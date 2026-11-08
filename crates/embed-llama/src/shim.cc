#include "tessar-embed-llama/src/shim.h"
#include "tessar-embed-llama/src/lib.rs.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "llama.h"

namespace tessar {

namespace {

// llama.cpp logs to stderr unless told otherwise, and a library should
// not. The callback keeps only the error lines of the current call on
// this thread, so an exception can say what llama.cpp complained about.
// With TESSAR_LLAMA_LOG set, everything also goes to stderr.
thread_local std::string errors;
thread_local bool in_error = false;
bool verbose = false;
// Set when the exit handler starts. The exiting thread's thread_locals are
// already destroyed by then, so llama.cpp's last log lines are dropped.
std::atomic<bool> exiting{false};

void log_errors(ggml_log_level level, const char * text, void *) {
    if (text == nullptr) {
        return;
    }
    if (verbose) {
        std::fputs(text, stderr);
    }
    // CONT continues the previous line at the previous level.
    if (level != GGML_LOG_LEVEL_CONT) {
        in_error = level == GGML_LOG_LEVEL_ERROR;
    }
    if (in_error && !exiting.load(std::memory_order_relaxed)) {
        errors += text;
    }
}

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char * env = std::getenv("TESSAR_LLAMA_LOG");
        verbose = env != nullptr && env[0] != '\0';
        llama_log_set(log_errors, nullptr);
        // Registers the backends linked in (CPU, Metal); once per process.
        llama_backend_init();
    });
}

[[noreturn]] void fail(std::string message) {
    std::string details = errors;
    errors.clear();
    while (!details.empty() && std::isspace(static_cast<unsigned char>(details.back()))) {
        details.pop_back();
    }
    if (!details.empty()) {
        message += " (" + details + ")";
    }
    throw std::runtime_error(message);
}

// ggml's Metal backend keeps its devices in a static list and asserts,
// when that list is destroyed at exit, that every Metal buffer was freed.
// An embedder still alive then (in a Rust static, leaked, or in use while
// some thread calls exit()) would abort the process. So live embedders
// are tracked here and released by an exit handler that runs first.
//
// Not covered: exit() while another thread is inside load(). Statics that
// load() builds are destroyed under it, which no handler order can
// prevent (undefined behavior in C++). Applications must not exit while a
// model is loading.
struct Live {
    std::mutex mutex;
    std::unordered_set<Embedder *> embedders;
};

// Never destroyed: an embedder may still unregister after exit handlers.
Live & live() {
    static Live * const instance = new Live;
    return *instance;
}

void release_live() {
    exiting.store(true, std::memory_order_relaxed);
    Live & l = live();
    std::lock_guard<std::mutex> lock(l.mutex);
    for (Embedder * embedder : l.embedders) {
        embedder->release();
    }
}

// Exit handlers and static destructors run in the reverse order of their
// registration. Registered after a model was loaded, so after ggml built
// its device list, release_live() runs before that list is destroyed.
void release_live_at_exit() {
    static std::once_flag once;
    std::call_once(once, [] { std::atexit(release_live); });
}

// Frees a batch from llama_batch_init on every way out.
struct Batch {
    llama_batch raw;
    explicit Batch(int32_t n_tokens) : raw(llama_batch_init(n_tokens, 0, 1)) {}
    ~Batch() { llama_batch_free(raw); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
};

// Cuts a text to `max` tokens. Keeps the last token when the model appends
// one (EOS or SEP): models that pool the last token read it.
void truncate(const llama_vocab * vocab, std::vector<llama_token> & tokens, std::size_t max) {
    if (tokens.size() <= max) {
        return;
    }
    const bool keep_last = llama_vocab_get_add_eos(vocab) || llama_vocab_get_add_sep(vocab);
    const llama_token last = tokens.back();
    tokens.resize(max);
    if (keep_last && max > 0) {
        tokens.back() = last;
    }
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const rust::String & text) {
    if (text.size() > static_cast<std::size_t>(INT32_MAX)) {
        fail("a text is too long to tokenize");
    }
    const auto length = static_cast<int32_t>(text.size());
    // add_special: BOS/EOS/SEP as the GGUF asks; a model that pools the
    // last token relies on that EOS. parse_special = false: the text is
    // data, so "<|endoftext|>" written in a note stays plain text.
    std::vector<llama_token> tokens(text.size() + 4);
    int32_t n = llama_tokenize(vocab, text.data(), length, tokens.data(),
                               static_cast<int32_t>(tokens.size()), true, false);
    if (n < 0 && n != INT32_MIN) {
        // A negative count is the number of tokens needed.
        tokens.resize(static_cast<std::size_t>(-n));
        n = llama_tokenize(vocab, text.data(), length, tokens.data(),
                           static_cast<int32_t>(tokens.size()), true, false);
    }
    if (n < 0) {
        fail("cannot tokenize a text");
    }
    tokens.resize(static_cast<std::size_t>(n));
    return tokens;
}

}  // namespace

void FreeModel::operator()(llama_model * model) const { llama_model_free(model); }

void FreeContext::operator()(llama_context * context) const { llama_free(context); }

Embedder::Embedder(Model model, Context context, bool truncate)
    : model_(std::move(model)), context_(std::move(context)), truncate_(truncate) {
    dims_ = static_cast<std::size_t>(llama_model_n_embd_out(model_.get()));
    // llama.cpp pads the context; a batch must fit the context, the
    // logical batch and the physical batch at once.
    max_tokens_ = std::min({llama_n_ctx(context_.get()), llama_n_batch(context_.get()),
                            llama_n_ubatch(context_.get())});
    max_texts_ = llama_n_seq_max(context_.get());
    Live & l = live();
    std::lock_guard<std::mutex> lock(l.mutex);
    l.embedders.insert(this);
}

Embedder::~Embedder() {
    // Release under the registry lock, so an exit in another thread waits
    // until this embedder's buffers are gone.
    Live & l = live();
    std::lock_guard<std::mutex> lock(l.mutex);
    l.embedders.erase(this);
    release();
}

void Embedder::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    context_.reset();
    model_.reset();
}

std::size_t Embedder::dims() const { return dims_; }

void Embedder::embed(rust::Slice<const rust::String> texts, rust::Vec<float> & out) {
    std::lock_guard<std::mutex> lock(mutex_);
    errors.clear();
    if (!context_) {
        fail("the model was released: the process is exiting");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model_.get());

    // Tokenize everything first, so a text that cannot fit fails the
    // call before any work is done.
    std::vector<std::vector<llama_token>> tokens;
    tokens.reserve(texts.size());
    for (std::size_t i = 0; i < texts.size(); ++i) {
        tokens.push_back(tokenize(vocab, texts[i]));
        const std::size_t n = tokens.back().size();
        if (n == 0) {
            fail("text " + std::to_string(i) + " has no tokens");
        }
        if (n > max_tokens_) {
            if (!truncate_) {
                fail("text " + std::to_string(i) + " takes " + std::to_string(n) +
                     " tokens; the context holds " + std::to_string(max_tokens_));
            }
            truncate(vocab, tokens.back(), max_tokens_);
        }
    }

    out.reserve(out.size() + texts.size() * dims_);
    Batch batch(static_cast<int32_t>(max_tokens_));
    std::size_t first = 0;
    while (first < tokens.size()) {
        // Pack texts while they fit; each text is its own sequence.
        std::size_t end = first;
        std::size_t n_tokens = 0;
        while (end < tokens.size() && end - first < max_texts_ &&
               n_tokens + tokens[end].size() <= max_tokens_) {
            n_tokens += tokens[end].size();
            ++end;
        }

        batch.raw.n_tokens = 0;
        for (std::size_t s = first; s < end; ++s) {
            const auto seq = static_cast<llama_seq_id>(s - first);
            for (std::size_t pos = 0; pos < tokens[s].size(); ++pos) {
                const int32_t i = batch.raw.n_tokens++;
                batch.raw.token[i] = tokens[s][pos];
                batch.raw.pos[i] = static_cast<llama_pos>(pos);
                batch.raw.n_seq_id[i] = 1;
                batch.raw.seq_id[i][0] = seq;
                // Pooling reads every token's state, so all are outputs.
                batch.raw.logits[i] = true;
            }
        }

        // Nothing carries over between batches. Clearing the metadata is
        // enough: freed cells are masked out and then overwritten.
        llama_memory_clear(llama_get_memory(context_.get()), false);
        // For models without a KV cache (BERT-like encoders) llama_decode
        // runs the encoder, so it serves both kinds.
        const int32_t rc = llama_decode(context_.get(), batch.raw);
        if (rc != 0) {
            fail("llama_decode failed with code " + std::to_string(rc));
        }

        for (std::size_t s = first; s < end; ++s) {
            const auto seq = static_cast<llama_seq_id>(s - first);
            // The pooled state after the model's output projection, if it
            // has one: llama_model_n_embd_out() floats.
            const float * vector = llama_get_embeddings_seq(context_.get(), seq);
            if (vector == nullptr) {
                fail("no embedding for text " + std::to_string(s));
            }
            double sum = 0.0;
            for (std::size_t j = 0; j < dims_; ++j) {
                sum += static_cast<double>(vector[j]) * vector[j];
            }
            const double norm = std::sqrt(sum);
            const float scale = norm > 0.0 ? static_cast<float>(1.0 / norm) : 0.0f;
            for (std::size_t j = 0; j < dims_; ++j) {
                out.push_back(vector[j] * scale);
            }
        }
        first = end;
    }
}

std::unique_ptr<Embedder> load(rust::Str path, const LoadOptions & options) {
    init_once();
    errors.clear();
    if (options.n_ctx == 0) {
        fail("the context must hold at least one token");
    }

    llama_model_params model_params = llama_model_default_params();
    // A negative count means all layers.
    model_params.n_gpu_layers = options.gpu_layers;
    const std::string file(path.data(), path.size());
    Model model(llama_model_load_from_file(file.c_str(), model_params));
    if (!model) {
        fail("cannot load the model");
    }
    release_live_at_exit();

    llama_context_params params = llama_context_default_params();
    params.embeddings = true;
    // pooling_type and attention_type stay UNSPECIFIED: what the GGUF says.
    params.n_ctx = options.n_ctx;
    // With embeddings on, llama.cpp wants every sequence whole in one
    // physical batch, so both batch sizes are the whole context.
    params.n_batch = options.n_ctx;
    params.n_ubatch = options.n_ctx;
    // One KV buffer for all sequences of a batch, so any text may take the
    // whole context (as in llama.cpp's examples/embedding). The compute
    // buffers grow with the number of sequences, so it stays small.
    params.kv_unified = true;
    params.n_seq_max = std::min(std::max(options.n_seq_max, 1u),
                                static_cast<uint32_t>(llama_max_parallel_sequences()));
    Context context(llama_init_from_model(model.get(), params));
    if (!context) {
        fail("cannot create a context");
    }

    // The `enum` tag is needed: a function of the same name hides the type.
    const enum llama_pooling_type pooling = llama_pooling_type(context.get());
    if (pooling == LLAMA_POOLING_TYPE_NONE || pooling == LLAMA_POOLING_TYPE_RANK) {
        fail("the model does not pool tokens into one vector per text; is it an embedding model?");
    }
    return std::make_unique<Embedder>(std::move(model), std::move(context), options.truncate);
}

}  // namespace tessar
