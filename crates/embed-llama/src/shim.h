// A small C++ layer between Rust and llama.cpp: load an embedding model,
// turn texts into vectors. Errors are exceptions; cxx turns them into
// `Err` on the Rust side.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "rust/cxx.h"

struct llama_context;
struct llama_model;

namespace tessar {

struct LoadOptions;

struct FreeModel {
    void operator()(llama_model * model) const;
};
struct FreeContext {
    void operator()(llama_context * context) const;
};
using Model = std::unique_ptr<llama_model, FreeModel>;
using Context = std::unique_ptr<llama_context, FreeContext>;

// A model with one context in embeddings mode. Not thread-safe: the Rust
// side serializes calls.
class Embedder {
public:
    Embedder(Model model, Context context);
    ~Embedder();
    Embedder(const Embedder &) = delete;
    Embedder & operator=(const Embedder &) = delete;

    // The length of every vector.
    std::size_t dims() const;

    // Appends one L2-normalized vector per text to `out`, in order. Texts
    // are packed several to a decode while they fit in the context.
    // Throws if a text is longer than the context or llama.cpp fails;
    // `out` may then hold vectors of the texts before the failing batch.
    void embed(rust::Slice<const rust::String> texts, rust::Vec<float> & out);

    // Frees the model and context; later calls to embed() throw. Runs at
    // process exit for embedders still alive (see shim.cc).
    void release();

private:
    // Guards model_ and context_ against release() at exit.
    std::mutex mutex_;
    // Declared in this order so the context is freed before its model.
    Model model_;
    Context context_;
    std::size_t dims_;
    // The most tokens and texts one decode takes.
    std::size_t max_tokens_;
    std::size_t max_texts_;
};

// Loads a GGUF embedding model. Throws if the file cannot be loaded or
// the model does not pool token states into one vector per text.
std::unique_ptr<Embedder> load(rust::Str path, const LoadOptions & options);

}  // namespace tessar
