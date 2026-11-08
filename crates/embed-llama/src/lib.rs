//! In-process embedder for tessar: llama.cpp, linked statically, behind a
//! small C++ shim (`src/shim.cc`) bridged to Rust with [`cxx`].
//!
//! [`LlamaEmbedder`] loads a GGUF embedding model into the process — on
//! Apple silicon all layers go to the GPU through Metal — so no model
//! server has to run next to tessar. llama.cpp is a git submodule pinned
//! to the release the embedder was measured with (see `build.rs`).
//!
//! The default model, `voyage-4-nano` in Q8_0, lives at
//! `~/.tessar/models/voyage-4-nano-Q8_0.gguf` ([`voyage_nano_path`]):
//!
//! ```no_run
//! use tessar_core::embed::Embedder;
//! use tessar_embed_llama::{LlamaEmbedder, voyage_nano_path};
//!
//! let path = voyage_nano_path().expect("HOME is set");
//! let embedder = LlamaEmbedder::voyage_nano(&path)?;
//! let vector = embedder.embed_query("where do we keep the models?")?;
//! assert_eq!(vector.len(), embedder.dims());
//! # Ok::<(), tessar_core::embed::EmbedError>(())
//! ```
//!
//! With the default [`Options`] — texts up to 8192 tokens, up to 8 texts
//! per batch — this model takes about 1.6 GB of GPU memory: 351 MB of
//! weights, a 384 MB KV cache and ~840 MB of compute buffers. Both limits
//! trade memory for longer texts and faster bulk indexing.
//!
//! llama.cpp's own logging is silenced; its error lines end up in the
//! [`EmbedError`]s. Set `TESSAR_LLAMA_LOG=1` to see all of it on stderr.
//! Embedders still alive when the process exits (in a static, or in use
//! by another thread) are released by an exit handler before ggml shuts
//! its Metal backend down; calls after that fail. Exiting while another
//! thread is still inside [`LlamaEmbedder::load`] is not supported and may
//! crash: join the loading thread before the process exits.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use cxx::UniquePtr;
use tessar_core::embed::{EmbedError, Embedder, Prompts};

// The bridge expands to `unsafe extern` blocks and `unsafe impl`s; the
// declarations below are the whole C++ surface. Their contracts are
// documented in `src/shim.h`.
#[allow(unsafe_code)]
#[cxx::bridge(namespace = "tessar")]
mod ffi {
    /// How the shim loads a model; built from [`super::Options`].
    struct LoadOptions {
        /// Tokens in the context, which is also the batch size.
        n_ctx: u32,
        /// The most texts in one batch.
        n_seq_max: u32,
        /// Layers to offload to the GPU; negative means all.
        gpu_layers: i32,
        /// Cut texts longer than the context instead of failing.
        truncate: bool,
    }

    unsafe extern "C++" {
        include!("tessar-embed-llama/src/shim.h");

        /// A loaded model with its context; calls are serialized inside.
        type Embedder;

        /// Loads a GGUF embedding model.
        fn load(path: &str, options: &LoadOptions) -> Result<UniquePtr<Embedder>>;

        /// The length of the vectors the model returns.
        fn dims(self: &Embedder) -> usize;

        /// Appends one L2-normalized vector per text to `out`, in order.
        fn embed(self: Pin<&mut Embedder>, texts: &[String], out: &mut Vec<f32>) -> Result<()>;
    }
}

// SAFETY: the C++ `Embedder` owns a llama.cpp model and context and has
// no thread affinity: the object keeps no thread-local state, and the
// backends (CPU, Metal) accept work from any thread. What llama.cpp
// forbids is using one context from two threads at once; `LlamaEmbedder`
// keeps the only handle inside a `Mutex`, so every call holds the lock
// (and the exit handler in `shim.cc` takes the object's own mutex).
// `Sync` is not implemented: shared access goes through the `Mutex`.
#[allow(unsafe_code)]
unsafe impl Send for ffi::Embedder {}

/// How to load a model. Memory grows with both batch limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// Tokens in one batch, and so the most one text may take, prompt
    /// included. A longer text is cut or rejected, see `truncate`.
    pub context: u32,
    /// Texts in one batch, at least 1 and at most 256.
    pub batch: u32,
    /// Put all layers on the GPU (Metal on macOS); otherwise the CPU.
    pub gpu: bool,
    /// Cut a text longer than `context` to its first tokens instead of
    /// failing, as embedding APIs do. On by default: the store embeds
    /// every fact again when a memory is opened, and one fact that no
    /// longer fits (after `context` was lowered) must not keep the whole
    /// memory closed.
    pub truncate: bool,
}

impl Default for Options {
    /// 8192 tokens — a long memory note with room to spare — in batches
    /// of up to 8 texts, on the GPU; longer texts are cut.
    fn default() -> Self {
        Self {
            context: 8192,
            batch: 8,
            gpu: true,
            truncate: true,
        }
    }
}

/// An embedding model loaded into the process with llama.cpp.
///
/// Calls are serialized: one context embeds one batch at a time. Vectors
/// are L2-normalized and have [`LlamaEmbedder::dims`] components — the
/// model's output length, after its projection if it has one.
pub struct LlamaEmbedder {
    inner: Mutex<UniquePtr<ffi::Embedder>>,
    dims: usize,
    prompts: Prompts,
}

impl LlamaEmbedder {
    /// Loads the GGUF model at `path`, without prompts.
    pub fn load(path: impl AsRef<Path>, options: &Options) -> Result<Self, EmbedError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(EmbedError(format!(
                "{}: no such model file",
                path.display()
            )));
        }
        let utf8 = path
            .to_str()
            .ok_or_else(|| EmbedError(format!("{}: path is not UTF-8", path.display())))?;
        let options = ffi::LoadOptions {
            n_ctx: options.context,
            n_seq_max: options.batch,
            // A negative count means all layers.
            gpu_layers: if options.gpu { -1 } else { 0 },
            truncate: options.truncate,
        };
        let inner = ffi::load(utf8, &options)
            .map_err(|e| EmbedError(format!("{}: {}", path.display(), e.what())))?;
        let dims = inner.dims();
        Ok(Self {
            inner: Mutex::new(inner),
            dims,
            prompts: Prompts::none(),
        })
    }

    /// `voyage-4-nano` at `path` with its prompts and default [`Options`].
    pub fn voyage_nano(path: impl AsRef<Path>) -> Result<Self, EmbedError> {
        Ok(Self::load(path, &Options::default())?.with_prompts(Prompts::voyage()))
    }

    /// Sets the instructions before queries and documents.
    pub fn with_prompts(mut self, prompts: Prompts) -> Self {
        self.prompts = prompts;
        self
    }

    /// The length of every vector.
    pub fn dims(&self) -> usize {
        self.dims
    }

    fn embed(&self, prefix: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let input: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        let mut flat = Vec::with_capacity(texts.len() * self.dims);
        {
            // A panic cannot leave the context half-used: the shim resets
            // it before every batch. So a poisoned lock is still usable.
            let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            inner
                .pin_mut()
                .embed(&input, &mut flat)
                .map_err(|e| EmbedError(format!("llama.cpp: {}", e.what())))?;
        }
        if flat.len() != texts.len() * self.dims {
            return Err(EmbedError(format!(
                "llama.cpp: {} floats for {} texts of {} dims",
                flat.len(),
                texts.len(),
                self.dims
            )));
        }
        Ok(flat.chunks_exact(self.dims).map(<[f32]>::to_vec).collect())
    }
}

impl Embedder for LlamaEmbedder {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.embed(&self.prompts.document, texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut vectors = self.embed(&self.prompts.query, &[text.to_owned()])?;
        vectors
            .pop()
            .ok_or_else(|| EmbedError("llama.cpp: no vector".into()))
    }
}

impl std::fmt::Debug for LlamaEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaEmbedder")
            .field("dims", &self.dims)
            .field("prompts", &self.prompts)
            .finish_non_exhaustive()
    }
}

/// Where tessar keeps the default model:
/// `~/.tessar/models/voyage-4-nano-Q8_0.gguf`. `None` without `HOME`.
pub fn voyage_nano_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(
        PathBuf::from(home)
            .join(".tessar")
            .join("models")
            .join("voyage-4-nano-Q8_0.gguf"),
    )
}
