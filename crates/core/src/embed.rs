//! The embedder port: turning text into vectors for the meaning channel.
//!
//! The core never computes embeddings itself. An adapter — a local model
//! server, an API, a cache in front of either — implements [`Embedder`].
//! The store calls it before a write is committed, when the memory is
//! opened (vectors are derived data and are not in the log), and before a
//! query is searched.

use thiserror::Error;

/// Turns text into vectors.
///
/// Documents and queries are separate calls because many models prefix
/// them with different instructions. Vectors of one embedder must be
/// comparable with each other; their length may exceed the index length
/// (the store cuts them, which suits Matryoshka models).
pub trait Embedder: Send + Sync {
    /// Embeds facts for the index: one vector per text, in order.
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Embeds a query.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}

/// Instructions a model expects before queries and documents.
///
/// Adapters put [`Prompts::query`] before every query and
/// [`Prompts::document`] before every document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompts {
    /// Put before every query.
    pub query: String,
    /// Put before every document.
    pub document: String,
}

impl Prompts {
    /// No instructions.
    pub fn none() -> Self {
        Self {
            query: String::new(),
            document: String::new(),
        }
    }

    /// The Voyage 4 series (`voyage-4-nano`, `-lite`, `voyage-4`,
    /// `-large`): what their API adds for `input_type`.
    pub fn voyage() -> Self {
        Self {
            query: "Represent the query for retrieving supporting documents: ".into(),
            document: "Represent the document for retrieval: ".into(),
        }
    }
}

/// An embedder failure, or a vector the store cannot use.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct EmbedError(pub String);
