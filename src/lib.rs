//! tessar — a memory engine for AI agents.
//!
//! The core is deterministic and has no LLM inside: it stores typed facts
//! about entities with validity periods and full version history, and finds
//! them by entity and keyword with explanations.

mod error;
pub mod model;

pub use error::Error;
