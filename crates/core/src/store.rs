//! The store: memory state in RAM, persisted through a
//! [`Log`](crate::log::Log).
//!
//! [`Memory`] is a cheap-to-clone handle shared by every caller. Writes go
//! through a single writer and become visible only after they are durable;
//! reads never wait for the disk. See [`Memory`] for the concurrency model.

mod draft;
mod error;
mod memory;
mod op;
mod plan;
mod recall;
mod state;
mod text;

pub use draft::{
    BodyDraft, EntityRef, EpisodeDraft, FactDraft, ObjectDraft, Outcome, Reason, Recorded,
    Remember, Remembered, Replacement, Update, View,
};
pub use error::StoreError;
pub use memory::Memory;
pub use op::Op;
pub use recall::{Hit, Recall, Recalled, Why};
