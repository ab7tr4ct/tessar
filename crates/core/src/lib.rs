//! tessar — a memory engine for AI agents.
//!
//! The core is deterministic and has no LLM inside: it stores typed facts
//! about entities with validity periods and full version history, and finds
//! them by entity and keyword with explanations.
//!
//! This crate holds the data model, the memory rules and the ports — traits
//! for everything outside the core, such as [`log::Log`] for persistence.
//! Adapters (the file journal, the MCP server) live in other crates and
//! depend on this one, never the other way round.

pub mod clock;
pub mod log;
pub mod model;
pub mod store;
