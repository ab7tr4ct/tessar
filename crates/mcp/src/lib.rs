//! MCP server for tessar: memory for Claude Code and other agents.
//!
//! Three layers:
//!
//! - [`api`] maps tool requests to core calls and renders the results as
//!   text. It knows nothing about MCP and is tested over an in-memory log.
//! - [`server`] is the thin `rmcp` handler that exposes [`api`] as tools.
//! - `main.rs` opens the journal and the model and serves stdio.

pub mod api;
pub mod args;
pub mod server;
pub mod when;
