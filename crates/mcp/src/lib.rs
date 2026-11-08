//! MCP server for tessar: memory for Claude Code and other agents.
//!
//! Layers:
//!
//! - [`api`] maps tool requests to core calls and renders the results as
//!   text. It knows nothing about MCP and is tested over an in-memory log.
//! - [`server`] is the thin `rmcp` handler that exposes [`api`] as tools.
//! - [`daemon`] serves that handler to many sessions over a Unix socket, one
//!   project space per connection.
//! - [`proxy`] is the stdio side: it forwards a session to the daemon.
//! - `main.rs` opens the journal and the model and picks a mode.

pub mod api;
pub mod args;
pub mod daemon;
pub mod project;
pub mod proxy;
pub mod server;
pub mod when;
