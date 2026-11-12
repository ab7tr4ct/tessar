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
//! - [`hook`] is the logic of the Claude Code hooks (`tessar hook`).
//! - [`lifecycle`] is the daemon's pid file, its retirement when the binary
//!   changed, and `tessar daemon stop|status`.
//! - `main.rs` opens the journal and the model and picks a mode.

/// Writes one line to stderr, ignoring a failure to write it. `eprintln!`
/// panics when stderr is gone (a closed terminal, a full disk under the
/// log file), and neither the daemon nor a proxy may die of its own logging.
pub fn log(line: std::fmt::Arguments<'_>) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr().lock(), "{line}");
}

/// [`log()`] with `format!` arguments.
macro_rules! logln {
    ($($arg:tt)*) => {
        $crate::log(format_args!($($arg)*))
    };
}

pub mod api;
pub mod args;
pub mod daemon;
pub mod decisions;
pub mod fingerprint;
pub mod hook;
pub mod lesson;
pub mod lifecycle;
pub mod private;
pub mod project;
pub mod proxy;
pub mod purge;
pub mod server;
pub mod when;

/// Identifies this build: the crate version, the commit it was built from
/// and the unix time of the build. A daemon and a proxy with different ids
/// are different binaries.
pub const BUILD_ID: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "+",
    env!("TESSAR_GIT_HASH"),
    ".",
    env!("TESSAR_BUILD_ID")
);
