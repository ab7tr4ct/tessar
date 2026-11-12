//! The decision log, `<home>/decisions.jsonl`: one JSON object per line
//! about what the hooks considered, showed and held back, and which tools
//! the agent called afterwards, so that the hooks' thresholds can be judged
//! from evidence.
//!
//! The log holds the text of the user's messages and of `recall` queries
//! (cut to [`MAX_MESSAGE_CHARS`]) and the error text of failed tool calls,
//! grows without rotation and is removed by `tessar purge`, since a message
//! may carry text that is redacted later. Tool records never hold fact
//! texts; those are in the journal.
//!
//! The log must never break its writer: [`append`] makes one `write` of one
//! line and returns nothing, whatever happens. The file is opened with
//! `O_APPEND` and each record is one `write`, so records from several
//! processes do not interleave. Readers are not part of this module; a line
//! may gain fields, so a reader has to ignore the ones it does not know.

use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{SecondsFormat, Utc};
use serde::Serialize;

/// The message text kept in a record, in characters.
pub const MAX_MESSAGE_CHARS: usize = 2000;

/// The error text kept in a tool record, in characters.
pub const MAX_ERROR_CHARS: usize = 300;

/// `<home>/decisions.jsonl`.
pub fn log_path(home: &Path) -> PathBuf {
    home.join("decisions.jsonl")
}

/// The current time as RFC 3339 in UTC, for a record's `ts`.
pub fn now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// The first [`MAX_MESSAGE_CHARS`] characters of `text`.
pub fn clip_message(text: &str) -> String {
    text.chars().take(MAX_MESSAGE_CHARS).collect()
}

/// The first `max` characters of `text`.
pub fn clip(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

/// Appends `record` as one line to the log in `home`, with one `write`. The
/// file is created `0600`. Every error is ignored: logging is never a reason
/// for a hook or a tool call to fail.
pub fn append<T: Serialize>(home: &Path, record: &T) {
    let _ = try_append(home, record);
}

fn try_append<T: Serialize>(home: &Path, record: &T) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(record).map_err(std::io::Error::other)?;
    line.push(b'\n');
    let mut file = crate::private::file_options()
        .create(true)
        .append(true)
        .open(log_path(home))?;
    crate::private::tighten_file(&file);
    file.write_all(&line)
}

/// One line of the log made by a hook.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Record {
    /// A run of the `prompt` hook.
    Prompt(PromptRecord),
    /// A run of the `session` hook.
    Session(SessionRecord),
    /// A call of an MCP tool.
    Tool(ToolRecord),
}

/// The thresholds and the budget a `prompt` run worked under.
#[derive(Debug, Clone, Serialize)]
pub struct Thresholds {
    pub min_similarity: f64,
    pub entity_min_similarity: f64,
    pub entity_max: usize,
    pub lesson_min_similarity: f64,
    pub lesson_max: usize,
    pub budget: usize,
}

/// What became of one fact the hook looked at.
#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    /// The fact's handle, `f12`.
    pub id: String,
    pub space: String,
    /// `entity`, `words`, `meaning`, as the daemon reported them.
    pub channels: Vec<String>,
    pub similarity: Option<f64>,
    pub lesson: bool,
    /// A lesson printed shortened to fit its line.
    pub cut: bool,
    /// `shown` or `held`.
    pub decision: &'static str,
    /// Why it was held: `below_threshold`, `already_shown`, `over_limit` or
    /// `over_budget`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}

/// A run of the `prompt` hook.
#[derive(Debug, Clone, Serialize)]
pub struct PromptRecord {
    pub ts: String,
    pub session: String,
    /// `None` when the hook stopped before it looked the project up.
    pub project: Option<String>,
    pub message_chars: usize,
    /// The message cut to [`MAX_MESSAGE_CHARS`].
    pub message: String,
    pub thresholds: Thresholds,
    pub elapsed_ms: u64,
    /// Why the hook stayed silent on purpose: `short_message` or
    /// `slash_command`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<&'static str>,
    /// Why the run failed (no daemon, a timeout), when it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The characters printed.
    pub printed_chars: usize,
    pub candidates: Vec<Candidate>,
}

/// A run of the `session` hook.
#[derive(Debug, Clone, Serialize)]
pub struct SessionRecord {
    pub ts: String,
    pub session: String,
    /// `None` when the run failed before the project was known.
    pub project: Option<String>,
    /// `startup`, `resume`, `clear` or `compact`.
    pub source: String,
    pub elapsed_ms: u64,
    /// Why the run failed (no daemon, a timeout), when it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The blocks printed and their sizes in characters.
    pub blocks: Vec<Block>,
    pub printed_chars: usize,
    /// Lessons recorded for this project.
    pub lessons_project: usize,
    /// Lessons recorded in the user space.
    pub lessons_user: usize,
}

/// One block of a briefing.
#[derive(Debug, Clone, Serialize)]
pub struct Block {
    pub name: &'static str,
    pub chars: usize,
}

/// A call of an MCP tool, written by the process that owns the memory after
/// the result was produced.
#[derive(Debug, Clone, Serialize)]
pub struct ToolRecord {
    pub ts: String,
    pub tool: String,
    /// The project space's name.
    pub project: String,
    /// The space the call worked in: `user`, `project` or, for a `recall`
    /// that was not narrowed, `all`; `None` when the call does not say.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space: Option<String>,
    /// The daemon's connection number (`#185` in its log); `None` in the
    /// standalone mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conn: Option<u64>,
    /// Claude Code's session id, as the client announced it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    /// What called: `mcp` (the agent), `hook` or `cli`; absent when the
    /// client did not say (an older proxy).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
    pub elapsed_ms: u64,
    pub failed: bool,
    /// The error text cut to [`MAX_ERROR_CHARS`], when the call failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(flatten)]
    pub detail: ToolDetail,
}

/// What is specific to one tool; the variant is told by the field names.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ToolDetail {
    /// `vocabulary`, and a call that failed before it could be read.
    Plain {},
    Recall {
        /// The query cut to [`MAX_MESSAGE_CHARS`].
        query: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        as_of: Option<String>,
        hits: Vec<ToolHit>,
    },
    About {
        entity: ToolEntity,
        history: bool,
        handles: Vec<String>,
    },
    /// `remember` and `lesson`.
    Write {
        fact_count: usize,
        facts: Vec<ToolFact>,
    },
    Update {
        /// The handle addressed, when the argument is a handle.
        #[serde(skip_serializing_if = "Option::is_none")]
        fact: Option<String>,
        /// `changed`, `wrong` or `redact`, when the reason is one of them.
        #[serde(skip_serializing_if = "Option::is_none")]
        mode: Option<String>,
        /// The length of a reason that is none of them; its text is not kept.
        #[serde(skip_serializing_if = "Option::is_none")]
        reason_chars: Option<usize>,
        /// one of [`crate::written::WriteOutcome`]'s words, or `unknown`.
        #[serde(skip_serializing_if = "Option::is_none")]
        outcome: Option<String>,
        /// The version the call created, if it did.
        #[serde(skip_serializing_if = "Option::is_none")]
        new_handle: Option<String>,
    },
}

/// A hit `recall` returned.
#[derive(Debug, Clone, Serialize)]
pub struct ToolHit {
    pub id: String,
    pub space: String,
    pub channels: Vec<String>,
    pub similarity: Option<f64>,
}

/// The entity `about` was asked for.
#[derive(Debug, Clone, Serialize)]
pub struct ToolEntity {
    pub key: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space: Option<String>,
}

/// A fact `remember` or `lesson` reported.
#[derive(Debug, Clone, Serialize)]
pub struct ToolFact {
    pub id: Option<String>,
    /// One of [`crate::written::WriteOutcome`]'s words, or `unknown` when
    /// the result carried no structured content.
    pub outcome: Option<String>,
    /// `superseded`: the version replaced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous: Option<String>,
    /// `kept`: a contradicting value that was closed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub closed: Option<String>,
    /// `kept`: how many more were closed; absent when none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub closed_others: Option<u32>,
    /// `conflicting`: the asserted value it sits beside.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conflicts_with: Option<String>,
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    #[test]
    fn lines_are_appended_and_the_file_is_private() {
        let home = tempfile::tempdir().unwrap();
        append(home.path(), &serde_json::json!({"kind": "x", "n": 1}));
        append(home.path(), &serde_json::json!({"kind": "x", "n": 2}));
        let path = log_path(home.path());
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(lines[1]).unwrap()["n"],
            2
        );
        // A file an older run made too wide is tightened.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        append(home.path(), &serde_json::json!({"n": 3}));
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn an_unwritable_home_is_ignored() {
        let home = tempfile::tempdir().unwrap();
        append(&home.path().join("missing"), &serde_json::json!({"n": 1}));
        assert!(!home.path().join("missing").exists());
    }

    #[test]
    fn the_message_is_cut_by_characters() {
        let long = "ж".repeat(MAX_MESSAGE_CHARS + 50);
        assert_eq!(clip_message(&long).chars().count(), MAX_MESSAGE_CHARS);
        assert_eq!(clip_message("short"), "short");
    }
}
