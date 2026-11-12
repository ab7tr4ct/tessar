//! The tool half of the decision log: one `"tool"` record per MCP tool call,
//! written by the process that owns the memory (the daemon, or `tessar mcp
//! --standalone`) from [`crate::server::TessarServer::call_tool`], the one
//! place every call passes through. The proxy only forwards and never logs.
//!
//! A record is built after the result exists, from the request's arguments
//! and the result as the agent receives it; it cannot change the result. The
//! write is handed to the blocking pool and not awaited, so a slow disk
//! delays neither the answer nor any lock; the file is opened per write
//! ([`decisions::append`]), so a daemon that outlives `tessar purge` writes
//! to the new file. Fact texts are never recorded.
//!
//! Handles and outcomes of `remember`, `lesson` and `update` are read back
//! from the report text, whose shape the tests pin down; a report that is
//! not recognised is recorded with the outcome `unparsed` and no handle.
//! The robust source would be structured content on those two tools (handle
//! and outcome per fact), a public result change that is not made here.
//!
//! Free-form arguments are cut to [`MAX_ARG_CHARS`] and error texts lose the
//! content of every quoted span, since serde errors echo the offending value.
//!
//! Known limits: the session id is read from the proxy's environment when it
//! starts and may go stale if the MCP process outlives a `/clear`; a call
//! the client cancels after the journal write leaves no tool record.

use std::path::{Path, PathBuf};

use rmcp::model::{CallToolResponse, CallToolResult, JsonObject};
use serde_json::Value;

use crate::decisions::{
    self, MAX_ERROR_CHARS, MAX_MESSAGE_CHARS, ToolDetail, ToolEntity, ToolFact, ToolHit, ToolRecord,
};

/// The environment variable in which Claude Code announces its session id.
const SESSION_ENV: &str = "CLAUDE_CODE_SESSION_ID";

/// The longest free-form argument kept in a record, in characters.
pub const MAX_ARG_CHARS: usize = 64;

/// The kinds of client a header may announce.
const CLIENTS: [&str; 3] = ["mcp", "hook", "cli"];

/// The role of this process, told to the daemon in the handshake.
static CLIENT_ROLE: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();

/// Says what this process is (`mcp`, `hook` or `cli`); the first call wins.
pub fn set_client_role(role: &'static str) {
    let _ = CLIENT_ROLE.set(role);
}

/// The role set by [`set_client_role`], or empty.
pub fn client_role() -> &'static str {
    CLIENT_ROLE.get().copied().unwrap_or("")
}

/// `raw` when it names a known kind of client.
pub fn clean_client(raw: &str) -> Option<String> {
    CLIENTS.contains(&raw.trim()).then(|| raw.trim().to_owned())
}

/// The longest session id kept, in characters.
const MAX_SESSION_CHARS: usize = 128;

/// Claude Code's session id from the environment, if it is set.
pub fn session_from_env() -> Option<String> {
    clean_session(&std::env::var(SESSION_ENV).ok()?)
}

/// `raw` trimmed and cut, `None` when nothing is left.
pub fn clean_session(raw: &str) -> Option<String> {
    let id = decisions::clip(raw.trim(), MAX_SESSION_CHARS);
    (!id.is_empty()).then_some(id)
}

/// Where and for whom the tool calls of one connection are logged.
#[derive(Debug, Clone)]
pub struct ToolLog {
    home: PathBuf,
    conn: Option<u64>,
    session: Option<String>,
    client: Option<String>,
}

impl ToolLog {
    /// A log in `home` for connection number `conn` (none when standalone)
    /// of the client with session id `session` and kind `client`.
    pub fn new(
        home: &Path,
        conn: Option<u64>,
        session: Option<String>,
        client: Option<String>,
    ) -> Self {
        Self {
            home: home.to_owned(),
            conn,
            session,
            client,
        }
    }

    /// Logs one finished call, `tool` being a name the router knows. Never fails and never waits for the disk.
    pub fn record(
        &self,
        project: &str,
        tool: &str,
        args: Option<&JsonObject>,
        elapsed: std::time::Duration,
        outcome: &Result<CallToolResponse, rmcp::ErrorData>,
    ) {
        let record = self.build(project, tool, args, elapsed, outcome);
        let home = self.home.clone();
        // Not awaited: the result is on its way regardless of the disk.
        drop(tokio::task::spawn_blocking(move || {
            decisions::append(&home, &decisions::Record::Tool(record));
        }));
    }

    fn build(
        &self,
        project: &str,
        tool: &str,
        args: Option<&JsonObject>,
        elapsed: std::time::Duration,
        outcome: &Result<CallToolResponse, rmcp::ErrorData>,
    ) -> ToolRecord {
        let (failed, text, structured) = match outcome {
            Ok(CallToolResponse::Complete(result)) => (
                result.is_error == Some(true),
                result_text(result),
                result.structured_content.as_ref(),
            ),
            Ok(_) => (false, String::new(), None),
            Err(e) => (true, e.message.to_string(), None),
        };
        let error = failed.then(|| decisions::clip(&unquote(&text), MAX_ERROR_CHARS));
        let args = args.cloned().unwrap_or_default();
        let (space, detail) = if failed {
            (requested_space(tool, &args), failed_detail(tool, &args))
        } else {
            describe(tool, &args, &text, structured)
        };
        ToolRecord {
            ts: decisions::now(),
            tool: tool.to_owned(),
            project: project.to_owned(),
            space,
            conn: self.conn,
            session: self.session.clone(),
            client: self.client.clone(),
            elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            failed,
            error,
            detail,
        }
    }
}

/// The text of the first text block of `result`.
fn result_text(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .find_map(|c| c.as_text().map(|t| t.text.clone()))
        .unwrap_or_default()
}

/// A free-form argument, trimmed and cut to [`MAX_ARG_CHARS`].
fn str_arg(args: &JsonObject, name: &str) -> Option<String> {
    args.get(name)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| decisions::clip(s, MAX_ARG_CHARS))
}

/// An argument that must be one of `allowed`; anything else is not logged.
fn choice_arg(args: &JsonObject, name: &str, allowed: &[&str]) -> Option<String> {
    str_arg(args, name).filter(|v| allowed.contains(&v.as_str()))
}

/// Words the tools' own messages put in backticks: names of arguments,
/// values and tools. Anything else in backticks may be the caller's text.
const BACKTICK_WORDS: [&str; 38] = [
    "query",
    "fact",
    "facts",
    "text",
    "space",
    "limit",
    "as_of",
    "reason",
    "since",
    "value",
    "when",
    "entity",
    "history",
    "kind",
    "key",
    "predicate",
    "mentions",
    "source",
    "subject",
    "synonym_of",
    "single",
    "predicate_description",
    "force_new_predicate",
    "project",
    "user",
    "changed",
    "wrong",
    "redact",
    "all",
    "statement",
    "attribute",
    "event",
    "remember",
    "recall",
    "update",
    "about",
    "lesson",
    "vocabulary",
];

/// `text` with the content of every double-quoted span, and of every
/// backticked span that is not one of [`BACKTICK_WORDS`], replaced by `…`:
/// serde and the tools echo offending values in quotes.
fn unquote(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut open: Option<(char, String)> = None;
    let mut escaped = false;
    for c in text.chars() {
        match &mut open {
            None => {
                out.push(c);
                if c == '"' || c == '`' {
                    open = Some((c, String::new()));
                }
            }
            Some((q, inner)) => {
                if escaped {
                    escaped = false;
                    inner.push(c);
                } else if c == '\\' && *q == '"' {
                    escaped = true;
                    inner.push(c);
                } else if c == *q {
                    if *q == '`' && BACKTICK_WORDS.contains(&inner.as_str()) {
                        out.push_str(inner);
                    } else {
                        out.push('…');
                    }
                    out.push(c);
                    open = None;
                } else {
                    inner.push(c);
                }
            }
        }
    }
    // A span that never closed is cut at its opening quote.
    if open.is_some() {
        out.push('…');
    }
    out
}

/// `f12`, when `s` is a fact handle.
fn handle_arg(s: &str) -> Option<String> {
    let digits = s.trim().strip_prefix('f')?;
    (!digits.is_empty() && digits.len() <= 10 && digits.bytes().all(|b| b.is_ascii_digit()))
        .then(|| format!("f{digits}"))
}

/// The space a call named, or the one its tool defaults to.
fn requested_space(tool: &str, args: &JsonObject) -> Option<String> {
    match tool {
        "remember" | "lesson" => Some(
            choice_arg(args, "space", &["user", "project"]).unwrap_or_else(|| "project".into()),
        ),
        "recall" => Some(
            choice_arg(args, "space", &["user", "project", "all"]).unwrap_or_else(|| "all".into()),
        ),
        _ => None,
    }
}

fn entity_of(args: &JsonObject) -> ToolEntity {
    let e = args.get("entity").and_then(Value::as_object);
    let field = |name: &str| e.and_then(|e| str_arg(e, name));
    ToolEntity {
        key: field("key").unwrap_or_default(),
        kind: field("kind").unwrap_or_default(),
        space: field("space").filter(|s| s == "user" || s == "project"),
    }
}

/// What can be said about a call that failed: only what it was asked.
fn failed_detail(tool: &str, args: &JsonObject) -> ToolDetail {
    match tool {
        "recall" => recall_detail(args, Vec::new()),
        "about" => about_detail(args, Vec::new()),
        "remember" | "lesson" => ToolDetail::Write {
            fact_count: fact_count(tool, args),
            facts: Vec::new(),
        },
        "update" => update_detail(args, None),
        _ => ToolDetail::Plain {},
    }
}

fn fact_count(tool: &str, args: &JsonObject) -> usize {
    match tool {
        "lesson" => 1,
        _ => args
            .get("facts")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
    }
}

fn recall_detail(args: &JsonObject, hits: Vec<ToolHit>) -> ToolDetail {
    ToolDetail::Recall {
        query: decisions::clip(
            args.get("query").and_then(Value::as_str).unwrap_or(""),
            MAX_MESSAGE_CHARS,
        ),
        limit: args.get("limit").and_then(Value::as_u64),
        as_of: str_arg(args, "as_of")
            .filter(|d| d.len() <= 10 && d.bytes().all(|b| b.is_ascii_digit() || b == b'-')),
        hits,
    }
}

fn about_detail(args: &JsonObject, handles: Vec<String>) -> ToolDetail {
    ToolDetail::About {
        entity: entity_of(args),
        history: args
            .get("history")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        handles,
    }
}

fn update_detail(args: &JsonObject, done: Option<(String, Option<String>)>) -> ToolDetail {
    let (outcome, new_handle) = match done {
        Some((outcome, new)) => (Some(outcome), new),
        None => (None, None),
    };
    let mode = choice_arg(args, "reason", &["changed", "wrong", "redact"]);
    // Anything else is free text: only its length is kept.
    let reason_chars = match (&mode, args.get("reason").and_then(Value::as_str)) {
        (None, Some(r)) => Some(r.chars().count()),
        _ => None,
    };
    ToolDetail::Update {
        fact: args
            .get("fact")
            .and_then(Value::as_str)
            .and_then(handle_arg),
        mode,
        reason_chars,
        outcome,
        new_handle,
    }
}

/// The space and the detail of a call that succeeded.
fn describe(
    tool: &str,
    args: &JsonObject,
    text: &str,
    structured: Option<&Value>,
) -> (Option<String>, ToolDetail) {
    let space = requested_space(tool, args);
    let detail = match tool {
        "recall" => recall_detail(args, hits_of(structured)),
        "about" => about_detail(
            args,
            hits_of(structured).into_iter().map(|h| h.id).collect(),
        ),
        "remember" | "lesson" => ToolDetail::Write {
            fact_count: fact_count(tool, args),
            facts: parse_facts(text, fact_count(tool, args)),
        },
        "update" => update_detail(args, Some(parse_update(text))),
        _ => ToolDetail::Plain {},
    };
    (space, detail)
}

fn hits_of(structured: Option<&Value>) -> Vec<ToolHit> {
    let Some(hits) = structured
        .and_then(|s| s.get("hits"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    hits.iter()
        .map(|h| ToolHit {
            id: h["id"].as_str().unwrap_or_default().to_owned(),
            space: h["space"].as_str().unwrap_or_default().to_owned(),
            channels: h["channels"]
                .as_array()
                .map(|c| {
                    c.iter()
                        .filter_map(|c| c.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default(),
            similarity: h["similarity"].as_f64(),
        })
        .collect()
}

/// The leading run of ASCII letters of `s`.
fn first_word(s: &str) -> &str {
    let end = s
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(s.len());
    &s[..end]
}

/// `f12` at the start of `s`, when a space or the end follows it.
fn handle_at(s: &str) -> Option<String> {
    let digits = s.strip_prefix('f')?;
    let n = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    let rest = &digits[n..];
    (n > 0 && (rest.is_empty() || rest.starts_with(char::is_whitespace)))
        .then(|| format!("f{}", &digits[..n]))
}

/// The first `fNN` that follows a `: ` in `s` and starts a fact line
/// (`fNN [date] [space] ...`).
fn handle_after_colon(s: &str) -> Option<String> {
    s.match_indices(": f").find_map(|(i, _)| {
        let tail = &s[i + 2..];
        let handle = handle_at(tail)?;
        tail[handle.len()..]
            .trim_start()
            .starts_with('[')
            .then_some(handle)
    })
}

/// The outcomes a `remember` report names, as its first word.
const WRITE_OUTCOMES: [&str; 6] = [
    "created",
    "conflicting",
    "unchanged",
    "kept",
    "closed",
    "superseded",
];

/// The outcomes an `update` report names, as its first word.
const UPDATE_OUTCOMES: [&str; 7] = [
    "created",
    "superseded",
    "corrected",
    "closed",
    "unchanged",
    "kept",
    "redacted",
];

/// What a report line could not be read as.
const UNPARSED: &str = "unparsed";

/// The facts of a `remember` or `lesson` report, lines
/// `N. outcome: fNN [date] ...`, for a call of `expected` facts. A line that
/// is not recognised, or a missing one, becomes outcome `unparsed` with no
/// handle.
pub fn parse_facts(text: &str, expected: usize) -> Vec<ToolFact> {
    let unparsed = || ToolFact {
        id: None,
        outcome: Some(UNPARSED.to_owned()),
    };
    let mut facts = Vec::new();
    for n in 1..=expected {
        let prefix = format!("{n}. ");
        let fact = text
            .lines()
            .find_map(|l| l.strip_prefix(&prefix))
            .and_then(|rest| {
                let outcome = first_word(rest);
                WRITE_OUTCOMES.contains(&outcome).then(|| ToolFact {
                    id: handle_after_colon(rest),
                    outcome: Some(outcome.to_owned()),
                })
            });
        facts.push(fact.unwrap_or_else(unparsed));
    }
    facts
}

/// The outcome of an `update` report and the version it created, if any;
/// `unparsed` when the report is not recognised.
pub fn parse_update(text: &str) -> (String, Option<String>) {
    let word = first_word(text);
    let unparsed = || (UNPARSED.to_owned(), None);
    if !UPDATE_OUTCOMES.contains(&word) || !text[word.len()..].starts_with(':') {
        return unparsed();
    }
    match word {
        "superseded" | "created" => text[word.len()..]
            .strip_prefix(": ")
            .and_then(handle_at)
            .map_or_else(unparsed, |h| (word.to_owned(), Some(h))),
        "corrected" => (
            word.to_owned(),
            text.split_once("Replaced by: ")
                .and_then(|(_, tail)| handle_at(tail)),
        ),
        _ => (word.to_owned(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remember_reports_are_read() {
        let text = "Remembered 3 fact(s) in space `repo`:\n\
            1. created: f12 [2026-11-08] [repo] The build uses cmake\n\
            2. superseded (was: Berlin, recorded 2026-11-01): f13 [2026-11-08] [user] User lives in Paris [current; earlier: Berlin (2026-11-01)]\n\
            3. unchanged (already known): f7 [2026-11-02] [repo] Same text";
        let got: Vec<_> = parse_facts(text, 3)
            .iter()
            .map(|f| (f.id.clone().unwrap(), f.outcome.clone().unwrap()))
            .collect();
        assert_eq!(
            got,
            [
                ("f12".to_owned(), "created".to_owned()),
                ("f13".to_owned(), "superseded".to_owned()),
                ("f7".to_owned(), "unchanged".to_owned())
            ]
        );
    }

    #[test]
    fn outcomes_the_mcp_tools_cannot_reach_are_read_too() {
        let line = |verb: &str| {
            parse_facts(
                &format!("Remembered 1 fact(s):\n1. {verb}: f5 [2026-11-08] [repo] x"),
                1,
            )
        };
        for verb in ["conflicting", "closed", "kept (already known)"] {
            let f = &line(verb)[0];
            assert_eq!(f.id.as_deref(), Some("f5"), "{verb}");
            assert_eq!(f.outcome.as_deref(), Some(first_word(verb)), "{verb}");
        }
        assert_eq!(
            parse_update("created: f9 [2026-11-08] [repo] x"),
            ("created".to_owned(), Some("f9".to_owned()))
        );
    }

    #[test]
    fn reports_that_are_not_recognised_are_unparsed() {
        for text in [
            "",
            "Remembered 1 fact(s):\n1. frobnicated: f1 [d] [s] x",
            "Nothing",
        ] {
            let facts = parse_facts(text, 1);
            assert_eq!(facts.len(), 1);
            assert_eq!(facts[0].outcome.as_deref(), Some("unparsed"));
            assert!(facts[0].id.is_none());
        }
        // Fewer lines than facts: the missing one is unparsed.
        let two = parse_facts("1. created: f1 [d] [s] x", 2);
        assert_eq!(two[0].outcome.as_deref(), Some("created"));
        assert_eq!(two[1].outcome.as_deref(), Some("unparsed"));
        assert_eq!(
            parse_update("something new: f1"),
            ("unparsed".to_owned(), None)
        );
        assert_eq!(
            parse_update("superseded: nothing"),
            ("unparsed".to_owned(), None)
        );
    }

    #[test]
    fn quoted_spans_lose_their_content() {
        assert_eq!(
            unquote(r#"invalid type: string "secret \" text", expected a sequence"#),
            r#"invalid type: string "…", expected a sequence"#
        );
        assert_eq!(
            unquote("unknown variant `xyz`, expected"),
            "unknown variant `…`, expected"
        );
        assert_eq!(unquote("`query` is empty"), "`query` is empty");
        assert_eq!(unquote("open \"never closed"), "open \"…");
        assert_eq!(unquote("plain"), "plain");
    }

    #[test]
    fn session_ids_are_trimmed_and_cut() {
        assert_eq!(clean_session("  abc \n").as_deref(), Some("abc"));
        assert_eq!(clean_session("   "), None);
        assert_eq!(
            clean_session(&"x".repeat(500)).unwrap().chars().count(),
            MAX_SESSION_CHARS
        );
    }
}
