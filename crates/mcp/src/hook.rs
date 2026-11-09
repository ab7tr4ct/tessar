//! Claude Code hooks: `tessar hook prompt`, `tessar hook session` and
//! `tessar hook config`.
//!
//! A hook is a command Claude Code runs on an event, with a JSON object on
//! stdin. What it prints on stdout (exit status 0) is added to the model's
//! context. Memory being down must never block the user, so every failure
//! here ends as empty output; the binary only logs it to stderr.
//!
//! The logic is split from the process: [`prompt`] and [`session`] take a
//! connected daemon [`Peer`] and return the text to print, so tests drive
//! them over the test daemon without spawning the binary. [`run_prompt`] and
//! [`run_session`] add the connection and the time limit. The session hook
//! starts the daemon if needed and waits for it. The prompt hook runs before
//! every prompt and never waits: with no daemon listening it starts one for
//! the next prompt and prints nothing.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rmcp::model::CallToolRequestParams;
use rmcp::{Peer, RoleClient};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::daemon::socket_path;
use crate::lifecycle;
use crate::project::project_name;
use crate::proxy;

/// A hit is kept when its similarity is at least this, unless the entity
/// channel found it. A first guess, to be calibrated.
pub const DEFAULT_MIN_SIMILARITY: f64 = 0.45;
/// A hit the entity channel found passes with at least this similarity.
pub const DEFAULT_ENTITY_MIN_SIMILARITY: f64 = 0.35;
/// At most this many hits pass through the entity rule alone.
pub const DEFAULT_ENTITY_MAX: usize = 2;
/// Characters the `prompt` hook may print.
pub const DEFAULT_BUDGET: usize = 1500;
/// Characters the `session` hook may print.
pub const DEFAULT_SESSION_BUDGET: usize = 3000;
/// The `prompt` hook gives up (prints nothing) after this long, all told:
/// finding the project, the handshake with the daemon and the search. The
/// user's prompt waits for this hook, so it must not be held up by a daemon
/// that is down, restarting or loading its model.
pub const PROMPT_WAIT: Duration = Duration::from_secs(1);
/// The `session` hook gives up after this long: a cold daemon loads the model.
pub const SESSION_WAIT: Duration = Duration::from_secs(60);

/// A prompt with fewer words than this is not worth a search.
const MIN_WORDS: usize = 3;
/// How many facts `recall` is asked for.
const RECALL_LIMIT: usize = 8;
/// A pasted log is not a query: only this many characters are searched.
const MAX_QUERY_CHARS: usize = 2000;
/// Below this many characters left, a hit that does not fit is dropped
/// instead of cut.
const MIN_CUT: usize = 40;

const PROMPT_HEADER: &str = "Memory (tessar) relevant to this message; call `recall` for more:";

/// What Claude Code sends a hook on stdin. Unknown fields are ignored.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct HookInput {
    /// The conversation's id.
    pub session_id: String,
    /// The directory Claude Code runs in.
    pub cwd: String,
    /// `UserPromptSubmit`: the user's text.
    pub prompt: String,
    /// `SessionStart`: `startup`, `resume`, `clear` or `compact`.
    pub source: String,
}

impl HookInput {
    /// Parses the JSON on stdin.
    pub fn parse(stdin: &str) -> Result<Self, String> {
        serde_json::from_str(stdin).map_err(|e| format!("bad hook input: {e}"))
    }
}

/// The tunables, from the environment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// `TESSAR_HOOK_MIN_SIMILARITY`.
    pub min_similarity: f64,
    /// `TESSAR_HOOK_ENTITY_MIN_SIMILARITY`.
    pub entity_min_similarity: f64,
    /// `TESSAR_HOOK_ENTITY_MAX`.
    pub entity_max: usize,
    /// `TESSAR_HOOK_BUDGET`.
    pub budget: usize,
    /// `TESSAR_HOOK_SESSION_BUDGET`.
    pub session_budget: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            min_similarity: DEFAULT_MIN_SIMILARITY,
            entity_min_similarity: DEFAULT_ENTITY_MIN_SIMILARITY,
            entity_max: DEFAULT_ENTITY_MAX,
            budget: DEFAULT_BUDGET,
            session_budget: DEFAULT_SESSION_BUDGET,
        }
    }
}

impl Settings {
    /// Reads the variables through `env`; one that is missing or does not
    /// parse keeps its default.
    pub fn from_env(env: impl Fn(&str) -> Option<String>) -> Self {
        let d = Self::default();
        Self {
            min_similarity: env("TESSAR_HOOK_MIN_SIMILARITY")
                .and_then(|v| v.trim().parse().ok())
                .filter(|v: &f64| v.is_finite())
                .unwrap_or(d.min_similarity),
            entity_min_similarity: env("TESSAR_HOOK_ENTITY_MIN_SIMILARITY")
                .and_then(|v| v.trim().parse().ok())
                .filter(|v: &f64| v.is_finite())
                .unwrap_or(d.entity_min_similarity),
            entity_max: env("TESSAR_HOOK_ENTITY_MAX")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(d.entity_max),
            budget: env("TESSAR_HOOK_BUDGET")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(d.budget),
            session_budget: env("TESSAR_HOOK_SESSION_BUDGET")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(d.session_budget),
        }
    }
}

/// True for a prompt the `prompt` hook leaves alone: fewer than three words,
/// or a slash command.
pub fn skip_prompt(prompt: &str) -> bool {
    let prompt = prompt.trim();
    prompt.starts_with('/') || prompt.split_whitespace().count() < MIN_WORDS
}

/// One hit of `recall` or `about`, as the structured content gives it.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// The fact's handle, `f12`.
    pub id: String,
    /// `user` or a project name.
    pub space: String,
    /// The day it was recorded, `YYYY-MM-DD`.
    pub recorded: String,
    /// The fact's sentence with its annotations.
    pub text: String,
    /// Cosine similarity, when the meaning channel found it.
    pub similarity: Option<f64>,
    /// `entity`, `words`, `meaning`.
    pub channels: Vec<String>,
}

impl Hit {
    /// `- [date] [space] text`.
    fn line(&self) -> String {
        format!(
            "- [{}] [{}] {}",
            self.recorded,
            self.space,
            one_line(&self.text)
        )
    }

    /// `- [date] text`, for the briefing, where the space is the heading.
    fn short_line(&self) -> String {
        format!("- [{}] {}", self.recorded, one_line(&self.text))
    }
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The hits in `{"hits": [...]}`. Entries that lack an id are skipped.
pub fn parse_hits(structured: &Value) -> Vec<Hit> {
    let text = |h: &Value, key: &str| h[key].as_str().unwrap_or_default().to_owned();
    structured["hits"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|h| {
            let id = text(h, "id");
            (!id.is_empty()).then(|| Hit {
                id,
                space: text(h, "space"),
                recorded: text(h, "recorded"),
                text: text(h, "text"),
                similarity: h["similarity"].as_f64(),
                channels: h["channels"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|c| c.as_str().map(str::to_owned))
                    .collect(),
            })
        })
        .collect()
}

/// The hits worth showing, in order, not shown yet in this session. A hit
/// passes when its similarity is at least `min_similarity`, or when the
/// entity channel found it and its similarity is at least
/// `entity_min_similarity`; at most `entity_max` hits pass by the second
/// rule alone, since a prompt naming the project would otherwise pull in
/// every fact linked to it.
pub fn select(hits: Vec<Hit>, settings: &Settings, shown: &HashSet<String>) -> Vec<Hit> {
    let mut by_entity = 0;
    hits.into_iter()
        .filter(|h| !shown.contains(&h.id))
        .filter(|h| {
            let Some(sim) = h.similarity else {
                return false;
            };
            if sim >= settings.min_similarity {
                return true;
            }
            let entity = h.channels.iter().any(|c| c == "entity");
            if entity && sim >= settings.entity_min_similarity && by_entity < settings.entity_max {
                by_entity += 1;
                return true;
            }
            false
        })
        .collect()
}

/// Shortens `text` to `max` characters, ending in an ellipsis.
fn cut(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Takes `lines` in order while `used` plus the lines stays within
/// `budget` characters (a newline counts for each). The first line is cut
/// to fit when it is the only way to show anything.
fn fit(lines: &[String], used: usize, budget: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut used = used;
    for line in lines {
        let len = line.chars().count() + 1;
        if used + len <= budget {
            out.push(line.clone());
            used += len;
        } else {
            let room = budget.saturating_sub(used + 1);
            if out.is_empty() && room >= MIN_CUT {
                out.push(cut(line, room));
            }
            break;
        }
    }
    out
}

/// The text the `prompt` hook prints for `hits`, and the ids in it. Empty
/// when nothing fits.
pub fn render_prompt(hits: &[Hit], budget: usize) -> (String, Vec<String>) {
    let lines: Vec<String> = hits.iter().map(Hit::line).collect();
    let fitted = fit(&lines, PROMPT_HEADER.chars().count() + 1, budget);
    if fitted.is_empty() {
        return (String::new(), Vec::new());
    }
    let ids = hits
        .iter()
        .take(fitted.len())
        .map(|h| h.id.clone())
        .collect();
    (format!("{PROMPT_HEADER}\n{}\n", fitted.join("\n")), ids)
}

/// The text the `session` hook prints. `user` and `project` are the current
/// facts about each, newest first; `*_total` are their counts.
pub fn render_briefing(project: &str, user: &[Hit], project_hits: &[Hit], budget: usize) -> String {
    if user.is_empty() && project_hits.is_empty() {
        return "tessar memory is empty. Call `remember` to save facts: about the user to the \
                `user` space, about this project to the `project` space.\n"
            .to_owned();
    }
    let header = format!(
        "tessar memory: {} facts about you, {} about project `{project}`; tools: remember, \
         recall, about, update",
        user.len(),
        project_hits.len()
    );
    // The header and the closing newline come first; each list may then use
    // half of what is left, and what one leaves unused goes to the other.
    // An empty project space gets one sentence at the end; its room is kept.
    let hint = if project_hits.is_empty() {
        format!("\n{NO_PROJECT_FACTS}")
    } else {
        String::new()
    };
    let left = budget.saturating_sub(header.chars().count() + 1 + hint.chars().count());
    let user_lines: Vec<String> = user.iter().map(Hit::short_line).collect();
    let project_lines: Vec<String> = project_hits.iter().map(Hit::short_line).collect();
    let cost = |title: &str, lines: &[String]| {
        if lines.is_empty() {
            0
        } else {
            title.chars().count() + 1 + lines.iter().map(|l| l.chars().count() + 1).sum::<usize>()
        }
    };
    let user_cap = left - cost(PROJECT_TITLE, &project_lines).min(left / 2);
    let (user_text, user_used) = section(USER_TITLE, &user_lines, user_cap);
    let (project_text, _) = section(PROJECT_TITLE, &project_lines, left - user_used);
    format!("{header}{user_text}{project_text}{hint}\n")
}

const NO_PROJECT_FACTS: &str = "No facts about this repository yet: when you learn a convention, \
     a decision or a plan, save it with remember (space project).";

const USER_TITLE: &str = "About you:";
const PROJECT_TITLE: &str = "About this project:";

/// `\n<title>` and the lines that fit in `cap` characters, and the
/// characters used. Nothing for an empty list or one with no room.
fn section(title: &str, lines: &[String], cap: usize) -> (String, usize) {
    let title_len = title.chars().count() + 1;
    let fitted = fit(lines, title_len, cap);
    if fitted.is_empty() {
        return (String::new(), 0);
    }
    let used = title_len + fitted.iter().map(|l| l.chars().count() + 1).sum::<usize>();
    (format!("\n{title}\n{}", fitted.join("\n")), used)
}

// ---- the session's `.shown` file ---------------------------------------

/// `<home>/hooks/<session>.shown`, or `None` for a session id that is empty
/// (then nothing is remembered). Characters outside letters, digits, `-`
/// and `_` become `_`, so an id cannot leave the directory.
pub fn shown_path(home: &Path, session_id: &str) -> Option<PathBuf> {
    let id = session_id.trim();
    if id.is_empty() {
        return None;
    }
    let safe: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    Some(home.join("hooks").join(format!("{safe}.shown")))
}

/// The ids already shown in the session. Read errors count as none.
pub fn read_shown(home: &Path, session_id: &str) -> HashSet<String> {
    shown_path(home, session_id)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| {
            t.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Appends `ids` to the session's file, creating the directory.
pub fn append_shown(home: &Path, session_id: &str, ids: &[String]) -> std::io::Result<()> {
    use std::io::Write;
    let Some(path) = shown_path(home, session_id) else {
        return Ok(());
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for id in ids {
        writeln!(file, "{id}")?;
    }
    Ok(())
}

/// Forgets what was shown in the session: its context is fresh.
pub fn reset_shown(home: &Path, session_id: &str) {
    if let Some(path) = shown_path(home, session_id) {
        let _ = std::fs::remove_file(path);
    }
}

// ---- talking to the daemon ---------------------------------------------

/// Calls `tool` and returns its structured content.
async fn structured(
    peer: &Peer<RoleClient>,
    tool: &'static str,
    args: Value,
) -> Result<Value, String> {
    let mut params = CallToolRequestParams::new(tool);
    params.arguments = args.as_object().cloned();
    let result = peer
        .call_tool(params)
        .await
        .map_err(|e| format!("`{tool}` failed: {e}"))?;
    if result.is_error == Some(true) {
        return Err(format!("`{tool}` returned an error"));
    }
    result
        .structured_content
        .ok_or_else(|| format!("`{tool}` returned no structured content"))
}

/// The `UserPromptSubmit` hook: the facts relevant to the prompt, or an
/// empty string. Records the ids it prints in the session's `.shown` file.
pub async fn prompt(
    peer: &Peer<RoleClient>,
    input: &HookInput,
    settings: &Settings,
    home: &Path,
) -> Result<String, String> {
    if skip_prompt(&input.prompt) {
        return Ok(String::new());
    }
    let query: String = input.prompt.trim().chars().take(MAX_QUERY_CHARS).collect();
    let found = structured(
        peer,
        "recall",
        json!({ "query": query, "limit": RECALL_LIMIT }),
    )
    .await?;
    let shown = read_shown(home, &input.session_id);
    let chosen = select(parse_hits(&found), settings, &shown);
    let (text, ids) = render_prompt(&chosen, settings.budget);
    if !ids.is_empty()
        && let Err(e) = append_shown(home, &input.session_id, &ids)
    {
        logln!("tessar hook: cannot record shown facts: {e}");
    }
    Ok(text)
}

/// The `SessionStart` hook: forgets what was shown and prints a briefing of
/// what memory holds about the user and the project.
pub async fn session(
    peer: &Peer<RoleClient>,
    input: &HookInput,
    project: &str,
    settings: &Settings,
    home: &Path,
) -> Result<String, String> {
    reset_shown(home, &input.session_id);
    let newest_first = |found: Value| {
        let mut hits = parse_hits(&found);
        hits.reverse();
        hits
    };
    let user = newest_first(
        structured(
            peer,
            "about",
            json!({ "entity": { "key": "user", "kind": "person" } }),
        )
        .await?,
    );
    let project_hits = newest_first(
        structured(
            peer,
            "about",
            json!({ "entity": { "key": project, "kind": "project", "space": "project" } }),
        )
        .await?,
    );
    Ok(render_briefing(
        project,
        &user,
        &project_hits,
        settings.session_budget,
    ))
}

/// The project for a hook: the one of the hook's `cwd`, else of the
/// process's.
fn project_and_cwd(cwd: &str) -> (String, PathBuf) {
    let cwd = if cwd.trim().is_empty() {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    } else {
        PathBuf::from(cwd)
    };
    (project_name(&cwd), cwd)
}

/// Hooks never restart a daemon of another build; they say so and use it.
fn note_build(hello: &crate::daemon::Hello) {
    if hello.build != crate::BUILD_ID {
        logln!(
            "tessar hook: daemon build differs (daemon {}, hook {}); using it as it is",
            hello.build,
            crate::BUILD_ID
        );
    }
}

/// Connects to the daemon (starting it if none runs), runs `work` on the
/// connection and gives up after `wait`.
async fn with_daemon<F, Fut>(
    home: &Path,
    input: &HookInput,
    wait: Duration,
    work: F,
) -> Result<String, String>
where
    F: FnOnce(Peer<RoleClient>, String) -> Fut,
    Fut: std::future::Future<Output = Result<String, String>>,
{
    let (project, cwd) = project_and_cwd(&input.cwd);
    let attempt = async {
        std::fs::create_dir_all(home)
            .map_err(|e| format!("cannot create {}: {e}", home.display()))?;
        let stream = proxy::connect_or_start(home, wait).await?;
        let opened = proxy::open(stream, &project, &cwd).await?;
        note_build(opened.hello());
        let (server, link) = opened.serve().await?;
        let out = work(server.daemon().clone(), project).await;
        drop(link);
        out
    };
    tokio::time::timeout(wait, attempt)
        .await
        .unwrap_or_else(|_| {
            Err(format!(
                "no answer from the daemon within {}s",
                wait.as_secs()
            ))
        })
}

/// `tessar hook prompt` over the daemon in `home`.
pub async fn run_prompt(home: &Path, stdin: &str, settings: &Settings) -> Result<String, String> {
    run_prompt_with(home, stdin, settings, PROMPT_WAIT, &proxy::own_daemon()).await
}

/// [`run_prompt`] with its time limit and the way to start a daemon spelled
/// out.
///
/// It only connects. If nothing listens, it starts a daemon through `spawn`
/// without waiting for it (unless something already holds the journal: a
/// daemon on its way up) and fails at once; the next prompt finds the
/// daemon. Everything else, the `git` calls that name the project included,
/// is cut off after `cap`.
#[doc(hidden)]
pub async fn run_prompt_with(
    home: &Path,
    stdin: &str,
    settings: &Settings,
    cap: Duration,
    spawn: &proxy::Spawn,
) -> Result<String, String> {
    let input = HookInput::parse(stdin)?;
    if skip_prompt(&input.prompt) {
        return Ok(String::new());
    }
    let hook_input = &input;
    let attempt = async {
        let Ok(stream) = tokio::net::UnixStream::connect(socket_path(home)).await else {
            return Err(start_for_the_next_prompt(home, spawn));
        };
        // `git` runs off the runtime's threads so that the limit holds
        // even if it hangs.
        let named = {
            let cwd = hook_input.cwd.clone();
            tokio::task::spawn_blocking(move || project_and_cwd(&cwd))
        };
        let (project, cwd) = named.await.map_err(|e| format!("internal error: {e}"))?;
        let opened = proxy::open(stream, &project, &cwd).await?;
        note_build(opened.hello());
        let (server, link) = opened.serve().await?;
        let out = prompt(&server.daemon(), hook_input, settings, home).await;
        drop(link);
        out
    };
    tokio::time::timeout(cap, attempt)
        .await
        .unwrap_or_else(|_| {
            Err(format!(
                "no answer from the daemon within {} ms",
                cap.as_millis()
            ))
        })
}

/// Starts a daemon for `home` and does not wait for it. Returns what to
/// tell the log.
fn start_for_the_next_prompt(home: &Path, spawn: &proxy::Spawn) -> String {
    if lifecycle::journal_in_use(home) {
        return "no daemon is listening yet, but one holds the journal (it is starting)".to_owned();
    }
    let started = std::fs::create_dir_all(home)
        .map_err(|e| format!("cannot create {}: {e}", home.display()))
        .and_then(|()| spawn(home, proxy::Start::Fresh));
    match started {
        // Dropping the handle leaves the daemon running.
        Ok(_) => "no daemon is listening; started one for the next prompt".to_owned(),
        Err(e) => format!("no daemon is listening, and starting one failed: {e}"),
    }
}

/// `tessar hook session` over the daemon in `home`.
pub async fn run_session(home: &Path, stdin: &str, settings: &Settings) -> Result<String, String> {
    let input = HookInput::parse(stdin)?;
    let hook_input = &input;
    with_daemon(home, hook_input, SESSION_WAIT, |peer, project| async move {
        session(&peer, hook_input, &project, settings, home).await
    })
    .await
}

// ---- tessar hook config ------------------------------------------------

/// Quotes `path` for a shell command line when it needs it.
fn shell_word(path: &str) -> String {
    let plain = !path.is_empty()
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-+@:=,".contains(c));
    if plain {
        path.to_owned()
    } else {
        format!("'{}'", path.replace('\'', r"'\''"))
    }
}

/// The `hooks` object to paste into `~/.claude/settings.json`, with the
/// absolute path of the binary `exe`.
pub fn config_snippet(exe: &Path) -> String {
    let exe = shell_word(&exe.display().to_string());
    let snippet = json!({
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{
                    "type": "command",
                    "command": format!("{exe} hook prompt"),
                    "timeout": 20,
                }],
            }],
            "SessionStart": [{
                "matcher": "startup|resume|clear|compact",
                "hooks": [{
                    "type": "command",
                    "command": format!("{exe} hook session"),
                    "timeout": 60,
                }],
            }],
        },
    });
    serde_json::to_string_pretty(&snippet).expect("plain JSON") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, similarity: Option<f64>, channels: &[&str], text: &str) -> Hit {
        Hit {
            id: id.to_owned(),
            space: "user".to_owned(),
            recorded: "2026-11-08".to_owned(),
            text: text.to_owned(),
            similarity,
            channels: channels.iter().map(|c| (*c).to_owned()).collect(),
        }
    }

    #[test]
    fn short_prompts_and_slash_commands_are_skipped() {
        assert!(skip_prompt(""));
        assert!(skip_prompt("fix it"));
        assert!(skip_prompt("  ok thanks  "));
        assert!(skip_prompt("/compact please do it now"));
        assert!(skip_prompt("  /clear"));
        assert!(!skip_prompt("what do I prefer to eat"));
        assert!(!skip_prompt("one\ntwo\nthree"));
    }

    #[test]
    fn settings_from_env() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            Settings::from_env(move |k| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| (*v).to_owned())
            })
        };
        assert_eq!(env(&[]), Settings::default());
        assert_eq!(Settings::default().min_similarity, 0.45);
        assert_eq!(Settings::default().entity_min_similarity, 0.35);
        assert_eq!(Settings::default().entity_max, 2);
        assert_eq!(Settings::default().budget, 1500);
        assert_eq!(Settings::default().session_budget, 3000);
        let s = env(&[
            ("TESSAR_HOOK_MIN_SIMILARITY", "0.6"),
            ("TESSAR_HOOK_ENTITY_MIN_SIMILARITY", "0.2"),
            ("TESSAR_HOOK_ENTITY_MAX", "5"),
            ("TESSAR_HOOK_BUDGET", "200"),
            ("TESSAR_HOOK_SESSION_BUDGET", "nope"),
        ]);
        assert_eq!(s.min_similarity, 0.6);
        assert_eq!(s.entity_min_similarity, 0.2);
        assert_eq!(s.entity_max, 5);
        assert_eq!(s.budget, 200);
        assert_eq!(s.session_budget, 3000);
    }

    fn ids(hits: Vec<Hit>) -> Vec<String> {
        hits.into_iter().map(|h| h.id).collect()
    }

    #[test]
    fn threshold_or_entity_with_a_floor_decides() {
        let hits = vec![
            hit("f1", Some(0.7), &["meaning"], "close"),
            hit("f2", Some(0.45), &["meaning"], "on the line"),
            hit("f3", Some(0.40), &["meaning"], "far"),
            hit(
                "f4",
                Some(0.36),
                &["entity", "meaning"],
                "named, above the floor",
            ),
            hit(
                "f5",
                Some(0.10),
                &["entity", "meaning"],
                "named, below the floor",
            ),
            hit("f6", None, &["entity"], "named, no vector"),
            hit("f7", None, &["words"], "words only"),
            hit("f8", Some(0.5), &["entity", "meaning"], "named and close"),
        ];
        assert_eq!(
            ids(select(hits, &Settings::default(), &HashSet::new())),
            ["f1", "f2", "f4", "f8"]
        );
    }

    #[test]
    fn entity_only_hits_are_capped() {
        let hits: Vec<Hit> = (1..=5)
            .map(|n| {
                hit(
                    &format!("f{n}"),
                    Some(0.40),
                    &["entity", "meaning"],
                    "named",
                )
            })
            .chain([hit("f6", Some(0.8), &["meaning"], "close")])
            .collect();
        // Two pass by the entity rule; a close hit after them still passes.
        assert_eq!(
            ids(select(hits.clone(), &Settings::default(), &HashSet::new())),
            ["f1", "f2", "f6"]
        );
        let none = Settings {
            entity_max: 0,
            ..Settings::default()
        };
        assert_eq!(ids(select(hits, &none, &HashSet::new())), ["f6"]);
    }

    #[test]
    fn shown_hits_are_dropped() {
        let hits = vec![
            hit("f1", Some(0.9), &[], "a"),
            hit("f2", Some(0.9), &[], "b"),
        ];
        let shown: HashSet<String> = ["f1".to_owned()].into();
        assert_eq!(ids(select(hits, &Settings::default(), &shown)), ["f2"]);
    }

    #[test]
    fn prompt_output_format_and_budget() {
        let hits = vec![
            hit("f1", Some(0.9), &[], "User prefers Russian."),
            hit("f2", Some(0.8), &[], &"long ".repeat(100)),
            hit("f3", Some(0.7), &[], "Third."),
        ];
        let (text, ids) = render_prompt(&hits, 1500);
        assert_eq!(
            text.lines().next().unwrap(),
            "Memory (tessar) relevant to this message; call `recall` for more:"
        );
        assert_eq!(
            text.lines().nth(1).unwrap(),
            "- [2026-11-08] [user] User prefers Russian."
        );
        assert_eq!(ids, ["f1", "f2", "f3"]);
        assert!(text.chars().count() <= 1500);

        // Best first: the cut stops at the first hit that does not fit.
        let (text, ids) = render_prompt(&hits, 160);
        assert_eq!(ids, ["f1"]);
        assert!(text.chars().count() <= 160, "{text}");
        assert!(!text.contains("long"));

        // Nothing fits: nothing at all.
        assert_eq!(render_prompt(&hits, 50), (String::new(), Vec::new()));
        assert_eq!(render_prompt(&[], 1500), (String::new(), Vec::new()));
    }

    #[test]
    fn a_single_long_hit_is_cut_to_the_budget() {
        let hits = vec![hit("f1", Some(0.9), &[], &"word ".repeat(200))];
        let (text, ids) = render_prompt(&hits, 300);
        assert_eq!(ids, ["f1"]);
        assert!(text.chars().count() <= 300, "{}", text.chars().count());
        assert!(text.trim_end().ends_with('…'));
    }

    #[test]
    fn briefing_has_both_sections_newest_first_and_fits() {
        let user: Vec<Hit> = (0..40)
            .rev()
            .map(|n| {
                hit(
                    &format!("f{n}"),
                    None,
                    &[],
                    &format!("User fact number {n} is here."),
                )
            })
            .collect();
        let project: Vec<Hit> = (0..3)
            .map(|n| {
                hit(
                    &format!("f{}", 100 + n),
                    None,
                    &[],
                    &format!("Project fact {n}."),
                )
            })
            .collect();
        let text = render_briefing("tessar", &user, &project, 600);
        assert!(text.starts_with(
            "tessar memory: 40 facts about you, 3 about project `tessar`; tools: remember, \
             recall, about, update\nAbout you:\n- [2026-11-08] User fact number 39"
        ));
        assert!(text.contains("About this project:\n- [2026-11-08] Project fact 0."));
        assert!(
            text.contains("Project fact 2."),
            "the small list is whole: {text}"
        );
        assert!(text.chars().count() <= 600, "{}", text.chars().count());
        assert!(!text.contains("No facts about this repository"), "{text}");
    }

    #[test]
    fn briefing_nudges_when_the_project_has_no_facts() {
        let user: Vec<Hit> = (0..40)
            .map(|n| {
                hit(
                    &format!("f{n}"),
                    None,
                    &[],
                    &format!("User fact number {n}."),
                )
            })
            .collect();
        let text = render_briefing("tessar", &user, &[], 600);
        assert!(
            text.ends_with(
                "\nNo facts about this repository yet: when you learn a convention, a decision \
                 or a plan, save it with remember (space project).\n"
            ),
            "{text}"
        );
        assert!(text.contains("0 about project `tessar`"));
        assert!(text.contains("About you:"));
        // The sentence keeps its room inside the budget even with a long list.
        assert!(text.chars().count() <= 600, "{}", text.chars().count());
    }

    #[test]
    fn briefing_of_one_side_and_of_nothing() {
        let project = vec![hit("f1", None, &[], "Uses Rust.")];
        let text = render_briefing("tessar", &[], &project, 3000);
        assert!(text.contains("0 facts about you, 1 about project `tessar`"));
        assert!(!text.contains("About you:"));
        assert!(text.contains("About this project:\n- [2026-11-08] Uses Rust."));
        assert!(!text.contains("No facts about this repository"), "{text}");
        let empty = render_briefing("tessar", &[], &[], 3000);
        assert!(empty.contains("empty"));
        assert!(empty.contains("`remember`"));
        assert_eq!(empty.lines().count(), 1);
        assert!(!empty.contains("No facts about this repository"), "{empty}");
    }

    #[test]
    fn shown_file_round_trip_and_safe_names() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_shown(dir.path(), "s1").is_empty());
        append_shown(dir.path(), "s1", &["f1".to_owned(), "f2".to_owned()]).unwrap();
        append_shown(dir.path(), "s1", &["f3".to_owned()]).unwrap();
        let shown = read_shown(dir.path(), "s1");
        assert_eq!(shown.len(), 3);
        assert!(shown.contains("f2"));
        assert!(read_shown(dir.path(), "other").is_empty());
        reset_shown(dir.path(), "s1");
        assert!(read_shown(dir.path(), "s1").is_empty());
        reset_shown(dir.path(), "s1");

        let p = shown_path(dir.path(), "../../etc/x y").unwrap();
        assert_eq!(p.parent().unwrap(), dir.path().join("hooks"));
        assert!(shown_path(dir.path(), " ").is_none());
        assert!(append_shown(dir.path(), "", &["f1".to_owned()]).is_ok());
    }

    #[test]
    fn config_snippet_has_both_hooks_with_the_binary() {
        let snippet = config_snippet(Path::new("/opt/tessar/bin/tessar"));
        let v: Value = serde_json::from_str(&snippet).unwrap();
        let prompt = &v["hooks"]["UserPromptSubmit"][0]["hooks"][0];
        assert_eq!(prompt["type"], "command");
        assert_eq!(prompt["command"], "/opt/tessar/bin/tessar hook prompt");
        assert_eq!(prompt["timeout"], 20);
        let start = &v["hooks"]["SessionStart"][0];
        assert_eq!(start["matcher"], "startup|resume|clear|compact");
        assert_eq!(
            start["hooks"][0]["command"],
            "/opt/tessar/bin/tessar hook session"
        );
        assert_eq!(start["hooks"][0]["timeout"], 60);
        let spaced = config_snippet(Path::new("/My Apps/it's/tessar"));
        let v: Value = serde_json::from_str(&spaced).unwrap();
        assert_eq!(
            v["hooks"]["SessionStart"][0]["hooks"][0]["command"],
            r"'/My Apps/it'\''s/tessar' hook session"
        );
    }
}
