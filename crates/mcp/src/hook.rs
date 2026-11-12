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
use std::time::{Duration, Instant};

use rmcp::model::CallToolRequestParams;
use rmcp::{Peer, RoleClient};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::daemon::socket_path;
use crate::decisions::{self, Block, Candidate, PromptRecord, Record, SessionRecord, Thresholds};
use crate::lesson;
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
/// A lesson is kept when its similarity is at least this: lessons are
/// phrased through a situation and compete badly under the general
/// threshold.
pub const LESSON_MIN_SIMILARITY: f64 = 0.35;
/// At most this many lessons are printed per prompt.
pub const LESSON_MAX: usize = 2;
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
/// How many facts `recall` is asked for: more than the ordinary window, so
/// that lessons, which rank below the facts the general threshold keeps, are
/// among them.
const RECALL_LIMIT: usize = 16;
/// A lesson's line in the prompt hook is cut to this many characters, its
/// head (`- lesson fNN [date]: `) included.
const LESSON_LINE_CHARS: usize = 300;
/// Ordinary facts are taken only from this many best hits, as before lessons
/// had a slot of their own.
const ORDINARY_WINDOW: usize = 8;
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
    skip_reason(prompt).is_some()
}

/// Why [`skip_prompt`] is true, as the decision log names it.
fn skip_reason(prompt: &str) -> Option<&'static str> {
    let prompt = prompt.trim();
    if prompt.starts_with('/') {
        Some("slash_command")
    } else if prompt.split_whitespace().count() < MIN_WORDS {
        Some("short_message")
    } else {
        None
    }
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

    /// True for a lesson, as [`lesson::is_lesson`] says.
    pub fn is_lesson(&self) -> bool {
        lesson::is_lesson(&self.text)
    }

    /// The lesson without its `Lesson (date):` header, on one line.
    fn lesson_body(&self) -> String {
        one_line(lesson::body(&self.text))
    }

    /// `- lesson f88 [date]: body`: the handle comes first so that the
    /// agent can act on it.
    fn lesson_line(&self) -> String {
        format!(
            "- lesson {} [{}]: {}",
            self.id,
            self.recorded,
            self.lesson_body()
        )
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

/// Why a hit was held back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hold {
    /// Shown earlier in this session.
    AlreadyShown,
    /// Its similarity is under the threshold of its rule.
    BelowThreshold,
    /// It passes a rule, but that rule's quota is used up.
    OverLimit,
    /// It did not fit in the character budget.
    OverBudget,
}

impl Hold {
    /// The name in the decision log.
    pub fn name(self) -> &'static str {
        match self {
            Self::AlreadyShown => "already_shown",
            Self::BelowThreshold => "below_threshold",
            Self::OverLimit => "over_limit",
            Self::OverBudget => "over_budget",
        }
    }
}

/// What the rules say about one hit, before the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Ordinary,
    Lesson,
    Held(Hold),
}

/// Judges every hit, in order. A hit passes when its similarity is at least
/// `min_similarity`, or when the entity channel found it and its similarity
/// is at least `entity_min_similarity`; at most `entity_max` hits pass by the
/// second rule alone, since a prompt naming the project would otherwise pull
/// in every fact linked to it. With `lessons` on, a lesson has its own rule:
/// at least [`LESSON_MIN_SIMILARITY`], at most [`LESSON_MAX`] of them.
fn judge(
    hits: Vec<Hit>,
    settings: &Settings,
    shown: &HashSet<String>,
    lessons: bool,
) -> Vec<(Hit, Verdict)> {
    let mut by_entity = 0;
    let mut lesson_count = 0;
    hits.into_iter()
        .map(|h| {
            let verdict = if shown.contains(&h.id) {
                Verdict::Held(Hold::AlreadyShown)
            } else if let Some(sim) = h.similarity {
                if lessons && h.is_lesson() {
                    if sim < LESSON_MIN_SIMILARITY.min(settings.min_similarity) {
                        Verdict::Held(Hold::BelowThreshold)
                    } else if lesson_count >= LESSON_MAX {
                        Verdict::Held(Hold::OverLimit)
                    } else {
                        lesson_count += 1;
                        Verdict::Lesson
                    }
                } else if sim >= settings.min_similarity {
                    Verdict::Ordinary
                } else if h.channels.iter().any(|c| c == "entity")
                    && sim >= settings.entity_min_similarity
                {
                    if by_entity < settings.entity_max {
                        by_entity += 1;
                        Verdict::Ordinary
                    } else {
                        Verdict::Held(Hold::OverLimit)
                    }
                } else {
                    Verdict::Held(Hold::BelowThreshold)
                }
            } else {
                Verdict::Held(Hold::BelowThreshold)
            };
            (h, verdict)
        })
        .collect()
}

/// The hits worth showing, in order, not shown yet in this session; see
/// [`judge`] for the rules. Lessons are not told apart here.
pub fn select(hits: Vec<Hit>, settings: &Settings, shown: &HashSet<String>) -> Vec<Hit> {
    judge(hits, settings, shown, false)
        .into_iter()
        .filter(|(_, v)| *v == Verdict::Ordinary)
        .map(|(h, _)| h)
        .collect()
}

/// Holds back the ordinary hits after the first [`ORDINARY_WINDOW`]: only
/// lessons are taken from further down the list.
fn limit_ordinary(judged: &mut [(Hit, Verdict)]) {
    for (_, verdict) in judged.iter_mut().skip(ORDINARY_WINDOW) {
        if *verdict == Verdict::Ordinary {
            *verdict = Verdict::Held(Hold::OverLimit);
        }
    }
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
    let (text, ids, _) = render_prompt_with_lessons(hits, &[], budget);
    (text, ids)
}

/// [`render_prompt`] with `lessons`. The lessons are fitted first, each cut
/// to [`LESSON_LINE_CHARS`] characters, so that a long ordinary fact cannot
/// crowd them out; the ordinary lines get what is left of the budget. They
/// are printed in the other order: ordinary lines, then lessons. Returns the
/// text, the ids printed and the ids of lessons that were cut.
pub fn render_prompt_with_lessons(
    hits: &[Hit],
    lessons: &[Hit],
    budget: usize,
) -> (String, Vec<String>, Vec<String>) {
    let head = PROMPT_HEADER.chars().count() + 1;
    let full: Vec<String> = lessons.iter().map(Hit::lesson_line).collect();
    let capped: Vec<String> = full.iter().map(|l| cut(l, LESSON_LINE_CHARS)).collect();
    let lesson_lines = fit(&capped, head, budget);
    let used = head
        + lesson_lines
            .iter()
            .map(|l| l.chars().count() + 1)
            .sum::<usize>();
    let ordinary: Vec<String> = hits.iter().map(Hit::line).collect();
    let ordinary_lines = fit(&ordinary, used, budget);
    if ordinary_lines.is_empty() && lesson_lines.is_empty() {
        return (String::new(), Vec::new(), Vec::new());
    }
    let mut ids: Vec<String> = hits
        .iter()
        .take(ordinary_lines.len())
        .map(|h| h.id.clone())
        .collect();
    let mut cut_ids = Vec::new();
    for (i, h) in lessons.iter().take(lesson_lines.len()).enumerate() {
        ids.push(h.id.clone());
        if lesson_lines[i] != full[i] {
            cut_ids.push(h.id.clone());
        }
    }
    let lines: Vec<&str> = ordinary_lines
        .iter()
        .chain(&lesson_lines)
        .map(String::as_str)
        .collect();
    (
        format!("{PROMPT_HEADER}\n{}\n", lines.join("\n")),
        ids,
        cut_ids,
    )
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

/// Longest lessons block of the briefing, in characters.
const LESSONS_BLOCK_MAX: usize = 600;
/// How many recent lessons the briefing lists.
const LESSONS_LISTED: usize = 3;
/// Characters of a lesson in the briefing's list.
const LESSON_LIST_CHARS: usize = 90;

const LESSON_HINT: &str = "Record a lesson with the `lesson` tool when the user corrects you, a \
     diagnosis turns out wrong, a review finds a defect in your work, or you revert; phrase it \
     through the situation that should trigger it.";

/// The lessons block of the briefing: the counts, the newest lessons and
/// when to record one; only the last with no lessons. At most `room`
/// characters (and [`LESSONS_BLOCK_MAX`]): the list is cut from its end,
/// then the counts, then the block is left out. Empty when nothing fits.
pub fn render_lessons_block(project: &[Hit], user: &[Hit], room: usize) -> String {
    let room = room.min(LESSONS_BLOCK_MAX);
    let mut newest: Vec<&Hit> = project.iter().chain(user).collect();
    // Handles grow with time.
    newest.sort_by_key(|h| std::cmp::Reverse(handle_number(&h.id)));
    newest.truncate(LESSONS_LISTED);
    let lines: Vec<String> = newest
        .iter()
        .map(|h| {
            format!(
                "- {} [{}] {}",
                h.id,
                h.recorded,
                cut(&h.lesson_body(), LESSON_LIST_CHARS)
            )
        })
        .collect();
    if !project.is_empty() || !user.is_empty() {
        let head = format!(
            "Lessons: {} for this project, {} for you; latest:",
            project.len(),
            user.len()
        );
        for keep in (0..=lines.len()).rev() {
            let mut parts = vec![head.as_str()];
            parts.extend(lines[..keep].iter().map(String::as_str));
            parts.push(LESSON_HINT);
            let text = parts.join("\n");
            if text.chars().count() <= room {
                return text;
            }
        }
    }
    if LESSON_HINT.chars().count() <= room {
        LESSON_HINT.to_owned()
    } else {
        String::new()
    }
}

fn handle_number(id: &str) -> u64 {
    id.trim_start_matches('f').parse().unwrap_or(0)
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
        crate::private::create_own_dir_all(home, dir)?;
    }
    let mut file = crate::private::file_options()
        .create(true)
        .append(true)
        .open(path)?;
    crate::private::tighten_file(&file);
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
    prompt_traced(peer, input, settings, home)
        .await
        .map(|(text, _)| text)
}

/// [`prompt`] and what it decided about each fact it looked at.
async fn prompt_traced(
    peer: &Peer<RoleClient>,
    input: &HookInput,
    settings: &Settings,
    home: &Path,
) -> Result<(String, Vec<Candidate>), String> {
    if skip_prompt(&input.prompt) {
        return Ok((String::new(), Vec::new()));
    }
    let query: String = input.prompt.trim().chars().take(MAX_QUERY_CHARS).collect();
    let found = structured(
        peer,
        "recall",
        json!({ "query": query, "limit": RECALL_LIMIT }),
    )
    .await?;
    let shown = read_shown(home, &input.session_id);
    let mut judged = judge(parse_hits(&found), settings, &shown, true);
    limit_ordinary(&mut judged);
    let pick = |wanted: Verdict| -> Vec<Hit> {
        judged
            .iter()
            .filter(|(_, v)| *v == wanted)
            .map(|(h, _)| h.clone())
            .collect()
    };
    let (ordinary, lessons) = (pick(Verdict::Ordinary), pick(Verdict::Lesson));
    let (text, ids, cut_ids) = render_prompt_with_lessons(&ordinary, &lessons, settings.budget);
    if !ids.is_empty()
        && let Err(e) = append_shown(home, &input.session_id, &ids)
    {
        logln!("tessar hook: cannot record shown facts: {e}");
    }
    let candidates = judged
        .into_iter()
        .map(|(h, verdict)| {
            let hold = match verdict {
                Verdict::Held(hold) => Some(hold),
                _ if ids.contains(&h.id) => None,
                _ => Some(Hold::OverBudget),
            };
            Candidate {
                lesson: h.is_lesson(),
                cut: cut_ids.contains(&h.id),
                id: h.id,
                space: h.space,
                channels: h.channels,
                similarity: h.similarity,
                decision: if hold.is_some() { "held" } else { "shown" },
                reason: hold.map(Hold::name),
            }
        })
        .collect();
    Ok((text, candidates))
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
    session_traced(peer, input, project, settings, home)
        .await
        .map(|(text, _)| text)
}

/// The lessons of one space; a failure counts as none, since the lessons
/// are an extra on top of the briefing.
async fn lessons_in(peer: &Peer<RoleClient>, space: &str) -> Vec<Hit> {
    let args = json!({ "entity": {
        "key": lesson::ENTITY_KEY, "kind": lesson::ENTITY_KIND, "space": space,
    } });
    match structured(peer, "about", args).await {
        Ok(found) => parse_hits(&found)
            .into_iter()
            .filter(Hit::is_lesson)
            .collect(),
        Err(e) => {
            logln!("tessar hook: no lessons from the {space} space: {e}");
            Vec::new()
        }
    }
}

/// [`session`] and the record of the run for the decision log.
async fn session_traced(
    peer: &Peer<RoleClient>,
    input: &HookInput,
    project: &str,
    settings: &Settings,
    home: &Path,
) -> Result<(String, SessionRecord), String> {
    let started = Instant::now();
    reset_shown(home, &input.session_id);
    // Lessons have their own block; they are not listed twice.
    let newest_first = |found: Value| {
        let mut hits: Vec<Hit> = parse_hits(&found)
            .into_iter()
            .filter(|h| !h.is_lesson())
            .collect();
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
    // The lessons block is built first, with a room of its own, so that a
    // full briefing cannot crowd it out; the briefing gets the rest.
    let lessons_project = lessons_in(peer, "project").await;
    let lessons_user = lessons_in(peer, "user").await;
    let block = render_lessons_block(
        &lessons_project,
        &lessons_user,
        LESSONS_BLOCK_MAX.min(settings.session_budget / 5),
    );
    // The block's closing newline is part of what it costs.
    let lessons_chars = if block.is_empty() {
        0
    } else {
        block.chars().count() + 1
    };
    let mut text = if user.is_empty()
        && project_hits.is_empty()
        && !(lessons_project.is_empty() && lessons_user.is_empty())
    {
        // Not an empty memory: it holds lessons, which have their own block.
        format!(
            "tessar memory has no facts about you or project `{project}` yet. Call `remember` \
             to save facts: about the user to the `user` space, about this project to the \
             `project` space.\n"
        )
    } else {
        render_briefing(
            project,
            &user,
            &project_hits,
            settings.session_budget.saturating_sub(lessons_chars),
        )
    };
    let briefing_chars = text.chars().count();
    if !block.is_empty() {
        text.push_str(&block);
        text.push('\n');
    }
    let record = SessionRecord {
        ts: decisions::now(),
        session: input.session_id.clone(),
        project: Some(project.to_owned()),
        source: input.source.clone(),
        elapsed_ms: elapsed_ms(started),
        error: None,
        blocks: vec![
            Block {
                name: "briefing",
                chars: briefing_chars,
            },
            Block {
                name: "lessons",
                chars: lessons_chars,
            },
        ],
        printed_chars: text.chars().count(),
        lessons_project: lessons_project.len(),
        lessons_user: lessons_user.len(),
    };
    Ok((text, record))
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
async fn with_daemon<T, F, Fut>(
    home: &Path,
    input: &HookInput,
    wait: Duration,
    work: F,
) -> Result<T, String>
where
    F: FnOnce(Peer<RoleClient>, String) -> Fut,
    Fut: std::future::Future<Output = Result<T, String>>,
{
    let (project, cwd) = project_and_cwd(&input.cwd);
    let attempt = async {
        crate::private::create_home(home)
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

/// What a hook run produced: the text for Claude Code, a failure to log and
/// the record for the decision log. The binary prints the text, then calls
/// [`HookRun::finish`]; the record is written only after the output is out.
#[derive(Debug)]
pub struct HookRun {
    /// What to print: empty on a failure.
    pub output: String,
    /// Why the run failed, if it did; it goes to stderr, never to the model.
    pub error: Option<String>,
    /// The line for the decision log, when the run reached a decision.
    pub record: Option<Record>,
}

impl HookRun {
    fn failed(error: String, record: Option<Record>) -> Self {
        Self {
            output: String::new(),
            error: Some(error),
            record,
        }
    }

    /// Hands the output to `print` (nothing on a failure, which is logged to
    /// stderr instead), then appends the record, ignoring any error of its.
    pub fn finish(self, home: &Path, print: impl FnOnce(&str)) {
        match &self.error {
            Some(e) => logln!("tessar hook: {e}"),
            None => print(&self.output),
        }
        if let Some(record) = &self.record {
            decisions::append(home, record);
        }
    }

    /// The output, or the failure.
    pub fn into_result(self) -> Result<String, String> {
        match self.error {
            Some(e) => Err(e),
            None => Ok(self.output),
        }
    }
}

/// `tessar hook prompt` over the daemon in `home`.
pub async fn run_prompt(home: &Path, stdin: &str, settings: &Settings) -> HookRun {
    run_prompt_recorded(home, stdin, settings, PROMPT_WAIT, &proxy::own_daemon()).await
}

/// [`run_prompt`] with its time limit and the way to start a daemon spelled
/// out, returning only the text.
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
    run_prompt_recorded(home, stdin, settings, cap, spawn)
        .await
        .into_result()
}

/// [`run_prompt_with`] with the record of the run: a decision, a deliberate
/// silence or a failure (the latter with its reason and the time it took).
#[doc(hidden)]
pub async fn run_prompt_recorded(
    home: &Path,
    stdin: &str,
    settings: &Settings,
    cap: Duration,
    spawn: &proxy::Spawn,
) -> HookRun {
    let started = Instant::now();
    let input = match HookInput::parse(stdin) {
        Ok(input) => input,
        Err(e) => return HookRun::failed(e, None),
    };
    let record = |project, skipped, printed: &str, candidates, error| {
        Record::Prompt(PromptRecord {
            ts: decisions::now(),
            session: input.session_id.clone(),
            project,
            message_chars: input.prompt.chars().count(),
            message: decisions::clip_message(&input.prompt),
            thresholds: Thresholds {
                min_similarity: settings.min_similarity,
                entity_min_similarity: settings.entity_min_similarity,
                entity_max: settings.entity_max,
                lesson_min_similarity: LESSON_MIN_SIMILARITY,
                lesson_max: LESSON_MAX,
                budget: settings.budget,
            },
            elapsed_ms: elapsed_ms(started),
            skipped,
            error,
            printed_chars: printed.chars().count(),
            candidates,
        })
    };
    if let Some(why) = skip_reason(&input.prompt) {
        return HookRun {
            output: String::new(),
            error: None,
            record: Some(record(None, Some(why), "", Vec::new(), None)),
        };
    }
    let hook_input = &input;
    let attempt = async {
        let Ok(stream) = tokio::net::UnixStream::connect(socket_path(home)).await else {
            return Err((None, start_for_the_next_prompt(home, spawn)));
        };
        // `git` runs off the runtime's threads so that the limit holds
        // even if it hangs.
        let named = {
            let cwd = hook_input.cwd.clone();
            tokio::task::spawn_blocking(move || project_and_cwd(&cwd))
        };
        let (project, cwd) = named
            .await
            .map_err(|e| (None, format!("internal error: {e}")))?;
        let known = Some(project.clone());
        let work = async {
            let opened = proxy::open(stream, &project, &cwd).await?;
            note_build(opened.hello());
            let (server, link) = opened.serve().await?;
            let out = prompt_traced(&server.daemon(), hook_input, settings, home).await;
            drop(link);
            out
        };
        match work.await {
            Ok((text, candidates)) => Ok((text, candidates, project)),
            Err(e) => Err((known, e)),
        }
    };
    let result = tokio::time::timeout(cap, attempt)
        .await
        .unwrap_or_else(|_| {
            Err((
                None,
                format!("no answer from the daemon within {} ms", cap.as_millis()),
            ))
        });
    match result {
        Ok((text, candidates, project)) => HookRun {
            record: Some(record(Some(project), None, &text, candidates, None)),
            output: text,
            error: None,
        },
        Err((project, e)) => HookRun::failed(
            e.clone(),
            Some(record(project, None, "", Vec::new(), Some(e))),
        ),
    }
}

/// Starts a daemon for `home` and does not wait for it. Returns what to
/// tell the log.
fn start_for_the_next_prompt(home: &Path, spawn: &proxy::Spawn) -> String {
    if lifecycle::journal_in_use(home) {
        return "no daemon is listening yet, but one holds the journal (it is starting)".to_owned();
    }
    let started = crate::private::create_home(home)
        .map_err(|e| format!("cannot create {}: {e}", home.display()))
        .and_then(|()| spawn(home, proxy::Start::Fresh));
    match started {
        // Dropping the handle leaves the daemon running.
        Ok(_) => "no daemon is listening; started one for the next prompt".to_owned(),
        Err(e) => format!("no daemon is listening, and starting one failed: {e}"),
    }
}

/// `tessar hook session` over the daemon in `home`.
pub async fn run_session(home: &Path, stdin: &str, settings: &Settings) -> HookRun {
    let started = Instant::now();
    let input = match HookInput::parse(stdin) {
        Ok(input) => input,
        Err(e) => return HookRun::failed(e, None),
    };
    let hook_input = &input;
    let ran = with_daemon(home, hook_input, SESSION_WAIT, |peer, project| async move {
        session_traced(&peer, hook_input, &project, settings, home).await
    })
    .await;
    match ran {
        Ok((output, record)) => HookRun {
            output,
            error: None,
            record: Some(Record::Session(record)),
        },
        Err(e) => {
            let record = Record::Session(SessionRecord {
                ts: decisions::now(),
                session: input.session_id.clone(),
                project: None,
                source: input.source.clone(),
                elapsed_ms: elapsed_ms(started),
                error: Some(e.clone()),
                blocks: Vec::new(),
                printed_chars: 0,
                lessons_project: 0,
                lessons_user: 0,
            });
            HookRun::failed(e, Some(record))
        }
    }
}

/// [`session`] with the record of the run, for tests.
#[doc(hidden)]
pub async fn session_recorded(
    peer: &Peer<RoleClient>,
    input: &HookInput,
    project: &str,
    settings: &Settings,
    home: &Path,
) -> Result<HookRun, String> {
    let (output, record) = session_traced(peer, input, project, settings, home).await?;
    Ok(HookRun {
        output,
        error: None,
        record: Some(Record::Session(record)),
    })
}

fn elapsed_ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
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

    #[test]
    fn the_hooks_directory_and_shown_files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let home = tempfile::tempdir().unwrap();
        std::fs::write(crate::purge::journal_path(home.path()), b"").unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        append_shown(home.path(), "s1", &["a".to_owned()]).unwrap();
        let file = shown_path(home.path(), "s1").unwrap();
        let dir = home.path().join("hooks");
        assert_eq!(mode(&dir), 0o700);
        assert_eq!(mode(&file), 0o600);
        // What an older build left behind is tightened.
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
        append_shown(home.path(), "s1", &["b".to_owned()]).unwrap();
        assert_eq!(mode(&dir), 0o700);
        assert_eq!(mode(&file), 0o600);
    }

    #[test]
    fn a_foreign_hooks_directory_is_not_tightened() {
        use std::os::unix::fs::PermissionsExt;
        // A home without tessar's journal, such as a repository.
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join("hooks");
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        append_shown(home.path(), "s1", &["a".to_owned()]).unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&dir), 0o755);
        // Once the home holds the journal it is tessar's.
        std::fs::write(crate::purge::journal_path(home.path()), b"").unwrap();
        append_shown(home.path(), "s1", &["b".to_owned()]).unwrap();
        assert_eq!(mode(&dir), 0o700);
    }

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

    fn lesson(id: &str, similarity: Option<f64>, body: &str) -> Hit {
        hit(
            id,
            similarity,
            &["meaning"],
            &format!("Lesson (2026-11-09): {body}"),
        )
    }

    fn verdicts(hits: Vec<Hit>, shown: &HashSet<String>) -> Vec<(String, Verdict)> {
        judge(hits, &Settings::default(), shown, true)
            .into_iter()
            .map(|(h, v)| (h.id, v))
            .collect()
    }

    #[test]
    fn a_lesson_passes_below_the_general_threshold_and_an_ordinary_fact_does_not() {
        let hits = vec![
            lesson("f1", Some(0.40), "read the log first"),
            hit("f2", Some(0.40), &["meaning"], "an ordinary fact"),
            lesson("f3", Some(0.30), "too far"),
        ];
        assert_eq!(
            verdicts(hits, &HashSet::new()),
            [
                ("f1".to_owned(), Verdict::Lesson),
                ("f2".to_owned(), Verdict::Held(Hold::BelowThreshold)),
                ("f3".to_owned(), Verdict::Held(Hold::BelowThreshold)),
            ]
        );
    }

    #[test]
    fn at_most_two_lessons_and_none_repeated() {
        let hits = vec![
            lesson("f1", Some(0.9), "a"),
            lesson("f2", Some(0.8), "b"),
            lesson("f3", Some(0.7), "c"),
            lesson("f4", Some(0.6), "d"),
        ];
        let shown: HashSet<String> = ["f1".to_owned()].into();
        assert_eq!(
            verdicts(hits, &shown),
            [
                ("f1".to_owned(), Verdict::Held(Hold::AlreadyShown)),
                ("f2".to_owned(), Verdict::Lesson),
                ("f3".to_owned(), Verdict::Lesson),
                ("f4".to_owned(), Verdict::Held(Hold::OverLimit)),
            ]
        );
    }

    #[test]
    fn the_entity_quota_is_named_over_limit() {
        let hits: Vec<Hit> = (1..=3)
            .map(|n| hit(&format!("f{n}"), Some(0.40), &["entity"], "named"))
            .collect();
        let v = verdicts(hits, &HashSet::new());
        assert_eq!(v[2].1, Verdict::Held(Hold::OverLimit));
    }

    #[test]
    fn lesson_lines_start_with_the_handle_and_drop_the_prefix() {
        let l = lesson("f88", Some(0.5), "before declaring a bug, read the log");
        let (text, ids, _) =
            render_prompt_with_lessons(&[hit("f1", Some(0.9), &[], "Ordinary.")], &[l], 1500);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[1], "- [2026-11-08] [user] Ordinary.");
        assert_eq!(
            lines[2],
            "- lesson f88 [2026-11-08]: before declaring a bug, read the log"
        );
        assert_eq!(ids, ["f1", "f88"]);
        assert!(!text.contains("Lesson ("));
    }

    #[test]
    fn the_budget_holds_with_lessons_and_cuts_them_first() {
        let ordinary = vec![hit("f1", Some(0.9), &[], "Ordinary fact.")];
        let lessons = vec![
            lesson("f2", Some(0.5), &"long ".repeat(60)),
            lesson("f3", Some(0.5), "second"),
        ];
        for budget in [60, 100, 200, 400, 1500] {
            let (text, ids, _) = render_prompt_with_lessons(&ordinary, &lessons, budget);
            assert!(text.chars().count() <= budget, "{budget}: {text}");
            if ids.contains(&"f1".to_owned()) {
                assert_eq!(ids[0], "f1", "the ordinary fact is printed first");
            }
        }
        let (_, ids, _) = render_prompt_with_lessons(&ordinary, &lessons, 120);
        assert_eq!(ids, ["f2"], "the lesson is fitted first, even if cut");
    }

    fn lessons_n(n: usize) -> Vec<Hit> {
        (0..n)
            .map(|i| {
                lesson(
                    &format!("f{}", 10 + i),
                    None,
                    &format!("when the situation {i} arises, do the right thing"),
                )
            })
            .collect()
    }

    #[test]
    fn the_lessons_block_with_zero_two_and_five_lessons() {
        let zero = render_lessons_block(&[], &[], 600);
        assert_eq!(zero, LESSON_HINT);
        assert!(!zero.contains("Lessons:"));

        let two = render_lessons_block(&lessons_n(2), &[], 600);
        let lines: Vec<&str> = two.lines().collect();
        assert_eq!(lines[0], "Lessons: 2 for this project, 0 for you; latest:");
        // Newest first, handle first, no prefix.
        assert!(
            lines[1].starts_with("- f11 [2026-11-08] when the situation 1"),
            "{two}"
        );
        assert!(lines[2].starts_with("- f10 "), "{two}");
        assert_eq!(lines[3], LESSON_HINT);
        assert!(!two.contains("Lesson ("));

        let five = render_lessons_block(&lessons_n(3), &lessons_n(2), 600);
        assert!(five.contains("3 for this project, 2 for you"));
        assert_eq!(five.lines().filter(|l| l.starts_with("- f")).count(), 3);
        assert!(five.chars().count() <= 600, "{}", five.chars().count());
    }

    #[test]
    fn the_lessons_block_is_cut_to_the_room_and_hint_goes_last() {
        let all = render_lessons_block(&lessons_n(3), &lessons_n(2), 600);
        let tight = all.chars().count() - 10;
        let cut_list = render_lessons_block(&lessons_n(3), &lessons_n(2), tight);
        assert!(cut_list.chars().count() <= tight);
        assert!(cut_list.ends_with(LESSON_HINT));
        assert!(cut_list.lines().filter(|l| l.starts_with("- f")).count() < 3);
        let only_hint = render_lessons_block(&lessons_n(3), &[], LESSON_HINT.chars().count());
        assert_eq!(only_hint, LESSON_HINT);
        assert_eq!(render_lessons_block(&lessons_n(3), &[], 20), "");
    }

    fn words(n: usize) -> String {
        "word ".repeat(n)
    }

    #[test]
    fn a_top_lesson_survives_a_long_ordinary_line_that_does_not_fit() {
        let ordinary = vec![
            hit("f1", Some(0.9), &[], &words(100)),
            hit("f2", Some(0.8), &[], &words(300)),
            hit("f3", Some(0.7), &[], "Short."),
        ];
        let lessons = vec![lesson(
            "f88",
            Some(0.95),
            "before declaring a bug, read the log",
        )];
        let (text, ids, cut_ids) = render_prompt_with_lessons(&ordinary, &lessons, 1500);
        // The second ordinary line does not fit, so the rest of the ordinary
        // lines go; the lesson stays and is printed last.
        assert_eq!(ids, ["f1", "f88"], "{text}");
        assert!(cut_ids.is_empty());
        assert!(text.lines().last().unwrap().starts_with("- lesson f88 "));
        assert!(text.chars().count() <= 1500);
    }

    #[test]
    fn long_lessons_are_cut_to_their_line_and_ordinary_lines_get_the_rest() {
        let lessons = vec![
            lesson("f10", Some(0.9), &words(200)),
            lesson("f11", Some(0.8), &words(200)),
        ];
        let ordinary = vec![hit("f1", Some(0.9), &[], &words(100))];
        let (text, ids, cut_ids) = render_prompt_with_lessons(&ordinary, &lessons, 1500);
        assert_eq!(ids, ["f1", "f10", "f11"]);
        assert_eq!(cut_ids, ["f10", "f11"]);
        let lines: Vec<&str> = text.lines().collect();
        for l in &lines[2..] {
            assert_eq!(l.chars().count(), LESSON_LINE_CHARS, "{l}");
            assert!(l.ends_with('…'));
        }
        // Ordinary lines lose only what the lessons took.
        let head = PROMPT_HEADER.chars().count() + 1;
        let taken = 2 * (LESSON_LINE_CHARS + 1);
        let ord = ordinary[0].line().chars().count() + 1;
        let tight = head + taken + ord;
        let (_, ids, _) = render_prompt_with_lessons(&ordinary, &lessons, tight);
        assert_eq!(ids, ["f1", "f10", "f11"]);
        // A little less: the ordinary line is cut to the room left, or, with
        // less than MIN_CUT, dropped.
        let (text, ids, _) = render_prompt_with_lessons(&ordinary, &lessons, tight - 1);
        assert_eq!(ids, ["f1", "f10", "f11"]);
        assert!(text.chars().count() < tight && text.contains('…'));
        let (_, ids, _) = render_prompt_with_lessons(&ordinary, &lessons, head + taken + 10);
        assert_eq!(ids, ["f10", "f11"]);
    }

    #[test]
    fn a_multibyte_lesson_is_cut_on_a_character_boundary() {
        let l = lesson(
            "f5",
            Some(0.9),
            &"перед тем как объявить баг, прочитай лог ".repeat(20),
        );
        let (text, _, cut_ids) = render_prompt_with_lessons(&[], &[l], 1500);
        assert_eq!(cut_ids, ["f5"]);
        let line = text.lines().last().unwrap();
        assert_eq!(line.chars().count(), LESSON_LINE_CHARS);
        assert!(line.ends_with('…'));
    }

    #[test]
    fn without_lessons_the_prompt_is_what_it_was() {
        let ordinary = vec![
            hit("f1", Some(0.9), &[], &words(100)),
            hit("f2", Some(0.8), &[], &words(300)),
        ];
        let (text, ids, cut_ids) = render_prompt_with_lessons(&ordinary, &[], 1500);
        assert_eq!((text.clone(), ids.clone()), render_prompt(&ordinary, 1500));
        assert_eq!(ids, ["f1"]);
        assert!(cut_ids.is_empty());
    }
}
