//! Merging the tessar hooks into the text of `settings.json`.
//!
//! The file is never parsed into a value and printed again: that would sort
//! keys, reformat numbers and drop what it does not understand. Instead the
//! text is scanned for the byte ranges that matter (with
//! `serde_json::value::RawValue`, which hands out slices of the input) and
//! only those ranges are replaced:
//!
//! - the string of a tessar hook command whose path is stale: only the word
//!   that is the path changes, and the JSON string is written again;
//! - a missing group or event: inserted after the last item of its container,
//!   pretty-printed with two-space steps, as `tessar hook config` prints it;
//! - a missing `hooks` member: inserted after the last member.
//!
//! Everything else is byte for byte what it was: other settings, other hooks,
//! key order, numbers, escapes, white space. What can still change is the
//! text of an updated command (its JSON escapes are written afresh) and the
//! white space inside a container that was empty when an item is added to it.
//!
//! A file with duplicate keys in an object this code reads (the top level,
//! `hooks`, and the groups and hook entries inside it) is refused, since
//! parsers disagree about which of two equal keys counts.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;
use serde_json::value::RawValue;

use crate::hook::{config_snippet, shell_word};

/// The result of a merge.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Merged {
    /// The new text; equal to the old one when nothing changed.
    pub text: String,
    /// Events that got a tessar hook.
    pub added: Vec<String>,
    /// Events whose tessar hook now points to this binary.
    pub updated: Vec<String>,
    /// Events from which a duplicate of an updated hook was removed.
    pub removed: Vec<String>,
    /// Events with a hook that looks like tessar's but is in a form this code
    /// does not rewrite (a wrapper script, a redirect, a pipe). Left as it is,
    /// and no second hook is added next to it.
    pub conflicts: Vec<String>,
}

impl Merged {
    pub fn changed(&self) -> bool {
        !self.added.is_empty() || !self.updated.is_empty() || !self.removed.is_empty()
    }

    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if !self.added.is_empty() {
            parts.push(format!("add tessar hooks for {}", self.added.join(", ")));
        }
        if !self.updated.is_empty() {
            parts.push(format!(
                "point the tessar hooks for {} to this binary",
                self.updated.join(", ")
            ));
        }
        if !self.removed.is_empty() {
            parts.push(format!(
                "remove duplicate tessar hooks for {}",
                self.removed.join(", ")
            ));
        }
        parts.join(" and ")
    }
}

/// Merges the hooks of `tessar hook config` for `exe` into `text`, the content
/// of `settings.json` (`None` or blank: no file). Nothing is changed on `Err`.
#[cfg(test)]
pub(super) fn merge(text: Option<&str>, exe: &Path) -> Result<Merged, String> {
    merge_for(text, exe, None)
}

/// [`merge`] where `home` is what `~` and `$HOME` mean in a hook command.
pub(super) fn merge_for(
    text: Option<&str>,
    exe: &Path,
    home: Option<&Path>,
) -> Result<Merged, String> {
    let merged = merge_once(text, exe, home)?;
    if !merged.changed() {
        return Ok(merged);
    }
    // What was written must read as intended: the same other settings, and
    // nothing left to do.
    let again = merge_once(Some(&merged.text), exe, home)
        .map_err(|e| format!("internal error: the merged text does not parse ({e})"))?;
    if again.changed() || other_members(text, &merged.text) != Some(true) {
        return Err("internal error: the merged text is not what was meant".to_owned());
    }
    Ok(merged)
}

/// Whether every top-level member but `hooks` is the same text in both.
fn other_members(old: Option<&str>, new: &str) -> Option<bool> {
    let Some(old) = old.filter(|t| !t.trim().is_empty()) else {
        return Some(true);
    };
    let strip = |text: &str| -> Option<Vec<(String, String)>> {
        let root: &RawValue = serde_json::from_str(text).ok()?;
        let members = members(root.get()).ok()?;
        Some(
            members
                .into_iter()
                .filter(|(k, _)| k != "hooks")
                .map(|(k, v)| (k, v.get().to_owned()))
                .collect(),
        )
    };
    Some(strip(old)? == strip(new)?)
}

type Edit = (usize, usize, String);

fn merge_once(text: Option<&str>, exe: &Path, home: Option<&Path>) -> Result<Merged, String> {
    let snippet: Value =
        serde_json::from_str(&config_snippet(exe)).map_err(|e| format!("internal error: {e}"))?;
    let wanted = snippet
        .get("hooks")
        .and_then(Value::as_object)
        .ok_or("internal error: the hooks snippet has no `hooks`")?;

    let Some(text) = text.filter(|t| !t.trim().is_empty()) else {
        return Ok(Merged {
            text: config_snippet(exe),
            added: wanted.keys().cloned().collect(),
            ..Merged::default()
        });
    };

    let root: &RawValue = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if !root.get().starts_with('{') {
        return Err("the top level is not a JSON object".to_owned());
    }
    let top = members(root.get())?;
    let mut merged = Merged::default();
    let mut edits: Vec<Edit> = Vec::new();

    let Some((_, hooks)) = top.iter().find(|(k, _)| k == "hooks") else {
        // No `hooks`: the whole object goes in as one more member.
        let member = format!(
            "\"hooks\": {}",
            indent_after_first(&pretty(&Value::Object(wanted.clone())), 2)
        );
        edits.push(insertion(text, root.get(), &top, vec![member], 1));
        merged.added = wanted.keys().cloned().collect();
        return Ok(finish(text, edits, merged));
    };
    if !hooks.get().starts_with('{') {
        return Err("`hooks` is not an object".to_owned());
    }
    let events = members(hooks.get())?;
    let mut new_events = Vec::new();
    for (event, groups) in wanted {
        let Some(group) = groups.as_array().and_then(|g| g.first()) else {
            continue;
        };
        let kind = kind_of(event);
        let Some((_, array)) = events.iter().find(|(k, _)| k == event) else {
            new_events.push(format!(
                "\"{event}\": {}",
                indent_after_first(&pretty(groups), 4)
            ));
            merged.added.push(event.clone());
            continue;
        };
        if !array.get().starts_with('[') {
            return Err(format!("`hooks.{event}` is not an array"));
        }
        let groups = items(array.get())?;
        let mut found = false;
        // The commands the entries of this event have or get, and whether
        // one of them was updated.
        let mut seen: Vec<(String, bool)> = Vec::new();
        for group_raw in &groups {
            let group = hook_commands(text, group_raw)?;
            let mut gone: Vec<usize> = Vec::new();
            for entry in group.entries {
                let (result, updated) = match classify(&entry.command_text, kind, exe, home) {
                    Found::Other => continue,
                    Found::InPlace => (entry.command_text.clone(), false),
                    Found::Update(new) => (new, true),
                    Found::Conflict => {
                        found = true;
                        if !merged.conflicts.contains(event) {
                            merged.conflicts.push(event.clone());
                        }
                        continue;
                    }
                };
                found = true;
                // Two entries that end up the same: keep the first.
                if seen.iter().any(|(c, u)| *c == result && (*u || updated)) {
                    gone.push(entry.idx);
                    if !merged.removed.contains(event) {
                        merged.removed.push(event.clone());
                    }
                    continue;
                }
                seen.push((result.clone(), updated));
                if updated {
                    edits.push((entry.command.0, entry.command.1, json_string(&result)));
                    if !merged.updated.contains(event) {
                        merged.updated.push(event.clone());
                    }
                }
            }
            edits.extend(removals(&group.ranges, &gone));
        }
        if !found {
            edits.push(array_insertion(
                text,
                array.get(),
                &groups,
                indent_after_first(&pretty(group), 6),
                3,
            ));
            merged.added.push(event.clone());
        }
    }
    if !new_events.is_empty() {
        edits.push(insertion(text, hooks.get(), &events, new_events, 2));
    }
    Ok(finish(text, edits, merged))
}

fn finish(text: &str, mut edits: Vec<Edit>, mut merged: Merged) -> Merged {
    // From the end, so that earlier offsets stay valid.
    edits.sort_by_key(|(start, _, _)| std::cmp::Reverse(*start));
    let crlf = text.contains("\r\n");
    let mut out = text.to_owned();
    for (start, end, replacement) in edits {
        // New text takes the file's own line ending.
        let replacement = if crlf {
            replacement.replace('\n', "\r\n")
        } else {
            replacement
        };
        out.replace_range(start..end, &replacement);
    }
    merged.text = out;
    merged
}

/// A hook entry: where its `command` string is (byte range of the JSON
/// string), its value, and its place in the `hooks` array of its group.
struct Entry {
    command: (usize, usize),
    command_text: String,
    idx: usize,
}

/// The hook entries of one group and the byte ranges of all items of its
/// `hooks` array (entries that are not objects included).
#[derive(Default)]
struct GroupEntries {
    entries: Vec<Entry>,
    ranges: Vec<(usize, usize)>,
}

/// The edits that cut the items `gone` out of an array whose items are at
/// `ranges`. A run of adjacent items goes as one range, with the comma that
/// separates it from what stays: the one before it, or for a run at the start
/// the one after it.
fn removals(ranges: &[(usize, usize)], gone: &[usize]) -> Vec<Edit> {
    let mut edits = Vec::new();
    let mut i = 0;
    while i < ranges.len() {
        if !gone.contains(&i) {
            i += 1;
            continue;
        }
        let first = i;
        while i + 1 < ranges.len() && gone.contains(&(i + 1)) {
            i += 1;
        }
        let last = i;
        let range = if first > 0 {
            (ranges[first - 1].1, ranges[last].1)
        } else if last + 1 < ranges.len() {
            (ranges[first].0, ranges[last + 1].0)
        } else {
            (ranges[first].0, ranges[last].1)
        };
        edits.push((range.0, range.1, String::new()));
        i += 1;
    }
    edits
}

/// The hook entries of one group. A group or entry that is not an object is
/// not ours to read.
fn hook_commands(text: &str, group: &RawValue) -> Result<GroupEntries, String> {
    if !group.get().starts_with('{') {
        return Ok(GroupEntries::default());
    }
    let Some((_, hooks)) = members(group.get())?
        .into_iter()
        .find(|(k, _)| k == "hooks")
    else {
        return Ok(GroupEntries::default());
    };
    if !hooks.get().starts_with('[') {
        return Ok(GroupEntries::default());
    }
    let items = items(hooks.get())?;
    let mut found = GroupEntries {
        ranges: items
            .iter()
            .map(|e| (offset(text, e.get()), end_of(text, e)))
            .collect(),
        ..GroupEntries::default()
    };
    for (idx, entry) in items.iter().enumerate() {
        if !entry.get().starts_with('{') {
            continue;
        }
        let Some((_, command)) = members(entry.get())?
            .into_iter()
            .find(|(k, _)| k == "command")
        else {
            continue;
        };
        let Ok(command_text) = serde_json::from_str::<String>(command.get()) else {
            continue;
        };
        let at = offset(text, command.get());
        found.entries.push(Entry {
            command: (at, at + command.get().len()),
            command_text,
            idx,
        });
    }
    Ok(found)
}

fn kind_of(event: &str) -> &'static str {
    match event {
        "UserPromptSubmit" => "prompt",
        "SessionStart" => "session",
        "PreToolUse" => "agent",
        _ => "subagent",
    }
}

// ---- scanning ------------------------------------------------------------

struct Members<'a>(Vec<(String, &'a RawValue)>);

impl<'de: 'a, 'a> Deserialize<'de> for Members<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V<'a>(std::marker::PhantomData<&'a ()>);
        impl<'de: 'a, 'a> Visitor<'de> for V<'a> {
            type Value = Members<'a>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Members<'a>, M::Error> {
                let mut out: Vec<(String, &'a RawValue)> = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    let value: &'a RawValue = map.next_value()?;
                    if out.iter().any(|(k, _)| *k == key) {
                        return Err(de::Error::custom(format!("duplicate key `{key}`")));
                    }
                    out.push((key, value));
                }
                Ok(Members(out))
            }
        }
        deserializer.deserialize_map(V(std::marker::PhantomData))
    }
}

struct Items<'a>(Vec<&'a RawValue>);

impl<'de: 'a, 'a> Deserialize<'de> for Items<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V<'a>(std::marker::PhantomData<&'a ()>);
        impl<'de: 'a, 'a> Visitor<'de> for V<'a> {
            type Value = Items<'a>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON array")
            }

            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Items<'a>, S::Error> {
                let mut out = Vec::new();
                while let Some(item) = seq.next_element::<&'a RawValue>()? {
                    out.push(item);
                }
                Ok(Items(out))
            }
        }
        deserializer.deserialize_seq(V(std::marker::PhantomData))
    }
}

/// The members of the object `raw` as slices of it; duplicate keys are an error.
fn members(raw: &str) -> Result<Vec<(String, &RawValue)>, String> {
    serde_json::from_str::<Members<'_>>(raw)
        .map(|m| m.0)
        .map_err(|e| e.to_string())
}

fn items(raw: &str) -> Result<Vec<&RawValue>, String> {
    serde_json::from_str::<Items<'_>>(raw)
        .map(|i| i.0)
        .map_err(|e| e.to_string())
}

/// Where `part`, a slice of `base`, starts.
fn offset(base: &str, part: &str) -> usize {
    part.as_ptr() as usize - base.as_ptr() as usize
}

fn end_of(base: &str, part: &RawValue) -> usize {
    offset(base, part.get()) + part.get().len()
}

// ---- inserting -----------------------------------------------------------

fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).expect("a Value always serialises")
}

fn json_string(text: &str) -> String {
    serde_json::to_string(text).expect("a string always serialises")
}

/// Indents every line but the first by `n` spaces.
fn indent_after_first(text: &str, n: usize) -> String {
    text.replace('\n', &format!("\n{}", " ".repeat(n)))
}

/// Adds `new` (members, already formatted) to the object `container` whose
/// members are `existing`; `depth` is the nesting of the members (1: top level).
fn insertion(
    text: &str,
    container: &str,
    existing: &[(String, &RawValue)],
    new: Vec<String>,
    depth: usize,
) -> Edit {
    let last_end = existing.last().map(|(_, v)| end_of(text, v));
    add_items(text, container, last_end, new, depth)
}

fn array_insertion(
    text: &str,
    container: &str,
    existing: &[&RawValue],
    new: String,
    depth: usize,
) -> Edit {
    let last_end = existing.last().map(|v| end_of(text, v));
    add_items(text, container, last_end, vec![new], depth)
}

fn add_items(
    text: &str,
    container: &str,
    last_end: Option<usize>,
    new: Vec<String>,
    depth: usize,
) -> Edit {
    let indent = " ".repeat(2 * depth);
    match last_end {
        Some(end) => {
            let body: String = new.iter().map(|i| format!(",\n{indent}{i}")).collect();
            (end, end, body)
        }
        None => {
            let open = offset(text, container) + 1;
            let close = " ".repeat(2 * (depth - 1));
            let body = new
                .iter()
                .map(|i| format!("\n{indent}{i}"))
                .collect::<Vec<_>>()
                .join(",");
            (open, open, format!("{body}\n{close}"))
        }
    }
}

// ---- recognising a tessar hook -------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Found {
    /// Not a tessar hook of this kind.
    Other,
    /// Already points to this binary.
    InPlace,
    /// A tessar hook of another path; the new command.
    Update(String),
    /// Looks like one but is in a form that is not rewritten.
    Conflict,
}

struct Token {
    start: usize,
    end: usize,
    value: String,
}

/// Splits a command line into words with their byte ranges. `None` when it
/// uses more of the shell than words, quotes and backslashes (variables,
/// redirects, pipes, globs).
fn tokenize(line: &str) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut chars = line.char_indices().peekable();
    let mut current: Option<Token> = None;
    while let Some((i, c)) = chars.next() {
        let after = i + c.len_utf8();
        match c {
            c if c.is_whitespace() && c != '\n' => {
                tokens.extend(current.take());
            }
            // A word that starts with `~/` or `$HOME/` is a path, not shell syntax.
            ';' | '&' | '|' | '<' | '>' | '(' | ')' | '$' | '`' | '\n' | '*' | '?' | '[' | ']'
            | '{' | '}' | '~' | '#' | '!'
                if !(current.is_none() && starts_with_home(&line[i..])) =>
            {
                return None;
            }
            '\'' | '"' | '\\' => {
                let token = current.get_or_insert(Token {
                    start: i,
                    end: i,
                    value: String::new(),
                });
                match c {
                    '\\' => {
                        let (j, n) = chars.next()?;
                        token.value.push(n);
                        token.end = j + n.len_utf8();
                    }
                    '\'' => {
                        let mut closed = false;
                        for (j, n) in chars.by_ref() {
                            token.end = j + n.len_utf8();
                            if n == '\'' {
                                closed = true;
                                break;
                            }
                            token.value.push(n);
                        }
                        if !closed {
                            return None;
                        }
                    }
                    _ => {
                        let mut closed = false;
                        while let Some((j, n)) = chars.next() {
                            token.end = j + n.len_utf8();
                            match n {
                                '"' => {
                                    closed = true;
                                    break;
                                }
                                '$' if token.value.is_empty() && starts_with_home(&line[j..]) => {
                                    token.value.push('$');
                                }
                                '$' | '`' => return None,
                                '\\' => {
                                    let (k, e) = chars.next()?;
                                    token.end = k + e.len_utf8();
                                    if !matches!(e, '"' | '\\') {
                                        token.value.push('\\');
                                    }
                                    token.value.push(e);
                                }
                                _ => token.value.push(n),
                            }
                        }
                        if !closed {
                            return None;
                        }
                    }
                }
            }
            _ => {
                let token = current.get_or_insert(Token {
                    start: i,
                    end: i,
                    value: String::new(),
                });
                token.value.push(c);
                token.end = after;
            }
        }
    }
    tokens.extend(current);
    Some(tokens)
}

fn starts_with_home(text: &str) -> bool {
    text.starts_with("~/") || text.starts_with("$HOME/")
}

/// `~/x` and `$HOME/x` for the user's `home`; anything else as it is.
fn expand_home(word: &str, home: Option<&Path>) -> PathBuf {
    let rest = word
        .strip_prefix("~/")
        .or_else(|| word.strip_prefix("$HOME/"));
    match (rest, home) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(word),
    }
}

fn is_assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty()
            && !name.starts_with(|c: char| c.is_ascii_digit())
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

fn is_tessar_name(word: &str) -> bool {
    Path::new(word)
        .file_name()
        .is_some_and(|n| n == "tessar" || n == "tessar.exe")
}

fn is_script(path: &Path) -> bool {
    let mut magic = [0u8; 2];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok()
        && &magic == b"#!"
}

/// What `command`, a hook command, is with respect to `tessar hook <kind>`
/// for the binary `exe`. Only the word that is the path is ever replaced; the
/// environment assignments before it and the arguments after it stay.
fn classify(command: &str, kind: &str, exe: &Path, home: Option<&Path>) -> Found {
    let mentions = command.contains("tessar") && command.contains(&format!("hook {kind}"));
    let ambiguous = if mentions {
        Found::Conflict
    } else {
        Found::Other
    };
    let Some(tokens) = tokenize(command) else {
        return ambiguous;
    };
    let Some(at) = tokens.iter().position(|t| !is_assignment(&t.value)) else {
        return Found::Other;
    };
    let (word, args) = (&tokens[at], &tokens[at + 1..]);
    let shaped = is_tessar_name(&word.value)
        && args.first().is_some_and(|a| a.value == "hook")
        && args.get(1).is_some_and(|a| a.value == kind);
    if !shaped {
        // `env X=1 /p/tessar hook prompt` and the like.
        return if mentions
            && tokens
                .iter()
                .any(|t| t.value.split_whitespace().any(is_tessar_name))
        {
            Found::Conflict
        } else {
            Found::Other
        };
    }
    let path = expand_home(&word.value, home);
    if is_script(&path) {
        return Found::Conflict;
    }
    if path == exe {
        return Found::InPlace;
    }
    let mut new = command.to_owned();
    new.replace_range(
        word.start..word.end,
        &shell_word(&exe.display().to_string()),
    );
    Found::Update(new)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXE: &str = "/opt/tessar/bin/tessar";

    fn exe() -> &'static Path {
        Path::new(EXE)
    }

    fn run(text: &str) -> Merged {
        merge(Some(text), exe()).unwrap()
    }

    #[test]
    fn a_missing_file_gets_the_snippet() {
        let m = merge(None, exe()).unwrap();
        assert_eq!(m.text, config_snippet(exe()));
        assert_eq!(m.added.len(), 4);
        assert_eq!(merge(Some(" \n"), exe()).unwrap().text, m.text);
    }

    #[test]
    fn everything_else_is_byte_for_byte() {
        let text = "{\n\t\"big\": 12345678901234567890123,\n  \"f\": 1.0e2, \"s\": \"\\u00e9\\n\\\"x\\\" \u{43f}\u{440}\",\n  \"nested\": {\"b\": [1, {\"z\": null, \"a\": 0.10}], \"a\": {}},\n  \"zeta\": true\n}\n";
        let m = run(text);
        assert_eq!(m.added.len(), 4);
        // Cut out the inserted `hooks` member: the rest is the original.
        let at = m.text.find(",\n  \"hooks\"").unwrap();
        let tail = m.text[at..].find("\n}\n").unwrap();
        let rest = format!("{}{}", &m.text[..at], &m.text[at + tail..]);
        assert_eq!(rest, text);
        for needle in [
            "12345678901234567890123",
            "1.0e2",
            "\\u00e9",
            "0.10",
            "\u{43f}\u{440}",
        ] {
            assert!(m.text.contains(needle), "{needle}");
        }
        let v: Value = serde_json::from_str(&m.text).unwrap();
        assert_eq!(v["hooks"]["SessionStart"][0]["hooks"][0]["timeout"], 60);
        // Idempotent.
        assert!(!run(&m.text).changed());
        assert_eq!(run(&m.text).text, m.text);
    }

    #[test]
    fn an_existing_hooks_member_keeps_its_place_and_neighbours() {
        let text = r#"{
  "a": 1,
  "hooks": {
    "Stop": [ { "hooks": [ { "type": "command", "command": "x" } ] } ],
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "/old/tessar hook prompt" } ] }
    ]
  },
  "z": 9007199254740993
}
"#;
        let m = run(text);
        assert_eq!(m.updated, ["UserPromptSubmit"]);
        assert!(m.text.contains(&format!("\"{EXE} hook prompt\"")));
        assert!(m.text.starts_with("{\n  \"a\": 1,\n  \"hooks\": {\n    \"Stop\": [ { \"hooks\": [ { \"type\": \"command\", \"command\": \"x\" } ] } ],"));
        assert!(
            m.text.ends_with("\n  },\n  \"z\": 9007199254740993\n}\n"),
            "{}",
            m.text
        );
        assert!(m.text.contains("\"SessionStart\"") && m.text.contains("\"SubagentStart\""));
        assert!(!m.text.contains("/old/"));
    }

    #[test]
    fn a_new_group_goes_after_the_foreign_ones() {
        let text =
            r#"{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"mine"}]}]}}"#;
        let m = run(text);
        let mine = m.text.find("\"mine\"").unwrap();
        let ours = m.text.find(&format!("{EXE} hook session")).unwrap();
        assert!(mine < ours);
        assert!(m.text.starts_with(
            r#"{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"mine"}]}"#
        ));
    }

    #[test]
    fn duplicate_keys_are_refused() {
        for text in [
            r#"{"hooks":{},"hooks":{}}"#,
            r#"{"a":1,"a":2}"#,
            r#"{"hooks":{"SessionStart":[],"SessionStart":[]}}"#,
            r#"{"hooks":{"SessionStart":[{"hooks":[],"hooks":[]}]}}"#,
        ] {
            let err = merge(Some(text), exe()).unwrap_err();
            assert!(err.contains("duplicate key"), "{text}: {err}");
        }
    }

    #[test]
    fn surprising_shapes_are_refused() {
        for text in [
            "[1]",
            "{\"hooks\": 3}",
            "{\"hooks\": null}",
            "{\"hooks\": {\"SessionStart\": {}}}",
            "{",
            "{} x",
        ] {
            assert!(merge(Some(text), exe()).is_err(), "{text}");
        }
    }

    #[test]
    fn hook_commands_are_classified() {
        let k = "prompt";
        let c = |s: &str| classify(s, k, exe(), None);
        assert_eq!(c(&format!("{EXE} hook prompt")), Found::InPlace);
        assert_eq!(c(&format!("{EXE} hook prompt --home /x")), Found::InPlace);
        assert_eq!(c("other-tool prompt"), Found::Other);
        assert_eq!(c("/x/not-tessar hook prompt"), Found::Other);
        assert_eq!(c("/p/tessar hook session"), Found::Other);
        // Only the path word is replaced.
        assert_eq!(
            c("TESSAR_HOME=/x /p/tessar hook prompt"),
            Found::Update(format!("TESSAR_HOME=/x {EXE} hook prompt"))
        );
        assert_eq!(
            c("\"/My Apps/tessar\" hook prompt --home /x"),
            Found::Update(format!("{EXE} hook prompt --home /x"))
        );
        assert_eq!(
            c("'/My Apps/it'\\''s/tessar' hook prompt"),
            Found::Update(format!("{EXE} hook prompt"))
        );
        // Forms that are not understood are conflicts, not duplicates.
        assert_eq!(c("/p/tessar hook prompt 2>/dev/null"), Found::Conflict);
        assert_eq!(c("/p/tessar hook prompt | tee /tmp/x"), Found::Conflict);
        assert_eq!(c("env X=1 /p/tessar hook prompt"), Found::Conflict);
        assert_eq!(c("sh -c \"/p/tessar hook prompt\""), Found::Conflict);
        assert_eq!(c("echo $HOME > /tmp/x"), Found::Other);
    }

    #[test]
    fn a_wrapper_script_is_a_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("tessar");
        std::fs::write(&script, "#!/bin/sh\nexec real-tessar \"$@\"\n").unwrap();
        let command = format!("{} hook prompt", script.display());
        assert_eq!(classify(&command, "prompt", exe(), None), Found::Conflict);
        let text = format!(
            "{{\"hooks\":{{\"UserPromptSubmit\":[{{\"hooks\":[{{\"type\":\"command\",\"command\":{}}}]}}]}}}}",
            json_string(&command)
        );
        let m = run(&text);
        assert_eq!(m.conflicts, ["UserPromptSubmit"]);
        assert!(!m.added.contains(&"UserPromptSubmit".to_owned()));
        assert_eq!(m.text.matches(" hook prompt").count(), 1, "no duplicate");
    }

    #[test]
    fn a_stale_hook_with_a_prefix_keeps_it() {
        let text = r#"{"hooks":{"UserPromptSubmit":[{"hooks":[{"type":"command","command":"TESSAR_HOME=/x /p/tessar hook prompt","timeout":5}]}]}}"#;
        let m = run(text);
        assert!(
            m.text
                .contains(&format!("TESSAR_HOME=/x {EXE} hook prompt"))
        );
        assert!(m.text.contains("\"timeout\":5"));
        assert_eq!(m.text.matches("hook prompt").count(), 1);
    }

    #[test]
    fn a_home_relative_path_that_is_this_binary_is_in_place() {
        let home = Some(Path::new("/opt/tessar"));
        let c = |s: &str| classify(s, "prompt", exe(), home);
        assert_eq!(c("~/bin/tessar hook prompt"), Found::InPlace);
        assert_eq!(c("$HOME/bin/tessar hook prompt"), Found::InPlace);
        assert_eq!(c("\"$HOME/bin/tessar\" hook prompt"), Found::InPlace);
        assert_eq!(
            c("~/other/tessar hook prompt"),
            Found::Update(format!("{EXE} hook prompt"))
        );
        // `~` in the middle of a word, or a variable other than HOME, is shell syntax.
        assert_eq!(c("/p/~/tessar hook prompt"), Found::Conflict);
        assert_eq!(c("$X/tessar hook prompt"), Found::Conflict);
        // Without a known home the word is another path.
        assert_eq!(
            classify("~/bin/tessar hook prompt", "prompt", exe(), None),
            Found::Update(format!("{EXE} hook prompt"))
        );
        let text = r#"{"hooks":{"UserPromptSubmit":[{"hooks":[{"type":"command","command":"~/bin/tessar hook prompt"}]}]}}"#;
        let m = merge_for(Some(text), exe(), home).unwrap();
        assert!(m.conflicts.is_empty() && m.updated.is_empty());
        assert!(m.text.contains("~/bin/tessar hook prompt"));
    }

    #[test]
    fn two_stale_entries_for_one_hook_leave_one() {
        let entry = |cmd: &str| format!(r#"{{"type":"command","command":"{cmd}"}}"#);
        // In one group, and across two groups; the first, middle and last.
        for text in [
            format!(
                r#"{{"hooks":{{"SessionStart":[{{"hooks":[{},{},{}]}}]}}}}"#,
                entry("/a/tessar hook session"),
                entry("keep-me"),
                entry("/b/tessar hook session")
            ),
            format!(
                r#"{{"hooks":{{"SessionStart":[{{"hooks":[{}]}},{{"hooks":[{}]}}]}}}}"#,
                entry("/a/tessar hook session"),
                entry("/b/tessar hook session")
            ),
            format!(
                r#"{{"hooks":{{"SessionStart":[{{"hooks":[{},{}]}}]}}}}"#,
                entry("/a/tessar hook session"),
                entry("/b/tessar hook session")
            ),
        ] {
            let m = run(&text);
            assert_eq!(m.text.matches("hook session").count(), 1, "{}", m.text);
            assert!(m.text.contains(&format!("{EXE} hook session")));
            assert_eq!(m.removed, ["SessionStart"]);
            serde_json::from_str::<Value>(&m.text).unwrap();
            assert_eq!(text.contains("keep-me"), m.text.contains("keep-me"));
            assert!(!run(&m.text).changed());
        }
        // Two identical hooks that are already right are the user's business.
        let twice = format!(
            r#"{{"hooks":{{"SessionStart":[{{"hooks":[{},{}]}}]}}}}"#,
            entry(&format!("{EXE} hook session")),
            entry(&format!("{EXE} hook session"))
        );
        assert!(!run(&twice).removed.contains(&"SessionStart".to_owned()));
    }

    #[test]
    fn crlf_files_get_crlf_insertions() {
        let text = "{\r\n  \"a\": 1\r\n}\r\n";
        let m = run(text);
        assert!(
            m.text.starts_with("{\r\n  \"a\": 1,\r\n  \"hooks\""),
            "{:?}",
            m.text
        );
        assert_eq!(m.text.matches('\n').count(), m.text.matches("\r\n").count());
        assert!(!run(&m.text).changed());
    }

    #[test]
    fn a_number_that_a_value_would_reject_is_not_our_business() {
        let m = run(r#"{"x": 1e400, "n": 12345678901234567890123}"#);
        assert!(m.text.contains("1e400") && m.text.contains("12345678901234567890123"));
    }

    #[test]
    fn runs_of_adjacent_duplicates_go_as_one_range() {
        let entry = |cmd: &str| format!(r#"{{"type":"command","command":"{cmd}"}}"#);
        let group = |entries: &[String]| format!(r#"{{"hooks":[{}]}}"#, entries.join(", "));
        let ok = entry(&format!("{EXE} hook session"));
        let stale = |n: &str| entry(&format!("/{n}/tessar hook session"));
        let cases = [
            // A correct hook, then two stale ones in a row.
            vec![
                group(std::slice::from_ref(&ok)),
                group(&[stale("a"), stale("b")]),
            ],
            // Three in a row, alone in the group and after a foreign entry.
            vec![group(&[stale("a"), stale("b"), stale("c")])],
            vec![group(&[
                entry("foreign"),
                stale("a"),
                stale("b"),
                stale("c"),
            ])],
            vec![group(&[
                stale("a"),
                stale("b"),
                stale("c"),
                entry("foreign"),
            ])],
            vec![group(&[
                stale("a"),
                entry("foreign"),
                stale("b"),
                stale("c"),
            ])],
            vec![
                group(&[ok.clone(), stale("a")]),
                group(&[stale("b"), stale("c")]),
            ],
        ];
        for groups in cases {
            let text = format!(
                r#"{{"hooks":{{"SessionStart":[{}]}}}}"#,
                groups.join(",\n  ")
            );
            let m = run(&text);
            let v: Value = serde_json::from_str(&m.text).unwrap();
            assert_eq!(
                m.text.matches("hook session").count(),
                1,
                "{text}\n{}",
                m.text
            );
            assert_eq!(m.text.contains("foreign"), text.contains("foreign"));
            assert!(v["hooks"]["SessionStart"].is_array());
            assert!(!run(&m.text).changed(), "{}", m.text);
        }
    }
}
