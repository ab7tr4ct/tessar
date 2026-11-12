//! The stash that carries memory from the `agent` hook to the `subagent`
//! hook.
//!
//! `PreToolUse` on `Agent` sees the brief but not the future agent's id;
//! `SubagentStart` can place text in the agent's context but sees no task.
//! The first hook leaves the rendered text here, the second takes it. The
//! two events share the parent session id and a key: the agent's `name`
//! when it has one, else its type.
//!
//! One file per entry in `<home>/hooks/`, named
//! `stash.<session>.<key>.<unix nanos, 20 digits>.<pid>-<n>`; session and key
//! are encoded so that they hold no `.` and cannot leave the directory.
//!
//! Pairing. The `agent` hook leaves exactly one entry for every Agent call
//! it sees, empty (a marker) when nothing was selected, so that every
//! subagent start has an entry of its own to consume. A start takes the
//! newest live entry of its key and an empty one is consumed silently. After
//! an orphan (a call that was denied or interrupted after `PreToolUse`),
//! sequential calls of one key still pair each start with its own call, as
//! the newest entry is always the latest call's; the orphan can only be
//! picked up by a start that has no call of its own, a resumed agent for
//! instance, and only while it lives ([`TTL`]). With parallel calls of one
//! key the pairing is ambiguous in any order: telling them apart needs an id
//! that the two events share, and they have none.
//!
//! Both operations are atomic against other hook processes. An entry is
//! written under a `.tmp.` name and renamed into place, so a reader never
//! sees half of one. It is taken by renaming it to a `.taken.` name first:
//! of several processes that try the same file one rename succeeds, and only
//! that one reads and delivers the text. Entries older than [`TTL`] are
//! ignored and removed whenever a hook scans the directory (the `agent`,
//! `subagent` and `session` hooks); `tessar purge` removes all of them, as
//! they may hold text of facts that were redacted since.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::private;

/// An entry older than this is never delivered.
pub const TTL: Duration = Duration::from_secs(120);

const ENTRY: &str = "stash.";
const TMP: &str = ".tmp.";
const TAKEN: &str = ".taken.";
/// The longest encoded session id or key.
const MAX_PART: usize = 120;

/// What [`take`] hands over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Taken {
    pub text: String,
    /// How long the entry waited.
    pub age: Duration,
}

/// `<home>/hooks`.
fn dir(home: &Path) -> PathBuf {
    home.join("hooks")
}

/// Letters, digits and `-` stay; every other byte becomes `_xx`. Distinct
/// inputs give distinct outputs. `None` for an empty or too long input.
fn encode(part: &str) -> Option<String> {
    let part = part.trim();
    if part.is_empty() {
        return None;
    }
    let mut out = String::new();
    for b in part.bytes() {
        if b.is_ascii_alphanumeric() || b == b'-' {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("_{b:02x}"));
        }
    }
    (out.len() <= MAX_PART).then_some(out)
}

fn nanos(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

fn unique() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// The time a stash file was made, from its name, and its entry parts when
/// it is an entry.
struct Named<'a> {
    nanos: u64,
    entry: Option<(&'a str, &'a str)>,
}

fn parse(name: &str) -> Option<Named<'_>> {
    let parts: Vec<&str> = name.split('.').collect();
    match parts.as_slice() {
        ["stash", session, key, ts, _] => Some(Named {
            nanos: ts.parse().ok()?,
            entry: Some((session, key)),
        }),
        ["", "tmp" | "taken", ts, _] => Some(Named {
            nanos: ts.parse().ok()?,
            entry: None,
        }),
        _ => None,
    }
}

/// Removes what is older than [`TTL`] and lists the live entries as
/// `(nanos, file name, session, key)`, oldest first.
fn scan(dir: &Path, now: SystemTime) -> Vec<(u64, String, String, String)> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let now_ns = nanos(now);
    let ttl_ns = u64::try_from(TTL.as_nanos()).unwrap_or(u64::MAX);
    let mut live = Vec::new();
    for item in read.flatten() {
        let name = item.file_name().to_string_lossy().into_owned();
        let Some(named) = parse(&name) else { continue };
        if now_ns.saturating_sub(named.nanos) > ttl_ns {
            let _ = std::fs::remove_file(item.path());
        } else if let Some((session, key)) = named.entry {
            live.push((
                named.nanos,
                name.clone(),
                session.to_owned(),
                key.to_owned(),
            ));
        }
    }
    live.sort();
    live
}

/// Removes the entries that have expired; for hooks that do not otherwise
/// touch the stash.
pub fn sweep(home: &Path) {
    scan(&dir(home), SystemTime::now());
}

/// Removes every stash file, live or not, and returns how many.
pub fn remove_all(home: &Path) -> usize {
    let Ok(read) = std::fs::read_dir(dir(home)) else {
        return 0;
    };
    read.flatten()
        .filter(|e| parse(&e.file_name().to_string_lossy()).is_some())
        .filter(|e| std::fs::remove_file(e.path()).is_ok())
        .count()
}

/// Leaves `text` for the next agent of `key` in `session`.
pub fn put(home: &Path, session: &str, key: &str, text: &str) -> io::Result<()> {
    put_at(home, session, key, text, SystemTime::now())
}

/// [`put`] at the time `now`.
#[doc(hidden)]
pub fn put_at(
    home: &Path,
    session: &str,
    key: &str,
    text: &str,
    now: SystemTime,
) -> io::Result<()> {
    let (Some(session), Some(key)) = (encode(session), encode(key)) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "empty or too long session id or agent key",
        ));
    };
    let dir = dir(home);
    private::create_own_dir_all(home, &dir)?;
    scan(&dir, now);
    let (ts, unique) = (nanos(now), unique());
    let tmp = dir.join(format!("{TMP}{ts:020}.{unique}"));
    let written = (|| {
        let mut file = private::file_options()
            .create_new(true)
            .write(true)
            .open(&tmp)?;
        private::tighten_file(&file);
        file.write_all(text.as_bytes())
    })();
    match written.and_then(|()| {
        std::fs::rename(
            &tmp,
            dir.join(format!("{ENTRY}{session}.{key}.{ts:020}.{unique}")),
        )
    }) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Takes the newest live entry of `key` in `session`, if there is one. An
/// empty entry (a marker) is taken like any other and has an empty `text`.
pub fn take(home: &Path, session: &str, key: &str) -> Option<Taken> {
    take_at(home, session, key, SystemTime::now())
}

/// [`take`] at the time `now`.
#[doc(hidden)]
pub fn take_at(home: &Path, session: &str, key: &str, now: SystemTime) -> Option<Taken> {
    let (session, key) = (encode(session)?, encode(key)?);
    let dir = dir(home);
    for (ts, name, s, k) in scan(&dir, now).into_iter().rev() {
        if s != session || k != key {
            continue;
        }
        let mine = dir.join(format!("{TAKEN}{:020}.{}", nanos(now), unique()));
        // Of several takers of this file one rename succeeds; the others
        // move on to the next entry.
        if std::fs::rename(dir.join(&name), &mine).is_err() {
            continue;
        }
        let text = std::fs::read_to_string(&mine);
        let _ = std::fs::remove_file(&mine);
        if let Ok(text) = text {
            return Some(Taken {
                text,
                age: Duration::from_nanos(nanos(now).saturating_sub(ts)),
            });
        }
    }
    None
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn at(secs: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(secs)
    }

    fn files(home: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir(home))
            .map(|r| {
                r.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }

    #[test]
    fn an_entry_is_taken_once_and_privately_kept() {
        let home = tempfile::tempdir().unwrap();
        put_at(home.path(), "s1", "worker", "text", at(1000)).unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&dir(home.path())), 0o700);
        let name = &files(home.path())[0];
        assert_eq!(mode(&dir(home.path()).join(name)), 0o600);
        let got = take_at(home.path(), "s1", "worker", at(1030)).unwrap();
        assert_eq!(got.text, "text");
        assert_eq!(got.age, Duration::from_secs(30));
        assert!(take_at(home.path(), "s1", "worker", at(1031)).is_none());
        assert!(files(home.path()).is_empty());
    }

    #[test]
    fn the_newest_entry_wins_and_an_empty_one_is_consumed() {
        let home = tempfile::tempdir().unwrap();
        put_at(home.path(), "s", "k", "orphan", at(1000)).unwrap();
        put_at(home.path(), "s", "k", "", at(1010)).unwrap();
        put_at(home.path(), "s", "k", "latest", at(1020)).unwrap();
        let t = |n| take_at(home.path(), "s", "k", at(n));
        assert_eq!(t(1030).unwrap().text, "latest");
        assert_eq!(t(1031).unwrap().text, "");
        assert_eq!(t(1032).unwrap().text, "orphan");
        assert!(t(1033).is_none());
    }

    #[test]
    fn remove_all_takes_every_stash_file_and_nothing_else() {
        let home = tempfile::tempdir().unwrap();
        put_at(home.path(), "s", "k", "x", at(1000)).unwrap();
        put_at(home.path(), "t", "k", "y", at(1000)).unwrap();
        let hooks = dir(home.path());
        std::fs::write(hooks.join(".tmp.00000000000000001000.1-1"), "z").unwrap();
        std::fs::write(hooks.join(".taken.00000000000000001000.1-2"), "z").unwrap();
        std::fs::write(hooks.join("s.shown"), "f1\n").unwrap();
        assert_eq!(remove_all(home.path()), 4);
        assert_eq!(files(home.path()), ["s.shown"]);
    }

    #[test]
    fn other_sessions_and_keys_are_not_taken() {
        let home = tempfile::tempdir().unwrap();
        put_at(home.path(), "s", "k", "mine", at(1000)).unwrap();
        assert!(take_at(home.path(), "other", "k", at(1001)).is_none());
        assert!(take_at(home.path(), "s", "k2", at(1001)).is_none());
        // Names that would collide under a lossy encoding stay apart.
        put_at(home.path(), "s", "a b", "x", at(1000)).unwrap();
        assert!(take_at(home.path(), "s", "a_b", at(1001)).is_none());
        assert_eq!(
            take_at(home.path(), "s", "a b", at(1001)).unwrap().text,
            "x"
        );
        assert_eq!(
            take_at(home.path(), "s", "k", at(1001)).unwrap().text,
            "mine"
        );
    }

    #[test]
    fn an_old_entry_is_not_delivered_and_is_removed() {
        let home = tempfile::tempdir().unwrap();
        put_at(home.path(), "s", "k", "old", at(1000)).unwrap();
        put_at(home.path(), "s", "other", "old too", at(1000)).unwrap();
        let late = at(1000 + TTL.as_secs() + 1);
        assert!(take_at(home.path(), "s", "k", late).is_none());
        // Expired entries of every key go in the same pass.
        assert!(files(home.path()).is_empty());
        put_at(home.path(), "s", "k", "edge", at(1000)).unwrap();
        let edge = at(1000 + TTL.as_secs());
        assert_eq!(take_at(home.path(), "s", "k", edge).unwrap().text, "edge");
    }

    #[test]
    fn text_survives_as_it_is_and_bad_ids_are_refused() {
        let home = tempfile::tempdir().unwrap();
        let text = "line\n\"quoted\" back\\slash ж\n";
        put(home.path(), "../s", "../../k", text).unwrap();
        assert!(files(home.path()).iter().all(|f| !f.contains('/')));
        assert_eq!(take(home.path(), "../s", "../../k").unwrap().text, text);
        assert!(put(home.path(), "", "k", "x").is_err());
        assert!(put(home.path(), "s", " ", "x").is_err());
        assert!(put(home.path(), "s", &"k".repeat(500), "x").is_err());
        assert!(take(home.path(), "", "k").is_none());
    }

    #[test]
    fn parallel_takers_deliver_an_entry_once() {
        let home = tempfile::tempdir().unwrap();
        for i in 0..20 {
            put_at(home.path(), "s", "k", &format!("e{i}"), at(1000 + i)).unwrap();
        }
        let got: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    s.spawn(|| {
                        let mut mine = Vec::new();
                        while let Some(t) = take_at(home.path(), "s", "k", at(1100)) {
                            mine.push(t.text);
                        }
                        mine
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        let mut got = got;
        got.sort();
        let mut want: Vec<String> = (0..20).map(|i| format!("e{i}")).collect();
        want.sort();
        assert_eq!(got, want);
    }
}
