//! `tessar purge`: erasing redacted content from the files on disk.
//!
//! A redaction removes content from memory, but the journal is append-only
//! and still holds the original text. Purging rewrites the journal without
//! it, drops the vector cache, which may hold vectors of redacted texts, and
//! removes the decision log, which keeps the text of the user's messages.

use std::fs;
use std::path::{Path, PathBuf};

use tessar_core::store::{Op, purge_with_report};
use tessar_journal::FileJournal;

/// What a purge did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Purged {
    /// Facts whose content was removed from the journal.
    pub facts: usize,
    /// Episodes whose text was removed from the journal.
    pub episodes: usize,
    /// Journal size before, in bytes.
    pub bytes_before: u64,
    /// Journal size after, in bytes.
    pub bytes_after: u64,
    /// Vector cache files deleted.
    pub caches: usize,
    /// Whether a decision log (`decisions.jsonl`) was removed.
    pub decision_log: bool,
}

/// Where the journal lives under `home`.
pub fn journal_path(home: &Path) -> PathBuf {
    home.join("journal.jsonl")
}

/// Rewrites the journal under `home` without redacted content and deletes
/// every vector cache (`index/<fingerprint>/vectors.bin`). Nothing may hold the
/// journal open: stop the daemon first. Without a journal there is nothing
/// to do.
pub fn purge_home(home: &Path) -> Result<Purged, String> {
    let path = journal_path(home);
    if !path.exists() {
        return Err(format!("no journal at {}", path.display()));
    }
    let decision_log = remove_decision_log(home)?;
    let mut counts = None;
    let report = FileJournal::<Op>::rewrite(&path, |commits| {
        let (commits, report) = purge_with_report(commits);
        counts = Some(report);
        commits
    })
    .map_err(|e| format!("cannot rewrite {}: {e}", path.display()))?;
    let counts = counts.unwrap_or_default();
    Ok(Purged {
        facts: counts.facts,
        episodes: counts.episodes,
        bytes_before: report.bytes_before,
        bytes_after: report.bytes_after,
        caches: drop_vector_caches(home)?,
        decision_log,
    })
}

/// Deletes the decision log: it holds message text, so it may hold text
/// that was redacted since.
fn remove_decision_log(home: &Path) -> Result<bool, String> {
    let path = crate::decisions::log_path(home);
    match fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(format!("cannot delete {}: {e}", path.display())),
    }
}

/// Deletes the vector caches under `home/index`; they are rebuilt on the
/// next start.
fn drop_vector_caches(home: &Path) -> Result<usize, String> {
    let index = home.join("index");
    let Ok(models) = fs::read_dir(&index) else {
        return Ok(0);
    };
    let mut dropped = 0;
    for model in models.flatten() {
        let cache = model.path().join("vectors.bin");
        match fs::remove_file(&cache) {
            Ok(()) => dropped += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("cannot delete {}: {e}", cache.display())),
        }
    }
    Ok(dropped)
}

/// The line `tessar purge` prints. `stopped` says a daemon was stopped first.
pub fn summary(p: &Purged, stopped: bool) -> String {
    let daemon = if stopped {
        "stopped the daemon (it restarts with the next session); "
    } else {
        ""
    };
    let log = if p.decision_log {
        "; decision log removed"
    } else {
        ""
    };
    if p.facts == 0 && p.episodes == 0 {
        return format!(
            "{daemon}nothing to purge: the journal holds no redacted content ({} bytes); vector cache files deleted: {}{log}",
            p.bytes_after, p.caches
        );
    }
    format!(
        "{daemon}purged {} fact(s) and {} episode(s); journal {} bytes before, {} after; vector cache files deleted: {}{log}",
        p.facts, p.episodes, p.bytes_before, p.bytes_after, p.caches
    )
}
