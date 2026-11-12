//! `tessar purge` over a real journal file in a temp directory: no model.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde_json::json;
use tessar_core::clock::ManualClock;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::model::Timestamp;
use tessar_core::store::{Memory, Op, Semantic};
use tessar_journal::FileJournal;
use tessar_mcp::api::Tessar;
use tessar_mcp::args::{AboutArgs, RecallArgs, RememberArgs, UpdateArgs};
use tessar_mcp::purge::{journal_path, purge_home, summary};
use tessar_mcp::when::{Calendar, DAY_MS, days_from_civil};

const DIMS: usize = 64;

struct Bag;

impl Bag {
    fn vector(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; DIMS];
        v[DIMS - 1] = 0.05;
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .map(str::to_lowercase)
            .filter(|w| !w.is_empty())
        {
            let mut h = DefaultHasher::new();
            word.hash(&mut h);
            v[(h.finish() % (DIMS as u64 - 1)) as usize] += 1.0;
        }
        v
    }
}

impl Embedder for Bag {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| Self::vector(t)).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(Self::vector(text))
    }
}

/// Opens the journal under `home` as an api. Everything is dropped with it.
fn open(home: &std::path::Path) -> Tessar {
    let clock = Arc::new(ManualClock::new(Timestamp::from_millis(
        days_from_civil(2026, 11, 8) * DAY_MS + 12 * 3_600_000,
    )));
    let log = FileJournal::<Op>::open(journal_path(home)).unwrap();
    let mem = Memory::open_with(
        log,
        clock.clone(),
        Semantic {
            embedder: Arc::new(Bag),
            dims: DIMS,
            near_bonus: 0.05,
            predicate_threshold: 0.70,
        },
    )
    .unwrap();
    Tessar::bootstrap(mem, clock, "tessar", &|w| panic!("{w}"))
        .unwrap()
        .with_calendar(Calendar::utc())
}

fn handle(out: &str) -> String {
    out.split(|c: char| !c.is_alphanumeric())
        .find(|w| w.len() > 1 && w.starts_with('f') && w[1..].bytes().all(|b| b.is_ascii_digit()))
        .unwrap_or_else(|| panic!("no handle in {out}"))
        .to_owned()
}

fn remember(api: &Tessar, text: &str) -> String {
    let args: RememberArgs =
        serde_json::from_value(json!({"facts": [{"kind": "statement", "text": text}]})).unwrap();
    handle(&api.remember(args).unwrap())
}

#[test]
fn purge_erases_redacted_text_from_the_file_and_keeps_the_rest() {
    let home = tempfile::tempdir().unwrap();
    let keep = "The journal uses JSON lines with a checksum";
    let secret = "The deploy token is zebracrossing99";

    let secret_id = {
        let api = open(home.path());
        remember(&api, keep);
        let id = remember(&api, secret);
        let args: UpdateArgs =
            serde_json::from_value(json!({"fact": id, "reason": "redact"})).unwrap();
        api.update(args).unwrap();
        id
    };

    // Redacted in memory, still on disk.
    let before = std::fs::read_to_string(journal_path(home.path())).unwrap();
    assert!(before.contains("zebracrossing99"));

    // A stale vector cache from an earlier run.
    let cache = home.path().join("index").join("voyage-4-nano");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(cache.join("vectors.bin"), b"vectors").unwrap();

    let done = purge_home(home.path()).unwrap();
    assert_eq!((done.facts, done.episodes, done.caches), (1, 0, 1));
    assert!(done.bytes_after < done.bytes_before);
    assert!(!cache.join("vectors.bin").exists());
    let line = summary(&done, true);
    assert!(line.starts_with("stopped the daemon"), "{line}");
    assert!(line.contains("purged 1 fact(s) and 0 episode(s)"), "{line}");

    let after = std::fs::read_to_string(journal_path(home.path())).unwrap();
    assert!(!after.contains("zebracrossing99"), "secret still on disk");
    assert!(after.contains("JSON lines with a checksum"));

    // Reopen: the kept fact is intact, the redacted one stays redacted.
    let api = open(home.path());
    let found = api
        .recall(serde_json::from_value::<RecallArgs>(json!({"query": "checksum journal"})).unwrap())
        .unwrap();
    assert!(found.contains(keep), "{found}");
    let gone = api
        .recall(serde_json::from_value::<RecallArgs>(json!({"query": "zebracrossing99"})).unwrap())
        .unwrap();
    assert!(!gone.contains("zebracrossing99"), "{gone}");
    let again = api
        .update(
            serde_json::from_value::<UpdateArgs>(json!({"fact": secret_id, "reason": "redact"}))
                .unwrap(),
        )
        .unwrap_err();
    assert!(again.contains("already redacted"), "{again}");
    let project = api
        .about(
            serde_json::from_value::<AboutArgs>(
                json!({"entity": {"key": "tessar", "kind": "project"}}),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(
        project.contains(keep) && !project.contains("zebracrossing99"),
        "{project}"
    );
}

#[test]
fn purging_a_clean_journal_succeeds_and_says_so() {
    let home = tempfile::tempdir().unwrap();
    {
        let api = open(home.path());
        remember(&api, "Nothing secret here");
    }
    let done = purge_home(home.path()).unwrap();
    assert_eq!((done.facts, done.episodes, done.caches), (0, 0, 0));
    let line = summary(&done, false);
    assert!(line.starts_with("nothing to purge"), "{line}");

    let missing = tempfile::tempdir().unwrap();
    assert!(
        purge_home(missing.path())
            .unwrap_err()
            .contains("no journal")
    );
}

#[test]
fn purge_removes_the_decision_log() {
    let home = tempfile::tempdir().unwrap();
    std::fs::write(journal_path(home.path()), "").unwrap();
    let log = home.path().join("decisions.jsonl");
    std::fs::write(&log, "{\"message\":\"secret\"}\n").unwrap();
    let done = purge_home(home.path()).unwrap();
    assert!(done.decision_log);
    assert!(!log.exists());
    assert!(summary(&done, false).contains("decision log removed"));
    let again = purge_home(home.path()).unwrap();
    assert!(!again.decision_log);
    assert!(!summary(&again, false).contains("decision log"));
}

#[test]
fn purge_removes_the_hooks_stash() {
    let home = tempfile::tempdir().unwrap();
    std::fs::write(journal_path(home.path()), "").unwrap();
    tessar_mcp::stash::put(home.path(), "s1", "scout", "a redacted fact").unwrap();
    tessar_mcp::stash::put(home.path(), "s1", "scout", "").unwrap();
    let hooks = home.path().join("hooks");
    std::fs::write(hooks.join(".tmp.00000000000000000001.1-1"), "x").unwrap();
    std::fs::write(hooks.join("s1.shown"), "f1\n").unwrap();
    let done = purge_home(home.path()).unwrap();
    assert_eq!(done.stash_files, 3);
    assert!(summary(&done, false).contains("stash files removed: 3"));
    let left: Vec<String> = std::fs::read_dir(&hooks)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(left, ["s1.shown"]);
}
