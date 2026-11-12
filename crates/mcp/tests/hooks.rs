//! The Claude Code hooks over the test daemon: a Unix socket in a temp
//! directory, an in-memory log and a fake embedder. No model, no GPU.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rmcp::model::CallToolRequestParams;
use rmcp::{Peer, RoleClient};
use serde_json::{Value, json};
use tessar_core::clock::SystemClock;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::log::MemLog;
use tessar_core::store::{Memory, Semantic};
use tessar_mcp::daemon::{self, Projects, socket_path};
use tessar_mcp::hook::{self, HookInput, Settings};
use tessar_mcp::proxy;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

const DIMS: usize = 64;

/// Bag of words hashed into 64 buckets: the same words, the same vector.
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

fn projects() -> Arc<Projects> {
    let clock = Arc::new(SystemClock);
    let mem = Memory::open_with(
        MemLog::new(),
        clock.clone(),
        Semantic {
            embedder: Arc::new(Bag),
            dims: DIMS,
            near_bonus: 0.05,
            predicate_threshold: 0.70,
        },
    )
    .unwrap();
    Arc::new(Projects::new(mem, clock))
}

struct Daemon {
    stop: oneshot::Sender<()>,
    task: JoinHandle<()>,
}

impl Daemon {
    fn start(home: &Path, idle_exit: Option<Duration>) -> Self {
        let listener = UnixListener::bind(socket_path(home)).unwrap();
        let (stop, rx) = oneshot::channel::<()>();
        let task = tokio::spawn(daemon::run(listener, projects(), idle_exit, async {
            let _ = rx.await;
        }));
        Self { stop, task }
    }

    async fn shutdown(self) {
        let _ = self.stop.send(());
        tokio::time::timeout(Duration::from_secs(15), self.task)
            .await
            .expect("the daemon did not stop")
            .unwrap();
    }
}

/// A hook's connection: the daemon's peer, and the link keeping it open.
struct Hook {
    peer: Peer<RoleClient>,
    _link: proxy::DaemonLink,
}

async fn connect(home: &Path, project: &str) -> Hook {
    let stream = UnixStream::connect(socket_path(home)).await.unwrap();
    let (server, link) = proxy::connect(stream, project, Path::new("/work/x"))
        .await
        .unwrap();
    Hook {
        peer: server.daemon().clone(),
        _link: link,
    }
}

async fn call(hook: &Hook, tool: &'static str, args: Value) -> rmcp::model::CallToolResult {
    let mut params = CallToolRequestParams::new(tool);
    params.arguments = args.as_object().cloned();
    hook.peer.call_tool(params).await.unwrap()
}

async fn remember(hook: &Hook, space: &str, text: &str) {
    let r = call(
        hook,
        "remember",
        json!({"space": space, "facts": [{"kind": "statement", "text": text}]}),
    )
    .await;
    assert_ne!(r.is_error, Some(true), "{r:?}");
}

fn input(session: &str, prompt: &str) -> HookInput {
    HookInput::parse(
        &json!({"session_id": session, "cwd": "/work/x", "hook_event_name": "UserPromptSubmit",
            "transcript_path": "/t", "prompt": prompt})
        .to_string(),
    )
    .unwrap()
}

async fn with_memory() -> (tempfile::TempDir, Daemon, Hook) {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let hook = connect(home.path(), "alpha").await;
    remember(&hook, "user", "User prefers to be talked to in Russian").await;
    remember(
        &hook,
        "user",
        "User lives in Berlin and works from a small flat",
    )
    .await;
    remember(
        &hook,
        "project",
        "The journal is a JSON Lines file with a checksum per line",
    )
    .await;
    (home, daemon, hook)
}

#[tokio::test]
async fn recall_and_about_carry_structured_hits_next_to_the_text() {
    let (_home, daemon, hook) = with_memory().await;

    let recalled = call(&hook, "recall", json!({"query": "journal checksum"})).await;
    let hits = recalled.structured_content.as_ref().unwrap()["hits"]
        .as_array()
        .unwrap()
        .clone();
    let hit = hits
        .iter()
        .find(|h| h["text"].as_str().unwrap().contains("JSON Lines"))
        .expect("the journal fact is a hit");
    assert!(hit["id"].as_str().unwrap().starts_with('f'));
    assert_eq!(hit["space"], "alpha");
    assert_eq!(hit["kind"], "statement");
    assert_eq!(hit["current"], true);
    assert_eq!(hit["recorded"].as_str().unwrap().len(), 10);
    assert!(hit["similarity"].as_f64().unwrap() > 0.3, "{hit}");
    let channels: Vec<&str> = hit["channels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap())
        .collect();
    assert!(
        channels.contains(&"meaning") && channels.contains(&"words"),
        "{channels:?}"
    );
    // The text is the line without its handle, date and space.
    assert_eq!(
        hit["text"],
        "The journal is a JSON Lines file with a checksum per line"
    );
    // The text content is the same as ever: handle, date, space, why.
    let text = serde_json::to_value(&recalled).unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        text.contains(&format!("{} [", hit["id"].as_str().unwrap())),
        "{text}"
    );
    assert!(text.contains("found by:"), "{text}");

    let about = call(
        &hook,
        "about",
        json!({"entity": {"key": "user", "kind": "person"}}),
    )
    .await;
    let hits = about.structured_content.as_ref().unwrap()["hits"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0]["space"], "user");
    assert!(hits[0]["similarity"].is_null());
    assert!(hits[0]["text"].as_str().unwrap().contains("Russian"));

    let none = call(
        &hook,
        "about",
        json!({"entity": {"key": "nobody", "kind": "person"}}),
    )
    .await;
    assert_eq!(none.structured_content.unwrap()["hits"], json!([]));

    daemon.shutdown().await;
}

#[tokio::test]
async fn prompt_hook_prints_relevant_facts_once_per_session() {
    let (home, daemon, hook) = with_memory().await;
    let settings = Settings::default();
    let hooks_home = home.path();

    let ask = input("sess-1", "in which language should the user be talked to");
    let out = hook::prompt(&hook.peer, &ask, &settings, hooks_home)
        .await
        .unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(
        lines[0],
        "Memory (tessar) relevant to this message; call `recall` for more:"
    );
    assert!(
        lines.iter().any(|l| l.starts_with("- [")
            && l.contains("] [user] User prefers to be talked to in Russian")),
        "{out}"
    );
    assert!(
        !out.contains("JSON Lines"),
        "unrelated facts stay out: {out}"
    );
    assert!(out.chars().count() <= 1500);
    let shown = std::fs::read_to_string(hooks_home.join("hooks/sess-1.shown")).unwrap();
    assert!(!shown.trim().is_empty());

    // The same question again in the same session: everything was shown.
    let again = hook::prompt(&hook.peer, &ask, &settings, hooks_home)
        .await
        .unwrap();
    assert_eq!(again, "");
    // Another session still sees it.
    let other = input("sess-2", "in which language should the user be talked to");
    let out = hook::prompt(&hook.peer, &other, &settings, hooks_home)
        .await
        .unwrap();
    assert!(out.contains("Russian"), "{out}");

    daemon.shutdown().await;
}

#[tokio::test]
async fn prompt_hook_skips_short_prompts_and_slash_commands() {
    let (home, daemon, hook) = with_memory().await;
    let settings = Settings::default();
    for prompt in ["Russian language", "/compact the user language Russian", ""] {
        let out = hook::prompt(&hook.peer, &input("s", prompt), &settings, home.path())
            .await
            .unwrap();
        assert_eq!(out, "", "{prompt:?}");
    }
    assert!(!home.path().join("hooks").exists(), "nothing was shown");
    daemon.shutdown().await;
}

#[tokio::test]
async fn an_entity_named_in_the_prompt_passes_above_the_entity_floor() {
    let (home, daemon, hook) = with_memory().await;
    // A threshold nothing reaches: only the entity rule can pass.
    let settings = Settings {
        min_similarity: 2.0,
        entity_min_similarity: 0.0,
        ..Settings::default()
    };
    let out = hook::prompt(
        &hook.peer,
        &input("s", "please tell me everything you know about alpha"),
        &settings,
        home.path(),
    )
    .await
    .unwrap();
    assert!(out.contains("JSON Lines"), "{out}");

    let out = hook::prompt(
        &hook.peer,
        &input("s2", "please tell me anything about quantum gardening"),
        &settings,
        home.path(),
    )
    .await
    .unwrap();
    assert_eq!(out, "");
    daemon.shutdown().await;
}

#[tokio::test]
async fn prompt_hook_respects_the_budget() {
    let (home, daemon, hook) = with_memory().await;
    let settings = Settings {
        budget: 120,
        ..Settings::default()
    };
    let out = hook::prompt(
        &hook.peer,
        &input("s", "what do you know about the user and the journal"),
        &settings,
        home.path(),
    )
    .await
    .unwrap();
    assert!(out.chars().count() <= 120, "{out}");
    daemon.shutdown().await;
}

#[tokio::test]
async fn session_briefing_has_both_sections_and_resets_the_shown_file() {
    let (home, daemon, hook) = with_memory().await;
    let settings = Settings::default();
    hook::append_shown(home.path(), "sess-1", &["f1".to_owned()]).unwrap();
    hook::append_shown(home.path(), "sess-2", &["f1".to_owned()]).unwrap();

    let start = HookInput::parse(
        &json!({"session_id": "sess-1", "cwd": "/work/x", "source": "compact"}).to_string(),
    )
    .unwrap();
    let out = hook::session(&hook.peer, &start, "alpha", &settings, home.path())
        .await
        .unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(
        lines[0],
        "tessar memory: 2 facts about you, 1 about project `alpha`; tools: remember, recall, \
         about, update"
    );
    assert_eq!(lines[1], "About you:");
    // Newest first: Berlin was recorded after Russian.
    assert!(lines[2].contains("Berlin"), "{out}");
    assert!(lines[3].contains("Russian"), "{out}");
    assert_eq!(lines[4], "About this project:");
    assert!(lines[5].contains("JSON Lines"), "{out}");
    assert!(out.chars().count() <= 3000);

    assert!(!home.path().join("hooks/sess-1.shown").exists(), "reset");
    assert!(
        home.path().join("hooks/sess-2.shown").exists(),
        "other sessions stay"
    );

    // After the reset the prompt hook shows facts again.
    let ask = input("sess-1", "in which language should the user be talked to");
    assert!(
        hook::prompt(&hook.peer, &ask, &settings, home.path())
            .await
            .unwrap()
            .contains("Russian")
    );

    daemon.shutdown().await;
}

#[tokio::test]
async fn session_briefing_of_an_empty_memory() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let hook = connect(home.path(), "alpha").await;
    let start =
        HookInput::parse(&json!({"session_id": "s", "source": "startup"}).to_string()).unwrap();
    let out = hook::session(
        &hook.peer,
        &start,
        "alpha",
        &Settings::default(),
        home.path(),
    )
    .await
    .unwrap();
    // The memory is empty, and so is the lessons block but for its hint.
    assert_eq!(out.lines().count(), 2, "{out}");
    assert!(out.contains("empty") && out.contains("`remember`"), "{out}");
    assert!(
        out.lines().nth(1).unwrap().starts_with("Record a lesson"),
        "{out}"
    );
    daemon.shutdown().await;
}

#[tokio::test]
async fn run_prompt_uses_the_running_daemon_and_the_cwd_project() {
    let (home, daemon, _hook) = with_memory().await;
    let cwd = tempfile::tempdir().unwrap();
    let stdin = json!({"session_id": "s", "cwd": cwd.path(),
        "prompt": "in which language should the user be talked to"})
    .to_string();
    // A generous limit: this test is about the answer, not the time.
    let out = hook::run_prompt_with(
        home.path(),
        &stdin,
        &Settings::default(),
        Duration::from_secs(20),
        &no_spawn(),
    )
    .await
    .unwrap();
    assert!(out.contains("Russian"), "{out}");
    // Garbage on stdin is an error for the caller to log, not a panic.
    assert!(
        hook::run_prompt(home.path(), "not json", &Settings::default())
            .await
            .error
            .is_some()
    );
    daemon.shutdown().await;
}

#[tokio::test]
async fn the_entity_floor_keeps_out_loosely_related_facts() {
    let (home, daemon, hook) = with_memory().await;
    let settings = Settings {
        min_similarity: 2.0,
        ..Settings::default()
    };
    let out = hook::prompt(
        &hook.peer,
        &input("s", "please tell me everything you know about alpha"),
        &settings,
        home.path(),
    )
    .await
    .unwrap();
    assert_eq!(out, "", "the cosine is below the default floor: {out}");
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_lesson_goes_through_the_daemon() {
    let (_home, daemon, hook) = with_memory().await;
    let line = proxy::call_tool_text(
        &hook.peer,
        "lesson",
        json!({"text": "Before blaming the cache, read its key.", "space": "user"}),
    )
    .await
    .unwrap();
    assert!(
        line.contains("space `user`") && line.contains("created"),
        "{line}"
    );
    let err = proxy::call_tool_text(&hook.peer, "lesson", json!({"text": " "}))
        .await
        .unwrap_err();
    assert!(err.contains("call `lesson` again"), "{err}");
    daemon.shutdown().await;
}

#[tokio::test]
async fn redact_fact_goes_through_the_daemon() {
    let (home, daemon, hook) = with_memory().await;
    remember(&hook, "user", "The vault code is quokka7788").await;
    let found = call(&hook, "recall", json!({"query": "vault code quokka7788"})).await;
    let id = found.structured_content.as_ref().unwrap()["hits"][0]["id"]
        .as_str()
        .unwrap()
        .to_owned();

    let line = proxy::redact_fact(&hook.peer, &id).await.unwrap();
    assert!(line.starts_with(&format!("redacted: {id}")), "{line}");
    let again = proxy::redact_fact(&hook.peer, &id).await.unwrap_err();
    assert!(
        again.starts_with(&format!("{id}: ")) && again.contains("already redacted"),
        "{again}"
    );
    assert!(proxy::redact_fact(&hook.peer, "nonsense").await.is_err());

    let after = call(&hook, "recall", json!({"query": "vault code quokka7788"})).await;
    assert!(
        !after
            .structured_content
            .unwrap()
            .to_string()
            .contains("quokka7788")
    );
    drop(hook);
    daemon.shutdown().await;
    drop(home);
}

/// A spawner that must not be called.
fn no_spawn() -> proxy::Spawn {
    Arc::new(|_, _| Err("the hook started a daemon".to_owned()))
}

/// A spawner that counts its calls and starts a process that just exits.
fn counting_spawn() -> (proxy::Spawn, Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&calls);
    let spawn: proxy::Spawn = Arc::new(move |_, start| {
        assert_eq!(start, proxy::Start::Fresh);
        count.fetch_add(1, Ordering::SeqCst);
        std::process::Command::new("true")
            .spawn()
            .map_err(|e| e.to_string())
    });
    (spawn, calls)
}

fn prompt_stdin(cwd: &Path) -> String {
    json!({"session_id": "s", "cwd": cwd,
        "prompt": "in which language should the user be talked to"})
    .to_string()
}

/// The prompt must not wait for a daemon that is down: the hook starts one
/// for the next prompt and returns, long before its time limit.
#[tokio::test]
async fn prompt_hook_without_a_daemon_starts_one_and_returns_at_once() {
    use std::sync::atomic::Ordering;
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let (spawn, calls) = counting_spawn();
    let limit = Duration::from_secs(30);
    let started = std::time::Instant::now();
    let out = hook::run_prompt_with(
        home.path(),
        &prompt_stdin(cwd.path()),
        &Settings::default(),
        limit,
        &spawn,
    )
    .await;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "waited {:?} for a daemon",
        started.elapsed()
    );
    let err = out.expect_err("nothing to print without a daemon");
    assert!(err.contains("started one for the next prompt"), "{err}");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // A daemon that holds the journal but does not listen yet (it loads
    // the model) is not started a second time.
    let journal = tessar_journal::FileJournal::<tessar_core::store::Op>::open(
        home.path().join("journal.jsonl"),
    )
    .unwrap();
    let err = hook::run_prompt_with(
        home.path(),
        &prompt_stdin(cwd.path()),
        &Settings::default(),
        limit,
        &spawn,
    )
    .await
    .expect_err("nothing to print without a daemon");
    assert!(err.contains("it is starting"), "{err}");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(journal);

    // A prompt that is skipped anyway starts nothing.
    let short = json!({"session_id": "s", "cwd": cwd.path(), "prompt": "ok"}).to_string();
    let out = hook::run_prompt_with(home.path(), &short, &Settings::default(), limit, &spawn)
        .await
        .unwrap();
    assert_eq!(out, "");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// A daemon that accepts the connection and then says nothing (one too old
/// for the handshake, or one that hangs) costs the time limit and no more.
#[tokio::test]
async fn prompt_hook_gives_up_on_a_silent_daemon_at_its_limit() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let listener = UnixListener::bind(socket_path(home.path())).unwrap();
    let silent = tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((stream, _)) = listener.accept().await {
            held.push(stream);
        }
    });
    let limit = Duration::from_millis(300);
    let started = std::time::Instant::now();
    let err = hook::run_prompt_with(
        home.path(),
        &prompt_stdin(cwd.path()),
        &Settings::default(),
        limit,
        &no_spawn(),
    )
    .await
    .expect_err("a silent daemon gives nothing to print");
    assert!(err.contains("no answer from the daemon"), "{err}");
    assert!(started.elapsed() >= limit);
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "waited {:?}, far past the limit",
        started.elapsed()
    );
    silent.abort();
}

#[test]
fn the_prompt_hook_is_capped_at_about_a_second() {
    assert!(hook::PROMPT_WAIT <= Duration::from_millis(1500));
    assert!(hook::PROMPT_WAIT < hook::SESSION_WAIT);
}

/// The real binary: `tessar hook prompt` with no daemon prints nothing,
/// exits with status 0 without waiting, and leaves a daemon starting. The
/// daemon it starts finds no model (the environment points it at a missing
/// file, so nothing outside the temp home is touched) and says so in the
/// log, which is how the test knows it was started.
#[test]
fn hook_prompt_binary_does_not_wait_for_a_daemon() {
    use std::io::Write;
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let started = std::time::Instant::now();
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_tessar"))
        .args(["hook", "prompt", "--home"])
        .arg(home.path())
        .env("HOME", home.path())
        .env("TESSAR_MODEL", home.path().join("no-such-model.gguf"))
        .env_remove("TESSAR_HOME")
        .current_dir(cwd.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(prompt_stdin(cwd.path()).as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let took = started.elapsed();
    assert!(out.status.success(), "{:?}", out.status);
    assert!(
        out.stdout.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(took < Duration::from_secs(5), "the hook took {took:?}");
    let log = home.path().join(daemon::LOG_FILE);
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        if text.contains("embedding model not found") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no daemon was started; log: {text:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

// ---- the decision log and lessons ---------------------------------------

fn records(home: &Path) -> Vec<Value> {
    std::fs::read_to_string(home.join("decisions.jsonl"))
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

async fn remember_lesson(hook: &Hook, space: &str, text: &str) {
    let r = call(
        hook,
        "remember",
        json!({"space": space, "facts": [{"kind": "statement", "text": text,
            "mentions": [{"key": "lessons", "kind": "topic", "space": space}]}]}),
    )
    .await;
    assert_ne!(r.is_error, Some(true), "{r:?}");
}

fn stdin_for(session: &str, prompt: &str) -> String {
    json!({"session_id": session, "cwd": "/work/x", "prompt": prompt}).to_string()
}

/// Runs the prompt hook as the binary does: the output is handed to a
/// closure, then the record is written. Returns what was printed and
/// whether the log already existed when it was printed.
async fn run_logged(home: &Path, stdin: &str, settings: &Settings) -> (String, bool) {
    let run =
        hook::run_prompt_recorded(home, stdin, settings, Duration::from_secs(20), &no_spawn())
            .await;
    let log = home.join("decisions.jsonl");
    let mut seen = None;
    run.finish(home, |text| seen = Some((text.to_owned(), log.exists())));
    seen.expect("the output was printed")
}

/// The session hook the same way.
async fn session_logged(
    hook: &Hook,
    start: &HookInput,
    settings: &Settings,
    home: &Path,
) -> String {
    let run = hook::session_recorded(&hook.peer, start, "alpha", settings, home)
        .await
        .unwrap();
    let mut printed = String::new();
    run.finish(home, |text| printed = text.to_owned());
    printed
}

#[tokio::test]
async fn a_prompt_run_is_logged_after_its_output() {
    use std::os::unix::fs::PermissionsExt;
    let (home, daemon, hook) = with_memory().await;
    remember_lesson(
        &hook,
        "project",
        "Lesson (2026-11-09): when the journal checksum fails, read the journal first",
    )
    .await;
    let message = "in which language should the user be talked to, and the journal checksum";
    let (out, log_existed) = run_logged(
        home.path(),
        &stdin_for("sess-1", message),
        &Settings::default(),
    )
    .await;
    assert!(out.contains("Russian"), "{out}");
    assert!(!log_existed, "the record is written after the output");

    let recs = records(home.path());
    assert_eq!(recs.len(), 1);
    let r = &recs[0];
    assert_eq!(r["kind"], "prompt");
    assert_eq!(r["session"], "sess-1");
    assert_eq!(r["message"], message);
    assert_eq!(r["message_chars"], message.chars().count());
    assert!(r["ts"].as_str().unwrap().ends_with('Z'));
    assert_eq!(r["thresholds"]["lesson_min_similarity"], 0.35);
    assert_eq!(r["thresholds"]["budget"], 1500);
    assert_eq!(r["printed_chars"], out.chars().count());
    assert!(r["elapsed_ms"].is_u64());
    let candidates = r["candidates"].as_array().unwrap();
    assert!(!candidates.is_empty());
    for c in candidates {
        assert!(c["id"].as_str().unwrap().starts_with('f'));
        assert!(c["channels"].is_array() && c["lesson"].is_boolean());
        match c["decision"].as_str().unwrap() {
            "shown" => assert!(c["reason"].is_null()),
            "held" => assert!(
                [
                    "below_threshold",
                    "already_shown",
                    "over_limit",
                    "over_budget"
                ]
                .contains(&c["reason"].as_str().unwrap()),
                "{c}"
            ),
            other => panic!("{other}"),
        }
    }
    assert!(
        candidates.iter().any(|c| c["lesson"] == true),
        "the lesson is a candidate: {candidates:?}"
    );
    let shown = candidates
        .iter()
        .filter(|c| c["decision"] == "shown")
        .count();
    assert_eq!(shown, out.lines().count() - 1, "{out}");

    let mode = std::fs::metadata(home.path().join("decisions.jsonl"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600);

    // The same prompt again: what was shown is held as already shown.
    run_logged(
        home.path(),
        &stdin_for("sess-1", message),
        &Settings::default(),
    )
    .await;
    let recs = records(home.path());
    assert_eq!(recs.len(), 2);
    assert!(
        recs[1]["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["reason"] == "already_shown")
    );
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_lesson_is_printed_with_its_handle_by_the_prompt_hook() {
    let (home, daemon, hook) = with_memory().await;
    remember_lesson(
        &hook,
        "project",
        "Lesson (2026-11-09): when the journal checksum fails, read the journal first",
    )
    .await;
    let out = hook::prompt(
        &hook.peer,
        &input("s", "the journal checksum fails again, what now"),
        &Settings::default(),
        home.path(),
    )
    .await
    .unwrap();
    let line = out
        .lines()
        .find(|l| l.starts_with("- lesson f"))
        .unwrap_or_else(|| panic!("{out}"));
    assert!(
        line.contains("]: when the journal checksum fails"),
        "{line}"
    );
    assert!(!out.contains("Lesson ("), "{out}");
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_skipped_prompt_is_logged_with_its_reason() {
    let home = tempfile::tempdir().unwrap();
    let (out, _) = run_logged(
        home.path(),
        &stdin_for("s", "/compact now please"),
        &Settings::default(),
    )
    .await;
    assert_eq!(out, "");
    run_logged(
        home.path(),
        &stdin_for("s", "ok thanks"),
        &Settings::default(),
    )
    .await;
    let recs = records(home.path());
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0]["kind"], "prompt");
    assert_eq!(recs[0]["skipped"], "slash_command");
    assert_eq!(recs[1]["skipped"], "short_message");
    assert_eq!(recs[1]["candidates"], json!([]));
    assert!(recs[1]["project"].is_null());
}

#[tokio::test]
async fn a_failed_log_write_leaves_the_output_alone() {
    let (home, daemon, _hook) = with_memory().await;
    let message = "in which language should the user be talked to";
    let (with_log, _) =
        run_logged(home.path(), &stdin_for("a", message), &Settings::default()).await;
    // A directory where the log should be: opening it for writing fails.
    std::fs::remove_file(home.path().join("decisions.jsonl")).unwrap();
    std::fs::create_dir(home.path().join("decisions.jsonl")).unwrap();
    let (without, _) =
        run_logged(home.path(), &stdin_for("b", message), &Settings::default()).await;
    assert_eq!(with_log, without);
    daemon.shutdown().await;
}

#[tokio::test]
async fn the_session_hook_adds_the_lessons_block_and_logs_it() {
    let (home, daemon, hook) = with_memory().await;
    remember_lesson(
        &hook,
        "project",
        "Lesson (2026-11-01): when a test flakes, look for a sleep",
    )
    .await;
    remember_lesson(
        &hook,
        "user",
        "Lesson (2026-11-05): when asked for a plan, give one option",
    )
    .await;
    // A fact of the same entity that is not a lesson is not counted.
    remember(&hook, "project", "The lessons topic is where lessons live").await;
    let start = HookInput::parse(
        &json!({"session_id": "s", "cwd": "/work/x", "source": "startup"}).to_string(),
    )
    .unwrap();
    let out = session_logged(&hook, &start, &Settings::default(), home.path()).await;
    assert!(
        out.contains("Lessons: 1 for this project, 1 for you; latest:"),
        "{out}"
    );
    let latest: Vec<&str> = out
        .lines()
        .filter(|l| l.starts_with("- f") && l.contains("[2026-"))
        .collect();
    assert!(out.contains("when asked for a plan"), "{out}");
    assert!(out.contains("when a test flakes"), "{out}");
    assert!(
        !out[out.find("Lessons:").unwrap()..].contains("Lesson ("),
        "{out}"
    );
    assert!(!latest.is_empty());
    assert!(out.trim_end().ends_with("that should trigger it."), "{out}");
    assert!(out.chars().count() <= 3000);

    let recs = records(home.path());
    assert_eq!(recs.len(), 1);
    let r = &recs[0];
    assert_eq!(r["kind"], "session");
    assert_eq!(r["project"], "alpha");
    assert_eq!(r["source"], "startup");
    assert_eq!(r["lessons_project"], 1);
    assert_eq!(r["lessons_user"], 1);
    assert_eq!(r["printed_chars"], out.chars().count());
    let blocks = r["blocks"].as_array().unwrap();
    let size = |n: &str| {
        blocks.iter().find(|b| b["name"] == n).unwrap()["chars"]
            .as_u64()
            .unwrap()
    };
    assert!(size("lessons") > 0 && size("lessons") <= 601);
    assert_eq!(
        (size("briefing") + size("lessons")) as usize,
        out.chars().count()
    );

    // Lessons are not listed twice: not among the facts about you.
    assert_eq!(out.matches("when asked for a plan").count(), 1, "{out}");
    assert!(out.contains("2 facts about you, 2 about project"), "{out}");

    // A budget too small for the block's room (a fifth of it) leaves it out;
    // the briefing stays.
    let tight = Settings {
        session_budget: 400,
        ..Settings::default()
    };
    let out = session_logged(&hook, &start, &tight, home.path()).await;
    assert!(out.chars().count() <= 400, "{}", out.chars().count());
    assert!(out.contains("About you:"), "{out}");
    assert!(
        !out.contains("Lessons:") && !out.contains("Record a lesson"),
        "{out}"
    );
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_full_briefing_still_has_the_lessons_block() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let hook = connect(home.path(), "alpha").await;
    for n in 0..40 {
        remember(
            &hook,
            "user",
            &format!("User fact number {n} about habit {n} and food {n}"),
        )
        .await;
        remember(
            &hook,
            "project",
            &format!("Project fact number {n} on part {n}"),
        )
        .await;
    }
    remember_lesson(
        &hook,
        "project",
        "Lesson (2026-11-01): when a test flakes, look for a sleep",
    )
    .await;
    let start =
        HookInput::parse(&json!({"session_id": "s", "source": "startup"}).to_string()).unwrap();
    let out = session_logged(&hook, &start, &Settings::default(), home.path()).await;
    assert!(out.chars().count() <= 3000, "{}", out.chars().count());
    assert!(
        out.contains("Lessons: 1 for this project, 0 for you"),
        "{out}"
    );
    assert!(
        out.contains("Record a lesson with the `lesson` tool"),
        "{out}"
    );
    assert!(
        out.contains("About you:") && out.contains("About this project:"),
        "{out}"
    );
    daemon.shutdown().await;
}

#[tokio::test]
async fn ordinary_lines_come_from_the_best_eight_hits_only() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let hook = connect(home.path(), "alpha").await;
    for n in 0..12 {
        remember(
            &hook,
            "project",
            &format!("the harbour crane schedule note {n}"),
        )
        .await;
    }
    let (out, _) = run_logged(
        home.path(),
        &stdin_for("s", "the harbour crane schedule note"),
        &Settings::default(),
    )
    .await;
    assert_eq!(out.lines().count() - 1, 8, "{out}");
    let recs = records(home.path());
    let held: Vec<&Value> = recs[0]["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["decision"] == "held")
        .collect();
    assert!(!held.is_empty());
    assert!(held.iter().all(|c| c["reason"] == "over_limit"), "{held:?}");
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_failed_prompt_run_is_logged_with_its_reason() {
    let home = tempfile::tempdir().unwrap();
    // No daemon: the hook starts none (the test spawner refuses) and fails.
    let run = hook::run_prompt_recorded(
        home.path(),
        &stdin_for("s1", "what do I prefer to eat for lunch"),
        &Settings::default(),
        Duration::from_secs(5),
        &no_spawn(),
    )
    .await;
    assert!(run.error.is_some() && run.output.is_empty());
    run.finish(home.path(), |_| panic!("nothing is printed on a failure"));
    let recs = records(home.path());
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0]["kind"], "prompt");
    assert_eq!(recs[0]["session"], "s1");
    assert!(
        recs[0]["error"].as_str().unwrap().contains("daemon"),
        "{}",
        recs[0]
    );
    assert!(recs[0]["elapsed_ms"].is_u64());
    assert_eq!(recs[0]["candidates"], json!([]));
}

#[tokio::test]
async fn a_memory_of_only_lessons_is_not_called_empty() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let hook = connect(home.path(), "alpha").await;
    remember_lesson(
        &hook,
        "project",
        "Lesson (2026-11-01): when a test flakes, look for a sleep",
    )
    .await;
    let start =
        HookInput::parse(&json!({"session_id": "s", "source": "startup"}).to_string()).unwrap();
    let out = session_logged(&hook, &start, &Settings::default(), home.path()).await;
    assert!(!out.contains("empty"), "{out}");
    assert!(
        out.starts_with("tessar memory has no facts about you or project `alpha` yet."),
        "{out}"
    );
    assert!(
        out.contains("`remember`") && out.contains("Lessons: 1 for this project"),
        "{out}"
    );
    daemon.shutdown().await;
}

// ---- memory for subagents: the `agent` and `subagent` hooks ---------------

const LANGUAGE_BRIEF: &str = "in which language should the user be talked to";

fn agent_stdin(session: &str, brief: &str, name: Option<&str>, kind: &str) -> String {
    let mut tool_input = json!({"prompt": brief, "description": "d", "subagent_type": kind});
    if let Some(name) = name {
        tool_input["name"] = json!(name);
    }
    json!({"session_id": session, "cwd": "/work/x", "hook_event_name": "PreToolUse",
        "tool_name": "Agent", "tool_use_id": "t1", "tool_input": tool_input})
    .to_string()
}

fn subagent_stdin(session: &str, agent_type: &str) -> String {
    json!({"session_id": session, "cwd": "/work/x", "hook_event_name": "SubagentStart",
        "agent_id": "a1", "agent_type": agent_type})
    .to_string()
}

/// Runs the `agent` hook as the binary does; returns what it printed.
async fn agent_logged(home: &Path, stdin: &str) -> String {
    let run = hook::run_agent_recorded(
        home,
        stdin,
        &Settings::default(),
        Duration::from_secs(20),
        &no_spawn(),
    )
    .await;
    let mut printed = None;
    run.finish(home, |text| printed = Some(text.to_owned()));
    printed.expect("the output was printed")
}

/// Runs the `subagent` hook the same way.
fn subagent_logged(home: &Path, stdin: &str) -> String {
    let mut printed = None;
    hook::run_subagent(home, stdin).finish(home, |text| printed = Some(text.to_owned()));
    printed.expect("the output was printed")
}

fn stash_files(home: &Path) -> Vec<String> {
    std::fs::read_dir(home.join("hooks"))
        .map(|r| {
            r.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with("stash."))
                .collect()
        })
        .unwrap_or_default()
}

fn delivered(printed: &str) -> String {
    let v: Value = serde_json::from_str(printed).expect("valid JSON");
    assert_eq!(v["hookSpecificOutput"]["hookEventName"], "SubagentStart");
    v["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap()
        .to_owned()
}

#[tokio::test]
async fn the_agent_hook_stashes_what_the_subagent_hook_delivers_once() {
    use std::os::unix::fs::PermissionsExt;
    let (home, daemon, _hook) = with_memory().await;
    let printed = agent_logged(
        home.path(),
        &agent_stdin("s1", LANGUAGE_BRIEF, None, "Explore"),
    )
    .await;
    assert_eq!(printed, "");
    let files = stash_files(home.path());
    assert_eq!(files.len(), 1, "{files:?}");
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&home.path().join("hooks")), 0o700);
    assert_eq!(mode(&home.path().join("hooks").join(&files[0])), 0o600);

    let printed = subagent_logged(home.path(), &subagent_stdin("s1", "Explore"));
    let text = delivered(&printed);
    assert!(
        text.starts_with("Memory (tessar) relevant to this task; call `recall` for more:\n- ["),
        "{text}"
    );
    assert!(
        text.contains("User prefers to be talked to in Russian"),
        "{text}"
    );
    assert!(!text.contains("JSON Lines"), "{text}");
    assert!(text.chars().count() <= 1500);
    assert!(stash_files(home.path()).is_empty(), "the entry is consumed");
    // A resume or a later message gets nothing.
    assert_eq!(
        subagent_logged(home.path(), &subagent_stdin("s1", "Explore")),
        ""
    );
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_brief_that_matches_nothing_leaves_an_empty_entry_that_is_consumed_silently() {
    let (home, daemon, _hook) = with_memory().await;
    let printed = agent_logged(
        home.path(),
        &agent_stdin(
            "s1",
            "zebra quartz xylophone volcano nebula",
            None,
            "Explore",
        ),
    )
    .await;
    assert_eq!(printed, "");
    assert_eq!(stash_files(home.path()).len(), 1, "a marker");
    assert_eq!(
        subagent_logged(home.path(), &subagent_stdin("s1", "Explore")),
        ""
    );
    assert!(
        stash_files(home.path()).is_empty(),
        "the marker is consumed"
    );
    let recs = records(home.path());
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0]["kind"], "agent");
    assert_eq!(
        (recs[0]["stashed"].as_bool(), recs[0]["marker"].as_bool()),
        (Some(false), Some(true))
    );
    assert_eq!(recs[1]["consumed"], true);
    assert_eq!(recs[1]["delivered"], false);
    // A brief too short to search leaves a marker too.
    agent_logged(home.path(), &agent_stdin("s1", "ok", None, "Explore")).await;
    assert_eq!(stash_files(home.path()).len(), 1);
    assert_eq!(
        records(home.path()).pop().unwrap()["skipped"],
        "short_message"
    );
    daemon.shutdown().await;
}

#[tokio::test]
async fn an_orphan_does_not_shift_the_later_deliveries() {
    let (home, daemon, _hook) = with_memory().await;
    let h = home.path();
    // A call whose agent never started leaves memory behind.
    agent_logged(h, &agent_stdin("s1", LANGUAGE_BRIEF, None, "Explore")).await;
    // The next call finds nothing; its start must not get the orphan's text.
    agent_logged(
        h,
        &agent_stdin(
            "s1",
            "zebra quartz xylophone volcano nebula",
            None,
            "Explore",
        ),
    )
    .await;
    assert_eq!(subagent_logged(h, &subagent_stdin("s1", "Explore")), "");
    // And the next pair is its own again.
    agent_logged(
        h,
        &agent_stdin(
            "s1",
            "where does the user live and work from",
            None,
            "Explore",
        ),
    )
    .await;
    let text = delivered(&subagent_logged(h, &subagent_stdin("s1", "Explore")));
    assert!(
        text.lines().nth(1).is_some_and(|l| l.contains("Berlin")),
        "{text}"
    );
    daemon.shutdown().await;
}

#[test]
fn an_entry_lives_two_minutes() {
    let home = tempfile::tempdir().unwrap();
    let ago = |secs| std::time::SystemTime::now() - Duration::from_secs(secs);
    tessar_mcp::stash::put_at(home.path(), "s1", "k", "fresh", ago(110)).unwrap();
    assert_eq!(
        delivered(&subagent_logged(home.path(), &subagent_stdin("s1", "k"))),
        "fresh"
    );
    tessar_mcp::stash::put_at(home.path(), "s1", "k", "stale", ago(130)).unwrap();
    assert_eq!(subagent_logged(home.path(), &subagent_stdin("s1", "k")), "");
    assert!(stash_files(home.path()).is_empty());
}

#[tokio::test]
async fn an_agent_call_without_name_or_type_is_keyed_general_purpose() {
    let (home, daemon, _hook) = with_memory().await;
    let stdin = json!({"session_id": "s1", "cwd": "/work/x", "tool_name": "Agent",
        "tool_input": {"prompt": LANGUAGE_BRIEF}})
    .to_string();
    agent_logged(home.path(), &stdin).await;
    assert_eq!(records(home.path())[0]["key"], "general-purpose");
    let text = delivered(&subagent_logged(
        home.path(),
        &subagent_stdin("s1", "general-purpose"),
    ));
    assert!(text.contains("Russian"), "{text}");
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_failed_agent_run_prints_nothing_leaves_a_marker_and_is_logged() {
    use std::sync::atomic::Ordering;
    let home = tempfile::tempdir().unwrap();
    let (spawn, calls) = counting_spawn();
    let stdin = agent_stdin("s1", LANGUAGE_BRIEF, Some("scout"), "Explore");

    // No daemon.
    let run = hook::run_agent_recorded(
        home.path(),
        &stdin,
        &Settings::default(),
        Duration::from_secs(30),
        &spawn,
    )
    .await;
    let mut printed = None;
    run.finish(home.path(), |t| printed = Some(t.to_owned()));
    assert_eq!(printed, None, "a failed run prints nothing");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(stash_files(home.path()).len(), 1);
    assert_eq!(
        subagent_logged(home.path(), &subagent_stdin("s1", "scout")),
        ""
    );
    assert!(stash_files(home.path()).is_empty());
    let a = &records(home.path())[0];
    assert_eq!(a["kind"], "agent");
    assert!(a["error"].as_str().unwrap().contains("no daemon"), "{a}");
    assert_eq!(
        (a["stashed"].as_bool(), a["marker"].as_bool()),
        (Some(false), Some(true))
    );

    // A daemon that does not answer in time.
    let listener = UnixListener::bind(socket_path(home.path())).unwrap();
    let silent = tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((stream, _)) = listener.accept().await {
            held.push(stream);
        }
    });
    let started = std::time::Instant::now();
    let run = hook::run_agent_recorded(
        home.path(),
        &stdin,
        &Settings::default(),
        Duration::from_millis(300),
        &no_spawn(),
    )
    .await;
    assert!(started.elapsed() < Duration::from_secs(3));
    let mut printed = None;
    run.finish(home.path(), |t| printed = Some(t.to_owned()));
    assert_eq!(printed, None, "a failed run prints nothing");
    assert_eq!(stash_files(home.path()).len(), 1);
    let a = records(home.path()).pop().unwrap();
    assert!(a["error"].as_str().unwrap().contains("no answer"), "{a}");
    assert_eq!(a["marker"], true);
    silent.abort();
}

#[tokio::test]
async fn the_session_hook_removes_expired_stash_entries_after_its_output() {
    let (home, daemon, hook) = with_memory().await;
    let old = std::time::SystemTime::now() - Duration::from_secs(600);
    // The stale one goes in last: a put sweeps the directory itself.
    tessar_mcp::stash::put(home.path(), "other", "j", "fresh").unwrap();
    tessar_mcp::stash::put_at(home.path(), "other", "k", "stale", old).unwrap();
    let start = HookInput::parse(
        &json!({"session_id": "s9", "cwd": "/work/x", "source": "startup"}).to_string(),
    )
    .unwrap();
    let run = hook::session_recorded(
        &hook.peer,
        &start,
        "alpha",
        &Settings::default(),
        home.path(),
    )
    .await
    .unwrap();
    let mut at_output = None;
    run.finish(home.path(), |_| {
        at_output = Some(stash_files(home.path()).len())
    });
    assert_eq!(at_output, Some(2), "nothing is swept before the output");
    assert_eq!(stash_files(home.path()).len(), 1);
    daemon.shutdown().await;
}

#[tokio::test]
async fn the_agent_hook_leaves_the_parents_no_repeat_state_alone() {
    let (home, daemon, _hook) = with_memory().await;
    let shown_file = home.path().join("hooks/s1.shown");
    // The parent has seen the language fact; the agent hook must neither
    // read that (the subagent has not) nor change it.
    run_logged(
        home.path(),
        &stdin_for("s1", LANGUAGE_BRIEF),
        &Settings::default(),
    )
    .await;
    let before = std::fs::read_to_string(&shown_file).unwrap();
    agent_logged(
        home.path(),
        &agent_stdin("s1", LANGUAGE_BRIEF, None, "Explore"),
    )
    .await;
    let text = delivered(&subagent_logged(
        home.path(),
        &subagent_stdin("s1", "Explore"),
    ));
    assert!(
        text.contains("Russian"),
        "already shown to the parent: {text}"
    );
    assert_eq!(std::fs::read_to_string(&shown_file).unwrap(), before);

    // And a fact stashed for a subagent still reaches the parent.
    agent_logged(
        home.path(),
        &agent_stdin(
            "s2",
            "where does the user live and work from",
            None,
            "Explore",
        ),
    )
    .await;
    assert!(!home.path().join("hooks/s2.shown").exists());
    let (out, _) = run_logged(
        home.path(),
        &stdin_for("s2", "where does the user live and work from"),
        &Settings::default(),
    )
    .await;
    assert!(out.contains("Berlin"), "{out}");
    daemon.shutdown().await;
}

#[tokio::test]
async fn entries_match_by_name_or_type_session_and_order() {
    let (home, daemon, _hook) = with_memory().await;
    let h = home.path();
    let berlin = "where does the user live and work from";
    // Two calls with the same name: a start takes the newest entry first.
    agent_logged(
        h,
        &agent_stdin("s1", LANGUAGE_BRIEF, Some("scout"), "Explore"),
    )
    .await;
    agent_logged(h, &agent_stdin("s1", berlin, Some("scout"), "Explore")).await;
    assert_eq!(stash_files(h).len(), 2);
    // A named agent is matched by name, not by type; another session gets
    // nothing.
    assert_eq!(subagent_logged(h, &subagent_stdin("s1", "Explore")), "");
    assert_eq!(subagent_logged(h, &subagent_stdin("s2", "scout")), "");
    assert_eq!(stash_files(h).len(), 2);
    let first = delivered(&subagent_logged(h, &subagent_stdin("s1", "scout")));
    assert!(first.contains("Berlin"), "{first}");
    let second = delivered(&subagent_logged(h, &subagent_stdin("s1", "scout")));
    assert!(second.contains("Russian"), "{second}");
    assert_eq!(subagent_logged(h, &subagent_stdin("s1", "scout")), "");
    // An unnamed agent is matched by its type.
    agent_logged(h, &agent_stdin("s1", LANGUAGE_BRIEF, None, "Explore")).await;
    assert_eq!(subagent_logged(h, &subagent_stdin("s1", "scout")), "");
    assert!(subagent_logged(h, &subagent_stdin("s1", "Explore")).contains("Russian"));
    daemon.shutdown().await;
}

#[test]
fn the_subagent_hook_escapes_the_text_and_ignores_old_entries() {
    let home = tempfile::tempdir().unwrap();
    let text = "Memory:\n- \"quoted\" back\\slash, Привет, 日本語\n";
    tessar_mcp::stash::put(home.path(), "s1", "k", text).unwrap();
    let printed = subagent_logged(home.path(), &subagent_stdin("s1", "k"));
    assert_eq!(delivered(&printed), text);
    assert_eq!(printed.lines().count(), 1, "one line of JSON: {printed}");

    let old = std::time::SystemTime::now() - Duration::from_secs(11 * 60);
    tessar_mcp::stash::put_at(home.path(), "s1", "k", "stale", old).unwrap();
    assert_eq!(stash_files(home.path()).len(), 1);
    assert_eq!(subagent_logged(home.path(), &subagent_stdin("s1", "k")), "");
    assert!(stash_files(home.path()).is_empty(), "the orphan is removed");
}

#[tokio::test]
async fn both_hooks_are_logged() {
    let (home, daemon, _hook) = with_memory().await;
    agent_logged(
        home.path(),
        &agent_stdin("s1", LANGUAGE_BRIEF, Some("scout"), "Explore"),
    )
    .await;
    subagent_logged(home.path(), &subagent_stdin("s1", "scout"));
    subagent_logged(home.path(), &subagent_stdin("s1", "scout"));
    let recs = records(home.path());
    assert_eq!(recs.len(), 3);

    let a = &recs[0];
    assert_eq!(a["kind"], "agent");
    assert_eq!(a["session"], "s1");
    assert_eq!(a["key"], "scout");
    assert_eq!(a["brief"], LANGUAGE_BRIEF);
    assert_eq!(a["brief_chars"], LANGUAGE_BRIEF.chars().count());
    assert_eq!(a["stashed"], true);
    assert!(a["stashed_chars"].as_u64().unwrap() > 0);
    assert_eq!(a["thresholds"]["budget"], 1500);
    assert!(a["elapsed_ms"].is_u64() && a["project"].is_string());
    assert!(
        a["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["decision"] == "shown")
    );

    let d = &recs[1];
    assert_eq!(d["kind"], "subagent");
    assert_eq!(
        (d["session"].as_str(), d["key"].as_str()),
        (Some("s1"), Some("scout"))
    );
    assert_eq!(d["delivered"], true);
    assert_eq!(d["chars"], a["stashed_chars"]);
    assert!(d["age_ms"].is_u64());

    let n = &recs[2];
    assert_eq!(n["delivered"], false);
    assert_eq!(n["chars"], 0);
    assert!(n.get("age_ms").is_none());

    // A long brief is cut.
    let long = format!("{LANGUAGE_BRIEF} {}", "x ".repeat(2000));
    agent_logged(home.path(), &agent_stdin("s1", &long, None, "Explore")).await;
    let last = records(home.path()).pop().unwrap();
    assert_eq!(last["brief"].as_str().unwrap().chars().count(), 2000);
    assert_eq!(last["brief_chars"], long.chars().count());
    daemon.shutdown().await;
}

/// The real binary: `hook subagent` prints the JSON and nothing else, `hook
/// agent` prints nothing even when it fails, `hook config` has both entries
/// and `--help` lists the subcommands.
#[test]
fn the_binaries_new_hooks_print_only_what_they_should() {
    use std::io::Write;
    let home = tempfile::tempdir().unwrap();
    let run = |args: &[&str], stdin: &str| {
        let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_tessar"))
            .args(args)
            .arg("--home")
            .arg(home.path())
            .env("HOME", home.path())
            .env("TESSAR_MODEL", home.path().join("no-such-model.gguf"))
            .env_remove("TESSAR_HOME")
            .current_dir(home.path())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(stdin.as_bytes())
            .unwrap();
        child.wait_with_output().unwrap()
    };
    tessar_mcp::stash::put(home.path(), "s1", "scout", "from the stash\n").unwrap();
    let out = run(&["hook", "subagent"], &subagent_stdin("s1", "scout"));
    assert!(out.status.success());
    assert_eq!(
        delivered(&String::from_utf8_lossy(&out.stdout)),
        "from the stash\n"
    );
    let out = run(&["hook", "subagent"], "not json");
    assert!(out.status.success() && out.stdout.is_empty());

    // No daemon: nothing printed, and only an empty entry is left.
    let out = run(
        &["hook", "agent"],
        &agent_stdin("s1", LANGUAGE_BRIEF, Some("scout"), "Explore"),
    );
    assert!(out.status.success() && out.stdout.is_empty());
    assert_eq!(stash_files(home.path()).len(), 1);
    let out = run(&["hook", "subagent"], &subagent_stdin("s1", "scout"));
    assert!(out.status.success() && out.stdout.is_empty());
    assert!(stash_files(home.path()).is_empty());

    let config = std::process::Command::new(env!("CARGO_BIN_EXE_tessar"))
        .args(["hook", "config"])
        .output()
        .unwrap();
    let v: Value = serde_json::from_slice(&config.stdout).unwrap();
    assert_eq!(v["hooks"]["PreToolUse"][0]["matcher"], "Agent");
    assert!(
        v["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .ends_with(" hook agent")
    );
    assert!(
        v["hooks"]["SubagentStart"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .ends_with(" hook subagent")
    );
    let help = std::process::Command::new(env!("CARGO_BIN_EXE_tessar"))
        .arg("--help")
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&help.stdout);
    assert!(help.contains("prompt|session|agent|subagent"), "{help}");
}
