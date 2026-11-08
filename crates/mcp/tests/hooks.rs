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
async fn an_entity_named_in_the_prompt_passes_whatever_the_cosine() {
    let (home, daemon, hook) = with_memory().await;
    // A threshold nothing reaches: only the entity channel can pass.
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
    assert_eq!(out.lines().count(), 1, "{out}");
    assert!(out.contains("empty") && out.contains("`remember`"), "{out}");
    daemon.shutdown().await;
}

#[tokio::test]
async fn run_prompt_uses_the_running_daemon_and_the_cwd_project() {
    let (home, daemon, _hook) = with_memory().await;
    let cwd = tempfile::tempdir().unwrap();
    let stdin = json!({"session_id": "s", "cwd": cwd.path(),
        "prompt": "in which language should the user be talked to"})
    .to_string();
    let out = hook::run_prompt(home.path(), &stdin, &Settings::default())
        .await
        .unwrap();
    assert!(out.contains("Russian"), "{out}");
    // Garbage on stdin is an error for the caller to log, not a panic.
    assert!(
        hook::run_prompt(home.path(), "not json", &Settings::default())
            .await
            .is_err()
    );
    daemon.shutdown().await;
}
