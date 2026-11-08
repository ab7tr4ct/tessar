//! The daemon and the proxy end to end over a real Unix socket in a temp
//! directory, with an in-memory log and a fake embedder: no model, no GPU.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use serde_json::{Value, json};
use tessar_core::clock::SystemClock;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::log::MemLog;
use tessar_core::store::{Memory, Semantic};
use tessar_mcp::daemon::{self, Projects, socket_path};
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

type Client = rmcp::service::RunningService<rmcp::RoleClient, ()>;

/// A session as an agent sees it: an MCP client on one end of a pipe, the
/// proxy serving the other end and forwarding to the daemon.
async fn session(home: &Path, project: &str) -> Client {
    let stream = UnixStream::connect(socket_path(home)).await.unwrap();
    let (server, link) = proxy::connect(stream, project, Path::new("/work/x"))
        .await
        .unwrap();
    let (agent, stdio) = tokio::io::duplex(1 << 20);
    tokio::spawn(proxy::bridge(server, link, stdio));
    ().serve(agent).await.unwrap()
}

async fn call(client: &Client, tool: &'static str, args: Value) -> CallToolResult {
    let args = args.as_object().cloned();
    let mut params = CallToolRequestParams::new(tool);
    params.arguments = args;
    client.peer().call_tool(params).await.unwrap()
}

fn text(result: &CallToolResult) -> String {
    serde_json::to_value(result).unwrap()["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

#[tokio::test]
async fn proxy_forwards_the_tools_and_uses_the_header_project() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let alpha = session(home.path(), "alpha").await;

    let info = alpha.peer().peer_info().unwrap();
    assert_eq!(info.server_info.as_ref().unwrap().name, "tessar");
    assert!(info.capabilities.tools.is_some());
    assert!(
        info.instructions
            .as_deref()
            .unwrap()
            .contains("long-term memory"),
        "{:?}",
        info.instructions
    );

    let tools = alpha.peer().list_all_tools().await.unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    for want in ["remember", "recall", "vocabulary", "about", "update"] {
        assert!(names.contains(&want), "{names:?}");
    }
    let remember = tools.iter().find(|t| t.name == "remember").unwrap();
    assert!(
        remember
            .description
            .as_deref()
            .unwrap()
            .contains("lives_in (single)"),
        "the vocabulary is part of the description"
    );

    let written = call(
        &alpha,
        "remember",
        json!({"facts": [{"kind": "statement",
            "text": "The journal is a JSON Lines file with a checksum per line"}]}),
    )
    .await;
    assert_ne!(written.is_error, Some(true), "{}", text(&written));
    let recalled = call(&alpha, "recall", json!({"query": "journal checksum"})).await;
    let out = text(&recalled);
    assert!(out.contains("JSON Lines file"), "{out}");
    assert!(out.contains("alpha"), "the hit shows its space: {out}");

    // The project entity named in the header exists and carries the fact.
    let about = call(
        &alpha,
        "about",
        json!({"entity": {"key": "alpha", "kind": "project"}}),
    )
    .await;
    assert_ne!(about.is_error, Some(true), "{}", text(&about));
    assert!(text(&about).contains("JSON Lines file"), "{}", text(&about));

    // A tool error comes through as an error result, not a lost connection.
    let bad = call(
        &alpha,
        "update",
        json!({"fact": "f9999", "reason": "wrong"}),
    )
    .await;
    assert_eq!(bad.is_error, Some(true), "{}", text(&bad));
    assert_eq!(text(&bad), "no fact f9999");

    drop(alpha);
    daemon.shutdown().await;
}

#[tokio::test]
async fn sessions_share_one_memory_with_their_own_project() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let alpha = session(home.path(), "alpha").await;
    let beta = session(home.path(), "beta").await;

    call(
        &alpha,
        "remember",
        json!({"facts": [{"kind": "statement", "text": "Alpha deploys on Fridays"}]}),
    )
    .await;
    // Another session sees it through recall, labelled with alpha's space.
    let seen = text(&call(&beta, "recall", json!({"query": "deploys on Fridays"})).await);
    assert!(seen.contains("Alpha deploys on Fridays"), "{seen}");
    assert!(seen.contains("alpha"), "{seen}");
    // But its own project entity is its own.
    let own = call(
        &beta,
        "about",
        json!({"entity": {"key": "beta", "kind": "project"}}),
    )
    .await;
    assert_ne!(own.is_error, Some(true), "{}", text(&own));
    assert!(!text(&own).contains("Fridays"), "{}", text(&own));
    // And a fact written in beta's default space goes to beta.
    call(
        &beta,
        "remember",
        json!({"facts": [{"kind": "statement", "text": "Beta ships on Mondays"}]}),
    )
    .await;
    let own = text(
        &call(
            &beta,
            "about",
            json!({"entity": {"key": "beta", "kind": "project"}}),
        )
        .await,
    );
    assert!(own.contains("Beta ships on Mondays"), "{own}");

    drop((alpha, beta));
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_bad_header_closes_the_connection() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let mut stream = UnixStream::connect(socket_path(home.path())).await.unwrap();
    stream
        .write_all(b"{\"tessar\": 7, \"project\": \"x\"}\n")
        .await
        .unwrap();
    let mut rest = Vec::new();
    let n = tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut rest))
        .await
        .expect("the daemon kept a bad connection open")
        .unwrap();
    assert_eq!(n, 0);
    // The daemon still serves.
    let ok = session(home.path(), "alpha").await;
    drop(ok);
    daemon.shutdown().await;
}

#[tokio::test]
async fn shutdown_ends_open_sessions() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let alpha = session(home.path(), "alpha").await;
    // The session is still open when the daemon is told to stop.
    daemon.shutdown().await;
    assert!(
        tokio::time::timeout(Duration::from_secs(5), alpha.waiting())
            .await
            .is_ok(),
        "the session outlived the daemon"
    );
}

#[tokio::test]
async fn idle_daemon_exits_after_the_last_session() {
    let home = tempfile::tempdir().unwrap();
    let listener = UnixListener::bind(socket_path(home.path())).unwrap();
    let task = tokio::spawn(daemon::run(
        listener,
        projects(),
        Some(Duration::from_millis(400)),
        std::future::pending(),
    ));
    let alpha = session(home.path(), "alpha").await;
    // Busy: still running well past the idle time.
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert!(!task.is_finished(), "exited with a session open");
    drop(alpha);
    tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .expect("did not exit when idle")
        .unwrap();
}
