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

#[tokio::test]
async fn daemon_greets_with_its_build_and_pid_before_mcp() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let mut stream = UnixStream::connect(socket_path(home.path())).await.unwrap();
    stream
        .write_all(b"{\"tessar\":1,\"project\":\"alpha\",\"build\":\"x\"}\n")
        .await
        .unwrap();
    let mut line = String::new();
    BufReader::new(&mut stream)
        .read_line(&mut line)
        .await
        .unwrap();
    assert_eq!(
        line,
        format!(
            "{{\"tessar\":1,\"build\":\"{}\",\"pid\":{}}}\n",
            tessar_mcp::BUILD_ID,
            std::process::id()
        )
    );
    drop(stream);
    daemon.shutdown().await;
}

#[tokio::test]
async fn open_reports_the_daemons_build_and_a_session_still_works() {
    let home = tempfile::tempdir().unwrap();
    let (stop, rx) = oneshot::channel::<()>();
    let listener = UnixListener::bind(socket_path(home.path())).unwrap();
    let task = tokio::spawn(daemon::run_as(
        "0.0.1+old.1",
        listener,
        projects(),
        None,
        async {
            let _ = rx.await;
        },
    ));
    let stream = UnixStream::connect(socket_path(home.path())).await.unwrap();
    let opened = proxy::open(stream, "alpha", Path::new("/work/x"))
        .await
        .unwrap();
    assert_eq!(opened.hello().build, "0.0.1+old.1");
    assert_eq!(opened.hello().pid, std::process::id());
    assert!(proxy::needs_restart(
        tessar_mcp::BUILD_ID,
        &opened.hello().build,
        false
    ));
    let (_server, link) = opened.serve().await.unwrap();

    // Status while a session is open.
    let stream = UnixStream::connect(socket_path(home.path())).await.unwrap();
    let status = proxy::status(stream).await.unwrap();
    assert_eq!(status.build, "0.0.1+old.1");
    assert_eq!(status.sessions, Some(1));
    assert!(status.uptime_secs.is_some());

    drop(link);
    let _ = stop.send(());
    tokio::time::timeout(Duration::from_secs(15), task)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn open_gives_up_on_a_daemon_without_a_hello() {
    let home = tempfile::tempdir().unwrap();
    let listener = UnixListener::bind(socket_path(home.path())).unwrap();
    // An old daemon: accepts, reads, says nothing.
    let silent = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        tokio::time::sleep(Duration::from_secs(30)).await;
        drop(stream);
    });
    let stream = UnixStream::connect(socket_path(home.path())).await.unwrap();
    let err = proxy::open(stream, "alpha", Path::new("/work/x"))
        .await
        .err()
        .expect("no hello");
    assert!(err.contains("predates the build handshake"), "{err}");
    silent.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lost_daemon_fails_calls_one_by_one_and_the_proxy_reconnects() {
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().to_path_buf();
    let first = Daemon::start(&dir, None);
    let stream = UnixStream::connect(socket_path(&dir)).await.unwrap();
    let (server, link) = proxy::connect(stream, "alpha", Path::new("/work/x"))
        .await
        .unwrap();
    let reconnect: proxy::Reconnect = {
        let dir = dir.clone();
        Arc::new(move || {
            let dir = dir.clone();
            Box::pin(async move {
                let stream = UnixStream::connect(socket_path(&dir))
                    .await
                    .map_err(|e| e.to_string())?;
                let (server, link) = proxy::connect(stream, "alpha", Path::new("/work/x")).await?;
                Ok((server.daemon(), link))
            })
        })
    };
    let (agent, stdio) = tokio::io::duplex(1 << 20);
    let bridge = tokio::spawn(proxy::bridge_reconnecting(server, link, stdio, reconnect));
    let client: Client = ().serve(agent).await.unwrap();

    let remember = |text: &'static str| {
        let mut params = CallToolRequestParams::new("remember");
        params.arguments = json!({"facts": [{"kind": "statement", "text": text}]})
            .as_object()
            .cloned();
        params
    };
    client
        .peer()
        .call_tool(remember("Before the restart"))
        .await
        .unwrap();

    first.shutdown().await;
    // The call fails for itself; the session and the proxy stay up.
    let lost = client.peer().call_tool(remember("While it is down")).await;
    assert!(lost.is_err(), "{lost:?}");
    assert!(!bridge.is_finished(), "the proxy exited with the daemon");

    let _ = std::fs::remove_file(socket_path(&dir));
    let second = Daemon::start(&dir, None);
    let mut healed = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if client
            .peer()
            .call_tool(remember("After the restart"))
            .await
            .is_ok()
        {
            healed = true;
            break;
        }
    }
    assert!(healed, "the proxy did not reconnect");

    drop(client);
    assert!(bridge.await.unwrap().unwrap(), "the session ended normally");
    second.shutdown().await;
}

/// A proxy whose daemon drops every session keeps trying, but with growing
/// pauses: at once, then after the pause, then after twice that, and so on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_daemon_that_drops_every_session_is_not_hammered() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().to_path_buf();
    let first = Daemon::start(&dir, None);
    let stream = UnixStream::connect(socket_path(&dir)).await.unwrap();
    let (server, link) = proxy::connect(stream, "alpha", Path::new("/work/x"))
        .await
        .unwrap();
    let reconnect: proxy::Reconnect = {
        let dir = dir.clone();
        Arc::new(move || {
            let dir = dir.clone();
            Box::pin(async move {
                let stream = UnixStream::connect(socket_path(&dir))
                    .await
                    .map_err(|e| e.to_string())?;
                let (server, link) = proxy::connect(stream, "alpha", Path::new("/work/x")).await?;
                Ok((server.daemon(), link))
            })
        })
    };
    let backoff = proxy::Backoff {
        pause: Duration::from_millis(200),
        cap: Duration::from_secs(30),
    };
    let (agent, stdio) = tokio::io::duplex(1 << 20);
    let bridge = tokio::spawn(proxy::bridge_reconnecting_with(
        server, link, stdio, reconnect, backoff,
    ));
    let client: Client = ().serve(agent).await.unwrap();
    // The first connection lives longer than the pause, so losing it is a
    // fresh loss and the proxy reconnects at once.
    tokio::time::sleep(backoff.pause * 2).await;

    // A daemon that greets and hangs up at once takes over the socket. The
    // new listener is renamed into place before the old daemon stops, so
    // the proxy never finds the socket without a listener.
    let staging = dir.join("staging.sock");
    let listener = UnixListener::bind(&staging).unwrap();
    std::fs::rename(&staging, socket_path(&dir)).unwrap();
    let accepted = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&accepted);
    let fake = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            count.fetch_add(1, Ordering::SeqCst);
            let (r, mut w) = tokio::io::split(stream);
            let mut line = String::new();
            let _ = BufReader::new(r).read_line(&mut line).await;
            let _ = w
                .write_all(b"{\"tessar\":1,\"build\":\"x\",\"pid\":1}\n")
                .await;
        }
    });
    first.shutdown().await;
    // Attempts come at about 0, 0.2, 0.6, 1.4 and 3.0 s after the loss.
    tokio::time::sleep(Duration::from_millis(1000)).await;
    let attempts = accepted.load(Ordering::SeqCst);
    assert!(
        attempts >= 2,
        "only {attempts} reconnects in 1 s: it gave up"
    );
    assert!(attempts <= 4, "{attempts} reconnects in 1 s: no backoff");

    fake.abort();
    drop(client);
    assert!(bridge.await.unwrap().unwrap());
}

/// The real `tessar mcp` binary on stdio, in front of the in-process daemon,
/// with two `tools/call` requests written back to back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stdio_proxy_survives_back_to_back_calls() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_tessar"))
        .args(["mcp", "--keep-daemon"])
        .env("TESSAR_HOME", home.path())
        .current_dir(home.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap()).lines();
    let call = |id: u64, space: &str, n: usize| {
        let facts: Vec<Value> = (0..n)
            .map(|i| {
                json!({"kind": "statement", "space": space,
                    "text": format!("Stdio {space} fact number {i}")})
            })
            .collect();
        json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {
            "name": "remember", "arguments": {"facts": facts}, "_meta": {"progressToken": id}}})
    };
    let lines = [
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        call(2, "project", 5),
        call(3, "user", 2),
    ];
    // One write, as a client that does not wait would produce.
    let burst: String = lines.iter().map(|l| format!("{l}\n")).collect();
    stdin.write_all(burst.as_bytes()).await.unwrap();
    stdin.flush().await.unwrap();

    let mut seen = std::collections::BTreeMap::new();
    while seen.len() < 3 {
        let line = tokio::time::timeout(Duration::from_secs(20), out.next_line())
            .await
            .expect("no response from the proxy")
            .unwrap()
            .expect("the proxy closed stdout before answering both calls");
        let v: Value = serde_json::from_str(&line).unwrap();
        seen.insert(v["id"].as_u64().unwrap(), v);
    }
    for id in [2, 3] {
        assert!(seen[&id]["error"].is_null(), "{}", seen[&id]);
        assert_eq!(seen[&id]["result"]["isError"], false, "{}", seen[&id]);
    }
    drop(stdin);
    let status = tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .expect("the proxy did not exit on stdin EOF")
        .unwrap();
    assert!(status.success(), "{status}");
    daemon.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_remember_calls_in_two_spaces_both_succeed() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let alpha = session(home.path(), "alpha").await;

    let facts = |space: &str, n: usize| {
        let facts: Vec<Value> = (0..n)
            .map(|i| {
                json!({"kind": "statement", "space": space,
                    "text": format!("Parallel {space} fact number {i}")})
            })
            .collect();
        json!({ "facts": facts })
    };
    let (a, b) = tokio::join!(
        call(&alpha, "remember", facts("project", 5)),
        call(&alpha, "remember", facts("user", 2)),
    );
    assert_ne!(a.is_error, Some(true), "{}", text(&a));
    assert_ne!(b.is_error, Some(true), "{}", text(&b));
    let recalled = call(&alpha, "recall", json!({"query": "Parallel fact number"})).await;
    let out = text(&recalled);
    assert!(out.contains("Parallel project fact"), "{out}");
    assert!(out.contains("Parallel user fact"), "{out}");

    drop(alpha);
    daemon.shutdown().await;
}

// ---- the daemon's life: silent clients, handover, starting ------------

/// A client that connects and sends nothing must not keep the daemon from
/// stopping for the whole grace period.
#[tokio::test]
async fn a_silent_client_does_not_hold_up_the_shutdown() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), None);
    let silent = UnixStream::connect(socket_path(home.path())).await.unwrap();
    // Let the daemon accept it.
    let ok = session(home.path(), "alpha").await;
    drop(ok);
    let started = std::time::Instant::now();
    daemon.shutdown().await;
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "the shutdown waited {:?} for a client that never sent a header",
        started.elapsed()
    );
    drop(silent);
}

/// The same client is dropped after a few seconds without a shutdown, so it
/// cannot keep an idle daemon alive for ever. The clock is paused: the wait
/// is the daemon's own limit, whatever it is, and takes no real time.
#[tokio::test(start_paused = true)]
async fn a_client_that_sends_no_header_is_dropped() {
    let (client, server) = tokio::io::duplex(1024);
    let (_stop, stopped) = tokio::sync::watch::channel(false);
    let state = Arc::new(daemon::State::new("test"));
    let started = tokio::time::Instant::now();
    let err = daemon::serve_connection(server, projects(), state, stopped, 1)
        .await
        .expect_err("served a connection without a header");
    assert!(err.contains("no header within"), "{err}");
    assert!(started.elapsed() <= Duration::from_secs(30));
    drop(client);
}

/// A spawner that counts its calls and hands each one to `start`.
fn counting(
    start: impl Fn(usize, &Path, proxy::Start) -> Result<std::process::Child, String>
    + Send
    + Sync
    + 'static,
) -> (proxy::Spawn, Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&calls);
    let spawn: proxy::Spawn = Arc::new(move |home, mode| {
        let nth = count.fetch_add(1, Ordering::SeqCst) + 1;
        start(nth, home, mode)
    });
    (spawn, calls)
}

/// Processes standing in for daemons that keep running. Each one reads a
/// pipe whose other end is kept here, so they all end when this value is
/// dropped, however the test ends; whoever holds the `Child` reaps it.
#[derive(Clone, Default)]
struct StandIns(Arc<std::sync::Mutex<Vec<std::process::ChildStdin>>>);

impl StandIns {
    fn start(&self) -> Result<std::process::Child, String> {
        let mut child = std::process::Command::new("cat")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;
        self.0.lock().unwrap().push(child.stdin.take().unwrap());
        Ok(child)
    }
}

/// A process standing in for an old daemon: something for a SIGTERM to hit.
/// On the signal it runs `on_signal`, lingers for `linger` seconds, exits
/// and runs `on_exit`. It also ends, without either, when the value
/// returned is dropped: it waits on a pipe whose other end is kept there.
struct OldDaemonProcess {
    pid: u32,
    _alive: std::process::ChildStdin,
}

fn old_daemon_process(
    linger: &str,
    on_signal: impl FnOnce() + Send + 'static,
    on_exit: impl FnOnce() + Send + 'static,
) -> OldDaemonProcess {
    use std::io::BufRead;
    let script = format!(
        "trap 'echo term; sleep {linger}; exit 0' TERM; exec 3<&0; echo ready; \
         cat <&3 >/dev/null & wait"
    );
    let mut child = std::process::Command::new("sh")
        .args(["-c", &script])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let alive = child.stdin.take().unwrap();
    let mut out = std::io::BufReader::new(child.stdout.take().unwrap());
    // Not before it handles the signal.
    let mut said = String::new();
    out.read_line(&mut said).unwrap();
    assert_eq!(said, "ready\n");
    std::thread::spawn(move || {
        let mut said = String::new();
        let _ = out.read_line(&mut said);
        if said == "term\n" {
            on_signal();
            let _ = child.wait();
            on_exit();
        } else {
            let _ = child.wait();
        }
    });
    OldDaemonProcess { pid, _alive: alive }
}

/// A listener on the socket of a home that answers every connection as a
/// daemon with the given pid would: it reads the header and sends a hello.
/// It runs on a thread of its own, for tests without a runtime, and stops
/// when dropped.
struct Greeter {
    stop: Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Greeter {
    fn start(home: &Path, pid: u32) -> Self {
        use std::io::{BufRead, Write};
        use std::sync::atomic::{AtomicBool, Ordering};
        let listener = std::os::unix::net::UnixListener::bind(socket_path(home)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            while !stopped.load(Ordering::SeqCst) {
                let Ok((stream, _)) = listener.accept() else {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                };
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
                let mut header = String::new();
                let _ = std::io::BufReader::new(&stream).read_line(&mut header);
                let hello = format!("{{\"tessar\":1,\"build\":\"0.0.1+old.1\",\"pid\":{pid}}}\n");
                let _ = (&stream).write_all(hello.as_bytes());
            }
        });
        Self {
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for Greeter {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Runs the real `tessar` with `args` on `home`, pointed at a missing model
/// and away from the user's own home, so that nothing outside `home` is
/// touched: a daemon that gets the journal fails on the model.
fn tessar(home: &Path, args: &[&str]) -> std::process::Command {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_tessar"));
    command
        .args(args)
        .arg("--home")
        .arg(home)
        .env("HOME", home)
        .env("TESSAR_MODEL", home.join("no-such-model.gguf"))
        .env_remove("TESSAR_HOME")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    command
}

/// The first run of a binary that was just linked can take most of a second
/// (the system checks it). Paying for that first lets the daemons in a test
/// start as fast as real ones.
fn warm_up_the_binary() {
    let version = std::process::Command::new(env!("CARGO_BIN_EXE_tessar"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
}

/// Waits up to `wait` for `child` to exit; kills it if it does not.
fn exit_within(
    mut child: std::process::Child,
    wait: Duration,
) -> Result<std::process::Output, String> {
    let deadline = std::time::Instant::now() + wait;
    while child.try_wait().unwrap().is_none() {
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("still running after {wait:?}"));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(child.wait_with_output().unwrap())
}

/// A process standing in for a daemon that exits at once with `status`.
fn exited_process(status: &str) -> Result<std::process::Child, String> {
    std::process::Command::new(status)
        .spawn()
        .map_err(|e| e.to_string())
}

/// A proxy that lost its daemon only looks for a new one at first: the
/// daemon is most likely being replaced, and one started now from this
/// proxy's binary could take the journal from the replacement.
#[tokio::test]
async fn a_reconnecting_proxy_starts_no_daemon_while_a_replacement_comes_up() {
    use std::sync::atomic::Ordering;
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().to_path_buf();
    let (spawn, calls) = counting(|_, _, _| Err("started a daemon during the grace".to_owned()));
    let launcher = proxy::Launcher {
        wait: Duration::from_secs(10),
        grace: Duration::from_secs(10),
        respawn_pause: Duration::from_secs(1),
        spawn,
    };
    // The replacement listens a little later.
    let replacement = {
        let dir = dir.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            Daemon::start(&dir, None)
        })
    };
    let stream = proxy::connect_or_start_with(&dir, &launcher)
        .await
        .expect("did not find the replacement");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    drop(stream);
    replacement.await.unwrap().shutdown().await;

    // The default for a first connection has no grace, a reconnecting one
    // has a few seconds.
    let first = proxy::Launcher::new(Duration::from_secs(60));
    assert_eq!(first.grace, Duration::ZERO);
    let again = first.reconnecting();
    assert_eq!(again.grace, proxy::RECONNECT_GRACE);
    assert!(again.grace >= Duration::from_secs(2) && again.grace <= Duration::from_secs(5));
}

/// With nothing listening after the grace, the proxy starts a daemon. If
/// that daemon exits cleanly (another process held the journal) and still
/// nothing listens a pause later, the holder was on its way out: the proxy
/// starts a daemon again instead of waiting out its whole time limit.
#[tokio::test]
async fn a_daemon_is_started_again_after_one_that_stepped_aside() {
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().to_path_buf();
    let second: Arc<Mutex<Option<Daemon>>> = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&second);
    let stand_ins = StandIns::default();
    let (spawn, calls) = counting(move |nth, home, mode| {
        assert_eq!(mode, proxy::Start::Fresh);
        if nth == 1 {
            return exited_process("true");
        }
        *slot.lock().unwrap() = Some(Daemon::start(home, None));
        stand_ins.start()
    });
    let launcher = proxy::Launcher {
        wait: Duration::from_secs(20),
        grace: Duration::from_millis(300),
        respawn_pause: Duration::from_millis(300),
        spawn,
    };
    let started = std::time::Instant::now();
    let stream = proxy::connect_or_start_with(&dir, &launcher)
        .await
        .expect("no daemon after the first one stepped aside");
    assert!(started.elapsed() >= launcher.grace + launcher.respawn_pause);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    drop(stream);
    let daemon = second.lock().unwrap().take().unwrap();
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_daemon_that_fails_to_start_is_an_error_and_is_not_started_again() {
    use std::sync::atomic::Ordering;
    let home = tempfile::tempdir().unwrap();
    let (spawn, calls) = counting(|_, _, _| exited_process("false"));
    let launcher = proxy::Launcher {
        wait: Duration::from_secs(20),
        grace: Duration::ZERO,
        respawn_pause: Duration::from_millis(100),
        spawn,
    };
    let err = proxy::connect_or_start_with(home.path(), &launcher)
        .await
        .expect_err("connected to nothing");
    assert!(err.contains("the daemon exited with"), "{err}");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// The upgrade, with a client of the old kind in the way.
///
/// An older daemon holds the journal and serves two clients: a proxy of the
/// build before the handover, and then the new proxy, which retires the
/// daemon. The old proxy does what that build did when its connection
/// dropped: it starts a daemon from its own binary at once (here the real
/// `tessar daemon`, pointed at a missing model: if it gets the journal it
/// fails on the model and exits with an error; if the journal is taken it
/// steps aside with status 0).
///
/// The upgrade sticks only if the replacement gets the journal: it is in
/// line for it before it tells the old daemon to stop.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_upgrade_survives_an_old_proxy_that_starts_its_own_daemon_at_once() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use tessar_core::store::Op;
    use tessar_journal::FileJournal;
    use tessar_mcp::lifecycle;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

    let home = tempfile::tempdir().unwrap();
    let dir = home.path().to_path_buf();
    // So that the old proxy's daemon below does reach the journal while it
    // is free.
    warm_up_the_binary();

    // The old daemon. Its pid is a stand-in process, so the SIGTERM of the
    // replacement has something to end. On the signal the daemon stops as
    // the real one does: no listener, sessions closed, the journal released
    // at once; the process lingers for a moment and removes the socket file
    // last. So the journal is free for a while with nobody but a daemon
    // already waiting for it to take it: the window the old proxy's daemon
    // falls into.
    let journal = FileJournal::<Op>::open(dir.join("journal.jsonl")).unwrap();
    let (on_signal, mut signalled) = oneshot::channel::<()>();
    let (on_exit, exited) = oneshot::channel::<()>();
    let old_process = old_daemon_process(
        "0.5",
        move || {
            let _ = on_signal.send(());
        },
        move || {
            let _ = on_exit.send(());
        },
    );
    let old_pid = old_process.pid;
    let listener = UnixListener::bind(socket_path(&dir)).unwrap();
    let old_daemon = {
        let socket = socket_path(&dir);
        tokio::spawn(async move {
            let mut sessions = Vec::new();
            loop {
                tokio::select! {
                    _ = &mut signalled => break,
                    accepted = listener.accept() => {
                        let (stream, _) = accepted.unwrap();
                        let (r, mut w) = tokio::io::split(stream);
                        let mut r = BufReader::new(r);
                        let mut header = String::new();
                        r.read_line(&mut header).await.unwrap();
                        let hello = format!(
                            "{{\"tessar\":1,\"build\":\"0.0.1+old.1\",\"pid\":{old_pid}}}\n"
                        );
                        w.write_all(hello.as_bytes()).await.unwrap();
                        sessions.push((r, w));
                    }
                }
            }
            drop(listener);
            drop(sessions);
            drop(journal);
            let _ = exited.await;
            let _ = std::fs::remove_file(socket);
        })
    };

    // The old proxy.
    let (connected, old_proxy_is_connected) = oneshot::channel::<()>();
    let old_proxy = {
        let dir = dir.clone();
        tokio::spawn(async move {
            let mut stream = UnixStream::connect(socket_path(&dir)).await.unwrap();
            stream
                .write_all(b"{\"tessar\":1,\"project\":\"beta\",\"build\":\"0.0.1+old.1\"}\n")
                .await
                .unwrap();
            let mut hello = [0u8; 1];
            stream.read_exact(&mut hello).await.unwrap();
            let _ = connected.send(());
            // The connection drops when the daemon is retired...
            let mut rest = Vec::new();
            let _ = stream.read_to_end(&mut rest).await;
            // ...nothing listens any more, and it starts its own daemon.
            assert!(UnixStream::connect(socket_path(&dir)).await.is_err());
            tokio::process::Command::from(tessar(&dir, &["daemon"]))
                .output()
                .await
                .unwrap()
        })
    };
    old_proxy_is_connected.await.unwrap();

    // The new proxy. Its replacement daemon runs in this process: it gets
    // the journal the way `tessar daemon --takeover PID` does (it stops the
    // old daemon itself, once it is in line), holds it, and listens only
    // after a while, as a daemon that loads a model first.
    let has_journal = Arc::new(AtomicBool::new(false));
    let stop_replacement = Arc::new(tokio::sync::Notify::new());
    let (replacement_done, replacement_ended) = std::sync::mpsc::channel::<()>();
    let stand_ins = StandIns::default();
    let (spawn, calls) = {
        let has_journal = Arc::clone(&has_journal);
        let stop = Arc::clone(&stop_replacement);
        let stand_ins = stand_ins.clone();
        counting(move |nth, home, mode| {
            assert_eq!(nth, 1, "a second daemon was started");
            assert_eq!(mode, proxy::Start::Takeover(old_pid));
            let home = home.to_path_buf();
            let runtime = tokio::runtime::Handle::current();
            let (has_journal, stop) = (Arc::clone(&has_journal), Arc::clone(&stop));
            let done = replacement_done.clone();
            std::thread::spawn(move || {
                let journal = match lifecycle::take_over(&home, old_pid, Duration::from_secs(30)) {
                    Ok(lifecycle::Takeover::Journal(journal)) => journal,
                    Ok(lifecycle::Takeover::Aside(why)) => {
                        panic!("the replacement did not get the journal: {why}")
                    }
                    Err(e) => panic!("the replacement did not get the journal: {e}"),
                };
                has_journal.store(true, Ordering::SeqCst);
                // Well after the old daemon has removed its socket file.
                std::thread::sleep(Duration::from_millis(1500));
                runtime.block_on(async {
                    let socket = socket_path(&home);
                    let _ = std::fs::remove_file(&socket);
                    let listener = UnixListener::bind(&socket).unwrap();
                    daemon::run(listener, projects(), None, stop.notified()).await;
                });
                drop(journal);
                let _ = done.send(());
            });
            stand_ins.start()
        })
    };
    let launcher = proxy::Launcher {
        wait: Duration::from_secs(30),
        grace: Duration::ZERO,
        respawn_pause: Duration::from_secs(1),
        spawn,
    };
    let opened = proxy::open_current_with(&dir, "alpha", Path::new("/work/x"), false, &launcher)
        .await
        .expect("the new proxy did not get a daemon");
    assert_eq!(
        opened.hello().build,
        tessar_mcp::BUILD_ID,
        "the new proxy ended up on another build"
    );
    assert!(has_journal.load(Ordering::SeqCst));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // The old proxy's daemon found the journal taken and stepped aside.
    let theirs = tokio::time::timeout(Duration::from_secs(30), old_proxy)
        .await
        .expect("the old proxy's daemon did not exit")
        .unwrap();
    let said = String::from_utf8_lossy(&theirs.stderr);
    assert!(
        theirs.status.success() && said.contains("already running"),
        "the old proxy's daemon took the journal from the replacement: {} {said}",
        theirs.status
    );
    // And the old proxy, looking again, finds the newer daemon.
    let stream = UnixStream::connect(socket_path(&dir)).await.unwrap();
    assert_eq!(
        proxy::status(stream).await.unwrap().build,
        tessar_mcp::BUILD_ID
    );

    drop(opened);
    old_daemon.await.unwrap();
    stop_replacement.notify_one();
    replacement_ended
        .recv_timeout(Duration::from_secs(15))
        .expect("the replacement daemon did not stop");
}

/// What the takeover tests below share: the real `tessar daemon --takeover
/// PID` against a journal held here and a process standing in for the old
/// daemon, which on its SIGTERM keeps the journal for `hold` more and then
/// lets go; at that very moment a daemon of the kind an old proxy starts
/// (the real `tessar daemon`) goes for the journal too.
///
/// The takeover signals only a process it can tell is the daemon of the
/// home, so the stand-in is given the daemon's voice: a listener on the
/// socket that answers every connection with a hello naming its pid.
///
/// The model is missing, so a daemon that gets the journal would fail at
/// once and let it go again. To see who has it, the pid file is a FIFO: the
/// daemon writes its pid as soon as it holds the journal, and that blocks
/// until the test reads. Whoever holds the journal is stuck there, holding
/// it, and its pid is what the test reads. (A FIFO is never read as a pid
/// file, so nothing blocks on it but that write. Once the test has read
/// it, the takeover goes on, finds no model and exits; the guard below
/// kills it if it has not by then, and reaps it.)
fn takeover_by_the_binary(hold: Duration) {
    use tessar_core::store::Op;
    use tessar_journal::FileJournal;
    use tessar_mcp::lifecycle;

    warm_up_the_binary();
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().to_path_buf();
    let fifo = lifecycle::pid_path(&dir);
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    let journal = FileJournal::<Op>::open(dir.join("journal.jsonl")).unwrap();

    enum Event {
        Signalled,
        Released(std::process::Child),
    }
    let (events, seen) = std::sync::mpsc::channel();
    let old = {
        let dir = dir.clone();
        old_daemon_process(
            "0.2",
            move || {
                let _ = events.send(Event::Signalled);
                std::thread::sleep(hold);
                drop(journal);
                let eager = tessar(&dir, &["daemon"]).spawn().unwrap();
                let _ = events.send(Event::Released(eager));
            },
            || {},
        )
    };

    let _voice = Greeter::start(&dir, old.pid);

    let takeover = tessar(&dir, &["daemon", "--takeover", &old.pid.to_string()])
        .spawn()
        .unwrap();
    let takeover_pid = takeover.id();
    // Killed and reaped however the test ends.
    struct Ended(Option<std::process::Child>);
    impl Drop for Ended {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let mut takeover = Ended(Some(takeover));

    // It tells the process it was given, and nobody else, to stop.
    let wait = Duration::from_secs(30);
    assert!(
        matches!(seen.recv_timeout(wait), Ok(Event::Signalled)),
        "the takeover did not signal the old daemon"
    );
    // And waits for the journal for as long as that one keeps it.
    if !hold.is_zero() {
        std::thread::sleep(hold / 2);
        let running = takeover.0.as_mut().unwrap().try_wait().unwrap().is_none();
        assert!(running, "the takeover did not wait for the journal");
        assert!(lifecycle::journal_in_use(&dir));
    }
    let Ok(Event::Released(eager)) = seen.recv_timeout(wait) else {
        panic!("the old daemon did not let go");
    };
    // The other daemon finds the journal taken and steps aside. If it had
    // got the journal it would be stuck on the pid file and never exit.
    let eager = exit_within(eager, wait).expect("the other daemon took the journal");
    let said = String::from_utf8_lossy(&eager.stderr);
    assert!(
        eager.status.success() && said.contains("already running"),
        "{} {said}",
        eager.status
    );
    // The journal is with the takeover: its pid is in the pid file.
    let (read, pid) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = read.send(std::fs::read_to_string(fifo));
    });
    let pid = pid
        .recv_timeout(wait)
        .expect("nobody wrote the pid file")
        .unwrap();
    assert_eq!(pid.trim(), takeover_pid.to_string());
    drop(old);
}

/// The takeover waits for a journal that is held, and has it when it is
/// released.
#[test]
fn the_takeover_binary_waits_for_the_journal_and_takes_it() {
    takeover_by_the_binary(Duration::from_millis(600));
}

/// The worst case for the order: the old daemon lets go the moment it is
/// signalled, and another daemon is started in the same moment. The
/// takeover wins because it was in line before it sent the signal.
#[test]
fn the_takeover_binary_is_in_line_before_it_stops_the_old_daemon() {
    takeover_by_the_binary(Duration::ZERO);
}

/// A daemon that finds the journal locked looks once more before it steps
/// aside: a check whether the journal is in use holds the lock for an
/// instant, and must not make a daemon give up with nobody running.
#[test]
fn a_daemon_does_not_step_aside_for_a_lock_held_for_an_instant() {
    use tessar_core::store::Op;
    use tessar_journal::FileJournal;

    warm_up_the_binary();
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().to_path_buf();
    // Held when the daemon first looks, free well before it looks again.
    let journal = FileJournal::<Op>::open(dir.join("journal.jsonl")).unwrap();
    // A daemon takes some 10 ms to start and look, and looks again 50 ms
    // later. (Should it be slower than this pause, it finds the journal
    // free at its first look, and the test passes without testing much.)
    let daemon = tessar(&dir, &["daemon"]).spawn().unwrap();
    std::thread::sleep(Duration::from_millis(25));
    drop(journal);
    let out = exit_within(daemon, Duration::from_secs(30)).unwrap();
    let said = String::from_utf8_lossy(&out.stderr);
    // It got the journal on the second look and went on to the model.
    assert!(
        !out.status.success() && said.contains("embedding model not found"),
        "{} {said}",
        out.status
    );

    // A journal that stays held is still stepped aside for.
    let journal = FileJournal::<Op>::open(dir.join("journal.jsonl")).unwrap();
    let out = exit_within(
        tessar(&dir, &["daemon"]).spawn().unwrap(),
        Duration::from_secs(30),
    )
    .unwrap();
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && said.contains("already running"),
        "{} {said}",
        out.status
    );
    drop(journal);
}

/// An old daemon that is hung: it greeted this proxy once, and then accepts
/// connections without ever answering. While the proxy waits for the
/// replacement, a daemon that says nothing is not the replacement: the wait
/// must end with "no daemon answered", not hand that connection on to fail
/// later as a daemon "without the handshake".
#[tokio::test]
async fn a_daemon_that_says_nothing_is_not_taken_for_the_replacement() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().to_path_buf();
    let listener = UnixListener::bind(socket_path(&dir)).unwrap();
    let old_pid = std::process::id();
    let old_daemon = tokio::spawn(async move {
        let mut held = Vec::new();
        let mut first = true;
        while let Ok((stream, _)) = listener.accept().await {
            let (r, mut w) = tokio::io::split(stream);
            let mut r = BufReader::new(r);
            if std::mem::take(&mut first) {
                let mut header = String::new();
                let _ = r.read_line(&mut header).await;
                let hello =
                    format!("{{\"tessar\":1,\"build\":\"0.0.1+old.1\",\"pid\":{old_pid}}}\n");
                let _ = w.write_all(hello.as_bytes()).await;
            }
            held.push((r, w));
        }
    });
    // The replacement never comes up; nothing here signals anybody.
    let stand_ins = StandIns::default();
    let (spawn, calls) = {
        let stand_ins = stand_ins.clone();
        counting(move |_, _, mode| {
            assert_eq!(mode, proxy::Start::Takeover(old_pid));
            stand_ins.start()
        })
    };
    let launcher = proxy::Launcher {
        wait: Duration::from_millis(700),
        grace: Duration::ZERO,
        respawn_pause: Duration::from_secs(60),
        spawn,
    };
    let started = std::time::Instant::now();
    let err = proxy::open_current_with(&dir, "alpha", Path::new("/work/x"), false, &launcher)
        .await
        .err()
        .expect("a daemon that never answered was taken for the new one");
    assert!(err.contains("no daemon answered"), "{err}");
    assert!(!err.contains("predates"), "{err}");
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    old_daemon.abort();
}
