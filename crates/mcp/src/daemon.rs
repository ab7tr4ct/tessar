//! The daemon: one process per machine holds the memory and the model and
//! serves MCP to many sessions over a Unix socket.
//!
//! # Protocol
//!
//! A connection starts with one line of JSON from the client, the
//! [`Header`], naming the project the session runs in and the client's
//! build id:
//!
//! ```text
//! {"tessar":1,"project":"repo","cwd":"/work/repo","build":"0.1.0+3ff4d59.1731100000"}
//! ```
//!
//! A client that knows Claude Code's session id adds `"session":"<id>"`, and
//! one that knows its role `"client":"mcp"` (or `hook`, `cli`); the daemon
//! records both in its tool log, and an older daemon ignores the fields.
//!
//! The daemon answers with one line, the [`Hello`], naming itself, before
//! any MCP traffic:
//!
//! ```text
//! {"tessar":1,"build":"0.1.0+3ff4d59.1731100000","pid":4242}
//! ```
//!
//! Then the rest of the stream is plain MCP, served by a [`TessarServer`]
//! whose project space is the header's. The client compares the build ids.
//! If they differ, a proxy retires the daemon (SIGTERM to `pid`, see
//! [`crate::lifecycle`]) and starts a new one; a hook uses the daemon as
//! it is.
//!
//! A header `{"tessar":1,"status":true}` asks for the daemon's state
//! instead: the daemon answers one hello line extended with `uptime_secs`
//! and `sessions` (open MCP sessions) and closes the connection. A daemon
//! older than the handshake sends no hello and never answers; clients give
//! up on it after a few seconds.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use rmcp::ServiceExt;
use serde::{Deserialize, Serialize};
use tessar_core::clock::Clock;
use tessar_core::store::{Memory, StoreError};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::task::JoinSet;
use tokio::time::Instant;

use crate::BUILD_ID;
use crate::api::Tessar;
use crate::server::TessarServer;
use crate::tool_log::ToolLog;

/// The protocol version of the header line.
pub const PROTOCOL: u32 = 1;

/// The longest header line accepted, in bytes.
const MAX_HEADER: u64 = 64 * 1024;

/// How long a client gets to send its header. Clients send it right after
/// connecting, so one that stays silent is not a tessar client; without a
/// limit it would hold a connection open for good and keep an idle daemon
/// from exiting.
const HEADER_WAIT: Duration = Duration::from_secs(5);

/// How long connections get to wind down after a shutdown signal.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// The socket file name inside the home directory.
pub const SOCKET_FILE: &str = "tessar.sock";

/// The daemon's log file name inside the home directory.
pub const LOG_FILE: &str = "daemon.log";

/// The socket of the daemon whose memory lives in `home`.
pub fn socket_path(home: &Path) -> PathBuf {
    home.join(SOCKET_FILE)
}

/// The first line of a connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    /// The header protocol version, [`PROTOCOL`].
    pub tessar: u32,
    /// The project name, as `tessar mcp` computed it. Empty for a status
    /// request.
    #[serde(default)]
    pub project: String,
    /// The session's working directory, for the log.
    #[serde(default)]
    pub cwd: String,
    /// The client's build id; empty from clients that predate it.
    #[serde(default)]
    pub build: String,
    /// Claude Code's session id, when the client knows it; empty from
    /// clients that predate it or run outside Claude Code. Older daemons
    /// ignore the field.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub session: String,
    /// What the client is: `mcp` (an agent's proxy), `hook` or `cli`; empty
    /// from older proxies. Recorded in the tool log, ignored by older daemons.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub client: String,
    /// Ask for the daemon's state instead of opening a session.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub status: bool,
}

impl Header {
    /// A header for `project` and `cwd`.
    pub fn new(project: &str, cwd: &Path) -> Self {
        Self {
            tessar: PROTOCOL,
            project: project.to_owned(),
            cwd: cwd.display().to_string(),
            build: BUILD_ID.to_owned(),
            session: crate::tool_log::session_from_env().unwrap_or_default(),
            client: crate::tool_log::client_role().to_owned(),
            status: false,
        }
    }

    /// A header asking for the daemon's state.
    pub fn status_request() -> Self {
        Self {
            tessar: PROTOCOL,
            project: String::new(),
            cwd: String::new(),
            build: BUILD_ID.to_owned(),
            session: String::new(),
            client: String::new(),
            status: true,
        }
    }

    /// The header as a line, newline included.
    pub fn line(&self) -> String {
        let mut line = serde_json::to_string(self).expect("a header is plain data");
        line.push('\n');
        line
    }

    /// Parses one header line.
    pub fn parse(line: &str) -> Result<Self, String> {
        let header: Self =
            serde_json::from_str(line.trim()).map_err(|e| format!("bad connection header: {e}"))?;
        if header.tessar != PROTOCOL {
            return Err(format!(
                "unsupported header version {} (this daemon speaks {PROTOCOL})",
                header.tessar
            ));
        }
        if !header.status && header.project.trim().is_empty() {
            return Err("bad connection header: empty project".to_owned());
        }
        Ok(header)
    }
}

/// The daemon's first line on a connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    /// The header protocol version, [`PROTOCOL`].
    pub tessar: u32,
    /// The daemon's build id.
    pub build: String,
    /// The daemon's process id.
    pub pid: u32,
    /// Seconds since the daemon started; status answers only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uptime_secs: Option<u64>,
    /// Open MCP sessions; status answers only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sessions: Option<usize>,
}

impl Hello {
    /// The hello line, newline included.
    pub fn line(&self) -> String {
        let mut line = serde_json::to_string(self).expect("a hello is plain data");
        line.push('\n');
        line
    }

    /// Parses one hello line.
    pub fn parse(line: &str) -> Result<Self, String> {
        let hello: Self =
            serde_json::from_str(line.trim()).map_err(|e| format!("bad daemon hello: {e}"))?;
        if hello.tessar != PROTOCOL {
            return Err(format!(
                "the daemon speaks protocol {}, this binary {PROTOCOL}",
                hello.tessar
            ));
        }
        Ok(hello)
    }
}

/// Who the daemon is, and what it is doing.
pub struct State {
    build: String,
    pid: u32,
    started: std::time::Instant,
    sessions: AtomicUsize,
}

impl State {
    /// The state of a daemon that has just started, with build id `build`.
    pub fn new(build: &str) -> Self {
        Self {
            build: build.to_owned(),
            pid: std::process::id(),
            started: std::time::Instant::now(),
            sessions: AtomicUsize::new(0),
        }
    }

    fn hello(&self) -> Hello {
        Hello {
            tessar: PROTOCOL,
            build: self.build.clone(),
            pid: self.pid,
            uptime_secs: None,
            sessions: None,
        }
    }

    fn status(&self) -> Hello {
        Hello {
            uptime_secs: Some(self.started.elapsed().as_secs()),
            sessions: Some(self.sessions.load(Ordering::Relaxed)),
            ..self.hello()
        }
    }
}

/// Counts an open MCP session while alive.
struct SessionGuard<'a>(&'a AtomicUsize);

impl<'a> SessionGuard<'a> {
    fn new(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self(counter)
    }
}

impl Drop for SessionGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The memory shared by all connections, and the per-project [`Tessar`]
/// views over it, created on first use.
pub struct Projects {
    mem: Memory,
    clock: Arc<dyn Clock>,
    cache: Mutex<HashMap<String, Arc<Tessar>>>,
    /// The home whose decision log gets the tool calls; none logs nothing.
    log_home: Option<PathBuf>,
}

impl Projects {
    /// Projects over `mem`.
    pub fn new(mem: Memory, clock: Arc<dyn Clock>) -> Self {
        Self {
            mem,
            clock,
            cache: Mutex::new(HashMap::new()),
            log_home: None,
        }
    }

    /// Logs the tool calls of every connection to the decision log in `home`.
    #[must_use]
    pub fn with_decision_log(mut self, home: &Path) -> Self {
        self.log_home = Some(home.to_owned());
        self
    }

    /// The view for `project`. The first call for a name creates its space,
    /// kinds and entity (bootstrapping is idempotent, so a name already in
    /// the journal is only looked up); the lock is held meanwhile, which
    /// serializes first uses. Blocking: embeds on a first use.
    pub fn get(&self, project: &str) -> Result<Arc<Tessar>, StoreError> {
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(api) = cache.get(project) {
            return Ok(Arc::clone(api));
        }
        let api = Arc::new(Tessar::bootstrap(
            self.mem.clone(),
            self.clock.clone(),
            project,
            &|w| logln!("tessar: {w}"),
        )?);
        cache.insert(project.to_owned(), Arc::clone(&api));
        Ok(api)
    }
}

/// Reads the header line, answers with the hello, then serves MCP on the
/// rest of `stream` until the peer closes it or `stop` fires. A status
/// request gets the extended hello and nothing more. A peer that sends no
/// header within [`HEADER_WAIT`], or before `stop` fires, is dropped.
pub async fn serve_connection<S>(
    stream: S,
    projects: Arc<Projects>,
    state: Arc<State>,
    stop: tokio::sync::watch::Receiver<bool>,
    peer: u64,
) -> Result<(), String>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    let (reader, mut writer) = tokio::io::split(stream);
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    let mut stop = stop;
    // `take` bounds a peer that never sends a newline, the timeout one that
    // sends nothing.
    let mut bounded = (&mut reader).take(MAX_HEADER);
    let header = tokio::time::timeout(HEADER_WAIT, bounded.read_line(&mut line));
    let read = tokio::select! {
        read = header => read
            .map_err(|_| format!("no header within {}s", HEADER_WAIT.as_secs()))?
            .map_err(|e| format!("cannot read the header: {e}"))?,
        _ = stop.wait_for(|s| *s) => return Err("shutdown before the header".to_owned()),
    };
    if read == 0 {
        return Err("closed before the header".to_owned());
    }
    let header = Header::parse(&line)?;
    if header.status {
        writer
            .write_all(state.status().line().as_bytes())
            .await
            .map_err(|e| format!("cannot answer the status request: {e}"))?;
        return writer.shutdown().await.map_err(|e| e.to_string());
    }
    writer
        .write_all(state.hello().line().as_bytes())
        .await
        .and(writer.flush().await)
        .map_err(|e| format!("cannot write the hello: {e}"))?;
    logln!(
        "tessar: #{peer} opened: project `{}` in {} (client build {})",
        header.project,
        header.cwd,
        if header.build.is_empty() {
            "unknown"
        } else {
            &header.build
        }
    );
    let _session = SessionGuard::new(&state.sessions);
    let project = header.project.clone();
    let api = {
        let projects = Arc::clone(&projects);
        tokio::task::spawn_blocking(move || projects.get(&project))
            .await
            .map_err(|e| format!("internal error: {e}"))?
            .map_err(|e| format!("cannot prepare project `{}`: {e}", header.project))?
    };
    let mut server = TessarServer::shared(api);
    if let Some(home) = &projects.log_home {
        let session = crate::tool_log::clean_session(&header.session);
        server = server.with_tool_log(ToolLog::new(
            home,
            Some(peer),
            session,
            crate::tool_log::clean_client(&header.client),
        ));
    }
    let service = server
        .serve((reader, writer))
        .await
        .map_err(|e| format!("handshake failed: {e}"))?;
    let token = service.cancellation_token();
    let watcher = tokio::spawn(async move {
        if stop.wait_for(|s| *s).await.is_ok() {
            token.cancel();
        }
    });
    let ended = service.waiting().await;
    watcher.abort();
    ended.map(|_| ()).map_err(|e| e.to_string())
}

/// Accepts connections on `listener` until `shutdown` resolves or, with
/// `idle_exit`, no connection has been open for that long. In-flight calls
/// get a grace period to finish.
pub async fn run<F>(
    listener: UnixListener,
    projects: Arc<Projects>,
    idle_exit: Option<Duration>,
    shutdown: F,
) where
    F: Future<Output = ()>,
{
    run_as(BUILD_ID, listener, projects, idle_exit, shutdown).await;
}

/// [`run`] announcing build id `build`, so tests can play an older daemon.
pub async fn run_as<F>(
    build: &str,
    listener: UnixListener,
    projects: Arc<Projects>,
    idle_exit: Option<Duration>,
    shutdown: F,
) where
    F: Future<Output = ()>,
{
    let state = Arc::new(State::new(build));
    tokio::pin!(shutdown);
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let mut tasks: JoinSet<()> = JoinSet::new();
    let mut idle_since = Instant::now();
    let mut next_id = 0u64;
    loop {
        let idle_at = idle_exit
            .filter(|_| tasks.is_empty())
            .map(|d| idle_since + d);
        // In this order: a connection that is already waiting when the idle
        // time runs out is served, not dropped with the listener.
        tokio::select! {
            biased;
            () = &mut shutdown => {
                logln!("tessar: shutting down");
                break;
            }
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    next_id += 1;
                    let id = next_id;
                    let projects = Arc::clone(&projects);
                    let state = Arc::clone(&state);
                    let stop = stop_rx.clone();
                    tasks.spawn(async move {
                        match serve_connection(stream, projects, state, stop, id).await {
                            Ok(()) => logln!("tessar: #{id} closed"),
                            Err(e) => logln!("tessar: #{id} closed: {e}"),
                        }
                    });
                }
                Err(e) => {
                    logln!("tessar: accept failed: {e}");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            },
            Some(_) = tasks.join_next(), if !tasks.is_empty() => {
                if tasks.is_empty() {
                    idle_since = Instant::now();
                }
            }
            () = async { tokio::time::sleep_until(idle_at.expect("guarded")).await }, if idle_at.is_some() => {
                logln!("tessar: idle, exiting");
                break;
            }
        }
    }
    drop(listener);
    let _ = stop_tx.send(true);
    let drained = tokio::time::timeout(SHUTDOWN_GRACE, async {
        while tasks.join_next().await.is_some() {}
    })
    .await;
    if drained.is_err() {
        logln!("tessar: connections did not close in time; dropping them");
        tasks.abort_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let header = Header::new("tessar", Path::new("/work/tessar"));
        let line = header.line();
        assert!(line.ends_with('\n'));
        assert_eq!(line.matches('\n').count(), 1);
        assert_eq!(Header::parse(&line).unwrap(), header);
    }

    #[test]
    fn header_accepts_the_documented_form() {
        let h = Header::parse(r#"{"tessar": 1, "project": "repo", "cwd": "/x y"}"#).unwrap();
        assert_eq!(h.project, "repo");
        assert_eq!(h.cwd, "/x y");
        // The working directory is optional, unknown fields are ignored.
        let h = Header::parse(r#"{"tessar":1,"project":"repo","extra":true}"#).unwrap();
        assert_eq!(h.cwd, "");
    }

    #[test]
    fn header_rejects_bad_lines() {
        for bad in [
            "",
            "not json",
            r#"{"project": "repo"}"#,
            r#"{"tessar": 2, "project": "repo"}"#,
            r#"{"tessar": 1, "project": "  "}"#,
            r#"{"tessar": 1}"#,
        ] {
            assert!(Header::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn header_carries_the_build_and_the_status_flag() {
        let h = Header::new("repo", Path::new("/x"));
        assert_eq!(h.build, BUILD_ID);
        assert!(!h.line().contains("status"));
        let h = Header::parse(&Header::status_request().line()).unwrap();
        assert!(h.status);
        assert!(h.project.is_empty());
        // Headers from before the build id still parse.
        let h = Header::parse(r#"{"tessar":1,"project":"repo"}"#).unwrap();
        assert_eq!(h.build, "");
    }

    #[test]
    fn hello_round_trip_and_documented_form() {
        let hello = State::new("1.2+abc.99").hello();
        let line = hello.line();
        assert_eq!(
            line,
            format!(
                "{{\"tessar\":1,\"build\":\"1.2+abc.99\",\"pid\":{}}}\n",
                hello.pid
            )
        );
        assert_eq!(Hello::parse(&line).unwrap(), hello);
        let status = State::new("b").status();
        assert_eq!(status.sessions, Some(0));
        assert_eq!(Hello::parse(&status.line()).unwrap(), status);
        assert!(Hello::parse(r#"{"tessar":2,"build":"b","pid":1}"#).is_err());
        assert!(Hello::parse("").is_err());
    }

    #[test]
    fn socket_lives_in_the_home() {
        assert_eq!(
            socket_path(Path::new("/h/.tessar")),
            Path::new("/h/.tessar/tessar.sock")
        );
    }
}
