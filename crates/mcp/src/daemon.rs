//! The daemon: one process per machine holds the memory and the model and
//! serves MCP to many sessions over a Unix socket.
//!
//! A connection starts with one line of JSON, the [`Header`], naming the
//! project the session runs in. The rest of the stream is plain MCP,
//! served by a [`TessarServer`] whose project space is the header's.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use rmcp::ServiceExt;
use serde::{Deserialize, Serialize};
use tessar_core::clock::Clock;
use tessar_core::store::{Memory, StoreError};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, BufReader};
use tokio::net::UnixListener;
use tokio::task::JoinSet;
use tokio::time::Instant;

use crate::api::Tessar;
use crate::server::TessarServer;

/// The protocol version of the header line.
pub const PROTOCOL: u32 = 1;

/// The longest header line accepted, in bytes.
const MAX_HEADER: u64 = 64 * 1024;

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
    /// The project name, as `tessar mcp` computed it.
    pub project: String,
    /// The session's working directory, for the log.
    #[serde(default)]
    pub cwd: String,
}

impl Header {
    /// A header for `project` and `cwd`.
    pub fn new(project: &str, cwd: &Path) -> Self {
        Self {
            tessar: PROTOCOL,
            project: project.to_owned(),
            cwd: cwd.display().to_string(),
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
        if header.project.trim().is_empty() {
            return Err("bad connection header: empty project".to_owned());
        }
        Ok(header)
    }
}

/// The memory shared by all connections, and the per-project [`Tessar`]
/// views over it, created on first use.
pub struct Projects {
    mem: Memory,
    clock: Arc<dyn Clock>,
    cache: Mutex<HashMap<String, Arc<Tessar>>>,
}

impl Projects {
    /// Projects over `mem`.
    pub fn new(mem: Memory, clock: Arc<dyn Clock>) -> Self {
        Self {
            mem,
            clock,
            cache: Mutex::new(HashMap::new()),
        }
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
            &|w| eprintln!("tessar: {w}"),
        )?);
        cache.insert(project.to_owned(), Arc::clone(&api));
        Ok(api)
    }
}

/// Reads the header line, then serves MCP on the rest of `stream` until the
/// peer closes it or `stop` fires. Returns the header's project name.
pub async fn serve_connection<S>(
    stream: S,
    projects: Arc<Projects>,
    stop: tokio::sync::watch::Receiver<bool>,
    peer: u64,
) -> Result<(), String>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    let (reader, writer) = tokio::io::split(stream);
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    // `take` bounds a peer that never sends a newline.
    let read = (&mut reader)
        .take(MAX_HEADER)
        .read_line(&mut line)
        .await
        .map_err(|e| format!("cannot read the header: {e}"))?;
    if read == 0 {
        return Err("closed before the header".to_owned());
    }
    let header = Header::parse(&line)?;
    eprintln!(
        "tessar: #{peer} opened: project `{}` in {}",
        header.project, header.cwd
    );
    let project = header.project.clone();
    let api = {
        let projects = Arc::clone(&projects);
        tokio::task::spawn_blocking(move || projects.get(&project))
            .await
            .map_err(|e| format!("internal error: {e}"))?
            .map_err(|e| format!("cannot prepare project `{}`: {e}", header.project))?
    };
    let service = TessarServer::shared(api)
        .serve((reader, writer))
        .await
        .map_err(|e| format!("handshake failed: {e}"))?;
    let token = service.cancellation_token();
    let mut stop = stop;
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
    tokio::pin!(shutdown);
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let mut tasks: JoinSet<()> = JoinSet::new();
    let mut idle_since = Instant::now();
    let mut next_id = 0u64;
    loop {
        let idle_at = idle_exit
            .filter(|_| tasks.is_empty())
            .map(|d| idle_since + d);
        tokio::select! {
            () = &mut shutdown => {
                eprintln!("tessar: shutting down");
                break;
            }
            () = async { tokio::time::sleep_until(idle_at.expect("guarded")).await }, if idle_at.is_some() => {
                eprintln!("tessar: idle, exiting");
                break;
            }
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    next_id += 1;
                    let id = next_id;
                    let projects = Arc::clone(&projects);
                    let stop = stop_rx.clone();
                    tasks.spawn(async move {
                        match serve_connection(stream, projects, stop, id).await {
                            Ok(()) => eprintln!("tessar: #{id} closed"),
                            Err(e) => eprintln!("tessar: #{id} closed: {e}"),
                        }
                    });
                }
                Err(e) => {
                    eprintln!("tessar: accept failed: {e}");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            },
            Some(_) = tasks.join_next(), if !tasks.is_empty() => {
                if tasks.is_empty() {
                    idle_since = Instant::now();
                }
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
        eprintln!("tessar: connections did not close in time; dropping them");
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
    fn socket_lives_in_the_home() {
        assert_eq!(
            socket_path(Path::new("/h/.tessar")),
            Path::new("/h/.tessar/tessar.sock")
        );
    }
}
