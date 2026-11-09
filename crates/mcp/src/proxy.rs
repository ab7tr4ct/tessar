//! The stdio side of the daemon: `tessar mcp` serves one session on stdio
//! and forwards every tool call to the daemon through an MCP client.

use std::borrow::Cow;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, PoisonError, RwLock};
use std::time::{Duration, Instant};

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, ErrorData, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerConfig,
};
use rmcp::service::{RequestContext, RunningService};
use rmcp::{Peer, RoleClient, RoleServer, ServerHandler, ServiceError, ServiceExt};
use tokio::io::{
    AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, ReadHalf,
    WriteHalf,
};
use tokio::net::UnixStream;

use crate::BUILD_ID;
use crate::daemon::{Header, Hello, LOG_FILE, socket_path};

const DAEMON_POLL: Duration = Duration::from_millis(200);

/// How long a proxy that lost its daemon only tries to connect before it
/// starts a daemon itself. A lost daemon is most often one being replaced
/// by a newer build, and a daemon started from this proxy's binary in that
/// moment could take the journal ahead of the replacement and undo the
/// upgrade. The dangerous time is from the old daemon releasing the journal
/// to the new one holding it: nothing when the replacement was started
/// first and waits in line ([`Start::Takeover`]), and otherwise the poll of
/// whoever retired the daemon (100 ms) plus the start of a process, a few
/// tenths of a second in all. Three seconds cover that with a margin, and
/// are what a session waits longer when its daemon simply died.
pub const RECONNECT_GRACE: Duration = Duration::from_secs(3);

/// How often a reconnecting proxy looks for a daemon during the grace time.
const GRACE_POLL: Duration = Duration::from_millis(100);

/// After a daemon this process started stepped aside for another one (it
/// exited cleanly), how long to keep looking for that other one before
/// starting a daemon again. The other one may have been on its way out.
#[doc(hidden)]
pub const RESPAWN_PAUSE: Duration = Duration::from_secs(1);

/// How long a daemon gets to send its hello. It sends it before doing any
/// work, so a daemon that stays silent predates the handshake.
const HELLO_WAIT: Duration = Duration::from_secs(5);

/// The longest hello line accepted, in bytes.
const MAX_HELLO: u64 = 4096;

/// A server whose tools are the daemon's: lists and calls are forwarded
/// verbatim, and `initialize` reports what the daemon reported.
#[derive(Clone)]
pub struct ProxyServer {
    /// The current daemon connection; replaced when the proxy reconnects.
    daemon: Arc<RwLock<Peer<RoleClient>>>,
    info: ServerConfig,
}

impl ProxyServer {
    /// The daemon's current peer.
    pub fn daemon(&self) -> Peer<RoleClient> {
        self.daemon
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn replace_peer(&self, peer: Peer<RoleClient>) {
        *self.daemon.write().unwrap_or_else(PoisonError::into_inner) = peer;
    }
}

/// Opens a new daemon connection for a proxy whose old one is gone.
pub type Reconnect = Arc<
    dyn Fn() -> Pin<Box<dyn Future<Output = Result<(Peer<RoleClient>, DaemonLink), String>> + Send>>
        + Send
        + Sync,
>;

/// How long to wait between attempts to reach a lost daemon.
const RECONNECT_PAUSE: Duration = Duration::from_secs(2);

/// The longest wait between attempts after repeated failures.
const RECONNECT_CAP: Duration = Duration::from_secs(30);

/// The pauses of a proxy that reconnects to a lost daemon.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backoff {
    /// The first pause. A connection that lived shorter than this counts as
    /// dropped by the daemon, and is not replaced at once.
    pub pause: Duration,
    /// The longest pause; each failure doubles the pause up to this.
    pub cap: Duration,
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            pause: RECONNECT_PAUSE,
            cap: RECONNECT_CAP,
        }
    }
}

/// The daemon connection a [`ProxyServer`] forwards to. Dropping it closes
/// the connection.
pub type DaemonLink = RunningService<RoleClient, ()>;

/// Turns a failure of the forwarding path into the JSON-RPC error for the
/// one request that hit it, and says so on stderr. `what` names the request.
fn forwarded(what: &str, e: ServiceError) -> ErrorData {
    match e {
        // The daemon's own answer: pass it on as it is.
        ServiceError::McpError(e) => e,
        other => {
            logln!("tessar: forwarding {what} to the daemon failed: {other}");
            ErrorData::internal_error(
                Cow::Owned(format!("tessar daemon: {what} failed: {other}")),
                None,
            )
        }
    }
}

impl ServerHandler for ProxyServer {
    fn get_info(&self) -> ServerConfig {
        self.info.clone()
    }

    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        self.daemon()
            .list_tools(request)
            .await
            .map_err(|e| forwarded("tools/list", e))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let what = format!("tools/call `{}`", request.name);
        self.daemon()
            .call_tool_once(request)
            .await
            .map_err(|e| forwarded(&what, e))
    }
}

/// A connection whose header is sent and whose hello is read, before any
/// MCP traffic: the point where the build ids can be compared.
pub struct Opened<S> {
    reader: BufReader<ReadHalf<S>>,
    writer: WriteHalf<S>,
    hello: Hello,
}

impl<S> Opened<S>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    /// What the daemon said about itself.
    pub fn hello(&self) -> &Hello {
        &self.hello
    }

    /// Runs the MCP handshake and returns the handler to serve the session
    /// with, plus the live link.
    pub async fn serve(self) -> Result<(ProxyServer, DaemonLink), String> {
        let link = ()
            .serve((self.reader, self.writer))
            .await
            .map_err(|e| format!("handshake with the daemon failed: {e}"))?;
        let peer = link
            .peer()
            .peer_info()
            .ok_or("the daemon sent no server info")?;
        let mut info = ServerConfig::new(peer.capabilities.clone());
        info.server_info = peer
            .server_info
            .clone()
            .unwrap_or_else(|| Implementation::new("tessar", env!("CARGO_PKG_VERSION")));
        info.instructions.clone_from(&peer.instructions);
        let server = ProxyServer {
            daemon: Arc::new(RwLock::new(link.peer().clone())),
            info,
        };
        Ok((server, link))
    }
}

/// Sends `header` on `stream` and reads the daemon's hello.
async fn open_with<S>(stream: S, header: &Header) -> Result<Opened<S>, String>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    let (reader, mut writer) = tokio::io::split(stream);
    writer
        .write_all(header.line().as_bytes())
        .await
        .and(writer.flush().await)
        .map_err(|e| format!("cannot write the header: {e}"))?;
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    let read = tokio::time::timeout(
        HELLO_WAIT,
        (&mut reader).take(MAX_HELLO).read_line(&mut line),
    )
    .await
    .map_err(|_| {
        "the daemon sent no hello; it predates the build handshake, stop it once with \
         `pkill -f 'tessar daemon'`"
            .to_owned()
    })?
    .map_err(|e| format!("cannot read the daemon's hello: {e}"))?;
    if read == 0 {
        return Err("the daemon closed the connection before its hello".to_owned());
    }
    let hello = Hello::parse(&line)?;
    Ok(Opened {
        reader,
        writer,
        hello,
    })
}

/// Sends the session header on `stream` and reads the hello.
pub async fn open<S>(stream: S, project: &str, cwd: &Path) -> Result<Opened<S>, String>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    open_with(stream, &Header::new(project, cwd)).await
}

/// Sends the header on `stream`, reads the hello, runs the MCP handshake
/// with the daemon and returns the handler to serve the session with, plus
/// the live link. Takes the daemon as it is, whatever its build.
pub async fn connect<S>(
    stream: S,
    project: &str,
    cwd: &Path,
) -> Result<(ProxyServer, DaemonLink), String>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    open(stream, project, cwd).await?.serve().await
}

/// Asks the daemon on `stream` for its state.
pub async fn status<S>(stream: S) -> Result<Hello, String>
where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    Ok(open_with(stream, &Header::status_request()).await?.hello)
}

/// The build time in a build id `<version>+<git>.<unix time>`.
fn build_stamp(id: &str) -> Option<u64> {
    id.rsplit_once('.')?.1.parse().ok()
}

/// Whether a daemon of build `daemon` is at least as new as `ours`, by build
/// time (builds in the same second count as equal). False when the ids
/// cannot be ordered.
pub fn daemon_is_newer(ours: &str, daemon: &str) -> bool {
    matches!((build_stamp(ours), build_stamp(daemon)), (Some(o), Some(d)) if d >= o)
}

/// Whether a proxy should retire the daemon: only when the builds differ,
/// the daemon is not newer than this binary (a newer one was started by
/// another session's upgrade; restarting it would make two proxies fight
/// over it) and the user did not ask to keep it. A daemon that reports no
/// build counts as older.
pub fn needs_restart(ours: &str, daemon: &str, keep_daemon: bool) -> bool {
    !keep_daemon && ours != daemon && !daemon_is_newer(ours, daemon)
}

/// Like [`connect_or_start`] followed by [`open`], and for a proxy: if the
/// daemon found is another build than this binary, retires it and talks to
/// a fresh one. With `keep_daemon` it takes the daemon as it is.
pub async fn open_current(
    home: &Path,
    project: &str,
    cwd: &Path,
    wait: Duration,
    keep_daemon: bool,
) -> Result<Opened<UnixStream>, String> {
    open_current_with(home, project, cwd, keep_daemon, &Launcher::new(wait)).await
}

/// [`open_current`] with the way to reach and start the daemon spelled out.
///
/// An older daemon is not stopped from here. Its replacement is started
/// with the old daemon's pid ([`Start::Takeover`]) and stops it itself, once
/// it waits in line for the journal: it has the journal the moment the old
/// daemon lets go. Other sessions' proxies lose their connection at that
/// point and may start daemons of their own build; those find the journal
/// taken and step aside. This proxy only waits for the new daemon.
pub async fn open_current_with(
    home: &Path,
    project: &str,
    cwd: &Path,
    keep_daemon: bool,
    launcher: &Launcher,
) -> Result<Opened<UnixStream>, String> {
    let stream = connect_or_start_with(home, launcher).await?;
    let opened = open(stream, project, cwd).await?;
    let theirs = opened.hello().build.clone();
    if !needs_restart(BUILD_ID, &theirs, keep_daemon) {
        if !keep_daemon && daemon_is_newer(BUILD_ID, &theirs) {
            logln!(
                "tessar: the daemon is build {theirs}, this proxy is {BUILD_ID}; \
                 serving through the newer daemon"
            );
        }
        return Ok(opened);
    }
    logln!(
        "tessar: the daemon is build {theirs}, this is {BUILD_ID}; restarting it \
         (other Claude Code sessions will show tessar as disconnected until reconnected)"
    );
    let pid = opened.hello().pid;
    drop(opened);
    let replacement = (launcher.spawn)(home, Start::Takeover(pid))?;
    let stream = await_daemon(home, launcher, replacement, Some(pid)).await?;
    let opened = open(stream, project, cwd).await?;
    if opened.hello().build != BUILD_ID {
        logln!(
            "tessar: the new daemon is build {}, not {BUILD_ID}; using it anyway",
            opened.hello().build
        );
    }
    Ok(opened)
}

/// Calls the daemon's `update` tool with `reason: "redact"` for `handle`
/// and returns the text it answers: the success line, or the error text
/// the tool gave.
pub async fn redact_fact(peer: &Peer<RoleClient>, handle: &str) -> Result<String, String> {
    let mut params = rmcp::model::CallToolRequestParams::new("update");
    params.arguments = serde_json::json!({"fact": handle, "reason": "redact"})
        .as_object()
        .cloned();
    let result = peer
        .call_tool(params)
        .await
        .map_err(|e| format!("{handle}: the daemon failed the call: {e}"))?;
    let text = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    if result.is_error == Some(true) {
        Err(format!("{handle}: {text}"))
    } else {
        Ok(text)
    }
}

/// Serves `server` on `transport` until the session ends or the daemon
/// goes away. `Ok(true)` means the session ended normally.
pub async fn bridge<T, E, A>(
    server: ProxyServer,
    link: DaemonLink,
    transport: T,
) -> Result<bool, String>
where
    T: rmcp::transport::IntoTransport<RoleServer, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    bridge_with(server, link, transport, None).await
}

/// Like [`bridge`], but when the daemon goes away it keeps the session open,
/// answers calls with errors meanwhile and reconnects through `reconnect`.
/// It returns only when the session ends.
pub async fn bridge_reconnecting<T, E, A>(
    server: ProxyServer,
    link: DaemonLink,
    transport: T,
    reconnect: Reconnect,
) -> Result<bool, String>
where
    T: rmcp::transport::IntoTransport<RoleServer, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    bridge_reconnecting_with(server, link, transport, reconnect, Backoff::default()).await
}

/// [`bridge_reconnecting`] with its pauses spelled out.
#[doc(hidden)]
pub async fn bridge_reconnecting_with<T, E, A>(
    server: ProxyServer,
    link: DaemonLink,
    transport: T,
    reconnect: Reconnect,
    backoff: Backoff,
) -> Result<bool, String>
where
    T: rmcp::transport::IntoTransport<RoleServer, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    bridge_with(server, link, transport, Some((reconnect, backoff))).await
}

async fn bridge_with<T, E, A>(
    server: ProxyServer,
    link: DaemonLink,
    transport: T,
    reconnect: Option<(Reconnect, Backoff)>,
) -> Result<bool, String>
where
    T: rmcp::transport::IntoTransport<RoleServer, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    let handle = server.clone();
    let session = server
        .serve(transport)
        .await
        .map_err(|e| format!("cannot serve the session: {e}"))?;
    let session = session.waiting();
    tokio::pin!(session);
    let mut watcher = tokio::spawn(link.waiting());
    let mut linked_at = Instant::now();
    let limits = reconnect.as_ref().map(|(_, b)| *b).unwrap_or_default();
    let mut backoff = limits.pause;
    loop {
        tokio::select! {
            _ = &mut session => {
                watcher.abort();
                return Ok(true);
            }
            _ = &mut watcher => {}
        }
        let Some((reconnect, _)) = &reconnect else {
            return Ok(false);
        };
        logln!("tessar: the daemon connection was lost; reconnecting");
        // A link that lived a while is a fresh loss: try again at once. One
        // that died young means the daemon drops sessions: back off.
        if linked_at.elapsed() < limits.pause {
            tokio::select! {
                _ = &mut session => return Ok(true),
                () = tokio::time::sleep(backoff) => {}
            }
            backoff = (backoff * 2).min(limits.cap);
        } else {
            backoff = limits.pause;
        }
        let link = loop {
            tokio::select! {
                _ = &mut session => return Ok(true),
                attempt = reconnect() => match attempt {
                    Ok((peer, link)) => {
                        handle.replace_peer(peer);
                        logln!("tessar: reconnected to the daemon");
                        break link;
                    }
                    Err(e) => logln!("tessar: cannot reconnect to the daemon: {e}"),
                },
            }
            tokio::select! {
                _ = &mut session => return Ok(true),
                () = tokio::time::sleep(backoff) => {}
            }
            backoff = (backoff * 2).min(limits.cap);
        };
        linked_at = Instant::now();
        watcher = tokio::spawn(link.waiting());
    }
}

/// How a daemon is asked to start.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// Take the journal, or step aside (exit cleanly) if another process
    /// holds it.
    Fresh,
    /// Replace the daemon with this process id: get in line for the
    /// journal, tell that daemon to stop, and wait up to
    /// [`crate::lifecycle::TAKEOVER_WAIT`] for it to let go. `tessar daemon
    /// --takeover PID`; see [`crate::lifecycle::take_over`].
    Takeover(u32),
}

/// Starts a daemon process for a home directory.
#[doc(hidden)]
pub type Spawn = Arc<dyn Fn(&Path, Start) -> Result<Child, String> + Send + Sync>;

/// A [`Spawn`] that runs this binary as `tessar daemon`, in its own process
/// group, so it outlives this process and the terminal's signals, with its
/// output appended to the daemon log in the home directory.
pub(crate) fn own_daemon() -> Spawn {
    Arc::new(|home, start| spawn_daemon(home, &home.join(LOG_FILE), start))
}

/// How a client reaches the daemon and starts one.
#[derive(Clone)]
pub struct Launcher {
    /// How long a started daemon gets to listen (the model loads first).
    #[doc(hidden)]
    pub wait: Duration,
    /// How long to only look for a daemon before starting one. Zero for a
    /// first connection, [`RECONNECT_GRACE`] for a proxy that lost its
    /// daemon.
    #[doc(hidden)]
    pub grace: Duration,
    /// See [`RESPAWN_PAUSE`].
    #[doc(hidden)]
    pub respawn_pause: Duration,
    /// Starts the daemon.
    #[doc(hidden)]
    pub spawn: Spawn,
}

impl Launcher {
    /// For a first connection: starts this binary as the daemon at once if
    /// none listens, and gives it `wait` to do so.
    pub fn new(wait: Duration) -> Self {
        Self {
            wait,
            grace: Duration::ZERO,
            respawn_pause: RESPAWN_PAUSE,
            spawn: own_daemon(),
        }
    }

    /// For a proxy that had a daemon and lost it: looks for a new one for
    /// [`RECONNECT_GRACE`] before starting one.
    #[must_use]
    pub fn reconnecting(mut self) -> Self {
        self.grace = RECONNECT_GRACE;
        self
    }
}

/// Connects to the daemon's socket, starting a daemon first if none answers
/// and waiting up to `wait` for it to listen (the model loads first).
pub async fn connect_or_start(home: &Path, wait: Duration) -> Result<UnixStream, String> {
    connect_or_start_with(home, &Launcher::new(wait)).await
}

/// [`connect_or_start`] with the way to reach and start the daemon spelled
/// out: nothing is started while a daemon shows up within the launcher's
/// grace time.
#[doc(hidden)]
pub async fn connect_or_start_with(home: &Path, launcher: &Launcher) -> Result<UnixStream, String> {
    let socket = socket_path(home);
    let started = Instant::now();
    loop {
        if let Ok(stream) = UnixStream::connect(&socket).await {
            return Ok(stream);
        }
        let left = launcher.grace.saturating_sub(started.elapsed());
        if left.is_zero() {
            break;
        }
        tokio::time::sleep(GRACE_POLL.min(left)).await;
    }
    let child = (launcher.spawn)(home, Start::Fresh)?;
    logln!(
        "tessar: starting the daemon (log: {})",
        home.join(LOG_FILE).display()
    );
    await_daemon(home, launcher, child, None).await
}

/// A daemon this process started. When the handle is let go while the
/// daemon runs, a thread waits for it, so that the daemon does not stay a
/// zombie for as long as this process lives once it has ended.
struct Started(Option<Child>);

impl Started {
    /// The exit status, if the daemon has ended.
    fn ended(&mut self) -> Option<std::process::ExitStatus> {
        self.0.as_mut()?.try_wait().ok().flatten()
    }
}

impl Drop for Started {
    fn drop(&mut self) {
        let Some(mut child) = self.0.take() else {
            return;
        };
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        // Without the thread the daemon is only not reaped; nothing breaks.
        let _ = std::thread::Builder::new()
            .name("daemon-reaper".to_owned())
            .spawn(move || {
                let _ = child.wait();
            });
    }
}

/// Waits up to the launcher's `wait` for a daemon to listen, `child` being
/// the one just started. With `old`, the daemon wanted is the replacement
/// for the one with that pid, which may still listen for a moment: only a
/// daemon that says it is another one counts, not one that says it is `old`
/// and not one that says nothing.
///
/// A daemon that failed (no model, say) will never listen: that is an
/// error. One that exited cleanly found another process holding the
/// journal. That is usually a daemon about to listen, but it may also be
/// one that was shutting down, so if still nothing listens a pause later, a
/// daemon is started again. A replacement that exited cleanly while the old
/// daemon still serves could not make it stop: an error too.
async fn await_daemon(
    home: &Path,
    launcher: &Launcher,
    child: Child,
    old: Option<u32>,
) -> Result<UnixStream, String> {
    let socket = socket_path(home);
    let log = home.join(LOG_FILE);
    let mut child = Started(Some(child));
    let started = Instant::now();
    // When the daemon last started was seen to have stepped aside.
    let mut stepped_aside: Option<Instant> = None;
    loop {
        tokio::time::sleep(DAEMON_POLL.min(launcher.wait)).await;
        let mut old_serves = false;
        if let Ok(stream) = UnixStream::connect(&socket).await {
            let Some(old) = old else {
                return Ok(stream);
            };
            // Only a daemon that says it is another one than the old is the
            // replacement. One that says nothing may be the old daemon,
            // hung or busy: keep waiting, for no longer than is left.
            let left = launcher.wait.saturating_sub(started.elapsed());
            let said = tokio::time::timeout(left.max(DAEMON_POLL), status(stream)).await;
            match said {
                Ok(Ok(hello)) if hello.pid == old => old_serves = true,
                Ok(Ok(_)) => {
                    if let Ok(stream) = UnixStream::connect(&socket).await {
                        return Ok(stream);
                    }
                }
                Ok(Err(_)) | Err(_) => {}
            }
        }
        if stepped_aside.is_none()
            && let Some(status) = child.ended()
        {
            if !status.success() {
                return Err(format!(
                    "the daemon exited with {status}; see {}",
                    log.display()
                ));
            }
            stepped_aside = Some(Instant::now());
        }
        if let (true, Some(_), Some(old)) = (old_serves, stepped_aside, old) {
            return Err(format!(
                "the old daemon (pid {old}) did not make way for a new one; see {}",
                log.display()
            ));
        }
        if started.elapsed() > launcher.wait {
            return Err(format!(
                "no daemon answered on {} within {}s; see {}",
                socket.display(),
                launcher.wait.as_secs(),
                log.display()
            ));
        }
        if stepped_aside.is_some_and(|at| at.elapsed() >= launcher.respawn_pause) {
            child = Started(Some((launcher.spawn)(home, Start::Fresh)?));
            stepped_aside = None;
        }
    }
}

/// Starts `tessar daemon` in its own process group, so it outlives this
/// proxy and the terminal's signals, with its output appended to `log`.
fn spawn_daemon(home: &Path, log: &Path, start: Start) -> Result<Child, String> {
    use std::os::unix::process::CommandExt;
    let exe = std::env::current_exe().map_err(|e| format!("cannot find my own binary: {e}"))?;
    let out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .map_err(|e| format!("cannot open {}: {e}", log.display()))?;
    let err = out
        .try_clone()
        .map_err(|e| format!("cannot open {}: {e}", log.display()))?;
    let mut command = Command::new(exe);
    command.arg("daemon").arg("--home").arg(home);
    if let Start::Takeover(old) = start {
        command.arg("--takeover").arg(old.to_string());
    }
    command
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .spawn()
        .map_err(|e| format!("cannot start the daemon: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_only_when_the_builds_differ() {
        assert!(!needs_restart("a", "a", false));
        assert!(needs_restart("a", "b", false));
        assert!(needs_restart("a", "", false));
        // Ids with build times: only an older daemon is retired.
        let (old, new) = ("0.1.0+aaa.1000", "0.1.0+bbb.2000");
        assert!(needs_restart(new, old, false));
        assert!(!needs_restart(old, new, false));
        assert!(!needs_restart(new, new, false));
        assert!(daemon_is_newer(old, new) && !daemon_is_newer(new, old));
        // Same second, different commits (parallel worktrees): hands off.
        let (a, b) = ("0.1.0+aaa.1500", "0.1.0+bbb.1500");
        assert!(!needs_restart(a, b, false));
        assert!(!needs_restart(b, a, false));
        // No stamp on the daemon's side counts as older.
        assert!(needs_restart(a, "0.1.0+ccc", false));
        // --keep-daemon never restarts.
        assert!(!needs_restart("a", "b", true));
        assert!(!needs_restart("a", "a", true));
    }
}
