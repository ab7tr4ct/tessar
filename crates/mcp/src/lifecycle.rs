//! The daemon's life from outside: its pid file, the journal lock that
//! makes it the only one, retiring it when the binary changed, and `tessar
//! daemon stop|status`.
//!
//! Signals go through the `kill` command: the workspace forbids `unsafe`
//! and has no libc binding, and `kill` is on every macOS and Linux box.
//!
//! # Whom `stop` may signal
//!
//! A pid file outlives a daemon that was killed or lost to a reboot, and
//! the number in it may by then belong to any process. So the pid comes
//! from the daemon itself, in the hello it sends on its socket. The pid
//! file is believed only while something holds the journal lock: the
//! daemon writes the file as soon as it has the lock, and a process that
//! takes the lock for another reason removes a file left behind. With the
//! journal free, the file is stale and is deleted without a signal.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tessar_core::log::LogError;
use tessar_core::store::Op;
use tessar_journal::FileJournal;
use tokio::net::UnixStream;

use crate::daemon::{Header, Hello, socket_path};
use crate::proxy;
use crate::purge::journal_path;

/// The pid file name inside the home directory.
pub const PID_FILE: &str = "tessar.pid";

/// How long a retiring daemon gets to go away. The daemon itself allows its
/// connections ten seconds to wind down.
pub const RETIRE_WAIT: Duration = Duration::from_secs(15);

const POLL: Duration = Duration::from_millis(100);

/// How long a daemon started to replace another one waits for the journal:
/// longer than a retiring daemon may take ([`RETIRE_WAIT`]), and bounded,
/// so a replacement whose predecessor never goes does not wait for ever.
pub const TAKEOVER_WAIT: Duration = Duration::from_secs(20);

/// How long a replacement daemon lets its lock thread run before it tells
/// the old daemon to stop. The thread has reported that it is running; what
/// is left is opening the file and entering the lock call, microseconds of
/// work. The pause is there for a loaded machine that takes the thread off
/// the processor in between, and it is all an upgrade pays for the order.
const LOCK_HEAD_START: Duration = Duration::from_millis(50);

/// How often a replacement daemon that waits for the journal looks whether
/// another daemon serves already.
const TAKEOVER_SLICE: Duration = Duration::from_millis(200);

/// How long a daemon gets to answer when asked who it is.
const WHO_WAIT: Duration = Duration::from_secs(1);

/// After a daemon is gone, how long `stop` waits for the journal lock to be
/// released: the socket goes a moment before the lock does.
const SETTLE_WAIT: Duration = Duration::from_secs(2);

const SETTLE_POLL: Duration = Duration::from_millis(10);

/// The pid file of the daemon whose memory lives in `home`.
pub fn pid_path(home: &Path) -> PathBuf {
    home.join(PID_FILE)
}

/// Records the current process as the daemon of `home`. The daemon calls
/// this as soon as it holds the journal lock.
pub fn write_pid_file(home: &Path) -> std::io::Result<()> {
    std::fs::write(pid_path(home), format!("{}\n", std::process::id()))
}

/// Removes the pid file, if it still names this process.
pub fn remove_pid_file(home: &Path) {
    let path = pid_path(home);
    if read_pid_file(home) == Some(std::process::id()) {
        let _ = std::fs::remove_file(path);
    }
}

/// The pid in the pid file, if there is a sane one.
pub fn read_pid_file(home: &Path) -> Option<u32> {
    parse_pid(&std::fs::read_to_string(pid_path(home)).ok()?)
}

/// A pid that is safe to signal: not 0 (our process group), not 1 (init),
/// and positive as a `pid_t`.
pub fn parse_pid(text: &str) -> Option<u32> {
    text.trim()
        .parse::<u32>()
        .ok()
        .filter(|p| (2..=i32::MAX as u32).contains(p))
}

fn kill(signal: &str, pid: u32) -> bool {
    Command::new("kill")
        .arg(signal)
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Asks process `pid` to terminate.
fn terminate(pid: u32) -> Result<(), String> {
    if parse_pid(&pid.to_string()).is_none() || pid == std::process::id() {
        return Err(format!("refusing to signal pid {pid}"));
    }
    if kill("-TERM", pid) {
        Ok(())
    } else {
        Err(format!("cannot signal the daemon (pid {pid})"))
    }
}

/// [`terminate`], where a process that went away by itself meanwhile is as
/// good as one that was told to.
fn terminate_or_gone(pid: u32) -> Result<(), String> {
    let signalable = parse_pid(&pid.to_string()).is_some() && pid != std::process::id();
    match terminate(pid) {
        Err(e) if !signalable || kill("-0", pid) => Err(e),
        _ => Ok(()),
    }
}

/// Whether the daemon of `home` is gone: its socket file is removed (it
/// does that when it has stopped serving, just before it releases the
/// journal; older builds just after) or its process no longer exists. The
/// socket matters because a daemon this process started stays a zombie,
/// which still "exists", until this process ends.
fn gone(home: &Path, pid: u32) -> bool {
    !socket_path(home).exists() || !kill("-0", pid)
}

/// Waits up to `wait` for the daemon to go away.
async fn wait_gone(home: &Path, pid: u32, wait: Duration) -> bool {
    let started = Instant::now();
    loop {
        if gone(home, pid) {
            return true;
        }
        if started.elapsed() > wait {
            return false;
        }
        tokio::time::sleep(POLL).await;
    }
}

/// Stops the daemon with process id `pid` and waits for it to go.
pub async fn retire(home: &Path, pid: u32) -> Result<(), String> {
    terminate_or_gone(pid)?;
    if wait_gone(home, pid, RETIRE_WAIT).await {
        Ok(())
    } else {
        Err(format!(
            "the old daemon (pid {pid}) did not stop within {}s",
            RETIRE_WAIT.as_secs()
        ))
    }
}

/// What became of a daemon started to replace another one.
pub enum Takeover {
    /// It holds the journal and is the daemon now.
    Journal(FileJournal<Op>),
    /// It is not needed, for the reason given: it exits cleanly.
    Aside(String),
}

/// How `tessar daemon --takeover PID` gets the journal: it replaces the
/// daemon with process id `old`, which holds it.
///
/// The order is what matters. Other sessions' proxies lose their daemon the
/// moment it stops and may start daemons of their own, older build; the
/// replacement must have the journal before any of them. So it first gets
/// in line for the lock, on a thread of its own that blocks in the lock
/// call, and only then sends the old daemon SIGTERM itself: when the old
/// daemon lets go, the lock passes to the one already waiting. A process
/// `old` that is gone already is fine, the lock is then free or soon will
/// be.
///
/// The wait ends without the journal ([`Takeover::Aside`]) after `wait`, or
/// as soon as a daemon other than `old` answers on the socket: two sessions
/// that upgrade together both start a replacement, and the one that lost
/// must not stay in line, where it would take the journal from whatever
/// needs it next (a purge, say). Its lock thread stays blocked; the daemon
/// exits then, which ends it.
pub fn take_over(home: &Path, old: u32, wait: Duration) -> Result<Takeover, String> {
    let path = journal_path(home);
    let failed = |e: LogError| format!("cannot open the journal: {e}");
    match FileJournal::<Op>::open(&path) {
        Ok(journal) => return Ok(Takeover::Journal(journal)),
        Err(LogError::Locked) => {}
        Err(e) => return Err(failed(e)),
    }
    let (running, is_running) = std::sync::mpsc::channel();
    let (opened, journal) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("journal-lock".to_owned())
        .spawn(move || {
            let _ = running.send(());
            let _ = opened.send(FileJournal::<Op>::open_blocking(path));
        })
        .map_err(|e| format!("cannot start a thread: {e}"))?;
    let _ = is_running.recv();
    std::thread::sleep(LOCK_HEAD_START);
    let mut early = journal.try_recv().ok();
    if early.is_none() {
        terminate_or_gone(old)?;
    }
    let started = Instant::now();
    loop {
        let got = match early.take() {
            Some(got) => got,
            None => match journal.recv_timeout(TAKEOVER_SLICE.min(wait)) {
                Ok(got) => got,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("the thread waiting for the journal died".to_owned());
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if let Some(pid) = serving_pid(home).filter(|pid| *pid != old) {
                        return Ok(Takeover::Aside(format!(
                            "another daemon (pid {pid}) serves already"
                        )));
                    }
                    if started.elapsed() >= wait {
                        return Ok(Takeover::Aside(format!(
                            "the daemon it was to replace (pid {old}) kept the journal for {}s",
                            wait.as_secs()
                        )));
                    }
                    continue;
                }
            },
        };
        return got.map(Takeover::Journal).map_err(failed);
    }
}

/// The pid of the daemon that answers on the socket of `home`, if one
/// does. Blocking, for the daemon before it has a runtime.
fn serving_pid(home: &Path) -> Option<u32> {
    use std::io::{BufRead, BufReader, Read, Write};
    let mut stream = std::os::unix::net::UnixStream::connect(socket_path(home)).ok()?;
    stream.set_read_timeout(Some(WHO_WAIT)).ok()?;
    stream.set_write_timeout(Some(WHO_WAIT)).ok()?;
    stream
        .write_all(Header::status_request().line().as_bytes())
        .ok()?;
    let mut line = String::new();
    BufReader::new(stream)
        .take(4096)
        .read_line(&mut line)
        .ok()?;
    Hello::parse(&line).ok().map(|hello| hello.pid)
}

/// Whether some process holds the journal of `home`: a daemon, also one
/// that is still loading its model and does not listen yet, a `tessar mcp
/// --standalone` or a purge. False without a journal. See
/// [`FileJournal::is_locked`] for what the check costs others.
pub fn journal_in_use(home: &Path) -> bool {
    FileJournal::<Op>::is_locked(journal_path(home)).unwrap_or(false)
}

/// Waits up to `wait` for the journal of `home` to be free.
async fn wait_journal_free(home: &Path, wait: Duration) {
    let started = Instant::now();
    while journal_in_use(home) && started.elapsed() < wait {
        tokio::time::sleep(SETTLE_POLL).await;
    }
}

/// `tessar daemon stop`: returns the message to print.
///
/// The daemon is the process that answers on the socket and names its pid
/// there. If none answers, the pid file is used, but only while something
/// holds the journal lock (a daemon that is loading its model, or one too
/// old to say hello); otherwise the file is stale and is removed.
pub async fn stop(home: &Path) -> Result<String, String> {
    if let Ok(stream) = UnixStream::connect(socket_path(home)).await {
        match proxy::status(stream).await {
            Ok(hello) => return stop_pid(home, hello.pid).await,
            Err(e) => logln!("tessar: the process on the socket did not say who it is: {e}"),
        }
    }
    let on_file = pid_path(home).exists();
    let pid = read_pid_file(home);
    if !journal_in_use(home) {
        if !on_file {
            return Ok("no daemon is running (no pid file)".to_owned());
        }
        let _ = std::fs::remove_file(pid_path(home));
        return Ok(match pid {
            Some(pid) => format!(
                "no daemon is running (nothing holds the journal; removed the stale pid file \
                 naming pid {pid})"
            ),
            None => "no daemon is running (removed the stale pid file)".to_owned(),
        });
    }
    match pid.filter(|pid| kill("-0", *pid)) {
        Some(pid) => stop_pid(home, pid).await,
        None => Err(format!(
            "{} is in use, but no daemon answers on {} and the pid file names no running \
             process; if a daemon is starting, try again in a moment (a `tessar mcp \
             --standalone` holds the journal too)",
            journal_path(home).display(),
            socket_path(home).display()
        )),
    }
}

/// Retires the daemon `pid` and waits a moment for the journal to be free,
/// so that what follows a stop (a purge) can open it.
async fn stop_pid(home: &Path, pid: u32) -> Result<String, String> {
    retire(home, pid).await?;
    wait_journal_free(home, SETTLE_WAIT).await;
    Ok(format!("stopped the daemon (pid {pid})"))
}

/// `tessar daemon status`: returns the report, or an error if no daemon
/// answers.
pub async fn status(home: &Path) -> Result<String, String> {
    let socket = socket_path(home);
    let stream = UnixStream::connect(&socket).await.map_err(|_| {
        format!(
            "no daemon is running (nothing listens on {})",
            socket.display()
        )
    })?;
    let hello = proxy::status(stream).await?;
    Ok(render_status(&hello))
}

fn render_status(hello: &Hello) -> String {
    let current = if hello.build == crate::BUILD_ID {
        "same as this binary"
    } else {
        "differs from this binary; the next `tessar mcp` restarts it"
    };
    format!(
        "build:    {} ({current})\npid:      {}\nuptime:   {}\nsessions: {}",
        hello.build,
        hello.pid,
        duration(hello.uptime_secs.unwrap_or(0)),
        hello.sessions.unwrap_or(0)
    )
}

fn duration(secs: u64) -> String {
    let (d, h, m, s) = (secs / 86400, secs / 3600 % 24, secs / 60 % 60, secs % 60);
    match (d, h, m) {
        (0, 0, 0) => format!("{s}s"),
        (0, 0, _) => format!("{m}m {s}s"),
        (0, _, _) => format!("{h}h {m}m"),
        _ => format!("{d}d {h}h"),
    }
}

#[cfg(test)]
mod tests {
    use std::process::Child;

    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixListener;

    use super::*;

    /// A process that is not a daemon but dies on SIGTERM. It is ended and
    /// reaped when the test lets go of it, however the test ends.
    struct Bystander(std::sync::Arc<std::sync::Mutex<Child>>);

    fn bystander() -> Bystander {
        let child = Command::new("sleep").arg("60").spawn().unwrap();
        Bystander(std::sync::Arc::new(std::sync::Mutex::new(child)))
    }

    impl Bystander {
        fn pid(&self) -> u32 {
            self.0.lock().unwrap().id()
        }

        /// The exit status, once it has ended. Reaps it, so that it does
        /// not stay a zombie that still answers `kill -0`.
        fn ended(&self) -> Option<std::process::ExitStatus> {
            self.0.lock().unwrap().try_wait().unwrap()
        }

        /// Runs `then` on a thread as soon as the process has ended and
        /// gives its exit status; `None` if it is still running after 30 s.
        fn when_ended(
            &self,
            then: impl FnOnce() + Send + 'static,
        ) -> std::thread::JoinHandle<Option<std::process::ExitStatus>> {
            let child = std::sync::Arc::clone(&self.0);
            std::thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(30);
                loop {
                    let status = child.lock().unwrap().try_wait().unwrap();
                    if status.is_some() || Instant::now() > deadline {
                        then();
                        return status;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
        }

        fn assert_alive(&self) {
            assert!(
                self.ended().is_none(),
                "a process that is not the daemon was signalled"
            );
        }
    }

    impl Drop for Bystander {
        fn drop(&mut self) {
            if let Ok(mut child) = self.0.lock() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    /// Whether the journal is free, give or take a moment. A lock outlives
    /// its file in this process for an instant when another test starts a
    /// child process just then: until it executes its program, the child
    /// holds a copy of every open file.
    fn journal_becomes_free(home: &Path) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while journal_in_use(home) {
            if Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        true
    }

    fn hold_journal(home: &Path) -> FileJournal<Op> {
        FileJournal::<Op>::open(journal_path(home)).unwrap()
    }

    /// Listens on the socket of `home` and greets as a daemon with `pid`.
    fn fake_daemon(home: &Path, pid: u32) -> tokio::task::JoinHandle<()> {
        let listener = UnixListener::bind(socket_path(home)).unwrap();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let (r, mut w) = tokio::io::split(stream);
                let mut line = String::new();
                let _ = BufReader::new(r).read_line(&mut line).await;
                let hello = Hello {
                    tessar: crate::daemon::PROTOCOL,
                    build: "0.0.1+fake.1".to_owned(),
                    pid,
                    uptime_secs: Some(1),
                    sessions: Some(0),
                };
                let _ = w.write_all(hello.line().as_bytes()).await;
            }
        })
    }

    #[test]
    fn pids_that_must_never_be_signalled_are_refused() {
        for bad in ["", "0", "1", "-1", "abc", "4294967295", "2147483648"] {
            assert_eq!(parse_pid(bad), None, "{bad:?}");
        }
        assert_eq!(parse_pid("4242\n"), Some(4242));
    }

    #[test]
    fn pid_file_round_trip() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(read_pid_file(home.path()), None);
        write_pid_file(home.path()).unwrap();
        assert_eq!(read_pid_file(home.path()), Some(std::process::id()));
        remove_pid_file(home.path());
        assert_eq!(read_pid_file(home.path()), None);
    }

    #[test]
    fn pid_file_of_another_daemon_is_left_alone() {
        let home = tempfile::tempdir().unwrap();
        std::fs::write(pid_path(home.path()), "999999\n").unwrap();
        remove_pid_file(home.path());
        assert_eq!(read_pid_file(home.path()), Some(999_999));
    }

    #[test]
    fn the_journal_is_in_use_only_while_someone_holds_it() {
        let home = tempfile::tempdir().unwrap();
        assert!(!journal_in_use(home.path()), "no journal at all");
        assert!(!journal_path(home.path()).exists(), "the check created it");
        let journal = hold_journal(home.path());
        assert!(journal_in_use(home.path()));
        drop(journal);
        assert!(journal_becomes_free(home.path()));
        // The check left the lock free.
        drop(hold_journal(home.path()));
    }

    fn took(result: Result<Takeover, String>) -> FileJournal<Op> {
        match result {
            Ok(Takeover::Journal(journal)) => journal,
            Ok(Takeover::Aside(why)) => panic!("stepped aside: {why}"),
            Err(e) => panic!("{e}"),
        }
    }

    fn stepped_aside(result: Result<Takeover, String>) -> String {
        match result {
            Ok(Takeover::Aside(why)) => why,
            Ok(Takeover::Journal(_)) => panic!("took a journal that is held"),
            Err(e) => panic!("{e}"),
        }
    }

    /// The replacement tells the old daemon to stop, and has the journal
    /// when that one lets go.
    #[test]
    fn a_takeover_stops_the_old_daemon_and_gets_its_journal() {
        let home = tempfile::tempdir().unwrap();
        let old = bystander();
        let journal = hold_journal(home.path());
        // The "daemon" lets go of the journal when it dies, as one does.
        let released = old.when_ended(move || drop(journal));
        let journal = took(take_over(home.path(), old.pid(), Duration::from_secs(20)));
        let status = released
            .join()
            .unwrap()
            .expect("the old daemon was not stopped");
        assert!(!status.success());
        assert!(journal_in_use(home.path()));
        drop(journal);
    }

    /// A free journal is taken at once, and nobody is signalled for it.
    #[test]
    fn a_takeover_of_a_free_journal_signals_nobody() {
        let home = tempfile::tempdir().unwrap();
        let other = bystander();
        drop(took(take_over(
            home.path(),
            other.pid(),
            Duration::from_secs(20),
        )));
        other.assert_alive();
    }

    /// The old daemon may be gone by the time its replacement starts; the
    /// journal it held is released a moment later.
    #[test]
    fn a_takeover_from_a_daemon_that_is_gone_just_waits_for_the_journal() {
        let home = tempfile::tempdir().unwrap();
        let gone = bystander();
        let pid = gone.pid();
        drop(gone);
        let journal = hold_journal(home.path());
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            drop(journal);
        });
        drop(took(take_over(home.path(), pid, Duration::from_secs(20))));
        release.join().unwrap();
    }

    #[test]
    fn a_takeover_gives_up_when_the_journal_is_not_released() {
        let home = tempfile::tempdir().unwrap();
        let gone = bystander();
        let pid = gone.pid();
        drop(gone);
        let _journal = hold_journal(home.path());
        let started = Instant::now();
        let why = stepped_aside(take_over(home.path(), pid, Duration::from_millis(300)));
        assert!(why.contains("kept the journal"), "{why}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// Two sessions that upgrade together start two replacements. The one
    /// that lost leaves the line as soon as the other serves, long before
    /// its time is up, so it does not take the journal from a later purge.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_takeover_leaves_the_line_when_another_daemon_serves() {
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().to_path_buf();
        let gone = bystander();
        let old = gone.pid();
        drop(gone);
        let _journal = hold_journal(&dir);
        let fake = fake_daemon(&dir, std::process::id());
        let started = Instant::now();
        let waited = tokio::task::spawn_blocking(move || {
            stepped_aside(take_over(&dir, old, Duration::from_secs(60)))
        });
        let why = waited.await.unwrap();
        assert!(why.contains("serves already"), "{why}");
        assert!(started.elapsed() < Duration::from_secs(10));
        fake.abort();
    }

    /// While the daemon on the socket is the old one itself, the
    /// replacement stays in line.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_takeover_stays_in_line_while_only_the_old_daemon_serves() {
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().to_path_buf();
        let old = bystander();
        let old_pid = old.pid();
        let journal = hold_journal(&dir);
        let fake = fake_daemon(&dir, old_pid);
        // The old daemon is slow to let go: several looks at the socket.
        let released = old.when_ended(move || {
            std::thread::sleep(Duration::from_millis(700));
            drop(journal);
        });
        let journal = tokio::task::spawn_blocking(move || {
            took(take_over(&dir, old_pid, Duration::from_secs(20)))
        })
        .await
        .unwrap();
        assert!(released.join().unwrap().is_some());
        drop(journal);
        fake.abort();
    }

    /// The hazard this module guards against: after a daemon was killed, its
    /// pid file names a number that now belongs to some other process.
    #[tokio::test]
    async fn stop_does_not_signal_the_process_in_a_stale_pid_file() {
        let home = tempfile::tempdir().unwrap();
        let child = bystander();
        std::fs::write(pid_path(home.path()), format!("{}\n", child.pid())).unwrap();
        // No journal at all.
        let message = stop(home.path()).await.unwrap();
        assert!(
            message.contains("no daemon") && message.contains("stale"),
            "{message}"
        );
        assert!(!pid_path(home.path()).exists());
        // A journal nobody holds.
        drop(hold_journal(home.path()));
        assert!(journal_becomes_free(home.path()));
        std::fs::write(pid_path(home.path()), format!("{}\n", child.pid())).unwrap();
        let message = stop(home.path()).await.unwrap();
        assert!(
            message.contains("no daemon") && message.contains("stale"),
            "{message}"
        );
        assert!(!pid_path(home.path()).exists());
        child.assert_alive();
    }

    #[tokio::test]
    async fn stop_takes_the_pid_from_the_daemon_not_from_the_pid_file() {
        let home = tempfile::tempdir().unwrap();
        let (daemon, other) = (bystander(), bystander());
        let daemon_pid = daemon.pid();
        std::fs::write(pid_path(home.path()), format!("{}\n", other.pid())).unwrap();
        let journal = hold_journal(home.path());
        let fake = fake_daemon(home.path(), daemon_pid);
        // The journal is let go when the "daemon" dies, as a daemon does.
        let release = daemon.when_ended(move || drop(journal));
        let message = stop(home.path()).await.unwrap();
        assert_eq!(message, format!("stopped the daemon (pid {daemon_pid})"));
        let status = release
            .join()
            .unwrap()
            .expect("the daemon was not signalled");
        assert!(!status.success());
        assert!(
            journal_becomes_free(home.path()),
            "stop returned before the journal was free"
        );
        other.assert_alive();
        fake.abort();
    }

    /// A daemon that holds the journal but does not answer yet (it is
    /// loading the model) is found through its pid file.
    #[tokio::test]
    async fn stop_uses_the_pid_file_while_the_journal_is_held() {
        let home = tempfile::tempdir().unwrap();
        let child = bystander();
        let pid = child.pid();
        std::fs::write(pid_path(home.path()), format!("{pid}\n")).unwrap();
        let journal = hold_journal(home.path());
        let release = child.when_ended(move || drop(journal));
        let message = stop(home.path()).await.unwrap();
        assert_eq!(message, format!("stopped the daemon (pid {pid})"));
        let status = release
            .join()
            .unwrap()
            .expect("the daemon was not signalled");
        assert!(!status.success());
    }

    #[tokio::test]
    async fn stop_without_a_daemon_is_not_an_error() {
        let home = tempfile::tempdir().unwrap();
        assert!(stop(home.path()).await.unwrap().contains("no daemon"));
        std::fs::write(pid_path(home.path()), "999999\n").unwrap();
        assert!(stop(home.path()).await.unwrap().contains("stale"));
        assert_eq!(read_pid_file(home.path()), None);
        // Not a number at all.
        std::fs::write(pid_path(home.path()), "junk\n").unwrap();
        assert!(stop(home.path()).await.unwrap().contains("stale"));
        assert!(!pid_path(home.path()).exists());
    }

    /// The journal is held, but by nothing `stop` can name: it says so and
    /// signals nobody.
    #[tokio::test]
    async fn stop_reports_a_journal_held_by_an_unknown_process() {
        let home = tempfile::tempdir().unwrap();
        let _journal = hold_journal(home.path());
        let err = stop(home.path()).await.unwrap_err();
        assert!(err.contains("is in use"), "{err}");
        std::fs::write(pid_path(home.path()), "999999\n").unwrap();
        let err = stop(home.path()).await.unwrap_err();
        assert!(err.contains("is in use"), "{err}");
    }

    #[tokio::test]
    async fn retiring_a_daemon_that_is_already_gone_succeeds() {
        let home = tempfile::tempdir().unwrap();
        let child = bystander();
        let pid = child.pid();
        drop(child);
        retire(home.path(), pid).await.unwrap();
        // A pid that must never be signalled is still refused.
        assert!(retire(home.path(), 1).await.is_err());
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(duration(5), "5s");
        assert_eq!(duration(125), "2m 5s");
        assert_eq!(duration(3 * 3600 + 120), "3h 2m");
        assert_eq!(duration(2 * 86400 + 5 * 3600), "2d 5h");
    }
}
