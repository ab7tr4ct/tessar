//! The daemon's life from outside: its pid file, retiring it when the
//! binary changed, and `tessar daemon stop|status`.
//!
//! Signals go through the `kill` command: the workspace forbids `unsafe`
//! and has no libc binding, and `kill` is on every macOS and Linux box.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tokio::net::UnixStream;

use crate::daemon::{Hello, socket_path};
use crate::proxy;

/// The pid file name inside the home directory.
pub const PID_FILE: &str = "tessar.pid";

/// How long a retiring daemon gets to go away. The daemon itself allows its
/// connections ten seconds to wind down.
pub const RETIRE_WAIT: Duration = Duration::from_secs(15);

const POLL: Duration = Duration::from_millis(100);

/// The pid file of the daemon whose memory lives in `home`.
pub fn pid_path(home: &Path) -> PathBuf {
    home.join(PID_FILE)
}

/// Records the current process as the daemon of `home`.
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
fn parse_pid(text: &str) -> Option<u32> {
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
    if parse_pid(&pid.to_string()).is_none() {
        return Err(format!("refusing to signal pid {pid}"));
    }
    if kill("-TERM", pid) {
        Ok(())
    } else {
        Err(format!("cannot signal the daemon (pid {pid})"))
    }
}

/// Whether the daemon of `home` is gone: its socket file is removed (the
/// last thing it does, after the journal lock is released) or its process
/// no longer exists.
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
    terminate(pid)?;
    if wait_gone(home, pid, RETIRE_WAIT).await {
        Ok(())
    } else {
        Err(format!(
            "the old daemon (pid {pid}) did not stop within {}s",
            RETIRE_WAIT.as_secs()
        ))
    }
}

/// `tessar daemon stop`: returns the message to print.
pub async fn stop(home: &Path) -> Result<String, String> {
    let Some(pid) = read_pid_file(home) else {
        return Ok("no daemon is running (no pid file)".to_owned());
    };
    if !kill("-0", pid) {
        let _ = std::fs::remove_file(pid_path(home));
        return Ok(format!(
            "no daemon is running (pid {pid} is gone; removed the stale pid file)"
        ));
    }
    retire(home, pid).await?;
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
    use super::*;

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

    #[tokio::test]
    async fn stop_terminates_the_process_in_the_pid_file() {
        let home = tempfile::tempdir().unwrap();
        // Not a daemon, but a process that dies on SIGTERM.
        let mut child = Command::new("sleep").arg("60").spawn().unwrap();
        std::fs::write(pid_path(home.path()), format!("{}\n", child.id())).unwrap();
        let reaper = std::thread::spawn(move || child.wait().unwrap());
        let message = stop(home.path()).await.unwrap();
        assert!(message.starts_with("stopped the daemon"), "{message}");
        assert!(!reaper.join().unwrap().success());
    }

    #[tokio::test]
    async fn stop_without_a_daemon_is_not_an_error() {
        let home = tempfile::tempdir().unwrap();
        assert!(stop(home.path()).await.unwrap().contains("no daemon"));
        std::fs::write(pid_path(home.path()), "999999\n").unwrap();
        assert!(stop(home.path()).await.unwrap().contains("stale"));
        assert_eq!(read_pid_file(home.path()), None);
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(duration(5), "5s");
        assert_eq!(duration(125), "2m 5s");
        assert_eq!(duration(3 * 3600 + 120), "3h 2m");
        assert_eq!(duration(2 * 86400 + 5 * 3600), "2d 5h");
    }
}
