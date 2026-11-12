//! `tessar daemon` and two concurrent `tessar mcp` proxies on a temporary
//! home, with the real model.
//!
//! Needs the real model (`~/.tessar/models/voyage-4-nano-Q8_0.gguf`, or
//! `TESSAR_MODEL`) and uses the GPU, so it is ignored by default:
//!
//! ```sh
//! cargo test -p tessar --test daemon_e2e -- --ignored --nocapture
//! ```

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct Session {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Session {
    /// Starts `tessar mcp` in `cwd`; the first call may start the daemon.
    fn start(home: &Path, cwd: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_tessar"))
            .arg("mcp")
            .env("TESSAR_HOME", home)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Self {
            child,
            stdin,
            lines,
        }
    }

    fn send(&mut self, v: Value) {
        writeln!(self.stdin, "{v}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn request(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        loop {
            let line = self
                .lines
                .recv_timeout(Duration::from_secs(180))
                .expect("no response from the proxy");
            let v: Value = serde_json::from_str(&line).expect("stdout line is not JSON");
            if v["id"] == json!(id) {
                return v;
            }
        }
    }

    fn initialize(&mut self) {
        let init = self.request(
            1,
            "initialize",
            json!({"protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"}}),
        );
        assert_eq!(init["result"]["serverInfo"]["name"], "tessar", "{init}");
        assert!(
            init["result"]["instructions"]
                .as_str()
                .unwrap()
                .contains("long-term memory")
        );
        self.send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    }

    fn call(&mut self, id: u64, tool: &str, args: Value) -> String {
        let r = self.request(id, "tools/call", json!({"name": tool, "arguments": args}));
        assert_eq!(r["result"]["isError"], false, "{r}");
        r["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_owned()
    }

    fn finish(mut self) {
        drop(self.stdin);
        assert!(self.child.wait().unwrap().success());
    }
}

/// A directory that is its own project: the name of the directory.
fn project_dir(root: &Path, name: &str) -> std::path::PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
#[ignore = "loads the real model onto the GPU"]
fn two_proxies_share_one_daemon() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let socket = home.path().join("tessar.sock");

    // The daemon, started by hand. Both proxies then find it.
    let mut daemon = Command::new(env!("CARGO_BIN_EXE_tessar"))
        .args(["daemon", "--home"])
        .arg(home.path())
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(180);
    while !socket.exists() {
        assert!(Instant::now() < deadline, "the daemon never listened");
        assert!(daemon.try_wait().unwrap().is_none(), "the daemon exited");
        std::thread::sleep(Duration::from_millis(200));
    }

    // A second daemon on the same home steps aside.
    let second = Command::new(env!("CARGO_BIN_EXE_tessar"))
        .args(["daemon", "--home"])
        .arg(home.path())
        .output()
        .unwrap();
    assert!(second.status.success());
    assert!(
        String::from_utf8_lossy(&second.stderr).contains("already running"),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );

    let mut a = Session::start(home.path(), &project_dir(work.path(), "alpha"));
    let mut b = Session::start(home.path(), &project_dir(work.path(), "beta"));
    // Both are connected at once.
    a.initialize();
    b.initialize();

    let tools = a.request(2, "tools/list", json!({}));
    assert!(tools["result"]["tools"].as_array().unwrap().len() >= 5);

    a.call(
        3,
        "remember",
        json!({"facts": [{"kind": "statement",
            "text": "Alpha stores its journal as a JSON Lines file with a checksum per line"}]}),
    );
    // The other session, in another project, recalls it from the shared memory.
    let seen = b.call(
        3,
        "recall",
        json!({"query": "how does alpha store its journal"}),
    );
    assert!(seen.contains("JSON Lines file"), "{seen}");
    assert!(seen.contains("alpha"), "{seen}");

    a.finish();
    b.finish();
    // The daemon outlives its sessions and stops on SIGTERM, removing the socket.
    assert!(daemon.try_wait().unwrap().is_none());
    assert!(
        Command::new("kill")
            .arg(daemon.id().to_string())
            .status()
            .unwrap()
            .success()
    );
    assert!(daemon.wait().unwrap().success());
    assert!(!socket.exists(), "the socket file was left behind");
}

#[test]
#[ignore = "loads the real model onto the GPU"]
fn the_first_proxy_starts_the_daemon() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let mut a = Session::start(home.path(), &project_dir(work.path(), "alpha"));
    a.initialize();
    a.call(
        2,
        "remember",
        json!({"facts": [{"kind": "statement", "text": "The daemon was started by the proxy"}]}),
    );
    a.finish();
    assert!(home.path().join("daemon.log").is_file());
    assert!(home.path().join("tessar.sock").exists());
    // Stop it: the pid is not known here, so ask through the socket owner.
    let out = Command::new("pkill")
        .args(["-TERM", "-f"])
        .arg(format!("daemon --home {}", home.path().display()))
        .status()
        .unwrap();
    assert!(out.success());
}
