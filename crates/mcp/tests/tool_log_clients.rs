//! The kind of client a tool record names, with the real binary on the
//! client side: a hook and an MCP proxy against an in-process daemon.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tessar_core::clock::SystemClock;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::log::MemLog;
use tessar_core::store::{Memory, Semantic};
use tessar_mcp::daemon::{self, Projects, socket_path};
use tessar_mcp::decisions;
use tokio::net::UnixListener;
use tokio::sync::oneshot;

const DIMS: usize = 64;

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

fn projects(home: &Path) -> Arc<Projects> {
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
    Arc::new(Projects::new(mem, clock).with_decision_log(home))
}

/// The tool records once `ready` accepts them; the daemon writes off the
/// tools' path, so this waits for it.
async fn tool_records(home: &Path, ready: impl Fn(&[Value]) -> bool) -> Vec<Value> {
    for _ in 0..3000 {
        if let Ok(text) = std::fs::read_to_string(decisions::log_path(home)) {
            let records: Vec<Value> = text
                .lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .filter(|r: &Value| r["kind"] == "tool")
                .collect();
            if ready(&records) {
                return records;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the tool records never became what the test waits for");
}

fn run_hook(home: &Path, cwd: &Path) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tessar"))
        .args(["hook", "session", "--home"])
        .arg(home)
        .env("HOME", home)
        .env_remove("TESSAR_HOME")
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let input = json!({"session_id": "s", "cwd": cwd, "source": "startup"});
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.to_string().as_bytes())
        .unwrap();
    assert!(child.wait_with_output().unwrap().status.success());
}

fn run_proxy(home: &Path, cwd: &Path) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tessar"))
        .args(["mcp", "--home"])
        .arg(home)
        .env("HOME", home)
        .env_remove("TESSAR_HOME")
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut send = |v: Value| {
        writeln!(stdin, "{v}").unwrap();
        stdin.flush().unwrap();
    };
    send(
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"}}}),
    );
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    send(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "vocabulary", "arguments": {}}}));
    loop {
        line.clear();
        assert!(stdout.read_line(&mut line).unwrap() > 0, "proxy closed");
        let v: Value = serde_json::from_str(&line).unwrap();
        if v["id"] == json!(2) {
            assert_eq!(v["result"]["isError"], false, "{v}");
            break;
        }
    }
    drop(send);
    drop(stdin);
    assert!(child.wait().unwrap().success());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hook_calls_are_marked_hook_and_the_proxy_is_marked_mcp() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let listener = UnixListener::bind(socket_path(home.path())).unwrap();
    let (stop, rx) = oneshot::channel::<()>();
    let daemon = tokio::spawn(daemon::run(listener, projects(home.path()), None, async {
        let _ = rx.await;
    }));

    let (h, c) = (home.path().to_owned(), cwd.path().to_owned());
    tokio::task::spawn_blocking(move || run_hook(&h, &c))
        .await
        .unwrap();
    let records = tool_records(home.path(), |r| r.iter().any(|x| x["tool"] == "about")).await;
    assert!(records.iter().all(|r| r["client"] == "hook"), "{records:?}");
    assert!(records.iter().any(|r| r["tool"] == "about"));

    let (h, c) = (home.path().to_owned(), cwd.path().to_owned());
    tokio::task::spawn_blocking(move || run_proxy(&h, &c))
        .await
        .unwrap();
    let records = tool_records(home.path(), |r| r.iter().any(|x| x["tool"] == "vocabulary")).await;
    let vocabulary: Vec<&Value> = records
        .iter()
        .filter(|r| r["tool"] == "vocabulary")
        .collect();
    assert_eq!(vocabulary.len(), 1, "{records:?}");
    assert_eq!(vocabulary[0]["client"], "mcp");

    let _ = stop.send(());
    tokio::time::timeout(Duration::from_secs(15), daemon)
        .await
        .unwrap()
        .unwrap();
}
