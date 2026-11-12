//! Spawns the built binary (`tessar mcp --standalone`) and talks MCP to it
//! over stdio.
//!
//! Needs the real model (`~/.tessar/models/voyage-4-nano-Q8_0.gguf`, or
//! `TESSAR_MODEL`) and uses the GPU, so it is ignored by default:
//!
//! ```sh
//! cargo test -p tessar-mcp --test stdio -- --ignored --nocapture
//! ```

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{Value, json};

#[test]
#[ignore = "loads the real model onto the GPU"]
fn stdout_carries_only_json_rpc() {
    let home = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_tessar"))
        .args(["mcp", "--standalone"])
        .env("TESSAR_HOME", home.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    let stdout = child.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let mut lines: Vec<String> = Vec::new();
    let send = |stdin: &mut std::process::ChildStdin, v: Value| {
        writeln!(stdin, "{v}").unwrap();
        stdin.flush().unwrap();
    };
    // Loading the model happens before the server starts reading.
    let await_id = |id: u64, lines: &mut Vec<String>| -> Value {
        loop {
            let line = rx
                .recv_timeout(Duration::from_secs(120))
                .expect("no response from the server");
            lines.push(line.clone());
            let v: Value = serde_json::from_str(&line).expect("stdout line is not JSON");
            if v["id"] == json!(id) {
                return v;
            }
        }
    };

    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}}),
    );
    let init = await_id(1, &mut lines);
    assert_eq!(init["result"]["serverInfo"]["name"], "tessar");
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    );

    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    );
    let tools = await_id(2, &mut lines);
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for want in [
        "remember",
        "recall",
        "vocabulary",
        "about",
        "update",
        "lesson",
    ] {
        assert!(names.contains(&want), "{names:?}");
    }
    let remember = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "remember")
        .unwrap();
    assert!(
        remember["description"]
            .as_str()
            .unwrap()
            .contains("lives_in (single)"),
        "the vocabulary is part of the description"
    );

    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "remember", "arguments": {"facts": [
                {"kind": "statement", "text": "The journal is a JSON Lines file with a checksum per line"}]}}}),
    );
    let remembered = await_id(3, &mut lines);
    assert_eq!(remembered["result"]["isError"], false, "{remembered}");
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "recall", "arguments": {"query": "how is the journal stored"}}}),
    );
    let recalled = await_id(4, &mut lines);
    let text = recalled["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("JSON Lines file"), "{text}");

    drop(stdin);
    let status = child.wait().unwrap();
    assert!(status.success(), "{status}");
    while let Ok(line) = rx.try_recv() {
        lines.push(line);
    }
    for line in &lines {
        let v: Value = serde_json::from_str(line).unwrap_or_else(|_| panic!("not JSON: {line:?}"));
        assert_eq!(v["jsonrpc"], "2.0", "{line}");
    }
}
