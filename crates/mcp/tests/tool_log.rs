//! Tool calls in the decision log, through a real daemon over a Unix socket
//! in a temp directory, with an in-memory journal and a fake embedder.

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
use tessar_mcp::daemon::{self, Header, Projects, socket_path};
use tessar_mcp::decisions::{self, Record, SessionRecord};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
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

/// Embeds like [`Bag`] until a text holds the word `quokka`, then fails the
/// way a provider that echoes its input does.
struct Echoing;

impl Embedder for Echoing {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        match texts.iter().find(|t| t.contains("quokka")) {
            Some(t) => Err(EmbedError(format!("provider said: \"bad input {t}\""))),
            None => Bag.embed_documents(texts),
        }
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Bag.embed_query(text)
    }
}

fn projects(log_home: Option<&Path>) -> Arc<Projects> {
    projects_with(log_home, Arc::new(Bag))
}

fn projects_with(log_home: Option<&Path>, embedder: Arc<dyn Embedder>) -> Arc<Projects> {
    let clock = Arc::new(SystemClock);
    let mem = Memory::open_with(
        MemLog::new(),
        clock.clone(),
        Semantic {
            embedder,
            dims: DIMS,
            near_bonus: 0.05,
            predicate_threshold: 0.70,
        },
    )
    .unwrap();
    let projects = Projects::new(mem, clock);
    Arc::new(match log_home {
        Some(home) => projects.with_decision_log(home),
        None => projects,
    })
}

struct Daemon {
    stop: oneshot::Sender<()>,
    task: JoinHandle<()>,
}

impl Daemon {
    /// A daemon in `home` that logs tool calls into `log_home`, if given.
    fn start(home: &Path, log_home: Option<&Path>) -> Self {
        Self::start_with(home, projects(log_home))
    }

    fn start_with(home: &Path, projects: Arc<Projects>) -> Self {
        let listener = UnixListener::bind(socket_path(home)).unwrap();
        let (stop, rx) = oneshot::channel::<()>();
        let task = tokio::spawn(daemon::run(listener, projects, None, async {
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

/// An MCP client straight on the daemon's socket, with a header written by
/// hand: `session` is what Claude Code's id would be, if any.
async fn session(home: &Path, project: &str, session: Option<&str>) -> Client {
    session_as(home, project, session, "").await
}

/// [`session`] announcing itself as kind `client` (empty: an older proxy).
async fn session_as(home: &Path, project: &str, session: Option<&str>, client: &str) -> Client {
    let mut header = Header::new(project, Path::new("/work/x"));
    header.session = session.unwrap_or_default().to_owned();
    header.client = client.to_owned();
    let mut stream = UnixStream::connect(socket_path(home)).await.unwrap();
    stream.write_all(header.line().as_bytes()).await.unwrap();
    // The hello is one line; read it byte by byte so that no MCP traffic is
    // consumed with it.
    let mut hello = Vec::new();
    loop {
        let byte = stream.read_u8().await.unwrap();
        if byte == b'\n' {
            break;
        }
        hello.push(byte);
    }
    assert!(String::from_utf8(hello).unwrap().contains("\"pid\""));
    ().serve(stream).await.unwrap()
}

async fn call(client: &Client, tool: &'static str, args: Value) -> CallToolResult {
    let mut params = CallToolRequestParams::new(tool);
    params.arguments = args.as_object().cloned();
    client.peer().call_tool(params).await.unwrap()
}

/// The lines of the log once there are at least `n` of them. The daemon
/// writes off the tool's path, so this waits for the blocking pool; the
/// verdict does not depend on how long that takes.
async fn lines(home: &Path, n: usize) -> Vec<String> {
    let path = decisions::log_path(home);
    for _ in 0..2000 {
        if let Ok(text) = std::fs::read_to_string(&path) {
            let lines: Vec<String> = text.lines().map(str::to_owned).collect();
            if lines.len() >= n {
                return lines;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("fewer than {n} lines in the decision log");
}

fn parse(lines: &[String]) -> Vec<Value> {
    lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn only<'a>(records: &'a [Value], tool: &str) -> &'a Value {
    let mut found = records.iter().filter(|r| r["tool"] == tool);
    let first = found.next().unwrap_or_else(|| panic!("no {tool} record"));
    assert!(found.next().is_none(), "two {tool} records");
    first
}

const FACT: &str = "The quokka marmalade protocol uses sqlite";

#[tokio::test]
async fn every_tool_call_is_one_record_without_fact_texts() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), Some(home.path()));
    let client = session(home.path(), "alpha", Some("sess-42")).await;

    let facts = json!({"facts": [{"kind": "statement", "text": FACT}]});
    let first = call(&client, "remember", facts.clone()).await;
    let second = call(&client, "remember", facts).await;
    assert_ne!(first.is_error, Some(true));
    let recalled = call(
        &client,
        "recall",
        json!({"query": "protocol", "space": "project", "limit": 5}),
    )
    .await;
    let hits = recalled.structured_content.unwrap()["hits"].clone();
    let id = hits[0]["id"].as_str().unwrap().to_owned();
    call(
        &client,
        "about",
        json!({"entity": {"key": "alpha", "kind": "project"}, "history": true}),
    )
    .await;
    let updated = call(
        &client,
        "update",
        json!({"fact": id, "reason": "changed", "text": "The quokka marmalade protocol uses postgres"}),
    )
    .await;
    assert_ne!(updated.is_error, Some(true));
    call(&client, "vocabulary", json!({})).await;
    call(
        &client,
        "lesson",
        json!({"text": "Before touching the zebrafish cache, read its notes"}),
    )
    .await;
    // Two failures: an empty query and a handle that is none.
    let bad_recall = call(&client, "recall", json!({"query": "  "})).await;
    assert_eq!(bad_recall.is_error, Some(true));
    call(&client, "update", json!({"fact": "zz9", "reason": "wrong"})).await;
    drop(second);

    let all = lines(home.path(), 9).await;
    assert_eq!(all.len(), 9, "one record per call");
    for line in &all {
        for secret in ["quokka", "postgres", "zebrafish"] {
            assert!(!line.contains(secret), "{line}");
        }
    }
    let records = parse(&all);
    for r in &records {
        assert_eq!(r["kind"], "tool");
        assert_eq!(r["project"], "alpha");
        assert_eq!(r["conn"], 1);
        assert_eq!(r["session"], "sess-42");
        assert!(r["ts"].as_str().unwrap().ends_with('Z'));
        assert!(r["elapsed_ms"].is_u64());
    }

    let writes: Vec<&Value> = records.iter().filter(|r| r["tool"] == "remember").collect();
    assert_eq!(writes.len(), 2);
    for w in &writes {
        assert_eq!(w["failed"], false);
        assert_eq!(w["space"], "project");
        assert_eq!(w["fact_count"], 1);
    }
    let mut ids: Vec<&str> = writes
        .iter()
        .map(|w| w["facts"][0]["id"].as_str().unwrap())
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, ["f0", "f1"]);
    assert!(writes.iter().all(|w| w["facts"][0]["outcome"] == "created"));

    let recall = records
        .iter()
        .find(|r| r["tool"] == "recall" && r["failed"] == false)
        .unwrap();
    assert_eq!(recall["query"], "protocol");
    assert_eq!(recall["space"], "project");
    assert_eq!(recall["limit"], 5);
    assert_eq!(recall["hits"][0]["id"], id.as_str());
    assert!(recall["hits"][0]["channels"].is_array());
    assert!(recall["hits"][0].get("similarity").is_some());
    assert!(recall["hits"][0].get("text").is_none());

    let about = only(&records, "about");
    assert_eq!(about["entity"], json!({"key": "alpha", "kind": "project"}));
    assert_eq!(about["history"], true);
    assert!(about["handles"].as_array().unwrap().contains(&json!(id)));

    let update = records
        .iter()
        .find(|r| r["tool"] == "update" && r["failed"] == false)
        .unwrap();
    assert_eq!(update["fact"], id.as_str());
    assert_eq!(update["mode"], "changed");
    assert_eq!(update["outcome"], "superseded");
    assert!(update["new_handle"].as_str().unwrap().starts_with('f'));
    assert_ne!(update["new_handle"], id.as_str());

    let vocabulary = only(&records, "vocabulary");
    assert_eq!(vocabulary["failed"], false);
    assert!(vocabulary.get("hits").is_none());

    let lesson = only(&records, "lesson");
    assert_eq!(lesson["fact_count"], 1);
    assert_eq!(lesson["facts"][0]["outcome"], "created");
    assert!(lesson["facts"][0]["id"].as_str().unwrap().starts_with('f'));

    let failed_recall = records
        .iter()
        .find(|r| r["tool"] == "recall" && r["failed"] == true)
        .unwrap();
    assert!(
        failed_recall["error"]
            .as_str()
            .unwrap()
            .contains("`query` is empty")
    );
    let failed_update = records
        .iter()
        .find(|r| r["tool"] == "update" && r["failed"] == true)
        .unwrap();
    // Not a handle: not logged, nor is the text echoed in the error.
    assert!(failed_update.get("fact").is_none());
    assert_eq!(failed_update["mode"], "wrong");
    assert!(!failed_update["error"].as_str().unwrap().contains("zz9"));
    assert!(failed_update["error"].as_str().unwrap().chars().count() <= 300);

    drop(client);
    daemon.shutdown().await;
}

#[tokio::test]
async fn a_client_without_a_session_id_is_recorded_without_one() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), Some(home.path()));
    let first = session(home.path(), "alpha", None).await;
    let second = session(home.path(), "beta", Some("sess-2")).await;
    call(&first, "vocabulary", json!({})).await;
    call(&second, "vocabulary", json!({})).await;
    let records = parse(&lines(home.path(), 2).await);
    let a = records.iter().find(|r| r["project"] == "alpha").unwrap();
    let b = records.iter().find(|r| r["project"] == "beta").unwrap();
    assert!(a.get("session").is_none(), "{a}");
    assert_eq!(a["conn"], 1);
    assert_eq!(b["session"], "sess-2");
    assert_eq!(b["conn"], 2);
    drop((first, second));
    daemon.shutdown().await;
}

#[tokio::test]
async fn long_queries_are_cut_and_bad_arguments_are_not_echoed() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), Some(home.path()));
    let client = session(home.path(), "alpha", None).await;
    call(&client, "recall", json!({"query": "ж".repeat(5000)})).await;
    call(
        &client,
        "update",
        json!({"fact": "ы".repeat(900), "reason": "wrong"}),
    )
    .await;
    let records = parse(&lines(home.path(), 2).await);
    let recall = only(&records, "recall");
    assert_eq!(recall["query"].as_str().unwrap().chars().count(), 2000);
    let update = only(&records, "update");
    assert_eq!(update["failed"], true);
    assert!(update.get("fact").is_none(), "{update}");
    assert!(!update["error"].as_str().unwrap().contains('ы'), "{update}");
    assert!(update["error"].as_str().unwrap().chars().count() <= 300);
    drop(client);
    daemon.shutdown().await;
}

/// The same calls, in order; every result in full.
async fn script(home: &Path, between: impl Fn()) -> Vec<Value> {
    let client = session(home, "alpha", Some("s")).await;
    let facts = json!({"facts": [{"kind": "statement", "text": FACT}]});
    let mut out = Vec::new();
    for (tool, args) in [
        ("remember", facts.clone()),
        ("recall", json!({"query": "protocol"})),
        (
            "about",
            json!({"entity": {"key": "alpha", "kind": "project"}}),
        ),
        ("update", json!({"fact": "f0", "reason": "wrong"})),
        ("remember", facts),
        ("recall", json!({"query": " "})),
        ("vocabulary", json!({})),
    ] {
        out.push(serde_json::to_value(call(&client, tool, args).await).unwrap());
        between();
    }
    drop(client);
    out
}

#[tokio::test]
async fn results_do_not_depend_on_the_log() {
    // No logging at all.
    let plain_home = tempfile::tempdir().unwrap();
    let plain_daemon = Daemon::start(plain_home.path(), None);
    let plain = script(plain_home.path(), || {}).await;
    plain_daemon.shutdown().await;
    assert!(!decisions::log_path(plain_home.path()).exists());

    // Logging on.
    let on_home = tempfile::tempdir().unwrap();
    let on_daemon = Daemon::start(on_home.path(), Some(on_home.path()));
    let on = script(on_home.path(), || {}).await;
    on_daemon.shutdown().await;
    assert_eq!(on, plain);
    assert_eq!(lines(on_home.path(), 7).await.len(), 7);

    // The log path cannot be written: a directory sits where the file goes,
    // and the log home is missing altogether in a second daemon.
    let blocked_home = tempfile::tempdir().unwrap();
    std::fs::create_dir(decisions::log_path(blocked_home.path())).unwrap();
    let blocked_daemon = Daemon::start(blocked_home.path(), Some(blocked_home.path()));
    let blocked = script(blocked_home.path(), || {}).await;
    blocked_daemon.shutdown().await;
    assert_eq!(blocked, plain);

    let missing_home = tempfile::tempdir().unwrap();
    let missing = missing_home.path().join("nowhere");
    let missing_daemon = Daemon::start(missing_home.path(), Some(&missing));
    let gone = script(missing_home.path(), || {}).await;
    missing_daemon.shutdown().await;
    assert_eq!(gone, plain);
    assert!(!missing.exists());
}

#[tokio::test]
async fn a_running_daemon_follows_the_file_after_purge_removes_it() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), Some(home.path()));
    let client = session(home.path(), "alpha", None).await;
    call(&client, "vocabulary", json!({})).await;
    assert_eq!(lines(home.path(), 1).await.len(), 1);

    std::fs::remove_file(decisions::log_path(home.path())).unwrap();
    call(&client, "recall", json!({"query": "anything"})).await;
    let records = parse(&lines(home.path(), 1).await);
    assert_eq!(records.len(), 1, "the new file holds only the new call");
    assert_eq!(records[0]["tool"], "recall");

    drop(client);
    daemon.shutdown().await;
}

#[tokio::test]
async fn hook_records_around_tool_records_still_parse() {
    let home = tempfile::tempdir().unwrap();
    let session_record = |source: &str| {
        Record::Session(SessionRecord {
            ts: decisions::now(),
            session: "sess-1".to_owned(),
            project: Some("alpha".to_owned()),
            source: source.to_owned(),
            elapsed_ms: 3,
            error: None,
            blocks: Vec::new(),
            printed_chars: 0,
            lessons_project: 0,
            lessons_user: 0,
        })
    };
    decisions::append(home.path(), &session_record("startup"));
    let daemon = Daemon::start(home.path(), Some(home.path()));
    let client = session(home.path(), "alpha", Some("sess-1")).await;
    call(&client, "vocabulary", json!({})).await;
    lines(home.path(), 2).await;
    decisions::append(home.path(), &session_record("resume"));

    let records = parse(&lines(home.path(), 3).await);
    let kinds: Vec<&str> = records
        .iter()
        .map(|r| r["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["session", "tool", "session"]);
    assert_eq!(records[0]["source"], "startup");
    assert_eq!(records[2]["session"], "sess-1");
    assert_eq!(records[1]["session"], records[2]["session"]);
    drop(client);
    daemon.shutdown().await;
}

#[tokio::test]
async fn deserialisation_errors_do_not_echo_fact_text() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), Some(home.path()));
    let client = session(home.path(), "alpha", None).await;
    // Facts as a JSON string, as models do.
    let as_string = json!({"facts": format!("[{{\"kind\":\"statement\",\"text\":\"{FACT}\"}}]")});
    let result = call(&client, "remember", as_string).await;
    assert_eq!(result.is_error, Some(true));
    call(
        &client,
        "update",
        json!({"fact": "f0", "reason": format!("because {FACT}")}),
    )
    .await;
    call(
        &client,
        "recall",
        json!({"query": "x", "as_of": FACT, "space": FACT}),
    )
    .await;
    let all = lines(home.path(), 3).await;
    for line in &all {
        assert!(!line.contains("quokka"), "{line}");
        assert!(!line.contains("marmalade"), "{line}");
    }
    let records = parse(&all);
    let update = only(&records, "update");
    assert!(update.get("mode").is_none());
    assert_eq!(
        update["reason_chars"],
        format!("because {FACT}").chars().count()
    );
    let remember = only(&records, "remember");
    assert_eq!(remember["failed"], true);
    assert!(remember["error"].as_str().unwrap().contains("invalid type"));
    drop(client);
    daemon.shutdown().await;
}

#[tokio::test]
async fn the_kind_of_client_is_recorded_when_it_is_known() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), Some(home.path()));
    for (project, client) in [
        ("a", "mcp"),
        ("b", "hook"),
        ("c", "cli"),
        ("d", ""),
        ("e", "bogus"),
    ] {
        let c = session_as(home.path(), project, None, client).await;
        call(&c, "vocabulary", json!({})).await;
        drop(c);
    }
    let records = parse(&lines(home.path(), 5).await);
    let client_of = |p: &str| {
        records
            .iter()
            .find(|r| r["project"] == p)
            .unwrap()
            .get("client")
            .cloned()
    };
    assert_eq!(client_of("a"), Some(json!("mcp")));
    assert_eq!(client_of("b"), Some(json!("hook")));
    assert_eq!(client_of("c"), Some(json!("cli")));
    assert_eq!(client_of("d"), None);
    assert_eq!(client_of("e"), None);
    daemon.shutdown().await;
}

#[tokio::test]
async fn echoed_values_with_quotes_and_newlines_do_not_reach_the_log() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start(home.path(), Some(home.path()));
    let client = session(home.path(), "alpha", None).await;
    let nasty = "x` lives in \"Berlin\"\nsecret` tail";
    let day = call(&client, "recall", json!({"query": "q", "as_of": nasty})).await;
    assert_eq!(day.is_error, Some(true));
    let synonym = call(
        &client,
        "remember",
        json!({"facts": [{"kind": "attribute", "subject": {"key": "user", "kind": "person"},
            "predicate": "fine_name", "value": "v", "synonym_of": nasty}]}),
    )
    .await;
    assert_eq!(synonym.is_error, Some(true));
    let predicate = call(
        &client,
        "remember",
        json!({"facts": [{"kind": "attribute", "subject": {"key": "user", "kind": "person"},
            "predicate": nasty, "value": "v"}]}),
    )
    .await;
    assert_eq!(predicate.is_error, Some(true));
    let lines = lines(home.path(), 3).await;
    for line in &lines {
        for leaked in ["Berlin", "secret", "tail", "lives in"] {
            assert!(!line.contains(leaked), "{line}");
        }
    }
    drop(client);
    daemon.shutdown().await;
}

#[tokio::test]
async fn an_embedder_error_that_echoes_the_text_does_not_reach_the_log() {
    let home = tempfile::tempdir().unwrap();
    let daemon = Daemon::start_with(
        home.path(),
        projects_with(Some(home.path()), Arc::new(Echoing)),
    );
    let client = session(home.path(), "alpha", None).await;
    let result = call(
        &client,
        "remember",
        json!({"facts": [{"kind": "statement", "text": FACT}]}),
    )
    .await;
    assert_eq!(result.is_error, Some(true));
    // The agent still sees what went wrong.
    assert!(
        text(&result).contains("the embedder failed"),
        "{}",
        text(&result)
    );
    let all = lines(home.path(), 1).await;
    assert!(!all[0].contains("quokka"), "{}", all[0]);
    let record: Value = serde_json::from_str(&all[0]).unwrap();
    assert_eq!(record["failed"], true);
    assert!(
        record["error"]
            .as_str()
            .unwrap()
            .contains("the embedder failed")
    );
    drop(client);
    daemon.shutdown().await;
}

fn text(result: &CallToolResult) -> String {
    serde_json::to_value(result).unwrap()["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}
