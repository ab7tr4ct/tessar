use std::collections::HashMap;
use std::sync::Mutex;

use chrono::TimeZone;
use serde_json::{Value, json};

use super::*;
use crate::hook::config_snippet;

const EXE: &str = "/opt/tessar/bin/tessar";

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// Serves fixed bytes per url, or an error.
#[derive(Default)]
struct FakeFetcher {
    bodies: HashMap<String, Vec<u8>>,
    no_curl: bool,
    calls: Mutex<Vec<String>>,
}

impl Fetcher for FakeFetcher {
    fn fetch(&self, url: &str, dest: &Path) -> Result<(), FetchError> {
        self.calls.lock().unwrap().push(url.to_owned());
        if self.no_curl {
            return Err(FetchError::NoCurl);
        }
        match self.bodies.get(url) {
            Some(body) => fs::write(dest, body).map_err(|e| FetchError::Failed(e.to_string())),
            None => Err(FetchError::Failed("HTTP 404".to_owned())),
        }
    }
}

impl FakeFetcher {
    fn calls(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

type Script = Box<dyn Fn(&[&str]) -> Result<CmdOutput, RunError> + Send + Sync>;

/// A `claude` that only ever sees `mcp add` and `mcp remove`: `get` and
/// `list` connect to the servers, so asking for them is a failure of the test.
struct FakeRunner {
    script: Script,
    calls: Mutex<Vec<Vec<String>>>,
}

impl FakeRunner {
    fn new(
        script: impl Fn(&[&str]) -> Result<CmdOutput, RunError> + Send + Sync + 'static,
    ) -> Self {
        FakeRunner {
            script: Box::new(script),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Every command works.
    fn working() -> Self {
        Self::new(|_| Ok(ok("")))
    }

    fn calls(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|c| c.join(" "))
            .collect()
    }
}

impl Runner for FakeRunner {
    fn run(&self, program: &str, args: &[&str]) -> Result<CmdOutput, RunError> {
        assert_eq!(program, "claude");
        assert!(
            !matches!(args.get(1), Some(&("get" | "list"))),
            "`claude mcp get` and `list` start the servers: {args:?}"
        );
        self.calls
            .lock()
            .unwrap()
            .push(args.iter().map(|a| (*a).to_owned()).collect());
        (self.script)(args)
    }
}

fn ok(stdout: &str) -> CmdOutput {
    CmdOutput {
        success: true,
        stdout: stdout.to_owned(),
        stderr: String::new(),
    }
}

fn failed(stderr: &str) -> CmdOutput {
    CmdOutput {
        success: false,
        stdout: String::new(),
        stderr: stderr.to_owned(),
    }
}

#[derive(Default)]
struct Answer {
    yes: bool,
    asked: Vec<String>,
    /// Runs while the question is on screen.
    meanwhile: Option<Box<dyn FnMut()>>,
}

impl Prompt for Answer {
    fn confirm(&mut self, question: &str) -> bool {
        self.asked.push(question.to_owned());
        if let Some(f) = self.meanwhile.as_mut() {
            f();
        }
        self.yes
    }
}

struct Fixture {
    dir: tempfile::TempDir,
    plan: Plan,
    fetcher: FakeFetcher,
}

const MODEL: &[u8] = b"model bytes";
const TOKENIZER: &[u8] = b"tokenizer bytes";

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let models = home.join("models");
        let file = |label, url: &str, dest, body: &[u8]| ModelFile {
            label,
            url: url.to_owned(),
            dest,
            sha256: hex(body),
            size: body.len() as u64,
            env: ("TESSAR_TEST", None),
        };
        let plan = Plan {
            home: home.clone(),
            exe: PathBuf::from(EXE),
            settings: dir.path().join("claude").join("settings.json"),
            claude_json: dir.path().join("claude.json"),
            cwd: PathBuf::from("/work/project"),
            user_home: Some(PathBuf::from("/opt/tessar")),
            files: vec![
                file(
                    "model",
                    "https://example.test/m.gguf",
                    models.join("m.gguf"),
                    MODEL,
                ),
                file(
                    "tokenizer",
                    "https://example.test/t.json",
                    models.join("sub").join("tokenizer.json"),
                    TOKENIZER,
                ),
            ],
            now: Utc.with_ymd_and_hms(2026, 11, 12, 10, 30, 5).unwrap(),
        };
        let mut fetcher = FakeFetcher::default();
        fetcher
            .bodies
            .insert("https://example.test/m.gguf".into(), MODEL.to_vec());
        fetcher
            .bodies
            .insert("https://example.test/t.json".into(), TOKENIZER.to_vec());
        Fixture { dir, plan, fetcher }
    }

    fn run(&self, opts: Options, runner: &FakeRunner, yes: bool) -> (Outcome, String, Vec<String>) {
        let mut prompt = Answer {
            yes,
            ..Answer::default()
        };
        let (outcome, out) = self.run_with(opts, runner, &mut prompt);
        (outcome, out, prompt.asked)
    }

    fn run_with(
        &self,
        opts: Options,
        runner: &FakeRunner,
        prompt: &mut Answer,
    ) -> (Outcome, String) {
        let mut out = Vec::new();
        let outcome = Setup::new(&self.plan, opts, &self.fetcher, runner, prompt, &mut out).run();
        (outcome, String::from_utf8(out).unwrap())
    }

    fn settings(&self) -> &Path {
        &self.plan.settings
    }

    fn read_settings(&self) -> String {
        fs::read_to_string(self.settings()).unwrap()
    }

    fn write_settings(&self, text: &str) {
        fs::create_dir_all(self.settings().parent().unwrap()).unwrap();
        fs::write(self.settings(), text).unwrap();
    }

    /// Writes Claude Code's `.claude.json` with the given `tessar` entries.
    fn claude_json(&self, user: Option<Value>, local: Option<Value>) {
        let mut root = json!({"numStartups": 3, "mcpServers": {"other": {"command": "x"}}});
        if let Some(user) = user {
            root["mcpServers"]["tessar"] = user;
        }
        root["projects"] = json!({"/work/project": {"allowedTools": []}});
        if let Some(local) = local {
            root["projects"]["/work/project"]["mcpServers"] = json!({"tessar": local});
        }
        fs::write(&self.plan.claude_json, root.to_string()).unwrap();
    }

    fn backups(&self) -> Vec<PathBuf> {
        let Ok(rd) = fs::read_dir(self.settings().parent().unwrap()) else {
            return Vec::new();
        };
        let mut found: Vec<PathBuf> = rd
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().contains("tessar-backup"))
            .collect();
        found.sort();
        found
    }

    fn leftovers(&self) -> Vec<String> {
        let mut names = Vec::new();
        for dir in [
            self.plan.home.join("models"),
            self.plan.home.join("models/sub"),
            self.settings().parent().unwrap().to_owned(),
        ] {
            if let Ok(rd) = fs::read_dir(dir) {
                for e in rd {
                    let name = e.unwrap().file_name().to_string_lossy().into_owned();
                    if name.contains(".part") || name.contains("tessar-tmp") {
                        names.push(name);
                    }
                }
            }
        }
        names
    }
}

fn stdio(command: &str) -> Value {
    json!({"type": "stdio", "command": command, "args": ["mcp"], "env": {}})
}

fn only(models: bool, mcp: bool, hooks: bool) -> Options {
    Options {
        dry_run: false,
        models,
        mcp,
        hooks,
        allow_unverified: false,
    }
}

fn all() -> Options {
    only(true, true, true)
}

fn mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

// ---- models ----------------------------------------------------------------

#[test]
fn the_pinned_downloads_are_not_the_moving_main() {
    for (url, commit_len) in [(MODEL_URL, 40), (TOKENIZER_URL, 40)] {
        assert!(url.starts_with("https://huggingface.co/"));
        assert!(!url.contains("/main/"), "{url}");
        let commit = url
            .split("/resolve/")
            .nth(1)
            .unwrap()
            .split('/')
            .next()
            .unwrap();
        assert_eq!(commit.len(), commit_len, "{url}");
        assert!(commit.chars().all(|c| c.is_ascii_hexdigit()));
    }
    assert_eq!(MODEL_SHA256.len(), 64);
    assert_eq!(TOKENIZER_SHA256.len(), 64);
}

#[test]
fn models_with_the_right_hash_are_left_alone() {
    let f = Fixture::new();
    for (file, body) in f.plan.files.iter().zip([MODEL, TOKENIZER]) {
        fs::create_dir_all(file.dest.parent().unwrap()).unwrap();
        fs::write(&file.dest, body).unwrap();
    }
    let (outcome, out, asked) = f.run(only(true, false, false), &FakeRunner::working(), false);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(f.fetcher.calls(), 0);
    assert!(asked.is_empty());
    assert_eq!(out.matches("\n  in place").count(), 2, "{out}");
    assert!(out.contains("nothing to change"), "{out}");
}

#[test]
fn absent_models_are_fetched_verified_and_private() {
    let f = Fixture::new();
    let runner = FakeRunner::working();
    let (outcome, out, asked) = f.run(only(true, false, false), &runner, false);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert!(
        asked.is_empty(),
        "downloading into the home needs no question"
    );
    for (file, body) in f.plan.files.iter().zip([MODEL, TOKENIZER]) {
        assert_eq!(fs::read(&file.dest).unwrap(), body);
        assert_eq!(mode(&file.dest), 0o600);
        assert_eq!(mode(file.dest.parent().unwrap()), 0o700);
    }
    assert_eq!(mode(&f.plan.home), 0o700);
    assert!(f.leftovers().is_empty());
    // A line before each download says what and how big.
    assert!(
        out.contains("downloading the model (1 MB) from https://example.test/m.gguf"),
        "{out}"
    );
    assert!(out.contains("next: restart Claude Code"), "{out}");
    // The second run changes nothing and does not fetch again.
    let (outcome, out, _) = f.run(only(true, false, false), &runner, false);
    assert_eq!(outcome, Outcome::Complete);
    assert_eq!(f.fetcher.calls(), 2);
    assert!(out.contains("nothing to change"), "{out}");
}

#[test]
fn a_download_with_a_wrong_hash_is_not_used() {
    let mut f = Fixture::new();
    f.fetcher
        .bodies
        .insert("https://example.test/m.gguf".into(), b"tampered".to_vec());
    let (outcome, out, _) = f.run(only(true, false, false), &FakeRunner::working(), false);
    assert_eq!(outcome, Outcome::Failed, "{out}");
    let model = &f.plan.files[0].dest;
    assert!(!model.exists());
    assert_eq!(
        fs::read(model.with_file_name("m.gguf.unverified")).unwrap(),
        b"tampered"
    );
    assert!(out.contains("--allow-unverified"), "{out}");
    assert!(out.contains(&hex(b"tampered")), "{out}");
    assert!(f.plan.files[1].dest.is_file());
    assert!(f.leftovers().is_empty());
}

#[test]
fn an_existing_file_with_a_wrong_hash_is_not_touched() {
    let f = Fixture::new();
    let model = &f.plan.files[0].dest;
    fs::create_dir_all(model.parent().unwrap()).unwrap();
    fs::write(model, b"mine").unwrap();
    let runner = FakeRunner::working();
    let (outcome, out, _) = f.run(only(true, false, false), &runner, true);
    assert_eq!(outcome, Outcome::Failed);
    assert_eq!(fs::read(model).unwrap(), b"mine");
    assert_eq!(f.fetcher.calls(), 1, "only the tokenizer is fetched");
    assert!(out.contains("--allow-unverified"), "{out}");
    assert!(
        !out.contains("not used"),
        "the file is what tessar loads: {out}"
    );
    let opts = Options {
        allow_unverified: true,
        ..only(true, false, false)
    };
    let (outcome, out, _) = f.run(opts, &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(fs::read(model).unwrap(), b"mine");
    assert!(out.contains("without verification"), "{out}");
}

#[test]
fn allow_unverified_accepts_what_was_downloaded_and_says_so() {
    let mut f = Fixture::new();
    f.fetcher
        .bodies
        .insert("https://example.test/m.gguf".into(), b"other".to_vec());
    let opts = Options {
        allow_unverified: true,
        ..only(true, false, false)
    };
    let (outcome, out, _) = f.run(opts, &FakeRunner::working(), false);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(fs::read(&f.plan.files[0].dest).unwrap(), b"other");
    assert!(out.contains("without verification"), "{out}");
    assert!(out.contains(&hex(b"other")), "{out}");
}

#[test]
fn without_curl_the_urls_and_destinations_are_printed() {
    let mut f = Fixture::new();
    f.fetcher.no_curl = true;
    let (outcome, out, _) = f.run(only(true, false, false), &FakeRunner::working(), false);
    assert_eq!(outcome, Outcome::Failed);
    for file in &f.plan.files {
        assert!(out.contains(&file.url), "{out}");
        assert!(out.contains(&file.dest.display().to_string()), "{out}");
    }
    assert!(!f.plan.files[0].dest.exists());
    assert!(f.leftovers().is_empty());
}

#[test]
fn a_failed_download_says_what_to_do_by_hand() {
    let mut f = Fixture::new();
    f.fetcher.bodies.clear();
    let (outcome, out, _) = f.run(only(true, false, false), &FakeRunner::working(), false);
    assert_eq!(outcome, Outcome::Failed);
    assert!(out.contains("HTTP 404"), "{out}");
    assert!(out.contains("curl -L --fail"), "{out}");
    assert!(f.leftovers().is_empty());
}

#[test]
fn an_env_override_replaces_the_download() {
    let mut f = Fixture::new();
    let elsewhere = f.dir.path().join("elsewhere.gguf");
    fs::write(&elsewhere, b"x").unwrap();
    f.plan.files[0].env = ("TESSAR_MODEL", Some(elsewhere.clone()));
    f.plan.files[1].env = ("TESSAR_TOKENIZER", Some(f.dir.path().join("missing.json")));
    let (outcome, out, _) = f.run(only(true, false, false), &FakeRunner::working(), false);
    assert_eq!(
        outcome,
        Outcome::Failed,
        "a missing override file fails: {out}"
    );
    assert_eq!(f.fetcher.calls(), 0);
    assert!(
        out.contains("taken from") && out.contains("TESSAR_MODEL"),
        "{out}"
    );
    assert!(
        out.contains("TESSAR_TOKENIZER") && out.contains("not a file"),
        "{out}"
    );
    assert!(!f.plan.home.exists());
}

#[test]
fn a_dry_run_touches_nothing() {
    let f = Fixture::new();
    f.write_settings("{\"theme\":\"dark\"}\n");
    f.claude_json(None, None);
    let before = (f.read_settings(), fs::read(&f.plan.claude_json).unwrap());
    let runner = FakeRunner::working();
    let opts = Options {
        dry_run: true,
        ..all()
    };
    let (outcome, out, asked) = f.run(opts, &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(f.fetcher.calls(), 0);
    assert!(asked.is_empty());
    assert!(!f.plan.home.exists());
    assert_eq!(f.read_settings(), before.0);
    assert_eq!(fs::read(&f.plan.claude_json).unwrap(), before.1);
    assert!(f.backups().is_empty());
    assert!(
        runner.calls().is_empty(),
        "a dry run does not run claude at all"
    );
    assert_eq!(out.matches("\n  would").count(), 4, "{out}");
    assert!(!out.contains("next:"), "{out}");
}

// ---- MCP -------------------------------------------------------------------

fn mcp() -> Options {
    only(false, true, false)
}

fn add_cmd() -> String {
    format!("mcp add --scope user tessar -- {EXE} mcp")
}

#[test]
fn an_unregistered_server_is_added_after_a_yes() {
    let f = Fixture::new();
    f.claude_json(None, None);
    let runner = FakeRunner::working();
    let (outcome, out, asked) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(asked.len(), 1);
    assert_eq!(runner.calls(), vec![add_cmd()]);
    assert!(out.contains("done"), "{out}");
}

#[test]
fn without_claude_code_ever_run_nothing_is_registered() {
    let f = Fixture::new();
    assert!(!f.plan.claude_json.exists());
    let runner = FakeRunner::working();
    let (outcome, _, _) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Complete);
    assert_eq!(runner.calls(), vec![add_cmd()]);
}

#[test]
fn without_claude_the_step_is_skipped_with_the_command() {
    let f = Fixture::new();
    f.claude_json(None, None);
    let runner = FakeRunner::new(|_| Err(RunError::NotFound));
    let (outcome, out, _) = f.run(mcp(), &runner, true);
    assert_eq!(
        outcome,
        Outcome::Failed,
        "an add that cannot run is a failure: {out}"
    );
    assert!(out.contains("`claude` was not found"), "{out}");
    assert!(out.contains(&format!("claude {}", add_cmd())), "{out}");
}

#[test]
fn an_unregistered_server_is_not_added_after_a_no() {
    let f = Fixture::new();
    f.claude_json(None, None);
    let runner = FakeRunner::working();
    let (outcome, out, asked) = f.run(mcp(), &runner, false);
    assert_eq!(outcome, Outcome::Incomplete, "{out}");
    assert_eq!(asked.len(), 1);
    assert!(runner.calls().is_empty());
    assert!(out.contains("skipped"), "{out}");
    assert!(out.contains("claude mcp add --scope user tessar"), "{out}");
    assert!(!out.contains("next:"), "{out}");
    assert!(!out.contains("already in place"), "{out}");
}

#[test]
fn a_server_registered_with_this_binary_is_in_place() {
    let f = Fixture::new();
    f.claude_json(Some(stdio(EXE)), None);
    let runner = FakeRunner::working();
    let (outcome, out, asked) = f.run(mcp(), &runner, false);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert!(asked.is_empty());
    assert!(runner.calls().is_empty());
    assert!(out.contains("in place"), "{out}");
}

#[test]
fn a_server_registered_elsewhere_is_replaced_after_a_yes() {
    let f = Fixture::new();
    f.claude_json(Some(stdio("/repo/target/release/tessar")), None);
    let runner = FakeRunner::working();
    let (outcome, out, asked) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert!(
        asked[0].contains("/repo/target/release/tessar"),
        "{asked:?}"
    );
    assert_eq!(
        runner.calls(),
        vec!["mcp remove tessar --scope user".to_owned(), add_cmd()]
    );
}

#[test]
fn a_server_registered_elsewhere_is_kept_after_a_no() {
    let f = Fixture::new();
    f.claude_json(Some(stdio("/repo/target/release/tessar")), None);
    let runner = FakeRunner::working();
    let (outcome, out, _) = f.run(mcp(), &runner, false);
    assert_eq!(outcome, Outcome::Incomplete, "{out}");
    assert!(runner.calls().is_empty());
    assert!(
        out.contains("claude mcp remove tessar --scope user"),
        "{out}"
    );
}

#[test]
fn other_arguments_make_a_registration_stale() {
    let f = Fixture::new();
    f.claude_json(
        Some(json!({"type": "stdio", "command": EXE, "args": ["mcp", "--home", "/x"]})),
        None,
    );
    let runner = FakeRunner::working();
    let (outcome, out, _) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(runner.calls().len(), 2);
}

#[test]
fn a_failing_add_puts_the_old_registration_back() {
    let f = Fixture::new();
    f.claude_json(
        Some(
            json!({"type": "stdio", "command": "/old/tessar", "args": ["mcp"], "env": {"K": "v"}}),
        ),
        None,
    );
    let runner = FakeRunner::new(|args| {
        let adds_us = args.contains(&EXE);
        Ok(if adds_us { failed("boom") } else { ok("") })
    });
    let (outcome, out, _) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Failed, "{out}");
    assert_eq!(
        runner.calls(),
        vec![
            "mcp remove tessar --scope user".to_owned(),
            "mcp add --scope user tessar -e K=v -- /opt/tessar/bin/tessar mcp".to_owned(),
            "mcp add --scope user tessar -e K=v -- /old/tessar mcp".to_owned(),
        ]
    );
    assert!(
        out.contains("boom") && out.contains("was put back"),
        "{out}"
    );
}

#[test]
fn a_failed_restore_is_not_called_a_restore() {
    let f = Fixture::new();
    f.claude_json(Some(stdio("/old/tessar")), None);
    let runner = FakeRunner::new(|args| {
        Ok(if args.get(1) == Some(&"remove") {
            ok("")
        } else {
            failed("boom")
        })
    });
    let (outcome, out, _) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Failed);
    assert!(!out.contains("was put back"), "{out}");
    assert!(
        out.contains("failed too") && out.contains("/old/tessar"),
        "{out}"
    );
}

#[test]
fn a_local_registration_of_another_path_goes_after_the_user_one_is_added() {
    let f = Fixture::new();
    f.claude_json(None, Some(stdio("/repo/target/release/tessar")));
    let runner = FakeRunner::working();
    let (outcome, out, _) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(
        runner.calls(),
        vec![add_cmd(), "mcp remove tessar --scope local".to_owned()]
    );
}

#[test]
fn both_scopes_stale_are_both_replaced() {
    let f = Fixture::new();
    f.claude_json(Some(stdio("/a/tessar")), Some(stdio("/b/tessar")));
    let runner = FakeRunner::working();
    let (outcome, out, asked) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(asked.len(), 1);
    assert_eq!(
        runner.calls(),
        vec![
            "mcp remove tessar --scope user".to_owned(),
            add_cmd(),
            "mcp remove tessar --scope local".to_owned(),
        ]
    );
}

#[test]
fn a_user_registration_in_place_with_a_stale_local_one_only_removes_the_local() {
    let f = Fixture::new();
    f.claude_json(Some(stdio(EXE)), Some(stdio("/b/tessar")));
    let runner = FakeRunner::working();
    let (outcome, out, _) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(runner.calls(), vec!["mcp remove tessar --scope local"]);
}

#[test]
fn a_registration_that_is_not_a_command_is_not_replaced() {
    let f = Fixture::new();
    f.claude_json(
        Some(json!({"type": "http", "url": "https://x.test/mcp"})),
        None,
    );
    let runner = FakeRunner::working();
    let (outcome, out, asked) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Incomplete, "{out}");
    assert!(asked.is_empty());
    assert!(runner.calls().is_empty());
    assert!(
        out.contains("claude mcp remove tessar --scope user"),
        "{out}"
    );
}

#[test]
fn an_unreadable_registry_means_unknown_and_nothing_is_run() {
    let f = Fixture::new();
    for text in ["{ not json", "{\"mcpServers\":{\"tessar\":7}}"] {
        fs::write(&f.plan.claude_json, text).unwrap();
        let runner = FakeRunner::working();
        let (outcome, out, asked) = f.run(mcp(), &runner, true);
        assert_eq!(outcome, Outcome::Incomplete, "{text}: {out}");
        assert!(asked.is_empty());
        assert!(runner.calls().is_empty());
        assert!(out.contains("cannot tell whether"), "{out}");
        assert!(out.contains(&format!("claude {}", add_cmd())), "{out}");
        assert_eq!(fs::read_to_string(&f.plan.claude_json).unwrap(), text);
    }
}

#[test]
fn the_registry_file_is_never_written() {
    let f = Fixture::new();
    f.claude_json(Some(stdio("/a/tessar")), None);
    let before = fs::read(&f.plan.claude_json).unwrap();
    f.run(mcp(), &FakeRunner::working(), true);
    assert_eq!(fs::read(&f.plan.claude_json).unwrap(), before);
}

#[test]
fn the_registry_path_follows_claude_config_dir() {
    let env = |name: &str| match name {
        "HOME" => Some("/u".into()),
        "CLAUDE_CONFIG_DIR" => Some("/cfg".into()),
        _ => None,
    };
    let plan = Plan::system(Path::new("/h"), EXE.into(), "/w".into(), env).unwrap();
    assert_eq!(plan.claude_json, Path::new("/cfg/.claude.json"));
    assert_eq!(plan.settings, Path::new("/cfg/settings.json"));
    let plan = Plan::system(Path::new("/h"), EXE.into(), "/w".into(), |n| {
        (n == "HOME").then(|| "/u".into())
    })
    .unwrap();
    assert_eq!(plan.claude_json, Path::new("/u/.claude.json"));
    assert_eq!(plan.settings, Path::new("/u/.claude/settings.json"));
}

// ---- hooks -----------------------------------------------------------------

fn hooks() -> Options {
    only(false, false, true)
}

#[test]
fn a_missing_settings_file_is_created_with_the_snippet() {
    let f = Fixture::new();
    let (outcome, out, asked) = f.run(hooks(), &FakeRunner::working(), true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(asked.len(), 1);
    assert_eq!(f.read_settings(), config_snippet(Path::new(EXE)));
    assert!(f.backups().is_empty(), "there was nothing to back up");
    assert_eq!(mode(f.settings()), 0o600);
    assert!(out.contains("next: restart Claude Code"), "{out}");
}

const FOREIGN: &str = r#"{
  "zeta": 1,
  "big": 12345678901234567890123,
  "alpha": {"b": [true, null], "a": 2.50},
  "text": "caf\u00e9 \"quoted\" \u043f\u0440\u0438\u0432\u0435\u0442",
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {"type": "command", "command": "notify-send done"}
        ]
      }
    ],
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "other-tool check"}]}
    ]
  },
  "model": "opus"
}
"#;

#[test]
fn other_settings_and_hooks_survive_byte_for_byte() {
    let f = Fixture::new();
    f.write_settings(FOREIGN);
    let mut prompt = Answer {
        yes: true,
        ..Answer::default()
    };
    let (outcome, out) = f.run_with(hooks(), &FakeRunner::working(), &mut prompt);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    let text = f.read_settings();
    // Everything before the first insertion and after the last is untouched.
    let up_to = FOREIGN
        .find("\n      {\"hooks\": [{\"type\": \"command\", \"command\": \"other-tool")
        .unwrap();
    assert!(text.starts_with(&FOREIGN[..up_to]), "{text}");
    assert!(
        text.ends_with("\n  },\n  \"model\": \"opus\"\n}\n"),
        "{text}"
    );
    for kept in [
        "12345678901234567890123",
        "2.50",
        "caf\\u00e9 \\\"quoted\\\" \\u043f\\u0440\\u0438\\u0432\\u0435\\u0442",
        "{\"type\": \"command\", \"command\": \"notify-send done\"}",
    ] {
        assert!(text.contains(kept), "{kept}\n{text}");
    }
    let at = |needle: &str| {
        text.find(needle)
            .unwrap_or_else(|| panic!("{needle} in {text}"))
    };
    assert!(at("\"zeta\"") < at("\"big\""));
    assert!(at("\"Stop\"") < at("\"UserPromptSubmit\""));
    assert!(at("other-tool check") < at(&format!("{EXE} hook prompt")));
    for kind in ["prompt", "session", "agent", "subagent"] {
        assert_eq!(
            text.matches(&format!(" hook {kind}\"")).count(),
            1,
            "{kind}"
        );
    }
    let parsed: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        parsed["hooks"]["Stop"][0]["hooks"][0]["command"],
        "notify-send done"
    );
    // The old file is saved, byte for byte, and the output says where.
    let backups = f.backups();
    assert_eq!(backups.len(), 1);
    assert!(
        backups[0]
            .to_string_lossy()
            .ends_with("settings.json.tessar-backup-20261112T103005Z")
    );
    assert_eq!(fs::read_to_string(&backups[0]).unwrap(), FOREIGN);
    assert_eq!(mode(&backups[0]), 0o600);
    assert!(out.contains(&backups[0].display().to_string()), "{out}");
    assert!(
        prompt.asked[0].contains(&backups[0].display().to_string()),
        "{:?}",
        prompt.asked
    );
    assert!(f.leftovers().is_empty());
}

#[test]
fn a_second_run_is_a_byte_for_byte_no_op() {
    let mut f = Fixture::new();
    f.write_settings(FOREIGN);
    let runner = FakeRunner::working();
    assert_eq!(f.run(hooks(), &runner, true).0, Outcome::Complete);
    let after_first = fs::read(f.settings()).unwrap();
    f.plan.now = Utc.with_ymd_and_hms(2026, 11, 13, 0, 0, 0).unwrap();
    let (outcome, out, asked) = f.run(hooks(), &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert!(asked.is_empty(), "nothing to ask about");
    assert_eq!(fs::read(f.settings()).unwrap(), after_first);
    assert_eq!(f.backups().len(), 1, "no second backup");
    assert!(out.contains("nothing to change"), "{out}");
}

#[test]
fn hooks_of_another_path_are_updated_not_duplicated() {
    let f = Fixture::new();
    f.write_settings(
        r#"{"hooks":{
  "UserPromptSubmit":[{"hooks":[{"type":"command","command":"TESSAR_HOME=/x /repo/target/release/tessar hook prompt --home /x","timeout":5}]}],
  "SubagentStart":[{"hooks":[{"type":"command","command":"'/My Apps/tessar' hook subagent"}]}],
  "PreToolUse":[{"matcher":"Bash","hooks":[{"type":"command","command":"/x/not-tessar hook agent"}]}]
}}"#,
    );
    let (outcome, out, _) = f.run(hooks(), &FakeRunner::working(), true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    let text = f.read_settings();
    assert!(!text.contains("/repo/target"), "{text}");
    assert!(!text.contains("My Apps"), "{text}");
    // The environment prefix and the arguments stay.
    assert!(
        text.contains(&format!("TESSAR_HOME=/x {EXE} hook prompt --home /x")),
        "{text}"
    );
    for kind in ["prompt", "subagent"] {
        assert_eq!(
            text.matches(&format!(" hook {kind}")).count(),
            1,
            "{kind}: {text}"
        );
    }
    // A program that is not tessar is not tessar's to update: ours is added.
    assert!(text.contains("/x/not-tessar hook agent"), "{text}");
    assert!(text.contains(&format!("{EXE} hook agent")), "{text}");
    assert!(text.contains("\"timeout\":5"), "{text}");
}

#[test]
fn a_hook_in_a_form_that_is_not_understood_is_reported_and_not_duplicated() {
    let f = Fixture::new();
    let original = r#"{"hooks":{"UserPromptSubmit":[{"hooks":[{"type":"command","command":"/p/tessar hook prompt 2>/dev/null"}]}]}}"#;
    f.write_settings(original);
    let (outcome, out, _) = f.run(hooks(), &FakeRunner::working(), true);
    assert_eq!(outcome, Outcome::Failed, "{out}");
    let text = f.read_settings();
    assert_eq!(text.matches("hook prompt").count(), 1, "{text}");
    assert!(text.contains("/p/tessar hook prompt 2>/dev/null"));
    // The other three events were still added.
    assert!(text.contains("hook session") && text.contains("hook subagent"));
    assert!(
        out.contains("UserPromptSubmit") && out.contains("do not rewrite"),
        "{out}"
    );
}

#[test]
fn an_unparsable_file_is_left_untouched() {
    let f = Fixture::new();
    let broken = "{ \"hooks\": [ // comments are not JSON\n";
    f.write_settings(broken);
    let (outcome, out, asked) = f.run(hooks(), &FakeRunner::working(), true);
    assert_eq!(outcome, Outcome::Failed);
    assert!(asked.is_empty());
    assert_eq!(f.read_settings(), broken);
    assert!(f.backups().is_empty());
    assert!(out.contains("left untouched"), "{out}");
    assert!(out.contains("tessar hook config"), "{out}");
}

#[test]
fn duplicate_keys_and_surprising_shapes_are_left_untouched() {
    let f = Fixture::new();
    for text in [
        "[1]",
        "{\"hooks\": 3}",
        "{\"hooks\": {\"SessionStart\": {}}}",
        "{\"hooks\": {}, \"hooks\": {}}",
        "{\"a\": 1, \"a\": 2}",
    ] {
        f.write_settings(text);
        let (outcome, out, _) = f.run(hooks(), &FakeRunner::working(), true);
        assert_eq!(outcome, Outcome::Failed, "{text}: {out}");
        assert_eq!(f.read_settings(), text);
    }
    assert!(f.backups().is_empty());
}

#[test]
fn hooks_are_not_written_without_a_yes() {
    let f = Fixture::new();
    f.write_settings(FOREIGN);
    let (outcome, out, asked) = f.run(hooks(), &FakeRunner::working(), false);
    assert_eq!(outcome, Outcome::Incomplete, "{out}");
    assert_eq!(asked.len(), 1);
    assert_eq!(f.read_settings(), FOREIGN);
    assert!(f.backups().is_empty());
    assert!(out.contains("tessar hook config"), "{out}");
    assert!(!out.contains("already in place"), "{out}");
}

#[test]
fn a_change_made_while_the_question_is_open_is_not_overwritten() {
    let f = Fixture::new();
    f.write_settings(FOREIGN);
    let path = f.settings().to_owned();
    let theirs = "{\"theme\": \"light\"}\n";
    let mut prompt = Answer {
        yes: true,
        meanwhile: Some(Box::new(move || fs::write(&path, theirs).unwrap())),
        ..Answer::default()
    };
    let (outcome, out) = f.run_with(hooks(), &FakeRunner::working(), &mut prompt);
    assert_eq!(outcome, Outcome::Failed, "{out}");
    assert_eq!(f.read_settings(), theirs);
    assert!(f.backups().is_empty());
    assert!(f.leftovers().is_empty());
    assert!(out.contains("changed while I was waiting"), "{out}");
    // A file created meanwhile counts too.
    let g = Fixture::new();
    let path = g.settings().to_owned();
    let mut prompt = Answer {
        yes: true,
        meanwhile: Some(Box::new(move || {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, "{}").unwrap();
        })),
        ..Answer::default()
    };
    let (outcome, _) = g.run_with(hooks(), &FakeRunner::working(), &mut prompt);
    assert_eq!(outcome, Outcome::Failed);
    assert_eq!(g.read_settings(), "{}");
}

#[test]
fn a_symlinked_settings_file_is_written_through() {
    let f = Fixture::new();
    let real = f.dir.path().join("dotfiles").join("settings.json");
    fs::create_dir_all(real.parent().unwrap()).unwrap();
    fs::write(&real, "{}").unwrap();
    fs::create_dir_all(f.settings().parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(&real, f.settings()).unwrap();
    let (outcome, out, _) = f.run(hooks(), &FakeRunner::working(), true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert!(
        fs::symlink_metadata(f.settings())
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(fs::read_to_string(&real).unwrap().contains("hook session"));
}

#[test]
fn a_dangling_symlink_is_not_replaced_by_a_file() {
    let f = Fixture::new();
    fs::create_dir_all(f.settings().parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(f.dir.path().join("nowhere.json"), f.settings()).unwrap();
    let (outcome, out, asked) = f.run(hooks(), &FakeRunner::working(), true);
    assert_eq!(outcome, Outcome::Failed, "{out}");
    assert!(asked.is_empty());
    assert!(
        fs::symlink_metadata(f.settings())
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(out.contains("does not exist"), "{out}");
}

// ---- together --------------------------------------------------------------

#[test]
fn without_a_terminal_the_answer_is_no_and_a_yes_does_all_three() {
    let f = Fixture::new();
    f.write_settings("{}");
    f.claude_json(None, None);
    let runner = FakeRunner::working();
    let (outcome, out, asked) = f.run(all(), &runner, false);
    assert_eq!(outcome, Outcome::Incomplete, "{out}");
    assert_eq!(asked.len(), 2, "the models need no question");
    assert_eq!(f.read_settings(), "{}");
    assert!(runner.calls().is_empty());
    assert!(out.contains("tessar hook config"), "{out}");
    assert!(out.contains("some steps were left to you"), "{out}");
    assert!(!out.contains("already in place"), "{out}");
    assert!(
        f.plan.files[0].dest.is_file(),
        "downloads into the home do not ask"
    );

    let (outcome, out, asked) = f.run(all(), &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(asked.len(), 2);
    assert!(f.read_settings().contains("hook session"));
    assert_eq!(runner.calls(), vec![add_cmd()]);
    assert_eq!(
        out.lines().filter(|l| l.contains("done")).count(),
        2,
        "{out}"
    );
}

#[test]
fn everything_in_place_says_so_and_exits_clean() {
    let f = Fixture::new();
    f.write_settings("{}");
    f.claude_json(Some(stdio(EXE)), None);
    assert_eq!(
        f.run(all(), &FakeRunner::working(), true).0,
        Outcome::Complete
    );
    let runner = FakeRunner::working();
    let (outcome, out, asked) = f.run(all(), &runner, false);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(outcome.exit_code(), 0);
    assert!(asked.is_empty());
    assert!(runner.calls().is_empty());
    assert_eq!(out.matches("\n  in place").count(), 4, "{out}");
    assert!(
        out.ends_with("nothing to change: everything requested is already in place.\n"),
        "{out}"
    );
}

#[test]
fn exit_codes_tell_complete_from_incomplete_from_failed() {
    assert_eq!(Outcome::Complete.exit_code(), 0);
    assert_eq!(Outcome::Failed.exit_code(), 1);
    assert_eq!(Outcome::Incomplete.exit_code(), 2);
}

#[test]
fn skipped_steps_are_not_run() {
    let f = Fixture::new();
    let runner = FakeRunner::working();
    let (outcome, out, _) = f.run(only(false, false, false), &runner, true);
    assert_eq!(outcome, Outcome::Complete);
    assert!(runner.calls().is_empty());
    assert_eq!(f.fetcher.calls(), 0);
    assert!(!f.plan.home.exists());
    assert!(!out.contains("models"), "{out}");
}

#[test]
fn a_replaced_registration_keeps_its_env_and_says_only_the_names() {
    let secret = "s3cr3t-value-123";
    let f = Fixture::new();
    f.claude_json(
        Some(
            json!({"type": "stdio", "command": "/old/tessar", "args": ["mcp"],
            "env": {"TESSAR_HOME": secret}}),
        ),
        None,
    );
    let runner = FakeRunner::working();
    let dry = Options {
        dry_run: true,
        ..mcp()
    };
    let (_, out, _) = f.run(dry, &runner, true);
    assert!(
        out.contains("TESSAR_HOME") && !out.contains(secret),
        "{out}"
    );
    let (outcome, out, asked) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert_eq!(
        runner.calls(),
        vec![
            "mcp remove tessar --scope user".to_owned(),
            format!("mcp add --scope user tessar -e TESSAR_HOME={secret} -- {EXE} mcp"),
        ]
    );
    assert!(
        out.contains("environment variables TESSAR_HOME are kept"),
        "{out}"
    );
    assert!(
        !out.contains(secret) && !asked[0].contains(secret),
        "{out}{asked:?}"
    );
}

#[test]
fn secrets_never_reach_the_output() {
    let secret = "s3cr3t-value-123";
    let f = Fixture::new();
    f.claude_json(
        Some(
            json!({"type": "stdio", "command": "/old/tessar", "args": ["mcp"],
            "env": {"API_KEY": secret}}),
        ),
        None,
    );
    // Every add fails, the restore too; stderr stays quiet.
    let runner = FakeRunner::new(|args| {
        Ok(if args.get(1) == Some(&"add") {
            failed("")
        } else {
            ok("")
        })
    });
    let (outcome, out, asked) = f.run(mcp(), &runner, true);
    assert_eq!(outcome, Outcome::Failed);
    assert!(out.contains("API_KEY=\u{2026}"), "{out}");
    assert!(!out.contains(secret), "{out}");
    assert!(!asked.iter().any(|q| q.contains(secret)));
    assert_eq!(
        redacted(&["mcp", "add", "-e", "A=b", "--", "x"]),
        "mcp add -e A=\u{2026} -- x"
    );
}

#[test]
fn a_number_a_value_cannot_hold_is_not_a_write_error() {
    let f = Fixture::new();
    f.write_settings("{\"x\": 1e400}\n");
    let (outcome, out, _) = f.run(hooks(), &FakeRunner::working(), true);
    assert_eq!(outcome, Outcome::Complete, "{out}");
    assert!(f.read_settings().starts_with("{\"x\": 1e400,"));
}
