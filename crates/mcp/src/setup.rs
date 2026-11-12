//! `tessar setup`: the three steps a new user does by hand, in this order.
//!
//! 1. the embedding model and its tokenizer into `<home>/models/`,
//! 2. the MCP server registered with Claude Code (user scope),
//! 3. the hooks of `tessar hook config` merged into `settings.json`.
//!
//! Every step is idempotent and ends with one line: done, in place, would
//! (a dry run), skipped or failed. The outside world is behind three small
//! seams ([`Fetcher`], [`Runner`], [`Prompt`]) and the paths in [`Plan`], so
//! the logic runs in tests without a network, a `claude` or a real home.
//! Nothing here opens the journal or starts the daemon, and `claude` is run
//! only to add or remove a registration: never `claude mcp get` or `list`,
//! which connect to the servers and would start `tessar mcp`.

use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use crate::hook::shell_word;
use crate::private;
use registry::{Lookup, Registration};

mod registry;
mod settings;

/// Both files are pinned to a commit of their Hugging Face repository, not to
/// `main`, so that a push upstream cannot break this version. The hashes are
/// those of exactly these files (the model's is the upstream's LFS SHA-256).
const MODEL_URL: &str = "https://huggingface.co/MongoDB/voyage-4-nano-GGUF/resolve/03547551689253732502486d62de4cd0970d6564/voyage-4-nano-Q8_0.gguf";
const MODEL_SHA256: &str = "c6ed6cc898d50ae1a4492809affadc6c0d023f8ac05c07a37abe20c563152560";
const MODEL_SIZE: u64 = 374_128_832;
const TOKENIZER_URL: &str = "https://huggingface.co/MongoDB/voyage-4-nano/resolve/5fb34e870f49f19e79c94d5e118f5ee730133279/tokenizer.json";
const TOKENIZER_SHA256: &str = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
const TOKENIZER_SIZE: u64 = 7_031_645;

/// The name the MCP server is registered under.
const SERVER: &str = "tessar";

// ---- seams ---------------------------------------------------------------

/// Why a download did not happen.
#[derive(Debug, PartialEq, Eq)]
pub enum FetchError {
    /// There is no `curl` to run.
    NoCurl,
    Failed(String),
}

/// Downloads `url` into the file `dest`.
pub trait Fetcher {
    fn fetch(&self, url: &str, dest: &Path) -> Result<(), FetchError>;
}

/// What a finished command printed.
#[derive(Debug, Clone, Default)]
pub struct CmdOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Why a command did not run.
#[derive(Debug, PartialEq, Eq)]
pub enum RunError {
    NotFound,
    Other(String),
}

/// Runs an external program to completion.
pub trait Runner {
    fn run(&self, program: &str, args: &[&str]) -> Result<CmdOutput, RunError>;
}

/// Asks the user a yes/no question; the default answer is no.
pub trait Prompt {
    fn confirm(&mut self, question: &str) -> bool;
}

/// `curl -L --fail` into the destination file: https only (also across
/// redirects), three retries, and a progress bar when stderr is a terminal.
pub struct Curl;

impl Fetcher for Curl {
    fn fetch(&self, url: &str, dest: &Path) -> Result<(), FetchError> {
        let mut curl = Command::new("curl");
        curl.args(["--location", "--fail", "--show-error"])
            .args(["--proto", "=https", "--proto-redir", "=https"])
            .args(["--retry", "3"])
            .arg(if io::stderr().is_terminal() {
                "--progress-bar"
            } else {
                "--silent"
            })
            .arg("--output")
            .arg(dest)
            .arg(url)
            .stdin(Stdio::null());
        match curl.status() {
            Ok(s) if s.success() => Ok(()),
            Ok(s) => Err(FetchError::Failed(format!("curl exited with {s}"))),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Err(FetchError::NoCurl),
            Err(e) => Err(FetchError::Failed(format!("cannot run curl: {e}"))),
        }
    }
}

/// Real child processes.
pub struct System;

impl Runner for System {
    fn run(&self, program: &str, args: &[&str]) -> Result<CmdOutput, RunError> {
        match Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .output()
        {
            Ok(o) => Ok(CmdOutput {
                success: o.status.success(),
                stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Err(RunError::NotFound),
            Err(e) => Err(RunError::Other(e.to_string())),
        }
    }
}

/// The user at the terminal. Without a terminal the answer is no, unless
/// `yes` says otherwise.
pub struct Terminal {
    pub yes: bool,
}

impl Prompt for Terminal {
    fn confirm(&mut self, question: &str) -> bool {
        if self.yes {
            return true;
        }
        if !io::stdin().is_terminal() {
            return false;
        }
        print!("{question} [y/N] ");
        let _ = io::stdout().flush();
        let mut answer = String::new();
        if io::stdin().lock().read_line(&mut answer).is_err() {
            return false;
        }
        matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
    }
}

// ---- plan and options ----------------------------------------------------

/// Which steps to run and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub dry_run: bool,
    pub models: bool,
    pub mcp: bool,
    pub hooks: bool,
    pub allow_unverified: bool,
}

/// A file the embedder needs.
#[derive(Debug, Clone)]
pub struct ModelFile {
    pub label: &'static str,
    pub url: String,
    pub dest: PathBuf,
    /// Lower-case hex SHA-256 of the file.
    pub sha256: String,
    /// Its size in bytes, for the line printed before the download.
    pub size: u64,
    /// The environment variable that points the loader elsewhere, with its
    /// value when it is set.
    pub env: (&'static str, Option<PathBuf>),
}

/// Where everything is.
#[derive(Debug, Clone)]
pub struct Plan {
    pub home: PathBuf,
    /// The running binary, resolved.
    pub exe: PathBuf,
    /// `settings.json` of Claude Code.
    pub settings: PathBuf,
    /// `.claude.json` of Claude Code, where it keeps the MCP servers (read only).
    pub claude_json: PathBuf,
    /// The directory `claude` would treat as the project.
    pub cwd: PathBuf,
    /// What `~` and `$HOME` mean in a hook command.
    pub user_home: Option<PathBuf>,
    pub files: Vec<ModelFile>,
    pub now: DateTime<Utc>,
}

impl Plan {
    /// The plan for this machine. `env` reads an environment variable.
    pub fn system(
        home: &Path,
        exe: PathBuf,
        cwd: PathBuf,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Result<Plan, String> {
        let non_empty = |name: &str| env(name).filter(|v| !v.is_empty()).map(PathBuf::from);
        let user_home = non_empty("HOME");
        let hook_home = user_home.clone();
        // Claude Code keeps `.claude.json` inside CLAUDE_CONFIG_DIR when it
        // is set, else beside `~/.claude`.
        let (settings, claude_json) = match non_empty("CLAUDE_CONFIG_DIR") {
            Some(dir) => (dir.join("settings.json"), dir.join(".claude.json")),
            None => {
                let user_home = user_home.ok_or("HOME is not set, so ~/.claude cannot be found")?;
                (
                    user_home.join(".claude").join("settings.json"),
                    user_home.join(".claude.json"),
                )
            }
        };
        let models = home.join("models");
        Ok(Plan {
            home: home.to_owned(),
            exe,
            settings,
            claude_json,
            cwd,
            user_home: hook_home,
            files: vec![
                ModelFile {
                    label: "model",
                    url: MODEL_URL.to_owned(),
                    dest: models.join("voyage-4-nano-Q8_0.gguf"),
                    sha256: MODEL_SHA256.to_owned(),
                    size: MODEL_SIZE,
                    env: ("TESSAR_MODEL", non_empty("TESSAR_MODEL")),
                },
                ModelFile {
                    label: "tokenizer",
                    url: TOKENIZER_URL.to_owned(),
                    dest: models.join("voyage-4-nano").join("tokenizer.json"),
                    sha256: TOKENIZER_SHA256.to_owned(),
                    size: TOKENIZER_SIZE,
                    env: ("TESSAR_TOKENIZER", non_empty("TESSAR_TOKENIZER")),
                },
            ],
            now: Utc::now(),
        })
    }
}

// ---- the run -------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Done,
    InPlace,
    Would,
    /// Not done, and the user has to do it by hand.
    Skipped,
    Failed,
}

impl Status {
    fn word(self) -> &'static str {
        match self {
            Status::Done => "done",
            Status::InPlace => "in place",
            Status::Would => "would",
            Status::Skipped => "skipped",
            Status::Failed => "FAILED",
        }
    }
}

/// How a run ended; the exit code of `tessar setup`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Every requested step is done or was already in place: exit 0.
    Complete,
    /// No step failed, but some were left to the user (declined, no terminal
    /// to ask on, no `claude`, an unreadable registration): exit 2, so that a
    /// script does not take an unfinished install for a finished one.
    Incomplete,
    /// A step failed: exit 1.
    Failed,
}

impl Outcome {
    pub fn exit_code(self) -> u8 {
        match self {
            Outcome::Complete => 0,
            Outcome::Failed => 1,
            Outcome::Incomplete => 2,
        }
    }
}

/// One run of `tessar setup`.
pub struct Setup<'a> {
    pub plan: &'a Plan,
    pub opts: Options,
    pub fetcher: &'a dyn Fetcher,
    pub runner: &'a dyn Runner,
    pub prompt: &'a mut dyn Prompt,
    pub out: &'a mut dyn Write,
    failed: bool,
    changed: bool,
    pending: bool,
}

impl<'a> Setup<'a> {
    pub fn new(
        plan: &'a Plan,
        opts: Options,
        fetcher: &'a dyn Fetcher,
        runner: &'a dyn Runner,
        prompt: &'a mut dyn Prompt,
        out: &'a mut dyn Write,
    ) -> Self {
        Setup {
            plan,
            opts,
            fetcher,
            runner,
            prompt,
            out,
            failed: false,
            changed: false,
            pending: false,
        }
    }

    /// Runs the requested steps.
    pub fn run(&mut self) -> Outcome {
        if self.opts.dry_run {
            self.say("dry run: nothing will be changed");
        }
        if self.opts.models {
            self.say("models");
            let plan = self.plan;
            for file in &plan.files {
                self.model_file(file);
            }
        }
        if self.opts.mcp {
            self.say("MCP server");
            self.mcp();
        }
        if self.opts.hooks {
            self.say("hooks");
            self.hooks();
        }
        if self.failed {
            self.say("some steps failed: see the lines marked FAILED above.");
        } else if self.pending {
            self.say("some steps were left to you: run the commands printed above, then run");
            self.say("`tessar setup` again to check.");
        }
        if !self.opts.dry_run && !self.failed {
            if self.changed {
                self.say("next: restart Claude Code (or run /mcp there to reconnect the server).");
            } else if !self.pending {
                self.say("nothing to change: everything requested is already in place.");
            }
        }
        match (self.failed, self.pending) {
            (true, _) => Outcome::Failed,
            (false, true) => Outcome::Incomplete,
            (false, false) => Outcome::Complete,
        }
    }

    fn say(&mut self, text: &str) {
        let _ = writeln!(self.out, "{text}");
    }

    /// One result line; continuation lines of `text` are indented.
    fn report(&mut self, status: Status, label: &str, text: &str) {
        match status {
            Status::Failed => self.failed = true,
            Status::Done => self.changed = true,
            Status::Skipped => self.pending = true,
            Status::InPlace | Status::Would => {}
        }
        let mut lines = text.lines();
        let first = lines.next().unwrap_or_default();
        let _ = writeln!(self.out, "  {:<8} {label}: {first}", status.word());
        for line in lines {
            let _ = writeln!(self.out, "           {line}");
        }
    }

    // -- models

    fn model_file(&mut self, file: &ModelFile) {
        let (var, value) = &file.env;
        if let Some(path) = value {
            if path.is_file() {
                let text = format!("taken from {} ({var}); not downloading", path.display());
                self.report(Status::InPlace, file.label, &text);
            } else {
                let text = format!("{var} is {}, which is not a file", path.display());
                self.report(Status::Failed, file.label, &text);
            }
            return;
        }
        if file.dest.is_file() {
            let (status, text) = self.check_existing(file);
            self.report(status, file.label, &text);
            return;
        }
        if self.opts.dry_run {
            let text = format!("download {} to {}", file.url, file.dest.display());
            self.report(Status::Would, file.label, &text);
            return;
        }
        let (status, text) = self.download(file);
        self.report(status, file.label, &text);
    }

    fn check_existing(&self, file: &ModelFile) -> (Status, String) {
        let dest = file.dest.display();
        match sha256_file(&file.dest) {
            Err(e) => (Status::Failed, format!("cannot read {dest}: {e}")),
            Ok(hash) if hash == file.sha256 => (Status::InPlace, format!("{dest}")),
            Ok(hash) if self.opts.allow_unverified => (
                Status::InPlace,
                format!(
                    "{dest} (accepted without verification: its SHA-256 is {hash}, expected {})",
                    file.sha256
                ),
            ),
            Ok(hash) => (
                Status::Failed,
                format!(
                    "{dest} exists but its SHA-256 is {hash}, expected {}; it is left as it is \
                     (it is not the file this version was tested with).\nMove it away to \
                     download the right one, or run again with --allow-unverified to accept it.",
                    file.sha256
                ),
            ),
        }
    }

    fn download(&mut self, file: &ModelFile) -> (Status, String) {
        let manual = manual_download(file);
        let Some(dir) = file.dest.parent() else {
            return (
                Status::Failed,
                format!("{} has no directory", file.dest.display()),
            );
        };
        let prepared = private::create_home(&self.plan.home)
            .and_then(|()| private::create_own_dir_all(&self.plan.home, dir));
        if let Err(e) = prepared {
            return (
                Status::Failed,
                format!("cannot create {}: {e}", dir.display()),
            );
        }
        let name = file
            .dest
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let part = dir.join(format!(".{name}.part"));
        let _ = fs::remove_file(&part);
        let line = format!(
            "  downloading the {} ({} MB) from {}",
            file.label,
            file.size.div_ceil(1_000_000),
            file.url
        );
        self.say(&line);
        let fetcher = self.fetcher;
        if let Err(e) = fetcher.fetch(&file.url, &part) {
            let _ = fs::remove_file(&part);
            return match e {
                FetchError::NoCurl => (
                    Status::Failed,
                    format!("`curl` was not found. Download it by hand:\n{manual}"),
                ),
                FetchError::Failed(why) => (
                    Status::Failed,
                    format!("download failed ({why}). Download it by hand:\n{manual}"),
                ),
            };
        }
        let hash = match sha256_file(&part) {
            Ok(hash) => hash,
            Err(e) => {
                let _ = fs::remove_file(&part);
                return (Status::Failed, format!("cannot read the download: {e}"));
            }
        };
        let verified = hash == file.sha256;
        let (target, note) = if verified {
            (file.dest.clone(), String::new())
        } else if self.opts.allow_unverified {
            (
                file.dest.clone(),
                format!(
                    " (accepted without verification: its SHA-256 is {hash}, expected {})",
                    file.sha256
                ),
            )
        } else {
            (dir.join(format!("{name}.unverified")), String::new())
        };
        if let Ok(f) = fs::File::open(&part) {
            private::tighten_file(&f);
        }
        if let Err(e) = fs::rename(&part, &target) {
            let _ = fs::remove_file(&part);
            return (
                Status::Failed,
                format!("cannot move the download to {}: {e}", target.display()),
            );
        }
        if verified || self.opts.allow_unverified {
            (
                Status::Done,
                format!("downloaded to {}{note}", target.display()),
            )
        } else {
            (
                Status::Failed,
                format!(
                    "the download has SHA-256 {hash}, expected {}; it is not used and is kept \
                     as {}.\nIf you trust it anyway, move it to {} or run again with \
                     --allow-unverified.",
                    file.sha256,
                    target.display(),
                    file.dest.display()
                ),
            )
        }
    }

    // -- MCP

    fn mcp(&mut self) {
        let exe = self.plan.exe.display().to_string();
        let add = format!(
            "claude mcp add --scope user {SERVER} -- {} mcp",
            shell_word(&exe)
        );
        let regs = match registry::read(&self.plan.claude_json, &self.plan.cwd) {
            Lookup::Found(regs) => regs,
            Lookup::Unknown(why) => {
                let text = format!(
                    "cannot tell whether a `{SERVER}` server is registered ({why}).\n\
                     If it is not, register it by hand:\n{add}"
                );
                return self.report(Status::Skipped, SERVER, &text);
            }
        };
        let current = |r: &Registration| r.is_setup_of(&self.plan.exe);
        let user_ok = regs.iter().any(|r| r.scope == "user" && current(r));
        let stale: Vec<&Registration> = regs.iter().filter(|r| !current(r)).collect();
        if user_ok && stale.is_empty() {
            return self.report(Status::InPlace, SERVER, &format!("registered with {exe}"));
        }
        if let Some(r) = stale.iter().find(|r| r.command.is_none()) {
            // It cannot be put back if the replacement fails.
            let text = format!(
                "`{SERVER}` is registered as {}; not replacing that.\nRemove it by hand, then \
                 register this binary:\nclaude {}\n{add}",
                r.describe(),
                r.remove_args().join(" ")
            );
            return self.report(Status::Skipped, SERVER, &text);
        }
        let manual: String = stale
            .iter()
            .map(|r| format!("claude {}\n", r.remove_args().join(" ")))
            .chain((!user_ok).then(|| add.clone()))
            .collect();
        // The new registration keeps the old one's environment (it may hold
        // `TESSAR_HOME`); only the names are ever shown.
        let env: Vec<(String, String)> = stale
            .iter()
            .find(|r| r.scope == "user")
            .or(stale.first())
            .map(|r| r.env.clone())
            .unwrap_or_default();
        let names = env
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let kept = if env.is_empty() {
            String::new()
        } else {
            format!("the environment variables {names} are kept")
        };
        let manual = if env.is_empty() || user_ok {
            manual
        } else {
            format!(
                "{manual}\n(then give it the environment variables {names} again, with -e NAME=VALUE)"
            )
        };
        let (question, would) = if stale.is_empty() {
            (
                format!("Register the MCP server `{SERVER}` with Claude Code at {exe}?"),
                format!("register it at {exe}"),
            )
        } else {
            let old: Vec<String> = stale.iter().map(|r| r.describe()).collect();
            let note = if kept.is_empty() {
                String::new()
            } else {
                format!(" ({kept})")
            };
            (
                format!(
                    "The MCP server `{SERVER}` is registered as {}. Replace it with {exe}?{note}",
                    old.join(" and ")
                ),
                format!("replace {} with {exe}{note}", old.join(" and ")),
            )
        };
        if self.opts.dry_run {
            return self.report(Status::Would, SERVER, &would);
        }
        if !self.prompt.confirm(&question) {
            let text = format!("not confirmed. To do it by hand:\n{}", manual.trim_end());
            return self.report(Status::Skipped, SERVER, &text);
        }
        match self.register(&stale, user_ok, &env) {
            Ok(()) => {
                let mut text = format!("registered with {exe}");
                if !kept.is_empty() && !user_ok {
                    let _ = write!(text, "\n{kept}");
                }
                self.report(Status::Done, SERVER, &text);
            }
            Err(e) => {
                let text = format!("{e}\nBy hand:\n{}", manual.trim_end());
                self.report(Status::Failed, SERVER, &text);
            }
        }
    }

    /// Replaces the stale registrations: the user-scope one is removed and
    /// added again (and put back if that fails); one in another scope goes
    /// after the new one is in.
    fn register(
        &self,
        stale: &[&Registration],
        user_ok: bool,
        env: &[(String, String)],
    ) -> Result<(), String> {
        let exe = self.plan.exe.display().to_string();
        let mut new_args: Vec<String> = ["mcp", "add", "--scope", "user", SERVER]
            .map(str::to_owned)
            .into();
        for (name, value) in env {
            new_args.extend(["-e".to_owned(), format!("{name}={value}")]);
        }
        new_args.extend(["--".to_owned(), exe, "mcp".to_owned()]);
        let add_new = || self.claude(&new_args);
        let stale_user = stale.iter().find(|r| r.scope == "user");
        if let Some(old) = stale_user {
            self.claude(&old.remove_args())?;
            if let Err(e) = add_new() {
                let back = old.add_args().unwrap_or_default();
                return Err(match self.claude(&back) {
                    Ok(()) => format!("{e}. The previous registration was put back"),
                    Err(why) => format!(
                        "{e}. Putting the previous registration back failed too ({why}); it was {}",
                        old.describe()
                    ),
                });
            }
        } else if !user_ok {
            add_new()?;
        }
        for old in stale.iter().filter(|r| r.scope != "user") {
            self.claude(&old.remove_args())?;
        }
        Ok(())
    }

    fn claude<S: AsRef<str>>(&self, args: &[S]) -> Result<(), String> {
        let args: Vec<&str> = args.iter().map(AsRef::as_ref).collect();
        match self.runner.run("claude", &args) {
            Ok(o) if o.success => Ok(()),
            Ok(o) => Err(format!(
                "`claude {}` failed: {}",
                redacted(&args),
                o.stderr.trim()
            )),
            Err(RunError::NotFound) => Err("`claude` was not found on PATH".to_owned()),
            Err(RunError::Other(e)) => Err(format!("cannot run `claude`: {e}")),
        }
    }

    // -- hooks

    fn hooks(&mut self) {
        let label = "settings.json";
        let path = match resolve_settings(&self.plan.settings) {
            Ok(path) => path,
            Err(e) => return self.report(Status::Failed, label, &e),
        };
        let shown = path.display().to_string();
        let manual = format!("Merge the output of `tessar hook config` into {shown} by hand.");
        let original = match fs::read_to_string(&path) {
            Ok(text) => Some(text),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => {
                return self.report(Status::Failed, label, &format!("cannot read {shown}: {e}"));
            }
        };
        let merged = match settings::merge_for(
            original.as_deref(),
            &self.plan.exe,
            self.plan.user_home.as_deref(),
        ) {
            Ok(merged) => merged,
            Err(why) => {
                let text = format!("{shown}: {why}; it is left untouched.\n{manual}");
                return self.report(Status::Failed, label, &text);
            }
        };
        if !merged.changed() {
            if merged.conflicts.is_empty() {
                let text = format!("the hooks are in {shown}");
                self.report(Status::InPlace, label, &text);
            }
            return self.conflicts(&merged, &shown);
        }
        let what = merged.describe();
        let backup = backup_path(&path, self.plan.now);
        if self.opts.dry_run {
            self.report(Status::Would, label, &format!("{what} in {shown}"));
            return self.conflicts(&merged, &shown);
        }
        let mut question = format!("{} in {shown}?", capitalise(&what));
        if original.as_deref().is_some_and(|t| !t.trim().is_empty()) {
            let _ = write!(
                question,
                " The old file is kept as {} (it holds the same secrets as settings.json).",
                backup.display()
            );
        }
        if !self.prompt.confirm(&question) {
            let text = format!("not confirmed. To do it by hand:\n{manual}");
            self.report(Status::Skipped, label, &text);
            return self.conflicts(&merged, &shown);
        }
        match write_settings(&path, original.as_deref(), &merged.text, &backup) {
            Ok(saved) => {
                let mut text = format!("{what} in {shown}");
                if saved {
                    let _ = write!(text, "\nthe old file is saved as {}", backup.display());
                }
                self.report(Status::Done, label, &text);
            }
            Err(e) => self.report(Status::Failed, label, &format!("{e}\n{manual}")),
        }
        self.conflicts(&merged, &shown);
    }

    /// Hooks that look like tessar's but were left alone.
    fn conflicts(&mut self, merged: &settings::Merged, shown: &str) {
        if merged.conflicts.is_empty() {
            return;
        }
        let text = format!(
            "in {shown} the tessar hook for {} is in a form I do not rewrite (a wrapper, \
             a redirect, a pipe); it is left as it is and no second one is added.\nMake it \
             `{} hook <prompt|session|agent|subagent>` yourself, or remove it and run again.",
            merged.conflicts.join(", "),
            self.plan.exe.display()
        );
        self.report(Status::Failed, "settings.json", &text);
    }
}

/// The arguments for display: the value of every `-e NAME=VALUE` is hidden,
/// since it may be a secret.
fn redacted(args: &[&str]) -> String {
    let mut shown = Vec::with_capacity(args.len());
    let mut after_env = false;
    for arg in args {
        shown.push(match (after_env, arg.split_once('=')) {
            (true, Some((name, _))) => format!("{name}=\u{2026}"),
            _ => (*arg).to_owned(),
        });
        after_env = *arg == "-e";
    }
    shown.join(" ")
}

fn capitalise(text: &str) -> String {
    let mut chars = text.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

fn manual_download(file: &ModelFile) -> String {
    let dest = file.dest.display().to_string();
    let dir = file
        .dest
        .parent()
        .map(|d| d.display().to_string())
        .unwrap_or_default();
    format!(
        "mkdir -p {} && curl -L --fail -o {} {}",
        shell_word(&dir),
        shell_word(&dest),
        shell_word(&file.url)
    )
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    }))
}

// ---- writing settings.json -----------------------------------------------

/// The file to write: a symlink is followed (a `settings.json` kept in a
/// dotfiles repository stays a link), but a dangling one is not replaced by
/// a regular file.
fn resolve_settings(path: &Path) -> Result<PathBuf, String> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => path.canonicalize().map_err(|_| {
            format!(
                "{} is a symlink to a file that does not exist; not touching it",
                path.display()
            )
        }),
        _ => Ok(path.to_owned()),
    }
}

/// Where the copy of the old file goes: next to it, named by the UTC time.
fn backup_path(path: &Path, now: DateTime<Utc>) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!(
        "{name}.tessar-backup-{}",
        now.format("%Y%m%dT%H%M%SZ")
    ))
}

/// Writes `text` over `path`: through a temporary file that is synced and
/// read back, after checking that nobody changed `path` since `original` was
/// read (Claude Code rewrites it too), with a copy of the old content saved
/// at `backup` first. `Ok(true)` when a backup was made.
fn write_settings(
    path: &Path,
    original: Option<&str>,
    text: &str,
    backup: &Path,
) -> Result<bool, String> {
    let dir = path.parent().ok_or("settings.json has no directory")?;
    fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or("settings.json has no name")?;
    let temp = dir.join(format!("{name}.tessar-tmp"));

    let prepared = write_file(&temp, text, false)
        .map_err(|e| format!("cannot write {}: {e}", temp.display()))
        .and_then(|()| {
            if let Ok(meta) = fs::metadata(path) {
                let _ = fs::set_permissions(&temp, meta.permissions());
            }
            reads_back(&temp, text)
        });
    if let Err(e) = prepared {
        let _ = fs::remove_file(&temp);
        return Err(e);
    }
    // Lost update: whoever wrote the file in the meantime wins.
    let unchanged = match (fs::read_to_string(path), original) {
        (Ok(now), Some(old)) => now == old,
        (Err(e), None) => e.kind() == io::ErrorKind::NotFound,
        _ => false,
    };
    if !unchanged {
        let _ = fs::remove_file(&temp);
        return Err(format!(
            "{} changed while I was waiting (Claude Code writes it too); nothing was written, \
             run `tessar setup` again",
            path.display()
        ));
    }
    let backed_up = match original.filter(|t| !t.trim().is_empty()) {
        Some(old) => {
            if let Err(e) = write_file(backup, old, true) {
                let _ = fs::remove_file(&temp);
                return Err(format!("cannot write {}: {e}", backup.display()));
            }
            true
        }
        None => false,
    };
    if let Err(e) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(format!("cannot replace {}: {e}", path.display()));
    }
    if let Err(e) = reads_back(path, text) {
        let restored = match original {
            Some(old) => fs::write(path, old).is_ok(),
            None => fs::remove_file(path).is_ok(),
        };
        return Err(if restored {
            format!("{e}; the old content was restored")
        } else {
            format!(
                "{e}; restoring the old content failed too{}",
                if backed_up {
                    format!(": copy {} over it", backup.display())
                } else {
                    String::new()
                }
            )
        });
    }
    Ok(backed_up)
}

/// Creates `path` private and syncs it; with `new` it must not exist.
fn write_file(path: &Path, text: &str, new: bool) -> io::Result<()> {
    let mut options = private::file_options();
    options.write(true);
    if new {
        options.create_new(true);
    } else {
        options.create(true).truncate(true);
    }
    let mut file = options.open(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()
}

fn reads_back(path: &Path, expected: &str) -> Result<(), String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("cannot re-read {}: {e}", path.display()))?;
    if text != expected {
        return Err(format!("{} does not read back as written", path.display()));
    }
    // Syntax only: a number like 1e400 elsewhere in the file is not ours to judge.
    serde_json::from_str::<&serde_json::value::RawValue>(&text)
        .map(drop)
        .map_err(|e| format!("{} does not parse after writing: {e}", path.display()))
}

#[cfg(test)]
mod tests;
