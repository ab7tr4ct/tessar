//! `tessar`: the memory as an MCP server.
//!
//! - `tessar daemon` holds the memory and the model and listens on a Unix
//!   socket.
//! - `tessar mcp` is what an agent runs: a stdio proxy to the daemon,
//!   which it starts if none is running. `--standalone` serves in process.
//!
//! Stdout of `tessar mcp` carries the protocol and nothing else. Everything
//! meant for a human goes to stderr.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::transport::stdio;
use tessar_core::clock::{Clock, SystemClock};
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::log::LogError;
use tessar_core::store::{Memory, Op, Semantic};
use tessar_embed_cache::CachedEmbedder;
use tessar_journal::FileJournal;
use tessar_mcp::daemon::{self, Projects, socket_path};
use tessar_mcp::fingerprint::{Identity, cache_path, fingerprint};
use tessar_mcp::hook;
use tessar_mcp::lifecycle;
use tessar_mcp::project::project_name;
use tessar_mcp::proxy;
use tessar_mcp::purge;
use tessar_mcp::server::TessarServer;
use tessar_mcp::tool_log::ToolLog;
use tokio::net::{UnixListener, UnixStream};

/// A line on stderr that cannot panic; see [`tessar_mcp::log`].
macro_rules! logln {
    ($($arg:tt)*) => {
        tessar_mcp::log(format_args!($($arg)*))
    };
}

#[cfg(not(feature = "candle"))]
compile_error!("enable the `candle` feature of tessar");

const USAGE: &str = "\
tessar: a memory engine for AI agents

USAGE:
    tessar mcp [--home DIR] [--standalone] [--keep-daemon]
                               serve the memory as an MCP server on stdio. By
                               default a proxy to the daemon, which is started
                               when none runs and restarted when it is another
                               build than this binary (--keep-daemon: never
                               restart it); --standalone serves in process
    tessar daemon [--home DIR] [--idle-exit MINUTES] [--takeover PID]
                               hold the memory and the model and serve every
                               session on DIR/tessar.sock. If another process
                               holds the memory it exits at once. --takeover
                               replaces the daemon with that pid: it gets in
                               line for the memory, sends the old daemon
                               SIGTERM and waits up to 20 s for it to let go
                               (how `tessar mcp` upgrades a daemon)
    tessar daemon stop|status [--home DIR]
                               stop the daemon (SIGTERM to the pid it reports
                               on its socket), or print its build, pid, uptime
                               and open sessions
    tessar redact fNN... [--purge] [--home DIR]
                               destroy the content of facts by handle (starts
                               the daemon if needed); with --purge also erase
                               it from the journal file, as `tessar purge`
    tessar lesson \"TEXT\" [--space project|user] [--home DIR]
                               record a lesson from a mistake, as the `lesson`
                               tool does (starts the daemon if needed)
    tessar purge [--home DIR]  erase redacted content from the journal file and
                               drop the vector cache. Stops the daemon first;
                               the next session starts it again
    tessar hook prompt|session|agent|subagent [--home DIR]
                               Claude Code hooks: read the hook JSON on stdin,
                               print context for the model on stdout. `agent`
                               (PreToolUse on Agent) prints nothing: it leaves
                               the memory for `subagent` (SubagentStart)
    tessar hook config         print the hooks snippet for ~/.claude/settings.json
    tessar --version           print the build id
    tessar --help              show this text

The memory lives in DIR (default: $TESSAR_HOME, else ~/.tessar): journal.jsonl,
tessar.sock, tessar.pid and daemon.log, and the embedding model in
models/voyage-4-nano-Q8_0.gguf (and models/voyage-4-nano/tokenizer.json for
the candle embedder). Set TESSAR_MODEL or TESSAR_TOKENIZER to use files from
elsewhere. TESSAR_EMBEDDER=candle names the embedder (the only one in this build).";

/// Vector length in the index.
const INDEX_DIMS: usize = 1024;

/// The model file name under `<home>/models/`.
const MODEL_FILE: &str = "voyage-4-nano-Q8_0.gguf";

/// The model's name in the vector cache's fingerprint and path.
const MODEL_NAME: &str = "voyage-4-nano";

/// An embedder that reports the first problem of the vector cache behind it
/// to the log, once each, and otherwise passes every call through. Looking
/// after each call is the one place that sees every write: the memory opening,
/// the warm-up of its index and each write through the tools all embed here.
struct CacheWatch {
    cache: CachedEmbedder,
    write_logged: AtomicBool,
    mismatch_logged: AtomicBool,
    /// How many times the cache's state was looked at (for the tests).
    reports: AtomicUsize,
}

impl CacheWatch {
    fn new(cache: CachedEmbedder) -> Self {
        Self {
            cache,
            write_logged: AtomicBool::new(false),
            mismatch_logged: AtomicBool::new(false),
            reports: AtomicUsize::new(0),
        }
    }

    /// Never fails the call: the cache is derived data.
    fn report(&self) {
        self.reports.fetch_add(1, Ordering::Relaxed);
        if let Some(e) = self.cache.write_error()
            && !self.write_logged.swap(true, Ordering::Relaxed)
        {
            logln!("tessar: vector cache stopped writing to disk: {e}");
        }
        if let Some(e) = self.cache.length_mismatch()
            && !self.mismatch_logged.swap(true, Ordering::Relaxed)
        {
            logln!("tessar: vector cache: {e}");
        }
    }
}

impl Embedder for CacheWatch {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let out = self.cache.embed_documents(texts);
        self.report();
        out
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        // A query never changes the cache, and looking at it takes the lock
        // that a batch holds while it embeds and syncs: a query must not
        // wait for that.
        self.cache.embed_query(text)
    }
}

/// Wraps `inner` in the on-disk vector cache, in a file of its own for this
/// fingerprint: directories of earlier fingerprints stay on disk until
/// `tessar purge`. The cache is derived data, so any problem with it is
/// reported and the plain embedder is used.
fn cached(home: &Path, inner: Arc<dyn Embedder>, identity: &Identity<'_>) -> Arc<dyn Embedder> {
    let opened = fingerprint(MODEL_NAME, identity)
        .map_err(|e| format!("cannot identify the model files: {e}"))
        .and_then(|fp| {
            let path = cache_path(home, &fp);
            CachedEmbedder::open(&path, &fp, Arc::clone(&inner))
                .map(|cache| (path, cache))
                .map_err(|e| e.to_string())
        });
    match opened {
        Ok((path, cache)) => {
            logln!(
                "tessar: vector cache {}: {} vectors",
                path.display(),
                cache.len()
            );
            let watch = CacheWatch::new(cache);
            watch.report();
            Arc::new(watch)
        }
        Err(e) => {
            logln!("tessar: vector cache unusable, embedding without it: {e}");
            inner
        }
    }
}

/// How long the proxy waits for a freshly started daemon (the model loads).
const DAEMON_WAIT: Duration = Duration::from_secs(60);

/// What the command line asked for.
#[derive(Debug, PartialEq)]
enum Cli {
    Mcp {
        home: PathBuf,
        standalone: bool,
        keep_daemon: bool,
    },
    Daemon {
        home: PathBuf,
        idle_exit: Option<Duration>,
        /// The pid of the daemon to replace.
        takeover: Option<u32>,
    },
    DaemonStop {
        home: PathBuf,
    },
    DaemonStatus {
        home: PathBuf,
    },
    Purge {
        home: PathBuf,
    },
    Redact {
        home: PathBuf,
        handles: Vec<String>,
        purge: bool,
    },
    Hook {
        home: PathBuf,
        kind: HookKind,
    },
    Lesson {
        home: PathBuf,
        text: String,
        user: bool,
    },
    Version,
    Help,
}

/// Which hook.
#[derive(Debug, Clone, Copy, PartialEq)]
enum HookKind {
    Prompt,
    Session,
    Agent,
    Subagent,
    Config,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse(&args, |name| std::env::var_os(name)) {
        Ok(Cli::Mcp {
            home,
            standalone: false,
            keep_daemon,
        }) => run_proxy(&home, keep_daemon),
        Ok(Cli::Mcp {
            home,
            standalone: true,
            ..
        }) => run_standalone(&home),
        Ok(Cli::DaemonStop { home }) => run_lifecycle(lifecycle::stop(&home)),
        Ok(Cli::DaemonStatus { home }) => run_lifecycle(lifecycle::status(&home)),
        Ok(Cli::Version) => {
            println!("tessar {}", tessar_mcp::BUILD_ID);
            ExitCode::SUCCESS
        }
        Ok(Cli::Daemon {
            home,
            idle_exit,
            takeover,
        }) => run_daemon(&home, idle_exit, takeover),
        Ok(Cli::Purge { home }) => run_purge(&home),
        Ok(Cli::Redact {
            home,
            handles,
            purge,
        }) => run_redact(&home, &handles, purge),
        Ok(Cli::Hook { home, kind }) => run_hook(&home, kind),
        Ok(Cli::Lesson { home, text, user }) => run_lesson(&home, &text, user),
        Ok(Cli::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(e) => fail(&e),
    }
}

fn fail(message: &str) -> ExitCode {
    logln!("tessar: {message}");
    ExitCode::FAILURE
}

/// Parses the command line. `env` reads an environment variable.
fn parse(args: &[String], env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Result<Cli, String> {
    let Some(command) = args.first().map(String::as_str) else {
        return Ok(Cli::Help);
    };
    if matches!(command, "--help" | "-h" | "help") {
        return Ok(Cli::Help);
    }
    if matches!(command, "--version" | "-V" | "version") {
        return Ok(Cli::Version);
    }
    if command == "hook" {
        return parse_hook(&args[1..], env);
    }
    if command == "redact" {
        return parse_redact(&args[1..], env);
    }
    if command == "lesson" {
        return parse_lesson(&args[1..], env);
    }
    if command == "purge" {
        return parse_purge(&args[1..], env);
    }
    if !matches!(command, "mcp" | "daemon") {
        return Err(format!("unknown command `{command}`\n\n{USAGE}"));
    }
    // `daemon stop` and `daemon status` control a running daemon.
    let (control, rest) = match args.get(1).map(String::as_str) {
        Some(c @ ("stop" | "status")) if command == "daemon" => (Some(c), &args[2..]),
        _ => (None, &args[1..]),
    };
    let mut home = None;
    let mut standalone = false;
    let mut keep_daemon = false;
    let mut idle_exit = None;
    let mut takeover = None;
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--home" => {
                home = Some(PathBuf::from(it.next().ok_or("--home needs a directory")?));
            }
            a if a.starts_with("--home=") => home = Some(PathBuf::from(&a["--home=".len()..])),
            "--standalone" if command == "mcp" => standalone = true,
            "--keep-daemon" if command == "mcp" => keep_daemon = true,
            "--takeover" if command == "daemon" && control.is_none() => {
                takeover = Some(daemon_pid(it.next().ok_or("--takeover needs a pid")?)?);
            }
            a if a.starts_with("--takeover=") && command == "daemon" && control.is_none() => {
                takeover = Some(daemon_pid(&a["--takeover=".len()..])?);
            }
            "--idle-exit" if command == "daemon" && control.is_none() => {
                idle_exit = Some(minutes(it.next().ok_or("--idle-exit needs minutes")?)?);
            }
            a if a.starts_with("--idle-exit=") && command == "daemon" && control.is_none() => {
                idle_exit = Some(minutes(&a["--idle-exit=".len()..])?);
            }
            other => return Err(format!("unexpected argument `{other}`\n\n{USAGE}")),
        }
    }
    let home = match home {
        Some(home) => home,
        None => default_home(env)?,
    };
    Ok(match (command, control) {
        ("mcp", _) => Cli::Mcp {
            home,
            standalone,
            keep_daemon,
        },
        (_, Some("stop")) => Cli::DaemonStop { home },
        (_, Some(_)) => Cli::DaemonStatus { home },
        _ => Cli::Daemon {
            home,
            idle_exit,
            takeover,
        },
    })
}

fn parse_redact(
    args: &[String],
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<Cli, String> {
    let mut home = None;
    let mut purge = false;
    let mut handles = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--home" => {
                home = Some(PathBuf::from(it.next().ok_or("--home needs a directory")?));
            }
            a if a.starts_with("--home=") => home = Some(PathBuf::from(&a["--home=".len()..])),
            "--purge" => purge = true,
            a if a.starts_with('-') => {
                return Err(format!("unexpected argument `{a}`\n\n{USAGE}"));
            }
            handle => handles.push(handle.to_owned()),
        }
    }
    if handles.is_empty() {
        return Err(format!(
            "`redact` wants at least one fact handle, like f12\n\n{USAGE}"
        ));
    }
    let home = match home {
        Some(home) => home,
        None => default_home(env)?,
    };
    Ok(Cli::Redact {
        home,
        handles,
        purge,
    })
}

fn parse_lesson(
    args: &[String],
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<Cli, String> {
    let mut home = None;
    let mut user = false;
    let mut text = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let space = match arg.as_str() {
            "--home" => {
                home = Some(PathBuf::from(it.next().ok_or("--home needs a directory")?));
                continue;
            }
            a if a.starts_with("--home=") => {
                home = Some(PathBuf::from(&a["--home=".len()..]));
                continue;
            }
            "--space" => it.next().ok_or("--space needs project or user")?.as_str(),
            a if a.starts_with("--space=") => &a["--space=".len()..],
            a if a.starts_with('-') => {
                return Err(format!("unexpected argument `{a}`\n\n{USAGE}"));
            }
            t => {
                if text.replace(t.to_owned()).is_some() {
                    return Err(format!("`lesson` takes one text, in quotes\n\n{USAGE}"));
                }
                continue;
            }
        };
        user = match space {
            "project" => false,
            "user" => true,
            other => return Err(format!("--space is project or user, not `{other}`")),
        };
    }
    let text = text.ok_or_else(|| format!("`lesson` wants a text, in quotes\n\n{USAGE}"))?;
    let home = match home {
        Some(home) => home,
        None => default_home(env)?,
    };
    Ok(Cli::Lesson { home, text, user })
}

fn parse_purge(
    args: &[String],
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<Cli, String> {
    let mut home = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--home" => {
                home = Some(PathBuf::from(it.next().ok_or("--home needs a directory")?));
            }
            a if a.starts_with("--home=") => home = Some(PathBuf::from(&a["--home=".len()..])),
            other => return Err(format!("unexpected argument `{other}`\n\n{USAGE}")),
        }
    }
    let home = match home {
        Some(home) => home,
        None => default_home(env)?,
    };
    Ok(Cli::Purge { home })
}

fn parse_hook(
    args: &[String],
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<Cli, String> {
    let kind = match args.first().map(String::as_str) {
        Some("prompt") => HookKind::Prompt,
        Some("session") => HookKind::Session,
        Some("agent") => HookKind::Agent,
        Some("subagent") => HookKind::Subagent,
        Some("config") => HookKind::Config,
        _ => {
            return Err(format!(
                "`hook` wants prompt, session, agent, subagent or config\n\n{USAGE}"
            ));
        }
    };
    let mut home = None;
    let mut it = args[1..].iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--home" => {
                home = Some(PathBuf::from(it.next().ok_or("--home needs a directory")?));
            }
            a if a.starts_with("--home=") => home = Some(PathBuf::from(&a["--home=".len()..])),
            other => return Err(format!("unexpected argument `{other}`\n\n{USAGE}")),
        }
    }
    let home = match (home, kind) {
        (Some(home), _) => home,
        // The snippet needs no memory directory.
        (None, HookKind::Config) => PathBuf::new(),
        (None, _) => default_home(env)?,
    };
    Ok(Cli::Hook { home, kind })
}

/// The pid of a daemon to replace: one that may be signalled.
fn daemon_pid(text: &str) -> Result<u32, String> {
    lifecycle::parse_pid(text)
        .ok_or_else(|| format!("--takeover wants the pid of the daemon to replace, got `{text}`"))
}

fn minutes(text: &str) -> Result<Duration, String> {
    let m: f64 = text
        .parse()
        .ok()
        .filter(|m: &f64| m.is_finite() && *m > 0.0)
        .ok_or_else(|| format!("--idle-exit wants a positive number of minutes, got `{text}`"))?;
    Ok(Duration::from_secs_f64(m * 60.0))
}

/// `$TESSAR_HOME`, else `~/.tessar`.
fn default_home(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Result<PathBuf, String> {
    if let Some(home) = env("TESSAR_HOME").filter(|h| !h.is_empty()) {
        return Ok(PathBuf::from(home));
    }
    let user_home = env("HOME")
        .filter(|h| !h.is_empty())
        .ok_or("HOME is not set; pass --home DIR or set TESSAR_HOME")?;
    Ok(PathBuf::from(user_home).join(".tessar"))
}

/// The default model directory, `~/.tessar/models`.
fn default_models_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(PathBuf::from(home).join(".tessar").join("models"))
}

/// The model: `$TESSAR_MODEL`, else `<home>/models/…`, else the default
/// `~/.tessar/models/…`.
fn model_path(home: &Path) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("TESSAR_MODEL").filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        return if path.is_file() {
            Ok(path)
        } else {
            Err(format!(
                "embedding model not found: TESSAR_MODEL is {}, which is not a file",
                path.display()
            ))
        };
    }
    let local = home.join("models").join(MODEL_FILE);
    if local.is_file() {
        return Ok(local);
    }
    match default_models_dir().map(|d| d.join(MODEL_FILE)) {
        Some(default) if default.is_file() => Ok(default),
        Some(default) => Err(format!(
            "embedding model not found: expected {} (download voyage-4-nano in Q8_0 GGUF there, \
             or set TESSAR_MODEL)",
            default.display()
        )),
        None => Err(format!(
            "embedding model not found: expected {} (or set TESSAR_MODEL)",
            local.display()
        )),
    }
}

/// The tokenizer of the candle embedder: `$TESSAR_TOKENIZER`, else
/// `<home>/models/voyage-4-nano/tokenizer.json`, else the same under
/// `~/.tessar`.
fn tokenizer_path(home: &Path) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("TESSAR_TOKENIZER").filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        return if path.is_file() {
            Ok(path)
        } else {
            Err(format!(
                "tokenizer not found: TESSAR_TOKENIZER is {}, which is not a file",
                path.display()
            ))
        };
    }
    let local = home.join("models").join(MODEL_NAME).join("tokenizer.json");
    if local.is_file() {
        return Ok(local);
    }
    let default = default_models_dir()
        .map(|d| d.join(MODEL_NAME).join("tokenizer.json"))
        .filter(|p| p.is_file());
    default.ok_or_else(|| {
        format!(
            "tokenizer not found: expected {} (it is `tokenizer.json` from the \
             `voyageai/voyage-4-nano` repository on Hugging Face; or set TESSAR_TOKENIZER)",
            local.display()
        )
    })
}

/// Which embedder to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Candle,
}

/// `$TESSAR_EMBEDDER`: `candle`, or unset. The llama.cpp embedder lives in
/// `tessar-embed-llama`, which this binary does not link.
fn choose_backend(requested: Option<&str>) -> Result<Backend, String> {
    match requested.map(str::trim).filter(|r| !r.is_empty()) {
        None | Some("candle") => Ok(Backend::Candle),
        Some("llama") => Err(
            "TESSAR_EMBEDDER=llama is not available: this binary has only the candle embedder"
                .to_owned(),
        ),
        Some(other) => Err(format!(
            "TESSAR_EMBEDDER={other:?} is unknown; use `candle`"
        )),
    }
}

/// A loaded embedder with what the logs say about it.
struct Loaded {
    embedder: Arc<dyn Embedder>,
    dims: usize,
    description: String,
    /// What the cache's fingerprint needs besides the model file.
    backend: &'static str,
    tokenizer: Option<PathBuf>,
    settings: String,
}

fn load_candle(home: &Path, model: &Path) -> Result<Loaded, String> {
    let tokenizer = tokenizer_path(home)?;
    let e = tessar_embed_candle::CandleEmbedder::voyage_nano_gguf(model, &tokenizer)
        .map_err(|e| format!("cannot load the model: {e}"))?;
    let device = if e.is_metal() { "Metal" } else { "CPU" };
    Ok(Loaded {
        dims: e.dims(),
        description: format!("candle ({device})"),
        backend: "candle",
        tokenizer: Some(tokenizer),
        settings: tessar_embed_candle::CandleEmbedder::voyage_nano_gguf_settings(),
        embedder: Arc::new(e),
    })
}

/// Opens the journal, which takes its file lock.
fn open_journal(home: &Path) -> Result<FileJournal<Op>, LogError> {
    FileJournal::<Op>::open(home.join("journal.jsonl"))
}

/// Loads the model and opens the memory over `journal`. Slow: the model
/// loads and every fact is embedded again.
fn open_memory(
    home: &Path,
    journal: FileJournal<Op>,
    clock: Arc<dyn Clock>,
) -> Result<Memory, String> {
    let model = model_path(home)?;
    let backend = choose_backend(std::env::var("TESSAR_EMBEDDER").ok().as_deref())?;
    let loaded = match backend {
        Backend::Candle => load_candle(home, &model)?,
    };
    logln!("tessar: embedder {}", loaded.description);
    // The index holds 1024 components per vector; a shorter model cannot fill it.
    if loaded.dims < INDEX_DIMS {
        return Err(format!(
            "the model at {} gives {} dimensions per vector, at least {INDEX_DIMS} are needed",
            model.display(),
            loaded.dims
        ));
    }
    // The first embedding compiles the GPU kernels (about 15 s for a new
    // build on macOS); pay for it here, not in the first request.
    let started = std::time::Instant::now();
    if let Err(e) = loaded.embedder.embed_query("warm-up") {
        logln!("tessar: embedder warm-up failed: {e}");
    }
    logln!(
        "tessar: embedder warm-up took {} ms",
        started.elapsed().as_millis()
    );
    let identity = Identity {
        backend: loaded.backend,
        model: &model,
        tokenizer: loaded.tokenizer.as_deref(),
        settings: &loaded.settings,
    };
    let embedder = cached(home, loaded.embedder, &identity);
    Memory::open_with(
        journal,
        clock,
        Semantic {
            embedder,
            dims: INDEX_DIMS,
            near_bonus: 0.05,
            predicate_threshold: 0.70,
        },
    )
    .map_err(|e| format!("cannot open the memory: {e}"))
}

fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| format!("cannot start the runtime: {e}"))
}

fn create_home(home: &Path) -> Result<(), String> {
    tessar_mcp::private::create_home(home)
        .map_err(|e| format!("cannot create {}: {e}", home.display()))
}

// ---- tessar mcp --standalone ---------------------------------------

fn run_standalone(home: &Path) -> ExitCode {
    if let Err(e) = create_home(home) {
        return fail(&e);
    }
    // The journal first: a second instance must fail before it loads the
    // model onto the GPU.
    let journal = match open_journal(home) {
        Ok(j) => j,
        Err(LogError::Locked) => {
            return fail(&format!(
                "another tessar is using {} (a running daemon? use `tessar mcp` without --standalone)",
                home.join("journal.jsonl").display()
            ));
        }
        Err(e) => return fail(&format!("cannot open the journal: {e}")),
    };
    // Whoever holds the journal knows that no daemon runs: a pid file still
    // there is stale, and `tessar daemon stop` must not act on it.
    let _ = std::fs::remove_file(lifecycle::pid_path(home));
    let clock = Arc::new(SystemClock);
    let memory = match open_memory(home, journal, clock.clone()) {
        Ok(m) => m,
        Err(e) => return fail(&e),
    };
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project = project_name(&cwd);
    let api = match Projects::new(memory, clock).get(&project) {
        Ok(a) => a,
        Err(e) => return fail(&format!("cannot prepare the memory: {e}")),
    };
    logln!(
        "tessar: serving {} (project space `{project}`)",
        home.display()
    );
    let runtime = match runtime() {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    let served = runtime.block_on(async {
        let log = ToolLog::new(
            home,
            None,
            tessar_mcp::tool_log::session_from_env(),
            Some("mcp".to_owned()),
        );
        let service = TessarServer::shared(api)
            .with_tool_log(log)
            .serve(stdio())
            .await
            .map_err(|e| e.to_string())?;
        service
            .waiting()
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    });
    // Every commit is already durable; stdin closing is the normal end.
    match served {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail(&format!("server stopped: {e}")),
    }
}

// ---- tessar daemon -------------------------------------------------

/// The daemon's pid file, removed when the daemon ends, however it ends.
struct PidFile<'a>(&'a Path);

impl Drop for PidFile<'_> {
    fn drop(&mut self) {
        lifecycle::remove_pid_file(self.0);
    }
}

/// How long a daemon that found the journal locked waits before it looks
/// once more. A check whether the journal is in use (`tessar daemon stop`,
/// the prompt hook) holds the lock for an instant; a daemon that stepped
/// aside for that would leave nobody running.
const LOCKED_RETRY: Duration = Duration::from_millis(50);

/// Opens the journal for a daemon; `None` if another process keeps it.
fn daemon_journal(home: &Path) -> Result<Option<FileJournal<Op>>, LogError> {
    for attempt in 0..2 {
        if attempt > 0 {
            std::thread::sleep(LOCKED_RETRY);
        }
        match open_journal(home) {
            Ok(journal) => return Ok(Some(journal)),
            Err(LogError::Locked) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(None)
}

fn run_daemon(home: &Path, idle_exit: Option<Duration>, takeover: Option<u32>) -> ExitCode {
    if let Err(e) = create_home(home) {
        return fail(&e);
    }
    let journal = match takeover {
        // The replacement for a daemon: it gets in line for the journal and
        // then stops the old daemon itself.
        Some(old) => match lifecycle::take_over(home, old, lifecycle::TAKEOVER_WAIT) {
            Ok(lifecycle::Takeover::Journal(j)) => j,
            Ok(lifecycle::Takeover::Aside(why)) => {
                logln!("tessar: not taking over: {why}");
                return ExitCode::SUCCESS;
            }
            Err(e) => return fail(&e),
        },
        None => match daemon_journal(home) {
            Ok(Some(j)) => j,
            Ok(None) => {
                logln!("tessar: a tessar daemon is already running");
                return ExitCode::SUCCESS;
            }
            Err(e) => return fail(&format!("cannot open the journal: {e}")),
        },
    };
    // From here on this process is the daemon, also while the model loads:
    // the pid file says so for as long as the journal is held.
    if let Err(e) = lifecycle::write_pid_file(home) {
        logln!("tessar: cannot write the pid file: {e}");
    }
    let _pid_file = PidFile(home);
    // Loading the model takes seconds and a signal during it is not
    // handled: the default action ends the process, and nothing is lost
    // because nothing is served yet. Signals are installed below, once the
    // memory is open.
    let clock = Arc::new(SystemClock);
    let memory = match open_memory(home, journal, clock.clone()) {
        Ok(m) => m,
        Err(e) => return fail(&e),
    };
    let runtime = match runtime() {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    let socket = socket_path(home);
    let served = runtime.block_on(async {
        // A failure to bind leaves the socket alone: it may be another
        // process's.
        let listener = bind(&socket).await?;
        logln!(
            "tessar: daemon serving {} on {}",
            home.display(),
            socket.display()
        );
        logln!(
            "tessar: build {}, pid {}",
            tessar_mcp::BUILD_ID,
            std::process::id()
        );
        let projects = Arc::new(Projects::new(memory, clock).with_decision_log(home));
        daemon::run(listener, Arc::clone(&projects), idle_exit, termination()).await;
        // The socket file goes while the journal is still held, so a
        // daemon that takes the journal next never has its own socket
        // removed by this one.
        let _ = std::fs::remove_file(&socket);
        drop(projects);
        Ok::<(), String>(())
    });
    match served {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail(&e),
    }
}

/// Binds `socket`, first removing a stale file left by a daemon that died.
/// The journal lock makes this process the only daemon, so a socket that
/// still answers is a mistake worth reporting.
async fn bind(socket: &Path) -> Result<UnixListener, String> {
    if socket.exists() {
        if UnixStream::connect(socket).await.is_ok() {
            return Err(format!("{} is in use by another process", socket.display()));
        }
        std::fs::remove_file(socket)
            .map_err(|e| format!("cannot remove stale {}: {e}", socket.display()))?;
    }
    let listener = UnixListener::bind(socket)
        .map_err(|e| format!("cannot listen on {}: {e}", socket.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600));
    }
    Ok(listener)
}

/// Resolves on SIGTERM or SIGINT.
async fn termination() {
    use tokio::signal::unix::{SignalKind, signal};
    let (Ok(mut term), Ok(mut int)) = (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
    ) else {
        logln!("tessar: cannot install signal handlers");
        std::future::pending::<()>().await;
        return;
    };
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
}

// ---- tessar daemon stop|status --------------------------------------

/// Runs `work` to completion and prints its message.
fn run_lifecycle(work: impl std::future::Future<Output = Result<String, String>>) -> ExitCode {
    let result = runtime().and_then(|rt| rt.block_on(work));
    match result {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(e) => fail(&e),
    }
}

// ---- tessar purge --------------------------------------------------

/// Redacts each handle through the daemon, then with `purge` erases the
/// content from the files. Every handle is tried; the exit status is
/// failure if any failed.
fn run_redact(home: &Path, handles: &[String], purge: bool) -> ExitCode {
    tessar_mcp::tool_log::set_client_role("cli");
    if let Err(e) = create_home(home) {
        return fail(&e);
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project = project_name(&cwd);
    let runtime = match runtime() {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    let mut failed = false;
    let outcome: Result<(), String> = runtime.block_on(async {
        let opened = proxy::open_current(home, &project, &cwd, DAEMON_WAIT, false).await?;
        let (server, link) = opened.serve().await?;
        for handle in handles {
            match proxy::redact_fact(&server.daemon(), handle).await {
                Ok(line) => println!("{line}"),
                Err(e) => {
                    logln!("tessar: {e}");
                    failed = true;
                }
            }
        }
        drop((server, link));
        Ok(())
    });
    if let Err(e) = outcome {
        return fail(&e);
    }
    if purge && run_purge(home) != ExitCode::SUCCESS {
        return ExitCode::FAILURE;
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Writes a lesson through the daemon, as the `lesson` tool does, and prints
/// the one-line result.
fn run_lesson(home: &Path, text: &str, user: bool) -> ExitCode {
    tessar_mcp::tool_log::set_client_role("cli");
    if let Err(e) = create_home(home) {
        return fail(&e);
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project = project_name(&cwd);
    let runtime = match runtime() {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    let space = if user { "user" } else { "project" };
    let result: Result<String, String> = runtime.block_on(async {
        let opened = proxy::open_current(home, &project, &cwd, DAEMON_WAIT, false).await?;
        let (server, link) = opened.serve().await?;
        let line = proxy::call_tool_text(
            &server.daemon(),
            "lesson",
            serde_json::json!({"text": text, "space": space}),
        )
        .await;
        drop((server, link));
        line
    });
    match result {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => fail(&e),
    }
}

/// Stops the daemon if one runs, rewrites the journal without redacted
/// content and drops the vector cache.
fn run_purge(home: &Path) -> ExitCode {
    let stopped = match runtime().and_then(|rt| rt.block_on(lifecycle::stop(home))) {
        Ok(message) => {
            let stopped = message.starts_with("stopped");
            // Say what happened, unless there was nothing at all to stop.
            if stopped || message.contains("stale") {
                logln!("tessar: {message}");
            }
            stopped
        }
        Err(e) => return fail(&e),
    };
    match purge::purge_home(home) {
        Ok(done) => {
            println!("{}", purge::summary(&done, stopped));
            ExitCode::SUCCESS
        }
        Err(e) => fail(&e),
    }
}

// ---- tessar hook ---------------------------------------------------

/// Runs a hook. Whatever goes wrong, the status is 0 and stdout is empty:
/// memory being down must not block the user's prompt (status 2 would).
fn run_hook(home: &Path, kind: HookKind) -> ExitCode {
    use std::io::{Read, Write};
    if kind == HookKind::Config {
        return match std::env::current_exe() {
            Ok(exe) => {
                print!("{}", hook::config_snippet(&exe));
                ExitCode::SUCCESS
            }
            Err(e) => fail(&format!("cannot find my own binary: {e}")),
        };
    }
    let mut stdin = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut stdin) {
        logln!("tessar hook: cannot read stdin: {e}");
        return ExitCode::SUCCESS;
    }
    tessar_mcp::tool_log::set_client_role("hook");
    let settings = hook::Settings::from_env(|name| std::env::var(name).ok());
    let run = if kind == HookKind::Subagent {
        // Only the stash: no daemon, no runtime.
        Ok(hook::run_subagent(home, &stdin))
    } else {
        runtime().map(|rt| {
            let run = rt.block_on(async {
                match kind {
                    HookKind::Prompt => hook::run_prompt(home, &stdin, &settings).await,
                    HookKind::Agent => hook::run_agent(home, &stdin, &settings).await,
                    _ => hook::run_session(home, &stdin, &settings).await,
                }
            });
            // Do not wait for work the time limit left behind (a `git` that
            // hangs): the process is about to exit.
            rt.shutdown_background();
            run
        })
    };
    match run {
        // The output goes out first; the decision log is written after it.
        Ok(run) => run.finish(home, |text| {
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(text.as_bytes());
            let _ = out.flush();
        }),
        Err(e) => logln!("tessar hook: {e}"),
    }
    ExitCode::SUCCESS
}

// ---- tessar mcp ----------------------------------------------------

fn run_proxy(home: &Path, keep_daemon: bool) -> ExitCode {
    tessar_mcp::tool_log::set_client_role("mcp");
    if let Err(e) = create_home(home) {
        return fail(&e);
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project = project_name(&cwd);
    let runtime = match runtime() {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    let result = runtime.block_on(async {
        let opened = proxy::open_current(home, &project, &cwd, DAEMON_WAIT, keep_daemon).await?;
        let (server, link) = opened.serve().await?;
        logln!("tessar: connected to the daemon (project space `{project}`)");
        let reconnect: proxy::Reconnect = {
            let (home, project, cwd) = (home.to_path_buf(), project.clone(), cwd.clone());
            Arc::new(move || {
                let (home, project, cwd) = (home.clone(), project.clone(), cwd.clone());
                Box::pin(async move {
                    // Not at once: the daemon was most likely replaced, and
                    // the replacement must get the journal first.
                    let launcher = proxy::Launcher::new(DAEMON_WAIT).reconnecting();
                    let opened =
                        proxy::open_current_with(&home, &project, &cwd, keep_daemon, &launcher)
                            .await?;
                    let (server, link) = opened.serve().await?;
                    Ok((server.daemon(), link))
                })
            })
        };
        proxy::bridge_reconnecting(server, link, stdio(), reconnect).await
    });
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => fail("the daemon closed the connection"),
        Err(e) => fail(&e),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_documents_consult_the_cache_state() {
        struct One;
        impl Embedder for One {
            fn embed_documents(&self, t: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
                Ok(t.iter().map(|_| vec![1.0]).collect())
            }
            fn embed_query(&self, _: &str) -> Result<Vec<f32>, EmbedError> {
                Ok(vec![1.0])
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let cache = CachedEmbedder::open(dir.path().join("v.bin"), "fp", Arc::new(One)).unwrap();
        let watch = CacheWatch::new(cache);
        watch.embed_query("q").unwrap();
        assert_eq!(watch.reports.load(Ordering::Relaxed), 0);
        watch.embed_documents(&["d".to_owned()]).unwrap();
        assert_eq!(watch.reports.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn embedder_choice_accepts_candle_only() {
        assert!(choose_backend(Some("bogus")).is_err());
        assert_eq!(choose_backend(None), Ok(Backend::Candle));
        assert_eq!(choose_backend(Some("")), choose_backend(None));
        assert_eq!(choose_backend(Some("candle")), Ok(Backend::Candle));
        assert!(choose_backend(Some("llama")).is_err());
    }
    use super::*;

    fn p(args: &[&str], env: &[(&str, &str)]) -> Result<Cli, String> {
        let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
        let env: Vec<(String, String)> = env
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        parse(&args, |name| {
            env.iter().find(|(k, _)| k == name).map(|(_, v)| v.into())
        })
    }

    #[test]
    fn mcp_flags_and_home() {
        assert_eq!(
            p(&["mcp"], &[("TESSAR_HOME", "/t")]).unwrap(),
            Cli::Mcp {
                home: "/t".into(),
                standalone: false,
                keep_daemon: false
            }
        );
        assert_eq!(
            p(
                &["mcp", "--standalone", "--home", "/x"],
                &[("TESSAR_HOME", "/t")]
            )
            .unwrap(),
            Cli::Mcp {
                home: "/x".into(),
                standalone: true,
                keep_daemon: false
            }
        );
        assert_eq!(
            p(&["mcp"], &[("HOME", "/u")]).unwrap(),
            Cli::Mcp {
                home: "/u/.tessar".into(),
                standalone: false,
                keep_daemon: false
            }
        );
        assert!(p(&["mcp"], &[]).is_err());
        assert!(p(&["mcp", "--idle-exit", "5"], &[("HOME", "/u")]).is_err());
        assert_eq!(
            p(&["mcp", "--keep-daemon"], &[("HOME", "/u")]).unwrap(),
            Cli::Mcp {
                home: "/u/.tessar".into(),
                standalone: false,
                keep_daemon: true
            }
        );
        assert!(p(&["daemon", "--keep-daemon"], &[("HOME", "/u")]).is_err());
    }

    #[test]
    fn redact_takes_handles_and_flags() {
        assert_eq!(
            p(&["redact", "f1", "f22", "--purge", "--home", "/d"], &[]).unwrap(),
            Cli::Redact {
                home: "/d".into(),
                handles: vec!["f1".into(), "f22".into()],
                purge: true
            }
        );
        assert_eq!(
            p(&["redact", "f3"], &[("TESSAR_HOME", "/t")]).unwrap(),
            Cli::Redact {
                home: "/t".into(),
                handles: vec!["f3".into()],
                purge: false
            }
        );
        assert!(p(&["redact"], &[("HOME", "/u")]).is_err());
        assert!(p(&["redact", "--purge"], &[("HOME", "/u")]).is_err());
        assert!(p(&["redact", "f1", "--bogus"], &[("HOME", "/u")]).is_err());
    }

    #[test]
    fn lesson_takes_a_text_and_a_space() {
        assert_eq!(
            p(&["lesson", "Before X, read Y", "--home", "/d"], &[]).unwrap(),
            Cli::Lesson {
                home: "/d".into(),
                text: "Before X, read Y".into(),
                user: false
            }
        );
        assert_eq!(
            p(
                &["lesson", "--space", "user", "T"],
                &[("TESSAR_HOME", "/t")]
            )
            .unwrap(),
            Cli::Lesson {
                home: "/t".into(),
                text: "T".into(),
                user: true
            }
        );
        assert_eq!(
            p(&["lesson", "T", "--space=project"], &[("HOME", "/u")]).unwrap(),
            Cli::Lesson {
                home: "/u/.tessar".into(),
                text: "T".into(),
                user: false
            }
        );
        assert!(p(&["lesson"], &[("HOME", "/u")]).is_err());
        assert!(p(&["lesson", "a", "b"], &[("HOME", "/u")]).is_err());
        assert!(p(&["lesson", "T", "--space", "all"], &[("HOME", "/u")]).is_err());
        assert!(p(&["lesson", "T", "--bogus"], &[("HOME", "/u")]).is_err());
    }

    #[test]
    fn purge_takes_a_home() {
        assert_eq!(
            p(&["purge", "--home", "/d"], &[]).unwrap(),
            Cli::Purge { home: "/d".into() }
        );
        assert_eq!(
            p(&["purge"], &[("TESSAR_HOME", "/t")]).unwrap(),
            Cli::Purge { home: "/t".into() }
        );
        assert!(p(&["purge", "--standalone"], &[("HOME", "/u")]).is_err());
    }

    #[test]
    fn daemon_stop_status_and_version() {
        assert_eq!(
            p(&["daemon", "stop", "--home", "/d"], &[]).unwrap(),
            Cli::DaemonStop { home: "/d".into() }
        );
        assert_eq!(
            p(&["daemon", "status"], &[("TESSAR_HOME", "/t")]).unwrap(),
            Cli::DaemonStatus { home: "/t".into() }
        );
        assert!(p(&["daemon", "stop", "--idle-exit", "5"], &[("HOME", "/u")]).is_err());
        assert_eq!(p(&["--version"], &[]).unwrap(), Cli::Version);
        assert!(!tessar_mcp::BUILD_ID.is_empty());
        assert!(tessar_mcp::BUILD_ID.starts_with(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn daemon_flags() {
        assert_eq!(
            p(&["daemon", "--home=/d", "--idle-exit", "30"], &[]).unwrap(),
            Cli::Daemon {
                home: "/d".into(),
                idle_exit: Some(Duration::from_secs(1800)),
                takeover: None
            }
        );
        assert_eq!(
            p(&["daemon", "--home", "/d", "--takeover", "4242"], &[]).unwrap(),
            Cli::Daemon {
                home: "/d".into(),
                idle_exit: None,
                takeover: Some(4242)
            }
        );
        assert_eq!(
            p(&["daemon", "--takeover=4242"], &[("HOME", "/u")]).unwrap(),
            Cli::Daemon {
                home: "/u/.tessar".into(),
                idle_exit: None,
                takeover: Some(4242)
            }
        );
        // A pid is required, and it must be one that may be signalled.
        for bad in [
            &["daemon", "--takeover"][..],
            &["daemon", "--takeover", "1"],
            &["daemon", "--takeover=0"],
            &["daemon", "--takeover", "--home"],
        ] {
            assert!(p(bad, &[("HOME", "/u")]).is_err(), "{bad:?}");
        }
        assert!(p(&["mcp", "--takeover", "4242"], &[("HOME", "/u")]).is_err());
        assert!(p(&["daemon", "stop", "--takeover", "4242"], &[("HOME", "/u")]).is_err());
        assert_eq!(
            p(&["daemon"], &[("HOME", "/u")]).unwrap(),
            Cli::Daemon {
                home: "/u/.tessar".into(),
                idle_exit: None,
                takeover: None
            }
        );
        assert!(p(&["daemon", "--idle-exit", "0"], &[("HOME", "/u")]).is_err());
        assert!(p(&["daemon", "--idle-exit", "soon"], &[("HOME", "/u")]).is_err());
        assert!(p(&["daemon", "--standalone"], &[("HOME", "/u")]).is_err());
    }

    #[test]
    fn hook_commands() {
        assert_eq!(
            p(&["hook", "prompt"], &[("HOME", "/u")]).unwrap(),
            Cli::Hook {
                home: "/u/.tessar".into(),
                kind: HookKind::Prompt
            }
        );
        assert_eq!(
            p(&["hook", "session", "--home", "/h"], &[]).unwrap(),
            Cli::Hook {
                home: "/h".into(),
                kind: HookKind::Session
            }
        );
        // The snippet works without a home.
        assert!(matches!(
            p(&["hook", "config"], &[]).unwrap(),
            Cli::Hook {
                kind: HookKind::Config,
                ..
            }
        ));
        assert!(matches!(
            p(&["hook", "agent"], &[("HOME", "/u")]).unwrap(),
            Cli::Hook {
                kind: HookKind::Agent,
                ..
            }
        ));
        assert!(matches!(
            p(&["hook", "subagent"], &[("HOME", "/u")]).unwrap(),
            Cli::Hook {
                kind: HookKind::Subagent,
                ..
            }
        ));
        assert!(p(&["hook"], &[("HOME", "/u")]).is_err());
        assert!(p(&["hook", "other"], &[("HOME", "/u")]).is_err());
        assert!(p(&["hook", "prompt"], &[]).is_err());
    }

    #[test]
    fn help_and_unknown() {
        assert_eq!(p(&[], &[]).unwrap(), Cli::Help);
        assert_eq!(p(&["--help"], &[]).unwrap(), Cli::Help);
        assert!(p(&["serve"], &[]).is_err());
    }
}
