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
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::transport::stdio;
use tessar_core::clock::{Clock, SystemClock};
use tessar_core::embed::Embedder;
use tessar_core::log::LogError;
use tessar_core::store::{Memory, Op, Semantic};
use tessar_embed_cache::CachedEmbedder;
use tessar_journal::FileJournal;
use tessar_mcp::daemon::{self, Projects, socket_path};
use tessar_mcp::hook;
use tessar_mcp::lifecycle;
use tessar_mcp::project::project_name;
use tessar_mcp::proxy;
use tessar_mcp::purge;
use tessar_mcp::server::TessarServer;
use tokio::net::{UnixListener, UnixStream};

#[cfg(not(any(feature = "candle", feature = "llama")))]
compile_error!("enable at least one embedder feature of tessar-mcp: `candle` (default) or `llama`");

const USAGE: &str = "\
tessar: a memory engine for AI agents

USAGE:
    tessar mcp [--home DIR] [--standalone] [--keep-daemon]
                               serve the memory as an MCP server on stdio. By
                               default a proxy to the daemon, which is started
                               when none runs and restarted when it is another
                               build than this binary (--keep-daemon: never
                               restart it); --standalone serves in process
    tessar daemon [--home DIR] [--idle-exit MINUTES]
                               hold the memory and the model and serve every
                               session on DIR/tessar.sock
    tessar daemon stop|status [--home DIR]
                               stop the daemon (SIGTERM to the pid in
                               DIR/tessar.pid), or print its build, pid, uptime
                               and open sessions
    tessar redact fNN... [--purge] [--home DIR]
                               destroy the content of facts by handle (starts
                               the daemon if needed); with --purge also erase
                               it from the journal file, as `tessar purge`
    tessar purge [--home DIR]  erase redacted content from the journal file and
                               drop the vector cache. Stops the daemon first;
                               the next session starts it again
    tessar hook prompt|session [--home DIR]
                               Claude Code hooks: read the hook JSON on stdin,
                               print context for the model on stdout
    tessar hook config         print the hooks snippet for ~/.claude/settings.json
    tessar --version           print the build id
    tessar --help              show this text

The memory lives in DIR (default: $TESSAR_HOME, else ~/.tessar): journal.jsonl,
tessar.sock, tessar.pid and daemon.log, and the embedding model in
models/voyage-4-nano-Q8_0.gguf (and models/voyage-4-nano/tokenizer.json for
the candle embedder). Set TESSAR_MODEL or TESSAR_TOKENIZER to use files from
elsewhere. TESSAR_EMBEDDER=candle|llama picks the embedder (default: candle,
else llama, whichever this build has).";

/// Vector length in the index.
const INDEX_DIMS: usize = 1024;

/// The model file name under `<home>/models/`.
const MODEL_FILE: &str = "voyage-4-nano-Q8_0.gguf";

/// The model's name in the vector cache's key and path.
const MODEL_NAME: &str = "voyage-4-nano";

/// Where the vectors of `model` are cached under `home`.
fn vector_cache_path(home: &Path, model: &str) -> PathBuf {
    home.join("index").join(model).join("vectors.bin")
}

/// Wraps `inner` in the on-disk vector cache. The cache is derived data,
/// so any problem with it is reported and the plain embedder is used.
fn cached(home: &Path, inner: Arc<dyn Embedder>) -> Arc<dyn Embedder> {
    let path = vector_cache_path(home, MODEL_NAME);
    let opened = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .map_err(|e| format!("cannot create the cache directory: {e}"))
        .and_then(|()| {
            CachedEmbedder::open(&path, MODEL_NAME, Arc::clone(&inner)).map_err(|e| e.to_string())
        });
    match opened {
        Ok(cache) => {
            eprintln!(
                "tessar: vector cache {}: {} vectors",
                path.display(),
                cache.len()
            );
            Arc::new(cache)
        }
        Err(e) => {
            eprintln!(
                "tessar: vector cache {} unusable, embedding without it: {e}",
                path.display()
            );
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
    Version,
    Help,
}

/// Which hook.
#[derive(Debug, Clone, Copy, PartialEq)]
enum HookKind {
    Prompt,
    Session,
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
        Ok(Cli::Daemon { home, idle_exit }) => run_daemon(&home, idle_exit),
        Ok(Cli::Purge { home }) => run_purge(&home),
        Ok(Cli::Redact {
            home,
            handles,
            purge,
        }) => run_redact(&home, &handles, purge),
        Ok(Cli::Hook { home, kind }) => run_hook(&home, kind),
        Ok(Cli::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(e) => fail(&e),
    }
}

fn fail(message: &str) -> ExitCode {
    eprintln!("tessar: {message}");
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
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--home" => {
                home = Some(PathBuf::from(it.next().ok_or("--home needs a directory")?));
            }
            a if a.starts_with("--home=") => home = Some(PathBuf::from(&a["--home=".len()..])),
            "--standalone" if command == "mcp" => standalone = true,
            "--keep-daemon" if command == "mcp" => keep_daemon = true,
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
        _ => Cli::Daemon { home, idle_exit },
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
        Some("config") => HookKind::Config,
        _ => return Err(format!("`hook` wants prompt, session or config\n\n{USAGE}")),
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
#[cfg(feature = "candle")]
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
    Llama,
}

/// `$TESSAR_EMBEDDER` (`candle` or `llama`), else candle if compiled in,
/// else llama. Asking for a backend this build lacks is an error.
fn choose_backend(requested: Option<&str>) -> Result<Backend, String> {
    let compiled = |b: Backend| match b {
        Backend::Candle => cfg!(feature = "candle"),
        Backend::Llama => cfg!(feature = "llama"),
    };
    match requested.map(str::trim).filter(|r| !r.is_empty()) {
        None => Ok(if compiled(Backend::Candle) {
            Backend::Candle
        } else {
            Backend::Llama
        }),
        Some("candle") if compiled(Backend::Candle) => Ok(Backend::Candle),
        Some("llama") if compiled(Backend::Llama) => Ok(Backend::Llama),
        Some(f @ ("candle" | "llama")) => Err(format!(
            "TESSAR_EMBEDDER={f} is not compiled into this binary; rebuild with \
             `cargo build --release -p tessar-mcp --features {f}` \
             (add `--no-default-features` to drop the other embedder)"
        )),
        Some(other) => Err(format!(
            "TESSAR_EMBEDDER={other:?} is unknown; use `candle` or `llama`"
        )),
    }
}

/// A loaded embedder with what the logs say about it.
struct Loaded {
    embedder: Arc<dyn Embedder>,
    dims: usize,
    description: String,
}

#[cfg(feature = "candle")]
fn load_candle(home: &Path, model: &Path) -> Result<Loaded, String> {
    let tokenizer = tokenizer_path(home)?;
    let e = tessar_embed_candle::CandleEmbedder::voyage_nano_gguf(model, &tokenizer)
        .map_err(|e| format!("cannot load the model: {e}"))?;
    let device = if e.is_metal() { "Metal" } else { "CPU" };
    Ok(Loaded {
        dims: e.dims(),
        description: format!("candle ({device})"),
        embedder: Arc::new(e),
    })
}

#[cfg(not(feature = "candle"))]
fn load_candle(_: &Path, _: &Path) -> Result<Loaded, String> {
    unreachable!("choose_backend refuses candle when it is not compiled")
}

#[cfg(feature = "llama")]
fn load_llama(model: &Path) -> Result<Loaded, String> {
    let e = tessar_embed_llama::LlamaEmbedder::voyage_nano(model)
        .map_err(|e| format!("cannot load the model: {e}"))?;
    Ok(Loaded {
        dims: e.dims(),
        description: "llama.cpp".to_string(),
        embedder: Arc::new(e),
    })
}

#[cfg(not(feature = "llama"))]
fn load_llama(_: &Path) -> Result<Loaded, String> {
    unreachable!("choose_backend refuses llama when it is not compiled")
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
        Backend::Llama => load_llama(&model)?,
    };
    eprintln!("tessar: embedder {}", loaded.description);
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
        eprintln!("tessar: embedder warm-up failed: {e}");
    }
    eprintln!(
        "tessar: embedder warm-up took {} ms",
        started.elapsed().as_millis()
    );
    let embedder = cached(home, loaded.embedder);
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
    std::fs::create_dir_all(home).map_err(|e| format!("cannot create {}: {e}", home.display()))
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
    eprintln!(
        "tessar: serving {} (project space `{project}`)",
        home.display()
    );
    let runtime = match runtime() {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    let served = runtime.block_on(async {
        let service = TessarServer::shared(api)
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

fn run_daemon(home: &Path, idle_exit: Option<Duration>) -> ExitCode {
    if let Err(e) = create_home(home) {
        return fail(&e);
    }
    let journal = match open_journal(home) {
        Ok(j) => j,
        Err(LogError::Locked) => {
            eprintln!("tessar: a tessar daemon is already running");
            return ExitCode::SUCCESS;
        }
        Err(e) => return fail(&format!("cannot open the journal: {e}")),
    };
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
        let listener = bind(&socket).await?;
        eprintln!(
            "tessar: daemon serving {} on {}",
            home.display(),
            socket.display()
        );
        if let Err(e) = lifecycle::write_pid_file(home) {
            eprintln!("tessar: cannot write the pid file: {e}");
        }
        eprintln!(
            "tessar: build {}, pid {}",
            tessar_mcp::BUILD_ID,
            std::process::id()
        );
        let projects = Arc::new(Projects::new(memory, clock));
        daemon::run(listener, projects, idle_exit, termination()).await;
        Ok::<(), String>(())
    });
    let _ = std::fs::remove_file(&socket);
    lifecycle::remove_pid_file(home);
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
        eprintln!("tessar: cannot install signal handlers");
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
            match proxy::redact_fact(server.daemon(), handle).await {
                Ok(line) => println!("{line}"),
                Err(e) => {
                    eprintln!("tessar: {e}");
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

/// Stops the daemon if one runs, rewrites the journal without redacted
/// content and drops the vector cache.
fn run_purge(home: &Path) -> ExitCode {
    let stopped = match runtime().and_then(|rt| {
        let running = lifecycle::read_pid_file(home).is_some();
        let message = rt.block_on(lifecycle::stop(home))?;
        Ok((running, message))
    }) {
        Ok((running, message)) => {
            if running {
                eprintln!("tessar: {message}");
            }
            running && message.starts_with("stopped")
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
        eprintln!("tessar hook: cannot read stdin: {e}");
        return ExitCode::SUCCESS;
    }
    let settings = hook::Settings::from_env(|name| std::env::var(name).ok());
    let result = runtime().and_then(|rt| {
        rt.block_on(async {
            match kind {
                HookKind::Prompt => hook::run_prompt(home, &stdin, &settings).await,
                _ => hook::run_session(home, &stdin, &settings).await,
            }
        })
    });
    match result {
        Ok(text) => {
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(text.as_bytes());
            let _ = out.flush();
        }
        Err(e) => eprintln!("tessar hook: {e}"),
    }
    ExitCode::SUCCESS
}

// ---- tessar mcp ----------------------------------------------------

fn run_proxy(home: &Path, keep_daemon: bool) -> ExitCode {
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
        eprintln!("tessar: connected to the daemon (project space `{project}`)");
        proxy::bridge(server, link, stdio()).await
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
    fn vector_cache_lives_under_the_model_in_the_home() {
        assert_eq!(
            vector_cache_path(Path::new("/h/.tessar"), "voyage-4-nano"),
            Path::new("/h/.tessar/index/voyage-4-nano/vectors.bin")
        );
    }

    #[test]
    fn embedder_choice_follows_the_compiled_features() {
        assert!(choose_backend(Some("bogus")).is_err());
        assert!(choose_backend(None).is_ok());
        assert_eq!(choose_backend(Some("")), choose_backend(None));
        let candle = choose_backend(Some("candle"));
        assert_eq!(candle.is_ok(), cfg!(feature = "candle"));
        if let Err(e) = candle {
            assert!(e.contains("--features candle"), "{e}");
        }
        let llama = choose_backend(Some("llama"));
        assert_eq!(llama.is_ok(), cfg!(feature = "llama"));
        if let Err(e) = llama {
            assert!(e.contains("--features llama"), "{e}");
        }
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
                idle_exit: Some(Duration::from_secs(1800))
            }
        );
        assert_eq!(
            p(&["daemon"], &[("HOME", "/u")]).unwrap(),
            Cli::Daemon {
                home: "/u/.tessar".into(),
                idle_exit: None
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
