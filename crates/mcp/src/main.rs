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
use std::process::{Command, ExitCode, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rmcp::ServiceExt;
use rmcp::transport::stdio;
use tessar_core::clock::{Clock, SystemClock};
use tessar_core::log::LogError;
use tessar_core::store::{Memory, Op, Semantic};
use tessar_embed_llama::{LlamaEmbedder, voyage_nano_path};
use tessar_journal::FileJournal;
use tessar_mcp::daemon::{self, LOG_FILE, Projects, socket_path};
use tessar_mcp::project::project_name;
use tessar_mcp::proxy;
use tessar_mcp::server::TessarServer;
use tokio::net::{UnixListener, UnixStream};

const USAGE: &str = "\
tessar: a memory engine for AI agents

USAGE:
    tessar mcp [--home DIR] [--standalone]
                               serve the memory as an MCP server on stdio. By
                               default a proxy to the daemon, which is started
                               when none runs; --standalone serves in process
    tessar daemon [--home DIR] [--idle-exit MINUTES]
                               hold the memory and the model and serve every
                               session on DIR/tessar.sock
    tessar --help              show this text

The memory lives in DIR (default: $TESSAR_HOME, else ~/.tessar): journal.jsonl,
tessar.sock and daemon.log, and the embedding model in
models/voyage-4-nano-Q8_0.gguf. Set TESSAR_MODEL to use a model file from
elsewhere.";

/// Vector length in the index.
const INDEX_DIMS: usize = 1024;

/// The model file name under `<home>/models/`.
const MODEL_FILE: &str = "voyage-4-nano-Q8_0.gguf";

/// How long the proxy waits for a freshly started daemon (the model loads).
const DAEMON_WAIT: Duration = Duration::from_secs(60);
const DAEMON_POLL: Duration = Duration::from_millis(200);

/// What the command line asked for.
#[derive(Debug, PartialEq)]
enum Cli {
    Mcp {
        home: PathBuf,
        standalone: bool,
    },
    Daemon {
        home: PathBuf,
        idle_exit: Option<Duration>,
    },
    Help,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse(&args, |name| std::env::var_os(name)) {
        Ok(Cli::Mcp {
            home,
            standalone: false,
        }) => run_proxy(&home),
        Ok(Cli::Mcp {
            home,
            standalone: true,
        }) => run_standalone(&home),
        Ok(Cli::Daemon { home, idle_exit }) => run_daemon(&home, idle_exit),
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
    if !matches!(command, "mcp" | "daemon") {
        return Err(format!("unknown command `{command}`\n\n{USAGE}"));
    }
    let mut home = None;
    let mut standalone = false;
    let mut idle_exit = None;
    let mut it = args[1..].iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--home" => {
                home = Some(PathBuf::from(it.next().ok_or("--home needs a directory")?));
            }
            a if a.starts_with("--home=") => home = Some(PathBuf::from(&a["--home=".len()..])),
            "--standalone" if command == "mcp" => standalone = true,
            "--idle-exit" if command == "daemon" => {
                idle_exit = Some(minutes(it.next().ok_or("--idle-exit needs minutes")?)?);
            }
            a if a.starts_with("--idle-exit=") && command == "daemon" => {
                idle_exit = Some(minutes(&a["--idle-exit=".len()..])?);
            }
            other => return Err(format!("unexpected argument `{other}`\n\n{USAGE}")),
        }
    }
    let home = match home {
        Some(home) => home,
        None => default_home(env)?,
    };
    Ok(if command == "mcp" {
        Cli::Mcp { home, standalone }
    } else {
        Cli::Daemon { home, idle_exit }
    })
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

/// The model: `$TESSAR_MODEL`, else `<home>/models/…`, else the default
/// `~/.tessar/models/…`.
fn model_path(home: &Path) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("TESSAR_MODEL").filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    let local = home.join("models").join(MODEL_FILE);
    if local.is_file() {
        return Ok(local);
    }
    match voyage_nano_path() {
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
    let embedder = LlamaEmbedder::voyage_nano(&model)
        .map(Arc::new)
        .map_err(|e| format!("cannot load the model: {e}"))?;
    // The index holds 1024 components per vector; a shorter model cannot fill it.
    if embedder.dims() < INDEX_DIMS {
        return Err(format!(
            "the model at {} gives {} dimensions per vector, at least {INDEX_DIMS} are needed",
            model.display(),
            embedder.dims()
        ));
    }
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
        let projects = Arc::new(Projects::new(memory, clock));
        daemon::run(listener, projects, idle_exit, termination()).await;
        Ok::<(), String>(())
    });
    let _ = std::fs::remove_file(&socket);
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

// ---- tessar mcp ----------------------------------------------------

fn run_proxy(home: &Path) -> ExitCode {
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
        let stream = connect_or_start(home).await?;
        let (server, link) = proxy::connect(stream, &project, &cwd).await?;
        eprintln!("tessar: connected to the daemon (project space `{project}`)");
        proxy::bridge(server, link, stdio()).await
    });
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => fail("the daemon closed the connection"),
        Err(e) => fail(&e),
    }
}

/// Connects to the daemon's socket, starting a daemon first if none answers.
async fn connect_or_start(home: &Path) -> Result<UnixStream, String> {
    let socket = socket_path(home);
    if let Ok(stream) = UnixStream::connect(&socket).await {
        return Ok(stream);
    }
    let log = home.join(LOG_FILE);
    let mut child = spawn_daemon(home, &log)?;
    eprintln!("tessar: starting the daemon (log: {})", log.display());
    let started = Instant::now();
    loop {
        tokio::time::sleep(DAEMON_POLL).await;
        if let Ok(stream) = UnixStream::connect(&socket).await {
            return Ok(stream);
        }
        // A daemon that failed (no model, say) will never listen. One that
        // exited cleanly found another daemon holding the journal, which
        // is about to listen.
        if let Ok(Some(status)) = child.try_wait()
            && !status.success()
        {
            return Err(format!(
                "the daemon exited with {status}; see {}",
                log.display()
            ));
        }
        if started.elapsed() > DAEMON_WAIT {
            return Err(format!(
                "no daemon answered on {} within {}s; see {}",
                socket.display(),
                DAEMON_WAIT.as_secs(),
                log.display()
            ));
        }
    }
}

/// Starts `tessar daemon` in its own process group, so it outlives this
/// proxy and the terminal's signals, with its output appended to `log`.
fn spawn_daemon(home: &Path, log: &Path) -> Result<std::process::Child, String> {
    use std::os::unix::process::CommandExt;
    let exe = std::env::current_exe().map_err(|e| format!("cannot find my own binary: {e}"))?;
    let out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .map_err(|e| format!("cannot open {}: {e}", log.display()))?;
    let err = out
        .try_clone()
        .map_err(|e| format!("cannot open {}: {e}", log.display()))?;
    Command::new(exe)
        .arg("daemon")
        .arg("--home")
        .arg(home)
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .spawn()
        .map_err(|e| format!("cannot start the daemon: {e}"))
}

#[cfg(test)]
mod tests {
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
                standalone: false
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
                standalone: true
            }
        );
        assert_eq!(
            p(&["mcp"], &[("HOME", "/u")]).unwrap(),
            Cli::Mcp {
                home: "/u/.tessar".into(),
                standalone: false
            }
        );
        assert!(p(&["mcp"], &[]).is_err());
        assert!(p(&["mcp", "--idle-exit", "5"], &[("HOME", "/u")]).is_err());
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
    fn help_and_unknown() {
        assert_eq!(p(&[], &[]).unwrap(), Cli::Help);
        assert_eq!(p(&["--help"], &[]).unwrap(), Cli::Help);
        assert!(p(&["serve"], &[]).is_err());
    }
}
