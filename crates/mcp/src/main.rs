//! `tessar mcp`: the memory as an MCP server on stdio.
//!
//! Stdout carries the protocol and nothing else. Everything meant for a
//! human goes to stderr.

use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::sync::Arc;

use rmcp::ServiceExt;
use rmcp::transport::stdio;
use tessar_core::clock::SystemClock;
use tessar_core::log::LogError;
use tessar_core::store::{Memory, Op, Semantic};
use tessar_embed_llama::{LlamaEmbedder, voyage_nano_path};
use tessar_journal::FileJournal;
use tessar_mcp::api::Tessar;
use tessar_mcp::server::TessarServer;

const USAGE: &str = "\
tessar: a memory engine for AI agents

USAGE:
    tessar mcp [--home DIR]    serve the memory as an MCP server on stdio
    tessar --help              show this text

The memory lives in DIR (default: $TESSAR_HOME, else ~/.tessar): journal.jsonl,
and the embedding model in models/voyage-4-nano-Q8_0.gguf. Set TESSAR_MODEL to
use a model file from elsewhere.";

/// Vector length in the index.
const INDEX_DIMS: usize = 1024;

/// The model file name under `<home>/models/`.
const MODEL_FILE: &str = "voyage-4-nano-Q8_0.gguf";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("mcp") => match parse_mcp(&args[1..]) {
            Ok(home) => run(home),
            Err(e) => fail(&e),
        },
        Some("--help" | "-h" | "help") | None => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(other) => fail(&format!("unknown command `{other}`\n\n{USAGE}")),
    }
}

fn fail(message: &str) -> ExitCode {
    eprintln!("tessar: {message}");
    ExitCode::FAILURE
}

/// The home directory from `--home DIR`, `$TESSAR_HOME` or `~/.tessar`.
fn parse_mcp(args: &[String]) -> Result<PathBuf, String> {
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
    if let Some(home) = home {
        return Ok(home);
    }
    if let Some(home) = std::env::var_os("TESSAR_HOME").filter(|h| !h.is_empty()) {
        return Ok(PathBuf::from(home));
    }
    let user_home = std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .ok_or("HOME is not set; pass --home DIR or set TESSAR_HOME")?;
    Ok(PathBuf::from(user_home).join(".tessar"))
}

/// The model: `$TESSAR_MODEL`, else `<home>/models/…`, else the default
/// `~/.tessar/models/…`.
fn model_path(home: &std::path::Path) -> Result<PathBuf, String> {
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

/// The repository name from a remote URL: the last path component without
/// `.git`, lowercased. Handles `https://host/owner/repo.git`, the scp form
/// `git@host:owner/repo.git` and a trailing slash.
fn repo_name_from_url(url: &str) -> Option<String> {
    let url = url.trim().trim_end_matches('/');
    let url = url
        .strip_suffix(".git")
        .unwrap_or(url)
        .trim_end_matches('/');
    let name = url.rsplit(['/', ':']).next()?.trim();
    (!name.is_empty()).then(|| name.to_lowercase())
}

/// The repository name, lowercased. It comes from the `origin` remote, so a
/// second clone of a repository shares its space. Without an origin it is
/// the name of the main checkout (the parent of `git rev-parse
/// --git-common-dir`, so all worktrees agree), and outside git the name of
/// the current directory.
fn project_name() -> String {
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .args(args)
            .stderr(std::process::Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())?;
        String::from_utf8(out.stdout)
            .ok()
            .map(|s| s.trim().to_owned())
    };
    if let Some(name) = git(&["remote", "get-url", "origin"]).and_then(|u| repo_name_from_url(&u)) {
        return name;
    }
    let common = git(&["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .map(PathBuf::from)
        .filter(|p| p.is_absolute());
    let dir = match common {
        Some(git_dir) if git_dir.file_name().is_some_and(|n| n == ".git") => {
            git_dir.parent().map(std::path::Path::to_path_buf)
        }
        Some(git_dir) => Some(git_dir),
        None => std::env::current_dir().ok(),
    };
    dir.as_deref()
        .and_then(std::path::Path::file_name)
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "project".to_owned())
}

fn run(home: PathBuf) -> ExitCode {
    if let Err(e) = std::fs::create_dir_all(&home) {
        return fail(&format!("cannot create {}: {e}", home.display()));
    }
    // The journal first: a second instance must fail before it loads the
    // model onto the GPU.
    let journal_path = home.join("journal.jsonl");
    let journal = match FileJournal::<Op>::open(&journal_path) {
        Ok(j) => j,
        Err(LogError::Locked) => {
            return fail(&format!(
                "another tessar is using {}",
                journal_path.display()
            ));
        }
        Err(e) => return fail(&format!("cannot open {}: {e}", journal_path.display())),
    };
    let model = match model_path(&home) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let embedder = match LlamaEmbedder::voyage_nano(&model) {
        Ok(e) => Arc::new(e),
        Err(e) => return fail(&format!("cannot load the model: {e}")),
    };
    // The index holds 1024 components per vector; a shorter model cannot fill it.
    if embedder.dims() < INDEX_DIMS {
        return fail(&format!(
            "the model at {} gives {} dimensions per vector, at least {INDEX_DIMS} are needed",
            model.display(),
            embedder.dims()
        ));
    }
    let clock = Arc::new(SystemClock);
    let memory = match Memory::open_with(
        journal,
        clock.clone(),
        Semantic {
            embedder,
            dims: INDEX_DIMS,
            near_bonus: 0.05,
            predicate_threshold: 0.70,
        },
    ) {
        Ok(m) => m,
        Err(e) => return fail(&format!("cannot open the memory: {e}")),
    };
    let project = project_name();
    let api = match Tessar::bootstrap(memory, clock, &project, &|w| eprintln!("tessar: {w}")) {
        Ok(a) => a,
        Err(e) => return fail(&format!("cannot prepare the memory: {e}")),
    };
    eprintln!(
        "tessar: serving {} (project space `{project}`)",
        home.display()
    );

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => return fail(&format!("cannot start the runtime: {e}")),
    };
    let served = runtime.block_on(async {
        let service = TessarServer::new(api)
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

#[cfg(test)]
mod tests {
    use super::repo_name_from_url as name;

    #[test]
    fn repository_name_from_remote_urls() {
        assert_eq!(
            name("https://github.com/ab7tr4ct/tessar.git").as_deref(),
            Some("tessar")
        );
        assert_eq!(
            name("https://github.com/ab7tr4ct/Tessar").as_deref(),
            Some("tessar")
        );
        assert_eq!(
            name("https://github.com/ab7tr4ct/tessar/").as_deref(),
            Some("tessar")
        );
        assert_eq!(
            name("https://github.com/ab7tr4ct/tessar.git/").as_deref(),
            Some("tessar")
        );
        assert_eq!(
            name("git@github.com:ab7tr4ct/tessar.git").as_deref(),
            Some("tessar")
        );
        assert_eq!(
            name("git@github.com:tessar.git\n").as_deref(),
            Some("tessar")
        );
        assert_eq!(
            name("ssh://git@host:2222/o/Repo.git").as_deref(),
            Some("repo")
        );
        assert_eq!(name("/srv/git/tessar.git").as_deref(), Some("tessar"));
        assert_eq!(name(""), None);
        assert_eq!(name("https://github.com/"), Some("github.com".to_owned()));
    }
}
