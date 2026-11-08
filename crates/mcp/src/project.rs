//! Which project a session belongs to, from the directory it runs in.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The repository name from a remote URL: the last path component without
/// `.git`, lowercased. Handles `https://host/owner/repo.git`, the scp form
/// `git@host:owner/repo.git` and a trailing slash.
pub fn repo_name_from_url(url: &str) -> Option<String> {
    let url = url.trim().trim_end_matches('/');
    let url = url
        .strip_suffix(".git")
        .unwrap_or(url)
        .trim_end_matches('/');
    let name = url.rsplit(['/', ':']).next()?.trim();
    (!name.is_empty()).then(|| name.to_lowercase())
}

/// The repository name for `cwd`, lowercased. It comes from the `origin`
/// remote, so a second clone of a repository shares its space. Without an
/// origin it is the name of the main checkout (the parent of `git rev-parse
/// --git-common-dir`, so all worktrees agree), and outside git the name of
/// `cwd` itself.
pub fn project_name(cwd: &Path) -> String {
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .args(args)
            .current_dir(cwd)
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
            git_dir.parent().map(Path::to_path_buf)
        }
        Some(git_dir) => Some(git_dir),
        None => Some(cwd.to_path_buf()),
    };
    dir.as_deref()
        .and_then(Path::file_name)
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "project".to_owned())
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
