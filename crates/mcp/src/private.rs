//! Owner-only files and directories under the tessar home (directories
//! `0700`, files `0600`).
//!
//! New ones are created with that mode. Older builds created them wider, so
//! what is tessar's own is tightened when it is used: only group and other
//! bits are ever removed, and a failure (the file is not ours) is ignored.
//! On other than Unix systems all of this does nothing.
//!
//! The home may be any existing directory (`--home .`, `TESSAR_HOME=$HOME`),
//! so [`create_home`] tightens it only when it is recognisably tessar's.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use crate::purge::journal_path;

/// Options for a file that is created `0600` if it does not exist. An
/// existing file keeps its mode; follow up with [`tighten_file`].
pub fn file_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

/// Creates `dir` and its missing parents `0700`. An existing directory is
/// left as it is.
pub(crate) fn create_dir_all(dir: &Path) -> io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)
}

/// Creates tessar's own subdirectory `dir` of `home` (for example `hooks/`)
/// `0700`. An existing one is tightened only when `home` is recognised as
/// tessar's, by the same rule as [`create_home`]: with `--home .` in a
/// repository that has its own `hooks/`, that directory is not ours.
pub fn create_own_dir_all(home: &Path, dir: &Path) -> io::Result<()> {
    create_dir_all(dir)?;
    if is_tessars(home) {
        tighten_path(dir);
    }
    Ok(())
}

/// Creates the home directory `0700`. An existing home is tightened only
/// if it already holds tessar's journal and is not the user's home
/// directory itself, so a foreign directory is never changed.
pub fn create_home(home: &Path) -> io::Result<()> {
    let user_home = std::env::var_os("HOME").filter(|h| !h.is_empty());
    create_home_for(home, user_home.as_deref().map(Path::new))
}

fn create_home_for(home: &Path, user_home: Option<&Path>) -> io::Result<()> {
    create_dir_all(home)?;
    if is_tessars_for(home, user_home) {
        tighten_path(home);
    }
    Ok(())
}

fn is_tessars(home: &Path) -> bool {
    let user_home = std::env::var_os("HOME").filter(|h| !h.is_empty());
    is_tessars_for(home, user_home.as_deref().map(Path::new))
}

fn is_tessars_for(home: &Path, user_home: Option<&Path>) -> bool {
    journal_path(home).is_file() && !is_user_home(home, user_home)
}

fn is_user_home(dir: &Path, user_home: Option<&Path>) -> bool {
    let Some(user_home) = user_home else {
        return false;
    };
    match (dir.canonicalize(), user_home.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => dir == user_home,
    }
}

/// Removes group and other access from `file`, if it has any.
pub fn tighten_file(file: &File) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = file.metadata() {
            let mode = meta.permissions().mode();
            if mode & 0o077 != 0 {
                let _ = file.set_permissions(std::fs::Permissions::from_mode(mode & !0o077));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = file;
}

fn tighten_path(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            let mode = meta.permissions().mode();
            if mode & 0o077 != 0 {
                let _ =
                    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode & !0o077));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    fn set(path: &Path, mode: u32) {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    // Shows the creation mode only when the umask leaves group and other
    // bits free; the tightening tests prove the rest.
    #[test]
    fn a_new_home_is_private() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("new").join(".tessar");
        create_home_for(&home, None).unwrap();
        assert_eq!(mode(&home), 0o700);
        assert_eq!(mode(&root.path().join("new")), 0o700);
    }

    #[test]
    fn an_existing_directory_without_a_journal_is_left_alone() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("project");
        std::fs::create_dir(&dir).unwrap();
        set(&dir, 0o755);
        create_home_for(&dir, None).unwrap();
        assert_eq!(mode(&dir), 0o755);
    }

    #[test]
    fn an_existing_home_with_a_journal_is_tightened() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join(".tessar");
        std::fs::create_dir(&home).unwrap();
        std::fs::write(journal_path(&home), "").unwrap();
        set(&home, 0o755);
        create_home_for(&home, Some(root.path())).unwrap();
        assert_eq!(mode(&home), 0o700);
    }

    #[test]
    fn the_users_own_home_is_left_alone_even_with_a_journal() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("me");
        std::fs::create_dir(&home).unwrap();
        std::fs::write(journal_path(&home), "").unwrap();
        set(&home, 0o755);
        create_home_for(&home, Some(&home)).unwrap();
        assert_eq!(mode(&home), 0o755);
        // The same directory reached by another path.
        let via = home.join("..").join("me");
        create_home_for(&via, Some(&home)).unwrap();
        assert_eq!(mode(&home), 0o755);
    }

    #[test]
    fn own_subdirectories_are_tightened() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(journal_path(root.path()), "").unwrap();
        let dir = root.path().join("hooks");
        create_own_dir_all(root.path(), &dir).unwrap();
        assert_eq!(mode(&dir), 0o700);
        set(&dir, 0o755);
        create_own_dir_all(root.path(), &dir).unwrap();
        assert_eq!(mode(&dir), 0o700);
    }

    #[test]
    fn own_subdirectories_of_a_foreign_home_are_left_alone() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("hooks");
        std::fs::create_dir(&dir).unwrap();
        set(&dir, 0o755);
        create_own_dir_all(root.path(), &dir).unwrap();
        assert_eq!(mode(&dir), 0o755);
    }

    #[test]
    fn tightening_never_loosens() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("f");
        std::fs::write(&path, "x").unwrap();
        set(&path, 0o400);
        tighten_file(&File::open(&path).unwrap());
        assert_eq!(mode(&path), 0o400);
        set(&path, 0o664);
        tighten_file(&File::open(&path).unwrap());
        assert_eq!(mode(&path), 0o600);
    }
}
