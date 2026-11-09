use std::fs;
use std::path::{Path, PathBuf};

use proptest::prelude::*;
use tempfile::TempDir;
use tessar_core::log::{Commit, Log, LogError, Seq, contract};
use tessar_core::model::Timestamp;
use tessar_journal::FileJournal;

fn commit(seq: u64, ops: &[u32]) -> Commit<u32> {
    Commit {
        seq: Seq::new(seq),
        at: Timestamp::from_millis(seq as i64 * 1000),
        ops: ops.to_vec(),
    }
}

fn open(path: &Path) -> Result<FileJournal<u32>, LogError> {
    FileJournal::open(path)
}

/// Writes `commits` to a new journal and returns the file bytes.
fn write_journal(path: &Path, commits: &[Commit<u32>]) -> Vec<u8> {
    let mut journal = open(path).unwrap();
    for c in commits {
        journal.append(c).unwrap();
    }
    drop(journal);
    fs::read(path).unwrap()
}

fn sample() -> Vec<Commit<u32>> {
    vec![
        commit(1, &[1, 2]),
        commit(2, &[]),
        commit(3, &[3]),
        commit(4, &[4, 5, 6]),
    ]
}

/// Byte offsets where each commit line ends, newline included.
fn line_ends(bytes: &[u8]) -> Vec<usize> {
    bytes
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b == b'\n')
        .map(|(i, _)| i + 1)
        .skip(1) // the header
        .collect()
}

struct Dir {
    dir: TempDir,
    next: usize,
}

impl Dir {
    fn new() -> Self {
        Self {
            dir: tempfile::tempdir().unwrap(),
            next: 0,
        }
    }

    fn fresh_path(&mut self) -> PathBuf {
        self.next += 1;
        self.dir.path().join(format!("journal-{}.log", self.next))
    }
}

#[test]
fn passes_the_log_contract() {
    let mut dir = Dir::new();
    contract::all(|| open(&dir.fresh_path()).unwrap());
}

#[test]
fn new_file_starts_with_header_and_lines_are_checksummed() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    let bytes = write_journal(&path, &[commit(1, &[7])]);
    let text = String::from_utf8(bytes).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("tessar-journal 1"));
    let line = lines.next().unwrap();
    let (crc, json) = line.split_once('\t').unwrap();
    assert_eq!(json, r#"{"seq":1,"at":1000,"ops":[7]}"#);
    assert_eq!(
        u32::from_str_radix(crc, 16).unwrap(),
        crc32fast::hash(json.as_bytes())
    );
    assert_eq!(lines.next(), None);
}

#[test]
fn reopen_replays_and_continues_numbering() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    write_journal(&path, &sample());

    let mut journal = open(&path).unwrap();
    assert_eq!(journal.cut_on_open(), 0);
    assert_eq!(journal.next_seq(), Seq::new(5));
    assert_eq!(journal.replay().unwrap(), sample());
    // A second replay reads the file again and sees the same.
    assert_eq!(journal.replay().unwrap(), sample());
    journal.append(&commit(5, &[9])).unwrap();
    drop(journal);

    let mut journal = open(&path).unwrap();
    let mut expected = sample();
    expected.push(commit(5, &[9]));
    assert_eq!(journal.replay().unwrap(), expected);
}

#[test]
fn second_open_is_locked_out() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    let _first = open(&path).unwrap();
    assert!(matches!(open(&path), Err(LogError::Locked)));
}

#[test]
fn lock_is_released_on_drop() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    drop(open(&path).unwrap());
    assert!(open(&path).is_ok());
}

#[test]
fn is_locked_tells_whether_the_journal_is_held_and_changes_nothing() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    assert!(!FileJournal::<u32>::is_locked(&path).unwrap());
    assert!(!path.exists(), "the check created the file");
    let before = write_journal(&path, &sample());
    assert!(!FileJournal::<u32>::is_locked(&path).unwrap());
    let journal = open(&path).unwrap();
    assert!(FileJournal::<u32>::is_locked(&path).unwrap());
    drop(journal);
    assert!(!FileJournal::<u32>::is_locked(&path).unwrap());
    assert_eq!(fs::read(&path).unwrap(), before);
    // The check left the lock free.
    assert!(open(&path).is_ok());
}

/// Starts a blocking open on a thread; the journal arrives on the channel.
fn open_blocking_on_a_thread(
    path: &Path,
) -> std::sync::mpsc::Receiver<Result<FileJournal<u32>, LogError>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let path = path.to_owned();
    std::thread::spawn(move || {
        let _ = tx.send(FileJournal::<u32>::open_blocking(&path));
    });
    rx
}

const STILL_WAITING: std::time::Duration = std::time::Duration::from_millis(200);
const ARRIVES: std::time::Duration = std::time::Duration::from_secs(10);

#[test]
fn blocking_open_of_a_free_journal_does_not_wait() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    write_journal(&path, &sample());
    let mut journal = FileJournal::<u32>::open_blocking(&path).unwrap();
    assert_eq!(journal.replay().unwrap(), sample());
    // It holds the lock like any other open.
    assert!(matches!(open(&path), Err(LogError::Locked)));
}

#[test]
fn blocking_open_waits_for_the_holder_and_sees_what_it_wrote() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    let mut holder = open(&path).unwrap();
    let waiter = open_blocking_on_a_thread(&path);
    assert!(
        waiter.recv_timeout(STILL_WAITING).is_err(),
        "opened a journal that is locked"
    );
    // Written while the other open was already waiting.
    for c in sample() {
        holder.append(&c).unwrap();
    }
    drop(holder);
    let mut journal = waiter.recv_timeout(ARRIVES).unwrap().unwrap();
    assert_eq!(journal.replay().unwrap(), sample());
    assert!(matches!(open(&path), Err(LogError::Locked)));
    journal.append(&commit(5, &[9])).unwrap();
}

/// A rewrite renames a new file over the path. An open that waited on the
/// old file must not end up with a lock on a file that has no name.
#[cfg(unix)]
#[test]
fn blocking_open_follows_a_file_replaced_while_it_waited() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    write_journal(&path, &sample());
    let holder = open(&path).unwrap();
    let waiter = open_blocking_on_a_thread(&path);
    assert!(waiter.recv_timeout(STILL_WAITING).is_err());
    // What `rewrite` does while it holds the lock.
    let replacement = dir.fresh_path();
    write_journal(&replacement, &sample()[..2]);
    fs::rename(&replacement, &path).unwrap();
    drop(holder);
    let mut journal = waiter.recv_timeout(ARRIVES).unwrap().unwrap();
    assert_eq!(journal.replay().unwrap(), sample()[..2]);
    // The lock is on the file the path names now.
    assert!(matches!(open(&path), Err(LogError::Locked)));
    journal.append(&commit(3, &[7])).unwrap();
    drop(journal);
    let mut expected = sample()[..2].to_vec();
    expected.push(commit(3, &[7]));
    assert_eq!(open(&path).unwrap().replay().unwrap(), expected);
}

/// The process may die at any byte of an append. Whatever the cut, the
/// journal opens with exactly the commits whose lines are complete, and
/// keeps working after that.
#[test]
fn every_torn_tail_is_recovered() {
    let mut dir = Dir::new();
    let full = write_journal(&dir.fresh_path(), &sample());
    let ends = line_ends(&full);

    for cut in 0..=full.len() {
        let path = dir.fresh_path();
        fs::write(&path, &full[..cut]).unwrap();
        let complete = ends.iter().filter(|&&end| end <= cut).count();

        let mut journal = open(&path).unwrap_or_else(|e| panic!("cut {cut}: {e}"));
        assert_eq!(journal.replay().unwrap(), sample()[..complete], "cut {cut}");
        let next = complete as u64 + 1;
        assert_eq!(journal.next_seq(), Seq::new(next), "cut {cut}");

        journal.append(&commit(next, &[99])).unwrap();
        drop(journal);
        let mut journal = open(&path).unwrap();
        assert_eq!(journal.cut_on_open(), 0, "cut {cut}");
        assert_eq!(journal.replay().unwrap().len(), complete + 1, "cut {cut}");
    }
}

/// Flip any single byte: the journal either opens with a prefix of what
/// was written, or refuses with `Corrupted`. It never returns a commit that
/// was not written.
#[test]
fn damage_never_yields_wrong_commits() {
    let mut dir = Dir::new();
    let full = write_journal(&dir.fresh_path(), &sample());
    let header_len = full.iter().position(|&b| b == b'\n').unwrap() + 1;

    for at in header_len..full.len() {
        let mut bytes = full.clone();
        bytes[at] ^= 0x01;
        let path = dir.fresh_path();
        fs::write(&path, &bytes).unwrap();
        match open(&path) {
            Ok(mut journal) => {
                let got = journal.replay().unwrap();
                assert!(sample().starts_with(&got), "byte {at}: {got:?}");
            }
            Err(LogError::Corrupted { .. }) => {}
            Err(other) => panic!("byte {at}: unexpected {other}"),
        }
    }
}

#[test]
fn damage_in_the_middle_is_reported_with_line_number() {
    let mut dir = Dir::new();
    let full = write_journal(&dir.fresh_path(), &sample());
    let ends = line_ends(&full);
    // Damage the JSON of commit 2, which sits on file line 3.
    let mut bytes = full.clone();
    bytes[ends[0] + 12] ^= 0x01;
    let path = dir.fresh_path();
    fs::write(&path, &bytes).unwrap();
    match open(&path) {
        Err(LogError::Corrupted { line, .. }) => assert_eq!(line, 3),
        other => panic!("expected corruption, got {other:?}"),
    }
}

#[test]
fn intact_but_undecodable_line_is_corruption_even_at_the_end() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    let mut journal: FileJournal<String> = FileJournal::open(&path).unwrap();
    journal
        .append(&Commit {
            seq: Seq::FIRST,
            at: Timestamp::from_millis(0),
            ops: vec!["not a number".to_owned()],
        })
        .unwrap();
    drop(journal);
    assert!(matches!(
        open(&path),
        Err(LogError::Corrupted { line: 2, .. })
    ));
}

#[test]
fn gap_in_commit_numbers_is_corruption() {
    let mut dir = Dir::new();
    let source = write_journal(&dir.fresh_path(), &sample());
    let lines: Vec<&[u8]> = source.split_inclusive(|&b| b == b'\n').collect();
    // Header, commit 1, commit 3: commit 2 is missing.
    let bytes = [lines[0], lines[1], lines[3]].concat();
    let path = dir.fresh_path();
    fs::write(&path, bytes).unwrap();
    assert!(matches!(
        open(&path),
        Err(LogError::Corrupted { line: 3, .. })
    ));
}

#[test]
fn foreign_file_is_rejected() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    fs::write(&path, "hello world\n").unwrap();
    assert!(matches!(open(&path), Err(LogError::UnsupportedFormat(_))));
    fs::write(&path, "tessar-journal 2\n").unwrap();
    assert!(matches!(open(&path), Err(LogError::UnsupportedFormat(_))));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Any operations survive a round trip through the file.
    #[test]
    fn any_commits_round_trip(ops in prop::collection::vec(prop::collection::vec(any::<u32>(), 0..6), 0..8)) {
        let mut dir = Dir::new();
        let path = dir.fresh_path();
        let commits: Vec<Commit<u32>> = ops
            .iter()
            .enumerate()
            .map(|(i, ops)| commit(i as u64 + 1, ops))
            .collect();
        write_journal(&path, &commits);
        let mut journal = open(&path).unwrap();
        prop_assert_eq!(journal.replay().unwrap(), commits);
    }
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap().to_owned();
    name.push(".tmp");
    path.with_file_name(name)
}

#[test]
fn rewrite_through_identity_changes_nothing() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    let before = write_journal(&path, &sample());
    let report = FileJournal::<u32>::rewrite(&path, |commits| commits).unwrap();
    assert_eq!(report.commits_before, 4);
    assert_eq!(report.commits_after, 4);
    assert_eq!(report.bytes_before, before.len() as u64);
    assert_eq!(report.bytes_after, before.len() as u64);
    assert_eq!(
        fs::read(&path).unwrap(),
        before,
        "same bytes, same checksums"
    );
    assert!(!tmp_path(&path).exists());
    assert_eq!(open(&path).unwrap().replay().unwrap(), sample());
}

#[test]
fn rewrite_stores_the_changed_payload_with_fresh_checksums() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    write_journal(&path, &sample());
    FileJournal::<u32>::rewrite(&path, |mut commits| {
        commits[2].ops = vec![33, 34];
        commits
    })
    .unwrap();

    let mut expected = sample();
    expected[2].ops = vec![33, 34];
    let mut journal = open(&path).unwrap();
    assert_eq!(journal.cut_on_open(), 0);
    assert_eq!(journal.replay().unwrap(), expected);
    // The journal goes on from where it was.
    journal.append(&commit(5, &[9])).unwrap();
}

#[test]
fn rewrite_refuses_a_journal_that_is_open() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    let before = write_journal(&path, &sample());
    let journal = open(&path).unwrap();
    let err = FileJournal::<u32>::rewrite(&path, |_| Vec::new()).unwrap_err();
    assert!(matches!(err, LogError::Locked), "{err}");
    drop(journal);
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(!tmp_path(&path).exists());
}

#[test]
fn rewrite_refuses_commits_that_do_not_count_from_one() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    let before = write_journal(&path, &sample());
    let err = FileJournal::<u32>::rewrite(&path, |mut commits| {
        commits.remove(1);
        commits
    })
    .unwrap_err();
    assert!(matches!(err, LogError::OutOfOrder { .. }), "{err}");
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(!tmp_path(&path).exists());
}

#[test]
fn rewrite_needs_an_existing_journal() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    assert!(matches!(
        FileJournal::<u32>::rewrite(&path, |c| c),
        Err(LogError::Io(_))
    ));
    assert!(!path.exists(), "rewrite must not create a journal");
}

/// A crash while the new file is being written leaves `<name>.tmp`. It is
/// not part of the journal: the journal opens from the original, and the
/// next rewrite overwrites the leftover.
#[test]
fn a_torn_tmp_file_is_ignored_and_overwritten() {
    let mut dir = Dir::new();
    let path = dir.fresh_path();
    let before = write_journal(&path, &sample());
    let tmp = tmp_path(&path);
    fs::write(&tmp, &before[..before.len() / 2]).unwrap();

    let mut journal = open(&path).unwrap();
    assert_eq!(journal.cut_on_open(), 0);
    assert_eq!(journal.replay().unwrap(), sample());
    journal.append(&commit(5, &[9])).unwrap();
    drop(journal);
    assert_eq!(fs::read(&tmp).unwrap(), &before[..before.len() / 2]);

    FileJournal::<u32>::rewrite(&path, |commits| commits).unwrap();
    assert!(!tmp.exists(), "the leftover was replaced and renamed away");
    let mut expected = sample();
    expected.push(commit(5, &[9]));
    assert_eq!(open(&path).unwrap().replay().unwrap(), expected);
}
