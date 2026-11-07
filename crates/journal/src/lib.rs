//! File journal for tessar: an append-only log of commits in JSON Lines
//! with a checksum per line. Implements [`tessar_core::log::Log`].
//!
//! # Format, version 1
//!
//! ```text
//! tessar-journal 1
//! 9f0c2a1b<TAB>{"seq":1,"at":1730000000000,"ops":[…]}
//! 03d4e5f6<TAB>{"seq":2,"at":1730000000500,"ops":[…]}
//! ```
//!
//! The first line names the format. Every other line is one commit: the
//! CRC-32 of its JSON as 8 hex digits, a tab, the JSON. A commit is durable
//! once its line is written and synced to disk.
//!
//! # Recovery on open
//!
//! - A damaged or unfinished **last** line is a torn write — the process
//!   died while appending. It is cut off and the journal opens.
//! - A damaged line **followed by** more data is corruption. Opening fails
//!   with the line number: silently dropping data is worse than refusing.
//! - A line with a valid checksum that does not decode, or a gap in commit
//!   numbers, is corruption too, wherever it is.
//!
//! # Locking
//!
//! The file is locked for the life of the [`FileJournal`]. A second open,
//! from this process or another, fails with [`LogError::Locked`].

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;
use tessar_core::log::{Commit, Log, LogError, Seq};

/// The first line of every journal file.
const HEADER: &[u8] = b"tessar-journal 1\n";

/// A journal of commits with operations of type `T`, kept in one file.
#[derive(Debug)]
pub struct FileJournal<T> {
    file: File,
    path: PathBuf,
    next_seq: Seq,
    /// Commits read on open, handed out by `replay` until the first append.
    recovered: Option<Vec<Commit<T>>>,
    /// Bytes of a torn write cut off on open.
    cut_on_open: u64,
    /// Set after a failed append: the file may end in a partial line.
    poisoned: bool,
}

impl<T: Serialize + DeserializeOwned> FileJournal<T> {
    /// Opens the journal at `path`, creating it if missing, locking it and
    /// cutting off a torn write at the end.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LogError> {
        let path = path.as_ref().to_owned();
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)?;
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(LogError::Locked),
            Err(TryLockError::Error(err)) => return Err(err.into()),
        }

        let mut bytes = Vec::new();
        file.seek(SeekFrom::Start(0))?;
        file.read_to_end(&mut bytes)?;

        let Scan { commits, valid_len } = scan(&bytes)?;
        let cut_on_open = bytes.len() as u64 - valid_len;
        if valid_len < HEADER.len() as u64 {
            // New file, or the process died while writing the header.
            file.set_len(0)?;
            file.write_all(HEADER)?;
            file.sync_all()?;
        } else if cut_on_open > 0 {
            file.set_len(valid_len)?;
            file.sync_all()?;
        }

        let next_seq = commits.last().map_or(Seq::FIRST, |c| c.seq.next());
        Ok(Self {
            file,
            path,
            next_seq,
            recovered: Some(commits),
            cut_on_open,
            poisoned: false,
        })
    }

    /// The path of the journal file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// How many bytes of a torn write were cut off on open; 0 if none.
    pub fn cut_on_open(&self) -> u64 {
        self.cut_on_open
    }
}

impl<T: Serialize + DeserializeOwned> Log<T> for FileJournal<T> {
    fn replay(&mut self) -> Result<Vec<Commit<T>>, LogError> {
        if let Some(commits) = self.recovered.take() {
            return Ok(commits);
        }
        let mut bytes = Vec::new();
        self.file.seek(SeekFrom::Start(0))?;
        self.file.read_to_end(&mut bytes)?;
        Ok(scan(&bytes)?.commits)
    }

    fn append(&mut self, commit: &Commit<T>) -> Result<(), LogError> {
        if self.poisoned {
            return Err(LogError::Poisoned);
        }
        if commit.seq != self.next_seq {
            return Err(LogError::OutOfOrder {
                expected: self.next_seq,
                got: commit.seq,
            });
        }
        let line = encode_line(commit)?;
        // After a failed write or sync the file may end in a partial line
        // and the state on disk is unknown: retrying could write after
        // garbage. Refuse until reopened; recovery then cuts the tail.
        let written = self
            .file
            .write_all(&line)
            .and_then(|()| self.file.sync_data());
        if let Err(err) = written {
            self.poisoned = true;
            return Err(err.into());
        }
        self.next_seq = commit.seq.next();
        // The commits read on open are no longer the whole log.
        self.recovered = None;
        Ok(())
    }

    fn next_seq(&self) -> Seq {
        self.next_seq
    }
}

/// Encodes one commit as a journal line, newline included.
fn encode_line<T: Serialize>(commit: &Commit<T>) -> Result<Vec<u8>, LogError> {
    let json = serde_json::to_vec(commit).map_err(|err| LogError::Encode {
        seq: commit.seq,
        reason: err.to_string(),
    })?;
    let mut line = format!("{:08x}\t", crc32fast::hash(&json)).into_bytes();
    line.extend_from_slice(&json);
    line.push(b'\n');
    Ok(line)
}

/// What is wrong with a line.
enum LineError {
    /// The checksum or the framing does not match: the bytes are damaged.
    /// At the very end of the file this is a torn write.
    Damaged(String),
    /// The bytes are intact but mean nothing valid. Never a torn write.
    Invalid(String),
}

/// Decodes one line, without its newline.
fn decode_line<T: DeserializeOwned>(line: &[u8]) -> Result<Commit<T>, LineError> {
    let damaged = |reason: &str| LineError::Damaged(reason.to_owned());
    let (crc, json) = match line.split_first_chunk::<8>() {
        Some((crc, [b'\t', json @ ..])) => (crc, json),
        _ => return Err(damaged("no checksum")),
    };
    let crc = std::str::from_utf8(crc)
        .ok()
        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
        .ok_or_else(|| damaged("malformed checksum"))?;
    if crc32fast::hash(json) != crc {
        return Err(damaged("checksum mismatch"));
    }
    serde_json::from_slice(json).map_err(|err| LineError::Invalid(err.to_string()))
}

/// The result of reading a journal file.
struct Scan<T> {
    commits: Vec<Commit<T>>,
    /// Length of the valid prefix; anything after it is a torn write.
    valid_len: u64,
}

/// Reads all commits, finds where a torn write starts and checks
/// everything else.
fn scan<T: DeserializeOwned>(bytes: &[u8]) -> Result<Scan<T>, LogError> {
    if bytes.len() < HEADER.len() {
        if HEADER.starts_with(bytes) {
            // Empty, or the process died while writing the header.
            return Ok(Scan {
                commits: Vec::new(),
                valid_len: 0,
            });
        }
        return Err(LogError::UnsupportedFormat(first_line(bytes)));
    }
    if !bytes.starts_with(HEADER) {
        return Err(LogError::UnsupportedFormat(first_line(bytes)));
    }

    let mut commits = Vec::new();
    let mut expected = Seq::FIRST;
    let mut pos = HEADER.len();
    let mut line_no: u64 = 1;
    while pos < bytes.len() {
        line_no += 1;
        let rest = &bytes[pos..];
        let Some(len) = rest.iter().position(|&b| b == b'\n') else {
            // The last line never got its newline: a torn write.
            break;
        };
        let is_last = pos + len + 1 == bytes.len();
        let commit: Commit<T> = match decode_line(&rest[..len]) {
            Ok(commit) => commit,
            Err(LineError::Damaged(_)) if is_last => break,
            Err(LineError::Damaged(reason) | LineError::Invalid(reason)) => {
                return Err(LogError::Corrupted {
                    line: line_no,
                    reason,
                });
            }
        };
        if commit.seq != expected {
            return Err(LogError::Corrupted {
                line: line_no,
                reason: format!("expected commit {expected}, found {}", commit.seq),
            });
        }
        expected = commit.seq.next();
        commits.push(commit);
        pos += len + 1;
    }
    // `pos` is the end of the file, or the start of a torn last line.
    Ok(Scan {
        commits,
        valid_len: pos as u64,
    })
}

/// The first line of a file, for error messages.
fn first_line(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|&b| b == b'\n')
        .unwrap_or(bytes.len())
        .min(64);
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}
