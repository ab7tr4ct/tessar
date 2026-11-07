//! The persistence port: a durable, append-only log of commits.
//!
//! The in-memory state of the store is the result of replaying the log
//! from the start. One commit is one transaction: all of its operations are
//! applied, or none of them.
//!
//! The core knows nothing about files. [`MemLog`] keeps commits in memory
//! for tests; the file journal lives in the `tessar-journal` crate. Any new
//! implementation must pass the shared contract tests in `contract`
//! (feature `testing`).

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::Timestamp;

/// The number of a commit in the log. Commits are numbered from 1 with no
/// gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Seq(u64);

impl Seq {
    /// The number of the first commit.
    pub const FIRST: Seq = Seq(1);

    /// The commit number `raw`.
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw number.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The number of the commit after this one.
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for Seq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// One transaction: operations that are applied together or not at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Commit<T> {
    /// The commit number.
    pub seq: Seq,
    /// When the commit was made.
    pub at: Timestamp,
    /// The operations, in order.
    pub ops: Vec<T>,
}

/// Errors of a [`Log`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LogError {
    /// Another process holds the log.
    #[error("the log is locked by another process")]
    Locked,

    /// The operating system reported an error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Damaged data in the middle of the log. A torn write at the very end
    /// is not corruption: it is cut off on open.
    #[error("the log is corrupted at line {line}: {reason}")]
    Corrupted {
        /// The line, counted from 1.
        line: u64,
        /// What is wrong with it.
        reason: String,
    },

    /// The log was written in a format this version cannot read.
    #[error("unsupported log format: {0:?}")]
    UnsupportedFormat(String),

    /// A commit was appended out of order.
    #[error("expected commit {expected}, got {got}")]
    OutOfOrder {
        /// The number the next commit must have.
        expected: Seq,
        /// The number it had.
        got: Seq,
    },

    /// A commit could not be encoded.
    #[error("cannot encode commit {seq}: {reason}")]
    Encode {
        /// The commit number.
        seq: Seq,
        /// Why.
        reason: String,
    },

    /// An earlier append failed halfway; the log refuses further appends
    /// until it is reopened, which cuts the partial write off.
    #[error("the log is unusable after a failed append; reopen it to recover")]
    Poisoned,
}

/// A durable, append-only sequence of commits.
///
/// The store calls [`Log::replay`] once on start, then only
/// [`Log::append`].
pub trait Log<T> {
    /// Every commit from the start, oldest first.
    fn replay(&mut self) -> Result<Vec<Commit<T>>, LogError>;

    /// Appends `commit`. When this returns `Ok`, the commit is durable.
    ///
    /// `commit.seq` must equal [`Log::next_seq`]; otherwise the call fails
    /// with [`LogError::OutOfOrder`] and the log is unchanged.
    fn append(&mut self, commit: &Commit<T>) -> Result<(), LogError>;

    /// The number the next commit must have.
    fn next_seq(&self) -> Seq;
}

/// A log kept in memory. For tests; nothing survives the process.
#[derive(Debug)]
pub struct MemLog<T> {
    commits: Vec<Commit<T>>,
}

impl<T> MemLog<T> {
    /// An empty log.
    pub const fn new() -> Self {
        Self {
            commits: Vec::new(),
        }
    }

    /// The commits appended so far.
    pub fn commits(&self) -> &[Commit<T>] {
        &self.commits
    }
}

impl<T> Default for MemLog<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Log<T> for MemLog<T> {
    fn replay(&mut self) -> Result<Vec<Commit<T>>, LogError> {
        Ok(self.commits.clone())
    }

    fn append(&mut self, commit: &Commit<T>) -> Result<(), LogError> {
        let expected = self.next_seq();
        if commit.seq != expected {
            return Err(LogError::OutOfOrder {
                expected,
                got: commit.seq,
            });
        }
        self.commits.push(commit.clone());
        Ok(())
    }

    fn next_seq(&self) -> Seq {
        Seq(self.commits.len() as u64 + 1)
    }
}

/// Contract tests that every [`Log`] implementation must pass.
///
/// Each function takes a fresh, empty log and panics on a violation.
#[cfg(any(test, feature = "testing"))]
pub mod contract {
    use super::{Commit, Log, LogError, Seq};
    use crate::model::Timestamp;

    fn commit(seq: u64, ops: &[u32]) -> Commit<u32> {
        Commit {
            seq: Seq::new(seq),
            at: Timestamp::from_millis(seq as i64 * 1000),
            ops: ops.to_vec(),
        }
    }

    /// An empty log replays nothing and expects commit 1.
    pub fn starts_empty(mut log: impl Log<u32>) {
        assert_eq!(log.next_seq(), Seq::FIRST);
        assert!(log.replay().unwrap().is_empty());
    }

    /// Appended commits replay in order, unchanged, including empty ones.
    pub fn replays_what_was_appended(mut log: impl Log<u32>) {
        let commits = [commit(1, &[10, 11]), commit(2, &[]), commit(3, &[12])];
        for c in &commits {
            log.append(c).unwrap();
        }
        assert_eq!(log.next_seq(), Seq::new(4));
        assert_eq!(log.replay().unwrap(), commits);
    }

    /// A commit with the wrong number is rejected and changes nothing.
    pub fn rejects_out_of_order(mut log: impl Log<u32>) {
        log.append(&commit(1, &[1])).unwrap();
        for wrong in [1, 3, 0] {
            let err = log.append(&commit(wrong, &[2])).unwrap_err();
            assert!(
                matches!(err, LogError::OutOfOrder { expected, got }
                    if expected == Seq::new(2) && got == Seq::new(wrong)),
                "{err}"
            );
        }
        assert_eq!(log.next_seq(), Seq::new(2));
        assert_eq!(log.replay().unwrap(), [commit(1, &[1])]);
    }

    /// Runs every contract test, each on a fresh log from `fresh`.
    pub fn all<L: Log<u32>>(mut fresh: impl FnMut() -> L) {
        starts_empty(fresh());
        replays_what_was_appended(fresh());
        rejects_out_of_order(fresh());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_log_passes_the_contract() {
        contract::all(MemLog::new);
    }

    #[test]
    fn seq_counts_from_one() {
        assert_eq!(Seq::FIRST.get(), 1);
        assert_eq!(Seq::FIRST.next(), Seq::new(2));
        assert_eq!(Seq::new(42).to_string(), "#42");
    }

    #[test]
    fn commit_json_is_flat() {
        let c = Commit {
            seq: Seq::new(3),
            at: Timestamp::from_millis(5),
            ops: vec!["a".to_owned()],
        };
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, r#"{"seq":3,"at":5,"ops":["a"]}"#);
        assert_eq!(serde_json::from_str::<Commit<String>>(&json).unwrap(), c);
    }
}
