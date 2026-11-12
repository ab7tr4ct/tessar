//! Errors of the store.

use std::num::NonZeroU32;

use thiserror::Error;

use crate::embed::EmbedError;
use crate::log::{LogError, Seq};
use crate::model::{
    EntityId, EntityKindId, EpisodeId, FactId, Lineage, ModelError, PredicateId, SpaceId,
    Timestamp, ValueType,
};

/// Errors of the store.
///
/// A write that fails changes nothing: every check runs before anything is
/// written to the log.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoreError {
    /// A value breaks a model invariant.
    #[error(transparent)]
    Model(#[from] ModelError),

    /// The log failed.
    #[error(transparent)]
    Log(#[from] LogError),

    /// The embedder failed or returned a vector the index cannot hold.
    /// A write that needs vectors fails whole: nothing is written.
    #[error("embedder failed: {0}")]
    Embed(#[from] EmbedError),

    /// No such space.
    #[error("unknown space {0}")]
    UnknownSpace(SpaceId),

    /// No such entity kind.
    #[error("unknown entity kind {0}")]
    UnknownKind(EntityKindId),

    /// No such predicate.
    #[error("unknown predicate {0}")]
    UnknownPredicate(PredicateId),

    /// No such entity.
    #[error("unknown entity {0}")]
    UnknownEntity(EntityId),

    /// No such fact lineage.
    #[error("unknown fact lineage {0}")]
    UnknownLineage(Lineage),

    /// No such fact version.
    #[error("unknown fact {0}")]
    UnknownFact(FactId),

    /// No such episode.
    #[error("unknown episode {0}")]
    UnknownEpisode(EpisodeId),

    /// The fact version was redacted already.
    #[error("fact {0} is already redacted")]
    AlreadyRedacted(FactId),

    /// The episode was redacted already.
    #[error("episode {0} is already redacted")]
    EpisodeAlreadyRedacted(EpisodeId),

    /// The space is archived and read-only.
    #[error("space {0} is archived")]
    SpaceArchived(SpaceId),

    /// A name is already used by another record.
    #[error("{0} is already taken")]
    NameTaken(String),

    /// A predicate with this name exists with another type or cardinality.
    #[error("predicate {name} is already defined differently")]
    PredicateConflict {
        /// The predicate name.
        name: String,
    },

    /// A predicate name has more words than allowed.
    #[error("predicate name {name} has more than {max_words} words")]
    PredicateNameTooLong {
        /// The name.
        name: String,
        /// The most words a name may have.
        max_words: usize,
    },

    /// A word of a predicate name is a number or the name of an entity:
    /// `plans_trip_to_japan`. The thing named belongs in the value.
    #[error("predicate name {name} is not universal: {word:?} {reason}")]
    PredicateNotUniversal {
        /// The name.
        name: String,
        /// The offending word.
        word: String,
        /// Why the word is not allowed.
        reason: NotUniversal,
    },

    /// A predicate with a similar meaning already exists. The caller may
    /// use it, declare the new name its synonym, or insist with
    /// [`Memory::define_predicate_unchecked`](super::Memory::define_predicate_unchecked).
    #[error("predicate {name} is similar to existing {existing} (similarity {similarity:.2})")]
    PredicateSimilar {
        /// The proposed name.
        name: String,
        /// The most similar existing predicate (its canonical id).
        existing: PredicateId,
        /// Cosine similarity of the two, in the embedder's units.
        similarity: f32,
    },

    /// Two predicates cannot be merged: another value type or cardinality.
    #[error("cannot merge {from} into {into}: value type or cardinality differ")]
    MergeMismatch {
        /// The predicate that would become a synonym.
        from: PredicateId,
        /// The canonical predicate.
        into: PredicateId,
    },

    /// The value does not have the type the predicate accepts.
    #[error("predicate {predicate} expects {expected:?}")]
    TypeMismatch {
        /// The predicate.
        predicate: PredicateId,
        /// The type it accepts.
        expected: ValueType,
    },

    /// A text field is empty.
    #[error("text must not be empty")]
    EmptyText,

    /// One transaction sets two values of a single-valued predicate.
    #[error("two values of single-valued predicate {predicate} for {subject} in one transaction")]
    DuplicatePair {
        /// The subject.
        subject: EntityId,
        /// The predicate.
        predicate: PredicateId,
    },

    /// The writer saw another version than the current one: someone else
    /// changed the fact in between. Re-read and decide again.
    #[error("conflict: expected version {expected}, current is {current:?}")]
    Conflict {
        /// The lineage, if it exists.
        lineage: Option<Lineage>,
        /// The version the writer saw.
        expected: NonZeroU32,
        /// The current version; `None` if there is no current value.
        current: Option<NonZeroU32>,
    },

    /// The lineage has no current version to update.
    #[error("{0} has no current version")]
    NotCurrent(Lineage),

    /// A replacement changes the kind, subject or predicate of a fact.
    #[error("a replacement must keep the kind, subject and predicate of {0}")]
    ShapeMismatch(Lineage),

    /// A change or a new value is dated too early. Two cases share this
    /// error; in both, `valid_from` is the earliest date the write may
    /// carry.
    ///
    /// 1. A change of a current version dated before that version became
    ///    true: `valid_from` is the `valid_from` of the version.
    /// 2. A value for a single-valued pair that has no current value,
    ///    dated while an earlier value of the pair was still true (it
    ///    would overlap it in a read of that time): `valid_from` is then
    ///    not a start but the latest `valid_to` among the versions of the
    ///    pair closed as changed. Corrected and redacted versions do not
    ///    count. The message below describes case 1 only.
    #[error("change at {at:?} is before the version became true at {valid_from:?}")]
    ChangeBeforeValid {
        /// The date the write carried.
        at: Timestamp,
        /// The earliest date it may carry: the start of the replaced
        /// version (case 1) or the end of the last true value (case 2).
        valid_from: Timestamp,
    },

    /// A record vector is full.
    #[error("too many records")]
    TooManyRecords,

    /// The log holds a commit that cannot be applied.
    #[error("cannot replay commit {seq}: {reason}")]
    Replay {
        /// The commit.
        seq: Seq,
        /// Why.
        reason: String,
    },

    /// A durable commit could not be applied to the state. A bug.
    #[error("internal error: {0}")]
    Internal(String),

    /// A durable commit failed to apply earlier, so the state no longer
    /// matches the log. Writes are refused; reads still work.
    #[error(
        "memory state diverged from the journal after an internal error; reopen it to rebuild the state from the journal"
    )]
    Corrupted,
}

/// Why a word of a predicate name is not universal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotUniversal {
    /// The word is a number.
    Number,
    /// The word (or run of words) is the name of an entity.
    EntityName,
}

impl std::fmt::Display for NotUniversal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Number => "is a number",
            Self::EntityName => "is the name of an entity",
        })
    }
}
