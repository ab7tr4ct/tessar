//! Errors of the store.

use std::num::NonZeroU32;

use thiserror::Error;

use crate::log::{LogError, Seq};
use crate::model::{
    EntityId, EntityKindId, Lineage, ModelError, PredicateId, SpaceId, Timestamp, ValueType,
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

    /// A record from another space was used.
    #[error("{record} belongs to space {found}, not {expected}")]
    WrongSpace {
        /// The record, for example `e3`.
        record: String,
        /// The space of the write.
        expected: SpaceId,
        /// The space of the record.
        found: SpaceId,
    },

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

    /// A change is dated before the version it replaces became true.
    #[error("change at {at:?} is before the version became true at {valid_from:?}")]
    ChangeBeforeValid {
        /// The change time.
        at: Timestamp,
        /// When the replaced version became true.
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

    /// A writer panicked earlier; reopen the memory.
    #[error("a writer panicked earlier; reopen the memory")]
    Poisoned,
}
