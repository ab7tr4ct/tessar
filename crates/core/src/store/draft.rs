//! Requests and results of the store API.

use std::num::NonZeroU32;

use crate::model::{
    Cardinality, EntityId, EntityKindId, EpisodeId, Fact, FactId, Key, Lineage, Name, Origin,
    PredicateId, SpaceId, TimeSpan, Timestamp, Value, ValueType,
};

/// A predicate to define in a space schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateDraft {
    /// The name: `lives_in`, `trip_destination`. One to four words joined
    /// by `_`.
    pub name: Name,
    /// One line: what the predicate means and what its value is. May be
    /// empty, but then the similarity check sees the name alone.
    pub description: String,
    /// Which values are allowed.
    pub object: ValueType,
    /// How many current values one subject may have.
    pub cardinality: Cardinality,
}

/// An entity named in a write: by id, or by key — found or created.
#[derive(Debug, Clone, PartialEq)]
pub enum EntityRef {
    /// An existing entity.
    Id(EntityId),
    /// The entity of `kind` with this key or alias; created if missing.
    Key {
        /// The entity kind.
        kind: EntityKindId,
        /// The key or an alias.
        key: Key,
    },
}

/// The value of a triple in a write.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectDraft {
    /// A plain value. A [`Value::Entity`] here must be an existing entity.
    Value(Value),
    /// An entity by id or key; created if missing.
    Entity(EntityRef),
}

/// The content of a fact in a write.
#[derive(Debug, Clone, PartialEq)]
pub enum BodyDraft {
    /// A free-form statement.
    Statement {
        /// The text.
        text: String,
        /// When the described thing happens, if known.
        when: Option<TimeSpan>,
    },
    /// A (subject, predicate, value) triple.
    Triple {
        /// The subject.
        subject: EntityRef,
        /// A predicate from a space schema. A synonym resolves to its
        /// canonical predicate.
        predicate: PredicateId,
        /// The value.
        object: ObjectDraft,
        /// The fact in one sentence, if the writer has one.
        text: Option<String>,
    },
    /// An event.
    Event {
        /// What happened.
        text: String,
        /// Who or what took part.
        participants: Vec<EntityRef>,
        /// When.
        when: TimeSpan,
    },
}

/// A fact to remember.
#[derive(Debug, Clone, PartialEq)]
pub struct FactDraft {
    /// The content.
    pub body: BodyDraft,
    /// Entities mentioned beyond those in the content.
    pub mentions: Vec<EntityRef>,
    /// From when the fact is true in the world; now if `None`.
    pub valid_from: Option<Timestamp>,
    /// For a triple of a single-valued predicate: the current version the
    /// writer saw, or `None` to replace whatever is there. If the current
    /// version differs, the write fails with
    /// [`StoreError::Conflict`](super::StoreError::Conflict) instead of
    /// silently replacing a value the writer never saw.
    pub expected_version: Option<NonZeroU32>,
}

impl FactDraft {
    /// A draft with no mentions, true from now, with no version check.
    pub fn new(body: BodyDraft) -> Self {
        Self {
            body,
            mentions: Vec::new(),
            valid_from: None,
            expected_version: None,
        }
    }
}

/// The source text of a write.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeDraft {
    /// The text.
    pub text: String,
    /// When it happened; now if `None`.
    pub at: Option<Timestamp>,
    /// Where it came from.
    pub source: Key,
}

/// One write transaction: everything is written, or nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct Remember {
    /// The space to write to.
    pub space: SpaceId,
    /// Who writes.
    pub origin: Origin,
    /// The source text, if any.
    pub episode: Option<EpisodeDraft>,
    /// The facts.
    pub facts: Vec<FactDraft>,
}

/// What happened to one fact of a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// A new fact lineage was started.
    Created,
    /// The same value was already current; nothing was written.
    Unchanged,
    /// A new version replaced `previous`.
    Superseded {
        /// The replaced version.
        previous: FactId,
    },
    /// The version was closed without a replacement.
    Closed,
}

/// The result for one fact of a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recorded {
    /// The resulting version: new, or the existing one if unchanged or
    /// closed.
    pub fact: FactId,
    /// What happened.
    pub outcome: Outcome,
}

/// The result of [`Remember`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Remembered {
    /// The episode, if one was given.
    pub episode: Option<EpisodeId>,
    /// One result per fact draft, in order.
    pub facts: Vec<Recorded>,
}

/// Why the current version of a fact is replaced or closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The world changed: the old version was true until `at` (now if
    /// `None`), the new one is true from then on.
    Change {
        /// When the world changed.
        at: Option<Timestamp>,
    },
    /// The old version was a mistake: it was never true.
    Correction,
}

/// The new content of a fact in an [`Update`].
#[derive(Debug, Clone, PartialEq)]
pub struct Replacement {
    /// The content; must keep the kind, and for a triple the subject and
    /// predicate.
    pub body: BodyDraft,
    /// Entities mentioned beyond those in the content.
    pub mentions: Vec<EntityRef>,
}

/// Replaces or closes the current version of a fact.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// The fact.
    pub lineage: Lineage,
    /// The current version the writer saw. Required: an update always says
    /// what it updates.
    pub expected_version: NonZeroU32,
    /// A change in the world or a correction.
    pub reason: Reason,
    /// The new content; `None` closes the fact without a replacement.
    pub replacement: Option<Replacement>,
    /// Who updates.
    pub origin: Origin,
}

/// Which versions of facts a read returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// Current versions only.
    Current,
    /// Versions true in the world at the moment, by what memory knows now.
    TrueAt(Timestamp),
    /// Versions memory held as current at the moment.
    KnownAt(Timestamp),
}

impl View {
    /// Whether `fact` is visible in this view.
    pub fn includes(self, fact: &Fact) -> bool {
        match self {
            Self::Current => fact.is_current(),
            Self::TrueAt(t) => fact.was_true_at(t),
            Self::KnownAt(t) => fact.was_known_at(t),
        }
    }
}
