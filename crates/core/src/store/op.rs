//! Operations: changes to memory as they are written to the log.

use serde::{Deserialize, Serialize};

use crate::model::{
    Entity, EntityId, EntityKindDef, Episode, EpisodeId, Fact, FactEnd, FactId, Key, PredicateDef,
    PredicateId, Space,
};

/// One change to memory, as written to the log.
///
/// New records carry their ids. On replay the store checks that each new
/// record gets exactly the id it had, so a bug or a damaged log is caught
/// at once instead of silently mixing up links.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// A new space.
    CreateSpace(Space),
    /// A new entity kind in a space schema.
    DefineKind(EntityKindDef),
    /// A new predicate in a space schema.
    DefinePredicate(PredicateDef),
    /// `from` becomes a synonym of `into`: lookups by name and new writes
    /// with `from` resolve to `into`, and the two count as one predicate
    /// for the replacement rule. Facts already written with `from` keep
    /// it. Both must have the same value type and cardinality.
    ///
    /// Merging leaves existing values untouched, so a single-valued pair
    /// can end up with several current values. The next write to that pair
    /// closes them all (see `Memory::remember`).
    MergePredicate {
        /// The predicate that becomes a synonym.
        from: PredicateId,
        /// The canonical predicate.
        into: PredicateId,
    },
    /// A new entity.
    CreateEntity(Entity),
    /// Another name for an existing entity.
    AddAlias {
        /// The entity.
        entity: EntityId,
        /// The new name.
        alias: Key,
    },
    /// A new episode.
    AddEpisode(Episode),
    /// A new fact version: the first one of a lineage, or the next one.
    AddFact(Fact),
    /// Closes a fact version.
    CloseFact {
        /// The version to close.
        fact: FactId,
        /// How and when it was closed.
        end: FactEnd,
    },
    /// Removes the content of a fact version for good: its body becomes
    /// [`FactBody::Redacted`](crate::model::FactBody::Redacted), its
    /// mentions and episode link are dropped, and it leaves the keyword,
    /// entity and meaning indexes. Ids, chain links, times and the ending
    /// stay.
    ///
    /// Replay accepts a fact that is already redacted (a purged journal
    /// has the content gone before this op), so applying it twice is a
    /// no-op; the planner still refuses to redact twice.
    Redact {
        /// The version to redact.
        fact: FactId,
    },
    /// Blanks the text of an episode; the record and its id stay. Like
    /// [`Op::Redact`], harmless to apply twice.
    RedactEpisode {
        /// The episode to blank.
        episode: EpisodeId,
    },
}
