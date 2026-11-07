//! Operations: changes to memory as they are written to the log.

use serde::{Deserialize, Serialize};

use crate::model::{
    Entity, EntityId, EntityKindDef, Episode, Fact, FactEnd, FactId, Key, PredicateDef, Space,
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
}
