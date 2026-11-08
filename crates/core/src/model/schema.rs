//! Space schema: entity kinds and predicates.
//!
//! The compiler checks the shape of records; the schema checks their
//! meaning: that `lives_in` points to a place and that a person lives in one
//! place at a time.

use serde::{Deserialize, Serialize};

use super::ids::{EntityKindId, PredicateId, SpaceId};
use super::text::Name;
use super::value::ValueType;

/// How many current values a (subject, predicate) pair may have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cardinality {
    /// One: a new value closes the old one ("lives in", "project goal").
    Single,
    /// Many: values accumulate ("speaks language").
    Many,
}

/// An entity kind in a space schema: `person`, `project`, `event`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityKindDef {
    /// The kind id.
    pub id: EntityKindId,
    /// The space whose schema declares the kind.
    pub space: SpaceId,
    /// The kind name.
    pub name: Name,
}

/// A predicate in a space schema: `lives_in`, `goal`, `span`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateDef {
    /// The predicate id.
    pub id: PredicateId,
    /// The space whose schema declares the predicate.
    pub space: SpaceId,
    /// The predicate name.
    pub name: Name,
    /// Which values are allowed.
    pub object: ValueType,
    /// How many current values one subject may have.
    pub cardinality: Cardinality,
    /// One line: what the predicate means and what its value is, for the
    /// similarity check and for the agent's tool description. Empty for
    /// predicates defined before descriptions existed.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}
