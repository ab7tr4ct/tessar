//! Values of triple facts and their types in a schema.

use serde::{Deserialize, Serialize};

use super::ModelError;
use super::ids::{EntityId, EntityKindId};
use super::time::{TimeSpan, Timestamp};

/// A finite float: neither NaN nor infinite.
///
/// NaN is not equal to itself and breaks comparisons and sorting, so it is
/// kept out of memory entirely.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Finite(f64);

impl Finite {
    /// The number `x`; fails if it is NaN or infinite.
    pub const fn new(x: f64) -> Result<Self, ModelError> {
        if x.is_finite() {
            Ok(Self(x))
        } else {
            Err(ModelError::NonFiniteFloat(x))
        }
    }

    /// The value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Finite {
    type Error = ModelError;

    fn try_from(x: f64) -> Result<Self, ModelError> {
        Self::new(x)
    }
}

impl From<Finite> for f64 {
    fn from(x: Finite) -> f64 {
        x.0
    }
}

/// The value in a (subject, predicate, value) triple.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Value {
    /// Text.
    Text(String),
    /// An integer.
    Int(i64),
    /// A finite float.
    Float(Finite),
    /// Yes or no.
    Bool(bool),
    /// An instant.
    Time(Timestamp),
    /// An interval or a duration, for example the dates of a vacation.
    Span(TimeSpan),
    /// A reference to an entity; this turns a triple into a link between
    /// two entities.
    Entity(EntityId),
}

/// The type of value a predicate accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueType {
    /// [`Value::Text`].
    Text,
    /// [`Value::Int`].
    Int,
    /// [`Value::Float`].
    Float,
    /// [`Value::Bool`].
    Bool,
    /// [`Value::Time`].
    Time,
    /// [`Value::Span`].
    Span,
    /// [`Value::Entity`]; if a kind is given, only entities of that kind.
    Entity(Option<EntityKindId>),
}

impl ValueType {
    /// Whether `value` has this type.
    ///
    /// `kind_of` returns the kind of an entity by id, or `None` if there is
    /// no such entity; it is called only for [`Value::Entity`].
    pub fn accepts(
        self,
        value: &Value,
        kind_of: impl FnOnce(EntityId) -> Option<EntityKindId>,
    ) -> bool {
        match (self, value) {
            (Self::Text, Value::Text(_))
            | (Self::Int, Value::Int(_))
            | (Self::Float, Value::Float(_))
            | (Self::Bool, Value::Bool(_))
            | (Self::Time, Value::Time(_))
            | (Self::Span, Value::Span(_)) => true,
            (Self::Entity(required), Value::Entity(id)) => {
                kind_of(*id).is_some_and(|kind| required.is_none_or(|r| r == kind))
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PERSON: EntityKindId = EntityKindId::new(0);
    const PLACE: EntityKindId = EntityKindId::new(1);

    /// Entity 1 is a place; no other entities exist.
    fn kind_of(id: EntityId) -> Option<EntityKindId> {
        (id == EntityId::new(1)).then_some(PLACE)
    }

    #[test]
    fn finite_rejects_nan_and_infinity() {
        assert!(Finite::new(f64::NAN).is_err());
        assert!(Finite::new(f64::INFINITY).is_err());
        assert!(Finite::new(f64::NEG_INFINITY).is_err());
        assert_eq!(Finite::new(1.5).unwrap().get(), 1.5);
    }

    #[test]
    fn scalar_types_match_their_values() {
        let no_entities = |_| None;
        assert!(ValueType::Text.accepts(&Value::Text("x".into()), no_entities));
        assert!(ValueType::Int.accepts(&Value::Int(1), no_entities));
        assert!(!ValueType::Int.accepts(&Value::Text("1".into()), no_entities));
        assert!(!ValueType::Text.accepts(&Value::Entity(EntityId::new(1)), no_entities));
    }

    #[test]
    fn entity_type_checks_existence_and_kind() {
        let place = Value::Entity(EntityId::new(1));
        let missing = Value::Entity(EntityId::new(9));
        assert!(ValueType::Entity(None).accepts(&place, kind_of));
        assert!(ValueType::Entity(Some(PLACE)).accepts(&place, kind_of));
        assert!(!ValueType::Entity(Some(PERSON)).accepts(&place, kind_of));
        assert!(!ValueType::Entity(None).accepts(&missing, kind_of));
    }

    #[test]
    fn non_finite_cannot_be_read_from_journal() {
        // JSON has no NaN, but it can hold numbers beyond the f64 range.
        assert!(serde_json::from_str::<Value>(r#"{"float":1e400}"#).is_err());
        assert_eq!(
            serde_json::from_str::<Value>(r#"{"float":2.5}"#).unwrap(),
            Value::Float(Finite::new(2.5).unwrap())
        );
    }
}
