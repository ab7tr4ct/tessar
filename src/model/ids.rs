//! Record identifiers.
//!
//! Every id is the index of a record in a store vector, so lookup by id is
//! O(1). A separate type per record kind means a fact id cannot be passed
//! where an entity id is expected: such a mistake does not compile.

use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! id {
    ($(#[$doc:meta])* $name:ident, $prefix:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(u32);

        impl $name {
            /// The id of record number `raw`.
            pub const fn new(raw: u32) -> Self {
                Self(raw)
            }

            /// The record number.
            pub const fn get(self) -> u32 {
                self.0
            }

            /// The record number as an index into a store vector.
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{}", $prefix, self.0)
            }
        }
    };
}

id!(
    /// A memory space.
    SpaceId,
    "s"
);
id!(
    /// An entity kind from a space schema.
    EntityKindId,
    "k"
);
id!(
    /// A predicate from a space schema.
    PredicateId,
    "p"
);
id!(
    /// An entity.
    EntityId,
    "e"
);
id!(
    /// An episode: a piece of source text.
    EpisodeId,
    "ep"
);
id!(
    /// One specific version of a fact.
    FactId,
    "f"
);

/// The version chain of one fact, identified by its first version.
///
/// References work like in git: a [`Lineage`] is like a branch (always the
/// current version), a [`FactId`] is like a commit (exactly that version,
/// forever).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Lineage(FactId);

impl Lineage {
    /// The chain that starts with version `first`.
    pub const fn starting_at(first: FactId) -> Self {
        Self(first)
    }

    /// The first version of the chain.
    pub const fn first(self) -> FactId {
        self.0
    }
}

impl fmt::Display for Lineage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "l{}", self.0.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_has_prefix_per_record_kind() {
        assert_eq!(FactId::new(7).to_string(), "f7");
        assert_eq!(EntityId::new(3).to_string(), "e3");
        assert_eq!(EpisodeId::new(0).to_string(), "ep0");
        assert_eq!(Lineage::starting_at(FactId::new(7)).to_string(), "l7");
    }

    #[test]
    fn index_is_raw_number() {
        let id = EntityId::new(42);
        assert_eq!(id.get(), 42);
        assert_eq!(id.index(), 42);
    }

    #[test]
    fn serializes_as_plain_number() {
        assert_eq!(serde_json::to_string(&FactId::new(5)).unwrap(), "5");
        let lineage: Lineage = serde_json::from_str("5").unwrap();
        assert_eq!(lineage.first(), FactId::new(5));
    }
}
