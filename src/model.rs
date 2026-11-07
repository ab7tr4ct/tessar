//! Data model: typed memory records.
//!
//! Records live in the store's vectors and an id is an index into one of
//! them; links between records are ids too, never pointers. This module
//! holds only the types and their invariants. Store rules (transactions,
//! versioning, indexes) live elsewhere.

mod ids;
mod origin;
mod record;
mod schema;
mod text;
mod time;
mod value;

pub use ids::{EntityId, EntityKindId, EpisodeId, FactId, Lineage, PredicateId, SpaceId};
pub use origin::{Confidence, Origin};
pub use record::{Ending, Entity, Episode, Fact, FactBody, FactEnd, Space, SpaceKind, SpaceState};
pub use schema::{Cardinality, EntityKindDef, PredicateDef};
pub use text::{Key, Name};
pub use time::{Interval, TimeDelta, TimeSpan, Timestamp};
pub use value::{Finite, Value, ValueType};
