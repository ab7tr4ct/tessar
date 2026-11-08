//! The starter vocabulary: universal predicates every memory begins with.
//!
//! Data only. The consumer (the MCP server, an evaluation harness) applies
//! it to the user's space with
//! [`Memory::define_predicate_unchecked`](crate::store::Memory::define_predicate_unchecked)
//! and shows it to the agent in the description of its `remember` tool.
//! Predicates here are universal: they name a relation, never the thing
//! related (`trip_destination`, not `plans_trip_to_japan`).

use crate::store::PredicateDraft;

/// The starter predicates, in a stable order.
///
/// TODO(core-B): about twenty predicates with one-line descriptions, value
/// types and cardinalities: `lives_in`, `works_at`, `job_title`, `owns`,
/// `has_pet`, `prefers`, `goal`, `plans_to`, `trip_destination`,
/// `personal_best`, `count_of`, `hobby`, `interest`, `age`, `birthday`,
/// `relation_to_user`, `uses`, `currently_reading`, `health_condition`,
/// `diet`…
pub fn starter() -> Vec<PredicateDraft> {
    Vec::new()
}
