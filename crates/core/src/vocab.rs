//! The starter vocabulary: universal predicates every memory begins with.
//!
//! Data only. The consumer (the MCP server, an evaluation harness) applies
//! it to the user's space with
//! [`Memory::define_predicate_unchecked`](crate::store::Memory::define_predicate_unchecked)
//! (the starter names are universal by construction, and they should not be
//! rejected for resembling each other) and shows it to the agent in the
//! description of its `remember` tool.
//! A predicate is many-valued unless declared single: `Single` is only for
//! relations where a new value always replaces the old one.
//! Predicates here are universal: they name a relation, never the thing
//! related (`trip_destination`, not `plans_trip_to_japan`). Values are text,
//! as an LLM writer produces them.

use crate::model::{Cardinality, Name, ValueType};
use crate::store::PredicateDraft;

/// The starter predicates, in a stable order.
pub fn starter() -> Vec<PredicateDraft> {
    use Cardinality::{Many, Single};
    const STARTER: &[(&str, &str, Cardinality)] = &[
        ("lives_in", "the place where the subject lives", Single),
        (
            "works_at",
            "the organization where the subject works",
            Single,
        ),
        ("job_title", "the subject's position at work", Single),
        ("occupation", "what the subject does for a living", Single),
        ("owns", "a thing the subject owns", Many),
        (
            "has_pet",
            "an animal the subject keeps, with its name or kind",
            Many,
        ),
        (
            "prefers",
            "something the subject likes better than the alternatives",
            Many,
        ),
        ("dislikes", "something the subject does not like", Many),
        ("goal", "something the subject wants to achieve", Many),
        ("plans_to", "something the subject intends to do", Many),
        (
            "trip_destination",
            "where the subject plans to travel",
            Many,
        ),
        (
            "personal_best",
            "the subject's best result: the discipline and the result",
            Many,
        ),
        (
            "count_of",
            "how many of something there are: the thing and the number",
            Many,
        ),
        ("hobby", "an activity the subject does for pleasure", Many),
        ("interest", "a topic the subject is interested in", Many),
        ("age", "how old the subject is", Single),
        ("birthday", "the date the subject was born", Single),
        ("relation_to_user", "who the subject is to the user", Single),
        ("uses", "a tool, service or product the subject uses", Many),
        (
            "currently_reading",
            "the book the subject is reading now",
            Many,
        ),
        (
            "health_condition",
            "a medical condition or limitation of the subject",
            Many,
        ),
        ("diet", "the way the subject eats", Single),
    ];
    STARTER
        .iter()
        .map(|&(name, description, cardinality)| PredicateDraft {
            name: Name::new(name).expect("starter names are valid"),
            description: description.to_owned(),
            object: ValueType::Text,
            cardinality,
        })
        .collect()
}
