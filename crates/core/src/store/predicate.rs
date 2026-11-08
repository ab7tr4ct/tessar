//! Planning the predicate vocabulary: definitions, checks and merges.
//!
//! The model names one thing many ways (`plans_to`, `plans_trip_to`,
//! `trip_destination`…), and the replacement rule only works on an exact
//! predicate match. So the agent extends the vocabulary and the core
//! decides, without an LLM, whether a new name may enter: it must be short
//! and universal, and not too close in meaning to an existing predicate.
//! A name that slipped through is later declared a synonym
//! ([`Op::MergePredicate`]).

use super::draft::PredicateDraft;
use super::error::StoreError;
use super::op::Op;
use super::plan::{Planner, mint};
use super::vector::Quantized;
use crate::model::{Name, PredicateDef, PredicateId, SpaceId, ValueType};

/// The most words a predicate name may have.
pub(crate) const MAX_NAME_WORDS: usize = 4;

/// How a new predicate is checked against the vocabulary.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PredicateCheck<'a> {
    /// Format only. For the starter vocabulary and for an agent that
    /// insists after seeing the similar predicate.
    Format,
    /// Format, universality and, with a vector, similarity: a cosine of
    /// `threshold` or more to an existing predicate rejects the name with
    /// [`StoreError::PredicateSimilar`].
    Full {
        /// The vector of "name words: description", if the memory has an
        /// embedder.
        vector: Option<&'a Quantized>,
        /// The cosine at which two predicates count as one.
        threshold: f32,
    },
}

/// The text a predicate is embedded from: the words of its name, then its
/// description. The same for the check and for the index.
pub(crate) fn predicate_text(name: &Name, description: &str) -> String {
    let words = name.as_str().replace('_', " ");
    if description.trim().is_empty() {
        words
    } else {
        format!("{words}: {}", description.trim())
    }
}

impl Planner<'_> {
    /// Finds or defines a predicate. Fails if the name exists with another
    /// type or cardinality, or if `check` rejects it.
    ///
    /// TODO(core-B): the checks. Format: at most [`MAX_NAME_WORDS`] words.
    /// Universality: no word is a number, no word (and not the whole name
    /// with `_` as spaces) is the normalized key or alias of an existing
    /// entity ([`State::entities_named`](super::state::State::entities_named)).
    /// Similarity: the best cosine against the vectors of all existing
    /// canonical predicates.
    pub(crate) fn define_predicate(
        &mut self,
        space: SpaceId,
        draft: PredicateDraft,
        check: PredicateCheck<'_>,
    ) -> Result<PredicateId, StoreError> {
        let _ = check;
        let state = self.state;
        self.open_space(space)?;
        if let Some(id) = state.predicate_named(space, &draft.name) {
            let existing = state
                .predicate(id)
                .ok_or(StoreError::UnknownPredicate(id))?;
            return if existing.object == draft.object && existing.cardinality == draft.cardinality {
                Ok(id)
            } else {
                Err(StoreError::PredicateConflict {
                    name: draft.name.to_string(),
                })
            };
        }
        if let ValueType::Entity(Some(kind)) = draft.object {
            state.kind(kind).ok_or(StoreError::UnknownKind(kind))?;
        }
        let id = mint(&mut self.next.predicates, PredicateId::new)?;
        self.ops.push(Op::DefinePredicate(PredicateDef {
            id,
            space,
            name: draft.name,
            object: draft.object,
            cardinality: draft.cardinality,
            description: draft.description,
        }));
        Ok(id)
    }

    /// Makes `from` a synonym of `into`'s canonical predicate.
    ///
    /// TODO(core-B): both must exist; `from` must be canonical and not
    /// `into`; value type and cardinality must match
    /// ([`StoreError::MergeMismatch`]). Merging into a synonym merges into
    /// its canonical predicate.
    pub(crate) fn merge_predicates(
        &mut self,
        from: PredicateId,
        into: PredicateId,
    ) -> Result<(), StoreError> {
        let _ = (from, into);
        todo!("core-B: merge_predicates")
    }

    /// Defines `name` as a synonym of `into`: a definition with `into`'s
    /// type and cardinality, then a merge, in one transaction. Returns the
    /// canonical predicate. If `name` already exists it must already be
    /// `into` or its synonym.
    ///
    /// TODO(core-B).
    pub(crate) fn alias_predicate(
        &mut self,
        space: SpaceId,
        name: Name,
        description: String,
        into: PredicateId,
    ) -> Result<PredicateId, StoreError> {
        let _ = (space, name, description, into);
        todo!("core-B: alias_predicate")
    }
}
