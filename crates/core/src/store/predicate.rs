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
use super::error::{NotUniversal, StoreError};
use super::op::Op;
use super::plan::{Planner, mint};
use super::text::name_candidates;
use super::vector::Quantized;
use crate::model::{Key, Name, PredicateDef, PredicateId, SpaceId, ValueType};

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
    /// Finds or defines a predicate. A name that already exists in the
    /// space is reused (or conflicts on type or cardinality) without any
    /// check. A new name must pass `check`:
    ///
    /// - format (always): at most [`MAX_NAME_WORDS`] words;
    /// - universality (`Full`): no word is a number, and no run of words
    ///   of the name with `_` as spaces is the key or alias of an entity
    ///   in any space;
    /// - similarity (`Full` with a vector): the best cosine to the vectors
    ///   of existing predicates is below the threshold. A threshold of 1.0
    ///   or more turns this check off: quantized vectors never reach
    ///   exactly 1.0 reliably, so it cannot mean "identical".
    pub(crate) fn define_predicate(
        &mut self,
        space: SpaceId,
        draft: PredicateDraft,
        check: PredicateCheck<'_>,
    ) -> Result<PredicateId, StoreError> {
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
        let words = name_words(&draft.name);
        check_format(&draft.name, &words)?;
        if let PredicateCheck::Full { vector, threshold } = check {
            self.check_universal(&draft.name, &words)?;
            if let Some(vector) = vector
                && let Some((best, similarity)) = state.most_similar_predicate(vector)
                && threshold < 1.0
                && similarity >= threshold
            {
                return Err(StoreError::PredicateSimilar {
                    name: draft.name.to_string(),
                    existing: state.canonical(best),
                    similarity,
                });
            }
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

    /// Rejects names with a number or the name of an entity in them.
    fn check_universal(&self, name: &Name, words: &[&str]) -> Result<(), StoreError> {
        let not_universal = |word: &str, reason: NotUniversal| StoreError::PredicateNotUniversal {
            name: name.to_string(),
            word: word.to_owned(),
            reason,
        };
        if let Some(word) = words.iter().find(|w| is_number(w)) {
            return Err(not_universal(word, NotUniversal::Number));
        }
        for candidate in name_candidates(&words.join(" "), MAX_NAME_WORDS) {
            let names_entity = Key::new(candidate.as_str())
                .is_ok_and(|k| !self.state.entities_named(&k.normalized()).is_empty());
            if names_entity {
                return Err(not_universal(&candidate, NotUniversal::EntityName));
            }
        }
        Ok(())
    }

    /// Makes `from` a synonym of `into`'s canonical predicate. Both must
    /// exist in the state (not only in this transaction); `from` must be
    /// canonical and not `into`'s canonical; value type and cardinality
    /// must match. Merging into a synonym merges into its canonical.
    ///
    /// Errors: [`StoreError::UnknownPredicate`]; [`StoreError::MergeMismatch`]
    /// for another type or cardinality, for `from` equal to the target and
    /// for a `from` that is already a synonym.
    pub(crate) fn merge_predicates(
        &mut self,
        from: PredicateId,
        into: PredicateId,
    ) -> Result<(), StoreError> {
        let state = self.state;
        let f = state
            .predicate(from)
            .ok_or(StoreError::UnknownPredicate(from))?;
        state
            .predicate(into)
            .ok_or(StoreError::UnknownPredicate(into))?;
        let into = state.canonical(into);
        let t = state
            .predicate(into)
            .ok_or(StoreError::UnknownPredicate(into))?;
        if from == into
            || state.canonical(from) != from
            || f.object != t.object
            || f.cardinality != t.cardinality
        {
            return Err(StoreError::MergeMismatch { from, into });
        }
        self.ops.push(Op::MergePredicate { from, into });
        Ok(())
    }

    /// Defines `name` as a synonym of `into`: a definition with `into`'s
    /// type and cardinality (format check only), then a merge, in one
    /// transaction. Returns the canonical predicate. If `name` already
    /// exists it must already be `into` or its synonym, else
    /// [`StoreError::NameTaken`].
    pub(crate) fn alias_predicate(
        &mut self,
        space: SpaceId,
        name: Name,
        description: String,
        into: PredicateId,
    ) -> Result<PredicateId, StoreError> {
        let state = self.state;
        self.open_space(space)?;
        state
            .predicate(into)
            .ok_or(StoreError::UnknownPredicate(into))?;
        let into = state.canonical(into);
        let target = state
            .predicate(into)
            .ok_or(StoreError::UnknownPredicate(into))?;
        if let Some(existing) = state.predicate_named(space, &name) {
            return if existing == into {
                Ok(into)
            } else {
                Err(StoreError::NameTaken(format!("predicate {name}")))
            };
        }
        check_format(&name, &name_words(&name))?;
        let (object, cardinality) = (target.object, target.cardinality);
        let id = mint(&mut self.next.predicates, PredicateId::new)?;
        self.ops.push(Op::DefinePredicate(PredicateDef {
            id,
            space,
            name,
            object,
            cardinality,
            description,
        }));
        self.ops.push(Op::MergePredicate { from: id, into });
        Ok(into)
    }
}

/// The non-empty parts of a name between underscores.
pub(crate) fn name_words(name: &Name) -> Vec<&str> {
    name.as_str().split('_').filter(|w| !w.is_empty()).collect()
}

pub(crate) fn check_format(name: &Name, words: &[&str]) -> Result<(), StoreError> {
    if words.len() > MAX_NAME_WORDS {
        return Err(StoreError::PredicateNameTooLong {
            name: name.to_string(),
            max_words: MAX_NAME_WORDS,
        });
    }
    Ok(())
}

/// Whether a word is a number. `inf` and `nan` are words, not numbers.
fn is_number(word: &str) -> bool {
    word.chars().any(|c| c.is_ascii_digit()) && word.parse::<f64>().is_ok()
}
