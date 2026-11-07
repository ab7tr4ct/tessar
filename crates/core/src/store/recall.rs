//! Recall: finding facts for a query, with an explanation for each.
//!
//! Each channel ranks candidate facts its own way. The rankings are fused
//! with reciprocal rank fusion (RRF): the scores of different channels are
//! not comparable, their places are. Channels in v0: entities named in the
//! query, and BM25 over words. Time is a filter (the view) plus a bonus
//! for facts inside the query window.
//!
//! The fused score is multiplied by **coverage**: the share of the query's
//! concepts the fact covers (like `coord` in classic Lucene scoring). A
//! concept is an entity named in the query, or a query word that is not
//! part of such a name. Without it, a fact that matches one entity through
//! both channels — by link and by its name as a word — would outrank a fact
//! that covers the whole question.

use std::collections::{HashMap, HashSet};

use super::draft::View;
use super::state::State;
use super::text::{fact_text, name_candidates, words};
use crate::model::{Ending, EntityId, Fact, FactId, Interval, SpaceId, Timestamp};

/// RRF constant: how quickly the weight of lower places falls.
const RRF_K: f32 = 60.0;
/// How many candidates each channel passes to fusion.
const CANDIDATES: usize = 100;
/// The longest entity name looked up in a query, in words.
const MAX_NAME_WORDS: usize = 5;
/// Score multiplier for a fact inside the query window.
const WINDOW_BONUS: f32 = 1.5;

/// A recall query.
#[derive(Debug, Clone, PartialEq)]
pub struct Recall {
    /// The question, in natural language.
    pub text: String,
    /// Spaces to search; empty means all.
    pub spaces: Vec<SpaceId>,
    /// Entities to look for besides those named in the text.
    pub entities: Vec<EntityId>,
    /// Which versions to search.
    pub view: View,
    /// Facts inside this window get a bonus.
    pub window: Option<Interval>,
    /// How many facts to return at most.
    pub limit: usize,
}

impl Recall {
    /// A query over current facts in all spaces, up to 10 results.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            spaces: Vec::new(),
            entities: Vec::new(),
            view: View::Current,
            window: None,
            limit: 10,
        }
    }
}

/// The result of a recall.
#[derive(Debug, Clone, PartialEq)]
pub struct Recalled {
    /// Entities named in the query text or given explicitly.
    pub entities: Vec<EntityId>,
    /// Facts, best first.
    pub hits: Vec<Hit>,
}

/// One found fact.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// The fact version.
    pub fact: Fact,
    /// The fused score; only the order matters.
    pub score: f32,
    /// Why it was found.
    pub why: Why,
}

/// Why a fact was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Why {
    /// Query entities the fact refers to.
    pub entities: Vec<EntityId>,
    /// Query words the fact contains.
    pub words: Vec<String>,
    /// Place in the entity channel, from 1.
    pub entity_rank: Option<usize>,
    /// Place in the keyword channel, from 1.
    pub keyword_rank: Option<usize>,
    /// Whether the fact falls inside the query window.
    pub in_window: bool,
    /// How many query concepts the fact covers.
    pub covered: usize,
    /// How many concepts the query has.
    pub concepts: usize,
}

/// A query prepared for the channels.
struct Prepared<'q> {
    query: &'q Recall,
    /// Distinct query words, in order.
    words: Vec<String>,
    /// Distinct query entities, in order, each with the words of the name
    /// it was found by (empty if given explicitly).
    entities: Vec<(EntityId, Vec<String>)>,
    /// Query words that are not part of an entity name.
    plain_words: Vec<String>,
}

impl Prepared<'_> {
    fn visible(&self, fact: &Fact) -> bool {
        (self.query.spaces.is_empty() || self.query.spaces.contains(&fact.space))
            && self.query.view.includes(fact)
    }

    fn concepts(&self) -> usize {
        self.entities.len() + self.plain_words.len()
    }

    /// How many concepts a fact covers. An entity counts if the fact links
    /// to it or contains every word of its name.
    fn covered(&self, fact_entities: &[EntityId], fact_words: &HashSet<String>) -> usize {
        let entities = self
            .entities
            .iter()
            .filter(|(e, name)| {
                fact_entities.contains(e)
                    || (!name.is_empty() && name.iter().all(|w| fact_words.contains(w)))
            })
            .count();
        let words = self
            .plain_words
            .iter()
            .filter(|w| fact_words.contains(*w))
            .count();
        entities + words
    }
}

/// A way to find candidate facts.
trait Channel {
    /// Visible candidate facts, best first, at most [`CANDIDATES`].
    fn search(&self, state: &State, query: &Prepared<'_>) -> Vec<FactId>;
}

/// Facts that refer to the query entities; more entities first, then newer.
struct EntityChannel;

impl Channel for EntityChannel {
    fn search(&self, state: &State, query: &Prepared<'_>) -> Vec<FactId> {
        let mut matched: HashMap<FactId, usize> = HashMap::new();
        for &(entity, _) in &query.entities {
            for &fact in state.facts_of(entity) {
                *matched.entry(fact).or_default() += 1;
            }
        }
        let mut ranked: Vec<(FactId, usize, Timestamp)> = matched
            .into_iter()
            .filter_map(|(id, n)| {
                let fact = state.fact(id)?;
                query.visible(fact).then_some((id, n, fact.recorded_at))
            })
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));
        ranked.into_iter().take(CANDIDATES).map(|r| r.0).collect()
    }
}

/// BM25 over the words of facts.
struct KeywordChannel;

impl Channel for KeywordChannel {
    fn search(&self, state: &State, query: &Prepared<'_>) -> Vec<FactId> {
        let mut ranked: Vec<(FactId, f32, Timestamp)> = state
            .text()
            .search(&query.words)
            .into_iter()
            .filter_map(|(id, score)| {
                let fact = state.fact(id)?;
                query.visible(fact).then_some((id, score, fact.recorded_at))
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));
        ranked.into_iter().take(CANDIDATES).map(|r| r.0).collect()
    }
}

/// Runs a recall query against the state.
pub(crate) fn recall(state: &State, query: &Recall) -> Recalled {
    let prepared = prepare(state, query);
    let channels: [&dyn Channel; 2] = [&EntityChannel, &KeywordChannel];

    let mut fused: HashMap<FactId, (f32, [Option<usize>; 2])> = HashMap::new();
    for (c, channel) in channels.iter().enumerate() {
        for (i, fact) in channel.search(state, &prepared).into_iter().enumerate() {
            let entry = fused.entry(fact).or_default();
            entry.0 += 1.0 / (RRF_K + (i + 1) as f32);
            entry.1[c] = Some(i + 1);
        }
    }

    let concepts = prepared.concepts();
    let mut hits: Vec<Hit> = fused
        .into_iter()
        .filter_map(|(id, (rrf, ranks))| {
            let fact = state.fact(id)?;
            let fact_entities = fact.entities();
            let fact_words: HashSet<String> = words(&fact_text(state, fact)).collect();
            let covered = prepared.covered(&fact_entities, &fact_words);
            let in_window = query.window.is_some_and(|w| in_window(fact, w));
            let mut score = rrf * covered as f32 / concepts.max(1) as f32;
            if in_window {
                score *= WINDOW_BONUS;
            }
            Some(Hit {
                fact: fact.clone(),
                score,
                why: Why {
                    entities: prepared
                        .entities
                        .iter()
                        .map(|(e, _)| *e)
                        .filter(|e| fact_entities.contains(e))
                        .collect(),
                    words: prepared
                        .words
                        .iter()
                        .filter(|w| fact_words.contains(*w))
                        .cloned()
                        .collect(),
                    entity_rank: ranks[0],
                    keyword_rank: ranks[1],
                    in_window,
                    covered,
                    concepts,
                },
            })
        })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(b.fact.recorded_at.cmp(&a.fact.recorded_at))
            .then(a.fact.id.cmp(&b.fact.id))
    });
    hits.truncate(query.limit);
    Recalled {
        entities: prepared.entities.into_iter().map(|(e, _)| e).collect(),
        hits,
    }
}

fn prepare<'q>(state: &State, query: &'q Recall) -> Prepared<'q> {
    let mut query_words = Vec::new();
    for word in words(&query.text) {
        if !query_words.contains(&word) {
            query_words.push(word);
        }
    }
    let in_scope = |e: &EntityId| {
        query.spaces.is_empty()
            || state
                .entity(*e)
                .is_some_and(|entity| query.spaces.contains(&entity.space))
    };
    let mut entities: Vec<(EntityId, Vec<String>)> = Vec::new();
    let mut add = |e: EntityId, name: Vec<String>| {
        if in_scope(&e) && !entities.iter().any(|(known, _)| *known == e) {
            entities.push((e, name));
        }
    };
    for &e in &query.entities {
        add(e, Vec::new());
    }
    for name in name_candidates(&query.text, MAX_NAME_WORDS) {
        for &e in state.entities_named(&name) {
            add(e, words(&name).collect());
        }
    }
    let name_words: HashSet<&String> = entities.iter().flat_map(|(_, name)| name).collect();
    let plain_words = query_words
        .iter()
        .filter(|w| !name_words.contains(w))
        .cloned()
        .collect();
    Prepared {
        query,
        words: query_words,
        entities,
        plain_words,
    }
}

/// Whether a fact falls inside a window: by when it happens, or else by
/// when it is true.
fn in_window(fact: &Fact, window: Interval) -> bool {
    if let Some(span) = fact.body.when() {
        return span.overlaps(window) == Some(true);
    }
    let valid_to = match fact.end.map(|end| end.ending) {
        None => None,
        Some(Ending::Changed { valid_to }) => Some(valid_to),
        Some(Ending::Corrected) => return false,
    };
    fact.valid_from < window.end() && valid_to.is_none_or(|to| window.start() < to)
}
