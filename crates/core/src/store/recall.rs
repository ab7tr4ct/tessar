//! Recall: finding facts for a query, with an explanation for each.
//!
//! Each channel ranks candidate facts its own way. The rankings are fused
//! with reciprocal rank fusion (RRF): the scores of different channels are
//! not comparable, their places are. Channels: entities named in the
//! query, BM25 over words, and — if the memory has an embedder — meaning,
//! by cosine similarity of vectors. Time is a filter (the view) plus a
//! bonus for facts inside the query window.
//!
//! The lexical part of the fused score (entities and words) is multiplied
//! by **coverage**: the share of the query's concepts the fact covers (like
//! `coord` in classic Lucene scoring). A concept is an entity named in the
//! query, or a query word that is not part of such a name. Without it, a
//! fact that matches one entity through both channels — by link and by its
//! name as a word — would outrank a fact that covers the whole question.
//! The meaning part is not multiplied: a fact in another language covers
//! no query words and still answers it.
//!
//! **Near spaces** ([`Recall::near`]) — the asking project, the user's
//! space — get a bonus in the meaning channel. Other spaces are still
//! searched: a strong match elsewhere wins over a weak one nearby.

use std::collections::{HashMap, HashSet};

use super::draft::View;
use super::state::State;
use super::text::{fact_text, name_candidates, words};
use super::vector::Quantized;
use crate::model::{Ending, EntityId, Fact, FactId, Interval, SpaceId, TimeSpan, Timestamp};

/// Default RRF constant: how quickly the weight of lower places falls.
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
    /// Spaces whose facts are searched; empty means all. Entities named in
    /// the query may lie in any space: the user, who lives in their own
    /// space, is the subject of facts in every project.
    pub spaces: Vec<SpaceId>,
    /// Spaces close to the asker — their project, their own space. Their
    /// facts get a bonus in the meaning channel; the rest are still
    /// searched.
    pub near: Vec<SpaceId>,
    /// Entities to look for besides those named in the text.
    pub entities: Vec<EntityId>,
    /// Which versions to search.
    pub view: View,
    /// Facts inside this window get a bonus.
    pub window: Option<Interval>,
    /// How many facts to return at most.
    pub limit: usize,
    /// Weight of each channel in fusion.
    pub channels: Channels,
    /// The RRF constant `k` in `1 / (k + place)`: the larger, the flatter
    /// the weight of lower places. Default 60.
    pub rrf_k: f32,
    /// How the channels are combined.
    pub fusion: Fusion,
    /// If set, the window bonus depends on how tightly the fact's `when`
    /// fits the window: with `span` the length of the fact's `when` and
    /// `len` the window length, the bonus is
    /// `1 + (WINDOW_BONUS - 1) * min(1, len / span)`, `WINDOW_BONUS`
    /// being 1.5. A point (and a fact without `when`, which is placed by
    /// its validity) is tight and keeps the full bonus; an open span
    /// ([`TimeSpan::Since`], [`TimeSpan::Until`]) is infinitely wide and
    /// gets none. Default off: any fact inside the window gets the full
    /// bonus.
    pub time_tightness: bool,
}

/// How the channels are combined into one score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fusion {
    /// Weighted reciprocal rank fusion over all channels: every channel
    /// can bring candidates.
    #[default]
    Rrf,
    /// Candidates come only from the meaning channel; the entity and
    /// keyword channels only re-rank them. The score is the meaning RRF
    /// term times `1 + coverage * (entities * [entity hit] + keywords *
    /// [keyword hit])`, where the weights are those of [`Channels`] and
    /// coverage is the share of query concepts the fact covers. Facts
    /// found only by the lexical channels are dropped. Mem0 calls BM25
    /// "a boost signal, not a recall expander". Without the meaning
    /// channel (no embedder, or weight zero) this is the same as
    /// [`Fusion::Rrf`].
    Boost,
}

/// Weight of each channel in fusion: its RRF term is multiplied by it.
/// Zero turns a channel off, which answers "what does this channel add" —
/// the measurement every channel has to pass.
///
/// By default the lexical channels weigh 0.05 against 1 for meaning. With
/// equal weights, a weak keyword match pulled facts above the best match
/// by meaning: RRF keeps places, not how sure a channel is. On a benchmark
/// of 485 real questions, equal weights gave 61% hit@1 against 84% for
/// meaning alone; at 0.05 the words only break near-ties. Without an
/// embedder the weights do not change the order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Channels {
    /// Entities named in the query.
    pub entities: f32,
    /// BM25 over words.
    pub keywords: f32,
    /// Meaning, if the memory has an embedder.
    pub meaning: f32,
}

impl Default for Channels {
    fn default() -> Self {
        Self {
            entities: 0.05,
            keywords: 0.05,
            meaning: 1.0,
        }
    }
}

impl Recall {
    /// A query over current facts in all spaces, up to 10 results.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            spaces: Vec::new(),
            near: Vec::new(),
            entities: Vec::new(),
            view: View::Current,
            window: None,
            limit: 10,
            channels: Channels::default(),
            rrf_k: RRF_K,
            fusion: Fusion::Rrf,
            time_tightness: false,
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
    /// Set if a channel was skipped, with the reason — for example, the
    /// embedder was down.
    pub degraded: Option<String>,
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
#[derive(Debug, Clone, PartialEq)]
pub struct Why {
    /// Query entities the fact refers to.
    pub entities: Vec<EntityId>,
    /// Query words the fact contains.
    pub words: Vec<String>,
    /// Place in the entity channel, from 1.
    pub entity_rank: Option<usize>,
    /// Place in the keyword channel, from 1.
    pub keyword_rank: Option<usize>,
    /// Place in the meaning channel, from 1.
    pub meaning_rank: Option<usize>,
    /// Cosine similarity to the query, without the near bonus; set if the
    /// meaning channel ran.
    pub similarity: Option<f32>,
    /// Whether the fact is in a near space.
    pub near: bool,
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
    /// The query vector, if the meaning channel runs.
    vector: Option<&'q Quantized>,
    /// Added to the similarity of facts in near spaces.
    near_bonus: f32,
}

impl Prepared<'_> {
    fn visible(&self, fact: &Fact) -> bool {
        !fact.is_redacted()
            && (self.query.spaces.is_empty() || self.query.spaces.contains(&fact.space))
            && self.query.view.includes(fact)
    }

    fn near(&self, fact: &Fact) -> bool {
        self.query.near.contains(&fact.space)
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

/// Facts closest in meaning to the query; near spaces get a bonus.
struct MeaningChannel;

impl Channel for MeaningChannel {
    fn search(&self, state: &State, query: &Prepared<'_>) -> Vec<FactId> {
        let (Some(index), Some(vector)) = (state.vectors(), query.vector) else {
            return Vec::new();
        };
        let mut ranked: Vec<(FactId, f32)> = state
            .facts()
            .iter()
            .filter(|f| query.visible(f))
            .filter_map(|f| {
                let similarity = index.similarity(vector, f.id)?;
                let bonus = if query.near(f) { query.near_bonus } else { 0.0 };
                Some((f.id, similarity + bonus))
            })
            .collect();
        let order = |a: &(FactId, f32), b: &(FactId, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if ranked.len() > CANDIDATES {
            ranked.select_nth_unstable_by(CANDIDATES - 1, order);
            ranked.truncate(CANDIDATES);
        }
        ranked.sort_by(order);
        ranked.into_iter().map(|r| r.0).collect()
    }
}

/// Places of a fact in each channel, from 1.
#[derive(Default)]
struct Ranks {
    entity: Option<usize>,
    keyword: Option<usize>,
    meaning: Option<usize>,
}

/// Runs a recall query against the state. `vector` is the query embedded
/// by the caller; without it the meaning channel is skipped.
pub(crate) fn recall(
    state: &State,
    query: &Recall,
    vector: Option<&Quantized>,
    near_bonus: f32,
) -> Recalled {
    let prepared = prepare(state, query, vector, near_bonus);
    let k = query.rrf_k;
    let rrf = |i: usize| 1.0 / (k + (i + 1) as f32);

    let on = query.channels;
    let mut fused: HashMap<FactId, Ranks> = HashMap::new();
    let mut run = |channel: &dyn Channel, place: fn(&mut Ranks) -> &mut Option<usize>| {
        for (i, fact) in channel.search(state, &prepared).into_iter().enumerate() {
            *place(fused.entry(fact).or_default()) = Some(i + 1);
        }
    };
    if on.entities > 0.0 {
        run(&EntityChannel, |r| &mut r.entity);
    }
    if on.keywords > 0.0 {
        run(&KeywordChannel, |r| &mut r.keyword);
    }
    if on.meaning > 0.0 {
        run(&MeaningChannel, |r| &mut r.meaning);
    }

    let boost = query.fusion == Fusion::Boost && on.meaning > 0.0 && prepared.vector.is_some();
    let concepts = prepared.concepts();
    let mut hits: Vec<Hit> = fused
        .into_iter()
        .filter_map(|(id, ranks)| {
            let fact = state.fact(id).filter(|f| !f.is_redacted())?;
            let fact_entities = fact.entities();
            let fact_words: HashSet<String> = words(&fact_text(state, fact)).collect();
            let covered = prepared.covered(&fact_entities, &fact_words);
            let in_window = query.window.is_some_and(|w| in_window(fact, w));
            let lexical = on.entities * ranks.entity.map_or(0.0, |i| rrf(i - 1))
                + on.keywords * ranks.keyword.map_or(0.0, |i| rrf(i - 1));
            let meaning = on.meaning * ranks.meaning.map_or(0.0, |i| rrf(i - 1));
            let coverage = covered as f32 / concepts.max(1) as f32;
            let mut score = if boost {
                ranks.meaning?;
                let hit = |rank: Option<usize>| if rank.is_some() { 1.0 } else { 0.0 };
                meaning
                    * (1.0
                        + coverage
                            * (on.entities * hit(ranks.entity) + on.keywords * hit(ranks.keyword)))
            } else {
                lexical * coverage + meaning
            };
            if in_window {
                score *= window_bonus(fact, query);
            }
            let similarity = prepared
                .vector
                .zip(state.vectors())
                .and_then(|(v, index)| index.similarity(v, id));
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
                    entity_rank: ranks.entity,
                    keyword_rank: ranks.keyword,
                    meaning_rank: ranks.meaning,
                    similarity,
                    near: prepared.near(fact),
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
        degraded: None,
    }
}

fn prepare<'q>(
    state: &State,
    query: &'q Recall,
    vector: Option<&'q Quantized>,
    near_bonus: f32,
) -> Prepared<'q> {
    let mut query_words = Vec::new();
    for word in words(&query.text) {
        if !query_words.contains(&word) {
            query_words.push(word);
        }
    }
    let mut entities: Vec<(EntityId, Vec<String>)> = Vec::new();
    let mut add = |e: EntityId, name: Vec<String>| {
        if !entities.iter().any(|(known, _)| *known == e) {
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
        vector,
        near_bonus,
    }
}

/// The score multiplier for a fact inside the query window.
fn window_bonus(fact: &Fact, query: &Recall) -> f32 {
    let (Some(window), true) = (query.window, query.time_tightness) else {
        return WINDOW_BONUS;
    };
    let window_len = (window.end().as_millis() - window.start().as_millis()) as f64;
    let span_len = match fact.body.when() {
        None | Some(TimeSpan::Point(_)) => return WINDOW_BONUS,
        Some(TimeSpan::Range(r)) => (r.end().as_millis() - r.start().as_millis()) as f64,
        Some(TimeSpan::Duration(d)) => d.as_millis() as f64,
        Some(TimeSpan::Since(_) | TimeSpan::Until(_)) => f64::INFINITY,
    };
    let tight = if span_len <= window_len {
        1.0
    } else {
        window_len / span_len
    };
    1.0 + (WINDOW_BONUS - 1.0) * tight as f32
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
