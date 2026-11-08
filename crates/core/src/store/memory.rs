//! [`Memory`]: the shared handle to the store.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard};

use super::draft::{PredicateDraft, Recorded, Remember, Remembered, Update, View};
use super::error::StoreError;
use super::op::Op;
use super::plan::Planner;
use super::predicate::{PredicateCheck, check_format, name_words, predicate_text};
use super::recall::{Recall, Recalled, recall};
use super::state::State;
use super::text::{fact_text, fact_text_with};
use super::vector::Quantized;
use crate::clock::Clock;
use crate::embed::{EmbedError, Embedder};
use crate::log::{Commit, Log};
use crate::model::{
    Entity, EntityId, EntityKindId, Episode, EpisodeId, Fact, FactId, Key, Lineage, Name,
    PredicateDef, PredicateId, Space, SpaceId, SpaceKind, Timestamp,
};

/// How many texts go to the embedder in one call when the memory opens.
const EMBED_BATCH: usize = 64;

/// The meaning channel: how facts and queries become vectors.
#[derive(Clone)]
pub struct Semantic {
    /// Computes the vectors.
    pub embedder: Arc<dyn Embedder>,
    /// Length of vectors in the index. Longer ones are cut and
    /// renormalized, which suits Matryoshka models.
    pub dims: usize,
    /// Added to the cosine similarity of facts in the query's near spaces
    /// ([`Recall::near`]). In the model's cosine units, so it is tuned per
    /// model; 0.05 suits voyage-4-nano.
    pub near_bonus: f32,
    /// Cosine similarity at which a new predicate counts as a duplicate
    /// of an existing one and [`Memory::define_predicate`] rejects it. In
    /// the model's cosine units; calibrated per model on known synonyms.
    /// 1.0 or more turns the similarity check off.
    pub predicate_threshold: f32,
}

/// The memory store. Cheap to clone; every clone is the same store, safe
/// to use from any number of threads.
///
/// # Concurrency
///
/// Two locks:
///
/// - the **writer lock** — one writer at a time. The writer checks the
///   request against the state and plans the operations, mints ids as
///   vector lengths, appends the commit to the log and waits for the disk,
///   then applies it. Holding this lock through all of that is what makes
///   ids unique, entity creation race-free and every check still valid at
///   apply time;
/// - the **state lock** — readers share it. The writer takes it
///   exclusively only to apply an already durable commit, which takes
///   microseconds. Readers never wait for the disk.
///
/// A write becomes visible only after it is durable, so nobody ever reads
/// something a crash could take back.
///
/// With the meaning channel on ([`Memory::open_with`]), the writer embeds
/// the new facts after planning and before the log: under the writer lock,
/// not the state lock, so readers go on meanwhile. A fact becomes visible
/// in every channel at once.
///
/// Lost updates are caught by version checks: a writer passes the version
/// it saw, and if someone changed the fact in between the write fails with
/// [`StoreError::Conflict`].
///
/// # Invariant
///
/// The state always equals the replay of the log. If applying a durable
/// commit fails halfway (a bug), the state no longer matches the log and
/// the memory is marked **poisoned**: every later write fails with
/// [`StoreError::Corrupted`] without touching the state, so nothing is
/// planned on a diverged state and the log stays replayable. Reads keep
/// working. Reopening the memory rebuilds the state from the log.
#[derive(Clone)]
pub struct Memory {
    inner: Arc<Inner>,
}

struct Inner {
    writer: Mutex<Writer>,
    state: RwLock<State>,
    clock: Arc<dyn Clock>,
    semantic: Option<Semantic>,
    /// Set when a durable commit failed to apply; see [`Memory`].
    poisoned: AtomicBool,
}

struct Writer {
    log: Box<dyn Log<Op> + Send>,
    /// Time of the last commit; commit times never go back, even if the
    /// system clock does.
    last_at: Timestamp,
}

impl Memory {
    /// Opens the memory without the meaning channel: replays `log` to
    /// rebuild the state.
    pub fn open(
        log: impl Log<Op> + Send + 'static,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, StoreError> {
        Self::open_inner(log, clock, None)
    }

    /// Opens the memory with the meaning channel. Vectors are not in the
    /// log, so every fact version is embedded again here; put a cache in
    /// front of a slow embedder.
    pub fn open_with(
        log: impl Log<Op> + Send + 'static,
        clock: Arc<dyn Clock>,
        semantic: Semantic,
    ) -> Result<Self, StoreError> {
        Self::open_inner(log, clock, Some(semantic))
    }

    fn open_inner(
        mut log: impl Log<Op> + Send + 'static,
        clock: Arc<dyn Clock>,
        semantic: Option<Semantic>,
    ) -> Result<Self, StoreError> {
        let mut state = State::default();
        let mut last_at = Timestamp::from_millis(i64::MIN);
        for commit in log.replay()? {
            for op in commit.ops {
                state.apply(op).map_err(|reason| StoreError::Replay {
                    seq: commit.seq,
                    reason,
                })?;
            }
            last_at = commit.at;
        }
        if let Some(semantic) = &semantic {
            state.enable_vectors(semantic.dims);
            let predicates: Vec<String> = state
                .all_predicates()
                .iter()
                .map(|p| predicate_text(&p.name, &p.description))
                .collect();
            let mut next = 0;
            for batch in predicates.chunks(EMBED_BATCH) {
                for vector in embed_texts(semantic, batch)? {
                    state
                        .push_predicate_vector(PredicateId::new(next), vector)
                        .map_err(StoreError::Internal)?;
                    next += 1;
                }
            }
            // Redacted facts have nothing to embed: they get a cleared slot.
            let texts: Vec<String> = state
                .facts()
                .iter()
                .filter(|f| !f.is_redacted())
                .map(|f| fact_text(&state, f))
                .collect();
            let mut vectors = Vec::with_capacity(texts.len());
            for batch in texts.chunks(EMBED_BATCH) {
                vectors.extend(embed_texts(semantic, batch)?);
            }
            let ids: Vec<(FactId, bool)> = state
                .facts()
                .iter()
                .map(|f| (f.id, f.is_redacted()))
                .collect();
            let mut vectors = vectors.into_iter();
            for (fact, redacted) in ids {
                if redacted {
                    state.push_blank_vector(fact)
                } else {
                    let vector = vectors.next().ok_or_else(|| {
                        StoreError::Internal("fewer vectors than facts".to_owned())
                    })?;
                    state.push_vector(fact, vector)
                }
                .map_err(StoreError::Internal)?;
            }
        }
        Ok(Self {
            inner: Arc::new(Inner {
                writer: Mutex::new(Writer {
                    log: Box::new(log),
                    last_at,
                }),
                state: RwLock::new(state),
                clock,
                semantic,
                poisoned: AtomicBool::new(false),
            }),
        })
    }

    /// Whether the state diverged from the log after an internal error.
    /// A poisoned memory still reads, but every write fails with
    /// [`StoreError::Corrupted`]; reopen it to rebuild the state.
    pub fn is_poisoned(&self) -> bool {
        self.inner.poisoned.load(Ordering::SeqCst)
    }

    /// Finds or creates a space by name. Fails if a space with this name
    /// exists with another kind or parent.
    pub fn create_space(
        &self,
        name: Key,
        kind: SpaceKind,
        parent: Option<SpaceId>,
    ) -> Result<SpaceId, StoreError> {
        self.write(|p| p.create_space(name, kind, parent))
    }

    /// Finds or defines an entity kind in a space schema.
    pub fn define_kind(&self, space: SpaceId, name: Name) -> Result<EntityKindId, StoreError> {
        self.write(|p| p.define_kind(space, name))
    }

    /// Finds or defines a predicate in a space schema, checking that the
    /// name is short and universal and that no existing predicate means
    /// the same (with the meaning channel on). Fails with
    /// [`StoreError::PredicateSimilar`] naming the closest predicate, so
    /// the caller can reuse it or declare a synonym
    /// ([`Memory::alias_predicate`]); with the name existing with another
    /// type or cardinality, [`StoreError::PredicateConflict`].
    ///
    /// Without the meaning channel the similarity check is skipped (the
    /// format and universality checks still run). With it, the draft is
    /// embedded before the writer lock is taken, so a slow embedder does
    /// not block other writers.
    pub fn define_predicate(
        &self,
        space: SpaceId,
        draft: PredicateDraft,
    ) -> Result<PredicateId, StoreError> {
        // Cheap checks first: a name that already exists is reused (or
        // conflicts) without a vector, and a malformed name never reaches
        // the embedder.
        let known = self.state().predicate_named(space, &draft.name).is_some();
        let semantic = match &self.inner.semantic {
            Some(semantic) if !known => {
                check_format(&draft.name, &name_words(&draft.name))?;
                semantic
            }
            _ => {
                return self.write(|p| {
                    p.define_predicate(
                        space,
                        draft,
                        PredicateCheck::Full {
                            vector: None,
                            threshold: 1.0,
                        },
                    )
                });
            }
        };
        let text = predicate_text(&draft.name, &draft.description);
        let vector = embed_texts(semantic, std::slice::from_ref(&text))?
            .pop()
            .ok_or_else(|| EmbedError("no vector for the predicate".to_owned()))?;
        let threshold = semantic.predicate_threshold;
        self.write_with(Some((&text, &vector)), |p| {
            p.define_predicate(
                space,
                draft,
                PredicateCheck::Full {
                    vector: Some(&vector),
                    threshold,
                },
            )
        })
    }

    /// Finds or defines a predicate without the universality and
    /// similarity checks: for the starter vocabulary, or for an agent that
    /// saw the similar predicate and insists. The format is still checked.
    pub fn define_predicate_unchecked(
        &self,
        space: SpaceId,
        draft: PredicateDraft,
    ) -> Result<PredicateId, StoreError> {
        self.write(|p| p.define_predicate(space, draft, PredicateCheck::Format))
    }

    /// Declares `name` a synonym of `into`: new writes with the name go to
    /// `into`, and the two count as one for the replacement rule. Returns
    /// the canonical predicate.
    pub fn alias_predicate(
        &self,
        space: SpaceId,
        name: Name,
        description: String,
        into: PredicateId,
    ) -> Result<PredicateId, StoreError> {
        self.write(|p| p.alias_predicate(space, name, description, into))
    }

    /// Makes the existing predicate `from` a synonym of `into`. Facts
    /// already written with `from` keep it; see [`Op::MergePredicate`].
    pub fn merge_predicates(&self, from: PredicateId, into: PredicateId) -> Result<(), StoreError> {
        self.write(|p| p.merge_predicates(from, into))
    }

    /// A predicate by id, synonym or canonical.
    pub fn predicate(&self, id: PredicateId) -> Option<PredicateDef> {
        self.state().predicate(id).cloned()
    }

    /// The canonical predicate of `id`: itself unless merged into another.
    pub fn canonical_predicate(&self, id: PredicateId) -> Option<PredicateId> {
        let state = self.state();
        state.predicate(id).map(|_| state.canonical(id))
    }

    /// The synonyms of the canonical predicate of `id` (it excluded), in id
    /// order. Empty for an unknown id.
    pub fn synonyms_of(&self, id: PredicateId) -> Vec<PredicateDef> {
        self.state().synonyms_of(id).cloned().collect()
    }

    /// The vocabulary: every canonical predicate, in id order. Synonyms are
    /// left out.
    pub fn predicates(&self) -> Vec<PredicateDef> {
        self.state().canonical_predicates().cloned().collect()
    }

    /// Finds or creates the entity of `kind` with `key` (or that alias), in
    /// the kind's space. Returns its id. An existing entity writes nothing.
    pub fn create_entity(&self, kind: EntityKindId, key: Key) -> Result<EntityId, StoreError> {
        self.write(|p| p.create_entity(kind, key))
    }

    /// Adds another name to an entity.
    pub fn add_alias(&self, entity: EntityId, alias: Key) -> Result<(), StoreError> {
        self.write(|p| p.add_alias(entity, alias))
    }

    /// Writes facts in one transaction: all of them, or none.
    ///
    /// Entities named by key are found or created. A triple of a
    /// single-valued predicate replaces the current value as its next
    /// version. A value that is already current is not written again.
    pub fn remember(&self, request: Remember) -> Result<Remembered, StoreError> {
        self.write(|p| p.remember(request))
    }

    /// Replaces or closes the current version of a fact.
    pub fn update(&self, update: Update) -> Result<Recorded, StoreError> {
        self.write(|p| p.update(update))
    }

    /// Removes the content of a fact version for good: its text leaves
    /// the keyword, entity and meaning indexes, and the version stays in
    /// its chain as [`FactBody::Redacted`](crate::model::FactBody::Redacted)
    /// with its id and times. A current version is closed first as a
    /// correction without a successor, so a single-valued pair accepts a
    /// new value afterwards.
    ///
    /// The text stays in the journal file until `purge`
    /// rewrites it. Fails if the fact is unknown or already redacted.
    pub fn redact(&self, fact: FactId) -> Result<(), StoreError> {
        self.write(|p| p.redact(fact))
    }

    /// Blanks the text of an episode; the record and its id stay. Facts
    /// taken from it keep their link until they are redacted themselves.
    /// Fails if the episode is unknown or already blank.
    pub fn redact_episode(&self, episode: EpisodeId) -> Result<(), StoreError> {
        self.write(|p| p.redact_episode(episode))
    }

    /// An episode by id.
    pub fn episode(&self, id: EpisodeId) -> Option<Episode> {
        self.state().episode(id).cloned()
    }

    /// A space by id.
    pub fn space(&self, id: SpaceId) -> Option<Space> {
        self.state().space(id).cloned()
    }

    /// A space by name.
    pub fn space_named(&self, name: &str) -> Option<Space> {
        let name = Key::new(name).ok()?.normalized();
        let state = self.state();
        state
            .space_named(&name)
            .and_then(|id| state.space(id).cloned())
    }

    /// An entity kind by name.
    pub fn kind_named(&self, space: SpaceId, name: &str) -> Option<EntityKindId> {
        self.state().kind_named(space, &Name::new(name).ok()?)
    }

    /// A predicate by name; a synonym's name gives its canonical predicate.
    pub fn predicate_named(&self, space: SpaceId, name: &str) -> Option<PredicateDef> {
        let name = Name::new(name).ok()?;
        let state = self.state();
        state
            .predicate_named(space, &name)
            .and_then(|id| state.predicate(id).cloned())
    }

    /// An entity by id.
    pub fn entity(&self, id: EntityId) -> Option<Entity> {
        self.state().entity(id).cloned()
    }

    /// An entity by key or alias.
    pub fn find_entity(&self, space: SpaceId, kind: EntityKindId, name: &str) -> Option<Entity> {
        let name = Key::new(name).ok()?.normalized();
        let state = self.state();
        state
            .entity_named(space, kind, &name)
            .and_then(|id| state.entity(id).cloned())
    }

    /// One fact version.
    pub fn fact(&self, id: FactId) -> Option<Fact> {
        self.state().fact(id).cloned()
    }

    /// The current version of a fact, if it has one.
    pub fn current(&self, lineage: Lineage) -> Option<Fact> {
        self.state()
            .head(lineage)
            .filter(|f| f.is_current() && !f.is_redacted())
            .cloned()
    }

    /// Every version of a fact, oldest first.
    pub fn history(&self, lineage: Lineage) -> Vec<Fact> {
        let state = self.state();
        let mut versions = Vec::new();
        let mut next = state.head(lineage);
        while let Some(fact) = next {
            versions.push(fact.clone());
            next = fact.prev.and_then(|id| state.fact(id));
        }
        versions.reverse();
        versions
    }

    /// Facts about an entity — as subject, value, participant or mention —
    /// visible in `view`, oldest first.
    pub fn facts_about(&self, entity: EntityId, view: View) -> Vec<Fact> {
        let state = self.state();
        state
            .facts_of(entity)
            .iter()
            .filter_map(|&id| state.fact(id))
            .filter(|f| view.includes(f))
            .cloned()
            .collect()
    }

    /// Finds facts for a query: by entities named in it, by its words and,
    /// with the meaning channel on, by meaning; fused, best first, each
    /// with an explanation.
    ///
    /// If the embedder fails, the query runs without the meaning channel
    /// and [`Recalled::degraded`] says why.
    pub fn recall(&self, query: &Recall) -> Recalled {
        let mut degraded = None;
        let mut vector = None;
        if let Some(semantic) = &self.inner.semantic
            && query.channels.meaning > 0.0
            && !query.text.trim().is_empty()
        {
            match semantic
                .embedder
                .embed_query(&query.text)
                .and_then(|v| Quantized::new(&v, semantic.dims))
            {
                Ok(v) => vector = Some(v),
                Err(e) => degraded = Some(format!("meaning channel skipped: {e}")),
            }
        }
        let near_bonus = self.inner.semantic.as_ref().map_or(0.0, |s| s.near_bonus);
        let mut recalled = recall(&self.state(), query, vector.as_ref(), near_bonus);
        recalled.degraded = degraded;
        recalled
    }

    /// The current values of a (subject, predicate) pair; a synonym gives
    /// the values of its canonical predicate.
    ///
    /// A single-valued pair with an asserted and an inferred current value
    /// is a conflict ([`Outcome::Conflicting`](super::Outcome)): the
    /// inferred one never replaced the asserted one.
    pub fn values_of(&self, subject: EntityId, predicate: PredicateId) -> Vec<Fact> {
        let state = self.state();
        let predicate = state.canonical(predicate);
        state
            .lineages_of_pair(subject, predicate)
            .iter()
            .filter_map(|&l| state.head(l))
            .filter(|f| f.is_current() && !f.is_redacted())
            .cloned()
            .collect()
    }

    /// Runs one write transaction; see the type docs for the protocol.
    fn write<R>(
        &self,
        plan: impl FnOnce(&mut Planner<'_>) -> Result<R, StoreError>,
    ) -> Result<R, StoreError> {
        self.write_with(None, plan)
    }

    /// Like [`Memory::write`], with the vector of one text already
    /// embedded (the predicate being defined), so it is not embedded twice.
    fn write_with<R>(
        &self,
        known: Option<(&str, &Quantized)>,
        plan: impl FnOnce(&mut Planner<'_>) -> Result<R, StoreError>,
    ) -> Result<R, StoreError> {
        let mut writer = self.inner.writer.lock().map_err(|_| StoreError::Poisoned)?;
        // Checked under the writer lock: a writer that poisons the memory
        // holds it, so no later writer plans on a diverged state.
        if self.is_poisoned() {
            return Err(StoreError::Corrupted);
        }
        let at = self.inner.clock.now().max(writer.last_at);

        let (ops, result, texts, predicate_texts) = {
            let state = self.state();
            let mut planner = Planner::new(&state, at);
            let result = plan(&mut planner)?;
            let ops = planner.into_ops();
            let (texts, predicate_texts) = match self.inner.semantic {
                Some(_) => (new_fact_texts(&state, &ops), new_predicate_texts(&ops)),
                None => (Vec::new(), Vec::new()),
            };
            (ops, result, texts, predicate_texts)
        };
        if ops.is_empty() {
            // Nothing to change, for example a value that is already current.
            return Ok(result);
        }
        // Embedding is slow; it runs before the log, so a failure writes
        // nothing, and without the state lock, so readers go on.
        let mut predicate_vectors: Vec<(PredicateId, Quantized)> = Vec::new();
        // One batch for the new predicates, then the new facts. Ids and
        // texts are kept apart so no text is cloned.
        let mut embed_ids: Vec<PredicateId> = Vec::new();
        let mut strings: Vec<String> = Vec::new();
        for (id, text) in predicate_texts {
            match known {
                Some((t, q)) if t == text => predicate_vectors.push((id, q.clone())),
                _ => {
                    embed_ids.push(id);
                    strings.push(text);
                }
            }
        }
        let new_predicates = embed_ids.len();
        let (fact_ids, fact_strings): (Vec<FactId>, Vec<String>) = texts.into_iter().unzip();
        strings.extend(fact_strings);
        let vectors: Vec<(FactId, Quantized)> = match &self.inner.semantic {
            Some(semantic) if !strings.is_empty() => {
                let mut all = embed_texts(semantic, &strings)?;
                let fact_vectors = all.split_off(new_predicates);
                predicate_vectors.extend(embed_ids.into_iter().zip(all));
                fact_ids.into_iter().zip(fact_vectors).collect()
            }
            _ => Vec::new(),
        };

        let commit = Commit {
            seq: writer.log.next_seq(),
            at,
            ops,
        };
        // Protocol: check not poisoned, plan, embed, append, apply. Once the
        // append succeeds the commit is durable; an apply error after it
        // sets `poisoned` (see the invariant on `Memory`).
        // The slow part — the disk — runs without the state lock: readers
        // keep reading the previous state meanwhile.
        writer.log.append(&commit)?;
        writer.last_at = at;

        let mut state = self
            .inner
            .state
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        // The commit is durable, so a failure from here on leaves the state
        // partially applied: poison the memory before reporting it.
        let applied = (|| {
            for op in commit.ops {
                state.apply(op)?;
            }
            for (id, vector) in predicate_vectors {
                state.push_predicate_vector(id, vector)?;
            }
            for (fact, vector) in vectors {
                state.push_vector(fact, vector)?;
            }
            Ok(())
        })();
        if let Err(reason) = applied {
            self.inner.poisoned.store(true, Ordering::SeqCst);
            return Err(StoreError::Internal(reason));
        }
        Ok(result)
    }

    /// Shared access to the state.
    ///
    /// The state lock is poisoned only if a thread panicked while holding
    /// it. Writers mutate the state only by applying planned operations,
    /// which do not panic, so the state stays consistent and reads go on.
    fn state(&self) -> RwLockReadGuard<'_, State> {
        self.inner
            .state
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Texts of the facts a commit adds, as the indexes will see them.
fn new_fact_texts(state: &State, ops: &[Op]) -> Vec<(FactId, String)> {
    let new_entities: Vec<&Entity> = ops
        .iter()
        .filter_map(|op| match op {
            Op::CreateEntity(e) => Some(e),
            _ => None,
        })
        .collect();
    ops.iter()
        .filter_map(|op| match op {
            Op::AddFact(f) => Some((f.id, fact_text_with(state, f, &new_entities))),
            _ => None,
        })
        .collect()
}

/// Texts of the predicates a commit defines.
fn new_predicate_texts(ops: &[Op]) -> Vec<(PredicateId, String)> {
    ops.iter()
        .filter_map(|op| match op {
            Op::DefinePredicate(p) => Some((p.id, predicate_text(&p.name, &p.description))),
            _ => None,
        })
        .collect()
}

/// Embeds texts as documents and prepares the vectors for the index, one
/// per text.
fn embed_texts(semantic: &Semantic, texts: &[String]) -> Result<Vec<Quantized>, EmbedError> {
    let vectors = semantic.embedder.embed_documents(texts)?;
    if vectors.len() != texts.len() {
        return Err(EmbedError(format!(
            "{} vectors for {} texts",
            vectors.len(),
            texts.len()
        )));
    }
    vectors
        .iter()
        .map(|v| Quantized::new(v, semantic.dims))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;
    use crate::log::MemLog;
    use crate::model::SpaceKind;

    #[test]
    fn a_poisoned_memory_rejects_writes_and_keeps_reading() {
        let clock = Arc::new(ManualClock::new(Timestamp::from_millis(1)));
        let memory = Memory::open(MemLog::<Op>::default(), clock).unwrap();
        let key = |s: &str| Key::new(s).unwrap();
        let space = memory
            .create_space(key("work"), SpaceKind::LongTerm, None)
            .unwrap();
        assert!(!memory.is_poisoned());

        memory.inner.poisoned.store(true, Ordering::SeqCst);
        assert!(memory.is_poisoned());
        let err = memory
            .create_space(key("home"), SpaceKind::LongTerm, None)
            .unwrap_err();
        assert!(matches!(err, StoreError::Corrupted), "{err}");
        assert!(memory.space_named("home").is_none());
        assert_eq!(memory.space_named("work").map(|s| s.id), Some(space));
        assert!(memory.space(space).is_some());
    }
}
