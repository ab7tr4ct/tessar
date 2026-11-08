//! [`Memory`]: the shared handle to the store.

use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard};

use super::draft::{PredicateDraft, Recorded, Remember, Remembered, Update, View};
use super::error::StoreError;
use super::op::Op;
use super::plan::Planner;
use super::predicate::PredicateCheck;
use super::recall::{Recall, Recalled, recall};
use super::state::State;
use super::text::{fact_text, fact_text_with};
use super::vector::Quantized;
use crate::clock::Clock;
use crate::embed::{EmbedError, Embedder};
use crate::log::{Commit, Log};
use crate::model::{
    Entity, EntityId, EntityKindId, Fact, FactId, Key, Lineage, Name, PredicateDef, PredicateId,
    Space, SpaceId, SpaceKind, Timestamp,
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
#[derive(Clone)]
pub struct Memory {
    inner: Arc<Inner>,
}

struct Inner {
    writer: Mutex<Writer>,
    state: RwLock<State>,
    clock: Arc<dyn Clock>,
    semantic: Option<Semantic>,
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
            let texts: Vec<(FactId, String)> = state
                .facts()
                .iter()
                .map(|f| (f.id, fact_text(&state, f)))
                .collect();
            for batch in texts.chunks(EMBED_BATCH) {
                for (fact, vector) in embed(semantic, batch)? {
                    state
                        .push_vector(fact, vector)
                        .map_err(StoreError::Internal)?;
                }
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
            }),
        })
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
    /// TODO(core-B): embed `predicate_text(name, description)` before
    /// taking the writer lock and pass it in `PredicateCheck::Full`.
    pub fn define_predicate(
        &self,
        space: SpaceId,
        draft: PredicateDraft,
    ) -> Result<PredicateId, StoreError> {
        let threshold = self
            .inner
            .semantic
            .as_ref()
            .map_or(1.0, |s| s.predicate_threshold);
        self.write(|p| {
            p.define_predicate(
                space,
                draft,
                PredicateCheck::Full {
                    vector: None,
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

    /// The vocabulary: every canonical predicate, in id order. Synonyms are
    /// left out.
    ///
    /// TODO(core-B): implement via a `State` accessor.
    pub fn predicates(&self) -> Vec<PredicateDef> {
        todo!("core-B: predicates")
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
            .filter(|f| f.is_current())
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

    /// The current values of a (subject, predicate) pair.
    pub fn values_of(&self, subject: EntityId, predicate: PredicateId) -> Vec<Fact> {
        let state = self.state();
        state
            .lineages_of_pair(subject, predicate)
            .iter()
            .filter_map(|&l| state.head(l))
            .filter(|f| f.is_current())
            .cloned()
            .collect()
    }

    /// Runs one write transaction; see the type docs for the protocol.
    fn write<R>(
        &self,
        plan: impl FnOnce(&mut Planner<'_>) -> Result<R, StoreError>,
    ) -> Result<R, StoreError> {
        let mut writer = self.inner.writer.lock().map_err(|_| StoreError::Poisoned)?;
        let at = self.inner.clock.now().max(writer.last_at);

        let (ops, result, texts) = {
            let state = self.state();
            let mut planner = Planner::new(&state, at);
            let result = plan(&mut planner)?;
            let ops = planner.into_ops();
            let texts = match self.inner.semantic {
                Some(_) => new_fact_texts(&state, &ops),
                None => Vec::new(),
            };
            (ops, result, texts)
        };
        if ops.is_empty() {
            // Nothing to change, for example a value that is already current.
            return Ok(result);
        }
        // Embedding is slow; it runs before the log, so a failure writes
        // nothing, and without the state lock, so readers go on.
        let vectors = match &self.inner.semantic {
            Some(semantic) if !texts.is_empty() => embed(semantic, &texts)?,
            _ => Vec::new(),
        };

        let commit = Commit {
            seq: writer.log.next_seq(),
            at,
            ops,
        };
        // The slow part — the disk — runs without the state lock: readers
        // keep reading the previous state meanwhile.
        writer.log.append(&commit)?;
        writer.last_at = at;

        let mut state = self
            .inner
            .state
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        for op in commit.ops {
            state.apply(op).map_err(StoreError::Internal)?;
        }
        for (fact, vector) in vectors {
            state
                .push_vector(fact, vector)
                .map_err(StoreError::Internal)?;
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

/// Embeds fact texts and prepares the vectors for the index.
fn embed(
    semantic: &Semantic,
    texts: &[(FactId, String)],
) -> Result<Vec<(FactId, Quantized)>, EmbedError> {
    let strings: Vec<String> = texts.iter().map(|(_, t)| t.clone()).collect();
    let vectors = semantic.embedder.embed_documents(&strings)?;
    if vectors.len() != texts.len() {
        return Err(EmbedError(format!(
            "{} vectors for {} texts",
            vectors.len(),
            texts.len()
        )));
    }
    texts
        .iter()
        .zip(vectors)
        .map(|((fact, _), v)| Ok((*fact, Quantized::new(&v, semantic.dims)?)))
        .collect()
}
