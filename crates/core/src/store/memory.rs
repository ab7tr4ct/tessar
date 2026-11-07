//! [`Memory`]: the shared handle to the store.

use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard};

use super::draft::{Recorded, Remember, Remembered, Update, View};
use super::error::StoreError;
use super::op::Op;
use super::plan::Planner;
use super::state::State;
use crate::clock::Clock;
use crate::log::{Commit, Log};
use crate::model::{
    Cardinality, Entity, EntityId, EntityKindId, Fact, FactId, Key, Lineage, Name, PredicateDef,
    PredicateId, Space, SpaceId, SpaceKind, Timestamp, ValueType,
};

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
}

struct Writer {
    log: Box<dyn Log<Op> + Send>,
    /// Time of the last commit; commit times never go back, even if the
    /// system clock does.
    last_at: Timestamp,
}

impl Memory {
    /// Opens the memory: replays `log` to rebuild the state.
    pub fn open(
        mut log: impl Log<Op> + Send + 'static,
        clock: Arc<dyn Clock>,
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
        Ok(Self {
            inner: Arc::new(Inner {
                writer: Mutex::new(Writer {
                    log: Box::new(log),
                    last_at,
                }),
                state: RwLock::new(state),
                clock,
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

    /// Finds or defines a predicate in a space schema. Fails if it exists
    /// with another type or cardinality.
    pub fn define_predicate(
        &self,
        space: SpaceId,
        name: Name,
        object: ValueType,
        cardinality: Cardinality,
    ) -> Result<PredicateId, StoreError> {
        self.write(|p| p.define_predicate(space, name, object, cardinality))
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

    /// A predicate by name.
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

        let (ops, result) = {
            let state = self.state();
            let mut planner = Planner::new(&state, at);
            let result = plan(&mut planner)?;
            (planner.into_ops(), result)
        };
        if ops.is_empty() {
            // Nothing to change, for example a value that is already current.
            return Ok(result);
        }

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
