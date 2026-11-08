//! Shared test fixture.
#![allow(dead_code)]

use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

use tessar_core::clock::ManualClock;
use tessar_core::log::{Commit, Log, LogError, MemLog, Seq};
use tessar_core::model::{
    Cardinality, EntityId, EntityKindId, Fact, FactBody, Key, Name, Origin, PredicateId, SpaceId,
    SpaceKind, Timestamp, Value, ValueType,
};
use tessar_core::store::{
    BodyDraft, EntityRef, FactDraft, Memory, ObjectDraft, Op, PredicateDraft, Recorded, Remember,
    Remembered, Semantic, StoreError,
};

/// A log shared between several `Memory` instances, to test reopening.
#[derive(Clone, Default)]
pub struct SharedLog(Arc<Mutex<MemLog<Op>>>);

impl SharedLog {
    pub fn commits(&self) -> usize {
        self.0.lock().unwrap().commits().len()
    }

    /// Every commit appended so far.
    pub fn snapshot(&self) -> Vec<Commit<Op>> {
        self.0.lock().unwrap().commits().to_vec()
    }

    /// A log that already holds `commits`.
    pub fn from_commits(commits: Vec<Commit<Op>>) -> Self {
        let log = Self::default();
        for commit in &commits {
            log.0.lock().unwrap().append(commit).unwrap();
        }
        log
    }
}

impl Log<Op> for SharedLog {
    fn replay(&mut self) -> Result<Vec<Commit<Op>>, LogError> {
        self.0.lock().unwrap().replay()
    }

    fn append(&mut self, commit: &Commit<Op>) -> Result<(), LogError> {
        self.0.lock().unwrap().append(commit)
    }

    fn next_seq(&self) -> Seq {
        self.0.lock().unwrap().next_seq()
    }
}

pub fn key(s: &str) -> Key {
    Key::new(s).unwrap()
}

pub fn name(s: &str) -> Name {
    Name::new(s).unwrap()
}

pub fn ts(ms: i64) -> Timestamp {
    Timestamp::from_millis(ms)
}

pub fn v(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap()
}

/// A predicate draft with an empty description.
pub fn predicate(n: &str, object: ValueType, cardinality: Cardinality) -> PredicateDraft {
    PredicateDraft {
        name: name(n),
        description: String::new(),
        object,
        cardinality,
    }
}

pub struct World {
    pub memory: Memory,
    pub semantic: Option<Semantic>,
    pub log: SharedLog,
    pub clock: Arc<ManualClock>,
    pub space: SpaceId,
    pub person: EntityKindId,
    pub project: EntityKindId,
    pub place: EntityKindId,
    /// The user's own space; the entity `user` lives here.
    pub home: SpaceId,
    /// The `person` kind of the user's space.
    pub home_person: EntityKindId,
    /// person → place, single.
    pub lives_in: PredicateId,
    /// project → text, single.
    pub goal: PredicateId,
    /// person → text, many.
    pub knows: PredicateId,
    /// project → int, single.
    pub counter: PredicateId,
}

impl World {
    pub fn new() -> Self {
        Self::with(None)
    }

    /// A world whose memory has the meaning channel, if given.
    pub fn with(semantic: Option<Semantic>) -> Self {
        let log = SharedLog::default();
        let clock = Arc::new(ManualClock::new(ts(100)));
        let memory = open(log.clone(), clock.clone(), semantic.clone()).unwrap();
        let space = memory
            .create_space(key("tessar"), SpaceKind::LongTerm, None)
            .unwrap();
        let person = memory.define_kind(space, name("person")).unwrap();
        let project = memory.define_kind(space, name("project")).unwrap();
        let place = memory.define_kind(space, name("place")).unwrap();
        let home = memory
            .create_space(key("home"), SpaceKind::LongTerm, None)
            .unwrap();
        let home_person = memory.define_kind(home, name("person")).unwrap();
        let lives_in = memory
            .define_predicate(
                space,
                predicate(
                    "lives_in",
                    ValueType::Entity(Some(place)),
                    Cardinality::Single,
                ),
            )
            .unwrap();
        let goal = memory
            .define_predicate(
                space,
                predicate("goal", ValueType::Text, Cardinality::Single),
            )
            .unwrap();
        let knows = memory
            .define_predicate(
                space,
                predicate("knows", ValueType::Text, Cardinality::Many),
            )
            .unwrap();
        let counter = memory
            .define_predicate(
                space,
                predicate("counter", ValueType::Int, Cardinality::Single),
            )
            .unwrap();
        Self {
            memory,
            semantic,
            log,
            clock,
            space,
            person,
            project,
            place,
            home,
            home_person,
            lives_in,
            goal,
            knows,
            counter,
        }
    }

    pub fn reopen(&self) -> Memory {
        open(self.log.clone(), self.clock.clone(), self.semantic.clone()).unwrap()
    }

    pub fn person(&self, k: &str) -> EntityRef {
        EntityRef::Key {
            kind: self.person,
            key: key(k),
        }
    }

    pub fn project(&self, k: &str) -> EntityRef {
        EntityRef::Key {
            kind: self.project,
            key: key(k),
        }
    }

    pub fn place(&self, k: &str) -> EntityRef {
        EntityRef::Key {
            kind: self.place,
            key: key(k),
        }
    }

    /// The entity `user` of the user's space, by key.
    pub fn user(&self) -> EntityRef {
        EntityRef::Key {
            kind: self.home_person,
            key: key("user"),
        }
    }

    /// A new project space.
    pub fn project_space(&self, k: &str) -> SpaceId {
        self.memory
            .create_space(key(k), SpaceKind::LongTerm, None)
            .unwrap()
    }

    /// Remembers facts into `space`.
    pub fn remember_in(&self, space: SpaceId, facts: Vec<FactDraft>) -> Vec<Recorded> {
        self.memory
            .remember(Remember {
                space,
                origin: origin(),
                episode: None,
                facts,
            })
            .unwrap()
            .facts
    }

    pub fn goal(&self, text: &str) -> FactDraft {
        FactDraft::new(BodyDraft::Triple {
            subject: self.project("tessar"),
            predicate: self.goal,
            object: ObjectDraft::Value(Value::Text(text.into())),
            text: None,
            when: None,
        })
    }

    pub fn lives_in(&self, who: &str, city: &str) -> FactDraft {
        FactDraft::new(BodyDraft::Triple {
            subject: self.person(who),
            predicate: self.lives_in,
            object: ObjectDraft::Entity(self.place(city)),
            text: None,
            when: None,
        })
    }

    pub fn knows(&self, who: &str, what: &str) -> FactDraft {
        FactDraft::new(BodyDraft::Triple {
            subject: self.person(who),
            predicate: self.knows,
            object: ObjectDraft::Value(Value::Text(what.into())),
            text: None,
            when: None,
        })
    }

    pub fn try_remember(&self, facts: Vec<FactDraft>) -> Result<Remembered, StoreError> {
        self.memory.remember(Remember {
            space: self.space,
            origin: origin(),
            episode: None,
            facts,
        })
    }

    pub fn remember(&self, facts: Vec<FactDraft>) -> Vec<Recorded> {
        self.try_remember(facts).unwrap().facts
    }

    pub fn remember_one(&self, fact: FactDraft) -> Recorded {
        self.remember(vec![fact])[0]
    }

    pub fn entity(&self, kind: EntityKindId, k: &str) -> EntityId {
        self.memory.find_entity(self.space, kind, k).unwrap().id
    }

    pub fn fact(&self, recorded: Recorded) -> Fact {
        self.memory.fact(recorded.fact).unwrap()
    }
}

fn open(
    log: SharedLog,
    clock: Arc<ManualClock>,
    semantic: Option<Semantic>,
) -> Result<Memory, StoreError> {
    match semantic {
        Some(semantic) => Memory::open_with(log, clock, semantic),
        None => Memory::open(log, clock),
    }
}

pub fn origin() -> Origin {
    Origin::Asserted { by: key("test") }
}

pub fn text_of(fact: &Fact) -> &str {
    match &fact.body {
        FactBody::Triple {
            object: Value::Text(t),
            ..
        } => t,
        other => panic!("not a text triple: {other:?}"),
    }
}
