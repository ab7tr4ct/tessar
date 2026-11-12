//! The core invariant, over random sequences of API calls: the live state
//! equals the state rebuilt by replaying the log.
//!
//! The state is private, so the two memories are compared through every
//! read the API has: facts, chains, entities, pair values, views, the
//! vocabulary and recall (which goes through the keyword, entity and
//! vector indexes).

mod common;

use std::collections::BTreeSet;
use std::sync::Arc;

use common::*;
use proptest::prelude::*;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::model::{
    Cardinality, Confidence, Ending, Entity, EntityId, Fact, FactId, Lineage, Origin, PredicateDef,
    PredicateId, Timestamp, Value, ValueType,
};
use tessar_core::store::{
    BodyDraft, FactDraft, Memory, ObjectDraft, Outcome, Reason, Recall, Recalled, Remember,
    Replacement, Semantic, StoreError, Update, View,
};

const DIMS: usize = 8;
const PEOPLE: [&str; 3] = ["anna", "boris", "vera"];
const PLACES: [&str; 3] = ["moscow", "berlin", "oslo"];
/// The last two differ from the first two only by spaces around them.
const VALUES: [&str; 5] = ["red", "green", "blue", " red", "green "];

/// Words land on dimensions by their letters; no vector is zero.
struct Hashed;

impl Hashed {
    fn vector(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; DIMS];
        v[DIMS - 1] = 0.1;
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            if !word.is_empty() {
                v[word.bytes().map(usize::from).sum::<usize>() % (DIMS - 1)] += 1.0;
            }
        }
        v
    }
}

impl Embedder for Hashed {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| Self::vector(t)).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(Self::vector(text))
    }
}

/// One fact of a write. Indexes point into the constants above and are
/// taken modulo their length.
#[derive(Debug, Clone)]
enum Draft {
    /// A value of one of the single-valued text predicates.
    Single {
        who: usize,
        predicate: usize,
        value: usize,
        /// A person to mention, perhaps a new one.
        mention: Option<usize>,
        /// How far before now the value starts.
        back: Option<i64>,
    },
    /// A value of the many-valued predicate.
    Many { who: usize, value: usize },
    /// A link to an entity through a single-valued predicate.
    LivesIn { who: usize, place: usize },
    Statement {
        value: usize,
        mention: Option<usize>,
    },
}

#[derive(Debug, Clone)]
enum Action {
    Remember {
        inferred: bool,
        facts: Vec<Draft>,
    },
    /// Changes, corrects or closes the lineage of an earlier fact.
    Update {
        target: usize,
        wrong: bool,
        value: Option<usize>,
        inferred: bool,
        back: Option<i64>,
    },
    Merge {
        from: usize,
        into: usize,
    },
    Redact {
        target: usize,
    },
    Tick(i64),
}

fn draft() -> impl Strategy<Value = Draft> {
    let mention = || prop::option::weighted(0.3, 0..5usize);
    let back = prop::option::weighted(0.3, 0..150i64);
    prop_oneof![
        // A hot pair with few values: the same value is written again, by
        // both origins, often enough to reach the contradiction rules,
        // and sometimes backdated, to reach the lower bound.
        8 => (0..2usize, mention(), back.clone()).prop_map(|(value, mention, back)| {
            Draft::Single {
                who: 0,
                predicate: 0,
                value,
                mention,
                back,
            }
        }),
        4 => (0..3usize, 0..3usize, 0..5usize, mention(), back).prop_map(
            |(who, predicate, value, mention, back)| Draft::Single {
                who,
                predicate,
                value,
                mention,
                back,
            }
        ),
        1 => (0..3usize, 0..5usize).prop_map(|(who, value)| Draft::Many { who, value }),
        1 => (0..3usize, 0..3usize).prop_map(|(who, place)| Draft::LivesIn { who, place }),
        1 => (0..5usize, mention()).prop_map(|(value, mention)| Draft::Statement { value, mention }),
    ]
}

fn action() -> impl Strategy<Value = Action> {
    let back = prop::option::weighted(0.3, 0..150i64);
    prop_oneof![
        10 => (any::<bool>(), prop::collection::vec(draft(), 1..3))
            .prop_map(|(inferred, facts)| Action::Remember { inferred, facts }),
        3 => (
            any::<prop::sample::Index>(),
            any::<bool>(),
            prop::option::weighted(0.7, 0..5usize),
            any::<bool>(),
            back
        )
            .prop_map(|(target, wrong, value, inferred, back)| Action::Update {
                target: target.index(usize::MAX),
                wrong,
                value,
                inferred,
                back,
            }),
        1 => (0..3usize, 0..3usize).prop_map(|(from, into)| Action::Merge { from, into }),
        1 => any::<prop::sample::Index>()
            .prop_map(|target| Action::Redact { target: target.index(usize::MAX) }),
        2 => (1..120i64).prop_map(Action::Tick),
    ]
}

struct Run {
    w: World,
    /// Three single-valued text predicates, to merge with each other.
    singles: [PredicateId; 3],
    /// Every fact version a call returned.
    facts: Vec<FactId>,
    now: i64,
}

impl Run {
    fn new() -> Self {
        let semantic = Semantic {
            embedder: Arc::new(Hashed),
            dims: DIMS,
            near_bonus: 0.05,
            predicate_threshold: 1.0,
        };
        let w = World::with(Some(semantic));
        let singles = ["aim", "target", "purpose"].map(|n| {
            w.memory
                .define_predicate_unchecked(
                    w.space,
                    predicate(n, ValueType::Text, Cardinality::Single),
                )
                .unwrap()
        });
        Self {
            w,
            singles,
            facts: Vec::new(),
            now: 100,
        }
    }

    fn origin(inferred: bool) -> Origin {
        if inferred {
            Origin::Inferred {
                by: key("model"),
                confidence: Confidence::new(0.6).unwrap(),
            }
        } else {
            origin()
        }
    }

    fn person(&self, i: usize) -> tessar_core::store::EntityRef {
        // Index 3 and 4 are people nobody is a subject of.
        self.w
            .person(["anna", "boris", "vera", "ghost", "shade"][i % 5])
    }

    fn text_triple(&self, who: usize, predicate: PredicateId, value: usize) -> BodyDraft {
        BodyDraft::Triple {
            subject: self.person(who % 3),
            predicate,
            object: ObjectDraft::Value(Value::Text(VALUES[value % 5].into())),
            text: None,
            when: None,
        }
    }

    fn draft(&self, draft: &Draft) -> FactDraft {
        let mentions = |m: &Option<usize>| m.iter().map(|&i| self.person(i)).collect();
        match draft {
            Draft::Single {
                who,
                predicate,
                value,
                mention,
                back,
            } => FactDraft {
                mentions: mentions(mention),
                valid_from: back.map(|b| ts(self.now - b)),
                ..FactDraft::new(self.text_triple(*who, self.singles[*predicate % 3], *value))
            },
            Draft::Many { who, value } => {
                FactDraft::new(self.text_triple(*who, self.w.knows, *value))
            }
            Draft::LivesIn { who, place } => self.w.lives_in(PEOPLE[*who % 3], PLACES[*place % 3]),
            Draft::Statement { value, mention } => FactDraft {
                mentions: mentions(mention),
                ..FactDraft::new(BodyDraft::Statement {
                    text: format!("the wall is {}", VALUES[*value % 5]),
                    when: None,
                })
            },
        }
    }

    /// Runs one action. A refused call is fine, but it must write nothing.
    fn apply(&mut self, action: &Action) -> Result<(), TestCaseError> {
        let memory = &self.w.memory;
        let commits = self.w.log.commits();
        let wrote = match action {
            Action::Tick(ms) => {
                self.now += ms;
                self.w.clock.set(ts(self.now));
                return Ok(());
            }
            Action::Remember { inferred, facts } => {
                let request = Remember {
                    space: self.w.space,
                    origin: Self::origin(*inferred),
                    episode: None,
                    facts: facts.iter().map(|d| self.draft(d)).collect(),
                };
                match memory.remember(request) {
                    Ok(done) => {
                        let unchanged =
                            |r: &tessar_core::store::Recorded| r.outcome == Outcome::Unchanged;
                        if done.facts.iter().all(unchanged) {
                            prop_assert_eq!(
                                self.w.log.commits(),
                                commits,
                                "an unchanged write reached the log"
                            );
                        }
                        self.facts.extend(done.facts.iter().map(|r| r.fact));
                        self.repeat(*inferred, facts)?;
                        if !*inferred {
                            self.one_value_each(facts)?;
                        }
                        true
                    }
                    Err(e) => {
                        // Without an explicit date a write is never
                        // refused for starting before an earlier value
                        // ended: it starts now, and the clock only moves
                        // forward (no close is dated in the future).
                        let dated = facts
                            .iter()
                            .any(|d| matches!(d, Draft::Single { back: Some(_), .. }));
                        prop_assert!(
                            dated || !matches!(e, StoreError::ChangeBeforeValid { .. }),
                            "an undated write was refused: {e:?}"
                        );
                        false
                    }
                }
            }
            Action::Update {
                target,
                wrong,
                value,
                inferred,
                back,
            } => {
                let Some(fact) = self.pick(*target) else {
                    return Ok(());
                };
                let head = memory.history(fact.lineage).pop().unwrap();
                let replacement = value.and_then(|v| self.replacement(&head, v));
                let reason = if *wrong {
                    Reason::Correction
                } else {
                    Reason::Change {
                        at: back.map(|b| ts(self.now - b)),
                    }
                };
                let update = Update {
                    lineage: fact.lineage,
                    expected_version: head.version,
                    reason,
                    replacement,
                    origin: Self::origin(*inferred),
                };
                match memory.update(update) {
                    Ok(done) => {
                        self.facts.push(done.fact);
                        true
                    }
                    Err(_) => false,
                }
            }
            Action::Merge { from, into } => memory
                .merge_predicates(self.singles[*from], self.singles[*into])
                .is_ok(),
            Action::Redact { target } => match self.pick(*target) {
                Some(fact) => memory.redact(fact.id).is_ok(),
                None => return Ok(()),
            },
        };
        if !wrote {
            prop_assert_eq!(self.w.log.commits(), commits, "a refused call wrote");
        }
        prop_assert!(!memory.is_poisoned());
        if wrote {
            self.no_overlap()?;
        }
        Ok(())
    }

    /// Within one lineage, no two versions were true at the same time.
    /// Corrected and redacted versions were never (or are not shown as)
    /// true, and a version closed at or before its start was true at no
    /// time, so none of them counts.
    fn no_overlap(&self) -> Result<(), TestCaseError> {
        let lineages: BTreeSet<Lineage> = self
            .facts
            .iter()
            .filter_map(|&id| self.w.memory.fact(id))
            .map(|f| f.lineage)
            .collect();
        for lineage in lineages {
            // (start, end) with `None` for open.
            let mut spans: Vec<(Timestamp, Option<Timestamp>, FactId)> = Vec::new();
            for f in self.w.memory.history(lineage) {
                if f.is_redacted() {
                    continue;
                }
                let to = match f.end.map(|e| e.ending) {
                    None => None,
                    Some(Ending::Changed { valid_to }) => Some(valid_to),
                    Some(Ending::Corrected) => continue,
                };
                if to.is_some_and(|t| t <= f.valid_from) {
                    continue;
                }
                spans.push((f.valid_from, to, f.id));
            }
            for (i, a) in spans.iter().enumerate() {
                for b in &spans[i + 1..] {
                    let before =
                        |x: &(Timestamp, Option<Timestamp>, FactId),
                         y: &(Timestamp, Option<Timestamp>, FactId)| {
                            x.1.is_some_and(|end| end <= y.0)
                        };
                    prop_assert!(
                        before(a, b) || before(b, a),
                        "versions {} and {} of {} were true at the same time",
                        a.2,
                        b.2,
                        lineage
                    );
                }
            }
        }
        Ok(())
    }

    /// After an asserted write, each single-valued pair it touched has
    /// exactly one current value.
    fn one_value_each(&self, facts: &[Draft]) -> Result<(), TestCaseError> {
        for d in facts {
            let (key, predicate) = match d {
                Draft::Single { who, predicate, .. } => {
                    (PEOPLE[*who % 3], self.singles[*predicate % 3])
                }
                Draft::LivesIn { who, .. } => (PEOPLE[*who % 3], self.w.lives_in),
                Draft::Many { .. } | Draft::Statement { .. } => continue,
            };
            let subject = self.w.entity(self.w.person, key);
            let values = self.w.memory.values_of(subject, predicate);
            prop_assert_eq!(
                values.len(),
                1,
                "{} current values of {} after an asserted write",
                values.len(),
                key
            );
        }
        Ok(())
    }

    /// A write that went through is idempotent: the same request again is
    /// unchanged in every fact and appends nothing.
    ///
    /// Statements are left out of the repeated request: they are never
    /// compared with what is already there, by design, so each one is a
    /// new fact every time.
    fn repeat(&self, inferred: bool, facts: &[Draft]) -> Result<(), TestCaseError> {
        let again: Vec<FactDraft> = facts
            .iter()
            .filter(|d| !matches!(d, Draft::Statement { .. }))
            .map(|d| self.draft(d))
            .collect();
        if again.is_empty() {
            return Ok(());
        }
        let commits = self.w.log.commits();
        let done = self.w.memory.remember(Remember {
            space: self.w.space,
            origin: Self::origin(inferred),
            episode: None,
            facts: again,
        });
        let outcomes: Vec<Outcome> = match done {
            Ok(done) => done.facts.iter().map(|r| r.outcome).collect(),
            Err(e) => return Err(TestCaseError::fail(format!("the repeat failed: {e}"))),
        };
        prop_assert!(
            outcomes.iter().all(|o| *o == Outcome::Unchanged),
            "the repeat was not unchanged: {:?}",
            outcomes
        );
        prop_assert_eq!(self.w.log.commits(), commits, "the repeat wrote");
        Ok(())
    }

    fn pick(&self, target: usize) -> Option<Fact> {
        let id = *self.facts.get(target.checked_rem(self.facts.len())?)?;
        self.w.memory.fact(id)
    }

    /// A new value for the lineage of `head`, keeping its shape.
    fn replacement(&self, head: &Fact, value: usize) -> Option<Replacement> {
        use tessar_core::model::FactBody;
        let body = match &head.body {
            FactBody::Statement { .. } => BodyDraft::Statement {
                text: format!("the wall is {}", VALUES[value % 5]),
                when: None,
            },
            FactBody::Triple {
                subject,
                predicate,
                object,
                ..
            } => BodyDraft::Triple {
                subject: tessar_core::store::EntityRef::Id(*subject),
                predicate: *predicate,
                object: match object {
                    Value::Entity(_) => ObjectDraft::Entity(self.w.place(PLACES[value % 3])),
                    _ => ObjectDraft::Value(Value::Text(VALUES[value % 5].into())),
                },
                text: None,
                when: None,
            },
            FactBody::Event { .. } | FactBody::Redacted => return None,
        };
        Some(Replacement {
            body,
            mentions: vec![],
        })
    }

    /// Everything the API shows of a memory.
    fn snapshot(&self, memory: &Memory) -> Snapshot {
        let w = &self.w;
        let lineages: BTreeSet<Lineage> = self
            .facts
            .iter()
            .filter_map(|&id| memory.fact(id))
            .map(|f| f.lineage)
            .collect();
        let entities: Vec<Entity> = ["anna", "boris", "vera", "ghost", "shade"]
            .iter()
            .filter_map(|k| memory.find_entity(w.space, w.person, k))
            .chain(
                PLACES
                    .iter()
                    .filter_map(|k| memory.find_entity(w.space, w.place, k)),
            )
            .collect();
        let predicates: Vec<PredicateId> = self
            .singles
            .iter()
            .copied()
            .chain([w.knows, w.lives_in, w.goal])
            .collect();
        let views = [0, 60, 150, 300, 600, 5_000]
            .into_iter()
            .flat_map(|t| [View::TrueAt(ts(t)), View::KnownAt(ts(t))])
            .chain([View::Current]);
        let mut about = Vec::new();
        for view in views {
            for e in &entities {
                about.push(memory.facts_about(e.id, view));
            }
        }
        let mut values = Vec::new();
        for e in &entities {
            for &p in &predicates {
                values.push((e.id, p, memory.values_of(e.id, p)));
            }
        }
        let recalled = ["anna red wall", "green", "boris lives in berlin", "ghost"]
            .iter()
            .flat_map(|q| {
                let current = Recall::new(*q);
                let past = Recall {
                    view: View::TrueAt(ts(150)),
                    ..Recall::new(*q)
                };
                [memory.recall(&current), memory.recall(&past)]
            })
            .collect();
        Snapshot {
            facts: self.facts.iter().map(|&id| memory.fact(id)).collect(),
            histories: lineages.iter().map(|&l| memory.history(l)).collect(),
            current: lineages.iter().map(|&l| memory.current(l)).collect(),
            about,
            values,
            vocabulary: memory.predicates(),
            canonical: predicates
                .iter()
                .map(|&p| memory.canonical_predicate(p))
                .collect(),
            entities,
            recalled,
        }
    }
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    facts: Vec<Option<Fact>>,
    histories: Vec<Vec<Fact>>,
    current: Vec<Option<Fact>>,
    entities: Vec<Entity>,
    about: Vec<Vec<Fact>>,
    values: Vec<(EntityId, PredicateId, Vec<Fact>)>,
    vocabulary: Vec<PredicateDef>,
    canonical: Vec<Option<PredicateId>>,
    recalled: Vec<Recalled>,
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 300,
        // No regression file next to the tests: a failure prints the
        // shrunk list of actions, which is enough to reproduce it.
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn the_live_state_equals_the_replayed_log(
        actions in prop::collection::vec(action(), 1..28)
    ) {
        let mut run = Run::new();
        for action in &actions {
            run.apply(action)?;
        }
        let live = run.snapshot(&run.w.memory);
        let replayed = run.snapshot(&run.w.reopen());
        prop_assert_eq!(live, replayed);
    }
}
