mod common;

use std::num::NonZeroU32;
use std::sync::Arc;
use std::thread;

use common::*;
use tessar_core::model::{Cardinality, Ending, EntityId, FactBody, SpaceKind, Value, ValueType};
use tessar_core::store::{
    BodyDraft, EntityRef, FactDraft, ObjectDraft, Outcome, Reason, Remember, Replacement,
    StoreError, Update, View,
};

#[test]
fn entities_are_found_or_created_by_key() {
    let w = World::new();
    let r = w.remember(vec![w.knows("Anna", "Rust"), w.knows("  anna ", "Go")]);
    let anna = w.entity(w.person, "ANNA");
    for recorded in r {
        assert!(w.fact(recorded).entities().contains(&anna));
    }
    w.remember_one(w.knows("ANNA", "SQL"));
    assert_eq!(w.memory.facts_about(anna, View::Current).len(), 3);

    w.memory.add_alias(anna, key("my sister")).unwrap();
    let r = w.remember_one(w.knows("My  Sister", "Python"));
    assert!(w.fact(r).entities().contains(&anna));
    // The alias belongs to Anna now; nobody else can take it.
    let bob = {
        w.remember_one(w.knows("Bob", "C"));
        w.entity(w.person, "bob")
    };
    assert!(matches!(
        w.memory.add_alias(bob, key("MY SISTER")),
        Err(StoreError::NameTaken(_))
    ));
}

#[test]
fn single_valued_predicate_replaces_the_value_as_a_new_version() {
    let w = World::new();
    let first = w.remember_one(w.goal("graph database"));
    assert_eq!(first.outcome, Outcome::Created);

    w.clock.set(ts(200));
    let second = w.remember_one(w.goal("memory engine"));
    assert_eq!(
        second.outcome,
        Outcome::Superseded {
            previous: first.fact
        }
    );

    let old = w.fact(first);
    let new = w.fact(second);
    assert_eq!(old.lineage, new.lineage);
    assert_eq!(new.version.get(), 2);
    assert_eq!(new.prev, Some(old.id));
    assert_eq!(old.next(), Some(new.id));
    assert_eq!(
        old.end.unwrap().ending,
        Ending::Changed { valid_to: ts(200) }
    );

    let tessar = w.entity(w.project, "tessar");
    let current = w.memory.facts_about(tessar, View::Current);
    assert_eq!(current.len(), 1);
    assert_eq!(text_of(&current[0]), "memory engine");
    assert_eq!(
        text_of(&w.memory.facts_about(tessar, View::TrueAt(ts(150)))[0]),
        "graph database"
    );
    assert_eq!(
        text_of(&w.memory.facts_about(tessar, View::KnownAt(ts(150)))[0]),
        "graph database"
    );
    assert_eq!(
        w.memory
            .history(new.lineage)
            .iter()
            .map(text_of)
            .collect::<Vec<_>>(),
        ["graph database", "memory engine"]
    );
    assert_eq!(w.memory.current(new.lineage).unwrap().id, new.id);
}

#[test]
fn a_value_that_is_already_current_is_not_written_again() {
    let w = World::new();
    let first = w.remember_one(w.goal("memory engine"));
    let commits = w.log.commits();
    let again = w.remember_one(w.goal("memory engine"));
    assert_eq!(again.outcome, Outcome::Unchanged);
    assert_eq!(again.fact, first.fact);
    assert_eq!(w.log.commits(), commits, "nothing was written");
}

#[test]
fn many_valued_predicate_accumulates_values() {
    let w = World::new();
    w.remember(vec![w.knows("Anna", "Rust"), w.knows("Anna", "Go")]);
    let again = w.remember_one(w.knows("Anna", "Rust"));
    assert_eq!(again.outcome, Outcome::Unchanged);
    let anna = w.entity(w.person, "anna");
    assert_eq!(w.memory.values_of(anna, w.knows).len(), 2);
}

#[test]
fn a_stale_version_is_a_conflict_not_a_lost_update() {
    let w = World::new();
    let first = w.remember_one(w.goal("v1"));
    let lineage = w.fact(first).lineage;

    // Writers A and B both saw version 1. A writes first.
    let mut a = w.goal("from A");
    a.expected_version = Some(v(1));
    w.remember_one(a);

    let mut b = w.goal("from B");
    b.expected_version = Some(v(1));
    let commits = w.log.commits();
    match w.try_remember(vec![b]) {
        Err(StoreError::Conflict {
            lineage: l,
            expected,
            current,
        }) => {
            assert_eq!(l, Some(lineage));
            assert_eq!(expected, v(1));
            assert_eq!(current, Some(v(2)));
        }
        other => panic!("expected a conflict, got {other:?}"),
    }
    assert_eq!(w.log.commits(), commits);
    assert_eq!(text_of(&w.memory.current(lineage).unwrap()), "from A");
}

#[test]
fn expecting_a_value_that_does_not_exist_is_a_conflict() {
    let w = World::new();
    let mut draft = w.goal("anything");
    draft.expected_version = Some(v(1));
    assert!(matches!(
        w.try_remember(vec![draft]),
        Err(StoreError::Conflict {
            lineage: None,
            current: None,
            ..
        })
    ));
}

#[test]
fn a_correction_was_never_true_but_was_known() {
    let w = World::new();
    let mut moscow = w.lives_in("Anna", "Moscow");
    moscow.valid_from = Some(ts(50));
    let first = w.remember_one(moscow);
    let lineage = w.fact(first).lineage;

    w.clock.set(ts(200));
    let corrected = w
        .memory
        .update(Update {
            lineage,
            expected_version: v(1),
            reason: Reason::Correction,
            replacement: Some(Replacement {
                body: BodyDraft::Triple {
                    subject: w.person("Anna"),
                    predicate: w.lives_in,
                    object: ObjectDraft::Entity(w.place("Minsk")),
                    text: None,
                },
                mentions: vec![],
            }),
            origin: origin(),
        })
        .unwrap();

    let anna = w.entity(w.person, "anna");
    let minsk = w.entity(w.place, "minsk");
    let moscow = w.entity(w.place, "moscow");
    // The corrected value is true from the same moment the mistake claimed.
    assert_eq!(w.fact(corrected).valid_from, ts(50));
    let true_then = w.memory.facts_about(anna, View::TrueAt(ts(60)));
    assert_eq!(true_then.len(), 1);
    assert!(true_then[0].entities().contains(&minsk));
    // But at 150 memory still believed Moscow.
    let known_then = w.memory.facts_about(anna, View::KnownAt(ts(150)));
    assert_eq!(known_then.len(), 1);
    assert!(known_then[0].entities().contains(&moscow));
}

#[test]
fn a_backdated_change_splits_true_and_known() {
    let w = World::new();
    w.remember_one(w.lives_in("Anna", "Moscow"));
    // At 300 we learn that Anna moved at 200.
    w.clock.set(ts(300));
    let mut berlin = w.lives_in("Anna", "Berlin");
    berlin.valid_from = Some(ts(200));
    w.remember_one(berlin);

    let anna = w.entity(w.person, "anna");
    let berlin = w.entity(w.place, "berlin");
    let moscow = w.entity(w.place, "moscow");
    let at = |view| w.memory.facts_about(anna, view)[0].entities();
    assert!(at(View::TrueAt(ts(250))).contains(&berlin));
    assert!(at(View::KnownAt(ts(250))).contains(&moscow));
    assert!(at(View::TrueAt(ts(150))).contains(&moscow));
}

#[test]
fn a_change_before_the_old_value_became_true_is_rejected() {
    let w = World::new();
    w.clock.set(ts(500));
    w.remember_one(w.lives_in("Anna", "Moscow"));
    let mut berlin = w.lives_in("Anna", "Berlin");
    berlin.valid_from = Some(ts(400));
    assert!(matches!(
        w.try_remember(vec![berlin]),
        Err(StoreError::ChangeBeforeValid { .. })
    ));
}

#[test]
fn a_failed_transaction_writes_nothing() {
    let w = World::new();
    let commits = w.log.commits();
    let bad = FactDraft::new(BodyDraft::Triple {
        subject: w.person("Anna"),
        // A person is not a place.
        predicate: w.lives_in,
        object: ObjectDraft::Entity(w.person("Bob")),
        text: None,
    });
    let result = w.try_remember(vec![w.knows("Zed", "Rust"), bad]);
    assert!(matches!(result, Err(StoreError::TypeMismatch { .. })));
    assert_eq!(w.log.commits(), commits);
    for who in ["zed", "anna", "bob"] {
        assert!(
            w.memory.find_entity(w.space, w.person, who).is_none(),
            "{who}"
        );
    }
}

#[test]
fn two_values_of_a_single_predicate_in_one_transaction_are_rejected() {
    let w = World::new();
    assert!(matches!(
        w.try_remember(vec![w.goal("a"), w.goal("b")]),
        Err(StoreError::DuplicatePair { .. })
    ));
}

fn lives_in_user(w: &World, city: &str) -> FactDraft {
    FactDraft::new(BodyDraft::Triple {
        subject: w.user(),
        predicate: w.lives_in,
        object: ObjectDraft::Entity(w.place(city)),
        text: None,
    })
}

#[test]
fn a_fact_may_refer_to_an_entity_of_another_space() {
    let w = World::new();
    let project = w.project_space("cortex-db");
    // By key: the entity is created in the space of its kind.
    let by_key = w.remember_in(project, vec![lives_in_user(&w, "Moscow")])[0];
    let user = w.memory.find_entity(w.home, w.home_person, "user").unwrap();
    assert_eq!(user.space, w.home);
    assert!(
        w.memory
            .find_entity(project, w.home_person, "user")
            .is_none()
    );
    let fact = w.fact(by_key);
    assert_eq!(fact.space, project);
    assert!(fact.entities().contains(&user.id));

    // By id, as subject and as mention.
    let by_id = w.remember_in(
        project,
        vec![FactDraft {
            mentions: vec![EntityRef::Id(user.id)],
            ..FactDraft::new(BodyDraft::Triple {
                subject: EntityRef::Id(user.id),
                predicate: w.knows,
                object: ObjectDraft::Value(Value::Text("Rust".into())),
                text: None,
            })
        }],
    )[0];
    assert_eq!(w.fact(by_id).space, project);
    assert_eq!(w.fact(by_id).mentions, [user.id]);

    // A fact in another space about the same user.
    w.remember_one(FactDraft::new(BodyDraft::Triple {
        subject: w.user(),
        predicate: w.knows,
        object: ObjectDraft::Value(Value::Text("Go".into())),
        text: None,
    }));
    let facts = w.memory.facts_about(user.id, View::Current);
    let spaces: Vec<_> = facts.iter().map(|f| f.space).collect();
    assert_eq!(facts.len(), 3);
    assert!(spaces.contains(&project) && spaces.contains(&w.space));
}

#[test]
fn one_new_cross_space_key_is_one_entity_within_a_transaction() {
    let w = World::new();
    let project = w.project_space("cortex-db");
    w.remember_in(
        project,
        vec![
            w.knows("Zed", "Rust"),
            lives_in_user(&w, "Moscow"),
            FactDraft::new(BodyDraft::Triple {
                subject: w.user(),
                predicate: w.knows,
                object: ObjectDraft::Value(Value::Text("Go".into())),
                text: None,
            }),
        ],
    );
    let user = w.memory.find_entity(w.home, w.home_person, "user").unwrap();
    assert_eq!(w.memory.facts_about(user.id, View::Current).len(), 2);
}

#[test]
fn a_single_valued_pair_is_replaced_from_another_space() {
    let w = World::new();
    let a = w.project_space("project-a");
    let b = w.project_space("project-b");
    w.clock.set(ts(200));
    let first = w.remember_in(a, vec![lives_in_user(&w, "Moscow")])[0];
    w.clock.set(ts(300));
    let second = w.remember_in(b, vec![lives_in_user(&w, "Berlin")])[0];
    assert_eq!(
        second.outcome,
        Outcome::Superseded {
            previous: first.fact
        }
    );
    let (old, new) = (w.fact(first), w.fact(second));
    assert_eq!((old.space, new.space), (a, b));
    assert_eq!(old.lineage, new.lineage);
    assert_eq!(w.memory.history(new.lineage).len(), 2);
    assert_eq!(w.memory.current(new.lineage), Some(new));
}

#[test]
fn an_update_keeps_the_space_of_the_version() {
    let w = World::new();
    let a = w.project_space("project-a");
    let first = w.remember_in(a, vec![lives_in_user(&w, "Moscow")])[0];
    let lineage = w.fact(first).lineage;
    let replaced = w
        .memory
        .update(Update {
            lineage,
            expected_version: v(1),
            reason: Reason::Correction,
            replacement: Some(Replacement {
                body: lives_in_user(&w, "Berlin").body,
                mentions: Vec::new(),
            }),
            origin: origin(),
        })
        .unwrap();
    assert_eq!(w.fact(replaced).space, a);
}

fn goal_with_text(w: &World, goal: &str, text: &str) -> FactDraft {
    FactDraft::new(BodyDraft::Triple {
        subject: w.project("tessar"),
        predicate: w.goal,
        object: ObjectDraft::Value(Value::Text(goal.into())),
        text: Some(text.into()),
    })
}

fn stored_text(f: &tessar_core::model::Fact) -> Option<&str> {
    match &f.body {
        FactBody::Triple { text, .. } => text.as_deref(),
        other => panic!("not a triple: {other:?}"),
    }
}

#[test]
fn the_same_value_with_another_text_is_unchanged() {
    let w = World::new();
    let first = w.remember_one(goal_with_text(
        &w,
        "memory engine",
        "Tessar is a memory engine",
    ));
    let again = w.remember_one(goal_with_text(&w, "memory engine", "A store of facts"));
    assert_eq!(again.outcome, Outcome::Unchanged);
    assert_eq!(again.fact, first.fact);
    assert_eq!(
        stored_text(&w.fact(again)),
        Some("Tessar is a memory engine")
    );
    assert_eq!(w.memory.history(w.fact(first).lineage).len(), 1);

    // Without a text it is unchanged too.
    assert_eq!(
        w.remember_one(w.goal("memory engine")).outcome,
        Outcome::Unchanged
    );

    // A many-valued predicate: the same value with another text.
    let knows = |text: &str| {
        FactDraft::new(BodyDraft::Triple {
            subject: w.person("Anna"),
            predicate: w.knows,
            object: ObjectDraft::Value(Value::Text("Rust".into())),
            text: Some(text.into()),
        })
    };
    let one = w.remember_one(knows("Anna knows Rust"));
    let two = w.remember_one(knows("Anna writes Rust daily"));
    assert_eq!(two.outcome, Outcome::Unchanged);
    assert_eq!(two.fact, one.fact);
}

#[test]
fn another_value_carries_its_own_text() {
    let w = World::new();
    let first = w.remember_one(goal_with_text(
        &w,
        "graph database",
        "Tessar is a graph database",
    ));
    let second = w.remember_one(goal_with_text(
        &w,
        "memory engine",
        "Tessar is a memory engine",
    ));
    assert_eq!(
        second.outcome,
        Outcome::Superseded {
            previous: first.fact
        }
    );
    assert_eq!(
        stored_text(&w.fact(second)),
        Some("Tessar is a memory engine")
    );
    assert_eq!(
        stored_text(&w.fact(first)),
        Some("Tessar is a graph database")
    );
}

#[test]
fn an_update_with_the_same_value_and_another_text_is_unchanged() {
    let w = World::new();
    let first = w.remember_one(goal_with_text(
        &w,
        "memory engine",
        "Tessar is a memory engine",
    ));
    let result = w
        .memory
        .update(Update {
            lineage: w.fact(first).lineage,
            expected_version: v(1),
            reason: Reason::Correction,
            replacement: Some(Replacement {
                body: goal_with_text(&w, "memory engine", "A store of facts").body,
                mentions: Vec::new(),
            }),
            origin: origin(),
        })
        .unwrap();
    assert_eq!(result.outcome, Outcome::Unchanged);
    assert_eq!(result.fact, first.fact);
    assert_eq!(
        stored_text(&w.memory.fact(result.fact).unwrap()),
        Some("Tessar is a memory engine")
    );
}

#[test]
fn a_triple_text_is_found_by_a_word_only_it_has() {
    let w = World::new();
    w.remember_one(goal_with_text(
        &w,
        "memory engine",
        "Tessar remembers everything forever",
    ));
    let hits = w
        .memory
        .recall(&tessar_core::store::Recall::new("forever"))
        .hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(
        stored_text(&hits[0].fact),
        Some("Tessar remembers everything forever")
    );
}

#[test]
fn closing_without_replacement_and_reopening_later() {
    let w = World::new();
    let first = w.remember_one(w.goal("graph database"));
    let lineage = w.fact(first).lineage;
    let closed = w
        .memory
        .update(Update {
            lineage,
            expected_version: v(1),
            reason: Reason::Change { at: None },
            replacement: None,
            origin: origin(),
        })
        .unwrap();
    assert_eq!(closed.outcome, Outcome::Closed);
    assert!(w.memory.current(lineage).is_none());
    let tessar = w.entity(w.project, "tessar");
    assert!(w.memory.values_of(tessar, w.goal).is_empty());

    // There is nothing current to update any more.
    let err = w
        .memory
        .update(Update {
            lineage,
            expected_version: v(1),
            reason: Reason::Correction,
            replacement: None,
            origin: origin(),
        })
        .unwrap_err();
    assert!(
        matches!(err, StoreError::NotCurrent(l) if l == lineage),
        "{err}"
    );

    // A later value continues the same lineage.
    let again = w.remember_one(w.goal("memory engine"));
    assert_eq!(again.outcome, Outcome::Created);
    assert_eq!(w.fact(again).lineage, lineage);
    assert_eq!(w.memory.history(lineage).len(), 2);
}

#[test]
fn a_replacement_must_keep_its_shape() {
    let w = World::new();
    let first = w.remember_one(w.goal("x"));
    let err = w
        .memory
        .update(Update {
            lineage: w.fact(first).lineage,
            expected_version: v(1),
            reason: Reason::Change { at: None },
            replacement: Some(Replacement {
                body: BodyDraft::Statement {
                    text: "no longer a triple".into(),
                    when: None,
                },
                mentions: vec![],
            }),
            origin: origin(),
        })
        .unwrap_err();
    assert!(matches!(err, StoreError::ShapeMismatch(_)));
}

#[test]
fn schema_definitions_are_idempotent_and_conflicts_are_caught() {
    let w = World::new();
    let again = w
        .memory
        .define_predicate(
            w.space,
            predicate("goal", ValueType::Text, Cardinality::Single),
        )
        .unwrap();
    assert_eq!(again, w.goal);
    assert!(matches!(
        w.memory.define_predicate(
            w.space,
            predicate("goal", ValueType::Text, Cardinality::Many)
        ),
        Err(StoreError::PredicateConflict { .. })
    ));
    assert_eq!(
        w.memory.define_kind(w.space, name("person")).unwrap(),
        w.person
    );
    assert_eq!(
        w.memory
            .create_space(key("TESSAR"), SpaceKind::LongTerm, None)
            .unwrap(),
        w.space
    );
}

#[test]
fn commit_times_never_go_back() {
    let w = World::new();
    w.clock.set(ts(1000));
    let first = w.remember_one(w.knows("Anna", "Rust"));
    w.clock.set(ts(10)); // the system clock jumped back
    let second = w.remember_one(w.knows("Anna", "Go"));
    assert!(w.fact(second).recorded_at >= w.fact(first).recorded_at);
}

#[test]
fn replay_rebuilds_the_same_state() {
    let w = World::new();
    let goal1 = w.remember_one(w.goal("graph database"));
    w.clock.set(ts(200));
    let goal2 = w.remember_one(w.goal("memory engine"));
    w.remember(vec![w.knows("Anna", "Rust"), w.lives_in("Anna", "Moscow")]);
    let anna = w.entity(w.person, "anna");
    w.memory.add_alias(anna, key("my sister")).unwrap();
    let project = w.project_space("cortex-db");
    let cross = w.remember_in(project, vec![lives_in_user(&w, "Berlin")])[0];
    w.clock.set(ts(300));
    let cross2 = w.remember_in(project, vec![lives_in_user(&w, "Paris")])[0];

    let reopened = w.reopen();
    let lineage = w.fact(goal2).lineage;
    assert_eq!(reopened.history(lineage), w.memory.history(lineage));
    assert_eq!(reopened.fact(goal1.fact), w.memory.fact(goal1.fact));
    assert_eq!(reopened.fact(cross2.fact), w.memory.fact(cross2.fact));
    assert_eq!(
        reopened.history(w.fact(cross).lineage),
        w.memory.history(w.fact(cross).lineage)
    );
    let user = w
        .memory
        .find_entity(w.home, w.home_person, "user")
        .unwrap()
        .id;
    assert_eq!(
        reopened.facts_about(user, View::Current),
        w.memory.facts_about(user, View::Current)
    );
    for view in [View::Current, View::TrueAt(ts(150)), View::KnownAt(ts(150))] {
        assert_eq!(
            reopened.facts_about(anna, view),
            w.memory.facts_about(anna, view)
        );
    }
    assert_eq!(
        reopened
            .find_entity(w.space, w.person, "MY SISTER")
            .map(|e| e.id),
        Some(anna)
    );
    assert_eq!(
        reopened.predicate_named(w.space, "goal").unwrap().id,
        w.goal
    );

    // The reopened memory keeps minting the same ids the old one would.
    drop(reopened);
    let reopened = w.reopen();
    let next = reopened
        .remember(Remember {
            space: w.space,
            origin: origin(),
            episode: None,
            facts: vec![w.knows("Anna", "Zig")],
        })
        .unwrap();
    assert_eq!(
        reopened.fact(next.facts[0].fact).unwrap().id,
        next.facts[0].fact
    );
}

/// Many writers increment one counter, retrying on conflict. Optimistic
/// version checks must not lose a single increment.
#[test]
fn concurrent_increments_lose_nothing() {
    const THREADS: usize = 8;
    const INCREMENTS: usize = 25;

    let w = Arc::new(World::new());
    let counter = |n: i64, expected: Option<NonZeroU32>| FactDraft {
        expected_version: expected,
        ..FactDraft::new(BodyDraft::Triple {
            subject: w.project("tessar"),
            predicate: w.counter,
            object: ObjectDraft::Value(Value::Int(n)),
            text: None,
        })
    };
    let first = w.remember_one(counter(0, None));
    let lineage = w.fact(first).lineage;

    let workers: Vec<_> = (0..THREADS)
        .map(|_| {
            let w = Arc::clone(&w);
            thread::spawn(move || {
                let mut conflicts = 0;
                for _ in 0..INCREMENTS {
                    loop {
                        let current = w.memory.current(lineage).unwrap();
                        let FactBody::Triple {
                            object: Value::Int(n),
                            ..
                        } = current.body
                        else {
                            panic!("not a counter");
                        };
                        let draft = FactDraft {
                            expected_version: Some(current.version),
                            ..FactDraft::new(BodyDraft::Triple {
                                subject: w.project("tessar"),
                                predicate: w.counter,
                                object: ObjectDraft::Value(Value::Int(n + 1)),
                                text: None,
                            })
                        };
                        match w.try_remember(vec![draft]) {
                            Ok(_) => break,
                            Err(StoreError::Conflict { .. }) => conflicts += 1,
                            Err(other) => panic!("{other}"),
                        }
                    }
                }
                conflicts
            })
        })
        .collect();
    let conflicts: usize = workers.into_iter().map(|t| t.join().unwrap()).sum();

    let total = (THREADS * INCREMENTS) as i64;
    let current = w.memory.current(lineage).unwrap();
    assert!(
        matches!(current.body, FactBody::Triple { object: Value::Int(n), .. } if n == total),
        "lost updates: {:?} ({conflicts} conflicts retried)",
        current.body
    );
    assert_eq!(current.version.get() as i64, total + 1);
    assert_eq!(w.memory.history(lineage).len() as i64, total + 1);
}

/// Many writers name the same new entity at once: exactly one is created.
#[test]
fn concurrent_writers_create_one_entity() {
    let w = Arc::new(World::new());
    let workers: Vec<_> = (0..16)
        .map(|i| {
            let w = Arc::clone(&w);
            thread::spawn(move || {
                let r = w.remember_one(w.knows(" Shared ", &format!("skill {i}")));
                match w.fact(r).body {
                    FactBody::Triple { subject, .. } => subject,
                    _ => unreachable!(),
                }
            })
        })
        .collect();
    let ids: Vec<EntityId> = workers.into_iter().map(|t| t.join().unwrap()).collect();
    assert!(ids.windows(2).all(|p| p[0] == p[1]), "{ids:?}");
    assert_eq!(w.memory.facts_about(ids[0], View::Current).len(), 16);
}
