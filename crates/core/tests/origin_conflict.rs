//! An inferred value never replaces an asserted one.

mod common;

use common::*;
use tessar_core::model::{Cardinality, Confidence, Ending, Origin, PredicateId, ValueType};
use tessar_core::store::{
    BodyDraft, FactDraft, ObjectDraft, Outcome, Reason, Recorded, Remember, Replacement,
    StoreError, Update, View,
};

fn inferred() -> Origin {
    Origin::Inferred {
        by: key("model"),
        confidence: Confidence::new(0.6).unwrap(),
    }
}

fn write_draft(w: &World, origin: Origin, draft: FactDraft) -> Result<Recorded, StoreError> {
    w.memory
        .remember(Remember {
            space: w.space,
            origin,
            episode: None,
            facts: vec![draft],
        })
        .map(|r| r.facts[0])
}

fn write(w: &World, origin: Origin, city: &str) -> Recorded {
    write_draft(w, origin, w.lives_in("Anna", city)).unwrap()
}

fn single_text(w: &World, n: &str) -> PredicateId {
    w.memory
        .define_predicate_unchecked(w.space, predicate(n, ValueType::Text, Cardinality::Single))
        .unwrap()
}

fn set(w: &World, predicate: PredicateId, text: &str) -> FactDraft {
    FactDraft::new(BodyDraft::Triple {
        subject: w.person("Anna"),
        predicate,
        object: ObjectDraft::Value(tessar_core::model::Value::Text(text.into())),
        text: None,
        when: None,
    })
}

#[test]
fn inferred_beside_asserted_is_a_conflict() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let i = write(&w, inferred(), "Berlin");
    assert_eq!(i.outcome, Outcome::Conflicting { current: a.fact });
    let (af, inf) = (w.fact(a), w.fact(i));
    assert_ne!(af.lineage, inf.lineage);
    assert_eq!(w.memory.current(af.lineage), Some(af.clone()));
    let anna = w.entity(w.person, "Anna");
    assert_eq!(w.memory.values_of(anna, w.lives_in).len(), 2);

    // The same inferred value again is unchanged.
    assert_eq!(write(&w, inferred(), "Berlin").outcome, Outcome::Unchanged);
    // A new inferred value continues the inferred head only.
    let i2 = write(&w, inferred(), "Paris");
    assert_eq!(i2.outcome, Outcome::Superseded { previous: i.fact });
    assert_eq!(w.memory.current(af.lineage), Some(af.clone()));
    assert_eq!(w.memory.values_of(anna, w.lives_in).len(), 2);

    // An asserted write continues the asserted lineage and stops the
    // inferred one.
    w.clock.set(ts(200));
    let s = write(&w, origin(), "Oslo");
    assert_eq!(s.outcome, Outcome::Superseded { previous: a.fact });
    assert_eq!(w.fact(s).lineage, af.lineage);
    let history: Vec<_> = w.memory.history(af.lineage).iter().map(|f| f.id).collect();
    assert_eq!(history, [a.fact, s.fact]);
    let stopped = w.fact(i2).end.unwrap();
    assert_eq!(stopped.next, None);
    assert!(matches!(stopped.ending, Ending::Changed { .. }));
    let vals = w.memory.values_of(anna, w.lives_in);
    assert_eq!(vals.len(), 1);
    assert_eq!(vals[0].id, s.fact);
}

#[test]
fn inferred_after_inferred_supersedes() {
    let w = World::new();
    let a = write(&w, inferred(), "Moscow");
    let b = write(&w, inferred(), "Berlin");
    assert_eq!(b.outcome, Outcome::Superseded { previous: a.fact });
}

#[test]
fn asserted_after_inferred_supersedes() {
    let w = World::new();
    let a = write(&w, inferred(), "Moscow");
    let b = write(&w, origin(), "Berlin");
    assert_eq!(b.outcome, Outcome::Superseded { previous: a.fact });
}

#[test]
fn update_closes_an_asserted_head_with_any_origin() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let f = w.fact(a);
    let r = w
        .memory
        .update(Update {
            lineage: f.lineage,
            expected_version: v(1),
            reason: Reason::Correction,
            replacement: Some(Replacement {
                body: w.lives_in("Anna", "Berlin").body,
                mentions: vec![],
            }),
            origin: inferred(),
        })
        .unwrap();
    assert_eq!(r.outcome, Outcome::Superseded { previous: a.fact });
}

#[test]
fn an_inferred_write_closes_every_inferred_head_after_a_merge() {
    let w = World::new();
    let p1 = single_text(&w, "aim");
    let p2 = single_text(&w, "target");
    let x = write_draft(&w, inferred(), set(&w, p1, "x")).unwrap();
    w.clock.set(ts(200));
    let y = write_draft(&w, inferred(), set(&w, p2, "y")).unwrap();
    w.memory.merge_predicates(p1, p2).unwrap();
    let anna = w.entity(w.person, "Anna");
    assert_eq!(w.memory.values_of(anna, p2).len(), 2);

    w.clock.set(ts(300));
    let z = write_draft(&w, inferred(), set(&w, p2, "z")).unwrap();
    assert_eq!(z.outcome, Outcome::Superseded { previous: y.fact });
    assert_eq!(w.fact(x).end.unwrap().next, None);
    let now = w.memory.values_of(anna, p2);
    assert_eq!(now.len(), 1);
    assert_eq!(now[0].id, z.fact);
}

#[test]
fn an_inferred_write_after_a_merge_spares_the_asserted_head() {
    let w = World::new();
    let p1 = single_text(&w, "aim");
    let p2 = single_text(&w, "target");
    let x = write_draft(&w, inferred(), set(&w, p1, "x")).unwrap();
    w.clock.set(ts(200));
    let a = write_draft(&w, origin(), set(&w, p2, "a")).unwrap();
    w.memory.merge_predicates(p1, p2).unwrap();

    w.clock.set(ts(300));
    let z = write_draft(&w, inferred(), set(&w, p2, "z")).unwrap();
    assert_eq!(z.outcome, Outcome::Superseded { previous: x.fact });
    let anna = w.entity(w.person, "Anna");
    let mut ids: Vec<_> = w.memory.values_of(anna, p2).iter().map(|f| f.id).collect();
    ids.sort();
    let mut want = [a.fact, z.fact];
    want.sort();
    assert_eq!(ids, want);
}

#[test]
fn the_conflict_survives_replay() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let i = write(&w, inferred(), "Berlin");
    assert_eq!(i.outcome, Outcome::Conflicting { current: a.fact });
    let anna = w.entity(w.person, "Anna");
    let before = w.memory.values_of(anna, w.lives_in);
    let again = w.reopen();
    assert_eq!(again.values_of(anna, w.lives_in), before);
}

#[test]
fn expected_version_is_checked_against_the_lineage_the_write_continues() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let i1 = write(&w, inferred(), "Berlin");
    assert!(matches!(i1.outcome, Outcome::Conflicting { .. }));
    let i2 = write(&w, inferred(), "Paris");
    assert_eq!(w.fact(i2).version.get(), 2);
    let expecting = |n: u32, city: &str| FactDraft {
        expected_version: Some(v(n)),
        ..w.lives_in("Anna", city)
    };

    // The inferred head is at version 2, the asserted one at 1.
    let err = write_draft(&w, inferred(), expecting(1, "Rome")).unwrap_err();
    assert!(matches!(err, StoreError::Conflict { .. }), "{err:?}");
    let ok = write_draft(&w, inferred(), expecting(2, "Rome")).unwrap();
    assert_eq!(ok.outcome, Outcome::Superseded { previous: i2.fact });

    // An asserted write continues the asserted lineage, at version 1.
    let err = write_draft(&w, origin(), expecting(2, "Oslo")).unwrap_err();
    assert!(matches!(err, StoreError::Conflict { .. }), "{err:?}");
    let ok = write_draft(&w, origin(), expecting(1, "Oslo")).unwrap();
    assert_eq!(ok.outcome, Outcome::Superseded { previous: a.fact });
}

#[test]
fn expected_version_on_a_conflicting_write_fails() {
    let w = World::new();
    write(&w, origin(), "Moscow");
    let draft = FactDraft {
        expected_version: Some(v(1)),
        ..w.lives_in("Anna", "Berlin")
    };
    let err = write_draft(&w, inferred(), draft).unwrap_err();
    assert!(matches!(err, StoreError::Conflict { .. }), "{err:?}");
    let anna = w.entity(w.person, "Anna");
    assert_eq!(w.memory.values_of(anna, w.lives_in).len(), 1);
}

/// What a memory rebuilt from the log must agree on: the facts, their
/// chains and the current values of the pair.
fn assert_replays(w: &World, facts: &[Recorded], predicate: PredicateId) {
    let again = w.reopen();
    let anna = w.entity(w.person, "Anna");
    assert_eq!(
        again.values_of(anna, predicate),
        w.memory.values_of(anna, predicate)
    );
    for r in facts {
        let fact = w.fact(*r);
        assert_eq!(again.fact(r.fact), Some(fact.clone()));
        assert_eq!(again.history(fact.lineage), w.memory.history(fact.lineage));
    }
}

#[test]
fn an_asserted_write_confirms_the_inferred_value_of_a_conflict() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let i = write(&w, inferred(), "Berlin");
    assert_eq!(i.outcome, Outcome::Conflicting { current: a.fact });

    // The user confirms the inferred value: it becomes the asserted
    // current one, as the next version of the asserted lineage.
    w.clock.set(ts(200));
    let c = write(&w, origin(), "Berlin");
    assert_eq!(c.outcome, Outcome::Superseded { previous: a.fact });
    let confirmed = w.fact(c);
    assert!(matches!(confirmed.origin, Origin::Asserted { .. }));
    assert_eq!(confirmed.lineage, w.fact(a).lineage);
    assert_eq!(confirmed.version, v(2));
    assert_eq!(w.fact(a).end.unwrap().next, Some(c.fact));
    let stopped = w.fact(i).end.unwrap();
    assert_eq!(stopped.next, None);
    assert_eq!(stopped.ending, Ending::Changed { valid_to: ts(200) });
    let anna = w.entity(w.person, "Anna");
    assert_eq!(w.memory.values_of(anna, w.lives_in), [confirmed]);
    assert_replays(&w, &[a, i, c], w.lives_in);

    // Now it is the only current value: writing it again changes nothing.
    let commits = w.log.commits();
    let again = write(&w, origin(), "Berlin");
    assert_eq!(again.outcome, Outcome::Unchanged);
    assert_eq!(again.fact, c.fact);
    assert_eq!(write(&w, inferred(), "Berlin").outcome, Outcome::Unchanged);
    assert_eq!(w.log.commits(), commits);
}

#[test]
fn an_asserted_write_of_the_asserted_value_ends_the_conflict() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let i = write(&w, inferred(), "Berlin");
    w.clock.set(ts(200));
    let before = w.fact(a);
    let c = write(&w, origin(), "Moscow");
    // The asserted version is kept as it is; only the inferred one ends.
    let (closed, others) = (i.fact, 0);
    assert_eq!(c.outcome, Outcome::Resolved { closed, others });
    assert_eq!(c.fact, a.fact);
    assert_eq!(w.fact(a), before);
    assert_eq!(w.memory.history(before.lineage).len(), 1);
    let stopped = w.fact(i).end.unwrap();
    assert_eq!(stopped.next, None);
    assert_eq!(stopped.ending, Ending::Changed { valid_to: ts(200) });
    let anna = w.entity(w.person, "Anna");
    assert_eq!(w.memory.values_of(anna, w.lives_in), [before]);
    assert_replays(&w, &[a, i], w.lives_in);

    let commits = w.log.commits();
    assert_eq!(write(&w, origin(), "Moscow").outcome, Outcome::Unchanged);
    assert_eq!(w.log.commits(), commits);
}

/// The agent and the model keep disagreeing: the asserted lineage does
/// not grow, each repeated assertion only closes the new guess.
#[test]
fn alternating_asserted_and_inferred_values_do_not_grow_the_asserted_lineage() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let lineage = w.fact(a).lineage;
    let mut guess = write(&w, inferred(), "Berlin");
    for cycle in 1..4 {
        w.clock.set(ts(100 + cycle * 100));
        let again = write(&w, origin(), "Moscow");
        let (closed, others) = (guess.fact, 0);
        assert_eq!(again.outcome, Outcome::Resolved { closed, others });
        assert_eq!(again.fact, a.fact);
        guess = write(&w, inferred(), "Berlin");
        assert_eq!(guess.outcome, Outcome::Conflicting { current: a.fact });
    }
    assert_eq!(w.memory.history(lineage).len(), 1);
    assert_eq!(w.fact(a).valid_from, ts(100));
    let anna = w.entity(w.person, "Anna");
    assert_eq!(w.memory.values_of(anna, w.lives_in).len(), 2);
    assert_replays(&w, &[a, guess], w.lives_in);
}

/// A journal of the planner before `Outcome::Resolved`: the same value
/// current twice, asserted and inferred. It replays as it was written,
/// and the next write of the value removes the duplicate.
#[test]
fn a_duplicate_pair_from_an_old_journal_replays_and_is_resolved() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let i = write(&w, inferred(), "Berlin");
    // What the old planner wrote for an inferred "Moscow" here: the next
    // version of the inferred lineage, closing the inferred head.
    let mut commits = serde_json::to_value(w.log.snapshot()).unwrap();
    let list = commits.as_array_mut().unwrap();
    let mut last = list.last().unwrap().clone();
    let mut fact = last["ops"]
        .as_array()
        .unwrap()
        .iter()
        .find(|op| op["op"] == "add_fact")
        .unwrap()
        .clone();
    let old = serde_json::to_value(w.fact(a)).unwrap();
    let id = i.fact.get() + 1;
    fact["id"] = id.into();
    fact["version"] = 2.into();
    fact["prev"] = i.fact.get().into();
    fact["body"] = old["body"].clone();
    let seq = list.len() + 1;
    last["seq"] = seq.into();
    last["ops"] = serde_json::json!([
        fact,
        {"op": "close_fact", "fact": i.fact.get(),
         "end": {"ending": "changed", "valid_to": 100, "recorded_at": 100, "next": id}},
    ]);
    list.push(last);
    let log = SharedLog::from_commits(serde_json::from_value(commits).unwrap());
    let old = tessar_core::store::Memory::open(log.clone(), w.clock.clone()).unwrap();
    let anna = w.entity(w.person, "Anna");
    let both = old.values_of(anna, w.lives_in);
    assert_eq!(both.len(), 2);
    assert!(both[0].body.same_content(&both[1].body));

    let request = |origin| Remember {
        space: w.space,
        origin,
        episode: None,
        facts: vec![w.lives_in("Anna", "Moscow")],
    };
    let done = old.remember(request(inferred())).unwrap().facts[0];
    let (closed, others) = (both[1].id, 0);
    assert_eq!(done.outcome, Outcome::Resolved { closed, others });
    assert_eq!(done.fact, a.fact);
    assert_eq!(old.values_of(anna, w.lives_in), [w.fact(a)]);
    let again = tessar_core::store::Memory::open(log, w.clock.clone()).unwrap();
    assert_eq!(again.values_of(anna, w.lives_in), [w.fact(a)]);
}

#[test]
fn an_asserted_write_confirms_a_lone_inferred_value() {
    let w = World::new();
    let i = write(&w, inferred(), "Berlin");
    w.clock.set(ts(200));
    let c = write(&w, origin(), "Berlin");
    assert_eq!(c.outcome, Outcome::Superseded { previous: i.fact });
    let confirmed = w.fact(c);
    assert!(matches!(confirmed.origin, Origin::Asserted { .. }));
    assert_eq!(confirmed.lineage, w.fact(i).lineage);
    let anna = w.entity(w.person, "Anna");
    assert_eq!(w.memory.values_of(anna, w.lives_in), [w.fact(c)]);
    assert_replays(&w, &[i, c], w.lives_in);

    // The value did not change at 200: the asserted version is true for
    // the whole period, the inferred one at no time, so no read shows the
    // value twice. What memory held when is still told apart.
    assert_eq!(confirmed.valid_from, ts(100));
    let end = w.fact(i).end.unwrap();
    assert_eq!(end.ending, Ending::Changed { valid_to: ts(100) });
    assert_eq!(end.next, Some(c.fact));
    let ids = |view| -> Vec<_> {
        let facts = w.memory.facts_about(anna, view);
        facts.iter().map(|f| f.id).collect()
    };
    for t in [100, 150, 200, 300] {
        assert_eq!(ids(View::TrueAt(ts(t))), [c.fact], "true at {t}");
    }
    assert_eq!(ids(View::KnownAt(ts(150))), [i.fact]);
    assert_eq!(ids(View::KnownAt(ts(250))), [c.fact]);
    let history: Vec<_> = w.memory.history(confirmed.lineage);
    let history: Vec<_> = history.iter().map(|f| f.id).collect();
    assert_eq!(history, [i.fact, c.fact]);
}

/// A confirmation dated before an earlier value ended is rejected like any
/// other write of the pair, though the value is current as inferred.
#[test]
fn a_dated_confirmation_may_not_reach_back_over_an_earlier_value() {
    let w = World::new();
    let x = w.remember_one(w.goal("x"));
    w.clock.set(ts(300));
    w.memory
        .update(Update {
            lineage: w.fact(x).lineage,
            expected_version: v(1),
            reason: Reason::Change { at: Some(ts(200)) },
            replacement: None,
            origin: origin(),
        })
        .unwrap();
    w.clock.set(ts(400));
    write_draft(&w, inferred(), w.goal("a")).unwrap();
    w.clock.set(ts(500));
    let err = write_draft(
        &w,
        origin(),
        FactDraft {
            valid_from: Some(ts(150)),
            ..w.goal("a")
        },
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            StoreError::ChangeBeforeValid { at, valid_from }
                if at == ts(150) && valid_from == ts(200)
        ),
        "{err:?}"
    );
    // Nothing was written: only "x" was true at 170.
    let project = w.entity(w.project, "tessar");
    let at_170 = w.memory.facts_about(project, View::TrueAt(ts(170)));
    let ids: Vec<_> = at_170.iter().map(|f| f.id).collect();
    assert_eq!(ids, [x.fact]);
}

/// An undated confirmation continues the inferred lineage from the
/// inferred value's own start, even if an asserted value of another
/// lineage ended after that: the overlap is the conflict that was already
/// there, not one inside a lineage.
#[test]
fn an_undated_confirmation_is_not_bounded_by_another_lineage() {
    let w = World::new();
    let x = w.remember_one(w.goal("x"));
    w.clock.set(ts(200));
    let i = write_draft(&w, inferred(), w.goal("a")).unwrap();
    assert!(matches!(i.outcome, Outcome::Conflicting { .. }), "{i:?}");
    w.clock.set(ts(300));
    w.memory
        .update(Update {
            lineage: w.fact(x).lineage,
            expected_version: v(1),
            reason: Reason::Change { at: None },
            replacement: None,
            origin: origin(),
        })
        .unwrap();
    w.clock.set(ts(400));
    let done = write_draft(&w, origin(), w.goal("a")).unwrap();
    assert_eq!(done.outcome, Outcome::Superseded { previous: i.fact });
    assert_eq!(w.fact(done).valid_from, ts(200));
}

/// A confirmation dated before the inferred value is accepted, and the
/// other facts of its batch are written with it.
#[test]
fn a_backdated_confirmation_is_taken_at_its_word() {
    let w = World::new();
    w.clock.set(ts(1_000));
    let i = write(&w, inferred(), "Berlin");
    w.clock.set(ts(2_000));
    let done = w
        .try_remember(vec![
            w.knows("Zed", "Rust"),
            FactDraft {
                valid_from: Some(ts(500)),
                ..w.lives_in("Anna", "Berlin")
            },
        ])
        .unwrap()
        .facts;
    assert_eq!(done[0].outcome, Outcome::Created);
    assert_eq!(done[1].outcome, Outcome::Superseded { previous: i.fact });
    assert_eq!(w.fact(done[1]).valid_from, ts(500));
    // The inferred version never ends before it starts.
    let end = w.fact(i).end.unwrap().ending;
    assert_eq!(
        end,
        Ending::Changed {
            valid_to: ts(1_000)
        }
    );
    let anna = w.entity(w.person, "Anna");
    for t in [500, 1_000, 1_500] {
        let facts = w.memory.facts_about(anna, View::TrueAt(ts(t)));
        let ids: Vec<_> = facts.iter().map(|f| f.id).collect();
        assert_eq!(ids, [done[1].fact], "true at {t}");
    }

    // That empty period bounds nothing once the pair has no value.
    let lineage = w.fact(i).lineage;
    w.memory
        .update(Update {
            lineage,
            expected_version: v(2),
            reason: Reason::Change { at: Some(ts(600)) },
            replacement: None,
            origin: origin(),
        })
        .unwrap();
    let later = FactDraft {
        valid_from: Some(ts(700)),
        ..w.lives_in("Anna", "Oslo")
    };
    assert_eq!(
        write_draft(&w, origin(), later).unwrap().outcome,
        Outcome::Created
    );
    assert_replays(&w, &[i, done[1]], w.lives_in);
}

#[test]
fn an_inferred_write_of_the_asserted_value_is_unchanged() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let commits = w.log.commits();
    let same = write(&w, inferred(), "Moscow");
    assert_eq!(same.outcome, Outcome::Unchanged);
    assert_eq!(same.fact, a.fact);
    assert_eq!(w.log.commits(), commits);
}

#[test]
fn an_inferred_write_agreeing_with_the_asserted_value_closes_the_inferred_one() {
    let w = World::new();
    let a = write(&w, origin(), "Moscow");
    let i = write(&w, inferred(), "Berlin");
    let before = w.fact(a);
    let i2 = write(&w, inferred(), "Moscow");
    let (closed, others) = (i.fact, 0);
    assert_eq!(i2.outcome, Outcome::Resolved { closed, others });
    assert_eq!(i2.fact, a.fact);
    // The asserted value is untouched and is the only one: no inferred
    // copy of it beside it.
    let anna = w.entity(w.person, "Anna");
    assert_eq!(w.memory.values_of(anna, w.lives_in), [before]);
    assert_eq!(w.fact(i).end.unwrap().next, None);
    let commits = w.log.commits();
    assert_eq!(write(&w, inferred(), "Moscow").outcome, Outcome::Unchanged);
    assert_eq!(w.log.commits(), commits);
    assert_replays(&w, &[a, i], w.lives_in);
}

/// After a merge two asserted values are current. An inferred write of
/// one of them may close neither.
#[test]
fn an_inferred_write_never_closes_an_asserted_head() {
    let w = World::new();
    let p1 = single_text(&w, "aim");
    let p2 = single_text(&w, "target");
    let x = write_draft(&w, origin(), set(&w, p1, "x")).unwrap();
    let y = write_draft(&w, origin(), set(&w, p2, "y")).unwrap();
    w.memory.merge_predicates(p1, p2).unwrap();
    let commits = w.log.commits();
    let same = write_draft(&w, inferred(), set(&w, p2, "x")).unwrap();
    assert_eq!((same.outcome, same.fact), (Outcome::Unchanged, x.fact));
    assert_eq!(w.log.commits(), commits);
    let other = write_draft(&w, inferred(), set(&w, p2, "z")).unwrap();
    assert!(matches!(other.outcome, Outcome::Conflicting { .. }));
    assert!(w.fact(x).is_current() && w.fact(y).is_current());
}

#[test]
fn an_asserted_write_of_one_merged_value_closes_the_other() {
    let w = World::new();
    let p1 = single_text(&w, "aim");
    let p2 = single_text(&w, "target");
    let x = write_draft(&w, origin(), set(&w, p1, "x")).unwrap();
    w.clock.set(ts(200));
    let y = write_draft(&w, inferred(), set(&w, p2, "y")).unwrap();
    w.memory.merge_predicates(p1, p2).unwrap();

    // The inferred head is the newer one and the write equals the
    // asserted one: the asserted version stays, the inferred stops.
    w.clock.set(ts(300));
    let again = write_draft(&w, origin(), set(&w, p2, "x")).unwrap();
    let (closed, others) = (y.fact, 0);
    assert_eq!(again.outcome, Outcome::Resolved { closed, others });
    assert_eq!(again.fact, x.fact);
    assert_eq!(w.fact(y).end.unwrap().next, None);
    let anna = w.entity(w.person, "Anna");
    assert_eq!(w.memory.values_of(anna, p2), [w.fact(x)]);
    assert_replays(&w, &[x, y], p2);
}
