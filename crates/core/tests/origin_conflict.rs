//! An inferred value never replaces an asserted one.

mod common;

use common::*;
use tessar_core::model::{Cardinality, Confidence, Ending, Origin, PredicateId, ValueType};
use tessar_core::store::{
    BodyDraft, FactDraft, ObjectDraft, Outcome, Reason, Recorded, Remember, Replacement,
    StoreError, Update,
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
