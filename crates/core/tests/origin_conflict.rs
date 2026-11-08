//! An inferred value never replaces an asserted one.

mod common;

use common::*;
use tessar_core::model::{Confidence, Origin};
use tessar_core::store::{Outcome, Reason, Recorded, Remember, Replacement, Update};

fn inferred() -> Origin {
    Origin::Inferred {
        by: key("model"),
        confidence: Confidence::new(0.6).unwrap(),
    }
}

fn write(w: &World, origin: Origin, city: &str) -> Recorded {
    w.memory
        .remember(Remember {
            space: w.space,
            origin,
            episode: None,
            facts: vec![w.lives_in("Anna", city)],
        })
        .unwrap()
        .facts[0]
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

    // An asserted write closes both.
    let s = write(&w, origin(), "Oslo");
    assert!(matches!(s.outcome, Outcome::Superseded { .. }));
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
