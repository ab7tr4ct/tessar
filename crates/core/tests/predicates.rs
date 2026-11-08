//! The predicate vocabulary: checks on new names, synonyms, merges,
//! replay and the starter set.

mod common;

use std::sync::Arc;

use common::*;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::model::{Cardinality, FactBody, PredicateId, Value, ValueType};
use tessar_core::store::{
    BodyDraft, FactDraft, Memory, ObjectDraft, Outcome, PredicateDraft, Semantic, StoreError,
};
use tessar_core::vocab;

const DIMS: usize = 256;
const THRESHOLD: f32 = 0.6;

/// A bag of words: every word is a hashed dimension, so texts that share
/// words are similar and texts that share none are orthogonal.
struct Words;

impl Embedder for Words {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| embed(t)).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(embed(text))
    }
}

fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0; DIMS];
    for word in text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let hash = word.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
        v[(hash % DIMS as u64) as usize] += 1.0;
    }
    if v.iter().all(|&x| x == 0.0) {
        v[0] = 1.0;
    }
    v
}

fn semantic() -> Semantic {
    Semantic {
        embedder: Arc::new(Words),
        dims: DIMS,
        near_bonus: 0.0,
        predicate_threshold: THRESHOLD,
    }
}

/// A world with the meaning channel and a vocabulary that has
/// `trip destination: where the subject plans to travel`.
fn world_with_trip() -> (World, PredicateId) {
    let w = World::with(Some(semantic()));
    let trip = w
        .memory
        .define_predicate(
            w.space,
            described(
                "trip_destination",
                "where the subject plans to travel",
                ValueType::Text,
                Cardinality::Many,
            ),
        )
        .unwrap();
    (w, trip)
}

fn described(n: &str, description: &str, object: ValueType, c: Cardinality) -> PredicateDraft {
    PredicateDraft {
        name: name(n),
        description: description.to_owned(),
        object,
        cardinality: c,
    }
}

fn text_draft(n: &str) -> PredicateDraft {
    predicate(n, ValueType::Text, Cardinality::Many)
}

fn plans_visit() -> PredicateDraft {
    described(
        "plans_to_visit",
        "where the subject plans to travel",
        ValueType::Text,
        Cardinality::Many,
    )
}

/// The predicate id the store gave a name, found through a synonym's
/// definition order: ids are consecutive record numbers.
fn pid(n: u32) -> PredicateId {
    serde_json::from_str(&n.to_string()).unwrap()
}

fn set(w: &World, who: &str, predicate: PredicateId, text: &str) -> FactDraft {
    FactDraft::new(BodyDraft::Triple {
        subject: w.person(who),
        predicate,
        object: ObjectDraft::Value(Value::Text(text.into())),
        text: None,
    })
}

fn stored_predicate(w: &World, fact: tessar_core::store::Recorded) -> PredicateId {
    match w.fact(fact).body {
        FactBody::Triple { predicate, .. } => predicate,
        other => panic!("not a triple: {other:?}"),
    }
}

#[test]
fn existing_name_is_reused_and_conflicts_on_other_shape() {
    let w = World::new();
    let again = w
        .memory
        .define_predicate(
            w.space,
            predicate("goal", ValueType::Text, Cardinality::Single),
        )
        .unwrap();
    assert_eq!(again, w.goal);
    let err = w
        .memory
        .define_predicate(
            w.space,
            predicate("goal", ValueType::Text, Cardinality::Many),
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::PredicateConflict { .. }), "{err}");
    let err = w
        .memory
        .define_predicate_unchecked(
            w.space,
            predicate("goal", ValueType::Int, Cardinality::Single),
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::PredicateConflict { .. }), "{err}");
}

#[test]
fn reuse_wins_over_checks() {
    // A name that already exists is reused even if it would fail the
    // universality check today.
    let w = World::new();
    let id = w
        .memory
        .define_predicate_unchecked(w.space, text_draft("plans_trip_to_japan"))
        .unwrap();
    w.remember(vec![w.lives_in("anna", "Japan")]);
    let again = w
        .memory
        .define_predicate(w.space, text_draft("plans_trip_to_japan"))
        .unwrap();
    assert_eq!(again, id);
}

#[test]
fn too_many_words_is_rejected_by_both_checks() {
    let w = World::new();
    let long = "plans_to_visit_some_place";
    let err = w
        .memory
        .define_predicate(w.space, text_draft(long))
        .unwrap_err();
    assert!(
        matches!(err, StoreError::PredicateNameTooLong { max_words: 4, .. }),
        "{err}"
    );
    let err = w
        .memory
        .define_predicate_unchecked(w.space, text_draft(long))
        .unwrap_err();
    assert!(
        matches!(err, StoreError::PredicateNameTooLong { .. }),
        "{err}"
    );
    // Four words are fine, and empty parts do not count.
    w.memory
        .define_predicate(w.space, text_draft("plans__to_visit_x"))
        .unwrap();
}

#[test]
fn a_number_in_the_name_is_rejected() {
    let w = World::new();
    for n in ["top_10_songs", "best_5k_time", "level_2", "pi_3"] {
        let r = w.memory.define_predicate(w.space, text_draft(n));
        if n == "best_5k_time" {
            // "5k" is not a number.
            r.unwrap();
            continue;
        }
        match r.unwrap_err() {
            StoreError::PredicateNotUniversal { reason, .. } => {
                assert_eq!(reason, "is a number", "{n}");
            }
            other => panic!("{n}: {other}"),
        }
    }
    // The format-only path lets it in.
    w.memory
        .define_predicate_unchecked(w.space, text_draft("top_10_songs"))
        .unwrap();
    // Words that parse as floats in Rust are still words.
    w.memory
        .define_predicate(w.space, text_draft("nan_inf"))
        .unwrap();
}

#[test]
fn an_entity_name_in_the_name_is_rejected() {
    let w = World::new();
    w.remember(vec![w.lives_in("anna", "Japan")]);
    match w
        .memory
        .define_predicate(w.space, text_draft("plans_trip_to_japan"))
        .unwrap_err()
    {
        StoreError::PredicateNotUniversal { word, reason, .. } => {
            assert_eq!(word, "japan");
            assert_eq!(reason, "is the name of an entity");
        }
        other => panic!("{other}"),
    }
    // The person is an entity too, found by key in any kind.
    let err = w
        .memory
        .define_predicate(w.space, text_draft("anna_likes"))
        .unwrap_err();
    assert!(
        matches!(err, StoreError::PredicateNotUniversal { .. }),
        "{err}"
    );

    // A whole name that is an entity: the entity "Plans To".
    w.remember(vec![w.lives_in("anna", "Plans To")]);
    match w
        .memory
        .define_predicate(w.space, text_draft("plans_to"))
        .unwrap_err()
    {
        StoreError::PredicateNotUniversal { word, .. } => assert_eq!(word, "plans to"),
        other => panic!("{other}"),
    }
    // Aliases count as names.
    let japan = w.entity(w.place, "Japan");
    w.memory.add_alias(japan, key("Nippon")).unwrap();
    let err = w
        .memory
        .define_predicate(w.space, text_draft("visits_nippon"))
        .unwrap_err();
    assert!(
        matches!(err, StoreError::PredicateNotUniversal { .. }),
        "{err}"
    );
    // Format-only skips the universality check.
    w.memory
        .define_predicate_unchecked(w.space, text_draft("plans_trip_to_japan"))
        .unwrap();
}

#[test]
fn whole_name_matches_an_entity_with_several_words() {
    let w = World::new();
    w.remember(vec![w.lives_in("anna", "works at")]);
    match w
        .memory
        .define_predicate(w.space, text_draft("works_at"))
        .unwrap_err()
    {
        StoreError::PredicateNotUniversal { word, reason, .. } => {
            assert_eq!(word, "works at");
            assert_eq!(reason, "is the name of an entity");
        }
        other => panic!("{other}"),
    }
}

#[test]
fn a_similar_name_is_rejected_with_the_closest_predicate() {
    let (w, trip) = world_with_trip();
    match w
        .memory
        .define_predicate(w.space, plans_visit())
        .unwrap_err()
    {
        StoreError::PredicateSimilar {
            existing,
            similarity,
            name,
        } => {
            assert_eq!(existing, trip);
            assert!(similarity >= THRESHOLD, "{similarity}");
            assert_eq!(name, "plans_to_visit");
        }
        other => panic!("{other}"),
    }
    // An unrelated predicate passes.
    w.memory
        .define_predicate(
            w.space,
            described("owns", "", ValueType::Text, Cardinality::Many),
        )
        .unwrap();
    // The unchecked path accepts the similar one.
    let id = w
        .memory
        .define_predicate_unchecked(w.space, plans_visit())
        .unwrap();
    assert_ne!(id, trip);
}

#[test]
fn without_an_embedder_similarity_is_skipped() {
    let w = World::new();
    w.memory
        .define_predicate(
            w.space,
            described(
                "trip_destination",
                "where the subject plans to travel",
                ValueType::Text,
                Cardinality::Many,
            ),
        )
        .unwrap();
    w.memory.define_predicate(w.space, plans_visit()).unwrap();
}

#[test]
fn similarity_reports_the_canonical_of_a_synonym() {
    let (w, trip) = world_with_trip();
    // A synonym whose text is far from the canonical one.
    w.memory
        .alias_predicate(
            w.space,
            name("holiday_spot"),
            "vacation resort".into(),
            trip,
        )
        .unwrap();
    let err = w
        .memory
        .define_predicate(
            w.space,
            described(
                "vacation_resort",
                "vacation resort",
                ValueType::Text,
                Cardinality::Many,
            ),
        )
        .unwrap_err();
    match err {
        StoreError::PredicateSimilar { existing, .. } => assert_eq!(existing, trip),
        other => panic!("{other}"),
    }
}

#[test]
fn entity_kind_of_a_predicate_must_exist() {
    let w = World::new();
    // A kind id from nowhere: person+100 does not exist in `w`.
    let missing: tessar_core::model::EntityKindId = serde_json::from_str("100").unwrap();
    let err = w
        .memory
        .define_predicate(
            w.space,
            predicate(
                "friend_of",
                ValueType::Entity(Some(missing)),
                Cardinality::Many,
            ),
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::UnknownKind(_)), "{err}");
}

#[test]
fn alias_writes_under_the_canonical_predicate() {
    let w = World::new();
    let canonical = w
        .memory
        .alias_predicate(w.space, name("aim"), "what to achieve".into(), w.goal)
        .unwrap();
    assert_eq!(canonical, w.goal);
    let alias = pid(4); // the next id after the world's four predicates

    // The synonym is not in the vocabulary, but is a known predicate.
    let names: Vec<_> = w
        .memory
        .predicates()
        .iter()
        .map(|p| p.name.to_string())
        .collect();
    assert_eq!(names, ["lives_in", "goal", "knows", "counter"]);
    assert_eq!(w.memory.predicate(alias).unwrap().name.as_str(), "aim");
    assert_eq!(w.memory.canonical_predicate(alias), Some(w.goal));
    assert_eq!(w.memory.canonical_predicate(w.goal), Some(w.goal));
    assert_eq!(w.memory.canonical_predicate(pid(99)), None);

    // By name, the synonym gives the canonical definition.
    let def = w.memory.predicate_named(w.space, "aim").unwrap();
    assert_eq!(def.id, w.goal);

    // A write with the alias id is stored under the canonical id.
    let r = w.remember_one(set(&w, "anna", alias, "ship it"));
    assert_eq!(stored_predicate(&w, r), w.goal);
    let anna = w.entity(w.person, "anna");
    assert_eq!(w.memory.values_of(anna, w.goal).len(), 1);
    assert_eq!(w.memory.values_of(anna, alias).len(), 1);

    // Idempotent: the same name again, from either id.
    assert_eq!(
        w.memory
            .alias_predicate(w.space, name("aim"), String::new(), alias)
            .unwrap(),
        w.goal
    );
    // A name taken by an unrelated predicate.
    let err = w
        .memory
        .alias_predicate(w.space, name("knows"), String::new(), w.goal)
        .unwrap_err();
    assert!(matches!(err, StoreError::NameTaken(_)), "{err}");
    // Unknown target.
    let err = w
        .memory
        .alias_predicate(w.space, name("aim2"), String::new(), pid(99))
        .unwrap_err();
    assert!(matches!(err, StoreError::UnknownPredicate(_)), "{err}");
    // The format check still applies to aliases.
    let err = w
        .memory
        .alias_predicate(w.space, name("a_b_c_d_e"), String::new(), w.goal)
        .unwrap_err();
    assert!(
        matches!(err, StoreError::PredicateNameTooLong { .. }),
        "{err}"
    );
}

#[test]
fn single_valued_replacement_works_across_synonyms() {
    let w = World::new();
    let from = w
        .memory
        .define_predicate(
            w.space,
            predicate("target", ValueType::Text, Cardinality::Single),
        )
        .unwrap();
    let a = w.remember_one(set_single(&w, "anna", from, "v1"));
    assert_eq!(a.outcome, Outcome::Created);

    w.memory.merge_predicates(from, w.goal).unwrap();
    let b = w.remember_one(set_single(&w, "anna", w.goal, "v2"));
    assert_eq!(
        a.fact,
        match b.outcome {
            Outcome::Superseded { previous } => previous,
            other => panic!("{other:?}"),
        }
    );
    let anna = w.entity(w.person, "anna");
    let now = w.memory.values_of(anna, w.goal);
    assert_eq!(now.len(), 1);
    assert_eq!(now[0].id, b.fact);
    // The old fact keeps the predicate it was written with.
    assert_eq!(stored_predicate(&w, a), from);
    assert_eq!(stored_predicate(&w, b), w.goal);
    // Reading by the synonym finds the same.
    assert_eq!(w.memory.values_of(anna, from).len(), 1);
}

fn set_single(w: &World, who: &str, predicate: PredicateId, text: &str) -> FactDraft {
    set(w, who, predicate, text)
}

#[test]
fn merge_checks() {
    let w = World::new();
    // Another cardinality and another type.
    let err = w.memory.merge_predicates(w.knows, w.goal).unwrap_err();
    assert!(matches!(err, StoreError::MergeMismatch { .. }), "{err}");
    let err = w.memory.merge_predicates(w.counter, w.goal).unwrap_err();
    assert!(matches!(err, StoreError::MergeMismatch { .. }), "{err}");
    // Itself.
    let err = w.memory.merge_predicates(w.goal, w.goal).unwrap_err();
    assert!(matches!(err, StoreError::MergeMismatch { .. }), "{err}");
    // Unknown.
    let err = w.memory.merge_predicates(pid(99), w.goal).unwrap_err();
    assert!(matches!(err, StoreError::UnknownPredicate(_)), "{err}");
    let err = w.memory.merge_predicates(w.goal, pid(99)).unwrap_err();
    assert!(matches!(err, StoreError::UnknownPredicate(_)), "{err}");

    // An already merged predicate cannot be merged again.
    let a = single_text(&w, "aim");
    let b = single_text(&w, "target");
    w.memory.merge_predicates(a, w.goal).unwrap();
    let err = w.memory.merge_predicates(a, b).unwrap_err();
    assert!(matches!(err, StoreError::MergeMismatch { .. }), "{err}");

    // Merging into a synonym lands on its canonical.
    w.memory.merge_predicates(b, a).unwrap();
    assert_eq!(w.memory.canonical_predicate(b), Some(w.goal));
    let names: Vec<_> = w.memory.predicates().iter().map(|p| p.id).collect();
    assert_eq!(names, [w.lives_in, w.goal, w.knows, w.counter]);
}

fn single_text(w: &World, n: &str) -> PredicateId {
    w.memory
        .define_predicate_unchecked(w.space, predicate(n, ValueType::Text, Cardinality::Single))
        .unwrap()
}

#[test]
fn merging_moves_both_lineages_in_a_deterministic_order() {
    let w = World::new();
    let other = single_text(&w, "target");
    // Different subjects; the pair of `other` and `goal` for anna both
    // exist, so after the merge anna has two lineages under one pair.
    let first = w.remember_one(set(&w, "anna", other, "old"));
    let second = w.remember_one(set(&w, "anna", w.goal, "new"));
    w.memory.merge_predicates(other, w.goal).unwrap();
    let anna = w.entity(w.person, "anna");
    let ids: Vec<_> = w
        .memory
        .values_of(anna, w.goal)
        .iter()
        .map(|f| f.id)
        .collect();
    assert_eq!(ids, [first.fact, second.fact]);
}

#[test]
fn replay_rebuilds_synonyms_pairs_and_vectors() {
    let (w, trip) = world_with_trip();
    let canonical = w
        .memory
        .alias_predicate(w.space, name("aim"), String::new(), w.goal)
        .unwrap();
    assert_eq!(canonical, w.goal);
    let target = w
        .memory
        .define_predicate_unchecked(
            w.space,
            predicate("target", ValueType::Text, Cardinality::Single),
        )
        .unwrap();
    let before = w.remember_one(set(&w, "anna", target, "v1"));
    w.memory.merge_predicates(target, w.goal).unwrap();

    let reopened: Memory = w.reopen();
    assert_eq!(reopened.canonical_predicate(target), Some(w.goal));
    assert_eq!(reopened.predicate_named(w.space, "aim").unwrap().id, w.goal);
    assert_eq!(reopened.predicates(), w.memory.predicates());

    // A write after the reopen supersedes the value written under `from`.
    let r = reopened
        .remember(tessar_core::store::Remember {
            space: w.space,
            origin: origin(),
            episode: None,
            facts: vec![set(&w, "anna", w.goal, "v2")],
        })
        .unwrap()
        .facts[0];
    assert_eq!(
        r.outcome,
        Outcome::Superseded {
            previous: before.fact
        }
    );

    // Predicate vectors were rebuilt: the similar name is still rejected.
    let err = reopened
        .define_predicate(w.space, plans_visit())
        .unwrap_err();
    match err {
        StoreError::PredicateSimilar { existing, .. } => assert_eq!(existing, trip),
        other => panic!("{other}"),
    }
}

#[test]
fn old_journal_without_descriptions_replays() {
    // A definition without `description` in the journal still replays.
    let json = r#"{"op":"define_predicate","id":0,"space":0,"name":"goal","object":"text","cardinality":"single"}"#;
    let op: tessar_core::store::Op = serde_json::from_str(json).unwrap();
    match op {
        tessar_core::store::Op::DefinePredicate(p) => assert!(p.description.is_empty()),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_failed_embedder_writes_no_predicate() {
    struct Broken;
    impl Embedder for Broken {
        fn embed_documents(&self, _: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Err(EmbedError("down".into()))
        }
        fn embed_query(&self, _: &str) -> Result<Vec<f32>, EmbedError> {
            Err(EmbedError("down".into()))
        }
    }
    let log = SharedLog::default();
    let clock = Arc::new(tessar_core::clock::ManualClock::new(ts(1)));
    let memory = Memory::open_with(
        log.clone(),
        clock,
        Semantic {
            embedder: Arc::new(Broken),
            dims: 4,
            near_bonus: 0.0,
            predicate_threshold: 0.9,
        },
    )
    .unwrap();
    let space = memory
        .create_space(key("s"), tessar_core::model::SpaceKind::LongTerm, None)
        .unwrap();
    let commits = log.commits();
    let err = memory
        .define_predicate(space, text_draft("owns"))
        .unwrap_err();
    assert!(matches!(err, StoreError::Embed(_)), "{err}");
    let err = memory
        .define_predicate_unchecked(space, text_draft("owns"))
        .unwrap_err();
    assert!(matches!(err, StoreError::Embed(_)), "{err}");
    assert_eq!(log.commits(), commits);
    assert!(memory.predicates().is_empty());
}

#[test]
fn the_starter_vocabulary_applies_to_a_fresh_memory() {
    let starter = vocab::starter();
    assert!(starter.len() >= 20, "{}", starter.len());
    let mut names: Vec<_> = starter.iter().map(|d| d.name.to_string()).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), starter.len(), "duplicate names");
    assert!(starter.iter().all(|d| !d.description.is_empty()));
    assert!(starter.iter().all(|d| d.object == ValueType::Text));

    for semantic in [None, Some(semantic())] {
        let clock = Arc::new(tessar_core::clock::ManualClock::new(ts(1)));
        let memory = match semantic {
            Some(s) => Memory::open_with(SharedLog::default(), clock, s),
            None => Memory::open(SharedLog::default(), clock),
        }
        .unwrap();
        let space = memory
            .create_space(key("s"), tessar_core::model::SpaceKind::LongTerm, None)
            .unwrap();
        let before = memory.predicates().len();
        for draft in vocab::starter() {
            memory.define_predicate_unchecked(space, draft).unwrap();
        }
        let all = memory.predicates();
        assert_eq!(all.len(), before + starter.len());
        for d in &starter {
            let def = memory.predicate_named(space, d.name.as_str()).unwrap();
            assert_eq!(def.cardinality, d.cardinality);
            assert_eq!(def.description, d.description);
        }
        // Applying it twice changes nothing.
        for draft in vocab::starter() {
            memory.define_predicate_unchecked(space, draft).unwrap();
        }
        assert_eq!(memory.predicates().len(), all.len());
    }
}
