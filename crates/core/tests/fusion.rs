//! Recall tunables: the RRF constant, boost fusion, window tightness.

mod common;

use std::sync::Arc;

use common::*;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::model::{FactBody, Interval, TimeSpan};
use tessar_core::store::{BodyDraft, Channels, FactDraft, Fusion, Hit, Recall, Semantic};

const DIMS: usize = 4;
const CONCEPTS: [&str; 3] = ["deadline", "friday", "journal"];

/// One dimension per concept word; a little on the last one.
struct Concepts;

impl Concepts {
    fn vector(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; DIMS];
        v[DIMS - 1] = 0.1;
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .map(str::to_lowercase)
        {
            if let Some(c) = CONCEPTS.iter().position(|w| *w == word) {
                v[c] += 1.0;
            }
        }
        v
    }
}

impl Embedder for Concepts {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| Self::vector(t)).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(Self::vector(text))
    }
}

fn world() -> World {
    World::with(Some(Semantic {
        embedder: Arc::new(Concepts),
        dims: DIMS,
        near_bonus: 0.0,
        predicate_threshold: 1.0,
    }))
}

fn statement(text: &str) -> FactDraft {
    FactDraft::new(BodyDraft::Statement {
        text: text.into(),
        when: None,
    })
}

fn text(hit: &Hit) -> String {
    match &hit.fact.body {
        FactBody::Statement { text, .. } | FactBody::Event { text, .. } => text.clone(),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn defaults_are_the_documented_ones() {
    let q = Recall::new("x");
    assert_eq!(q.rrf_k, 60.0);
    assert_eq!(q.fusion, Fusion::Rrf);
    assert!(!q.time_tightness);
}

fn event(w: &World, who: &str, start: i64, end: i64) -> FactDraft {
    FactDraft::new(BodyDraft::Event {
        text: "release planned".into(),
        participants: vec![w.person(who)],
        when: TimeSpan::Range(Interval::new(ts(start), ts(end)).unwrap()),
    })
}

#[test]
fn rrf_k_changes_the_order() {
    let w = world();
    for i in 0..8 {
        w.clock.set(ts(200 + i));
        w.remember_one(statement(&format!("alpha filler{i}")));
    }
    // Best by meaning, but covers only one of the two query concepts.
    w.remember_one(statement("deadline"));
    w.remember_one(statement("deadline alpha"));
    let mut q = Recall::new("alpha deadline");
    q.channels = Channels {
        entities: 0.0,
        keywords: 1.0,
        meaning: 1.0,
    };
    let top = |q: &Recall| text(&w.memory.recall(q).hits[0]);
    q.rrf_k = 1.0;
    assert_eq!(top(&q), "deadline");
    q.rrf_k = 60.0;
    assert_eq!(top(&q), "deadline alpha");
}

#[test]
fn boost_drops_facts_found_only_by_keywords() {
    let w = world();
    // 101 facts closer in meaning push the keyword-only fact out of the
    // meaning channel's candidates.
    for i in 0..101 {
        w.clock.set(ts(200 + i));
        w.remember_one(statement("deadline"));
    }
    w.remember_one(statement("alpha beta"));
    let mut q = Recall::new("deadline alpha");
    q.limit = 200;
    let rrf = w.memory.recall(&q).hits;
    assert!(rrf.iter().any(|h| text(h) == "alpha beta"));
    q.fusion = Fusion::Boost;
    let boost = w.memory.recall(&q).hits;
    assert!(boost.iter().all(|h| text(h) != "alpha beta"));
    assert_eq!(boost.len(), 100);
}

#[test]
fn boost_reranks_meaning_candidates_by_lexical_hits() {
    let w = world();
    w.remember_one(statement("deadline beta"));
    w.clock.set(ts(300));
    w.remember_one(statement("deadline alpha"));
    let mut q = Recall::new("deadline alpha");
    q.fusion = Fusion::Boost;
    let hits = w.memory.recall(&q).hits;
    assert_eq!(text(&hits[0]), "deadline alpha");
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn boost_without_an_embedder_is_rrf() {
    let w = World::new();
    w.remember_one(statement("alpha beta"));
    let mut q = Recall::new("alpha");
    let rrf = w.memory.recall(&q);
    q.fusion = Fusion::Boost;
    assert_eq!(w.memory.recall(&q), rrf);
}

#[test]
fn time_tightness_prefers_the_tighter_fact() {
    let w = World::new();
    // The day-precise fact is recorded first, so by default the newer
    // year-wide one wins the tie.
    w.remember_one(event(&w, "Anna", 110_000, 111_000));
    w.clock.set(ts(200));
    w.remember_one(event(&w, "Bob", 0, 365_000));
    let mut q = Recall::new("release");
    q.window = Some(Interval::new(ts(100_000), ts(130_000)).unwrap());
    let anna = w.entity(w.person, "anna");
    let bob = w.entity(w.person, "bob");
    let hits = w.memory.recall(&q).hits;
    assert_eq!(hits[0].fact.entities(), [bob]);
    assert!(hits.iter().all(|h| h.why.in_window));
    q.time_tightness = true;
    let hits = w.memory.recall(&q).hits;
    assert_eq!(hits[0].fact.entities(), [anna]);
    assert!(hits[0].score > hits[1].score);
}
