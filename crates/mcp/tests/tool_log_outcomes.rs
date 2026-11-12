//! Every outcome the tools can report, read back by the tool log's parser
//! from the text the real mapping layer produces.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde_json::json;
use tessar_core::clock::ManualClock;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::log::MemLog;
use tessar_core::model::{Confidence, Key, Origin, Timestamp, Value};
use tessar_core::store::{
    BodyDraft, EntityRef, FactDraft, Memory, ObjectDraft, Remember, Semantic,
};
use tessar_mcp::api::Tessar;
use tessar_mcp::args::{RememberArgs, UpdateArgs};
use tessar_mcp::tool_log::{parse_facts, parse_update};
use tessar_mcp::when::{Calendar, DAY_MS, days_from_civil};

const DIMS: usize = 64;

struct Bag;

impl Bag {
    fn vector(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; DIMS];
        v[DIMS - 1] = 0.05;
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .map(str::to_lowercase)
            .filter(|w| !w.is_empty())
        {
            let mut h = DefaultHasher::new();
            word.hash(&mut h);
            v[(h.finish() % (DIMS as u64 - 1)) as usize] += 1.0;
        }
        v
    }
}

impl Embedder for Bag {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| Self::vector(t)).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(Self::vector(text))
    }
}

struct Fixture {
    api: Tessar,
    mem: Memory,
    clock: Arc<ManualClock>,
}

fn fixture() -> Fixture {
    let clock = Arc::new(ManualClock::new(Timestamp::from_millis(
        days_from_civil(2026, 11, 8) * DAY_MS + 12 * 3_600_000,
    )));
    let mem = Memory::open_with(
        MemLog::new(),
        clock.clone(),
        Semantic {
            embedder: Arc::new(Bag),
            dims: DIMS,
            near_bonus: 0.05,
            predicate_threshold: 0.70,
        },
    )
    .unwrap();
    let api = Tessar::bootstrap(mem.clone(), clock.clone(), "tessar", &|w| panic!("{w}"))
        .unwrap()
        .with_calendar(Calendar::utc());
    Fixture { api, mem, clock }
}

fn remember(f: &Fixture, v: serde_json::Value) -> String {
    f.api
        .remember(serde_json::from_value::<RememberArgs>(v).unwrap())
        .unwrap()
}

fn update(f: &Fixture, v: serde_json::Value) -> String {
    f.api
        .update(serde_json::from_value::<UpdateArgs>(v).unwrap())
        .unwrap()
}

fn lives_in(city: &str) -> serde_json::Value {
    json!({"space": "user", "facts": [{
        "kind": "attribute", "subject": {"key": "user", "kind": "person"},
        "predicate": "lives_in", "value": city, "text": format!("User lives in {city}")}]})
}

fn statement(text: &str) -> serde_json::Value {
    json!({"facts": [{"kind": "statement", "text": text}]})
}

/// The one fact a report of one fact names: handle and outcome.
fn one(text: &str) -> (Option<String>, String) {
    let facts = parse_facts(text, 1);
    (facts[0].id.clone(), facts[0].outcome.clone().unwrap())
}

fn handle(id: &str) -> Option<String> {
    Some(id.to_owned())
}

#[test]
fn remember_outcomes_round_trip() {
    let f = fixture();
    let out = remember(&f, lives_in("Boston"));
    let (id, outcome) = one(&out);
    assert_eq!(outcome, "created", "{out}");
    let created = id.unwrap();

    assert_eq!(
        one(&remember(&f, lives_in("Boston"))),
        (handle(&created), "unchanged".into())
    );

    f.clock.advance(DAY_MS);
    let out = remember(&f, lives_in("Denver"));
    let (id, outcome) = one(&out);
    assert_eq!(outcome, "superseded", "{out}");
    assert_ne!(id.unwrap(), created);

    // Several facts in one report, each read.
    let out = remember(
        &f,
        json!({"facts": [
            {"kind": "statement", "text": "alpha note"},
            {"kind": "statement", "text": "beta note"}]}),
    );
    let facts = parse_facts(&out, 2);
    assert!(
        facts
            .iter()
            .all(|x| x.outcome.as_deref() == Some("created") && x.id.is_some())
    );
}

/// An inferred value beside the asserted one is written past the MCP layer;
/// asserting the same value again keeps it ("kept").
#[test]
fn kept_round_trips() {
    let f = fixture();
    let (asserted, _) = one(&remember(&f, lives_in("Boston")));
    let space = f.mem.space_named("user").unwrap().id;
    let kind = f.mem.kind_named(space, "person").unwrap();
    let who = f.mem.find_entity(space, kind, "user").unwrap().id;
    let predicate = f.mem.predicate_named(space, "lives_in").unwrap().id;
    f.mem
        .remember(Remember {
            space,
            origin: Origin::Inferred {
                by: Key::new("model").unwrap(),
                confidence: Confidence::new(0.5).unwrap(),
            },
            episode: None,
            facts: vec![FactDraft::new(BodyDraft::Triple {
                subject: EntityRef::Id(who),
                predicate,
                object: ObjectDraft::Value(Value::Text("Denver".into())),
                text: None,
                when: None,
            })],
        })
        .unwrap();
    f.clock.advance(DAY_MS);
    assert_eq!(
        one(&remember(&f, lives_in("Boston"))),
        (asserted, "kept".to_owned())
    );
}

#[test]
fn update_outcomes_round_trip() {
    let f = fixture();
    let id = |out: &str| one(out).0.unwrap();

    // superseded
    let a = id(&remember(&f, statement("User prefers tea")));
    f.clock.advance(DAY_MS);
    let out = update(
        &f,
        json!({"fact": a, "reason": "changed", "text": "User prefers coffee"}),
    );
    let (outcome, new) = parse_update(&out);
    assert_eq!(outcome, "superseded", "{out}");
    let b = new.unwrap();
    assert_ne!(b, a);

    // unchanged
    let out = update(
        &f,
        json!({"fact": b, "reason": "changed", "text": "User prefers coffee"}),
    );
    assert_eq!(parse_update(&out), ("unchanged".to_owned(), None), "{out}");

    // closed
    let out = update(&f, json!({"fact": b, "reason": "changed"}));
    assert_eq!(parse_update(&out), ("closed".to_owned(), None), "{out}");

    // corrected, with and without a replacement
    let c = id(&remember(&f, statement("User owns a cat")));
    let out = update(&f, json!({"fact": c, "reason": "wrong"}));
    assert_eq!(parse_update(&out), ("corrected".to_owned(), None), "{out}");
    let d = id(&remember(&f, statement("User owns a dog")));
    let out = update(
        &f,
        json!({"fact": d, "reason": "wrong", "text": "User owns a fish"}),
    );
    let (outcome, new) = parse_update(&out);
    assert_eq!(outcome, "corrected", "{out}");
    assert!(new.is_some_and(|h| h != d), "{out}");

    // redacted
    let e = id(&remember(&f, statement("The token is hunter2")));
    let out = update(&f, json!({"fact": e, "reason": "redact"}));
    assert_eq!(parse_update(&out), ("redacted".to_owned(), None), "{out}");
}
