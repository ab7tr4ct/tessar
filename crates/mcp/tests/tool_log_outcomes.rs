//! Every outcome the write tools can report: the structured content has the
//! right handle and outcome and agrees with the text report.

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
use tessar_mcp::api::Rendered;
use tessar_mcp::api::Tessar;
use tessar_mcp::args::{LessonArgs, RememberArgs, UpdateArgs};
use tessar_mcp::when::{Calendar, DAY_MS, days_from_civil};
use tessar_mcp::written::WriteOutcome;

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

fn remember(f: &Fixture, v: serde_json::Value) -> Rendered {
    f.api
        .remember_rendered(serde_json::from_value::<RememberArgs>(v).unwrap())
        .unwrap()
}

fn update(f: &Fixture, v: serde_json::Value) -> Rendered {
    f.api
        .update_rendered(serde_json::from_value::<UpdateArgs>(v).unwrap())
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

/// The one fact of a result: its structured entry, after checking that the
/// text report starts the same way and that no word of `secret` leaked.
fn one(r: &Rendered, secret: &str) -> serde_json::Value {
    let facts = r.structured["facts"].as_array().unwrap();
    assert_eq!(facts.len(), 1, "{}", r.structured);
    let fact = facts[0].clone();
    let word = fact["outcome"].as_str().unwrap();
    assert!(
        r.text.contains(&format!("{word}: ")) || r.text.contains(&format!("{word} (")),
        "{word} not in {}",
        r.text
    );
    assert!(
        !r.structured.to_string().contains(secret),
        "{}",
        r.structured
    );
    fact
}

fn id(fact: &serde_json::Value) -> String {
    fact["id"].as_str().unwrap().to_owned()
}

#[test]
fn remember_outcomes() {
    let f = fixture();
    let r = remember(&f, lives_in("Boston"));
    assert_eq!(r.structured["space"], "user");
    let created = one(&r, "Boston");
    assert_eq!(created["outcome"], "created");
    assert!(r.text.contains(&id(&created)));

    let again = one(&remember(&f, lives_in("Boston")), "Boston");
    assert_eq!(again["outcome"], "unchanged");
    assert_eq!(again["id"], created["id"]);

    f.clock.advance(DAY_MS);
    let r = remember(&f, lives_in("Denver"));
    let next = one(&r, "Denver");
    assert_eq!(next["outcome"], "superseded");
    assert_eq!(next["previous"], created["id"]);
    assert_ne!(next["id"], created["id"]);
    assert!(r.text.contains(&id(&next)));

    // Several facts in one call, in call order.
    let r = remember(
        &f,
        json!({"facts": [
            {"kind": "statement", "text": "alpha note"},
            {"kind": "statement", "text": "beta note"}]}),
    );
    assert_eq!(r.structured["space"], "tessar");
    let facts = r.structured["facts"].as_array().unwrap();
    assert_eq!(facts.len(), 2);
    let (a, b) = (id(&facts[0]), id(&facts[1]));
    assert!(facts.iter().all(|x| x["outcome"] == "created"));
    assert!(
        r.text.find(&a).unwrap() < r.text.find(&b).unwrap(),
        "{}",
        r.text
    );
    assert!(!r.structured.to_string().contains("note"));
}

#[test]
fn a_lesson_has_the_same_shape() {
    let f = fixture();
    let r = f
        .api
        .lesson_rendered(
            serde_json::from_value::<LessonArgs>(
                json!({"text": "Before blaming the cache, read the log"}),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(r.structured["space"], "tessar");
    let fact = one(&r, "blaming");
    assert_eq!(fact["outcome"], "created");
    assert!(r.text.contains(&id(&fact)));
}

/// An inferred value beside the asserted one is written past the MCP layer;
/// asserting the same value again keeps it ("kept") and closes the other.
#[test]
fn kept_names_what_it_closed() {
    let f = fixture();
    let asserted = id(&one(&remember(&f, lives_in("Boston")), "Boston"));
    let space = f.mem.space_named("user").unwrap().id;
    let kind = f.mem.kind_named(space, "person").unwrap();
    let who = f.mem.find_entity(space, kind, "user").unwrap().id;
    let predicate = f.mem.predicate_named(space, "lives_in").unwrap().id;
    let done = f
        .mem
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
    let inferred = done.facts[0].fact.to_string();
    f.clock.advance(DAY_MS);
    let r = remember(&f, lives_in("Boston"));
    let kept = one(&r, "Denver");
    assert_eq!(kept["outcome"], "kept");
    assert_eq!(kept["id"], asserted.as_str());
    assert_eq!(kept["closed"], inferred.as_str());
    assert!(kept.get("closed_others").is_none());
    assert!(r.text.contains("kept (already known)"), "{}", r.text);
}

#[test]
fn update_outcomes() {
    let f = fixture();

    // superseded
    let a = id(&one(&remember(&f, statement("User prefers tea")), "tea"));
    f.clock.advance(DAY_MS);
    let r = update(
        &f,
        json!({"fact": a, "reason": "changed", "text": "User prefers coffee"}),
    );
    assert_eq!(r.structured["space"], "tessar");
    let sup = one(&r, "coffee");
    assert_eq!(sup["outcome"], "superseded");
    assert_eq!(sup["previous"], a.as_str());
    let b = id(&sup);
    assert_ne!(b, a);

    // unchanged
    let r = update(
        &f,
        json!({"fact": b, "reason": "changed", "text": "User prefers coffee"}),
    );
    let un = one(&r, "coffee");
    assert_eq!(
        (un["outcome"].as_str(), un["id"].as_str()),
        (Some("unchanged"), Some(b.as_str()))
    );

    // closed
    let r = update(&f, json!({"fact": b, "reason": "changed"}));
    let closed = one(&r, "coffee");
    assert_eq!(
        (closed["outcome"].as_str(), closed["id"].as_str()),
        (Some("closed"), Some(b.as_str()))
    );

    // corrected, without and with a replacement
    let c = id(&one(&remember(&f, statement("User owns a cat")), "cat"));
    let corrected = one(&update(&f, json!({"fact": c, "reason": "wrong"})), "cat");
    assert_eq!(corrected["outcome"], "corrected");
    assert_eq!(corrected["id"], c.as_str());
    assert!(corrected.get("previous").is_none());
    let d = id(&one(&remember(&f, statement("User owns a dog")), "dog"));
    let r = update(
        &f,
        json!({"fact": d, "reason": "wrong", "text": "User owns a fish"}),
    );
    let fixed = one(&r, "fish");
    assert_eq!(fixed["outcome"], "corrected");
    assert_eq!(fixed["previous"], d.as_str());
    assert_ne!(fixed["id"], d.as_str());
    assert!(r.text.contains(&id(&fixed)));

    // redacted
    let e = id(&one(
        &remember(&f, statement("The token is hunter2")),
        "hunter2",
    ));
    let r = update(&f, json!({"fact": e, "reason": "redact"}));
    let red = one(&r, "hunter2");
    assert_eq!(
        (red["outcome"].as_str(), red["id"].as_str()),
        (Some("redacted"), Some(e.as_str()))
    );
    assert_eq!(r.structured["space"], "tessar");
}

#[test]
fn outcome_words_are_stable() {
    // The enum is the only place the words live; the reports use `word()`.
    let words: Vec<&str> = WriteOutcome::ALL.iter().map(|o| o.word()).collect();
    assert_eq!(
        words,
        [
            "created",
            "conflicting",
            "unchanged",
            "kept",
            "closed",
            "superseded",
            "corrected",
            "redacted"
        ]
    );
}
