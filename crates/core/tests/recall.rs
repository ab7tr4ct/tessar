mod common;

use common::*;
use tessar_core::model::{Fact, FactBody, Interval, SpaceKind, TimeSpan, Value};
use tessar_core::store::{BodyDraft, EntityRef, FactDraft, Hit, ObjectDraft, Recall, View};

fn statement(text: &str, mentions: Vec<EntityRef>) -> FactDraft {
    FactDraft {
        mentions,
        ..FactDraft::new(BodyDraft::Statement {
            text: text.into(),
            when: None,
        })
    }
}

fn event(w: &World, text: &str, who: &str, start: i64, end: i64) -> FactDraft {
    FactDraft::new(BodyDraft::Event {
        text: text.into(),
        participants: vec![w.person(who)],
        when: TimeSpan::Range(Interval::new(ts(start), ts(end)).unwrap()),
    })
}

fn text(fact: &Fact) -> String {
    match &fact.body {
        FactBody::Statement { text, .. } | FactBody::Event { text, .. } => text.clone(),
        FactBody::Triple {
            object: Value::Text(t),
            ..
        } => t.clone(),
        other => format!("{other:?}"),
    }
}

fn texts(hits: &[Hit]) -> Vec<String> {
    hits.iter().map(|h| text(&h.fact)).collect()
}

fn recall(w: &World, query: &str) -> Vec<Hit> {
    w.memory.recall(&Recall::new(query)).hits
}

#[test]
fn finds_facts_about_an_entity_named_in_the_query() {
    let w = World::new();
    w.remember(vec![w.goal("memory engine"), w.knows("Anna", "Rust")]);
    let tessar = w.entity(w.project, "tessar");

    let result = w.memory.recall(&Recall::new("Что сейчас с tessar?"));
    assert_eq!(result.entities, [tessar]);
    assert_eq!(texts(&result.hits), ["memory engine"]);
    let why = &result.hits[0].why;
    assert_eq!(why.entities, [tessar]);
    assert_eq!(why.entity_rank, Some(1));
}

#[test]
fn entity_names_match_aliases_case_and_punctuation() {
    let w = World::new();
    w.remember_one(w.knows("Anna", "Rust"));
    let anna = w.entity(w.person, "anna");
    w.memory.add_alias(anna, key("my sister")).unwrap();

    let result = w.memory.recall(&Recall::new("What does «My Sister» know?"));
    assert_eq!(result.entities, [anna]);
    assert_eq!(texts(&result.hits), ["Rust"]);
}

#[test]
fn keys_with_inner_punctuation_are_found() {
    let w = World::new();
    w.remember_one(statement(
        "HGVL was the first attempt",
        vec![w.project("cortex-db")],
    ));
    let cortex = w.entity(w.project, "cortex-db");
    let result = w.memory.recall(&Recall::new("а что было в cortex-db?"));
    assert_eq!(result.entities, [cortex]);
    assert_eq!(texts(&result.hits), ["HGVL was the first attempt"]);
}

#[test]
fn finds_by_words_when_no_entity_is_named() {
    let w = World::new();
    w.remember(vec![
        statement("Решили не брать MongoDB", vec![]),
        statement("Журнал пишется в JSON Lines", vec![]),
    ]);
    let hits = recall(&w, "почему НЕ mongodb?");
    assert_eq!(texts(&hits), ["Решили не брать MongoDB"]);
    let why = &hits[0].why;
    assert_eq!(why.words, ["не", "mongodb"]);
    assert_eq!(why.keyword_rank, Some(1));
    assert_eq!(why.entity_rank, None);
}

#[test]
fn rare_words_outrank_common_ones() {
    let w = World::new();
    w.remember(vec![
        statement("память агента и журнал", vec![]),
        statement("память агента и индексы", vec![]),
        statement("память агента и векторы", vec![]),
    ]);
    // «память» is everywhere, «журнал» only in one fact.
    assert_eq!(
        recall(&w, "память журнал")[0].why.words,
        ["память", "журнал"]
    );
}

#[test]
fn a_fact_found_by_both_channels_comes_first() {
    let w = World::new();
    w.remember(vec![
        w.goal("memory engine"),
        statement("журнал на диске", vec![]),
        statement("журнал хранит историю", vec![w.project("tessar")]),
    ]);
    let hits = recall(&w, "журнал tessar");
    assert!(hits[0].why.entity_rank.is_some());
    assert!(hits[0].why.keyword_rank.is_some());
    assert_eq!(text(&hits[0].fact), "журнал хранит историю");
    assert_eq!(hits.len(), 3);
}

#[test]
fn replaced_values_are_hidden_unless_asked_for_the_past() {
    let w = World::new();
    w.remember_one(w.goal("graph database"));
    w.clock.set(ts(200));
    w.remember_one(w.goal("memory engine"));

    assert_eq!(texts(&recall(&w, "tessar goal")), ["memory engine"]);
    let mut past = Recall::new("tessar goal");
    past.view = View::TrueAt(ts(150));
    assert_eq!(texts(&w.memory.recall(&past).hits), ["graph database"]);
}

#[test]
fn spaces_limit_the_search() {
    let w = World::new();
    w.remember_one(w.goal("memory engine"));
    let other = w
        .memory
        .create_space(key("cortex-db"), SpaceKind::LongTerm, None)
        .unwrap();
    let project = w.memory.define_kind(other, name("project")).unwrap();
    w.memory
        .remember(tessar_core::store::Remember {
            space: other,
            origin: origin(),
            episode: None,
            facts: vec![statement(
                "tessar grew out of HGVL",
                vec![EntityRef::Key {
                    kind: project,
                    key: key("tessar"),
                }],
            )],
        })
        .unwrap();

    assert_eq!(recall(&w, "tessar").len(), 2);
    let mut here = Recall::new("tessar");
    here.spaces = vec![w.space];
    let result = w.memory.recall(&here);
    assert_eq!(texts(&result.hits), ["memory engine"]);
    // Query entities are not filtered by space, only facts are.
    assert_eq!(
        result.entities,
        [
            w.entity(w.project, "tessar"),
            w.memory.find_entity(other, project, "tessar").unwrap().id
        ]
    );
}

#[test]
fn facts_inside_the_window_get_a_bonus() {
    let w = World::new();
    // Recorded first, but happens in March.
    w.remember_one(event(&w, "release planned", "Anna", 3_000, 4_000));
    w.clock.set(ts(200));
    // Recorded later, happens in January.
    w.remember_one(event(&w, "release planned", "Bob", 1_000, 2_000));

    let plain = recall(&w, "release");
    assert_eq!(plain[0].fact.entities(), [w.entity(w.person, "bob")]);
    assert!(!plain[0].why.in_window);

    let mut march = Recall::new("release");
    march.window = Some(Interval::new(ts(3_000), ts(4_000)).unwrap());
    let hits = w.memory.recall(&march).hits;
    assert_eq!(hits[0].fact.entities(), [w.entity(w.person, "anna")]);
    assert!(hits[0].why.in_window);
    assert!(!hits[1].why.in_window);
}

#[test]
fn explicit_entities_need_no_text() {
    let w = World::new();
    w.remember(vec![w.knows("Anna", "Rust"), w.knows("Bob", "Go")]);
    let anna = w.entity(w.person, "anna");
    let mut query = Recall::new("");
    query.entities = vec![anna];
    assert_eq!(texts(&w.memory.recall(&query).hits), ["Rust"]);
}

#[test]
fn limit_and_empty_queries() {
    let w = World::new();
    w.remember(vec![w.knows("Anna", "Rust"), w.knows("Anna", "Go")]);
    assert!(recall(&w, "").is_empty());
    assert!(recall(&w, "nothing matches this").is_empty());
    let mut one = Recall::new("anna");
    one.limit = 1;
    assert_eq!(w.memory.recall(&one).hits.len(), 1);
}

#[test]
fn results_are_deterministic_and_survive_replay() {
    let w = World::new();
    w.remember(vec![
        w.goal("memory engine"),
        w.knows("Anna", "Rust"),
        statement("журнал хранит историю", vec![w.project("tessar")]),
        statement("Rust без сборщика мусора", vec![]),
    ]);
    let query = Recall::new("rust tessar журнал anna");
    let first = w.memory.recall(&query);
    assert_eq!(w.memory.recall(&query), first);
    assert_eq!(w.reopen().recall(&query), first);
    assert_eq!(first.hits.len(), 4);
}

#[test]
fn coverage_counts_each_concept_once() {
    let w = World::new();
    w.remember(vec![
        w.goal("memory engine"),
        statement("журнал хранит историю", vec![w.project("tessar")]),
        // Names tessar in its text, without a link.
        statement("tessar ведёт журнал", vec![]),
    ]);
    let hits = recall(&w, "журнал tessar");
    assert!(hits.iter().all(|h| h.why.concepts == 2));
    let find = |t: &str| hits.iter().find(|h| text(&h.fact) == t).unwrap();
    let goal = find("memory engine");
    let unlinked = find("tessar ведёт журнал");
    // The goal matches tessar twice — by link and by word — but it is one
    // concept out of two.
    assert_eq!(goal.why.covered, 1);
    assert_eq!(unlinked.why.covered, 2);
    assert!(goal.score < unlinked.score);
    assert_eq!(find("журнал хранит историю").why.covered, 2);
}

#[test]
fn a_query_entity_of_another_space_finds_facts_of_the_project() {
    let w = World::new();
    let project = w.project_space("cortex-db");
    w.remember_in(
        project,
        vec![FactDraft::new(BodyDraft::Triple {
            subject: w.user(),
            predicate: w.knows,
            object: ObjectDraft::Value(Value::Text("Rust".into())),
            text: None,
        })],
    );
    let user = w
        .memory
        .find_entity(w.home, w.home_person, "user")
        .unwrap()
        .id;

    let mut query = Recall::new("what does user know");
    query.spaces = vec![project];
    let result = w.memory.recall(&query);
    assert!(result.entities.contains(&user));
    assert_eq!(texts(&result.hits), ["Rust"]);

    // An explicit entity of another space works as well.
    let mut query = Recall::new("anything");
    query.spaces = vec![project];
    query.entities = vec![user];
    let result = w.memory.recall(&query);
    assert_eq!(result.entities, [user]);
    assert_eq!(texts(&result.hits), ["Rust"]);
}
