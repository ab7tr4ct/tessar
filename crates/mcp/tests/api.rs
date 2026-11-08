//! The mapping layer over an in-memory log and a fake embedder: no model,
//! no GPU.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde_json::json;
use tessar_core::clock::ManualClock;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::log::MemLog;
use tessar_core::model::Timestamp;
use tessar_core::store::{Memory, Semantic};
use tessar_mcp::api::Tessar;
use tessar_mcp::args::{AboutArgs, RecallArgs, RememberArgs};
use tessar_mcp::when::{DAY_MS, days_from_civil};

const DIMS: usize = 64;
const STOP: [&str; 9] = ["the", "a", "an", "of", "in", "to", "is", "where", "subject"];

/// Bag of words hashed into 64 buckets: the same words, the same vector.
struct Bag;

impl Bag {
    fn vector(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; DIMS];
        v[DIMS - 1] = 0.05;
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .map(str::to_lowercase)
            .filter(|w| !w.is_empty() && !STOP.contains(&w.as_str()))
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
    let api = Tessar::bootstrap(mem.clone(), clock.clone(), "tessar", &|w| panic!("{w}")).unwrap();
    Fixture { api, mem, clock }
}

fn remember(f: &Fixture, v: serde_json::Value) -> Result<String, String> {
    f.api
        .remember(serde_json::from_value::<RememberArgs>(v).unwrap())
}

fn recall(f: &Fixture, v: serde_json::Value) -> Result<String, String> {
    f.api
        .recall(serde_json::from_value::<RecallArgs>(v).unwrap())
}

fn about(f: &Fixture, v: serde_json::Value) -> Result<String, String> {
    f.api.about(serde_json::from_value::<AboutArgs>(v).unwrap())
}

fn user() -> serde_json::Value {
    json!({"key": "user", "kind": "person"})
}

#[test]
fn bootstrap_is_idempotent() {
    let f = fixture();
    let before = f.api.vocabulary();
    let again =
        Tessar::bootstrap(f.mem.clone(), f.clock.clone(), "tessar", &|w| panic!("{w}")).unwrap();
    assert_eq!(again.vocabulary(), before);
    assert!(f.mem.space_named("user").is_some());
    assert!(f.mem.space_named("tessar").is_some());
}

#[test]
fn statement_round_trip() {
    let f = fixture();
    let out = remember(
        &f,
        json!({"facts": [{"kind": "statement", "text": "The journal uses JSON lines with a checksum"}],
               "source": "chat about the journal"}),
    )
    .unwrap();
    assert!(out.contains("created"), "{out}");
    let found = recall(&f, json!({"query": "checksum"})).unwrap();
    assert!(found.contains("JSON lines with a checksum"), "{found}");
    assert!(found.contains("[tessar]"), "{found}");
    assert!(found.contains("found by:"), "{found}");
}

#[test]
fn attribute_round_trip_finds_text_by_word() {
    let f = fixture();
    remember(
        &f,
        json!({"space": "user", "facts": [{
            "kind": "attribute", "subject": user(), "predicate": "lives_in",
            "value": "Lisbon", "text": "User lives in Lisbon"}]}),
    )
    .unwrap();
    let found = recall(&f, json!({"query": "Lisbon", "space": "user"})).unwrap();
    assert!(found.contains("User lives in Lisbon"), "{found}");
    assert!(found.contains("[user]"), "{found}");
    // Without text the triple renders as "subject predicate: value".
    remember(
        &f,
        json!({"space": "user", "facts": [{
            "kind": "attribute", "subject": user(), "predicate": "occupation", "value": "pilot"}]}),
    )
    .unwrap();
    let found = recall(&f, json!({"query": "pilot"})).unwrap();
    assert!(found.contains("user occupation: pilot"), "{found}");
}

#[test]
fn single_valued_write_reports_superseded() {
    let f = fixture();
    let write = |city: &str| {
        remember(
            &f,
            json!({"space": "user", "facts": [{
                "kind": "attribute", "subject": user(), "predicate": "lives_in", "value": city,
                "text": format!("User lives in {city}")}]}),
        )
        .unwrap()
    };
    assert!(write("Boston").contains("created"));
    f.clock.advance(DAY_MS);
    let out = write("Denver");
    assert!(
        out.contains("superseded (was: Boston, recorded 2026-11-08)"),
        "{out}"
    );
    assert!(write("Denver").contains("unchanged"));

    let found = recall(&f, json!({"query": "where does the user live"})).unwrap();
    assert!(
        found.contains("User lives in Denver [current; earlier: Boston (2026-11-08)]"),
        "{found}"
    );
    assert!(!found.contains("User lives in Boston ["), "{found}");

    let hist = about(&f, json!({"entity": user(), "history": true})).unwrap();
    assert!(
        hist.contains("User lives in Boston") && hist.contains("User lives in Denver"),
        "{hist}"
    );
    assert!(hist.contains("[version 1, until 2026-11-09]"), "{hist}");
}

#[test]
fn as_of_shows_the_older_value() {
    let f = fixture();
    let write = |city: &str| {
        remember(
            &f,
            json!({"space": "user", "facts": [{
                "kind": "attribute", "subject": user(), "predicate": "lives_in", "value": city,
                "text": format!("User lives in {city}")}]}),
        )
        .unwrap()
    };
    write("Boston");
    f.clock.advance(3 * DAY_MS);
    write("Denver");
    let then = recall(&f, json!({"query": "User lives", "as_of": "2026-11-09"})).unwrap();
    assert!(then.contains("User lives in Boston"), "{then}");
    assert!(
        then.contains("[true on 2026-11-09; now: Denver; earlier:")
            || then.contains("[true on 2026-11-09; now: Denver]"),
        "{then}"
    );
    assert!(!then.contains("User lives in Denver"), "{then}");
    assert!(recall(&f, json!({"query": "x", "as_of": "yesterday"})).is_err());
}

#[test]
fn user_resolves_in_user_space_from_a_project_write() {
    let f = fixture();
    remember(
        &f,
        json!({"facts": [{
            "kind": "attribute", "subject": user(), "predicate": "prefers",
            "value": "tabs", "text": "User prefers tabs in this repository"}]}),
    )
    .unwrap();
    let user_space = f.mem.space_named("user").unwrap().id;
    let project_space = f.mem.space_named("tessar").unwrap().id;
    let person_user = f.mem.kind_named(user_space, "person").unwrap();
    let person_project = f.mem.kind_named(project_space, "person").unwrap();
    assert!(f.mem.find_entity(user_space, person_user, "user").is_some());
    assert!(
        f.mem
            .find_entity(project_space, person_project, "user")
            .is_none()
    );
    let list = about(&f, json!({"entity": user()})).unwrap();
    assert!(
        list.contains("User prefers tabs in this repository"),
        "{list}"
    );
    assert!(list.contains("in space `user`"), "{list}");
}

#[test]
fn new_predicate_with_description_lands_in_the_user_space() {
    let f = fixture();
    let out = remember(
        &f,
        json!({"facts": [{
            "kind": "attribute", "subject": {"key": "tessar", "kind": "project"},
            "predicate": "espresso_machine", "single": true,
            "predicate_description": "coffee brewing gadget model somebody operates",
            "value": "Rancilio", "text": "tessar espresso machine is a Rancilio"}]}),
    )
    .unwrap();
    assert!(out.contains("created"), "{out}");
    let user_space = f.mem.space_named("user").unwrap().id;
    let def = f
        .mem
        .predicate_named(user_space, "espresso_machine")
        .unwrap();
    assert_eq!(
        def.description,
        "coffee brewing gadget model somebody operates"
    );
    assert!(
        f.api
            .vocabulary()
            .contains("espresso_machine (single): coffee brewing")
    );
}

#[test]
fn new_predicate_without_description_is_refused() {
    let f = fixture();
    let err = remember(
        &f,
        json!({"facts": [{
            "kind": "attribute", "subject": user(), "predicate": "shoe_size", "value": "44"}]}),
    )
    .unwrap_err();
    assert!(err.contains("predicate_description"), "{err}");
    assert!(err.starts_with("Nothing was written"), "{err}");
}

#[test]
fn similar_predicate_is_a_tool_error_and_writes_no_facts() {
    let f = fixture();
    let err = remember(
        &f,
        json!({"facts": [
            {"kind": "statement", "text": "Should not be written"},
            {"kind": "attribute", "subject": user(), "predicate": "lives_place", "value": "Oslo",
             "predicate_description": "the place where the subject lives"}]}),
    )
    .unwrap_err();
    assert!(err.contains("fact 2"), "{err}");
    assert!(
        err.contains("predicate `lives_place` looks like `lives_in`"),
        "{err}"
    );
    assert!(
        err.contains("synonym_of: \"lives_in\"") && err.contains("force_new_predicate: true"),
        "{err}"
    );
    assert!(
        f.mem
            .recall(&tessar_core::store::Recall::new("written"))
            .hits
            .is_empty()
    );
    assert!(
        f.mem
            .predicate_named(f.mem.space_named("user").unwrap().id, "lives_place")
            .is_none()
    );
}

#[test]
fn synonym_of_declares_a_synonym() {
    let f = fixture();
    let out = remember(
        &f,
        json!({"space": "user", "facts": [{
            "kind": "attribute", "subject": user(), "predicate": "home_city", "value": "Oslo",
            "synonym_of": "lives_in", "predicate_description": "where the subject lives",
            "text": "User lives in Oslo"}]}),
    )
    .unwrap();
    assert!(out.contains("created"), "{out}");
    let vocab = f.api.vocabulary();
    assert!(
        vocab.contains("lives_in (single)") && vocab.contains("synonyms: home_city"),
        "{vocab}"
    );
    assert!(
        !vocab.lines().any(|l| l.starts_with("home_city")),
        "{vocab}"
    );
    // The synonym replaces the value written under the canonical name.
    f.clock.advance(1);
    let out = remember(
        &f,
        json!({"space": "user", "facts": [{
            "kind": "attribute", "subject": user(), "predicate": "lives_in", "value": "Rome",
            "text": "User lives in Rome"}]}),
    )
    .unwrap();
    assert!(out.contains("superseded (was: Oslo"), "{out}");
}

#[test]
fn force_new_predicate_defines_it_anyway() {
    let f = fixture();
    let out = remember(
        &f,
        json!({"space": "user", "facts": [{
            "kind": "attribute", "subject": user(), "predicate": "home_city", "value": "Oslo",
            "predicate_description": "the place where the subject lives",
            "force_new_predicate": true, "text": "User lives in Oslo"}]}),
    )
    .unwrap();
    assert!(out.contains("created"), "{out}");
    assert!(f.api.vocabulary().contains("home_city (many)"));
}

#[test]
fn non_universal_and_malformed_predicates_explain_themselves() {
    let f = fixture();
    let attr = |p: &str| {
        json!({"facts": [{"kind": "attribute", "subject": user(), "predicate": p, "value": "x",
                          "predicate_description": "something odd about the subject"}]})
    };
    let err = remember(&f, attr("plans_trip_to_2027")).unwrap_err();
    assert!(err.contains("not universal"), "{err}");
    let err = remember(&f, attr("Plans Trip")).unwrap_err();
    assert!(err.contains("snake_case"), "{err}");
    let err = remember(&f, attr("one_two_three_four_five")).unwrap_err();
    assert!(err.contains("more than"), "{err}");
}

#[test]
fn vocabulary_lists_the_starter_set() {
    let f = fixture();
    let vocab = f.api.vocabulary();
    let lines: Vec<&str> = vocab.lines().collect();
    assert_eq!(lines.len(), tessar_core::vocab::starter().len());
    assert_eq!(
        lines[0],
        "lives_in (single): the place where the subject lives"
    );
    assert!(lines.contains(&"owns (many): a thing the subject owns"));
    assert_eq!(f.api.vocabulary_summary(), vocab);
}

#[test]
fn about_lists_facts_across_spaces() {
    let f = fixture();
    remember(
        &f,
        json!({"space": "user", "facts": [{
            "kind": "statement", "text": "Anna is the user's sister",
            "mentions": [{"key": "Anna", "kind": "person"}]}]}),
    )
    .unwrap();
    remember(
        &f,
        json!({"facts": [{
            "kind": "event", "text": "Anna reviews the tessar design", "when": "2026-11-20",
            "mentions": [{"key": "Anna", "kind": "person"}]}]}),
    )
    .unwrap();
    let list = about(
        &f,
        json!({"entity": {"key": "anna", "kind": "person", "space": "user"}}),
    )
    .unwrap();
    assert!(list.contains("Anna is the user's sister"), "{list}");
    assert!(list.contains("Anna reviews the tessar design"), "{list}");
    assert!(
        list.contains("[tessar]") && list.contains("[user]"),
        "{list}"
    );
    assert!(list.contains("(future: 2026-11-20)"), "{list}");
    assert!(
        about(&f, json!({"entity": {"key": "Nobody", "kind": "person"}}))
            .unwrap()
            .contains("No person named")
    );
    assert!(about(&f, json!({"entity": {"key": "x", "kind": "robot"}})).is_err());
}

#[test]
fn events_need_a_valid_when_and_get_a_tense_tag() {
    let f = fixture();
    let err = remember(&f, json!({"facts": [{"kind": "event", "text": "Launch"}]})).unwrap_err();
    assert!(err.contains("`when`"), "{err}");
    let err = remember(
        &f,
        json!({"facts": [{"kind": "event", "text": "Launch", "when": "2026-02-30"}]}),
    )
    .unwrap_err();
    assert!(err.contains("no such day"), "{err}");
    remember(
        &f,
        json!({"facts": [
            {"kind": "event", "text": "Beta launch happened", "when": "2026-10-01"},
            {"kind": "event", "text": "Beta review week", "when": "2026-11-05/2026-11-12"},
            {"kind": "event", "text": "Public launch is planned", "when": "2027-03"}]}),
    )
    .unwrap();
    let found = recall(&f, json!({"query": "launch review beta", "limit": 5})).unwrap();
    assert!(
        found.contains("Beta launch happened (past: 2026-10-01)"),
        "{found}"
    );
    assert!(
        found.contains("Beta review week (now: 2026-11-05 to 2026-11-12)"),
        "{found}"
    );
    assert!(
        found.contains("Public launch is planned (future: 2027-03-01 to 2027-03-31)"),
        "{found}"
    );
}

#[test]
fn duplicate_pair_in_one_call_is_reported_with_a_hint() {
    let f = fixture();
    let one = |v: &str| json!({"kind": "attribute", "subject": user(), "predicate": "lives_in", "value": v});
    let err = remember(&f, json!({"space": "user", "facts": [one("A"), one("B")]})).unwrap_err();
    assert!(
        err.contains("two values of the single-valued predicate `lives_in`"),
        "{err}"
    );
}

#[test]
fn bad_input_is_a_tool_error() {
    let f = fixture();
    assert!(remember(&f, json!({"facts": []})).is_err());
    assert!(remember(&f, json!({"facts": [{"kind": "statement"}]})).is_err());
    assert!(remember(
        &f,
        json!({"facts": [{"kind": "statement", "text": "x", "mentions": [{"key": "a", "kind": "alien"}]}]})
    )
    .unwrap_err()
    .contains("person, place"));
    assert!(recall(&f, json!({"query": "  "})).is_err());
    let none = recall(&f, json!({"query": "anything"})).unwrap();
    assert_eq!(none, "Nothing found.");
}

#[test]
fn tools_carry_descriptions_and_schemas() {
    use rmcp::ServerHandler;
    let server = tessar_mcp::server::TessarServer::new(fixture().api);
    for name in ["remember", "recall", "vocabulary", "about"] {
        let tool = server
            .get_tool(name)
            .unwrap_or_else(|| panic!("no tool {name}"));
        assert!(
            tool.description.as_deref().is_some_and(|d| d.len() > 100),
            "{name}"
        );
        if std::env::var_os("TESSAR_PRINT_TOOLS").is_some() {
            println!(
                "=== {name}\n{}\n{}",
                tool.description.as_deref().unwrap(),
                serde_json::to_string_pretty(&*tool.input_schema).unwrap()
            );
        }
    }
    let remember = server.get_tool("remember").unwrap();
    let description = remember.description.unwrap();
    assert!(description.contains("lives_in (single): the place where the subject lives"));
    assert!(description.contains("`vocabulary` for the live list"));
    let schema =
        serde_json::to_string(&*server.get_tool("remember").unwrap().input_schema).unwrap();
    for field in [
        "synonym_of",
        "force_new_predicate",
        "predicate_description",
        "mentions",
        "source",
    ] {
        assert!(schema.contains(field), "{field}");
    }
    let info = server.get_info();
    assert_eq!(info.server_info.name, "tessar");
    assert!(info.capabilities.tools.is_some());
}

#[test]
fn a_failed_call_defines_no_predicates() {
    let f = fixture();
    let err = remember(
        &f,
        json!({"facts": [
            {"kind": "attribute", "subject": user(), "predicate": "espresso_machine",
             "predicate_description": "coffee brewing gadget model somebody operates", "value": "Rancilio"},
            {"kind": "statement"}]}),
    )
    .unwrap_err();
    assert!(
        err.starts_with("Nothing was written, no predicates either"),
        "{err}"
    );
    let user_space = f.mem.space_named("user").unwrap().id;
    assert!(
        f.mem
            .predicate_named(user_space, "espresso_machine")
            .is_none()
    );
}

#[test]
fn a_similar_predicate_after_a_good_one_reports_what_was_kept() {
    let f = fixture();
    let err = remember(
        &f,
        json!({"facts": [
            {"kind": "attribute", "subject": user(), "predicate": "espresso_machine",
             "predicate_description": "coffee brewing gadget model somebody operates", "value": "Rancilio"},
            {"kind": "attribute", "subject": user(), "predicate": "lives_place", "value": "Oslo",
             "predicate_description": "the place where the subject lives"}]}),
    )
    .unwrap_err();
    assert!(err.starts_with("No facts were written"), "{err}");
    assert!(
        err.contains("Predicates defined and kept: `espresso_machine`"),
        "{err}"
    );
    assert!(
        f.mem
            .recall(&tessar_core::store::Recall::new("Rancilio"))
            .hits
            .is_empty()
    );
}

#[test]
fn one_new_predicate_in_two_facts_is_defined_once() {
    let f = fixture();
    let fact = |v: &str| {
        json!({"kind": "attribute", "subject": user(), "predicate": "espresso_machine",
        "predicate_description": "coffee brewing gadget model somebody operates", "value": v})
    };
    let out = remember(&f, json!({"facts": [fact("Rancilio"), fact("Gaggia")]})).unwrap();
    assert!(out.matches("created").count() == 2, "{out}");
}

#[test]
fn when_on_an_attribute_is_when_it_happens_not_when_it_holds() {
    let f = fixture();
    remember(
        &f,
        json!({"space": "user", "facts": [{
            "kind": "attribute", "subject": user(), "predicate": "lives_in", "value": "Lisbon",
            "when": "2099-01-01", "text": "User lives in Lisbon"}]}),
    )
    .unwrap();
    // The value holds now, even though the thing it describes is in the future.
    let found = recall(&f, json!({"query": "Lisbon"})).unwrap();
    assert!(found.contains("User lives in Lisbon"), "{found}");
    assert!(found.contains("(future: 2099-01-01)"), "{found}");
    assert!(!found.contains("no longer current"), "{found}");
}

#[test]
fn an_existing_entity_is_found_before_a_new_one_is_made() {
    let f = fixture();
    remember(
        &f,
        json!({"space": "user", "facts": [{"kind": "statement", "text": "Anna is the user's sister",
            "mentions": [{"key": "Anna", "kind": "person"}]}]}),
    )
    .unwrap();
    remember(
        &f,
        json!({"facts": [{"kind": "statement", "text": "Anna reviews the design",
            "mentions": [{"key": "Anna", "kind": "person"}]}]}),
    )
    .unwrap();
    let project_space = f.mem.space_named("tessar").unwrap().id;
    let person = f.mem.kind_named(project_space, "person").unwrap();
    assert!(f.mem.find_entity(project_space, person, "Anna").is_none());
    let list = about(&f, json!({"entity": {"key": "Anna", "kind": "person"}})).unwrap();
    assert!(
        list.contains("sister") && list.contains("reviews the design"),
        "{list}"
    );
}
