//! Redaction: content removed for good, from every read and every index,
//! and purging it from the log.

mod common;

use std::sync::Arc;

use common::*;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::log::Commit;
use tessar_core::model::{Ending, FactBody, FactId};
use tessar_core::store::{
    BodyDraft, Channels, EpisodeDraft, FactDraft, Memory, Op, Outcome, Recall, Remember, Semantic,
    StoreError, View, purge, purge_with_report,
};

const DIMS: usize = 8;

/// Words land on dimensions by their letters; every text gets a little
/// of the last dimension so no vector is zero.
struct Hashed;

impl Hashed {
    fn vector(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; DIMS];
        v[DIMS - 1] = 0.1;
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
        {
            let slot = word.to_lowercase().bytes().map(usize::from).sum::<usize>() % (DIMS - 1);
            v[slot] += 1.0;
        }
        v
    }
}

impl Embedder for Hashed {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| Self::vector(t)).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(Self::vector(text))
    }
}

fn semantic() -> Semantic {
    Semantic {
        embedder: Arc::new(Hashed),
        dims: DIMS,
        near_bonus: 0.05,
        predicate_threshold: 1.0,
    }
}

fn statement(text: &str) -> FactDraft {
    FactDraft::new(BodyDraft::Statement {
        text: text.into(),
        when: None,
    })
}

fn only_meaning() -> Channels {
    Channels {
        entities: 0.0,
        keywords: 0.0,
        meaning: 1.0,
    }
}

fn found(w: &World, query: &str) -> Vec<FactId> {
    w.memory
        .recall(&Recall::new(query))
        .hits
        .iter()
        .map(|h| h.fact.id)
        .collect()
}

#[test]
fn a_redacted_statement_is_gone_from_every_channel_and_view() {
    let w = World::with(Some(semantic()));
    let anna = w.remember_one(w.knows("Anna", "Rust"));
    let secret = w.remember_one(FactDraft {
        mentions: vec![w.person("Anna")],
        ..statement("the vault code is hunter2")
    });
    let other = w.remember_one(statement("a plain note about berlin"));
    let anna_id = w.entity(w.person, "anna");
    let secret_fact = w.fact(secret);
    assert_eq!(secret_fact.mentions, vec![anna_id]);

    assert!(found(&w, "hunter2").contains(&secret.fact));
    assert!(found(&w, "Anna").contains(&secret.fact));
    let meaning = Recall {
        channels: only_meaning(),
        ..Recall::new("vault code hunter2")
    };
    assert_eq!(w.memory.recall(&meaning).hits[0].fact.id, secret.fact);

    w.clock.advance(10);
    w.memory.redact(secret.fact).unwrap();

    // Recall: by words, by entity, by meaning.
    assert!(!found(&w, "hunter2").contains(&secret.fact));
    assert!(!found(&w, "vault code").contains(&secret.fact));
    assert!(!found(&w, "Anna").contains(&secret.fact));
    assert!(found(&w, "Anna").contains(&anna.fact));
    let hits = w.memory.recall(&meaning).hits;
    assert!(hits.iter().all(|h| h.fact.id != secret.fact));
    assert!(hits.iter().any(|h| h.fact.id == other.fact));
    // Every view of the entity.
    for view in [View::Current, View::TrueAt(ts(105)), View::KnownAt(ts(105))] {
        assert!(
            w.memory
                .facts_about(anna_id, view)
                .iter()
                .all(|f| f.id != secret.fact),
            "{view:?}"
        );
    }
    assert!(w.memory.current(secret_fact.lineage).is_none());

    // The record stays, empty.
    let f = w.memory.fact(secret.fact).unwrap();
    assert_eq!(f.body, FactBody::Redacted);
    assert!(f.mentions.is_empty());
    assert_eq!(f.episode, None);
    assert_eq!(f.id, secret_fact.id);
    assert_eq!(f.space, secret_fact.space);
    assert_eq!(f.lineage, secret_fact.lineage);
    assert_eq!(f.version, secret_fact.version);
    assert_eq!(f.origin, secret_fact.origin);
    assert_eq!(f.recorded_at, secret_fact.recorded_at);
    assert_eq!(f.valid_from, secret_fact.valid_from);
    let end = f.end.expect("a current version is closed first");
    assert_eq!(end.ending, Ending::Corrected);
    assert_eq!(end.next, None);
    let history = w.memory.history(f.lineage);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].body, FactBody::Redacted);
}

#[test]
fn a_redacted_attribute_frees_the_pair_for_a_new_value() {
    let w = World::new();
    let first = w.remember_one(w.goal("launch codes are hunter2"));
    let lineage = w.fact(first).lineage;
    let project = w.entity(w.project, "tessar");
    w.memory.redact(first.fact).unwrap();

    assert!(w.memory.values_of(project, w.goal).is_empty());
    assert!(found(&w, "hunter2").is_empty());
    assert!(found(&w, "tessar").is_empty());

    let second = w.remember_one(w.goal("a memory engine"));
    assert_eq!(second.outcome, Outcome::Created);
    let f = w.fact(second);
    assert_eq!(f.lineage, lineage);
    assert_eq!(f.version.get(), 2);
    assert_eq!(f.prev, Some(first.fact));
    let history = w.memory.history(lineage);
    assert_eq!(history[0].body, FactBody::Redacted);
    assert_eq!(text_of(&history[1]), "a memory engine");
    assert_eq!(w.memory.values_of(project, w.goal).len(), 1);
}

#[test]
fn an_old_version_can_be_redacted_without_touching_the_current_one() {
    let w = World::new();
    let v1 = w.remember_one(w.goal("old secret plan"));
    w.clock.advance(10);
    let v2 = w.remember_one(w.goal("new public plan"));
    assert!(matches!(v2.outcome, Outcome::Superseded { .. }));

    w.memory.redact(v1.fact).unwrap();
    let f1 = w.memory.fact(v1.fact).unwrap();
    // Closed as changed, not a correction: redaction does not rewrite when
    // things were true.
    assert!(matches!(f1.end.unwrap().ending, Ending::Changed { .. }));
    assert_eq!(f1.next(), Some(v2.fact));
    assert!(found(&w, "secret").is_empty());
    assert_eq!(found(&w, "public"), vec![v2.fact]);
    assert!(w.memory.current(f1.lineage).is_some());
}

#[test]
fn redacting_twice_or_an_unknown_fact_fails() {
    let w = World::new();
    let a = w.remember_one(statement("one"));
    w.memory.redact(a.fact).unwrap();
    let commits = w.log.commits();
    assert!(matches!(
        w.memory.redact(a.fact),
        Err(StoreError::AlreadyRedacted(id)) if id == a.fact
    ));
    assert_eq!(w.log.commits(), commits, "a failed redact writes nothing");

    // An id the memory has not minted yet.
    let bigger = World::new();
    let ids: Vec<FactId> = (0..5)
        .map(|i| bigger.remember_one(statement(&format!("fact {i}"))).fact)
        .collect();
    assert!(matches!(
        w.memory.redact(ids[4]),
        Err(StoreError::UnknownFact(id)) if id == ids[4]
    ));
}

#[test]
fn an_episode_can_be_blanked_once() {
    let w = World::new();
    let r = w
        .memory
        .remember(Remember {
            space: w.space,
            origin: origin(),
            episode: Some(EpisodeDraft {
                text: "I told you the password is hunter2".into(),
                at: None,
                source: key("test"),
            }),
            facts: vec![statement("a note")],
        })
        .unwrap();
    let episode = r.episode.unwrap();
    assert_eq!(
        w.memory.episode(episode).unwrap().text,
        "I told you the password is hunter2"
    );
    w.memory.redact_episode(episode).unwrap();
    let e = w.memory.episode(episode).unwrap();
    assert_eq!(e.text, "");
    assert_eq!(e.id, episode);
    assert!(matches!(
        w.memory.redact_episode(episode),
        Err(StoreError::EpisodeAlreadyRedacted(_))
    ));
    // The fact keeps its link until it is redacted itself.
    assert_eq!(w.fact(r.facts[0]).episode, Some(episode));
    w.memory.redact(r.facts[0].fact).unwrap();
    assert_eq!(w.fact(r.facts[0]).episode, None);

    let reopened = w.reopen();
    assert_eq!(reopened.episode(episode).unwrap().text, "");
}

/// A log with redactions of every kind. Returns the ids of the facts, the
/// words that must not survive a purge, and the world.
fn world_with_secrets(semantic: Option<Semantic>) -> (World, Vec<FactId>) {
    let w = World::with(semantic);
    let mut ids = Vec::new();
    let r = w
        .memory
        .remember(Remember {
            space: w.space,
            origin: origin(),
            episode: Some(EpisodeDraft {
                text: "I whispered the hunter2 password".into(),
                at: None,
                source: key("test"),
            }),
            facts: vec![
                FactDraft {
                    mentions: vec![w.person("Anna")],
                    ..statement("the vault code is hunter2")
                },
                w.goal("zebra launch plan"),
                statement("a harmless note about tessar"),
            ],
        })
        .unwrap();
    ids.extend(r.facts.iter().map(|f| f.fact));
    let episode = r.episode.unwrap();
    w.clock.advance(10);
    let goal2 = w.remember_one(w.goal("a public goal"));
    ids.push(goal2.fact);
    w.clock.advance(10);
    let berlin = w.remember_one(w.lives_in("Anna", "Atlantis"));
    ids.push(berlin.fact);

    w.clock.advance(10);
    w.memory.redact(r.facts[0].fact).unwrap(); // current statement
    w.memory.redact(r.facts[1].fact).unwrap(); // old version of goal
    w.memory.redact_episode(episode).unwrap();
    w.memory.redact(berlin.fact).unwrap(); // current triple, lineage of one
    w.clock.advance(10);
    let paris = w.remember_one(w.lives_in("Anna", "Paris")); // continues it
    assert_eq!(paris.outcome, Outcome::Created);
    ids.push(paris.fact);
    (w, ids)
}

/// What a reader can see: every fact by id, the episode, the histories,
/// and the results of a few queries.
fn observe(w: &World, memory: &Memory, ids: &[FactId]) -> String {
    let mut out = String::new();
    for &id in ids {
        let f = memory.fact(id).unwrap();
        out += &format!("{f:?}\n{:?}\n", memory.history(f.lineage));
    }
    let anna = memory.find_entity(w.space, w.person, "anna").unwrap().id;
    for view in [View::Current, View::TrueAt(ts(125)), View::KnownAt(ts(125))] {
        out += &format!("{:?}\n", memory.facts_about(anna, view));
    }
    let queries = ["hunter2", "Anna", "zebra", "public goal", "tessar", "Paris"];
    for (i, query) in queries.into_iter().enumerate() {
        let r = memory.recall(&Recall::new(query));
        out += &format!("query {i}: {:?}\n", r.hits);
    }
    out
}

fn assert_purge_is_equivalent(semantic: Option<Semantic>) {
    let (w, ids) = world_with_secrets(semantic.clone());
    let original = w.log.snapshot();
    let (purged, report) = purge_with_report(original.clone());

    assert_eq!(report.facts, 3);
    assert_eq!(report.episodes, 1);
    // Same commits, same ops in the same order; only content differs.
    assert_eq!(purged.len(), original.len());
    for (p, o) in purged.iter().zip(&original) {
        assert_eq!((p.seq, p.at, p.ops.len()), (o.seq, o.at, o.ops.len()));
    }
    let leaked = |commits: &[Commit<Op>]| {
        let json = serde_json::to_string(commits).unwrap();
        ["hunter2", "whispered", "vault code", "zebra"]
            .into_iter()
            .filter(|word| json.contains(word))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        leaked(&original),
        ["hunter2", "whispered", "vault code", "zebra"]
    );
    assert_eq!(leaked(&purged), Vec::<&str>::new());
    // Purging again changes nothing.
    let (again, report) = purge_with_report(purged.clone());
    assert_eq!(again, purged);
    assert_eq!((report.facts, report.episodes), (0, 0));

    let open = |commits: Vec<Commit<Op>>| {
        let log = SharedLog::from_commits(commits);
        match &semantic {
            Some(s) => Memory::open_with(log, w.clock.clone(), s.clone()),
            None => Memory::open(log, w.clock.clone()),
        }
        .unwrap()
    };
    let from_original = open(original);
    let from_purged = open(purged.clone());
    let expected = observe(&w, &w.memory, &ids);
    assert_eq!(observe(&w, &from_original, &ids), expected);
    assert_eq!(observe(&w, &from_purged, &ids), expected);
    assert!(!expected.contains("hunter2") && !expected.contains("zebra"));

    // The purged memory keeps working: Paris is the current value of the
    // pair, and a new city supersedes it in the same lineage.
    let rome = from_purged
        .remember(Remember {
            space: w.space,
            origin: origin(),
            episode: None,
            facts: vec![w.lives_in("Anna", "Rome")],
        })
        .unwrap()
        .facts[0];
    assert!(matches!(rome.outcome, Outcome::Superseded { .. }));
    let anna = from_purged
        .find_entity(w.space, w.person, "anna")
        .unwrap()
        .id;
    assert_eq!(from_purged.values_of(anna, w.lives_in).len(), 1);
}

#[test]
fn purge_gives_the_same_memory_without_the_content() {
    assert_purge_is_equivalent(None);
}

#[test]
fn purge_gives_the_same_memory_with_the_meaning_channel() {
    assert_purge_is_equivalent(Some(semantic()));
}

#[test]
fn replay_rebuilds_a_redacted_state() {
    for semantic in [None, Some(semantic())] {
        let (w, ids) = world_with_secrets(semantic);
        let reopened = w.reopen();
        assert_eq!(observe(&w, &reopened, &ids), observe(&w, &w.memory, &ids));
        // The meaning channel may still return other facts; none is the secret.
        let hits = w.memory.recall(&Recall::new("hunter2")).hits;
        assert!(
            hits.iter()
                .all(|h| !format!("{:?}", h.fact).contains("hunter2"))
        );
    }
}

#[test]
fn a_log_without_redactions_is_purged_to_itself() {
    let w = World::new();
    w.remember(vec![w.goal("a goal"), statement("a note")]);
    w.remember_one(w.goal("a better goal"));
    let commits = w.log.snapshot();
    let (purged, report) = purge_with_report(commits.clone());
    assert_eq!(purged, commits);
    assert_eq!((report.facts, report.episodes), (0, 0));
    assert_eq!(purge(commits.clone()), commits);

    // And the JSON of such a log does not mention redaction at all.
    let json = serde_json::to_string(&commits).unwrap();
    assert!(!json.contains("redact"), "{json}");
}

#[test]
fn redacted_body_and_ops_round_trip_through_json() {
    let json = serde_json::to_string(&FactBody::Redacted).unwrap();
    assert_eq!(json, r#"{"kind":"redacted"}"#);
    assert_eq!(
        serde_json::from_str::<FactBody>(&json).unwrap(),
        FactBody::Redacted
    );
    let (w, _) = world_with_secrets(None);
    let commits = w.log.snapshot();
    let json = serde_json::to_string(&commits).unwrap();
    assert!(json.contains(r#""op":"redact""#));
    assert!(json.contains(r#""op":"redact_episode""#));
    assert_eq!(
        serde_json::from_str::<Vec<Commit<Op>>>(&json).unwrap(),
        commits
    );
}
