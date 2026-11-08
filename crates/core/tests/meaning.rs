//! The meaning channel: vectors from an embedder, fused with the lexical
//! channels, near-space bonus, failures and reopening.

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::*;
use tessar_core::embed::{EmbedError, Embedder};
use tessar_core::model::{FactBody, SpaceKind};
use tessar_core::store::{
    BodyDraft, Channels, FactDraft, Hit, Recall, Remember, Semantic, StoreError,
};

/// Index length in these tests.
const DIMS: usize = 8;

/// Words of the same meaning share a dimension, in any language. Words
/// it does not know add a little to the last dimension, so no text gets a
/// zero vector. Returns longer vectors than the index holds, to check the
/// cut.
#[derive(Default)]
struct Concepts {
    fail: AtomicBool,
    /// Return vectors of this length instead, if set.
    length: AtomicUsize,
    /// Every document text it was asked to embed.
    documents: Mutex<Vec<String>>,
}

const CONCEPTS: [&[&str]; 7] = [
    &["journal", "log", "журнал", "лог"],
    &["berlin", "берлин"],
    &["moved", "переехала", "переехал"],
    &["anna", "аня"],
    &["deadline", "срок", "дедлайн"],
    &["friday", "пятница", "пятницу"],
    &["memory", "память", "памяти"],
];

impl Concepts {
    fn vector(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if self.fail.load(Ordering::SeqCst) {
            return Err(EmbedError("server is down".into()));
        }
        let length = match self.length.load(Ordering::SeqCst) {
            0 => DIMS + 4,
            n => n,
        };
        let mut v = vec![0.0f32; length.max(DIMS)];
        v[DIMS - 1] = 0.1;
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .map(str::to_lowercase)
        {
            if let Some(c) = CONCEPTS.iter().position(|ws| ws.contains(&word.as_str())) {
                v[c] += 1.0;
            }
        }
        v.truncate(length);
        Ok(v)
    }

    fn fail(&self, fail: bool) {
        self.fail.store(fail, Ordering::SeqCst);
    }

    fn documents(&self) -> Vec<String> {
        self.documents.lock().unwrap().clone()
    }
}

impl Embedder for Concepts {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let vectors: Result<Vec<_>, _> = texts.iter().map(|t| self.vector(t)).collect();
        if vectors.is_ok() {
            self.documents.lock().unwrap().extend_from_slice(texts);
        }
        vectors
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.vector(text)
    }
}

fn world() -> (World, Arc<Concepts>) {
    let concepts = Arc::new(Concepts::default());
    let semantic = Semantic {
        embedder: concepts.clone(),
        dims: DIMS,
        near_bonus: 0.05,
        predicate_threshold: 1.0,
    };
    (World::with(Some(semantic)), concepts)
}

fn statement(text: &str) -> FactDraft {
    FactDraft::new(BodyDraft::Statement {
        text: text.into(),
        when: None,
    })
}

fn text(hit: &Hit) -> &str {
    match &hit.fact.body {
        FactBody::Statement { text, .. } => text,
        other => panic!("not a statement: {other:?}"),
    }
}

#[test]
fn finds_a_fact_in_another_language_by_meaning() {
    let (w, _) = world();
    w.remember(vec![
        statement("Anna moved to Berlin"),
        statement("The journal is append-only"),
    ]);

    let result = w.memory.recall(&Recall::new("Куда переехала Аня?"));
    assert_eq!(result.degraded, None);
    let top = &result.hits[0];
    assert_eq!(text(top), "Anna moved to Berlin");
    // No shared words: only the meaning channel found it, and coverage
    // does not cut its score.
    assert_eq!(top.why.keyword_rank, None);
    assert_eq!(top.why.meaning_rank, Some(1));
    assert!(top.why.similarity.unwrap() > 0.8);
    assert!(top.score > 0.0);
}

#[test]
fn without_an_embedder_there_is_no_meaning_channel() {
    let w = World::new();
    w.remember_one(statement("Anna moved to Berlin"));
    let result = w.memory.recall(&Recall::new("Куда переехала Аня?"));
    assert!(result.hits.is_empty());
    assert_eq!(result.degraded, None);
}

#[test]
fn channels_agree_on_the_best_fact() {
    let (w, _) = world();
    w.remember(vec![
        statement("Deadline is Friday"),
        statement("The journal is append-only"),
    ]);
    let top = &w.memory.recall(&Recall::new("journal")).hits[0];
    assert_eq!(text(top), "The journal is append-only");
    assert_eq!(top.why.keyword_rank, Some(1));
    assert_eq!(top.why.meaning_rank, Some(1));
}

#[test]
fn near_spaces_win_ties_but_not_strong_matches() {
    let (w, _) = world();
    let other = w
        .memory
        .create_space(key("other"), SpaceKind::LongTerm, None)
        .unwrap();
    let remember_in = |space, text: &str| {
        w.memory
            .remember(Remember {
                space,
                origin: origin(),
                episode: None,
                facts: vec![statement(text)],
            })
            .unwrap();
    };
    remember_in(w.space, "Deadline: Friday");
    remember_in(other, "Срок — пятница");

    let near = |space| Recall {
        near: vec![space],
        ..Recall::new("дедлайн пятницу")
    };
    // Same meaning in both spaces: the near one comes first.
    let here = w.memory.recall(&near(w.space)).hits;
    assert_eq!(text(&here[0]), "Deadline: Friday");
    assert!(here[0].why.near);
    let there = w.memory.recall(&near(other)).hits;
    assert_eq!(text(&there[0]), "Срок — пятница");

    // A strong match in another space beats a weak one nearby.
    remember_in(w.space, "Memory of deadlines");
    let hits = w
        .memory
        .recall(&Recall {
            near: vec![w.space],
            ..Recall::new("Anna moved to Berlin")
        })
        .hits;
    remember_in(other, "Anna moved to Berlin");
    let hits_after = w
        .memory
        .recall(&Recall {
            near: vec![w.space],
            ..Recall::new("Anna moved to Berlin")
        })
        .hits;
    assert_ne!(text(&hits[0]), "Anna moved to Berlin");
    assert_eq!(text(&hits_after[0]), "Anna moved to Berlin");
    assert!(!hits_after[0].why.near);
}

#[test]
fn a_write_fails_whole_if_the_embedder_fails() {
    let (w, concepts) = world();
    w.remember_one(statement("The journal is append-only"));
    let commits = w.log.commits();

    concepts.fail(true);
    let err = w
        .try_remember(vec![
            statement("Anna moved to Berlin"),
            w.knows("Anna", "Rust"),
        ])
        .unwrap_err();
    assert!(matches!(err, StoreError::Embed(_)), "{err:?}");
    assert_eq!(w.log.commits(), commits, "nothing reached the log");
    assert!(w.memory.find_entity(w.space, w.person, "anna").is_none());

    concepts.fail(false);
    w.remember_one(statement("Anna moved to Berlin"));
    let top = &w.memory.recall(&Recall::new("Аня переехала")).hits[0];
    assert_eq!(text(top), "Anna moved to Berlin");
}

#[test]
fn recall_runs_without_meaning_if_the_embedder_fails() {
    let (w, concepts) = world();
    w.remember(vec![
        statement("The journal is append-only"),
        statement("Anna moved to Berlin"),
    ]);
    concepts.fail(true);

    let result = w.memory.recall(&Recall::new("journal"));
    assert!(result.degraded.unwrap().contains("server is down"));
    assert_eq!(result.hits.len(), 1, "keyword channel only");
    assert_eq!(text(&result.hits[0]), "The journal is append-only");
    assert_eq!(result.hits[0].why.similarity, None);
}

#[test]
fn vectors_shorter_than_the_index_are_rejected() {
    let (w, concepts) = world();
    concepts.length.store(DIMS - 1, Ordering::SeqCst);
    let err = w.try_remember(vec![statement("journal")]).unwrap_err();
    assert!(matches!(err, StoreError::Embed(_)), "{err:?}");
}

#[test]
fn reopening_rebuilds_the_same_vectors() {
    let (w, concepts) = world();
    w.remember(vec![
        statement("Anna moved to Berlin"),
        w.lives_in("Anna", "Berlin"),
        statement("The journal is append-only"),
    ]);
    let query = Recall::new("Куда переехала Аня?");
    let before = w.memory.recall(&query).hits;
    let written = concepts.documents();

    let reopened = w.reopen();
    // The texts embedded on open are exactly those embedded on write —
    // including the key of an entity created in the same commit.
    let all = concepts.documents();
    assert_eq!(all[written.len()..], written[..]);
    assert!(
        written.contains(&"Anna lives in Berlin".to_owned()),
        "{written:?}"
    );

    let after = reopened.recall(&query).hits;
    assert_eq!(before, after);
}

#[test]
fn reopening_keeps_vectors_aligned_across_embedding_batches() {
    // Opening embeds facts in batches; with more facts than one batch,
    // every fact must still get its own vector (a zip over a shared id
    // iterator once lost one id per batch boundary).
    let (w, _) = world();
    for i in 0..150 {
        w.remember_one(statement(&format!("note number {i} about topic {}", i % 7)));
    }
    w.remember_one(statement("Anna moved to Berlin"));
    let query = Recall::new("Куда переехала Аня?");
    let before = w.memory.recall(&query).hits;
    assert!(before[0].why.similarity.is_some());

    let reopened = w.reopen();
    let after = reopened.recall(&query).hits;
    assert_eq!(before, after);
}

#[test]
fn opening_fails_if_the_embedder_fails() {
    let (w, concepts) = world();
    w.remember_one(statement("The journal is append-only"));
    concepts.fail(true);
    let semantic = w.semantic.clone().unwrap();
    let err = tessar_core::store::Memory::open_with(w.log.clone(), w.clock.clone(), semantic)
        .err()
        .unwrap();
    assert!(matches!(err, StoreError::Embed(_)), "{err:?}");
}

#[test]
fn channels_can_be_turned_off() {
    let (w, _) = world();
    w.remember(vec![
        statement("Anna moved to Berlin"),
        statement("The journal is append-only"),
    ]);
    let only = |channels| Recall {
        channels,
        ..Recall::new("journal Аня")
    };
    let lexical = w.memory.recall(&only(Channels {
        meaning: 0.0,
        ..Channels::default()
    }));
    assert_eq!(lexical.hits.len(), 1);
    assert!(lexical.hits.iter().all(|h| h.why.meaning_rank.is_none()));

    let meaning = w.memory.recall(&only(Channels {
        entities: 0.0,
        keywords: 0.0,
        meaning: 1.0,
    }));
    assert_eq!(meaning.hits.len(), 2);
    assert!(meaning.hits.iter().all(|h| h.why.keyword_rank.is_none()));
}
