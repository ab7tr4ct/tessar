//! The store on top of the file journal, end to end.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tessar_core::clock::{ManualClock, SystemClock};
use tessar_core::log::LogError;
use tessar_core::model::{
    Cardinality, FactBody, Key, Name, Origin, SpaceKind, Timestamp, Value, ValueType,
};
use tessar_core::store::{
    BodyDraft, EntityRef, FactDraft, Memory, ObjectDraft, Op, Remember, View,
};
use tessar_journal::FileJournal;

fn key(s: &str) -> Key {
    Key::new(s).unwrap()
}

fn name(s: &str) -> Name {
    Name::new(s).unwrap()
}

fn text(fact: &tessar_core::model::Fact) -> String {
    match &fact.body {
        FactBody::Triple {
            object: Value::Text(t),
            ..
        } => t.clone(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn memory_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal.log");
    let clock = Arc::new(ManualClock::new(Timestamp::from_millis(100)));

    let lineage = {
        let memory = Memory::open(FileJournal::<Op>::open(&path).unwrap(), clock.clone()).unwrap();
        let space = memory
            .create_space(key("tessar"), SpaceKind::LongTerm, None)
            .unwrap();
        let project = memory.define_kind(space, name("project")).unwrap();
        let goal = memory
            .define_predicate(space, name("goal"), ValueType::Text, Cardinality::Single)
            .unwrap();
        let draft = |text: &str| Remember {
            space,
            origin: Origin::Asserted { by: key("test") },
            episode: None,
            facts: vec![FactDraft::new(BodyDraft::Triple {
                subject: EntityRef::Key {
                    kind: project,
                    key: key("tessar"),
                },
                predicate: goal,
                object: ObjectDraft::Value(Value::Text(text.into())),
            })],
        };
        let first = memory.remember(draft("graph database")).unwrap();
        clock.set(Timestamp::from_millis(200));
        memory.remember(draft("memory engine")).unwrap();

        // While the memory is open, nobody else can open the journal.
        assert!(matches!(
            FileJournal::<Op>::open(&path),
            Err(LogError::Locked)
        ));
        memory.fact(first.facts[0].fact).unwrap().lineage
    };

    let memory = Memory::open(FileJournal::<Op>::open(&path).unwrap(), clock).unwrap();
    let history: Vec<String> = memory.history(lineage).iter().map(text).collect();
    assert_eq!(history, ["graph database", "memory engine"]);
    assert_eq!(text(&memory.current(lineage).unwrap()), "memory engine");
    let space = memory.space_named("tessar").unwrap().id;
    let project = memory.kind_named(space, "project").unwrap();
    let tessar = memory.find_entity(space, project, "TESSAR").unwrap().id;
    assert_eq!(
        text(&memory.facts_about(tessar, View::TrueAt(Timestamp::from_millis(150)))[0]),
        "graph database"
    );
}

/// Real disk sync: how many commits per second one writer gets, and how
/// long readers wait while writers are syncing.
#[test]
#[ignore = "measures the disk; run with --ignored --nocapture"]
fn write_throughput_and_read_latency() {
    const WRITERS: usize = 8;
    const PER_WRITER: usize = 40;

    let dir = tempfile::tempdir().unwrap();
    let journal = FileJournal::<Op>::open(dir.path().join("journal.log")).unwrap();
    let memory = Memory::open(journal, Arc::new(SystemClock)).unwrap();
    let space = memory
        .create_space(key("bench"), SpaceKind::LongTerm, None)
        .unwrap();
    let person = memory.define_kind(space, name("person")).unwrap();
    let knows = memory
        .define_predicate(space, name("knows"), ValueType::Text, Cardinality::Many)
        .unwrap();
    let draft = move |i: usize| Remember {
        space,
        origin: Origin::Asserted { by: key("bench") },
        episode: None,
        facts: vec![FactDraft::new(BodyDraft::Triple {
            subject: EntityRef::Key {
                kind: person,
                key: key("anna"),
            },
            predicate: knows,
            object: ObjectDraft::Value(Value::Text(format!("skill {i}"))),
        })],
    };

    // One writer, one commit at a time.
    let n = 100;
    let start = Instant::now();
    for i in 0..n {
        memory.remember(draft(i)).unwrap();
    }
    let sequential = start.elapsed();
    println!(
        "sequential: {n} commits in {sequential:?} = {:.2} ms/commit, {:.0} commits/s",
        sequential.as_secs_f64() * 1000.0 / n as f64,
        n as f64 / sequential.as_secs_f64()
    );

    // Many writers plus a reader measuring its own latency meanwhile.
    let anna = memory.find_entity(space, person, "anna").unwrap().id;
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reader = {
        let memory = memory.clone();
        let done = Arc::clone(&done);
        thread::spawn(move || {
            let mut worst = Duration::ZERO;
            let mut reads = 0u64;
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                let t = Instant::now();
                let facts = memory.facts_about(anna, View::Current);
                assert!(!facts.is_empty());
                worst = worst.max(t.elapsed());
                reads += 1;
            }
            (reads, worst)
        })
    };
    let start = Instant::now();
    let writers: Vec<_> = (0..WRITERS)
        .map(|w| {
            let memory = memory.clone();
            thread::spawn(move || {
                for i in 0..PER_WRITER {
                    memory.remember(draft(1000 + w * PER_WRITER + i)).unwrap();
                }
            })
        })
        .collect();
    for w in writers {
        w.join().unwrap();
    }
    let concurrent = start.elapsed();
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    let (reads, worst_read) = reader.join().unwrap();
    let total = WRITERS * PER_WRITER;
    println!(
        "{WRITERS} writers: {total} commits in {concurrent:?} = {:.0} commits/s",
        total as f64 / concurrent.as_secs_f64()
    );
    println!("reader meanwhile: {reads} reads, worst {worst_read:?}");
}
