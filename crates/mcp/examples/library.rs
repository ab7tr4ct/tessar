//! The core as a library: open a memory, define a predicate, remember a
//! fact, replace it, recall and read the history.
//!
//! Run with the model at `~/.tessar/models/voyage-4-nano-Q8_0.gguf`:
//!
//! ```sh
//! cargo run --release -p tessar-mcp --example library
//! ```

use std::error::Error;
use std::sync::Arc;

use tessar_core::clock::SystemClock;
use tessar_core::model::{Cardinality, Key, Name, Origin, SpaceKind, Value, ValueType};
use tessar_core::store::{
    BodyDraft, EntityRef, FactDraft, Memory, ObjectDraft, Op, PredicateDraft, Recall, Remember,
    Semantic,
};
use tessar_embed_llama::{LlamaEmbedder, voyage_nano_path};
use tessar_journal::FileJournal;

fn main() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let journal = FileJournal::<Op>::open(dir.path().join("journal.jsonl"))?;
    let model = voyage_nano_path().ok_or("no home directory")?;
    let embedder = Arc::new(LlamaEmbedder::voyage_nano(&model)?);
    let memory = Memory::open_with(
        journal,
        Arc::new(SystemClock),
        Semantic {
            embedder,
            dims: 1024,
            near_bonus: 0.05,
            predicate_threshold: 0.70,
        },
    )?;

    let space = memory.create_space(Key::new("notes")?, SpaceKind::LongTerm, None)?;
    let person = memory.define_kind(space, Name::new("person")?)?;
    let lives_in = memory.define_predicate_unchecked(
        space,
        PredicateDraft {
            name: Name::new("lives_in")?,
            description: "the place where the subject lives".into(),
            object: ValueType::Text,
            cardinality: Cardinality::Single,
        },
    )?;

    let anna_lives_in = |city: &str| FactDraft::new(BodyDraft::Triple {
        subject: EntityRef::Key {
            kind: person,
            key: Key::new("Anna").expect("valid key"),
        },
        predicate: lives_in,
        object: ObjectDraft::Value(Value::Text(city.into())),
        text: Some(format!("Anna lives in {city}.")),
        when: None,
    });
    let write = |fact: FactDraft| {
        memory.remember(Remember {
            space,
            origin: Origin::Asserted {
                by: Key::new("example").expect("valid key"),
            },
            episode: None,
            facts: vec![fact],
        })
    };

    let first = write(anna_lives_in("Berlin"))?;
    let second = write(anna_lives_in("Munich"))?;
    println!("first write: {:?}", first.facts[0].outcome);
    println!("second write: {:?}", second.facts[0].outcome);

    for hit in memory.recall(&Recall::new("where does Anna live?")).hits {
        println!("{:?}\n  because {:?}", hit.fact.body, hit.why);
    }

    let current = memory.fact(second.facts[0].fact).ok_or("fact vanished")?;
    for version in memory.history(current.lineage) {
        println!(
            "version {}: {:?}, current: {}",
            version.version,
            version.body,
            version.is_current()
        );
    }
    Ok(())
}
