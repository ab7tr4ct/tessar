# tessar

A memory engine for AI agents, in Rust. Deterministic, local, no LLM
inside.

An agent (Claude Code today, a voice agent next) writes typed facts and
reads them back across conversations and projects. The core keeps every
version of every fact in an append-only journal, answers "what is true
now" and "what was true on that date", finds facts by the entities,
words and meaning of a question, and explains every hit. Embeddings come
from `voyage-4-nano` running in the same process through llama.cpp.
Nothing leaves the machine.

Status: version 0.1. The core and the MCP server work end to end on macOS
with Apple Silicon; Linux and CUDA are untested. The API is not stable.

## What it does

**Typed facts, not blobs.** A fact is a `statement` (a sentence), an
`attribute` (subject, predicate, value, with the sentence it came from)
or an `event` (with a date). Each fact has three times: when memory
learned it, from when it holds in the world, and `when` the described
thing happens ("the trip is in March"). Values that must be single
(`lives_in`, `job_title`) replace the old value and keep it in history;
everything else accumulates. Nothing is ever deleted: a change closes a
version and opens the next one, a correction marks a version as never
true, and reads can be made `as_of` any moment.

**Entities and spaces.** Entities have a kind, a key and aliases, and are
the first thing recall looks for in a question. Facts live in spaces: one
for the person (`user`), one per repository, more later. A fact may refer
to an entity or predicate of any space, so `user` is one entity across all
projects. Spaces are soft boundaries: recall searches all of them and
gives the asker's own spaces a bonus.

**A vocabulary the agent extends but cannot pollute.** Predicates have a
description, a value type and a cardinality. A new predicate must be
short and universal: `trip_destination` with the value "Japan", never
`plans_trip_to_japan`. Names that contain a number or an entity's name
are rejected with the reason; names close in meaning to an existing
predicate get a hint ("looks like `goal`") and the agent decides whether
to reuse it, declare a synonym or insist. Synonyms merge for the
replacement rule without rewriting old facts. A starter vocabulary of
twenty-two predicates ships with the engine.

**Recall over three channels.** Entities named in the question, BM25
over words, and cosine similarity of int8 vectors, fused by reciprocal
rank fusion with a coverage term so that a fact answering the whole
question beats one matching a single word. Time is a filter and a bonus.
Every hit says which channels found it, at what rank, with what
similarity.

**A journal you can read.** JSON Lines with a CRC per line and `fsync`
per commit. A torn last line is cut off on open; damage in the middle
refuses to open rather than silently losing data. The in-memory state is
rebuilt by replaying the journal through the same code that applies live
writes, so the two can never diverge.

## Results

On LongMemEval (S, knowledge-update and temporal-reasoning, 179 test
questions; Claude Haiku writes the facts "as an agent", reads and judges;
8 000 characters of context per question):

| system | accuracy |
|---|---|
| tessar: extracted facts, recall, dates, versions | 87.2% |
| flat vector search over the same chats, same budget | 54.7% |

The gain (+32 points, 95% CI +23.5 to +40.8) comes from short dated facts
in place of raw chat turns. Honest notes from the same runs: version
history did not add accuracy on this benchmark, and the embedding
similarity hint for duplicate predicates is noisy, so the vocabulary
relies on its hard rules and the starter set. The evaluation harness is
not published yet.

## Quick start

Prerequisites: a recent stable Rust (edition 2024), `cmake` and a C++
toolchain (Xcode Command Line Tools on macOS). The build compiles
llama.cpp from the submodule once; it takes a few minutes.

```sh
git clone https://github.com/ab7tr4ct/tessar.git && cd tessar
scripts/setup.sh                      # llama.cpp submodule, sparse checkout
cargo build --release -p tessar-mcp   # the `tessar` binary

mkdir -p ~/.tessar/models
curl -L -o ~/.tessar/models/voyage-4-nano-Q8_0.gguf \
  https://huggingface.co/MongoDB/voyage-4-nano-GGUF/resolve/main/voyage-4-nano-Q8_0.gguf

claude mcp add tessar -- "$PWD/target/release/tessar" mcp
```

Claude Code then has four tools: `remember`, `recall`, `vocabulary` and
`about`. The tool descriptions are the agent's manual. Memory lives in
`~/.tessar/journal.jsonl`; the project space is named after the
repository's `origin`. Details, including `.mcp.json` and environment
variables, are in [`crates/mcp/README.md`](crates/mcp/README.md).

## Using the core as a library

```rust
use std::sync::Arc;
use tessar_core::clock::SystemClock;
use tessar_core::model::{Cardinality, Key, Name, Origin, SpaceKind, Value, ValueType};
use tessar_core::store::{
    BodyDraft, EntityRef, FactDraft, Memory, ObjectDraft, Op, PredicateDraft, Recall, Remember,
    Semantic,
};

let journal = tessar_journal::FileJournal::<Op>::open("journal.jsonl")?;
let embedder = Arc::new(tessar_embed_llama::LlamaEmbedder::voyage_nano("voyage-4-nano-Q8_0.gguf")?);
let memory = Memory::open_with(
    journal,
    Arc::new(SystemClock),
    Semantic { embedder, dims: 1024, near_bonus: 0.05, predicate_threshold: 0.70 },
)?;

let space = memory.create_space(Key::new("notes")?, SpaceKind::LongTerm, None)?;
let person = memory.define_kind(space, Name::new("person")?)?;
let lives_in = memory.define_predicate_unchecked(space, PredicateDraft {
    name: Name::new("lives_in")?,
    description: "the place where the subject lives".into(),
    object: ValueType::Text,
    cardinality: Cardinality::Single,
})?;

memory.remember(Remember {
    space,
    origin: Origin::Asserted { by: Key::new("me")? },
    episode: None,
    facts: vec![FactDraft::new(BodyDraft::Triple {
        subject: EntityRef::Key { kind: person, key: Key::new("Anna")? },
        predicate: lives_in,
        object: ObjectDraft::Value(Value::Text("Berlin".into())),
        text: Some("Anna lives in Berlin.".into()),
        when: None,
    })],
})?;

for hit in memory.recall(&Recall::new("where does Anna live?")).hits {
    println!("{:?} because {:?}", hit.fact.body, hit.why);
}
```

Writing "Anna lives in Munich" later supersedes the Berlin version;
`memory.history(lineage)` returns both, and a `Recall` with
`View::TrueAt(date)` returns the one that held on that date.

## Layout

| crate | what |
|---|---|
| `crates/core` | data model, store, recall, ports (`Log`, `Clock`, `Embedder`). No I/O, no network, no LLM. |
| `crates/journal` | the file journal: JSON Lines, CRC-32 per line, `fsync`, torn-write recovery, file lock. |
| `crates/embed-llama` | in-process embedder: a small C++ shim over llama.cpp bridged with `cxx`; Metal on macOS, CPU elsewhere. |
| `crates/embed` | HTTP embedder for `llama-server` and embedding APIs. |
| `crates/mcp` | the `tessar` binary: an MCP server on stdio. |
| `third_party/llama.cpp` | git submodule (release b12189); `scripts/setup.sh` checks out only the parts the build needs. |

Principles: the core has no LLM and is deterministic; nothing is deleted;
exact keys come before vectors; every result is explainable; spaces from
day one; no feature without a measurement.

## Roadmap

- Hooks for Claude Code: recall on every prompt within a token budget, a
  summary at session start, past / present / future tags on hits, stale
  value warnings.
- A vector cache on disk, so that opening a large memory is instant; a
  daemon shared by all sessions on the machine.
- A voice agent as the second consumer: memory between calls within a
  latency budget.
- Memory about code and documents: symbols and files with edges from the
  compiler (SCIP) and tree-sitter.

## License

MIT or Apache-2.0, at your option.
