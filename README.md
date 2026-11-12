# tessar

A memory engine for AI agents, in Rust. Deterministic, local, no LLM
inside.

An agent (today Claude Code, through MCP tools and hooks) writes typed
facts and reads them back across conversations and projects. The core
keeps every version of every fact in an append-only journal, answers
"what is true now" and "what was true on that date", finds facts by the
entities, words and meaning of a question, and explains every hit.
Embeddings come from `voyage-4-nano`, which runs in the same process
(pure Rust, on candle). Nothing leaves the machine.

Status: version 0.1. The core and the MCP server work end to end on macOS
with Apple Silicon; Linux and CUDA are untested. The API is not stable.

**Paused since 2026-11-12, for about two weeks.** No new features are
being built while the system runs in daily use and collects data for the
next round of tests: a local decision log records what the hooks show an
agent, what they hold back, and which memory tools the agent then calls.
What to build next, if anything, will be decided from those numbers.
Everything on `main` works as described below.

## Install

```sh
cargo install --locked tessar
tessar setup
```

`tessar setup` does the rest, one step at a time, and ends each step
with a line saying whether it was done, was already in place, or was
left to you:

1. It downloads the embedding model (about 375 MB) and the tokenizer
   (about 7 MB) into `~/.tessar/models` with `curl`. Both files are
   pinned to a commit of their Hugging Face repository and checked
   against a SHA-256 sum; a file that fails the check is not used.
2. It registers the binary as the MCP server `tessar` in Claude Code,
   in the user scope (`claude mcp add --scope user`). It asks first.
3. It merges four hooks into `~/.claude/settings.json`, leaving the rest
   of the file as it is, and keeps the old file next to it as
   `settings.json.tessar-backup-<UTC time>`. It asks first.

The first step does not ask. Without a terminal the answer to a question
is no, and `setup` prints the command to run by hand; `--yes` answers
yes. `tessar setup --dry-run` changes nothing and prints what it would
do. Every step is idempotent, so running it again is safe. The exit
status is 0 when everything requested is done or was in place, 1 when a
step failed and 2 when some steps were left to you. Afterwards restart
Claude Code (or run `/mcp` in it).

Prerequisites: Rust 1.89 or newer, `curl`, and the Claude Code CLI
(`claude`) on `PATH`. Tested on macOS with Apple Silicon; Linux is
untested. If you decline a step, or `claude` is missing, do it by hand
(`setup` prints the command it could not run):

```sh
# the model and the tokenizer; the SHA-256 sums are
#   c6ed6cc898d50ae1a4492809affadc6c0d023f8ac05c07a37abe20c563152560
#   c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
mkdir -p ~/.tessar/models/voyage-4-nano
curl -L --fail -o ~/.tessar/models/voyage-4-nano-Q8_0.gguf \
  https://huggingface.co/MongoDB/voyage-4-nano-GGUF/resolve/03547551689253732502486d62de4cd0970d6564/voyage-4-nano-Q8_0.gguf
curl -L --fail -o ~/.tessar/models/voyage-4-nano/tokenizer.json \
  https://huggingface.co/MongoDB/voyage-4-nano/resolve/5fb34e870f49f19e79c94d5e118f5ee730133279/tokenizer.json
shasum -a 256 ~/.tessar/models/voyage-4-nano-Q8_0.gguf \
  ~/.tessar/models/voyage-4-nano/tokenizer.json

# the MCP server
claude mcp add --scope user tessar -- "$(command -v tessar)" mcp

# the hooks: print the snippet, merge it into ~/.claude/settings.json
tessar hook config
```

To undo the installation:

```sh
tessar daemon stop                    # if a daemon is running
claude mcp remove tessar --scope user
rm -r ~/.tessar                       # the memory itself: its journal too
cargo uninstall tessar
```

and take the four tessar hooks (`PreToolUse`, `SessionStart`,
`SubagentStart`, `UserPromptSubmit`; their commands end in
`tessar hook agent|session|subagent|prompt`) out of
`~/.claude/settings.json`, or put back the
`settings.json.tessar-backup-<UTC time>` that `setup` left beside it,
which also drops any change made to the file since. If
`CLAUDE_CONFIG_DIR` is set, Claude Code's files are there, not in
`~/.claude`. `~/.tessar` holds the journal, so keep a copy of it if you
may want the memory back.

After that Claude Code has six tools: `remember`, `lesson`, `recall`,
`vocabulary`, `update` and `about`. The tool descriptions are the agent's
manual. Memory lives in `~/.tessar/journal.jsonl`; the project space is
named after the repository's `origin`. `tessar mcp` talks to a daemon
that holds the memory and the model for all sessions; it starts the
daemon when none runs and replaces it when it is an older build (the new
daemon takes over from the old one). The vector cache, which spares
re-embedding the facts at start, is keyed by a fingerprint of the model
file, the tokenizer, the embedding settings and a revision number of the
embedding code, so a change in any of them starts a fresh cache.
`TESSAR_MODEL` and `TESSAR_TOKENIZER` point to the model and tokenizer
files if they live elsewhere. On macOS the first start of a build may
take longer while Metal compiles the candle kernels (seen once at about
15 s, usually well under a second); the daemon warms them up at start
and logs the time. The candle CPU path on Linux is not measured yet.
The commands, the hooks and the environment variables are in
[`crates/mcp/README.md`](crates/mcp/README.md).

## What it does

**Typed facts, not blobs.** A fact is a `statement` (a sentence), an
`attribute` (subject, predicate, value, with the sentence it came from)
or an `event` (with a date). Each fact has three times: when memory
learned it, from when it holds in the world, and `when` the described
thing happens ("the trip is in March"). Values that must be single
(`lives_in`, `job_title`) replace the old value and keep it in history;
everything else accumulates. Nothing is ever deleted: a change closes a
version and opens the next one, a correction marks a version as never
true, and reads can be made `as_of` any moment. What the person asserted
outranks what an agent inferred: an inferred value never replaces an
asserted one, it sits beside it as a reported contradiction until the
next asserted write settles it.

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
twenty-nine predicates ships with the engine.

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

**Lessons.** An agent records its own mistakes with the `lesson` tool (or
you do, with `tessar lesson "TEXT"`). A lesson is an ordinary dated statement
about the topic `lessons`, in the project space or in `user` for lessons that
hold everywhere. The prompt hook shows a lesson when a later prompt is close
in meaning to it (at most two per prompt).

**Hooks.** `tessar hook config` prints a snippet for
`~/.claude/settings.json` with four Claude Code hooks: `prompt`
(UserPromptSubmit) prints the facts and lessons that fit the user's message;
`session` (SessionStart) prints a short summary of what memory holds;
`agent` (PreToolUse on the Agent tool) and `subagent` (SubagentStart) give a
subagent memory that fits its brief: the first picks the facts and leaves them
in a short-lived file in `~/.tessar/hooks/`, the second hands them to the
agent. When several unnamed agents of one type run in parallel, one may
receive another's memory.

**Forgetting and privacy.** `update` can mark a fact as changed, as never
true (`wrong`) or destroy its content (`redact`). Because the journal is
append-only, `tessar purge` then rewrites it without the redacted content and
drops the vector cache, the decision log and the hooks' leftover files;
`tessar redact fNN... [--purge]` does both steps from a shell. `~/.tessar`
is created `0700` and the files in it `0600`. The hooks also keep a local
decision log, `~/.tessar/decisions.jsonl`: what they showed or held back
and which tools were called afterwards. It holds the text of the user's
messages, of the briefs given to subagents and of `recall` queries (each cut
to 2000 characters) and the error text of failed tool calls, never the texts
of facts; it is never sent anywhere and
`tessar purge` removes it.

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

The embedder runs in-process on candle, in pure Rust, reading the same
Q8 GGUF file as llama.cpp. On an M1 Pro it produces vectors identical to
llama.cpp (cosine 1.0000 on 200 texts) in 676 MB of memory, at about
13 ms per query against 10 ms for llama.cpp and at half its bulk
throughput; the measurements, including the fusions and custom Metal
kernels that did not pay off, are in `crates/embed-candle/EXPERIMENT.md`.

## Using the core as a library

Depend on `tessar-core` (the model, the store, recall and the ports),
`tessar-journal` (the file journal) and `tessar-embed-candle` (the
embedder); `tessar-embed-cache` puts a persistent vector cache in front
of any embedder. The flow in a few lines, where `model.gguf` and
`tokenizer.json` are the files `tessar setup` downloads:

```rust
use std::sync::Arc;
use tessar_core::clock::SystemClock;
use tessar_core::model::{Cardinality, Key, Name, Origin, SpaceKind, Value, ValueType};
use tessar_core::store::{
    BodyDraft, EntityRef, FactDraft, Memory, ObjectDraft, Op, PredicateDraft, Recall, Remember,
    Semantic,
};
use tessar_embed_candle::CandleEmbedder;
use tessar_journal::FileJournal;

let journal = FileJournal::<Op>::open("journal.jsonl")?;
let embedder = Arc::new(CandleEmbedder::voyage_nano_gguf("model.gguf", "tokenizer.json")?);
let semantic = Semantic { embedder, dims: 1024, near_bonus: 0.05, predicate_threshold: 0.70 };
let memory = Memory::open_with(journal, Arc::new(SystemClock), semantic)?;
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

let found = memory.recall(&Recall::new("where does Anna live?"));
```

Writing "Anna lives in Munich" later supersedes the Berlin version;
`memory.history(lineage)` returns both, and a `Recall` with
`View::TrueAt(date)` returns the one that held on that date. The
complete example, with the second write and the history, is
[`crates/embed-llama/examples/library.rs`](crates/embed-llama/examples/library.rs);
it uses the llama.cpp embedder (see below).

## Building from source

```sh
git clone https://github.com/ab7tr4ct/tessar.git && cd tessar
cargo build --release -p tessar     # target/release/tessar
target/release/tessar setup
```

`setup` registers the binary that runs it, so a source build is
registered by its path in `target/release`. The default build is pure
Rust: no cmake, no C++ toolchain; it embeds with candle (Metal on macOS,
CPU elsewhere). `cargo test -p tessar` needs no model and no GPU.

The llama.cpp embedder is not part of the binary. It lives in the
`tessar-embed-llama` crate, which is not published and is used for the
comparison tests against candle and for the example above. Building it
needs cmake, a C++ toolchain (Xcode Command Line Tools on macOS) and
`scripts/setup.sh`, which checks out the parts of the `third_party/llama.cpp`
submodule that the build needs. `TESSAR_EMBEDDER` accepts `candle`; `llama`
is an error.

## Layout

| crate | what |
|---|---|
| `crates/core` | data model, store, recall, ports (`Log`, `Clock`, `Embedder`). No I/O, no network, no LLM. |
| `crates/journal` | the file journal: JSON Lines, CRC-32 per line, `fsync`, torn-write recovery, file lock. |
| `crates/embed-candle` | the in-process embedder: pure Rust on candle, GGUF Q8; Metal on macOS, CPU elsewhere. |
| `crates/embed-cache` | a persistent vector cache in front of any embedder, keyed by a fingerprint. |
| `crates/mcp` | the package `tessar`: an MCP server on stdio, the daemon, the hooks and the `setup`, `lesson`, `redact` and `purge` commands. |
| `crates/embed` | HTTP embedder for `llama-server` and embedding APIs. Not published. |
| `crates/embed-llama` | llama.cpp embedder, a small C++ shim bridged with `cxx`; needs cmake. Not published; for comparison tests. |
| `third_party/llama.cpp` | git submodule (release b12189), only for `embed-llama`. |

The first five crates are published: `tessar`, `tessar-core`,
`tessar-journal`, `tessar-embed-cache` and `tessar-embed-candle`.

Principles: the core has no LLM and is deterministic; nothing is deleted;
exact keys come before vectors; every result is explainable; spaces from
day one; no feature without a measurement.

## Where it stands

The core, the MCP server, the daemon and the hooks are built and in daily
use by the author. The decision log exists so that the next steps can be
chosen from data about what the hooks show and what agents do with it. The
only consumers today are the author and Claude Code. Nothing beyond what is
described above is promised.

## License

MIT, see [LICENSE](LICENSE).
