# tessar-core

The core of [tessar](https://github.com/ab7tr4ct/tessar), a memory engine for
AI agents. Deterministic, no LLM and no network inside: typed facts with
validity periods and immutable versions, entities with keys and aliases,
`as_of` reads, and recall over entities, BM25 and meaning with an explanation
for every hit.

This crate holds the data model, the memory rules and the ports (`Log`,
`Clock`, `Embedder`). Adapters live in other crates:
[`tessar-journal`](https://crates.io/crates/tessar-journal) persists to a file,
[`tessar-embed-candle`](https://crates.io/crates/tessar-embed-candle) makes
vectors, and the [`tessar`](https://crates.io/crates/tessar) binary serves it
over MCP.

Most people want the binary: `cargo install tessar`.
