# tessar-embed-cache

A persistent vector cache in front of any `tessar_core::embed::Embedder`.

Opening a memory embeds every fact again, because vectors are derived data
and never go to the journal. `CachedEmbedder` keeps them on disk, keyed by
fingerprint and text, so a reopen embeds only the facts that are new. The core
is unchanged: the cache is an ordinary embedder adapter.

- `embed_documents` looks every text up, embeds only the misses in one call
  to the inner embedder, keeps the order, and appends the new vectors to the
  file (`write_all` + `sync_data` per batch).
- `embed_query` passes straight through. Queries are not cached.
- Vectors are stored exactly as the inner embedder returned them (full
  length, f32). Changing `Semantic::dims` needs no invalidation. Changing the
  model does: the key and the file header include the model name, and
  opening a file written for another model fails, so pick another path.

## Fingerprint

The fingerprint is an opaque string chosen by the caller. It must change
whenever the vectors a text produces could change: model name, model file,
tokenizer, prompts, context length. The header and every key include it. A
file written for another fingerprint or format version is never served: it is
emptied and refilled. The cache is a rebuildable index, so losing it only
costs time. A fingerprint is 1 to 256 bytes with no spaces or newlines.

The `tessar` binary keeps one file per fingerprint,
`<home>/index/<fingerprint>/vectors.bin`, so two setups never empty each
other's cache. The library takes any path.

## File format

```text
tessar-vectors 2 <fingerprint> <dims>\n
record*   where record = key (32 bytes) ++ dims * f32 (little-endian)
```

The key is SHA-256 of the fingerprint, a NUL byte and the text. A trailing
partial record (a torn write) is cut off on open.

## Memory

The whole file is loaded on open: about `dims * 4 + 80` bytes per vector,
roughly 4 KiB per fact at 1024 dimensions.

## Use

```rust,ignore
// `inner` is any `Arc<dyn tessar_core::embed::Embedder>`, for example the
// one from `tessar-embed-candle`.
let cached = CachedEmbedder::open(
    home.join("index").join(fingerprint).join("vectors.bin"),
    fingerprint,
    inner,
)?;
let memory = Memory::open_with(journal, clock, Semantic {
    embedder: Arc::new(cached),
    ..semantic
})?;
```

A complete example that compiles is in the crate documentation on docs.rs.
