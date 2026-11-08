# tessar-embed-cache

A persistent vector cache in front of any `tessar_core::embed::Embedder`.

Opening a memory embeds every fact again, because vectors are derived data
and never go to the journal. `CachedEmbedder` keeps them on disk, keyed by
model and text, so a reopen embeds only the facts that are new. The core
is unchanged: the cache is an ordinary embedder adapter.

- `embed_documents` looks every text up, embeds only the misses in one call
  to the inner embedder, keeps the order, and appends the new vectors to the
  file (`write_all` + `sync_data` per batch).
- `embed_query` passes straight through. Queries are not cached.
- Vectors are stored exactly as the inner embedder returned them (full
  length, f32). Changing `Semantic::dims` needs no invalidation. Changing the
  model does: the key and the file header include the model name, and
  opening a file written for another model fails, so pick another path.

## File format

```text
tessar-vectors 1 <model> <dims>\n
record*      record = key (32 bytes) ++ dims * f32 little-endian
```

The key is SHA-256 of `model`, a NUL byte, and the text. A trailing partial
record (a torn write) is cut off on open.

## Memory

The whole file is loaded on open. A vector costs `dims * 4` bytes plus the
32-byte key and map overhead: about `dims * 4 + 80` bytes, roughly 4 KiB per
fact at 1024 dimensions.

## Wiring

```rust
let embedder = CachedEmbedder::open(
    home.join("index/voyage-4-nano/vectors.bin"),
    "voyage-4-nano",
    Arc::new(LlamaEmbedder::new(/* ... */)?),
)?;
let memory = Memory::open_with(journal, clock, Semantic {
    embedder: Arc::new(embedder),
    ..semantic
})?;
```
