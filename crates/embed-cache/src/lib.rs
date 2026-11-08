//! A persistent vector cache in front of any [`Embedder`].
//!
//! Opening a memory re-embeds every fact, because vectors are derived data
//! and never go to the journal. [`CachedEmbedder`] remembers the vectors on
//! disk, keyed by model and text, so a reopen embeds only what is new.
//!
//! The cache stores vectors exactly as the inner embedder returned them
//! (full length, `f32`). Changing the index length (`Semantic::dims`)
//! downstream therefore needs no invalidation; changing the model does, and
//! the key and the header both include the model name.
//!
//! # File format
//!
//! ```text
//! tessar-vectors 1 <model> <dims>\n
//! record*   where record = key (32 bytes) ++ dims * f32 (little-endian)
//! ```
//!
//! The key is SHA-256 of `model`, a NUL byte and the text. A trailing
//! partial record (a torn write) is cut off on open.
//!
//! # Memory
//!
//! The whole cache is loaded into RAM on open: `dims * 4` bytes of vector
//! plus 32 bytes of key and some map overhead per entry, about
//! `dims * 4 + 80` bytes (4176 for 1024 dimensions).

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use sha2::{Digest, Sha256};
use tessar_core::embed::{EmbedError, Embedder};

const MAGIC: &str = "tessar-vectors";
const VERSION: u32 = 1;
const KEY_LEN: usize = 32;
/// The longest header line accepted; a longer one is not our file.
const MAX_HEADER: usize = 1024;

type Key = [u8; KEY_LEN];

/// An embedder that caches document vectors on disk. Queries pass through
/// uncached.
pub struct CachedEmbedder {
    inner: Arc<dyn Embedder>,
    model: String,
    state: Mutex<Cache>,
}

struct Cache {
    file: File,
    /// Vector length: from the header, or from the first vector of a new file.
    dims: Option<usize>,
    vectors: HashMap<Key, Vec<f32>>,
}

fn io_err(what: &str, e: std::io::Error) -> EmbedError {
    EmbedError(format!("vector cache: {what}: {e}"))
}

fn not_a_cache() -> EmbedError {
    EmbedError("vector cache: not a vector cache file".into())
}

fn key_of(model: &str, text: &str) -> Key {
    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    hasher.update([0]);
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

impl CachedEmbedder {
    /// Opens the cache file at `path`, creating it and its parent
    /// directory if missing. `model` names the model behind `inner`; a
    /// file written for another model is an error, so pick another path.
    ///
    /// For a new file the vector length is learned from the first batch
    /// `inner` returns.
    pub fn open(
        path: impl AsRef<Path>,
        model: &str,
        inner: Arc<dyn Embedder>,
    ) -> Result<Self, EmbedError> {
        let path = path.as_ref();
        if model.is_empty() || model.contains([' ', '\n']) {
            return Err(EmbedError(format!(
                "vector cache: model name {model:?} must be non-empty and have no spaces"
            )));
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| io_err("create directory", e))?;
        }
        // Append mode: every write goes to the end, whatever `set_len` did.
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)
            .map_err(|e| io_err("open", e))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| io_err("read", e))?;
        let mut cache = Cache {
            file,
            dims: None,
            vectors: HashMap::new(),
        };
        cache.load(&bytes, model)?;
        Ok(Self {
            inner,
            model: model.to_owned(),
            state: Mutex::new(cache),
        })
    }

    /// How many vectors are cached.
    pub fn len(&self) -> usize {
        self.lock().vectors.len()
    }

    /// Whether the cache holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> MutexGuard<'_, Cache> {
        // The map changes only after the file write succeeded, so a panic
        // under the lock leaves both consistent.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Cache {
    fn load(&mut self, bytes: &[u8], model: &str) -> Result<(), EmbedError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let Some(end) = bytes.iter().take(MAX_HEADER).position(|&b| b == b'\n') else {
            // A torn header of a file that never got a vector: start over.
            if bytes.len() < MAX_HEADER
                && MAGIC
                    .as_bytes()
                    .starts_with(&bytes[..bytes.len().min(MAGIC.len())])
            {
                return self.cut(0);
            }
            return Err(not_a_cache());
        };
        let header = std::str::from_utf8(&bytes[..end]).map_err(|_| not_a_cache())?;
        let parts: Vec<&str> = header.split(' ').collect();
        let [magic, version, file_model, dims] = parts[..] else {
            return Err(not_a_cache());
        };
        if magic != MAGIC {
            return Err(not_a_cache());
        }
        if version != VERSION.to_string() {
            return Err(EmbedError(format!(
                "vector cache: unsupported version {version}"
            )));
        }
        if file_model != model {
            return Err(EmbedError(format!(
                "vector cache: the file holds vectors of model {file_model:?}, \
                 not {model:?}; use another file"
            )));
        }
        let dims: usize = dims
            .parse()
            .ok()
            .filter(|&d| d > 0)
            .ok_or_else(|| EmbedError("vector cache: bad dimensions in the header".into()))?;
        let record = KEY_LEN + dims * 4;
        let body = &bytes[end + 1..];
        for chunk in body.chunks_exact(record) {
            let mut key = [0u8; KEY_LEN];
            key.copy_from_slice(&chunk[..KEY_LEN]);
            let (words, _) = chunk[KEY_LEN..].as_chunks::<4>();
            let vector = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
            self.vectors.insert(key, vector);
        }
        self.dims = Some(dims);
        let good = end + 1 + body.len() / record * record;
        if good < bytes.len() {
            // A torn write: cut the partial record, as the journal does.
            self.cut(good as u64)?;
        }
        Ok(())
    }

    fn cut(&mut self, len: u64) -> Result<(), EmbedError> {
        self.file
            .set_len(len)
            .map_err(|e| io_err("cut a torn record", e))?;
        self.file.sync_data().map_err(|e| io_err("sync", e))
    }

    /// Appends records (and the header, for a new file) and syncs.
    fn append(&mut self, model: &str, entries: Vec<(Key, Vec<f32>)>) -> Result<(), EmbedError> {
        let Some(first) = entries.first() else {
            return Ok(());
        };
        let dims = self.dims.unwrap_or(first.1.len());
        if dims == 0 || entries.iter().any(|(_, v)| v.len() != dims) {
            return Err(EmbedError(format!(
                "vector cache: the embedder returned a vector of a length other than {dims}"
            )));
        }
        let mut buf = Vec::with_capacity(64 + entries.len() * (KEY_LEN + dims * 4));
        if self.dims.is_none() {
            buf.extend_from_slice(format!("{MAGIC} {VERSION} {model} {dims}\n").as_bytes());
        }
        for (key, vector) in &entries {
            buf.extend_from_slice(key);
            for x in vector {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        self.file.write_all(&buf).map_err(|e| io_err("write", e))?;
        self.file.sync_data().map_err(|e| io_err("sync", e))?;
        self.dims = Some(dims);
        self.vectors.extend(entries);
        Ok(())
    }
}

impl Embedder for CachedEmbedder {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let keys: Vec<Key> = texts.iter().map(|t| key_of(&self.model, t)).collect();
        // The lock is held through the inner call, so concurrent callers
        // wait instead of embedding the same texts twice.
        let mut cache = self.lock();
        let mut first_miss: HashMap<Key, usize> = HashMap::new();
        let mut missing: Vec<usize> = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if !cache.vectors.contains_key(key) && first_miss.insert(*key, i).is_none() {
                missing.push(i);
            }
        }
        if !missing.is_empty() {
            let todo: Vec<String> = missing.iter().map(|&i| texts[i].clone()).collect();
            let vectors = self.inner.embed_documents(&todo)?;
            if vectors.len() != todo.len() {
                return Err(EmbedError(format!(
                    "{} vectors for {} texts",
                    vectors.len(),
                    todo.len()
                )));
            }
            let entries = missing.iter().map(|&i| keys[i]).zip(vectors).collect();
            cache.append(&self.model, entries)?;
        }
        Ok(keys.iter().map(|k| cache.vectors[k].clone()).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.inner.embed_query(text)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Vectors derived from the text; records every batch it was asked for.
    #[derive(Default)]
    struct Counting {
        batches: Mutex<Vec<Vec<String>>>,
    }

    impl Counting {
        fn embedded(&self) -> usize {
            self.batches.lock().unwrap().iter().map(Vec::len).sum()
        }
        fn calls(&self) -> usize {
            self.batches.lock().unwrap().len()
        }
    }

    impl Embedder for Counting {
        fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
            self.batches.lock().unwrap().push(texts.to_vec());
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32, 0.5, -1.25])
                .collect())
        }
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![text.len() as f32, 9.0, 9.0])
        }
    }

    fn texts(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    fn open(dir: &Path, model: &str, inner: &Arc<Counting>) -> Result<CachedEmbedder, EmbedError> {
        CachedEmbedder::open(dir.join("sub/vectors.bin"), model, inner.clone())
    }

    #[test]
    fn the_second_call_embeds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &inner).unwrap();
        let a = cache.embed_documents(&texts(&["one", "three"])).unwrap();
        assert_eq!(inner.embedded(), 2);
        let b = cache.embed_documents(&texts(&["one", "three"])).unwrap();
        assert_eq!(inner.embedded(), 2);
        assert_eq!(inner.calls(), 1);
        assert_eq!(a, b);
    }

    #[test]
    fn order_is_kept_with_hits_and_misses() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &inner).unwrap();
        cache.embed_documents(&texts(&["bb"])).unwrap();
        let got = cache
            .embed_documents(&texts(&["a", "bb", "cccc", "a"]))
            .unwrap();
        let lens: Vec<f32> = got.iter().map(|v| v[0]).collect();
        assert_eq!(lens, [1.0, 2.0, 4.0, 1.0]);
        // One call for the misses, each distinct text once.
        let batches = inner.batches.lock().unwrap();
        assert_eq!(batches.last().unwrap(), &texts(&["a", "cccc"]));
    }

    #[test]
    fn reopening_reads_the_cache_back() {
        let dir = tempfile::tempdir().unwrap();
        let first = Arc::new(Counting::default());
        let expected = open(dir.path(), "m", &first)
            .unwrap()
            .embed_documents(&texts(&["x", "yy"]))
            .unwrap();
        let second = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &second).unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(
            cache.embed_documents(&texts(&["x", "yy"])).unwrap(),
            expected
        );
        assert_eq!(second.calls(), 0);
        // Appending after a reopen keeps the file readable.
        cache.embed_documents(&texts(&["zzz"])).unwrap();
        let third = Arc::new(Counting::default());
        assert_eq!(open(dir.path(), "m", &third).unwrap().len(), 3);
    }

    #[test]
    fn another_model_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        open(dir.path(), "m1", &inner)
            .unwrap()
            .embed_documents(&texts(&["x"]))
            .unwrap();
        let err = open(dir.path(), "m2", &inner).err().unwrap();
        assert!(err.0.contains("m1") && err.0.contains("m2"), "{err}");
    }

    #[test]
    fn a_torn_record_is_cut_off() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        open(dir.path(), "m", &inner)
            .unwrap()
            .embed_documents(&texts(&["x", "yy"]))
            .unwrap();
        let path = dir.path().join("sub/vectors.bin");
        let full = std::fs::metadata(&path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(full - 5).unwrap();
        drop(file);

        let again = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &again).unwrap();
        assert_eq!(cache.len(), 1);
        let header = "tessar-vectors 1 m 3\n".len() as u64;
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            header + (KEY_LEN + 12) as u64
        );
        cache.embed_documents(&texts(&["x", "yy"])).unwrap();
        assert_eq!(again.embedded(), 1);
        let last = Arc::new(Counting::default());
        assert_eq!(open(dir.path(), "m", &last).unwrap().len(), 2);
    }

    #[test]
    fn queries_pass_through() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &inner).unwrap();
        assert_eq!(cache.embed_query("abc").unwrap(), vec![3.0, 9.0, 9.0]);
        assert!(cache.is_empty());
    }

    #[test]
    fn it_works_from_several_threads() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let cache: Arc<dyn Embedder> = Arc::new(open(dir.path(), "m", &inner).unwrap());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cache = cache.clone();
                std::thread::spawn(move || {
                    cache.embed_documents(&texts(&["a", "bb", "ccc"])).unwrap()
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap().len(), 3);
        }
        assert_eq!(inner.embedded(), 3);
    }
}
