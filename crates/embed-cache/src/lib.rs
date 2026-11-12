//! A persistent vector cache in front of any [`Embedder`].
//!
//! Opening a memory re-embeds every fact, because vectors are derived data
//! and never go to the journal. [`CachedEmbedder`] remembers the vectors on
//! disk, keyed by fingerprint and text, so a reopen embeds only what is new.
//!
//! The cache stores vectors exactly as the inner embedder returned them
//! (full length, `f32`). Changing the index length (`Semantic::dims`)
//! downstream therefore needs no invalidation. Changing anything that alters
//! the vectors does, and that is the job of the *fingerprint*: an opaque
//! string chosen by the caller that must change whenever the vectors a given
//! text produces could change (model name, model file, tokenizer, prompts,
//! context length). The header and every key include it.
//!
//! The cache is a rebuildable index, not a source of truth: losing it only
//! costs time. So a file whose header carries another fingerprint, another
//! format version (including version 1, which stored a model name) or a
//! damaged header is **never served and is started afresh**: truncated to
//! empty and refilled. A file that does not start with the magic string is
//! not ours and is an error; it is left untouched.
//!
//! Writing the cache is best effort. If a write fails, the embedder still
//! returns the vectors it computed, the torn tail is cut off if possible, and
//! the cache stops writing for the rest of the process (vectors are then kept
//! in memory only). The next open cuts any torn tail itself.
//!
//! On Unix the directory is created `0700` and the file `0600`; existing ones
//! with group or other bits are tightened (never loosened) on open, and a
//! failure to tighten is ignored.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use tessar_core::embed::{EmbedError, Embedder};
//! use tessar_embed_cache::CachedEmbedder;
//!
//! struct Toy;
//! impl Embedder for Toy {
//!     fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
//!         Ok(texts.iter().map(|t| vec![t.len() as f32, 1.0]).collect())
//!     }
//!     fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
//!         Ok(vec![text.len() as f32, 1.0])
//!     }
//! }
//!
//! let dir = tempfile::tempdir()?;
//! let path = dir.path().join("index").join("toy-1").join("vectors.bin");
//! let cached = CachedEmbedder::open(&path, "toy-1", Arc::new(Toy))?;
//! let vectors = cached.embed_documents(&["hello".to_owned()])?;
//! assert_eq!(vectors, vec![vec![5.0, 1.0]]);
//! assert_eq!(cached.len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # File format
//!
//! ```text
//! tessar-vectors 2 <fingerprint> <dims>\n
//! record*   where record = key (32 bytes) ++ dims * f32 (little-endian)
//! ```
//!
//! The key is SHA-256 of `fingerprint`, a NUL byte and the text. A trailing
//! partial record (a torn write) is cut off on open. Vectors with a NaN or
//! an infinity, or of a length other than `dims`, are returned to the caller
//! but never written.
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
const VERSION: u32 = 2;
const KEY_LEN: usize = 32;
/// The longest header line accepted; a longer one is not our file.
const MAX_HEADER: usize = 1024;
/// The longest fingerprint accepted, well inside `MAX_HEADER`.
const MAX_FINGERPRINT: usize = 256;
/// The largest vector length accepted, from a header or from an embedder.
/// Keeps `KEY_LEN + dims * 4` far from overflow on a damaged header.
const MAX_DIMS: usize = 65536;

type Key = [u8; KEY_LEN];

/// An embedder that caches document vectors on disk. Queries pass through
/// uncached.
pub struct CachedEmbedder {
    inner: Arc<dyn Embedder>,
    fingerprint: String,
    state: Mutex<Cache>,
}

/// The file operations the cache needs. A seam so tests can inject failures
/// without relying on a full disk.
trait Backing: Send {
    fn append(&mut self, bytes: &[u8]) -> std::io::Result<()>;
    fn truncate(&mut self, len: u64) -> std::io::Result<()>;
    fn sync(&mut self) -> std::io::Result<()>;
}

impl Backing for File {
    fn append(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.write_all(bytes)
    }
    fn truncate(&mut self, len: u64) -> std::io::Result<()> {
        self.set_len(len)
    }
    fn sync(&mut self) -> std::io::Result<()> {
        self.sync_data()
    }
}

struct Cache {
    file: Box<dyn Backing>,
    /// Bytes of the file known to hold a header and whole records.
    good_len: u64,
    /// Vector length: from the header, or from the first vector of a new file.
    dims: Option<usize>,
    vectors: HashMap<Key, Vec<f32>>,
    /// Cleared after a failed write: appending after a torn record would
    /// shift every later record, and a file that failed once is not trusted
    /// again in this process. The next open repairs the tail.
    writable: bool,
    /// The first error that stopped the cache from writing.
    write_error: Option<String>,
    /// The first finite vector whose length differs from `dims`.
    mismatch: Option<String>,
}

fn io_err(what: &str, e: std::io::Error) -> EmbedError {
    EmbedError(format!("vector cache: {what}: {e}"))
}

fn not_a_cache() -> EmbedError {
    EmbedError("vector cache: not a vector cache file".into())
}

fn key_of(fingerprint: &str, text: &str) -> Key {
    let mut hasher = Sha256::new();
    hasher.update(fingerprint.as_bytes());
    hasher.update([0]);
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

/// Clears group and other bits and keeps the owner's. Failures are ignored:
/// a cache with loose permissions still works.
#[cfg(unix)]
fn tighten(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    // A symlink is not ours to chmod: its target may be anything.
    if let Ok(meta) = std::fs::symlink_metadata(path)
        && !meta.file_type().is_symlink()
    {
        let mode = meta.permissions().mode();
        if mode & 0o077 != 0 {
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode & !0o077));
        }
    }
}

#[cfg(not(unix))]
fn tighten(_path: &Path) {}

fn create_dir(dir: &Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)
}

fn is_finite(vector: &[f32]) -> bool {
    vector.iter().all(|x| x.is_finite())
}

impl CachedEmbedder {
    /// Opens the cache file at `path`, creating it and its parent
    /// directory if missing. `fingerprint` identifies everything that shapes
    /// the vectors `inner` returns (see the crate docs); it must be non-empty,
    /// at most 256 bytes and contain no spaces or newlines. A file written
    /// for another fingerprint or format version is not used: it is emptied
    /// and started afresh.
    ///
    /// For a new file the vector length is learned from the first batch
    /// `inner` returns.
    pub fn open(
        path: impl AsRef<Path>,
        fingerprint: &str,
        inner: Arc<dyn Embedder>,
    ) -> Result<Self, EmbedError> {
        Self::open_with(path.as_ref(), fingerprint, inner, |f| Box::new(f))
    }

    fn open_with(
        path: &Path,
        fingerprint: &str,
        inner: Arc<dyn Embedder>,
        wrap: impl FnOnce(File) -> Box<dyn Backing>,
    ) -> Result<Self, EmbedError> {
        if fingerprint.is_empty()
            || fingerprint.len() > MAX_FINGERPRINT
            || fingerprint.contains([' ', '\n'])
        {
            let head: String = fingerprint.chars().take(16).collect();
            return Err(EmbedError(format!(
                "vector cache: a fingerprint must be 1..={MAX_FINGERPRINT} bytes with no spaces \
                 or newlines (got {} bytes, starting {head:?})",
                fingerprint.len()
            )));
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            create_dir(parent).map_err(|e| io_err("create directory", e))?;
            tighten(parent);
        }
        // Append mode: every write goes to the end, whatever `set_len` did.
        let mut options = OpenOptions::new();
        options.read(true).append(true).create(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(path).map_err(|e| io_err("open", e))?;
        tighten(path);
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| io_err("read", e))?;
        let mut cache = Cache {
            file: wrap(file),
            good_len: 0,
            dims: None,
            vectors: HashMap::new(),
            writable: true,
            write_error: None,
            mismatch: None,
        };
        cache.load(&bytes, fingerprint)?;
        Ok(Self {
            inner,
            fingerprint: fingerprint.to_owned(),
            state: Mutex::new(cache),
        })
    }

    /// How many vectors are cached.
    pub fn len(&self) -> usize {
        self.lock().vectors.len()
    }

    /// The first error that made the cache stop writing to disk, if any. After
    /// it the cache keeps serving vectors from memory, but nothing more is
    /// persisted in this process. Meant to be logged once by the caller.
    pub fn write_error(&self) -> Option<String> {
        self.lock().write_error.clone()
    }

    /// The first time the embedder returned a finite vector whose length
    /// differs from the cache's, a message with both lengths. Such vectors are
    /// returned to the caller but not cached, and the file is left as it is;
    /// vectors of the right length keep being cached. Typically this means the
    /// fingerprint does not capture a change of model. Meant to be logged once.
    pub fn length_mismatch(&self) -> Option<String> {
        self.lock().mismatch.clone()
    }

    /// Whether the cache holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> MutexGuard<'_, Cache> {
        // The map changes only under the lock and never half-way, so a panic
        // under the lock leaves it consistent.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Cache {
    fn load(&mut self, bytes: &[u8], fingerprint: &str) -> Result<(), EmbedError> {
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
        if parts.first() != Some(&MAGIC) {
            return Err(not_a_cache());
        }
        // Anything but a well-formed header of our version and fingerprint
        // (version 1, another fingerprint, a damaged header) is stale: the
        // file is only an index, so start afresh.
        let dims = match parts[..] {
            [_, version, file_fingerprint, dims]
                if version == VERSION.to_string() && file_fingerprint == fingerprint =>
            {
                dims.parse::<usize>()
                    .ok()
                    .filter(|d| (1..=MAX_DIMS).contains(d))
            }
            _ => None,
        };
        let Some(dims) = dims else {
            return self.cut(0);
        };
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
        self.good_len = good as u64;
        if good < bytes.len() {
            // A torn write: cut the partial record, as the journal does.
            self.cut(good as u64)?;
        }
        Ok(())
    }

    fn cut(&mut self, len: u64) -> Result<(), EmbedError> {
        self.file
            .truncate(len)
            .map_err(|e| io_err("cut a torn record", e))?;
        self.file.sync().map_err(|e| io_err("sync", e))?;
        self.good_len = len;
        Ok(())
    }

    fn note_mismatch(&mut self, have: usize, got: usize) {
        self.mismatch.get_or_insert_with(|| {
            format!(
                "the embedder returned a vector of length {got}, the cache holds length {have}: \
                 vectors of other lengths are not being cached"
            )
        });
    }

    /// Remembers `entries`, which must all be finite and of one length, and
    /// writes them while the file is still trusted. Never fails: the caller
    /// has the vectors either way.
    fn store(&mut self, fingerprint: &str, entries: Vec<(Key, Vec<f32>)>) {
        let Some(first) = entries.first() else {
            return;
        };
        let dims = self.dims.unwrap_or(first.1.len());
        if !self.writable {
            // Memory only.
        } else if let Err(e) = self.write(fingerprint, dims, &entries) {
            self.writable = false;
            self.write_error = Some(e.to_string());
            // Drop the torn tail if we can; if not, the next open will.
            let _ = self.file.truncate(self.good_len);
        }
        self.dims = Some(dims);
        self.vectors.extend(entries);
    }

    fn write(
        &mut self,
        fingerprint: &str,
        dims: usize,
        entries: &[(Key, Vec<f32>)],
    ) -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(64 + entries.len() * (KEY_LEN + dims * 4));
        if self.dims.is_none() {
            buf.extend_from_slice(format!("{MAGIC} {VERSION} {fingerprint} {dims}\n").as_bytes());
        }
        for (key, vector) in entries {
            buf.extend_from_slice(key);
            for x in vector {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        self.file.append(&buf)?;
        self.file.sync()?;
        self.good_len += buf.len() as u64;
        Ok(())
    }
}

impl Embedder for CachedEmbedder {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let keys: Vec<Key> = texts.iter().map(|t| key_of(&self.fingerprint, t)).collect();
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
        // Vectors that must not be cached (non-finite, wrong length): the
        // caller gets them exactly as the embedder produced them.
        let mut uncached: HashMap<Key, Vec<f32>> = HashMap::new();
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
            let first = vectors
                .iter()
                .filter(|v| (1..=MAX_DIMS).contains(&v.len()) && is_finite(v))
                .map(Vec::len)
                .next();
            let dims = cache.dims.or(first);
            if let Some(have) = dims
                && let Some(got) = vectors
                    .iter()
                    .find(|v| !v.is_empty() && is_finite(v) && v.len() != have)
                    .map(Vec::len)
            {
                cache.note_mismatch(have, got);
            }
            let mut entries = Vec::new();
            for (&i, vector) in missing.iter().zip(vectors) {
                if Some(vector.len()) == dims && is_finite(&vector) {
                    entries.push((keys[i], vector));
                } else {
                    uncached.insert(keys[i], vector);
                }
            }
            cache.store(&self.fingerprint, entries);
        }
        Ok(keys
            .iter()
            .map(|k| {
                cache
                    .vectors
                    .get(k)
                    .or_else(|| uncached.get(k))
                    .cloned()
                    .expect("every key is cached or was just computed")
            })
            .collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.inner.embed_query(text)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    /// Vectors derived from the text; records every batch it was asked for.
    /// The texts "nan" and "inf" yield non-finite vectors.
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
                .map(|t| match t.as_str() {
                    "nan" => vec![f32::NAN, 0.5, -1.25],
                    "inf" => vec![1.0, f32::INFINITY, -1.25],
                    "odd" => vec![1.0, 2.0],
                    _ => vec![t.len() as f32, 0.5, -1.25],
                })
                .collect())
        }
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![text.len() as f32, 9.0, 9.0])
        }
    }

    /// A file whose writes and truncations fail on demand. A failing write
    /// lands half of the bytes first, like a full disk would.
    struct Flaky {
        file: File,
        fail_write: Arc<AtomicBool>,
        fail_truncate: Arc<AtomicBool>,
    }

    impl Backing for Flaky {
        fn append(&mut self, bytes: &[u8]) -> std::io::Result<()> {
            if self.fail_write.load(Ordering::SeqCst) {
                self.file.write_all(&bytes[..bytes.len() / 2])?;
                return Err(std::io::Error::other("injected write failure"));
            }
            self.file.write_all(bytes)
        }
        fn truncate(&mut self, len: u64) -> std::io::Result<()> {
            if self.fail_truncate.load(Ordering::SeqCst) {
                return Err(std::io::Error::other("injected truncate failure"));
            }
            self.file.set_len(len)
        }
        fn sync(&mut self) -> std::io::Result<()> {
            self.file.sync_data()
        }
    }

    fn texts(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    fn file_in(dir: &Path) -> std::path::PathBuf {
        dir.join("sub/vectors.bin")
    }

    fn open(dir: &Path, fp: &str, inner: &Arc<Counting>) -> Result<CachedEmbedder, EmbedError> {
        CachedEmbedder::open(file_in(dir), fp, inner.clone())
    }

    fn open_flaky(
        dir: &Path,
        inner: &Arc<Counting>,
    ) -> (CachedEmbedder, Arc<AtomicBool>, Arc<AtomicBool>) {
        let fail_write = Arc::new(AtomicBool::new(false));
        let fail_truncate = Arc::new(AtomicBool::new(false));
        let (w, t) = (fail_write.clone(), fail_truncate.clone());
        let cache = CachedEmbedder::open_with(&file_in(dir), "m", inner.clone(), move |file| {
            Box::new(Flaky {
                file,
                fail_write: w,
                fail_truncate: t,
            })
        })
        .unwrap();
        (cache, fail_write, fail_truncate)
    }

    fn file_len(dir: &Path) -> u64 {
        std::fs::metadata(file_in(dir)).unwrap().len()
    }

    const HEADER_M: &str = "tessar-vectors 2 m 3\n";

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
    fn another_fingerprint_is_not_served_and_starts_afresh() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        open(dir.path(), "f1", &inner)
            .unwrap()
            .embed_documents(&texts(&["x"]))
            .unwrap();
        assert_eq!(inner.embedded(), 1);

        let cache = open(dir.path(), "f2", &inner).unwrap();
        assert!(cache.is_empty());
        cache.embed_documents(&texts(&["x"])).unwrap();
        assert_eq!(inner.embedded(), 2, "the old vector must not be served");
        let bytes = std::fs::read(file_in(dir.path())).unwrap();
        assert!(bytes.starts_with(b"tessar-vectors 2 f2 3\n"));
        assert_eq!(bytes.len(), "tessar-vectors 2 f2 3\n".len() + KEY_LEN + 12);

        // The old fingerprint's records are gone: it starts afresh too.
        assert!(open(dir.path(), "f1", &inner).unwrap().is_empty());
    }

    #[test]
    fn a_version_1_file_counts_as_another_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let mut old = b"tessar-vectors 1 m 3\n".to_vec();
        old.extend_from_slice(&key_of("m", "x"));
        for v in [1.0f32, 0.5, -1.25] {
            old.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(file_in(dir.path()), old).unwrap();

        let inner = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &inner).unwrap();
        assert!(cache.is_empty());
        cache.embed_documents(&texts(&["x"])).unwrap();
        assert_eq!(inner.embedded(), 1);
        let bytes = std::fs::read(file_in(dir.path())).unwrap();
        assert!(bytes.starts_with(HEADER_M.as_bytes()));
    }

    #[test]
    fn a_foreign_file_is_an_error_and_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(file_in(dir.path()), b"hello, this is not a cache\n").unwrap();
        let inner = Arc::new(Counting::default());
        assert!(open(dir.path(), "m", &inner).is_err());
        assert_eq!(
            std::fs::read(file_in(dir.path())).unwrap(),
            b"hello, this is not a cache\n"
        );
    }

    #[test]
    fn a_bad_fingerprint_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        for fp in ["", "a b", "a\nb", &"x".repeat(MAX_FINGERPRINT + 1)] {
            assert!(open(dir.path(), fp, &inner).is_err(), "{fp:?}");
        }
        let long = "x".repeat(10_000);
        let err = open(dir.path(), &long, &inner).err().unwrap();
        assert!(err.0.len() < 300 && err.0.contains("newlines"), "{err}");
    }

    #[test]
    fn a_huge_dims_in_the_header_starts_afresh() {
        for dims in ["18446744073709551615", "4611686018427387904", "65537", "0"] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(dir.path().join("sub")).unwrap();
            let mut bytes = format!("tessar-vectors 2 m {dims}\n").into_bytes();
            bytes.extend_from_slice(&[7u8; 100]);
            std::fs::write(file_in(dir.path()), bytes).unwrap();
            let inner = Arc::new(Counting::default());
            let cache = open(dir.path(), "m", &inner).unwrap();
            assert!(cache.is_empty(), "{dims}");
            cache.embed_documents(&texts(&["x"])).unwrap();
            assert_eq!(open(dir.path(), "m", &inner).unwrap().len(), 1);
        }
    }

    #[test]
    fn an_odd_length_vector_is_returned_and_noted_but_the_file_is_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &inner).unwrap();
        cache.embed_documents(&texts(&["x", "yy"])).unwrap();
        let before = std::fs::read(file_in(dir.path())).unwrap();
        assert_eq!(cache.length_mismatch(), None);

        let got = cache.embed_documents(&texts(&["odd"])).unwrap();
        assert_eq!(got, vec![vec![1.0, 2.0]]);
        assert_eq!(std::fs::read(file_in(dir.path())).unwrap(), before);
        let note = cache.length_mismatch().unwrap();
        assert!(note.contains('2') && note.contains('3'), "{note}");
        assert_eq!(cache.write_error(), None);

        // Stored vectors are still served; matching ones still cached, in a
        // mixed batch and in later ones; the note is recorded once.
        let calls = inner.calls();
        assert_eq!(
            cache.embed_documents(&texts(&["x", "yy"])).unwrap().len(),
            2
        );
        assert_eq!(inner.calls(), calls);
        cache.embed_documents(&texts(&["odd", "zzz"])).unwrap();
        cache.embed_documents(&texts(&["wwww"])).unwrap();
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.length_mismatch().unwrap(), note);
        let again = Arc::new(Counting::default());
        assert_eq!(open(dir.path(), "m", &again).unwrap().len(), 4);
    }

    #[test]
    fn a_length_note_does_not_replace_a_write_error() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let (cache, fail_write, _) = open_flaky(dir.path(), &inner);
        cache.embed_documents(&texts(&["x"])).unwrap();
        fail_write.store(true, Ordering::SeqCst);
        cache.embed_documents(&texts(&["yy"])).unwrap();
        let err = cache.write_error().unwrap();
        cache.embed_documents(&texts(&["odd"])).unwrap();
        assert_eq!(cache.write_error().unwrap(), err);
        assert!(cache.length_mismatch().is_some());
    }

    #[test]
    fn the_first_write_error_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let (cache, fail_write, _) = open_flaky(dir.path(), &inner);
        assert_eq!(cache.write_error(), None);
        fail_write.store(true, Ordering::SeqCst);
        cache.embed_documents(&texts(&["x"])).unwrap();
        let first = cache.write_error().unwrap();
        assert!(first.contains("injected write failure"), "{first}");
        cache.embed_documents(&texts(&["yy"])).unwrap();
        assert_eq!(cache.write_error().unwrap(), first);
    }

    #[test]
    fn a_torn_record_is_cut_off() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        open(dir.path(), "m", &inner)
            .unwrap()
            .embed_documents(&texts(&["x", "yy"]))
            .unwrap();
        let path = file_in(dir.path());
        let full = std::fs::metadata(&path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(full - 5).unwrap();
        drop(file);

        let again = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &again).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            HEADER_M.len() as u64 + (KEY_LEN + 12) as u64
        );
        cache.embed_documents(&texts(&["x", "yy"])).unwrap();
        assert_eq!(again.embedded(), 1);
        let last = Arc::new(Counting::default());
        assert_eq!(open(dir.path(), "m", &last).unwrap().len(), 2);
    }

    #[test]
    fn a_failed_write_still_returns_vectors_and_the_file_stays_readable() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let (cache, fail_write, _) = open_flaky(dir.path(), &inner);
        cache.embed_documents(&texts(&["x"])).unwrap();
        let good = file_len(dir.path());

        fail_write.store(true, Ordering::SeqCst);
        let got = cache.embed_documents(&texts(&["yy", "zzz"])).unwrap();
        assert_eq!(got.iter().map(|v| v[0]).collect::<Vec<_>>(), [2.0, 3.0]);
        // The torn tail was truncated back.
        assert_eq!(file_len(dir.path()), good);

        // No more writes in this process, even once the disk recovers, and
        // the vectors computed meanwhile are served from memory.
        fail_write.store(false, Ordering::SeqCst);
        cache.embed_documents(&texts(&["yy", "q"])).unwrap();
        assert_eq!(inner.embedded(), 4, "x, yy, zzz, q: yy is not re-embedded");
        assert_eq!(file_len(dir.path()), good);

        let again = Arc::new(Counting::default());
        assert_eq!(open(dir.path(), "m", &again).unwrap().len(), 1);
    }

    #[test]
    fn a_failed_write_with_a_failed_truncate_is_repaired_by_the_next_open() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let (cache, fail_write, fail_truncate) = open_flaky(dir.path(), &inner);
        cache.embed_documents(&texts(&["x"])).unwrap();
        let good = file_len(dir.path());

        fail_write.store(true, Ordering::SeqCst);
        fail_truncate.store(true, Ordering::SeqCst);
        let got = cache
            .embed_documents(&texts(&["yy", "zzz", "wwww"]))
            .unwrap();
        assert_eq!(got.len(), 3);
        // One and a half records landed: the file is torn.
        assert!(file_len(dir.path()) > good);
        fail_write.store(false, Ordering::SeqCst);
        cache.embed_documents(&texts(&["q"])).unwrap();
        let torn = file_len(dir.path());

        let again = Arc::new(Counting::default());
        let reopened = open(dir.path(), "m", &again).unwrap();
        assert_eq!(reopened.len(), 2, "x and the one whole record that landed");
        assert!(file_len(dir.path()) < torn);
        reopened
            .embed_documents(&texts(&["x", "yy", "zzz", "wwww"]))
            .unwrap();
        assert_eq!(again.embedded(), 2);
        let last = Arc::new(Counting::default());
        assert_eq!(open(dir.path(), "m", &last).unwrap().len(), 4);
    }

    #[test]
    fn a_failed_first_write_leaves_a_usable_file() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let (cache, fail_write, _) = open_flaky(dir.path(), &inner);
        fail_write.store(true, Ordering::SeqCst);
        assert_eq!(cache.embed_documents(&texts(&["x"])).unwrap().len(), 1);
        assert_eq!(file_len(dir.path()), 0);
        let again = Arc::new(Counting::default());
        let reopened = open(dir.path(), "m", &again).unwrap();
        reopened.embed_documents(&texts(&["x"])).unwrap();
        assert_eq!(open(dir.path(), "m", &again).unwrap().len(), 1);
    }

    #[test]
    fn non_finite_vectors_are_returned_but_never_written() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &inner).unwrap();
        let got = cache
            .embed_documents(&texts(&["nan", "x", "inf", "nan"]))
            .unwrap();
        assert!(got[0][0].is_nan() && got[3][0].is_nan());
        assert_eq!(got[1], vec![1.0, 0.5, -1.25]);
        assert!(got[2][1].is_infinite());
        assert_eq!(cache.len(), 1);
        assert_eq!(file_len(dir.path()), (HEADER_M.len() + KEY_LEN + 12) as u64);
        let again = Arc::new(Counting::default());
        assert_eq!(open(dir.path(), "m", &again).unwrap().len(), 1);
    }

    #[test]
    fn a_batch_of_only_non_finite_vectors_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        let cache = open(dir.path(), "m", &inner).unwrap();
        assert_eq!(cache.embed_documents(&texts(&["nan"])).unwrap().len(), 1);
        assert_eq!(file_len(dir.path()), 0);
        assert!(cache.is_empty());
    }

    #[cfg(unix)]
    fn mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[cfg(unix)]
    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn new_directory_and_file_are_private() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        open(dir.path(), "m", &inner)
            .unwrap()
            .embed_documents(&texts(&["x"]))
            .unwrap();
        assert_eq!(mode(&dir.path().join("sub")), 0o700);
        assert_eq!(mode(&file_in(dir.path())), 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn loose_permissions_are_tightened_and_tight_ones_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(Counting::default());
        open(dir.path(), "m", &inner)
            .unwrap()
            .embed_documents(&texts(&["x"]))
            .unwrap();
        let sub = dir.path().join("sub");
        let file = file_in(dir.path());

        set_mode(&sub, 0o755);
        set_mode(&file, 0o644);
        let cache = open(dir.path(), "m", &inner).unwrap();
        assert_eq!(mode(&sub), 0o700);
        assert_eq!(mode(&file), 0o600);
        assert_eq!(cache.len(), 1, "tightening keeps the contents");

        // Owner bits are never widened or removed beyond group/other.
        set_mode(&file, 0o640);
        open(dir.path(), "m", &inner).unwrap();
        assert_eq!(mode(&file), 0o600);
        set_mode(&sub, 0o500);
        set_mode(&file, 0o600);
        open(dir.path(), "m", &inner).unwrap();
        assert_eq!(mode(&sub), 0o500);
        set_mode(&sub, 0o700);
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_not_tightened() {
        let dir = tempfile::tempdir().unwrap();
        let real_dir = dir.path().join("real");
        std::fs::create_dir(&real_dir).unwrap();
        set_mode(&real_dir, 0o755);
        std::os::unix::fs::symlink(&real_dir, dir.path().join("sub")).unwrap();
        let other = dir.path().join("other.bin");
        std::fs::write(&other, b"").unwrap();
        set_mode(&other, 0o644);
        std::os::unix::fs::symlink(&other, real_dir.join("vectors.bin")).unwrap();
        let inner = Arc::new(Counting::default());
        open(dir.path(), "m", &inner).unwrap();
        assert_eq!(mode(&real_dir), 0o755);
        assert_eq!(mode(&other), 0o644);
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
