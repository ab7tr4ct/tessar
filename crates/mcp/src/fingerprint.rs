//! The identity of the vector cache: a short string that changes whenever
//! something that shapes the vectors changes (see `tessar_embed_cache`).

use std::fmt::Write as _;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use sha2::{Digest, Sha256};

/// What shapes the vectors of one embedder.
#[derive(Debug, Clone, Copy)]
pub struct Identity<'a> {
    /// The backend, e.g. `candle` or `llama.cpp`.
    pub backend: &'a str,
    /// The model file.
    pub model: &'a Path,
    /// The tokenizer file, for backends that read one apart from the model.
    pub tokenizer: Option<&'a Path>,
    /// Prompts, token limit, pooling and the like, as one stable string.
    pub settings: &'a str,
}

/// `<name>-<32 hex digits>`: short enough for the cache (at most 256 bytes,
/// no spaces), and the digest covers the [`Identity`]. The model file is
/// identified by size and modification time and not by content: it is
/// hundreds of megabytes, and hashing it would add seconds to every daemon
/// start, while replacing it with a file of the same size and the same
/// modification time is not something a download or a copy does. The
/// tokenizer is small, so its content is hashed. A file that cannot be read
/// is an error; the caller then goes without the cache.
pub fn fingerprint(name: &str, identity: &Identity<'_>) -> io::Result<String> {
    let model = std::fs::metadata(identity.model)?;
    let modified = model
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut digest = Sha256::new();
    field(&mut digest, "backend", identity.backend);
    field(&mut digest, "model", &format!("{}:{modified}", model.len()));
    let tokenizer = match identity.tokenizer {
        Some(path) => hash_file(path)?,
        None => String::new(),
    };
    field(&mut digest, "tokenizer", &tokenizer);
    field(&mut digest, "settings", identity.settings);
    let mut hex = String::new();
    for byte in &digest.finalize()[..16] {
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(format!("{name}-{hex}"))
}

/// Where the vectors of `fingerprint` live under `home`. Every fingerprint
/// has its own file, so two setups never empty each other's.
pub fn cache_path(home: &Path, fingerprint: &str) -> PathBuf {
    home.join("index").join(fingerprint).join("vectors.bin")
}

/// Length-prefixed, so no two sets of fields feed the same bytes.
fn field(digest: &mut Sha256, label: &str, value: &str) {
    digest.update(format!("{label}={}:{value}\n", value.len()));
}

fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(digest.finalize().iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tessar_core::embed::{EmbedError, Embedder};
    use tessar_embed_cache::CachedEmbedder;

    use super::*;

    struct Fixed;
    impl Embedder for Fixed {
        fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Ok(texts.iter().map(|_| vec![1.0]).collect())
        }
        fn embed_query(&self, _: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![1.0])
        }
    }

    struct Files {
        _dir: tempfile::TempDir,
        model: PathBuf,
        tokenizer: PathBuf,
    }

    fn files() -> Files {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        let tokenizer = dir.path().join("tokenizer.json");
        std::fs::write(&model, b"weights").unwrap();
        std::fs::write(&tokenizer, b"{}").unwrap();
        Files {
            _dir: dir,
            model,
            tokenizer,
        }
    }

    fn fp(f: &Files, backend: &str, settings: &str) -> io::Result<String> {
        fingerprint(
            "voyage-4-nano",
            &Identity {
                backend,
                model: &f.model,
                tokenizer: Some(&f.tokenizer),
                settings,
            },
        )
    }

    #[test]
    fn stable_for_unchanged_inputs_and_accepted_by_the_cache() {
        let f = files();
        let a = fp(&f, "candle", "s").unwrap();
        assert_eq!(a, fp(&f, "candle", "s").unwrap());
        assert!(a.starts_with("voyage-4-nano-") && a.len() < 64, "{a}");
        // The cache's own validation.
        let dir = tempfile::tempdir().unwrap();
        let path = cache_path(dir.path(), &a);
        CachedEmbedder::open(&path, &a, Arc::new(Fixed)).unwrap();
        assert!(path.is_file());
        assert!(
            path.to_string_lossy()
                .contains(a.rsplit('-').next().unwrap())
        );
        assert_eq!(path, dir.path().join("index").join(&a).join("vectors.bin"));
    }

    #[test]
    fn changes_with_the_model_the_tokenizer_the_backend_and_the_settings() {
        let f = files();
        let base = fp(&f, "candle", "s").unwrap();
        assert_ne!(base, fp(&f, "llama.cpp", "s").unwrap());
        assert_ne!(base, fp(&f, "candle", "t").unwrap());

        std::fs::write(&f.tokenizer, b"{ }").unwrap();
        let other_tokenizer = fp(&f, "candle", "s").unwrap();
        assert_ne!(base, other_tokenizer);

        // A different size is enough for the model.
        std::fs::write(&f.model, b"other weights").unwrap();
        assert_ne!(other_tokenizer, fp(&f, "candle", "s").unwrap());
    }

    #[test]
    fn a_new_modification_time_changes_it() {
        let f = files();
        let before = fp(&f, "candle", "s").unwrap();
        let file = std::fs::File::options().write(true).open(&f.model).unwrap();
        file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5))
            .unwrap();
        assert_ne!(before, fp(&f, "candle", "s").unwrap());
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_panic() {
        let f = files();
        std::fs::remove_file(&f.tokenizer).unwrap();
        assert!(fp(&f, "candle", "s").is_err());
        std::fs::remove_file(&f.model).unwrap();
        let no_tokenizer = Identity {
            backend: "llama.cpp",
            model: &f.model,
            tokenizer: None,
            settings: "s",
        };
        assert!(fingerprint("m", &no_tokenizer).is_err());
    }

    #[test]
    fn no_tokenizer_is_fine() {
        let f = files();
        let id = Identity {
            backend: "llama.cpp",
            model: &f.model,
            tokenizer: None,
            settings: "s",
        };
        assert!(fingerprint("m", &id).is_ok());
    }
}
