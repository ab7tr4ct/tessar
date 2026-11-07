//! Loading failures; these need no model.

use std::io::Write;

use tessar_embed_llama::{LlamaEmbedder, Options};

#[test]
fn a_missing_file_is_an_error() {
    let err = LlamaEmbedder::load("/nonexistent/model.gguf", &Options::default()).unwrap_err();
    assert!(err.0.contains("no such model file"), "{err}");
}

#[test]
fn a_file_that_is_not_gguf_is_an_error() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(b"definitely not a GGUF model").unwrap();
    let err = LlamaEmbedder::load(file.path(), &Options::default()).unwrap_err();
    assert!(err.0.contains("cannot load the model"), "{err}");
}

#[test]
fn an_empty_context_is_an_error() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let options = Options {
        context: 0,
        ..Options::default()
    };
    let err = LlamaEmbedder::load(file.path(), &options).unwrap_err();
    assert!(err.0.contains("at least one token"), "{err}");
}
