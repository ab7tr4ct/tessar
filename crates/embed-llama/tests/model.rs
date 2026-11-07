//! `LlamaEmbedder` with the real model. Ignored by default: they need
//! `voyage-4-nano-Q8_0.gguf` at `$TESSAR_TEST_MODEL` or, without it, at
//! `~/.tessar/models/`. Run with
//! `cargo test -p tessar-embed-llama -- --ignored`.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::thread;

use tessar_core::embed::{Embedder, Prompts};
use tessar_embed_llama::{LlamaEmbedder, voyage_nano_path};

fn model_path() -> PathBuf {
    std::env::var_os("TESSAR_TEST_MODEL")
        .map(PathBuf::from)
        .or_else(voyage_nano_path)
        .expect("set TESSAR_TEST_MODEL or HOME")
}

/// One model for all tests: loading it per test would put several
/// copies on the GPU at once. Being a static, it is never dropped, which
/// also checks that a process exits cleanly with a live embedder.
fn embedder() -> &'static LlamaEmbedder {
    static EMBEDDER: OnceLock<LlamaEmbedder> = OnceLock::new();
    EMBEDDER.get_or_init(|| {
        let path = model_path();
        LlamaEmbedder::voyage_nano(&path)
            .unwrap_or_else(|e| panic!("cannot load {}: {e}", path.display()))
    })
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

fn texts(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

const NOTES: [&str; 4] = [
    "Release builds of the journal crate must use the frozen lockfile.",
    "Модели для эмбеддингов лежат в ~/.tessar/models, сервер больше не нужен.",
    "The cat sleeps on the windowsill every afternoon.",
    "Отпуск в марте: билеты в Тбилиси уже куплены.",
];

#[test]
#[ignore = "needs the model; see the module docs"]
fn vectors_have_the_model_output_length_and_unit_norm() {
    let e = embedder();
    assert_eq!(e.dims(), 2048);
    let vectors = e.embed_documents(&texts(&NOTES)).unwrap();
    assert_eq!(vectors.len(), NOTES.len());
    for v in &vectors {
        assert_eq!(v.len(), 2048);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
    }
    assert_eq!(e.embed_query("anything").unwrap().len(), 2048);
}

#[test]
#[ignore = "needs the model; see the module docs"]
fn no_texts_give_no_vectors() {
    assert!(embedder().embed_documents(&[]).unwrap().is_empty());
}

#[test]
#[ignore = "needs the model; see the module docs"]
fn a_text_alone_and_in_a_batch_gets_the_same_vector() {
    let e = embedder();
    let batch = e.embed_documents(&texts(&NOTES)).unwrap();
    for (i, note) in NOTES.iter().enumerate() {
        let alone = e.embed_documents(&texts(&[note])).unwrap().remove(0);
        let c = cos(&alone, &batch[i]);
        assert!(c > 0.9999, "text {i}: cosine {c}");
    }
}

#[test]
#[ignore = "needs the model; see the module docs"]
fn queries_and_documents_get_their_own_prompts() {
    let e = embedder();
    let text = "where are the embedding models kept";
    let query = e.embed_query(text).unwrap();
    let document = e.embed_documents(&texts(&[text])).unwrap().remove(0);
    let c = cos(&query, &document);
    assert!(c < 0.999, "the prompts changed nothing: cosine {c}");
    // Without prompts both sides embed the same string.
    let plain = LlamaEmbedder::voyage_nano(model_path())
        .unwrap()
        .with_prompts(Prompts::none());
    let q = plain.embed_query(text).unwrap();
    let d = plain.embed_documents(&texts(&[text])).unwrap().remove(0);
    assert!(cos(&q, &d) > 0.9999);
}

#[test]
#[ignore = "needs the model; see the module docs"]
fn queries_find_their_notes_in_russian_and_english() {
    let e = embedder();
    let notes = e.embed_documents(&texts(&NOTES)).unwrap();
    let cases = [
        ("which lockfile do release builds use?", 0),
        ("где лежат модели для эмбеддингов?", 1),
        ("where does the cat nap?", 2),
        ("куда я лечу в отпуск?", 3),
        // Across languages.
        ("where are the embedding models stored?", 1),
        ("что делает кот днём?", 2),
    ];
    for (query, want) in cases {
        let q = e.embed_query(query).unwrap();
        let best = (0..notes.len())
            .max_by(|&a, &b| cos(&q, &notes[a]).total_cmp(&cos(&q, &notes[b])))
            .unwrap();
        assert_eq!(best, want, "{query}");
    }
}

#[test]
#[ignore = "needs the model; see the module docs"]
fn a_text_longer_than_the_context_is_an_error() {
    let long = "memory ".repeat(20_000);
    let err = embedder().embed_documents(&[long]).unwrap_err();
    assert!(err.0.contains("the context holds"), "{err}");
}

#[test]
#[ignore = "needs the model; see the module docs"]
fn many_texts_span_several_batches() {
    let e = embedder();
    // More texts than one decode takes (256 sequences).
    let many: Vec<String> = (0..300).map(|i| format!("note number {i}")).collect();
    let vectors = e.embed_documents(&many).unwrap();
    assert_eq!(vectors.len(), 300);
    let last = e.embed_documents(&many[299..]).unwrap().remove(0);
    assert!(cos(&last, &vectors[299]) > 0.9999);
}

#[test]
#[ignore = "needs the model; see the module docs"]
fn threads_share_one_embedder() {
    let e = embedder();
    let want = e.embed_query(NOTES[0]).unwrap();
    thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| s.spawn(|| e.embed_query(NOTES[0]).unwrap()))
            .collect();
        for h in handles {
            assert!(cos(&h.join().unwrap(), &want) > 0.9999);
        }
    });
}
