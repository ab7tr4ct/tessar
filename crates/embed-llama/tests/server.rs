//! `LlamaEmbedder` against `llama-server` from the same llama.cpp
//! release: the same texts must get the same vectors. Ignored by default:
//! it needs the model (see `tests/model.rs`) and a `llama-server` binary.
//!
//! ```text
//! TESSAR_TEST_LLAMA_SERVER=/path/to/llama-server \
//!     cargo test -p tessar-embed-llama --test server -- --ignored --nocapture
//! ```

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tessar_core::embed::{Embedder, Prompts};
use tessar_embed_llama::{LlamaEmbedder, voyage_nano_path};

const TEXTS: [&str; 10] = [
    "Release builds of the journal crate must use the frozen lockfile.",
    "Модели для эмбеддингов лежат в ~/.tessar/models, сервер больше не нужен.",
    "where are the embedding models kept?",
    "куда я лечу в отпуск?",
    "The store cuts vectors to 1024 dimensions, renormalizes them and \
     quantizes them to int8 before they go into the index.",
    "Решили: канал смысла включён по умолчанию, лексика с весом 0.1, \
     ближнее пространство получает бонус 0.05.",
    "ok",
    "Mixed: тест на смешанный текст with English and русские слова, 2026-11-08.",
    "fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>",
    "Пользователь предпочитает короткие коммиты на английском; заметки — \
     на русском. Код и документация публичного репозитория — на английском.",
];

/// Kills the server when the test ends, also on panic.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[derive(Deserialize)]
struct Response {
    data: Vec<Item>,
}

#[derive(Deserialize)]
struct Item {
    index: usize,
    embedding: Vec<f32>,
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
#[ignore = "needs the model and a llama-server binary; see the module docs"]
fn vectors_match_llama_server() {
    let binary = std::env::var_os("TESSAR_TEST_LLAMA_SERVER")
        .map(PathBuf::from)
        .expect("set TESSAR_TEST_LLAMA_SERVER to a llama-server of the pinned release");
    let model = std::env::var_os("TESSAR_TEST_MODEL")
        .map(PathBuf::from)
        .or_else(voyage_nano_path)
        .expect("set TESSAR_TEST_MODEL or HOME");

    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    // The setup the earlier measurements used.
    let _server = Server(
        Command::new(&binary)
            .arg("-m")
            .arg(&model)
            .args(["--embedding", "--port", &port.to_string()])
            .args(["-ngl", "99", "-c", "65536", "-b", "8192", "-ub", "8192"])
            .args(["--parallel", "8"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| panic!("cannot start {}: {e}", binary.display())),
    );
    let base = format!("http://127.0.0.1:{port}");
    let started = Instant::now();
    while ureq::get(&format!("{base}/health")).call().is_err() {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "llama-server did not come up"
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    let native = LlamaEmbedder::voyage_nano(&model).unwrap();
    let prompts = Prompts::voyage();
    let texts: Vec<String> = TEXTS.iter().map(|t| (*t).to_owned()).collect();
    let mut cosines = Vec::new();
    for (kind, prefix) in [("document", &prompts.document), ("query", &prompts.query)] {
        let input: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        let mut response = ureq::post(&format!("{base}/v1/embeddings"))
            .send_json(serde_json::json!({ "input": input }))
            .unwrap();
        let mut served: Vec<Item> = response.body_mut().read_json::<Response>().unwrap().data;
        served.sort_by_key(|item| item.index);
        let ours = if kind == "document" {
            native.embed_documents(&texts).unwrap()
        } else {
            texts
                .iter()
                .map(|t| native.embed_query(t).unwrap())
                .collect()
        };
        for (i, (a, b)) in ours.iter().zip(&served).enumerate() {
            assert_eq!(a.len(), b.embedding.len());
            let c = cos(a, &b.embedding);
            println!("{kind:<8} {i:>2}  cosine {c:.6}");
            cosines.push(c);
        }
    }
    let min = cosines.iter().copied().fold(f32::INFINITY, f32::min);
    let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
    println!(
        "{} vectors: min cosine {min:.6}, mean {mean:.6}",
        cosines.len()
    );
    assert!(min > 0.999, "min cosine {min}");
}
