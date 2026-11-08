#![cfg(target_os = "macos")]

//! candle vs llama.cpp on the same texts. Ignored: needs the GPU and
//! `~/.tessar/models/voyage-4-nano-candle/` (safetensors) plus
//! `~/.tessar/models/voyage-4-nano-Q8_0.gguf`.
//!
//! ```sh
//! cargo test -p tessar-embed-llama --release --test candle_compare \
//!     -- --ignored --nocapture
//! ```
//!
//! Knobs (environment): `CANDLE_DTYPE` (f16|bf16|f32), `CANDLE_DEVICE`
//! (metal|cpu), `CANDLE_BATCH` (texts per forward pass), `CANDLE_EOS` (1 to
//! append `<|endoftext|>`), `LLAMA_ONLY`/`CANDLE_ONLY` (1).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use tessar_core::embed::{Embedder, Prompts};
use tessar_embed_candle::{Accel, CandleEmbedder, DType, Options};
use tessar_embed_llama::{LlamaEmbedder, Options as LlamaOptions, voyage_nano_path};

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME"))
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// 100 short bilingual texts, 5-30 words, built from parts.
fn texts() -> Vec<String> {
    let en_subj = [
        "The release build of the journal crate",
        "Our weekly planning meeting",
        "The embedding model running on the laptop",
        "A note about the vacation in Georgia",
        "The recall pipeline",
        "Yesterday the author",
        "The new memory space for the project",
        "Every fact stored in the log",
    ];
    let en_pred = [
        "must use the frozen lockfile so that builds stay reproducible across machines",
        "was moved to Thursday because two people were travelling and nobody could join",
        "returns a normalized vector for each text and never calls the network",
        "mentions train tickets, a small hotel near the old town and a long walk by the river",
        "fuses three channels with weighted reciprocal rank fusion and explains every hit",
        "decided to keep the data model small and to measure every change before adding it",
        "keeps validity periods and immutable versions so that older answers can be reproduced",
    ];
    let ru_subj = [
        "Сборка релизной версии журнала",
        "Еженедельная планерка команды",
        "Модель эмбеддингов на ноутбуке",
        "Заметка про отпуск в Грузии",
        "Конвейер поиска по памяти",
        "Вчера автор проекта",
        "Новое пространство памяти для проекта",
        "Каждый факт в журнале",
    ];
    let ru_pred = [
        "должна использовать замороженный lock-файл, чтобы сборки воспроизводились на разных машинах",
        "перенесена на четверг, потому что двое уехали в командировку и не смогли подключиться",
        "возвращает нормализованный вектор для каждого текста и никогда не ходит в сеть",
        "упоминает билеты на поезд, маленький отель у старого города и долгую прогулку вдоль реки",
        "объединяет три канала взвешенным слиянием рангов и объясняет каждый найденный результат",
        "решил оставить модель данных маленькой и измерять каждое изменение перед добавлением",
        "хранит периоды действия и неизменяемые версии, чтобы старые ответы можно было повторить",
    ];
    let mut out = Vec::new();
    for i in 0..50 {
        out.push(format!(
            "{} {}.",
            en_subj[i % 8],
            en_pred[(i * 3 + i / 8) % 7]
        ));
        out.push(format!(
            "{} {}.",
            ru_subj[(i + 1) % 8],
            ru_pred[(i * 5 + i / 8) % 7]
        ));
    }
    out
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
    let n = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i].as_secs_f64() * 1000.0
}

/// (p50, p95) in ms of `n` calls after `warm` warm-up calls.
fn bench(warm: usize, n: usize, mut f: impl FnMut(usize)) -> (f64, f64) {
    for i in 0..warm {
        f(i);
    }
    let mut times: Vec<Duration> = (0..n)
        .map(|i| {
            let t = Instant::now();
            f(i);
            t.elapsed()
        })
        .collect();
    times.sort();
    (pct(&times, 0.5), pct(&times, 0.95))
}

fn report(name: &str, texts: &[String], e: &dyn Embedder) -> Vec<Vec<f32>> {
    let vecs = e.embed_documents(texts).unwrap();
    let q = bench(5, 100, |i| {
        e.embed_query(&texts[i % texts.len()]).unwrap();
    });
    let docs: Vec<String> = texts.iter().take(64).cloned().collect();
    let b = bench(2, 10, |_| {
        e.embed_documents(&docs).unwrap();
    });
    println!(
        "RESULT {name}: query p50 {:.1} ms p95 {:.1} ms | batch64 p50 {:.0} ms p95 {:.0} ms",
        q.0, q.1, b.0, b.1
    );
    vecs
}

#[test]
#[ignore = "needs GPU and model files"]
fn candle_vs_llama() {
    let texts = texts();
    assert_eq!(texts.len(), 100);

    let dtype = match env("CANDLE_DTYPE").as_deref() {
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        _ => DType::F16,
    };
    let options = Options {
        context: 2048,
        batch: env("CANDLE_BATCH").map_or(8, |v| v.parse().unwrap()),
        device: if env("CANDLE_DEVICE").as_deref() == Some("cpu") {
            Accel::Cpu
        } else {
            Accel::Auto
        },
        dtype,
        append_eos: env("CANDLE_EOS").as_deref() == Some("1"),
        fused: env("TESSAR_CANDLE_NO_SDPA").is_none(),
    };

    let mut reference = None;
    if env("CANDLE_ONLY").is_none() {
        let t = Instant::now();
        let llama = LlamaEmbedder::load(
            voyage_nano_path().unwrap(),
            &LlamaOptions {
                context: 2048,
                batch: 64,
                gpu: true,
                truncate: true,
            },
        )
        .unwrap()
        .with_prompts(Prompts::voyage());
        println!("RESULT llama load: {:.2} s", t.elapsed().as_secs_f64());
        reference = Some(report("llama.cpp", &texts, &llama));
        if env("LLAMA_ONLY").is_some() {
            return;
        }
    }

    let t = Instant::now();
    let candle = CandleEmbedder::load(
        &home().join(".tessar/models/voyage-4-nano-candle"),
        &options,
    )
    .unwrap()
    .with_prompts(Prompts::voyage());
    println!(
        "RESULT candle load: {:.2} s (metal={}, {:?}, batch {}, eos {})",
        t.elapsed().as_secs_f64(),
        candle.is_metal(),
        dtype,
        options.batch,
        options.append_eos
    );
    assert_eq!(candle.dims(), 2048);
    let ours = report("candle", &texts, &candle);

    if let Some(reference) = reference {
        let mut c: Vec<f64> = reference
            .iter()
            .zip(&ours)
            .map(|(a, b)| cos(a, b))
            .collect();
        c.sort_by(f64::total_cmp);
        let mean = c.iter().sum::<f64>() / c.len() as f64;
        println!(
            "RESULT cosine: mean {mean:.5} min {:.5} p5 {:.5} max {:.5}",
            c[0],
            c[5],
            c[c.len() - 1]
        );
    }
}
