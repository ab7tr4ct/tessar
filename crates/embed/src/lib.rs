//! Embedders for tessar.
//!
//! [`HttpEmbedder`] talks to any OpenAI-compatible `/v1/embeddings`
//! endpoint: a `llama-server --embedding` — on this machine or another
//! one with a GPU, ~15 ms per query locally — or an API such as
//! OpenRouter, which is good for bulk indexing but too slow for queries
//! (300–700 ms). The default embedder runs the model inside the process
//! instead: `tessar-embed-llama`.
//!
//! Running `llama-server` for the default model:
//!
//! ```text
//! llama-server -m voyage-4-nano-Q8_0.gguf --embedding --port 8095 \
//!     -ngl 99 -c 65536 -b 8192 -ub 8192 --parallel 8
//! ```
//!
//! `--parallel` lets it embed a batch at once; the context is split
//! between the slots, so each slot must fit the longest text.

use std::time::Duration;

use serde::Deserialize;
pub use tessar_core::embed::Prompts;
use tessar_core::embed::{EmbedError, Embedder};

/// An embedder behind an OpenAI-compatible `/v1/embeddings` endpoint.
#[derive(Debug, Clone)]
pub struct HttpEmbedder {
    agent: ureq::Agent,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    prompts: Prompts,
    batch: usize,
}

impl HttpEmbedder {
    /// An embedder at `base_url` (`http://127.0.0.1:8095`, or
    /// `https://openrouter.ai/api` for OpenRouter), without prompts, in
    /// batches of 32, with a 60 s timeout.
    pub fn new(base_url: &str, model: impl Into<String>) -> Self {
        Self {
            agent: agent(Duration::from_secs(60)),
            endpoint: format!("{}/v1/embeddings", base_url.trim_end_matches('/')),
            model: model.into(),
            api_key: None,
            prompts: Prompts::none(),
            batch: 32,
        }
    }

    /// The default model, `voyage-4-nano`, on a local `llama-server`.
    pub fn voyage_nano(base_url: &str) -> Self {
        Self::new(base_url, "voyage-4-nano").with_prompts(Prompts::voyage())
    }

    /// Sets the instructions before queries and documents.
    pub fn with_prompts(mut self, prompts: Prompts) -> Self {
        self.prompts = prompts;
        self
    }

    /// Sends `Authorization: Bearer <key>`.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// How many texts go in one request; at least 1.
    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch.max(1);
        self
    }

    /// The timeout of one request.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.agent = agent(timeout);
        self
    }

    fn embed(&self, prefix: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch) {
            let input: Vec<String> = chunk.iter().map(|t| format!("{prefix}{t}")).collect();
            out.extend(self.request(&input)?);
        }
        Ok(out)
    }

    /// One request; vectors in input order.
    fn request(&self, input: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut request = self.agent.post(&self.endpoint);
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        let mut response = request
            .send_json(serde_json::json!({ "model": self.model, "input": input }))
            .map_err(|e| EmbedError(format!("{}: {e}", self.endpoint)))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.body_mut().read_to_string().unwrap_or_default();
            let body: String = body.chars().take(300).collect();
            return Err(EmbedError(format!(
                "{}: HTTP {status}: {body}",
                self.endpoint
            )));
        }
        let body: Response = response
            .body_mut()
            .read_json()
            .map_err(|e| EmbedError(format!("{}: bad response: {e}", self.endpoint)))?;
        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; input.len()];
        for item in body.data {
            let slot = vectors.get_mut(item.index).ok_or_else(|| {
                EmbedError(format!(
                    "{}: index {} out of range",
                    self.endpoint, item.index
                ))
            })?;
            *slot = Some(item.embedding);
        }
        vectors
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                v.ok_or_else(|| EmbedError(format!("{}: no vector for input {i}", self.endpoint)))
            })
            .collect()
    }
}

impl Embedder for HttpEmbedder {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.embed(&self.prompts.document, texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut vectors = self.embed(&self.prompts.query, &[text.to_owned()])?;
        vectors
            .pop()
            .ok_or_else(|| EmbedError(format!("{}: no vector", self.endpoint)))
    }
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into()
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
