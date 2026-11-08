//! Experimental pure-Rust embedder for `voyage-4-nano` on candle: no cmake,
//! no C++. See `EXPERIMENT.md`.

mod model;

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tessar_core::embed::{EmbedError, Embedder, Prompts};
use tokenizers::Tokenizer;

/// Where the weights run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accel {
    /// Metal when the crate is built with the `metal` feature, else CPU.
    Auto,
    /// Always the CPU.
    Cpu,
}

/// How to load the model.
#[derive(Debug, Clone)]
pub struct Options {
    /// Tokens per text, prompt included; longer texts are cut.
    pub context: usize,
    /// Most texts per forward pass.
    pub batch: usize,
    /// Device choice.
    pub device: Accel,
    /// Compute dtype (weights are bf16 on disk and converted on load).
    pub dtype: DType,
    /// Append `<|endoftext|>` to every text. The HF tokenizer does not.
    pub append_eos: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            context: 8192,
            batch: 8,
            device: Accel::Auto,
            dtype: DType::F16,
            append_eos: false,
        }
    }
}

fn err(e: impl std::fmt::Display) -> EmbedError {
    EmbedError(format!("candle: {e}"))
}

/// `voyage-4-nano` on candle.
pub struct CandleEmbedder {
    model: Mutex<model::Model>,
    tokenizer: Tokenizer,
    device: Device,
    prompts: Prompts,
    options: Options,
    eos: u32,
    dims: usize,
}

impl CandleEmbedder {
    /// Loads `config.json`, `tokenizer.json` and `model.safetensors`
    /// from `model_dir`, without prompts.
    pub fn load(model_dir: &Path, options: &Options) -> Result<Self, EmbedError> {
        let cfg: model::Config =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json")).map_err(err)?)
                .map_err(err)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(err)?;
        let eos = tokenizer.token_to_id("<|endoftext|>").unwrap_or(151_643);
        let device = match options.device {
            #[cfg(feature = "metal")]
            Accel::Auto => Device::new_metal(0).map_err(err)?,
            _ => Device::Cpu,
        };
        // Buffered read instead of mmap: mmap needs `unsafe` in candle's API.
        let bytes = std::fs::read(model_dir.join("model.safetensors")).map_err(err)?;
        let vb =
            VarBuilder::from_buffered_safetensors(bytes, options.dtype, &device).map_err(err)?;
        let model = model::Model::load(&cfg, vb, &device).map_err(err)?;
        Ok(Self {
            dims: model.dims,
            model: Mutex::new(model),
            tokenizer,
            device,
            prompts: Prompts::none(),
            options: options.clone(),
            eos,
        })
    }

    /// `voyage-4-nano` in `model_dir` with its prompts.
    pub fn voyage_nano(model_dir: &Path) -> Result<Self, EmbedError> {
        Ok(Self::load(model_dir, &Options::default())?.with_prompts(Prompts::voyage()))
    }

    /// Sets the instructions before queries and documents.
    pub fn with_prompts(mut self, prompts: Prompts) -> Self {
        self.prompts = prompts;
        self
    }

    /// The length of every vector.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Whether the model runs on Metal.
    pub fn is_metal(&self) -> bool {
        self.device.is_metal()
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, EmbedError> {
        let enc = self.tokenizer.encode(text, false).map_err(err)?;
        let mut ids = enc.get_ids().to_vec();
        let max = self.options.context.max(1);
        if self.options.append_eos {
            ids.truncate(max - 1);
            ids.push(self.eos);
        } else {
            ids.truncate(max);
        }
        if ids.is_empty() {
            return Err(EmbedError("candle: text has no tokens".into()));
        }
        Ok(ids)
    }

    fn embed(&self, prefix: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut tokens = texts
            .iter()
            .map(|t| self.encode(&format!("{prefix}{t}")))
            .collect::<Result<Vec<_>, _>>()?;
        // Sort by length so batches pad little; order is restored below.
        let mut order: Vec<usize> = (0..tokens.len()).collect();
        order.sort_by_key(|&i| tokens[i].len());
        let mut out = vec![Vec::new(); texts.len()];
        let model = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for chunk in order.chunks(self.options.batch.max(1)) {
            let len = chunk.iter().map(|&i| tokens[i].len()).max().unwrap_or(1);
            let mut ids = Vec::with_capacity(chunk.len() * len);
            let mut valid = Vec::with_capacity(chunk.len() * len);
            for &i in chunk {
                let t = std::mem::take(&mut tokens[i]);
                valid.extend((0..len).map(|p| if p < t.len() { 1f32 } else { 0.0 }));
                let pad = len - t.len();
                ids.extend(t);
                ids.extend(std::iter::repeat_n(self.eos, pad));
            }
            let ids = Tensor::from_vec(ids, (chunk.len(), len), &self.device).map_err(err)?;
            let valid = Tensor::from_vec(valid, (chunk.len(), len), &self.device).map_err(err)?;
            let v = model.forward(&ids, &valid).map_err(err)?;
            let v = v.to_vec2::<f32>().map_err(err)?;
            for (&i, mut row) in chunk.iter().zip(v) {
                let norm = row
                    .iter()
                    .map(|x| f64::from(*x).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let scale = if norm > 0.0 { (1.0 / norm) as f32 } else { 0.0 };
                row.iter_mut().for_each(|x| *x *= scale);
                out[i] = row;
            }
        }
        Ok(out)
    }
}

impl Embedder for CandleEmbedder {
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.embed(&self.prompts.document, texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed(&self.prompts.query, &[text.to_owned()])?
            .pop()
            .ok_or_else(|| EmbedError("candle: no vector".into()))
    }
}

impl std::fmt::Debug for CandleEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleEmbedder")
            .field("dims", &self.dims)
            .field("device", &self.device.location())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_dir_is_an_error() {
        let e = CandleEmbedder::load(Path::new("/nonexistent/tessar-model"), &Options::default())
            .unwrap_err();
        assert!(e.0.starts_with("candle:"), "{e}");
    }
}
