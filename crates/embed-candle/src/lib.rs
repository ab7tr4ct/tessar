//! Experimental pure-Rust embedder for `voyage-4-nano` on candle: no cmake,
//! no C++. See `EXPERIMENT.md`.

mod model;

pub use model::Joins;

use std::path::Path;
use std::sync::Mutex;

pub use candle_core::DType;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use tessar_core::embed::{EmbedError, Embedder, Prompts};
use tokenizers::{Tokenizer, TruncationParams};

/// Where the weights run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accel {
    /// Metal when the crate is built with the `metal` feature, else CPU.
    /// Decided at compile time: a Metal failure is an error, not a fallback.
    Auto,
    /// Always the CPU.
    Cpu,
}

/// How to load the model.
#[derive(Debug, Clone)]
pub struct Options {
    /// Token budget of one forward pass, and the most one text may take
    /// (prompt included; a longer text is cut). A pass holds as many
    /// texts as fit in `context` tokens when padded to the longest one,
    /// so memory stays bounded however many texts a call brings. The
    /// non-fused attention path (CPU) materializes the scores, so one text
    /// of 8192 tokens costs about `heads * len^2 * 4` bytes, ~4.3 GB in f32.
    pub context: usize,
    /// Most texts in one forward pass.
    pub batch: usize,
    /// Device choice.
    pub device: Accel,
    /// Compute dtype (the weights are bf16 on disk and converted on load).
    /// Ignored by `load_gguf`, which runs quantized weights on f32 activations.
    pub dtype: DType,
    /// Append `<|endoftext|>` to every text. The HF tokenizer does not.
    pub append_eos: bool,
    /// Use the fused Metal attention kernel (no effect on the CPU).
    pub fused: bool,
    /// `load_gguf` only: keep the activations between the quantized matmuls
    /// in f16 instead of f32 (casts happen around each matmul).
    pub f16_activations: bool,
    /// Join the q/k/v and gate/up projections into single layers. Exact,
    /// but slower on Metal in candle 0.11 (see `EXPERIMENT.md`).
    pub joins: Joins,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            context: 8192,
            batch: 8,
            device: Accel::Auto,
            dtype: DType::F16,
            append_eos: false,
            fused: true,
            f16_activations: false,
            joins: Joins::default(),
        }
    }
}

fn err(e: impl std::fmt::Display) -> EmbedError {
    EmbedError(format!("candle: {e}"))
}

#[cfg(feature = "metal")]
fn auto_device() -> Result<Device, EmbedError> {
    Device::new_metal(0).map_err(err)
}

#[cfg(not(feature = "metal"))]
#[allow(clippy::unnecessary_wraps)]
fn auto_device() -> Result<Device, EmbedError> {
    Ok(Device::Cpu)
}

fn pick_device(accel: Accel) -> Result<Device, EmbedError> {
    match accel {
        Accel::Cpu => Ok(Device::Cpu),
        Accel::Auto => auto_device(),
    }
}

/// L2-normalizes `row` in place. A zero vector stays zero; a vector with
/// NaN or infinite parts is an error, never an index entry.
fn normalize(row: &mut [f32]) -> Result<(), EmbedError> {
    let norm = row
        .iter()
        .map(|x| f64::from(*x).powi(2))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() {
        return Err(EmbedError("candle: non-finite vector".into()));
    }
    let scale = if norm > 0.0 { (1.0 / norm) as f32 } else { 0.0 };
    for x in row {
        *x *= scale;
    }
    Ok(())
}

/// Splits `order` (indices sorted by ascending length) into passes of at
/// most `batch` texts whose padded size fits the `context` token budget.
/// A text longer than the budget gets a pass of its own.
fn plan<'a>(order: &'a [usize], lens: &[usize], batch: usize, context: usize) -> Vec<&'a [usize]> {
    let mut passes = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        // Ascending lengths: the newest text is the longest.
        while end < order.len()
            && end - start < batch.max(1)
            && (end - start + 1) * lens[order[end]] <= context
        {
            end += 1;
        }
        passes.push(&order[start..end]);
        start = end;
    }
    passes
}

/// `voyage-4-nano` on candle.
pub struct CandleEmbedder {
    // Serializes use of the GPU only: `Model::forward` takes `&self`.
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
    pub fn load(model_dir: impl AsRef<Path>, options: &Options) -> Result<Self, EmbedError> {
        let dir = model_dir.as_ref();
        let cfg: model::Config =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).map_err(err)?)
                .map_err(err)?;
        cfg.check().map_err(err)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(err)?;
        let device = pick_device(options.device)?;
        // Buffered read instead of mmap: mmap needs `unsafe` in candle's API.
        let bytes = std::fs::read(dir.join("model.safetensors")).map_err(err)?;
        let vb =
            VarBuilder::from_buffered_safetensors(bytes, options.dtype, &device).map_err(err)?;
        let model = model::Model::load(&cfg, vb, options.fused, options.joins).map_err(err)?;
        Self::assemble(model, tokenizer, device, options)
    }

    /// Loads the quantized GGUF at `gguf` (the file llama.cpp uses) and the
    /// tokenizer from `tokenizer_json`. Linear layers stay quantized and run
    /// through `QMatMul`; norms are dequantized to f32.
    pub fn load_gguf(
        gguf: impl AsRef<Path>,
        tokenizer_json: impl AsRef<Path>,
        options: &Options,
    ) -> Result<Self, EmbedError> {
        let tokenizer = Tokenizer::from_file(tokenizer_json.as_ref()).map_err(err)?;
        let device = pick_device(options.device)?;
        let file = std::fs::File::open(gguf.as_ref()).map_err(err)?;
        let act = if options.f16_activations {
            DType::F16
        } else {
            DType::F32
        };
        let model = model::Model::load_gguf(file, &device, options.fused, options.joins, act)
            .map_err(err)?;
        Self::assemble(model, tokenizer, device, options)
    }

    fn assemble(
        model: model::Model,
        mut tokenizer: Tokenizer,
        device: Device,
        options: &Options,
    ) -> Result<Self, EmbedError> {
        // The id is only used for `append_eos`; padding is masked and uses 0.
        let eos = match tokenizer.token_to_id("<|endoftext|>") {
            Some(id) => id,
            None if options.append_eos => {
                return Err(err("the tokenizer has no <|endoftext|> token"));
            }
            None => 0,
        };
        let reserved = usize::from(options.append_eos);
        if options.context <= reserved {
            return Err(err(format!(
                "context {} leaves no room for text",
                options.context
            )));
        }
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: options.context - reserved,
                ..TruncationParams::default()
            }))
            .map_err(err)?;
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
    pub fn voyage_nano(model_dir: impl AsRef<Path>) -> Result<Self, EmbedError> {
        Ok(Self::load(model_dir, &Options::default())?.with_prompts(Prompts::voyage()))
    }

    /// Sets the instructions before queries and documents.
    #[must_use]
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
        // The tokenizer cuts to `context` minus the EOS slot.
        let enc = self.tokenizer.encode(text, false).map_err(err)?;
        let mut ids = enc.get_ids().to_vec();
        if self.options.append_eos {
            ids.push(self.eos);
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
        let lens: Vec<usize> = tokens.iter().map(Vec::len).collect();
        // Sort by length so passes pad little; order is restored below.
        let mut order: Vec<usize> = (0..tokens.len()).collect();
        order.sort_by_key(|&i| lens[i]);
        let mut out = vec![Vec::new(); texts.len()];
        let model = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for chunk in plan(&order, &lens, self.options.batch, self.options.context) {
            let len = chunk.iter().map(|&i| lens[i]).max().unwrap_or(1);
            let mut ids = Vec::with_capacity(chunk.len() * len);
            let mut valid = Vec::with_capacity(chunk.len() * len);
            for &i in chunk {
                let t = std::mem::take(&mut tokens[i]);
                valid.extend((0..len).map(|p| if p < t.len() { 1f32 } else { 0.0 }));
                let pad = len - t.len();
                ids.extend(t);
                // Padding is masked out, any valid id will do.
                ids.extend(std::iter::repeat_n(0u32, pad));
            }
            let valid = Tensor::from_vec(valid, (chunk.len(), len), &self.device).map_err(err)?;
            let v = model.forward(&ids, chunk.len(), len, &valid).map_err(err)?;
            let v = v.to_vec2::<f32>().map_err(err)?;
            for (&i, mut row) in chunk.iter().zip(v) {
                normalize(&mut row)?;
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
        let e = CandleEmbedder::load("/nonexistent/tessar-model", &Options::default()).unwrap_err();
        assert!(e.0.starts_with("candle:"), "{e}");
    }

    #[test]
    fn plan_respects_batch_and_token_budget() {
        let lens = [10, 10, 10, 10, 50, 50, 300];
        let order: Vec<usize> = (0..lens.len()).collect();
        let sizes = |batch, context| -> Vec<usize> {
            plan(&order, &lens, batch, context)
                .iter()
                .map(|c| c.len())
                .collect()
        };
        assert_eq!(sizes(8, 1000), vec![6, 1]);
        assert_eq!(sizes(2, 1000), vec![2, 2, 2, 1]);
        // 4 x 10 fits in 40 tokens; 2 x 50 does not fit in 60.
        assert_eq!(sizes(8, 60), vec![4, 1, 1, 1]);
    }

    #[test]
    fn normalize_scales_to_unit_length() {
        let mut row = [3.0f32, 4.0];
        normalize(&mut row).unwrap();
        assert!((row[0] - 0.6).abs() < 1e-6 && (row[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_keeps_zero_and_rejects_non_finite() {
        let mut zero = [0.0f32; 3];
        normalize(&mut zero).unwrap();
        assert_eq!(zero, [0.0; 3]);
        let mut nan = [1.0f32, f32::NAN];
        assert!(normalize(&mut nan).is_err());
        let mut inf = [1.0f32, f32::INFINITY];
        assert!(normalize(&mut inf).is_err());
    }
}
