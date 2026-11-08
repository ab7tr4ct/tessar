//! `voyage-4-nano`: Qwen3 with bidirectional attention, followed by a
//! bias-free linear projection (1024 -> 2048). candle-transformers' own
//! `qwen3` is causal and has a KV cache, so this is a small re-implementation
//! of the encoder path only. Weights come from HF safetensors (dense) or
//! from the llama.cpp GGUF (Q8_0, `QMatMul`).

use std::fs::File;

use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::rotary_emb::rope_thd;
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub num_labels: usize,
    pub use_bidirectional_attention: Option<bool>,
    pub use_sliding_window: Option<bool>,
}

impl Config {
    /// Rejects configs this implementation does not model.
    pub fn check(&self) -> std::result::Result<(), String> {
        if self.use_bidirectional_attention == Some(false) {
            return Err("the model uses causal attention; only bidirectional is supported".into());
        }
        if self.use_sliding_window == Some(true) {
            return Err("sliding-window attention is not supported".into());
        }
        Ok(())
    }

    /// The model's shape from GGUF metadata (`qwen3.*`).
    fn from_gguf(content: &Content) -> Result<Self> {
        match content.metadata.get("qwen3.attention.causal") {
            Some(Value::Bool(false)) => {}
            other => candle_core::bail!("GGUF qwen3.attention.causal is {other:?}, want false"),
        }
        match content.metadata.get("qwen3.pooling_type") {
            Some(Value::U32(1)) => {} // mean
            other => candle_core::bail!("GGUF pooling_type is {other:?}, want mean (1)"),
        }
        let vocab_size = content
            .tensor_infos
            .get("token_embd.weight")
            .map(|i| i.shape.dims()[0])
            .ok_or_else(|| candle_core::Error::Msg("GGUF has no token_embd.weight".into()))?;
        Ok(Self {
            hidden_size: meta_u32(content, "qwen3.embedding_length")?,
            intermediate_size: meta_u32(content, "qwen3.feed_forward_length")?,
            num_hidden_layers: meta_u32(content, "qwen3.block_count")?,
            num_attention_heads: meta_u32(content, "qwen3.attention.head_count")?,
            num_key_value_heads: meta_u32(content, "qwen3.attention.head_count_kv")?,
            head_dim: meta_u32(content, "qwen3.attention.key_length")?,
            rms_norm_eps: meta_f32(content, "qwen3.attention.layer_norm_rms_epsilon")?,
            rope_theta: meta_f32(content, "qwen3.rope.freq_base")?,
            vocab_size,
            num_labels: meta_u32(content, "qwen3.embedding_length_out")?,
            use_bidirectional_attention: Some(true),
            use_sliding_window: None,
        })
    }
}

fn meta_u32(content: &Content, key: &str) -> Result<usize> {
    match content.metadata.get(key) {
        Some(Value::U32(v)) => Ok(*v as usize),
        other => candle_core::bail!("GGUF key {key}: want u32, found {other:?}"),
    }
}

fn meta_f32(content: &Content, key: &str) -> Result<f64> {
    match content.metadata.get(key) {
        Some(Value::F32(v)) => Ok(f64::from(*v)),
        other => candle_core::bail!("GGUF key {key}: want f32, found {other:?}"),
    }
}

/// A bias-free linear layer: dense (safetensors) or quantized (GGUF).
enum Lin {
    Dense(Linear),
    /// On Metal, `QMatMul` takes and returns f32: a matrix-vector kernel for
    /// a single row, a dedicated quantized matrix-matrix kernel
    /// (`call_quantized_matmul_mm_t`) for more rows. No dequantization.
    Quant(QMatMul),
}

impl Lin {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quant(q) => q.forward(x),
        }
    }
}

const Q8_BLOCK: usize = 32;
/// An f16 scale and 32 `i8`.
const Q8_BLOCK_BYTES: usize = 34;

/// Token embedding: a device table, or the Q8_0 table kept in host memory
/// from which only the rows in use get dequantized.
enum Embed {
    Dense(Embedding),
    Q8 { table: Box<QTensor>, hidden: usize },
}

impl Embed {
    fn forward(&self, ids: &[u32], b: usize, l: usize, device: &Device) -> Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(&Tensor::from_slice(ids, (b, l), device)?),
            Self::Q8 { table, hidden } => {
                // On the CPU `data()` borrows the quantized bytes.
                let bytes = table.data()?;
                let row_bytes = hidden / Q8_BLOCK * Q8_BLOCK_BYTES;
                let mut out = Vec::with_capacity(ids.len() * hidden);
                for &id in ids {
                    let at = id as usize * row_bytes;
                    let row = bytes.get(at..at + row_bytes).ok_or_else(|| {
                        candle_core::Error::Msg(format!("token {id} is out of range"))
                    })?;
                    for block in row.as_chunks::<Q8_BLOCK_BYTES>().0.iter() {
                        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                        out.extend(block[2..].iter().map(|&q| f32::from(q.cast_signed()) * d));
                    }
                }
                Tensor::from_vec(out, (b, l, *hidden), device)
            }
        }
    }
}

/// Where tensors come from, with the name each format gives them.
enum Source {
    Safetensors(VarBuilder<'static>),
    Gguf {
        content: Content,
        file: File,
        device: Device,
    },
}

impl Source {
    fn linear(&mut self, st: &str, gg: &str, out: usize, inp: usize) -> Result<Lin> {
        self.linear_cat(&[(st.to_owned(), gg.to_owned(), out)], inp)
    }

    /// One layer whose output rows are the given layers' rows one after
    /// another (`(safetensors name, GGUF name, out)` per part). Rows of a
    /// row-major `[out, in]` matrix are independent, and for Q8_0 so are
    /// their 32-value blocks, so the raw blocks can simply be joined.
    fn linear_cat(&mut self, parts: &[(String, String, usize)], inp: usize) -> Result<Lin> {
        let total: usize = parts.iter().map(|p| p.2).sum();
        match self {
            Self::Safetensors(vb) => {
                let ws = parts
                    .iter()
                    .map(|(st, _, out)| vb.pp(st).get((*out, inp), "weight"))
                    .collect::<Result<Vec<_>>>()?;
                let w = if ws.len() == 1 {
                    ws.into_iter().next().unwrap()
                } else {
                    Tensor::cat(&ws, 0)?
                };
                Ok(Lin::Dense(Linear::new(w, None)))
            }
            Self::Gguf {
                content,
                file,
                device,
            } => {
                let join = parts.len() > 1;
                let mut bytes = Vec::new();
                let mut single = None;
                for (_, gg, out) in parts {
                    let name = format!("{gg}.weight");
                    // Joined parts are read on the CPU, as raw blocks.
                    let at = if join { &Device::Cpu } else { &*device };
                    let q = content.tensor(file, &name, at)?;
                    if q.shape().dims() != [*out, inp] {
                        candle_core::bail!("{gg}: shape {:?}, want [{out}, {inp}]", q.shape());
                    }
                    if join {
                        if q.dtype() != GgmlDType::Q8_0 {
                            candle_core::bail!("{gg}: only Q8_0 weights can be joined");
                        }
                        bytes.extend_from_slice(&q.data()?);
                    } else {
                        single = Some(q);
                    }
                }
                let q = match single {
                    Some(q) => q,
                    None => QTensor::new(
                        QStorage::from_data(bytes.into(), device, GgmlDType::Q8_0)?,
                        (total, inp),
                    )?,
                };
                Ok(Lin::Quant(QMatMul::from_qtensor(q)?))
            }
        }
    }

    fn norm(&mut self, st: &str, gg: &str, size: usize, eps: f64) -> Result<RmsNorm> {
        let w = match self {
            Self::Safetensors(vb) => vb.pp(st).get(size, "weight")?,
            Self::Gguf {
                content,
                file,
                device,
            } => content
                .tensor(file, &format!("{gg}.weight"), device)?
                .dequantize(device)?,
        };
        Ok(RmsNorm::new(w, eps))
    }

    fn embedding(&mut self, vocab: usize, hidden: usize) -> Result<Embed> {
        match self {
            Self::Safetensors(vb) => Ok(Embed::Dense(candle_nn::embedding(
                vocab,
                hidden,
                vb.pp("model.embed_tokens"),
            )?)),
            Self::Gguf { content, file, .. } => {
                let q = content.tensor(file, "token_embd.weight", &Device::Cpu)?;
                if q.dtype() != GgmlDType::Q8_0
                    || q.shape().dims() != [vocab, hidden]
                    || !hidden.is_multiple_of(Q8_BLOCK)
                {
                    candle_core::bail!("token_embd must be a Q8_0 [{vocab}, {hidden}] table");
                }
                Ok(Embed::Q8 {
                    table: Box::new(q),
                    hidden,
                })
            }
        }
    }

    fn device(&self) -> Device {
        match self {
            Self::Safetensors(vb) => vb.device().clone(),
            Self::Gguf { device, .. } => device.clone(),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Safetensors(vb) => vb.dtype(),
            // QMatMul takes and returns f32.
            Self::Gguf { .. } => DType::F32,
        }
    }
}

/// Which projections are joined into one layer. Joining is exact (rows of
/// the weight matrices are concatenated) but measured slower on Metal, see
/// `EXPERIMENT.md`, so both are off by default.
#[derive(Debug, Clone, Copy, Default)]
pub struct Joins {
    /// One layer for `[q | k | v]`.
    pub qkv: bool,
    /// One layer for `[gate | up]`.
    pub gate_up: bool,
}

/// The query, key and value projections: three layers, or one whose output
/// rows are `[q | k | v]`.
enum Qkv {
    Split(Box<[Lin; 3]>),
    Fused(Lin),
}

struct Attention {
    qkv: Qkv,
    o: Lin,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    fused: bool,
}

impl Attention {
    fn load(c: &Config, src: &mut Source, i: usize, fused: bool, joins: Joins) -> Result<Self> {
        let (h, d) = (c.num_attention_heads, c.head_dim);
        let kv = c.num_key_value_heads;
        let hid = c.hidden_size;
        let st = format!("model.layers.{i}.self_attn");
        let gg = format!("blk.{i}");
        let part =
            |n: &str, g: &str, out: usize| (format!("{st}.{n}_proj"), format!("{gg}.{g}"), out);
        let parts = [
            part("q", "attn_q", h * d),
            part("k", "attn_k", kv * d),
            part("v", "attn_v", kv * d),
        ];
        let qkv = if joins.qkv {
            Qkv::Fused(src.linear_cat(&parts, hid)?)
        } else {
            let mut l = parts
                .iter()
                .map(|(st, gg, out)| src.linear(st, gg, *out, hid));
            Qkv::Split(Box::new([
                l.next().unwrap()?,
                l.next().unwrap()?,
                l.next().unwrap()?,
            ]))
        };
        Ok(Self {
            qkv,
            o: src.linear(
                &format!("{st}.o_proj"),
                &format!("{gg}.attn_output"),
                hid,
                h * d,
            )?,
            q_norm: src.norm(
                &format!("{st}.q_norm"),
                &format!("{gg}.attn_q_norm"),
                d,
                c.rms_norm_eps,
            )?,
            k_norm: src.norm(
                &format!("{st}.k_norm"),
                &format!("{gg}.attn_k_norm"),
                d,
                c.rms_norm_eps,
            )?,
            heads: h,
            kv_heads: kv,
            head_dim: d,
            scale: (d as f32).powf(-0.5),
            fused,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (h, kv, d) = (self.heads, self.kv_heads, self.head_dim);
        // Everything up to attention stays in the (b, l, heads, d) layout the
        // projections produce: contiguous, so the fused rms_norm and
        // rope_thd kernels apply without copies.
        let (q, k, v) = match &self.qkv {
            Qkv::Split(p) => (
                p[0].forward(x)?.reshape((b, l, h, d))?,
                p[1].forward(x)?.reshape((b, l, kv, d))?,
                p[2].forward(x)?.reshape((b, l, kv, d))?,
            ),
            Qkv::Fused(p) => {
                // Heads are the rows' unit: [q heads | k heads | v heads]. The
                // norms need contiguous q and k; v goes to sdpa as a view.
                let y = p.forward(x)?.reshape((b, l, h + 2 * kv, d))?;
                (
                    y.narrow(2, 0, h)?.contiguous()?,
                    y.narrow(2, h, kv)?.contiguous()?,
                    y.narrow(2, h + kv, kv)?,
                )
            }
        };
        let q = rope_thd(&self.q_norm.forward(&q)?, cos, sin)?;
        let k = rope_thd(&self.k_norm.forward(&k)?, cos, sin)?;
        let out = if self.fused {
            // Fused Metal kernel; GQA and the key mask are handled inside.
            // It takes strides for k and v, so those stay transposed views;
            // q has to be contiguous (b, h, l, d).
            let mask = mask.expand((b, h, l, l))?;
            candle_nn::ops::sdpa(
                &q.transpose(1, 2)?.contiguous()?,
                &k.transpose(1, 2)?,
                &v.transpose(1, 2)?,
                Some(&mask),
                false,
                self.scale,
                1.0,
            )?
            .transpose(1, 2)?
            .reshape((b, l, h * d))?
        } else {
            // Grouped-query attention without copying K and V: the `rep`
            // query heads that share a KV head are stacked along the length.
            let rep = h / kv;
            let q = q.transpose(1, 2)?.reshape((b, kv, rep * l, d))?;
            let k = k.transpose(1, 2)?.transpose(2, 3)?.contiguous()?;
            let v = v.transpose(1, 2)?.contiguous()?;
            let scores = (q.matmul(&k)? * f64::from(self.scale))?;
            let scores = scores.broadcast_add(mask)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            probs
                .matmul(&v)?
                .reshape((b, h, l, d))?
                .transpose(1, 2)?
                .reshape((b, l, h * d))?
        };
        self.o.forward(&out)
    }
}

/// Gate and up projections: two layers, or one with rows `[gate | up]`.
enum GateUp {
    Split(Box<[Lin; 2]>),
    Fused(Lin),
}

struct Layer {
    attn: Attention,
    gate_up: GateUp,
    down: Lin,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Layer {
    fn load(c: &Config, src: &mut Source, i: usize, fused: bool, joins: Joins) -> Result<Self> {
        let (h, f) = (c.hidden_size, c.intermediate_size);
        let st = format!("model.layers.{i}");
        let gg = format!("blk.{i}");
        Ok(Self {
            attn: Attention::load(c, src, i, fused, joins)?,
            gate_up: {
                let g = (format!("{st}.mlp.gate_proj"), format!("{gg}.ffn_gate"), f);
                let u = (format!("{st}.mlp.up_proj"), format!("{gg}.ffn_up"), f);
                if joins.gate_up {
                    GateUp::Fused(src.linear_cat(&[g, u], h)?)
                } else {
                    GateUp::Split(Box::new([
                        src.linear(&g.0, &g.1, f, h)?,
                        src.linear(&u.0, &u.1, f, h)?,
                    ]))
                }
            },
            down: src.linear(
                &format!("{st}.mlp.down_proj"),
                &format!("{gg}.ffn_down"),
                h,
                f,
            )?,
            ln1: src.norm(
                &format!("{st}.input_layernorm"),
                &format!("{gg}.attn_norm"),
                h,
                c.rms_norm_eps,
            )?,
            ln2: src.norm(
                &format!("{st}.post_attention_layernorm"),
                &format!("{gg}.ffn_norm"),
                h,
                c.rms_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln1.forward(x)?, cos, sin, mask)?)?;
        let h = self.ln2.forward(&x)?;
        let m = match &self.gate_up {
            GateUp::Split(p) => (candle_nn::ops::silu(&p[0].forward(&h)?)? * p[1].forward(&h)?)?,
            GateUp::Fused(p) => {
                let y = p.forward(&h)?;
                let f = y.dim(D::Minus1)? / 2;
                (candle_nn::ops::silu(&y.narrow(D::Minus1, 0, f)?)? * y.narrow(D::Minus1, f, f)?)?
            }
        };
        x + self.down.forward(&m)?
    }
}

pub struct Model {
    embed: Embed,
    layers: Vec<Layer>,
    norm: RmsNorm,
    proj: Lin,
    inv_freq: Vec<f32>,
    dtype: DType,
    device: Device,
    pub dims: usize,
}

impl Model {
    pub fn load(c: &Config, vb: VarBuilder<'static>, fused: bool, joins: Joins) -> Result<Self> {
        Self::build(c, &mut Source::Safetensors(vb), fused, joins)
    }

    pub fn load_gguf(mut file: File, device: &Device, fused: bool, joins: Joins) -> Result<Self> {
        let content = Content::read(&mut file)?;
        let c = Config::from_gguf(&content)?;
        let mut src = Source::Gguf {
            content,
            file,
            device: device.clone(),
        };
        Self::build(&c, &mut src, fused, joins)
    }

    fn build(c: &Config, src: &mut Source, fused: bool, joins: Joins) -> Result<Self> {
        let device = src.device();
        let fused = device.is_metal() && fused;
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load(c, src, i, fused, joins))
            .collect::<Result<Vec<_>>>()?;
        let inv_freq = (0..c.head_dim / 2)
            .map(|i| (1.0 / c.rope_theta.powf(2.0 * i as f64 / c.head_dim as f64)) as f32)
            .collect();
        Ok(Self {
            embed: src.embedding(c.vocab_size, c.hidden_size)?,
            layers,
            norm: src.norm("model.norm", "output_norm", c.hidden_size, c.rms_norm_eps)?,
            proj: src.linear("linear", "dense_2", c.num_labels, c.hidden_size)?,
            inv_freq,
            dtype: src.dtype(),
            device,
            dims: c.num_labels,
        })
    }

    /// `ids`: `b * l` token ids, right-padded rows; `valid`: `[b, l]` f32
    /// with 1.0 for real tokens. Returns the masked mean of the projected
    /// hidden states, `[b, dims]` in f32 (not yet normalized).
    pub fn forward(&self, ids: &[u32], b: usize, l: usize, valid: &Tensor) -> Result<Tensor> {
        let pos: Vec<f32> = (0..l).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (l, 1), &self.device)?;
        let inv = Tensor::from_slice(&self.inv_freq, (1, self.inv_freq.len()), &self.device)?;
        let freqs = pos.matmul(&inv)?;
        let cos = freqs.cos()?.to_dtype(self.dtype)?;
        let sin = freqs.sin()?.to_dtype(self.dtype)?;
        // Additive mask over keys: 0 for real tokens, large negative for padding.
        let mask = ((1.0 - valid)? * -1e4)?
            .reshape((b, 1, 1, l))?
            .to_dtype(self.dtype)?;
        let mut x = self
            .embed
            .forward(ids, b, l, &self.device)?
            .to_dtype(self.dtype)?;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, &mask)?;
        }
        let x = self
            .proj
            .forward(&self.norm.forward(&x)?)?
            .to_dtype(DType::F32)?;
        let w = valid.unsqueeze(D::Minus1)?;
        let sum = x.broadcast_mul(&w)?.sum(1)?;
        let n = w.sum(1)?.clamp(1e-9, f64::MAX)?;
        sum.broadcast_div(&n)
    }
}
