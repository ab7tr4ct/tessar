//! `voyage-4-nano`: Qwen3 with bidirectional attention, followed by a
//! bias-free linear projection (1024 -> 2048). candle-transformers' own
//! `qwen3` is causal and has a KV cache, so this is a small re-implementation
//! of the encoder path only.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, linear_no_bias, rms_norm};
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
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    fused: bool,
}

fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    candle_nn::rotary_emb::rope(&x.contiguous()?, cos, sin)
}

impl Attention {
    fn load(c: &Config, vb: VarBuilder, fused: bool) -> Result<Self> {
        let (h, d) = (c.num_attention_heads, c.head_dim);
        let kv = c.num_key_value_heads;
        Ok(Self {
            q: linear_no_bias(c.hidden_size, h * d, vb.pp("q_proj"))?,
            k: linear_no_bias(c.hidden_size, kv * d, vb.pp("k_proj"))?,
            v: linear_no_bias(c.hidden_size, kv * d, vb.pp("v_proj"))?,
            o: linear_no_bias(h * d, c.hidden_size, vb.pp("o_proj"))?,
            q_norm: rms_norm(d, c.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(d, c.rms_norm_eps, vb.pp("k_norm"))?,
            heads: h,
            kv_heads: kv,
            head_dim: d,
            fused,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (h, kv, d) = (self.heads, self.kv_heads, self.head_dim);
        let q = self.q.forward(x)?.reshape((b, l, h, d))?.transpose(1, 2)?;
        let k = self.k.forward(x)?.reshape((b, l, kv, d))?.transpose(1, 2)?;
        let v = self.v.forward(x)?.reshape((b, l, kv, d))?.transpose(1, 2)?;
        let q = rope(&self.q_norm.forward(&q)?, cos, sin)?;
        let k = rope(&self.k_norm.forward(&k)?, cos, sin)?;
        let out = if self.fused {
            // Fused Metal kernel; GQA and the key mask are handled inside.
            let mask = mask.expand((b, h, l, l))?;
            candle_nn::ops::sdpa(
                &q,
                &k.contiguous()?,
                &v.contiguous()?,
                Some(&mask),
                false,
                (d as f32).powf(-0.5),
                1.0,
            )?
            .transpose(1, 2)?
            .reshape((b, l, h * d))?
        } else {
            // Grouped-query attention without copying K and V: the `rep`
            // query heads that share a KV head are stacked along the length.
            let rep = h / kv;
            let q = q.reshape((b, kv, rep * l, d))?;
            let k = k.transpose(2, 3)?.contiguous()?;
            let scores = (q.matmul(&k)? / (d as f64).sqrt())?;
            let scores = scores.broadcast_add(mask)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let out = probs.matmul(&v.contiguous()?)?;
            out.reshape((b, h, l, d))?
                .transpose(1, 2)?
                .reshape((b, l, h * d))?
        };
        self.o.forward(&out)
    }
}

struct Layer {
    attn: Attention,
    gate: Linear,
    up: Linear,
    down: Linear,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Layer {
    fn load(c: &Config, vb: VarBuilder, fused: bool) -> Result<Self> {
        let (h, i) = (c.hidden_size, c.intermediate_size);
        Ok(Self {
            attn: Attention::load(c, vb.pp("self_attn"), fused)?,
            gate: linear_no_bias(h, i, vb.pp("mlp.gate_proj"))?,
            up: linear_no_bias(h, i, vb.pp("mlp.up_proj"))?,
            down: linear_no_bias(i, h, vb.pp("mlp.down_proj"))?,
            ln1: rms_norm(h, c.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: rms_norm(h, c.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln1.forward(x)?, cos, sin, mask)?)?;
        let h = self.ln2.forward(&x)?;
        let m = (candle_nn::ops::silu(&self.gate.forward(&h)?)? * self.up.forward(&h)?)?;
        x + self.down.forward(&m)?
    }
}

/// `TESSAR_CANDLE_NO_SDPA=1` turns the fused kernel off (for measuring).
fn no_fused() -> bool {
    std::env::var_os("TESSAR_CANDLE_NO_SDPA").is_some()
}

pub struct Model {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    proj: Linear,
    inv_freq: Vec<f32>,
    dtype: DType,
    device: Device,
    pub dims: usize,
}

impl Model {
    pub fn load(c: &Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        let m = vb.pp("model");
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                Layer::load(
                    c,
                    m.pp(format!("layers.{i}")),
                    device.is_metal() && !no_fused(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let inv_freq = (0..c.head_dim / 2)
            .map(|i| (1.0 / c.rope_theta.powf(2.0 * i as f64 / c.head_dim as f64)) as f32)
            .collect();
        Ok(Self {
            embed: Embedding::new(
                m.pp("embed_tokens")
                    .get((c.vocab_size, c.hidden_size), "weight")?,
                c.hidden_size,
            ),
            layers,
            norm: rms_norm(c.hidden_size, c.rms_norm_eps, m.pp("norm"))?,
            proj: linear_no_bias(c.hidden_size, c.num_labels, vb.pp("linear"))?,
            inv_freq,
            dtype: vb.dtype(),
            device: device.clone(),
            dims: c.num_labels,
        })
    }

    /// `ids`: `[batch, len]` u32, right-padded; `valid`: `[batch, len]` f32
    /// with 1.0 for real tokens. Returns the masked mean of the projected
    /// hidden states, `[batch, dims]` in f32 (not yet normalized).
    pub fn forward(&self, ids: &Tensor, valid: &Tensor) -> Result<Tensor> {
        let (b, l) = ids.dims2()?;
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
        let mut x = self.embed.forward(ids)?;
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
