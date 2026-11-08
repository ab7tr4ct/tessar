# Experiment: a pure-Rust embedder on candle instead of llama.cpp

Date: 2026-11-08. Machine: Apple silicon (Metal), macOS. Branch `embed-candle`.

## Question

Can `crates/embed-llama` (C++ shim over llama.cpp, built with cmake) be
replaced by a pure-Rust adapter on candle behind the `Embedder` port, with the
same vectors and acceptable speed, so the build needs no cmake or C++?

Criteria fixed beforehand: mean cosine to llama.cpp >= 0.999 (min >= 0.995);
p95 single-query latency <= 2x llama.cpp; `cargo build` without cmake/C++.

## Findings: the architecture and candle's support

Sources (read 2026-11-08):

- <https://huggingface.co/voyageai/voyage-4-nano> (`config.json`,
  `modeling_qwen3_bidirectional.py`, `modules.json`, `1_Pooling/config.json`,
  `tokenizer.json`), Apache-2.0.
- crates.io: `candle-core` / `candle-nn` / `candle-transformers` 0.11.1,
  `tokenizers` 0.22.2 (0.23.x and 1.0.0-rc exist; 0.22 used).

| Item | Value |
| --- | --- |
| Architecture | `Qwen3Model` with **bidirectional** attention (`is_causal = false`, key-padding mask only), 12 layers, hidden 1024, 16 heads / 8 KV heads (GQA), head_dim 128, MLP 3072, RMSNorm eps 1e-6, RoPE theta 1e6, q/k norm |
| Head | bias-free `linear` 1024 -> 2048 after the final norm (`num_labels`) |
| Pooling | **mean** over real tokens, then L2 normalize (`1_Pooling`, `modules.json`) |
| Output | 2048 dims (Matryoshka 256/512/1024/2048) |
| Tokenizer | Qwen BPE, 151936 vocab, `tokenizer.json`; no BOS/EOS added |
| Context | 32768 (`sentence_bert_config`), 40960 positions |
| Weights | `model.safetensors`, bf16, 693 MB. **No GGUF in the HF repo** (we convert our own Q8_0) |
| Prompts | `Represent the query for retrieving supporting documents: ` / `Represent the document for retrieval: ` |

Candle support:

- `candle_transformers::models::qwen3` exists but is causal with a KV cache
  and has no hook for bidirectional attention or the extra linear head. It was
  not usable as is. I wrote the encoder path myself: `src/model.rs`, ~190
  lines, built from `candle_nn` pieces (`Linear`, `RmsNorm`, `Embedding`,
  `rotary_emb::rope`, `ops::sdpa`). Loading uses the stock safetensors names.
- f16, bf16 and f32 all work on Metal and CPU (CPU f16 works but is slow).
- Quantized GGUF (`candle_transformers::models::quantized_qwen3`) is a causal
  LM loader. A bidirectional encoder with the head would need its own
  quantized version (`candle_core::quantized::QMatMul` is available) and our
  own GGUF export. Not attempted; see risks.

## Setup

`crates/embed-candle`: `CandleEmbedder::load(model_dir, &Options)` implements
`Embedder`. Same behaviour as the llama.cpp adapter: Voyage prompts prepended,
truncation to the context, mean pooling (what the GGUF declares), L2
normalization in f64, vectors of 2048 floats. `tokenizers` is built with
`fancy-regex` (pure Rust, no oniguruma). Weights are read with a buffered
read, not mmap, to keep `unsafe` out. Texts are length-sorted inside a call to
cut padding. Features: `metal` (`candle-core/metal`), `accelerate`.

Comparison harness: `crates/embed-llama/tests/candle_compare.rs` (ignored, GPU). It generates
bilingual texts (100 for E15a, 200 for E15b; half en, half ru, about 15-25 words), same Voyage prompts for both
adapters, llama.cpp with context 2048 and batch 64, `n_gpu_layers` all. 5 warm-up
calls before timing, 100 timed single queries, 10 timed batches of 64 documents
(2 warm-up). Model: the HF safetensors for candle; `voyage-4-nano-Q8_0.gguf`
for llama.cpp. `tests/profile.rs`: Metal kernel micro-benchmark.

```sh
cargo test -p tessar-embed-llama --release --test candle_compare -- --ignored --nocapture
```

## Results

### Quality (candle f16 on Metal vs llama.cpp Q8_0, per text)

| Variant | mean cos | min cos |
| --- | --- | --- |
| no EOS appended (default) | **0.99992** | **0.99986** |
| EOS appended | 0.98562 | 0.94059 |

Both targets pass with a large margin. The tiny gap is the Q8_0 quantization
(and f16 rounding) in llama.cpp. llama.cpp does not append EOS for this GGUF,
so neither does the candle adapter. bf16 and f32 gave the same speed class;
cosine was measured for f16 only (run with `CANDLE_DTYPE=f32` to see the
f32 value).

### Speed (Metal; llama.cpp = Q8_0, all layers on GPU)

| | llama.cpp | candle (final: f16, sdpa, batch 8) |
| --- | --- | --- |
| single query p50 | 9.9 ms | 15.7 ms |
| single query p95 | 10.4 ms | 17.6 ms (1.7x) |
| batch of 64 docs p50 | 264 ms | 621 ms (2.4x) |
| batch of 64 docs p95 | 265 ms | 623 ms |
| model load | 0.2 s warm, 6.7 s cold | 0.4-0.7 s |
| peak RSS | 0.73 GB | 1.5 GB (footprint 2.4 GB) |

The first run (before the attention fixes) was query 20.0 / 24.0 ms and 64
documents 1410 ms; llama.cpp was 10.9 / 14.6 ms and 274 ms in that run (the
numbers differ run to run by a few percent).

### Ablations (candle on Metal unless noted; query p50/p95 ms, 64 docs p50 ms)

| Variant | query | 64 docs |
| --- | --- | --- |
| f16, naive attention + repeat_kv, batch 64 | 20.0 / 24.0 | 1410 |
| bf16, same | 21.8 / 25.2 | 1532 |
| f32, same | 24.1 / 27.6 | 1683 |
| f16, batch 8 / 1 (naive) | 19.2 / 21.8 ; 21.3 / 27.3 | 883 ; 1372 |
| `CANDLE_METAL_COMPUTE_PER_BUFFER=1000` | 21.9 / 26.6 | 883 (no change) |
| f16, GQA without copies (no repeat_kv), batch 64 | 17.4 / 19.5 | 1085 |
| + fused `ops::sdpa`, batch 64 | 15.6 / 17.7 | 972 |
| + fused `ops::sdpa`, batch 32 | 15.6 / 17.6 | 738 |
| + fused `ops::sdpa`, batch 16 | 15.7 / 17.5 | 650 |
| **+ fused `ops::sdpa`, batch 8** | **15.7 / 17.6** | **621** |
| same, without sdpa, batch 16 | 17.2 / 19.2 | 717 |
| CPU f16 (gemm), batch 8 | 60.8 / 87.3 | 2075 |
| CPU f32 (gemm), batch 8 | 97.6 / 120.5 | 3119 |
| CPU f32 + `accelerate`, batch 8 | 64.9 / 72.4 | 1258 |

Checks: device is Metal (`is_metal() == true` printed by the harness), compute
dtype f16. The model is loaded once and kept; the tokenizer is not rebuilt per
call. Warm-up is done before timing (first call compiles Metal pipelines).
Batch size: 8 is best; larger batches are slower, 1 is dispatch-bound
(about 16-21 ms per forward pass regardless of length).

Micro-benchmark (`tests/profile.rs`, 64 x 40 tokens, f16): dense matmuls run at
about 4.7 TFLOPS (3.4 ms for the 1024 -> 3072 projection, fine). The costly
pieces were `repeat_kv` (8.5 ms per call, a strided copy), batched
`q.kT` for small matrices (7.3 ms, 0.06 TFLOPS) and softmax (0.7 ms). Removing
them (GQA by stacking query heads, then the fused `sdpa` kernel) took the
batch from 1410 ms to 621 ms with identical vectors. What remains: the 7
projections per layer at about 17 ms x 12 = 200 ms in matmuls plus per-op
dispatch overhead (about 25 ops per layer, about 16 ms floor per forward).
llama.cpp fuses more and uses better matmul kernels for Q8_0.

### Build facts

- `cargo build -p tessar-embed-candle`: **no cmake, no C/C++ compiler**
  (`cargo tree` shows no `cmake`, `cc`, `cxx` or `-sys` crates in the normal
  and build graph, including with `--features metal`).
- Clean release build, no metal feature: 49 s wall (257 s CPU, 10 cores).
  With `metal`: about 67 s for the whole test binary incl. llama.cpp's cmake
  (the candle part alone is of the same order). The cmake build of
  `embed-llama` from clean took about 5 min CPU in this worktree.
- `unsafe_code = "forbid"` holds for the crate (buffered safetensors load, no
  mmap).
- Model on disk: 693 MB bf16 safetensors vs 351 MB Q8_0 GGUF. RSS 1.5 GB at
  load because the whole file is read and converted (the bf16 buffer is freed
  after load; the peak footprint is 2.4 GB).

## Verdict against the criteria

| Criterion | Result | Pass |
| --- | --- | --- |
| mean cosine >= 0.999 | 0.99992 | yes |
| min cosine >= 0.995 | 0.99986 | yes |
| p95 query latency <= 2x llama.cpp | 17.6 vs 10.4 ms = 1.7x | yes |
| build without cmake / C++ | confirmed | yes |

Not part of the criteria but worth knowing: bulk indexing (64 documents) is
2.4x slower (621 vs 264 ms), memory is about 2x higher, and the model on disk
is 2x larger. Re-embedding a memory of 10k facts on open would take about 97 s
instead of 41 s at these rates; the vector cache in `crates/mcp` hides most
of this.

Honest caveat: the 1.7x on the query is close to the limit and depends on a
single machine; the gap to llama.cpp is dispatch overhead, not arithmetic.

## Risks

- Maintenance: we now own a model implementation (~190 lines) instead of an
  upstream C++ library. A new Voyage model (e.g. `voyage-4-lite`, other
  architectures) needs porting work; llama.cpp gets new architectures for
  free. For Qwen3-class encoders the port is small.
- Candle's Metal kernels: batched small matmuls and strided copies were slow
  (found above); `ops::sdpa` constraints (mask must be `(b, heads, q, kv)`,
  head dim from a fixed list, mask dtype = q dtype) may change between
  versions. Pin candle (0.11.x) and keep the ignored comparison test as the
  regression gate.
- Quantization: no Q8_0 path yet. f16 weights are 2x the size of Q8_0 and
  use more bandwidth; a quantized encoder would need `QMatMul` plumbing and
  our own GGUF export of the head. Unknown speed; not measured.
- Memory: 1.5 GB resident vs 0.73 GB; the load path could stream tensors
  instead of reading the whole file (mmap would need `unsafe`; a chunked
  safetensors read is possible).
- Numerics: f16 gives cos 0.9999 to Q8_0; both are approximations of the
  bf16 reference. Not measured against the Python reference here.
- CPU fallback is slow (65 ms/query with Accelerate), fine only as a fallback.
- Linux/Windows GPU: only `cuda` would exist as a feature, untested here.

## Recommendation

Technically feasible and within all stated criteria: the candle adapter gives
the same vectors (cos 0.9999) and about 1.7x the query latency without
cmake or C++. Adopt it as the default if "no C++ in the build" is worth the
2.4x slower bulk embedding and 2x memory; keep `embed-llama` behind a feature
or as the fast path until the bulk gap closes. Concretely:

1. Do not delete `embed-llama` yet. Make the choice a cargo feature in `mcp`
   and compare on the real memory-bench workload (re-embed on open) before
   switching the default.
2. If bulk speed matters, next steps in order: (a) profile the layer with
   Metal system trace to see the dispatch gap, fuse `rms_norm` + residual and
   `silu * up` into custom ops or batch the Q/K/V projections into one matmul
   (3 -> 1 dispatch per layer); (b) a quantized (Q8_0 `QMatMul`) encoder;
   (c) a macOS-only adapter on `mlx-rs` (Apple's MLX, usually faster on Apple
   silicon, but C++ underneath so it loses the main point), with candle (or
   ONNX) elsewhere.
3. Keep `crates/embed-llama/tests/candle_compare.rs` as the gate: cosine >= 0.999 mean to the
   llama.cpp adapter on the same texts.

## E15b step 1: Q8_0 weights straight from the GGUF

`CandleEmbedder::load_gguf(gguf, tokenizer_json, &options)` reads the same
`voyage-4-nano-Q8_0.gguf` llama.cpp uses through
`candle_core::quantized::gguf_file`, so the 693 MB safetensors copy is not
needed (the tokenizer still comes from `tokenizer.json`; candle has no GGUF
tokenizer). Config is read from the `qwen3.*` metadata (`attention.causal`
must be false, `pooling_type` must be 1 = mean).

Tensor mapping (candle shapes are `[out, in]`; all linear weights are Q8_0,
norms F32):

| Model part | GGUF name | safetensors name | Handling |
| --- | --- | --- | --- |
| token embedding `[151936, 1024]` | `token_embd.weight` (Q8_0) | `model.embed_tokens.weight` | kept as raw Q8_0 bytes in host memory (165 MB); only the rows of the ids in use are dequantized per call |
| q / k / v proj | `blk.N.attn_q/k/v.weight` | `...self_attn.{q,k,v}_proj` | `QMatMul` |
| o proj `[1024, 2048]` | `blk.N.attn_output.weight` | `...self_attn.o_proj` | `QMatMul` |
| q / k norm `[128]` | `blk.N.attn_q_norm/attn_k_norm.weight` | `...self_attn.{q,k}_norm` | dequantized (F32) |
| input / post-attn norm | `blk.N.attn_norm.weight`, `blk.N.ffn_norm.weight` | `input_layernorm`, `post_attention_layernorm` | dequantized (F32) |
| gate / up / down | `blk.N.ffn_gate/up/down.weight` | `...mlp.{gate,up,down}_proj` | `QMatMul` |
| final norm | `output_norm.weight` | `model.norm.weight` | dequantized |
| **1024 -> 2048 projection** | **`dense_2.weight`** (Q8_0 `[2048, 1024]`) | `linear.weight` | `QMatMul` |

The token embedding is a gather, which candle has no op for on `QTensor`, so
the Q8_0 table (34 bytes per 32 values: f16 scale + 32 `i8`) stays on the CPU
and the needed rows are dequantized by hand (about 15 lines, `half` for the
scale). Costs nothing measurable per call and saves 450+ MB against a
dequantized table.

Which Metal path `QMatMul` takes (candle-core 0.11.1 `quantized/metal.rs`):
one input row goes to `call_quantized_matmul_mv_t`; more than one row goes to
`call_quantized_matmul_mm_t`, a dedicated quantized matrix-matrix kernel. The
weights are **not** dequantized to f16 for batches. The input has to be f32
and contiguous, the output is f32, so the whole model runs on f32
activations in this mode. Micro-benchmark (1024 -> 3072): Q8_0 `QMatMul`
2560 rows 4.18 ms (3.85 TFLOPS) vs f16 `Linear` 3.39 ms (4.75 TFLOPS); 40
rows 0.19 ms vs 0.35 ms (faster); 1 row 0.05 ms vs 0.07 ms. The kernel is
not the problem.

Measured on the same 200 texts against llama.cpp (raw numbers:
`evals/runs/e15b/step1-q8.md`):

| | llama.cpp Q8_0 | candle GGUF Q8_0 | candle safetensors f16 |
| --- | --- | --- | --- |
| cosine to llama.cpp, mean / min | - | 1.00000 / 1.00000 | 0.99992 / 0.99987 |
| max RSS | 737 MB (footprint 339 MB) | **674 MB** (footprint 665 MB) | 1122 MB (footprint 2475 MB) |
| query p50 / p95 | 10.1 / 10.6 ms | 17.6 / 20.2 ms | 16.0 / 17.8 ms |
| bulk (200 texts) | 241 texts/s | 81 texts/s | 106 texts/s |
| 64 docs p50 | 264 ms | 807 ms | 625 ms |
| load | 0.25-0.5 s | 0.7 s | 1.0 s |

Criteria, fixed in advance: cosine >= 0.999 (pass: identical weights give
1.00000), RSS <= 1 GB (pass: 674 MB), query p95 <= 1.5x llama.cpp (**fail**:
1.9x), bulk <= 2x llama.cpp (**fail**: 3.0x slower).

Reading: memory and quality are solved by taking Q8_0 straight from the
GGUF. Speed got worse than f16 safetensors (query p95 17.8 -> 20.2 ms, bulk
106 -> 81 texts/s) while the quantized matmul itself is as fast as dense
f16; the loss is that `QMatMul` forces f32 activations, so every non-matmul
op (norms, RoPE, SwiGLU, sdpa, residual adds, copies) moves twice the bytes.
Step 2 candidates: keep activations in f16 and cast only at the 7 `QMatMul`
inputs/outputs per layer (casts are cheap element-wise ops, but add
dispatches), or fuse norm + cast; merge Q/K/V into one `QMatMul` by
concatenating the quantized rows (needs `QTensor` concatenation, which is not
a public op, so repacking Q8_0 blocks by hand); reduce dispatch count
(about 16 ms floor per forward pass is what bounds the query).

## E15b step 2: fewer dispatches

Raw rows: `evals/runs/e15b/step2-fusion.md`. Setup as in step 1 (200 texts).

| step | GGUF Q8 query p50 / p95 | GGUF Q8 bulk | f16 query p50 / p95 | f16 bulk |
| --- | --- | --- | --- | --- |
| llama.cpp | 10.3 / 11.0 ms | 241 texts/s | same | same |
| step 1 | 17.6 / 20.2 | 81 | 16.0 / 17.8 | 106 |
| 2a `CANDLE_METAL_COMPUTE_PER_BUFFER` 200-5000 | no change (noise) | no change | no change | no change |
| **2b copies removed, fused norm and rope** | **13.2 / 14.4** | **123** | **11.8 / 13.1** | **153-157** |
| 2c+2d joined qkv, gate/up | 14.5 / 16.4 | 93 | 13.5 / 15.7 | 110 |
| 2e f16 activations (Q8 only) | 13.3 / 14.4 | 122 | - | - |

- 2a: the default is 50 ops per command buffer (candle-metal-kernels 0.11.1
  `commands.rs`). Raising it does nothing. It is read with `std::env::var`
  when the device is created and there is no API; setting it from our code
  would need `unsafe` `set_var` (forbidden), so it is not adopted.
- 2b gave the whole win. The main cause was not the launch count but that
  `q_norm`/`k_norm` ran on transposed (non-contiguous) views and so fell
  back to candle's composite `LayerNorm::forward` (9 small kernels each in f16).
  Applying them in the contiguous `(b, l, heads, d)` layout uses the fused
  `rms_norm`, then `rope_thd` needs no copy, and `sdpa` takes strided k and v
  (only q must be contiguous). Dispatches per layer: about 37 -> 20.
- 2c+2d: exact (Q8_0 blocks are joined byte for byte, cosine unchanged) and
  slower. After 2b the model is not launch-bound; the `narrow` copies and
  strided kernels cost more than the saved launches. Kept as
  `Options.joins` (since removed, see below).
- 2e: `QMatMul` on Metal asserts f32 input, so casts are required; they cost
  as much as the f16 element-wise work saves. Kept as
  `Options.f16_activations` (since removed, see below).

Verdict against the fixed criteria (GGUF Q8, 2b): cosine 1.00000 (pass), RSS
676 MB (pass), query p95 1.31x (pass), bulk 1.96x slower (pass, barely).
The joined-projection and f16-activation code (`Options.joins`, `Options.f16_activations`) was
removed from the tree; it lives in commits `2d441b8` and `abec331` if anyone wants it back.

Remaining ideas, none cheap: a fused SwiGLU or residual+norm custom op; sdpa
that accepts a strided q; a single command buffer per forward pass.
