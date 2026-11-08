// K2: residual add + RMSNorm in one kernel.
//
// A transformer layer has the pattern
//
//     x'  = x + delta               // the residual stream
//     h   = rmsnorm(x') * w         // the input of the next block
//
// with RMSNorm(v)[j] = v[j] / sqrt(mean(v^2) + eps), then scaled by a learned
// weight w[j]. With ordinary candle ops that is two kernels (an add and a
// norm) and the sum makes a round trip through memory in between. Here one
// kernel reads x and delta once and writes BOTH results:
//
//     out[0] = rmsnorm(x') * w      (the normalized copy)
//     out[1] = x'                   (the new residual)
//
// stacked in one buffer of shape [2, rows, h]. The Rust side slices it with
// `get`, which is free (a view).
//
// WHY THE NORMALIZED COPY COMES FIRST
// The normalized copy feeds a quantized matrix multiplication, and candle
// 0.11.1's quantized matrix-matrix kernel drops the start offset of its input
// (it rebuilds a fresh layout, so the offset of a view is lost). A view at
// offset 0 is safe, a view into the middle of the buffer is not. The residual
// only goes into the next add kernel, which honours offsets.
//
// HOW THE WORK IS SPLIT
// Normalization is per row (one token's hidden vector, h = 1024 values), and
// it needs a sum over the whole row before any element can be written. That
// is a reduction, and the natural unit for it is a THREADGROUP: a team of
// threads (here up to 256) that can share fast on-chip memory and wait for
// each other at a barrier. We launch ONE THREADGROUP PER ROW.
//
//   1. Each thread walks the row with a stride equal to the team size
//      ("grid-stride loop"), computes v = x + delta for its elements, writes
//      out[0], and keeps a private running sum of v*v.
//   2. The team adds its partial sums up. Metal groups threads into SIMD
//      groups of 32 that run in lockstep; `simd_sum` adds a value across a
//      SIMD group in a few cycles, without memory. Each SIMD group leader
//      parks its result in threadgroup memory, a barrier makes sure all
//      leaders are done, and every thread then adds the few leader values.
//   3. Everyone now knows the row's mean square, computes the scale
//      rsqrt(mean + eps), and walks the row a second time writing
//      out[0] = v * scale * w. v is recomputed from x and delta, which is
//      cheaper than keeping it: the row was just read, so it is in cache.
//
// PRECISION
// The sum of squares and the scale are float even for half inputs, because a
// sum of 1024 squares overflows or loses all precision in half. The residual
// v is rounded to T before it is squared, so the norm sees exactly the values
// stored in out[0], as the two separate candle ops would.

#include <metal_stdlib>
using namespace metal;

// At most 1024 threads per threadgroup, so at most 32 SIMD groups.
constant constexpr uint MAX_SIMD_GROUPS = 32;

template <typename T>
[[kernel]] void add_rmsnorm(
    device const T* x      [[buffer(0)]],  // residual stream, [rows, h]
    device const T* delta  [[buffer(1)]],  // output of the block, [rows, h]
    device const T* w      [[buffer(2)]],  // norm weight, [h]
    device T*       out    [[buffer(3)]],  // [2, rows, h]
    constant uint&  h      [[buffer(4)]],  // row length
    constant uint&  rows   [[buffer(5)]],  // number of rows
    constant float& eps    [[buffer(6)]],
    uint row               [[threadgroup_position_in_grid]],
    uint tid               [[thread_position_in_threadgroup]],
    uint tg_size           [[threads_per_threadgroup]],
    uint simd_lane         [[thread_index_in_simdgroup]],
    uint simd_group        [[simdgroup_index_in_threadgroup]],
    uint n_simd_groups     [[simdgroups_per_threadgroup]]) {
  // Memory shared by the whole threadgroup: one partial sum per SIMD group.
  threadgroup float partial[MAX_SIMD_GROUPS];

  const ulong base = static_cast<ulong>(row) * h;           // this row in x/delta/out[0]
  device T* normed_out = out + base;                                  // out[0]
  device T* residual_out = out + static_cast<ulong>(rows) * h + base;  // out[1]

  // Pass 1: residual add, and a private sum of squares.
  float local = 0.0f;
  for (uint j = tid; j < h; j += tg_size) {
    const T v = static_cast<T>(static_cast<float>(x[base + j]) +
                               static_cast<float>(delta[base + j]));
    residual_out[j] = v;
    const float f = static_cast<float>(v);
    local += f * f;
  }

  // Reduction: SIMD group first, then across the groups.
  const float simd_total = simd_sum(local);
  if (simd_lane == 0) {
    partial[simd_group] = simd_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint g = 0; g < n_simd_groups; ++g) {
    total += partial[g];
  }
  const float scale = rsqrt(total / static_cast<float>(h) + eps);

  // Pass 2: the normalized copy. Each thread reads back only the elements it
  // wrote itself in pass 1 (same stride), so no extra barrier is needed.
  for (uint j = tid; j < h; j += tg_size) {
    const float v = static_cast<float>(residual_out[j]);
    normed_out[j] = static_cast<T>(v * scale * static_cast<float>(w[j]));
  }
}

template [[host_name("add_rmsnorm_f32")]] [[kernel]] decltype(add_rmsnorm<float>) add_rmsnorm<float>;
template [[host_name("add_rmsnorm_f16")]] [[kernel]] decltype(add_rmsnorm<half>) add_rmsnorm<half>;
