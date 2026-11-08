// K1: SwiGLU, the activation of Qwen3's feed-forward block.
//
//     out[i] = silu(gate[i]) * up[i],     silu(g) = g / (1 + e^-g)
//
// The model computes gate = x W_gate and up = x W_up with two matrix
// multiplications. Done with ordinary candle ops, this step is two kernels:
// `silu(gate)` writes a temporary tensor, `* up` reads it back. Each kernel
// launch costs a few microseconds of GPU command overhead, and the temporary
// costs a round trip through memory. One kernel that reads `gate` and `up`
// once and writes `out` once does the same job with a single launch.
//
// ELEMENT-WISE KERNELS IN METAL, IN FIVE LINES
//   * A "thread" is one tiny worker. We launch one per output element.
//   * `thread_position_in_grid` is the thread's global index, our `i`.
//   * The grid is rounded up to a whole number of threadgroups, so the last
//     few threads may have no element: hence the `i >= n` guard.
//   * `device` marks memory in the GPU's main memory; `const` promises we
//     only read it. `constant` is the small read-only parameter space.
//   * Buffers are bound by index: buffer(0) is the first argument that the
//     Rust side sets (see `ops.rs`), buffer(1) the second, and so on.
//
// PRECISION
// Inputs and output have type T (float or half). The arithmetic is always
// done in float, and the result is rounded once on the way out. With T = half
// that is slightly more accurate than rounding after silu and again after
// the product, which is what the two separate candle kernels do.
//
// TEMPLATES
// One body serves both dtypes. Metal needs every instantiation to be named
// explicitly, because the host looks kernels up by string: the
// `[[host_name("...")]]` lines at the bottom create "swiglu_f32" and
// "swiglu_f16".

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void swiglu(
    device const T* gate [[buffer(0)]],   // x W_gate, n elements
    device const T* up   [[buffer(1)]],   // x W_up,   n elements
    device T*       out  [[buffer(2)]],   // silu(gate) * up, n elements
    constant uint&  n    [[buffer(3)]],   // element count
    uint            i    [[thread_position_in_grid]]) {
  if (i >= n) {
    return;  // rounding of the grid: this thread has no element
  }
  const float g = static_cast<float>(gate[i]);
  const float u = static_cast<float>(up[i]);
  // exp() of a large negative number is 0 and of a large positive one is inf;
  // g / (1 + inf) is 0 for very negative g, which is the correct limit.
  out[i] = static_cast<T>(g / (1.0f + exp(-g)) * u);
}

template [[host_name("swiglu_f32")]] [[kernel]] decltype(swiglu<float>) swiglu<float>;
template [[host_name("swiglu_f16")]] [[kernel]] decltype(swiglu<half>) swiglu<half>;
