//! Two fused operations as candle custom ops, for fewer GPU launches:
//!
//! * [`swiglu`]: `silu(gate) * up` (kernel `kernels/swiglu.metal`),
//! * [`add_rmsnorm`]: `x + delta` and its RMSNorm in one pass
//!   (kernel `kernels/add_rmsnorm.metal`).
//!
//! A candle custom op is a struct that implements `CustomOp2`/`CustomOp3`: one
//! method per backend that turns input storages into an output storage. The
//! CPU method here is plain safe Rust and is also the reference the tests
//! compare the Metal kernels with. The Metal method compiles the `.metal`
//! source once (Metal runtime compiler), then for every call binds the
//! buffers, sets the scalar arguments and dispatches. candle's command
//! encoder takes care of ordering against the other kernels of the pass.
//!
//! For dtypes other than f32/f16 the functions fall back to the composed
//! candle ops, so callers never have to care.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor};
use half::f16;

/// `silu(gate) * up` with ordinary candle ops (2 kernel launches on Metal).
pub fn swiglu_composed(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(gate)? * up
}

/// `[rmsnorm(x + delta) * weight, x + delta]` with ordinary candle ops,
/// stacked on a new first axis (2 launches on Metal).
pub fn add_rmsnorm_composed(
    x: &Tensor,
    delta: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let sum = (x + delta)?;
    let normed = candle_nn::ops::rms_norm(&sum.contiguous()?, weight, eps)?;
    Tensor::stack(&[normed, sum], 0)
}

fn supported(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16)
}

/// `silu(gate) * up`, one launch on Metal. Same shape and dtype as the inputs.
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if !supported(gate.dtype()) || gate.dims() != up.dims() {
        return swiglu_composed(gate, up);
    }
    gate.apply_op2_no_bwd(up, &Swiglu)
}

/// Returns a tensor of shape `[2, ..x.dims()]`: index 0 is
/// `rmsnorm(x + delta) * weight`, index 1 is `x + delta`. Use `.get(0)` /
/// `.get(1)` for views. The normalized copy is first on purpose: candle 0.11.1's
/// quantized matmul ignores the start offset of its input, so the tensor that
/// feeds it must be the view at offset 0.
pub fn add_rmsnorm(x: &Tensor, delta: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if !supported(x.dtype()) || x.dims() != delta.dims() || weight.dtype() != x.dtype() {
        return add_rmsnorm_composed(x, delta, weight, eps);
    }
    x.apply_op3_no_bwd(delta, weight, &AddRmsNorm { eps })
}

/// Contiguous element range of a layout, or an error: our kernels do not
/// take strides (the model only feeds them contiguous tensors or views at an
/// offset, which `contiguous_offsets` covers).
fn range(layout: &Layout, what: &str) -> Result<(usize, usize)> {
    layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg(format!("{what} must be contiguous")))
}

/// The elements of a CPU storage as f32, plus a function to turn f32 back.
fn to_f32(storage: &CpuStorage, layout: &Layout, what: &str) -> Result<Vec<f32>> {
    let (start, end) = range(layout, what)?;
    match storage {
        CpuStorage::F32(v) => Ok(v[start..end].to_vec()),
        CpuStorage::F16(v) => Ok(v[start..end].iter().map(|x| x.to_f32()).collect()),
        other => candle_core::bail!("{what}: unsupported dtype {:?}", other.dtype()),
    }
}

fn from_f32(values: Vec<f32>, dtype: DType) -> CpuStorage {
    match dtype {
        DType::F16 => CpuStorage::F16(values.into_iter().map(f16::from_f32).collect()),
        _ => CpuStorage::F32(values),
    }
}

/// Rounds like the output dtype does, so a value that is stored and read
/// back (the residual) is the same in the reference and in the kernel.
fn round(v: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => f16::from_f32(v).to_f32(),
        _ => v,
    }
}

struct Swiglu;

impl CustomOp2 for Swiglu {
    fn name(&self) -> &'static str {
        "tessar-swiglu"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let gate = to_f32(s1, l1, "gate")?;
        let up = to_f32(s2, l2, "up")?;
        let out = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        Ok((from_f32(out, s1.dtype()), l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        gate: &candle_core::MetalStorage,
        gl: &Layout,
        up: &candle_core::MetalStorage,
        ul: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::swiglu(gate, gl, up, ul)
    }
}

struct AddRmsNorm {
    eps: f32,
}

impl CustomOp3 for AddRmsNorm {
    fn name(&self) -> &'static str {
        "tessar-add-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dtype = s1.dtype();
        let x = to_f32(s1, l1, "x")?;
        let delta = to_f32(s2, l2, "delta")?;
        let w = to_f32(s3, l3, "weight")?;
        let h = w.len();
        if h == 0 || x.len() % h != 0 {
            candle_core::bail!(
                "add_rmsnorm: row length {h} does not divide {} elements",
                x.len()
            );
        }
        // Layout of the result: [normed | residual], each as long as x.
        let mut out = vec![0f32; 2 * x.len()];
        let (normed, residual) = out.split_at_mut(x.len());
        for (row, (xr, dr)) in x.chunks_exact(h).zip(delta.chunks_exact(h)).enumerate() {
            let at = row * h;
            for j in 0..h {
                residual[at + j] = round(xr[j] + dr[j], dtype);
            }
            let sum_sq: f32 = residual[at..at + h].iter().map(|v| v * v).sum();
            let scale = 1.0 / (sum_sq / h as f32 + self.eps).sqrt();
            for j in 0..h {
                normed[at + j] = residual[at + j] * scale * w[j];
            }
        }
        let mut dims = vec![2];
        dims.extend_from_slice(l1.shape().dims());
        Ok((from_f32(out, dtype), Shape::from(dims)))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x: &candle_core::MetalStorage,
        xl: &Layout,
        delta: &candle_core::MetalStorage,
        dl: &Layout,
        w: &candle_core::MetalStorage,
        wl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::add_rmsnorm(x, xl, delta, dl, w, wl, self.eps)
    }
}

#[cfg(feature = "metal")]
mod metal {
    //! Host side of the two Metal kernels: compile once, bind, dispatch.

    use std::sync::OnceLock;

    use candle_core::backend::BackendStorage;
    use candle_core::{DType, Layout, MetalStorage, Result, Shape};
    use candle_metal_kernels::metal::ComputePipeline;
    use objc2_metal::MTLSize;

    use super::range;

    const SWIGLU: &str = include_str!("kernels/swiglu.metal");
    const ADD_RMSNORM: &str = include_str!("kernels/add_rmsnorm.metal");

    /// Threads per threadgroup for the row-wise kernel: 256 threads walk a
    /// 1024-value row with 4 values each. Capped by what the pipeline allows.
    const ROW_THREADS: usize = 256;

    struct Pipelines {
        swiglu_f32: ComputePipeline,
        swiglu_f16: ComputePipeline,
        add_rmsnorm_f32: ComputePipeline,
        add_rmsnorm_f16: ComputePipeline,
    }

    /// Compiled on first use and shared. A pipeline belongs to the Metal
    /// device it was built on; the process has one (device 0).
    static PIPELINES: OnceLock<std::result::Result<Pipelines, String>> = OnceLock::new();

    fn build(device: &candle_core::MetalDevice) -> std::result::Result<Pipelines, String> {
        let metal = device.device();
        let pipeline = |source: &str, name: &str| -> std::result::Result<ComputePipeline, String> {
            let library = metal
                .new_library_with_source(source, None)
                .map_err(|e| format!("{name}: {e}"))?;
            let function = library
                .get_function(name, None)
                .map_err(|e| format!("{name}: {e}"))?;
            metal
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("{name}: {e}"))
        };
        Ok(Pipelines {
            swiglu_f32: pipeline(SWIGLU, "swiglu_f32")?,
            swiglu_f16: pipeline(SWIGLU, "swiglu_f16")?,
            add_rmsnorm_f32: pipeline(ADD_RMSNORM, "add_rmsnorm_f32")?,
            add_rmsnorm_f16: pipeline(ADD_RMSNORM, "add_rmsnorm_f16")?,
        })
    }

    fn pipelines(device: &candle_core::MetalDevice) -> Result<&'static Pipelines> {
        PIPELINES
            .get_or_init(|| build(device))
            .as_ref()
            .map_err(|e| candle_core::Error::Msg(format!("metal kernel build failed: {e}")))
    }

    fn check_contiguous(layout: &Layout, what: &str) -> Result<usize> {
        // The kernels index from a buffer offset: a view at an offset is
        // fine, a strided one is not.
        range(layout, what).map(|(start, _)| start)
    }

    pub fn swiglu(
        gate: &MetalStorage,
        gl: &Layout,
        up: &MetalStorage,
        ul: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (g0, g1) = range(gl, "gate")?;
        let u0 = check_contiguous(ul, "up")?;
        let n = g1 - g0;
        let dtype = gate.dtype();
        let size = dtype.size_in_bytes();
        let device = gate.device();
        let p = pipelines(device)?;
        let pipeline = match dtype {
            DType::F32 => &p.swiglu_f32,
            DType::F16 => &p.swiglu_f16,
            other => candle_core::bail!("swiglu: unsupported dtype {other:?}"),
        };
        let out = device.new_buffer(n, dtype, "swiglu")?;
        {
            let guard = device.command_encoder()?;
            let encoder = guard.as_ref();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_input_buffer(0, Some(gate.buffer()), g0 * size);
            encoder.set_input_buffer(1, Some(up.buffer()), u0 * size);
            encoder.set_output_buffer(2, Some(&out), 0);
            encoder.set_bytes(3, &(n as u32));
            // One thread per element, 256 per threadgroup; the grid is
            // rounded up, the kernel ignores the surplus threads.
            let group = 256usize.min(pipeline.max_total_threads_per_threadgroup());
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: n.div_ceil(group),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: group,
                    height: 1,
                    depth: 1,
                },
            );
        }
        let storage = MetalStorage::new(out, device.clone(), n, dtype);
        Ok((storage, gl.shape().clone()))
    }

    pub fn add_rmsnorm(
        x: &MetalStorage,
        xl: &Layout,
        delta: &MetalStorage,
        dl: &Layout,
        w: &MetalStorage,
        wl: &Layout,
        eps: f32,
    ) -> Result<(MetalStorage, Shape)> {
        let (x0, x1) = range(xl, "x")?;
        let d0 = check_contiguous(dl, "delta")?;
        let w0 = check_contiguous(wl, "weight")?;
        let dtype = x.dtype();
        let size = dtype.size_in_bytes();
        let h = wl.shape().elem_count();
        let n = x1 - x0;
        if h == 0 || n % h != 0 {
            candle_core::bail!("add_rmsnorm: row length {h} does not divide {n} elements");
        }
        let rows = n / h;
        let device = x.device();
        let p = pipelines(device)?;
        let pipeline = match dtype {
            DType::F32 => &p.add_rmsnorm_f32,
            DType::F16 => &p.add_rmsnorm_f16,
            other => candle_core::bail!("add_rmsnorm: unsupported dtype {other:?}"),
        };
        let out = device.new_buffer(2 * n, dtype, "add_rmsnorm")?;
        {
            let guard = device.command_encoder()?;
            let encoder = guard.as_ref();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_input_buffer(0, Some(x.buffer()), x0 * size);
            encoder.set_input_buffer(1, Some(delta.buffer()), d0 * size);
            encoder.set_input_buffer(2, Some(w.buffer()), w0 * size);
            encoder.set_output_buffer(3, Some(&out), 0);
            encoder.set_bytes(4, &(h as u32));
            encoder.set_bytes(5, &(rows as u32));
            encoder.set_bytes(6, &eps);
            // One threadgroup per row.
            let group = ROW_THREADS.min(pipeline.max_total_threads_per_threadgroup());
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: group,
                    height: 1,
                    depth: 1,
                },
            );
        }
        let mut dims = vec![2];
        dims.extend_from_slice(xl.shape().dims());
        let storage = MetalStorage::new(out, device.clone(), 2 * n, dtype);
        Ok((storage, Shape::from(dims)))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::*;

    fn data(n: usize, seed: u32) -> Vec<f32> {
        // A deterministic spread of values, both signs, up to about +-2: small
        // enough that f16 rounding (a relative error) stays under the
        // absolute tolerance of the tests.
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x >> 8) % 10_000) as f32 / 10_000.0 * 4.0 - 2.0
            })
            .collect()
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        a.to_dtype(DType::F32)
            .unwrap()
            .sub(&b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn check_swiglu(dev: &Device, dtype: DType, tol: f32) {
        let (g, u) = (data(5 * 7 * 33, 1), data(5 * 7 * 33, 2));
        let g = Tensor::from_vec(g, (5, 7, 33), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let u = Tensor::from_vec(u, (5, 7, 33), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let got = swiglu(&g, &u).unwrap();
        let want = swiglu_composed(&g, &u).unwrap();
        assert_eq!(got.dims(), want.dims());
        let d = max_diff(&got, &want);
        assert!(d <= tol, "swiglu {dtype:?} on {dev:?}: max diff {d}");
    }

    fn check_add_rmsnorm(dev: &Device, dtype: DType, tol: f32) {
        let h = 96;
        let rows = 4 * 5;
        let x = Tensor::from_vec(data(rows * h, 3), (4, 5, h), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let d = Tensor::from_vec(data(rows * h, 4), (4, 5, h), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let w = Tensor::from_vec(data(h, 5), h, dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let got = add_rmsnorm(&x, &d, &w, 1e-6).unwrap();
        let want = add_rmsnorm_composed(&x, &d, &w, 1e-6).unwrap();
        assert_eq!(got.dims(), &[2, 4, 5, h]);
        let diff = max_diff(&got, &want);
        assert!(
            diff <= tol,
            "add_rmsnorm {dtype:?} on {dev:?}: max diff {diff}"
        );
        // The residual half is exactly x + d.
        let residual = max_diff(&got.get(1).unwrap(), &(&x + &d).unwrap());
        assert!(residual <= tol, "residual differs by {residual}");
    }

    #[test]
    fn swiglu_matches_composed_ops_on_cpu() {
        check_swiglu(&Device::Cpu, DType::F32, 1e-5);
        check_swiglu(&Device::Cpu, DType::F16, 1e-2);
    }

    #[test]
    fn add_rmsnorm_matches_composed_ops_on_cpu() {
        check_add_rmsnorm(&Device::Cpu, DType::F32, 1e-5);
        check_add_rmsnorm(&Device::Cpu, DType::F16, 1e-2);
    }

    #[test]
    fn add_rmsnorm_accepts_views_at_an_offset() {
        // The model feeds a slice of a stacked [2, b, l, h] result.
        let dev = Device::Cpu;
        let h = 64;
        let stacked = Tensor::from_vec(data(2 * 3 * h, 7), (2, 3, h), &dev).unwrap();
        let x = stacked.get(1).unwrap();
        let d = Tensor::from_vec(data(3 * h, 8), (3, h), &dev).unwrap();
        let w = Tensor::from_vec(data(h, 9), h, &dev).unwrap();
        let got = add_rmsnorm(&x, &d, &w, 1e-6).unwrap();
        let want = add_rmsnorm_composed(&x, &d, &w, 1e-6).unwrap();
        assert!(max_diff(&got, &want) <= 1e-5);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn swiglu_matches_composed_ops_on_metal() {
        let dev = Device::new_metal(0).unwrap();
        check_swiglu(&dev, DType::F32, 1e-5);
        check_swiglu(&dev, DType::F16, 1e-2);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn add_rmsnorm_matches_composed_ops_on_metal() {
        let dev = Device::new_metal(0).unwrap();
        check_add_rmsnorm(&dev, DType::F32, 1e-5);
        check_add_rmsnorm(&dev, DType::F16, 1e-2);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn add_rmsnorm_on_metal_takes_a_view_at_an_offset() {
        let dev = Device::new_metal(0).unwrap();
        let h = 1024;
        let stacked = Tensor::from_vec(data(2 * 3 * h, 7), (2, 3, h), &dev).unwrap();
        let x = stacked.get(1).unwrap();
        let d = Tensor::from_vec(data(3 * h, 8), (3, h), &dev).unwrap();
        let w = Tensor::from_vec(data(h, 9), h, &dev).unwrap();
        let got = add_rmsnorm(&x, &d, &w, 1e-6).unwrap();
        let want = add_rmsnorm_composed(&x, &d, &w, 1e-6).unwrap();
        assert!(max_diff(&got, &want) <= 1e-4);
    }
}
