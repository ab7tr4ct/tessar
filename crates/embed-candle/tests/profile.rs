//! Micro-benchmark of candle's Metal kernels at the shapes of one batch of
//! 64 documents (about 64 x 40 tokens). Ignored; needs the `metal` feature.

#![cfg(feature = "metal")]

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, ops::softmax_last_dim};

fn time(name: &str, flops: f64, mut f: impl FnMut() -> Tensor) {
    let dev = f().device().clone();
    for _ in 0..3 {
        f();
    }
    dev.synchronize().unwrap();
    let t = Instant::now();
    let n = 20;
    for _ in 0..n {
        f();
    }
    dev.synchronize().unwrap();
    let s = t.elapsed().as_secs_f64() / f64::from(n);
    println!(
        "PROF {name}: {:.2} ms ({:.2} TFLOPS)",
        s * 1e3,
        flops / s / 1e12
    );
}

#[test]
#[ignore = "needs GPU"]
fn kernels() {
    let dev = Device::new_metal(0).unwrap();
    let dt = DType::F16;
    let (b, l, h, kv, d) = (64usize, 40usize, 16usize, 8usize, 128usize);
    let x = Tensor::randn(0f32, 1.0, (b, l, 1024), &dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap();
    let mk = |o, i| {
        Linear::new(
            Tensor::randn(0f32, 0.05, (o, i), &dev)
                .unwrap()
                .to_dtype(dt)
                .unwrap(),
            None,
        )
    };
    let toks = (b * l) as f64;
    let up = mk(3072, 1024);
    time(
        "linear 1024->3072 (3D input)",
        2.0 * toks * 1024.0 * 3072.0,
        || up.forward(&x).unwrap(),
    );
    let q = mk(h * d, 1024);
    time(
        "linear 1024->2048 (q_proj)",
        2.0 * toks * 1024.0 * 2048.0,
        || q.forward(&x).unwrap(),
    );
    let x2 = x.reshape((b * l, 1024)).unwrap();
    let w = Tensor::randn(0f32, 0.05, (1024, 3072), &dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap();
    time(
        "matmul 2D [2560,1024]x[1024,3072]",
        2.0 * toks * 1024.0 * 3072.0,
        || x2.matmul(&w).unwrap(),
    );
    let qh = Tensor::randn(0f32, 1.0, (b, h, l, d), &dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap();
    let kh = qh.clone();
    time(
        "scores q.kT batched",
        2.0 * (b * h * l * l * d) as f64,
        || {
            qh.matmul(&kh.transpose(2, 3).unwrap().contiguous().unwrap())
                .unwrap()
        },
    );
    let sc = Tensor::randn(0f32, 1.0, (b, h, l, l), &dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap();
    time("softmax", 1.0, || softmax_last_dim(&sc).unwrap());
    let kvt = Tensor::randn(0f32, 1.0, (b, kv, l, d), &dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap();
    time("repeat_kv expand+reshape", 1.0, || {
        kvt.unsqueeze(2)
            .unwrap()
            .expand((b, kv, 2, l, d))
            .unwrap()
            .reshape((b, h, l, d))
            .unwrap()
    });
    let wn = Tensor::ones(1024, dt, &dev).unwrap();
    time("rms_norm", 1.0, || {
        candle_nn::ops::rms_norm(&x, &wn, 1e-6).unwrap()
    });
    let cos = Tensor::randn(0f32, 1.0, (l, d / 2), &dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap();
    time("rope", 1.0, || {
        candle_nn::rotary_emb::rope(&qh, &cos, &cos).unwrap()
    });
    time("silu*mul", 1.0, || {
        let a = up.forward(&x).unwrap();
        (candle_nn::ops::silu(&a).unwrap() * &a).unwrap()
    });
}
