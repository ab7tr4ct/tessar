//! The meaning channel's index: one int8 vector per fact version, searched
//! by brute force.
//!
//! Each vector is cut to the index length, normalized to unit length and
//! stored as `i8` with its own scale: `x ≈ q * scale`. The cosine
//! similarity of two vectors is then `dot_i8(qa, qb) * scale_a * scale_b`.
//! A quarter of the memory of `f32`, and the plain integer dot product
//! compiles to SIMD (SDOT on ARM). Brute force is exact and deterministic;
//! an approximate index comes when a space outgrows it (trade-off K4).

use crate::embed::EmbedError;
use crate::model::FactId;

/// A vector cut, normalized and quantized for the index.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Quantized {
    values: Vec<i8>,
    scale: f32,
}

impl Quantized {
    /// Cuts `v` to `dims`, normalizes and quantizes it.
    pub(crate) fn new(v: &[f32], dims: usize) -> Result<Self, EmbedError> {
        if v.len() < dims {
            return Err(EmbedError(format!(
                "vector of length {}, expected at least {dims}",
                v.len()
            )));
        }
        let v = &v[..dims];
        // The norm in f64: stable and precise on long vectors.
        let mut norm_sq = 0.0f64;
        let mut max_abs = 0.0f32;
        for &x in v {
            if !x.is_finite() {
                return Err(EmbedError("vector has a non-finite value".into()));
            }
            norm_sq += f64::from(x) * f64::from(x);
            max_abs = max_abs.max(x.abs());
        }
        if norm_sq == 0.0 {
            return Err(EmbedError("vector is zero".into()));
        }
        let inv_norm = (1.0 / norm_sq.sqrt()) as f32;
        // The largest component of the normalized vector maps to ±127.
        let scale = max_abs * inv_norm / 127.0;
        let inv_scale = inv_norm / scale;
        let values = v
            .iter()
            .map(|&x| (x * inv_scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Ok(Self { values, scale })
    }
}

/// Vectors of every fact version, by fact index.
#[derive(Debug)]
pub(crate) struct VectorIndex {
    dims: usize,
    values: Vec<i8>,
    scales: Vec<f32>,
}

impl VectorIndex {
    pub(crate) fn new(dims: usize) -> Self {
        Self {
            dims,
            values: Vec::new(),
            scales: Vec::new(),
        }
    }

    /// How many facts have vectors.
    pub(crate) fn len(&self) -> usize {
        self.scales.len()
    }

    /// Adds the vector of the next fact. Facts must come in id order.
    pub(crate) fn push(&mut self, fact: FactId, vector: Quantized) -> Result<(), String> {
        if fact.index() != self.len() {
            return Err(format!(
                "vector index expected fact {}, got {fact}",
                self.len()
            ));
        }
        if vector.values.len() != self.dims {
            return Err(format!(
                "vector of length {}, index holds {}",
                vector.values.len(),
                self.dims
            ));
        }
        self.values.extend_from_slice(&vector.values);
        self.scales.push(vector.scale);
        Ok(())
    }

    /// Cosine similarity of a fact to a query, if the fact has a vector.
    pub(crate) fn similarity(&self, query: &Quantized, fact: FactId) -> Option<f32> {
        let i = fact.index();
        let scale = *self.scales.get(i)?;
        let v = &self.values[i * self.dims..(i + 1) * self.dims];
        Some(dot_i8(&query.values, v) as f32 * scale * query.scale)
    }
}

/// Dot product of two int8 vectors of the same length.
///
/// A plain sum on purpose: LLVM recognizes it and picks the instruction
/// (SDOT on ARM). Integer addition is associative, so the result does not
/// depend on how the compiler splits the sum. No overflow:
/// |127 · 127 · dims| < `i32::MAX` for dims < 133 000.
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(len: usize, phase: f32) -> Vec<f32> {
        (0..len).map(|i| (i as f32 * 0.37 + phase).sin()).collect()
    }

    #[test]
    fn self_similarity_is_one() {
        let q = Quantized::new(&wave(384, 0.0), 384).unwrap();
        let mut index = VectorIndex::new(384);
        index.push(FactId::new(0), q.clone()).unwrap();
        let s = index.similarity(&q, FactId::new(0)).unwrap();
        assert!((s - 1.0).abs() < 0.01, "self similarity = {s}");
    }

    #[test]
    fn similarity_follows_cosine() {
        let (a, b) = (wave(256, 0.0), wave(256, 0.8));
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let exact = dot / (norm(&a) * norm(&b));

        let mut index = VectorIndex::new(256);
        index
            .push(FactId::new(0), Quantized::new(&b, 256).unwrap())
            .unwrap();
        let s = index
            .similarity(&Quantized::new(&a, 256).unwrap(), FactId::new(0))
            .unwrap();
        assert!((s - exact).abs() < 0.01, "{s} vs {exact}");
    }

    #[test]
    fn long_vectors_are_cut_and_renormalized() {
        let long = wave(2048, 0.0);
        let q = Quantized::new(&long, 1024).unwrap();
        assert_eq!(q, Quantized::new(&long[..1024], 1024).unwrap());
    }

    #[test]
    fn rejects_bad_vectors() {
        assert!(Quantized::new(&[1.0, 2.0], 3).is_err(), "too short");
        assert!(Quantized::new(&[0.0, 0.0], 2).is_err(), "zero");
        assert!(Quantized::new(&[1.0, f32::NAN], 2).is_err(), "NaN");
        assert!(Quantized::new(&[f32::INFINITY], 1).is_err(), "infinite");
    }

    #[test]
    fn vectors_must_come_in_order() {
        let mut index = VectorIndex::new(2);
        let q = Quantized::new(&[1.0, 0.0], 2).unwrap();
        assert!(index.push(FactId::new(1), q.clone()).is_err());
        index.push(FactId::new(0), q).unwrap();
        assert!(
            index
                .similarity(&Quantized::new(&[0.0, 1.0], 2).unwrap(), FactId::new(1))
                .is_none()
        );
    }

    #[test]
    fn dot_matches_naive_for_odd_lengths() {
        for len in [0usize, 1, 7, 31, 32, 33, 64, 385] {
            let a: Vec<i8> = (0..len).map(|i| ((i * 37 + 11) % 255) as i8).collect();
            let b: Vec<i8> = (0..len).map(|i| ((i * 91 + 5) % 255) as i8).collect();
            let naive: i32 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| i32::from(x) * i32::from(y))
                .sum();
            assert_eq!(dot_i8(&a, &b), naive, "len = {len}");
        }
    }
}
