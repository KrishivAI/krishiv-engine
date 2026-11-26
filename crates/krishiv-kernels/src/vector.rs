// SPDX-License-Identifier: Apache-2.0
//! The `vector_*` kernels over `float` slices, bit for bit what Spark 4.2.0's
//! `VectorFunctionImplUtils` computes.
//!
//! The result of a float sum depends on its order, and Spark's order is
//! fixed: blocks of 8 elements, each block summed left to right and then
//! added to the accumulator, the tail added one element at a time, all in
//! `float`, with no fused multiply-add (Java never fuses). The kernels here
//! do exactly those operations in that order; nothing is reassociated, so a
//! 768-dimension result equals Spark's in every bit. The 8 products of a
//! block are independent, so the compiler still vectorizes them; the adds
//! stay in Spark's order.
//!
//! These functions take vectors already known to be free of NULL elements
//! and of equal length (the caller checks both, in Spark's order: a length
//! mismatch raises before a NULL element is looked at).
//!
//! Spark 4.3.0 accumulates in `double` and casts the result; that profile is
//! a separate set of kernels once the engine has a 4.3 version profile.

/// Spark's block sum: `((((((t0 + t1) + t2) + t3) + t4) + t5) + t6) + t7`.
#[inline(always)]
fn block_sum(t: [f32; 8]) -> f32 {
    t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7]
}

/// `sum(term(a[i], b[i]))` in Spark's order.
#[inline(always)]
fn sum2(a: &[f32], b: &[f32], term: impl Fn(f32, f32) -> f32) -> f32 {
    let (a_blocks, a_tail) = a.as_chunks::<8>();
    let (b_blocks, b_tail) = b.as_chunks::<8>();
    let mut acc = 0.0_f32;
    for (x, y) in a_blocks.iter().zip(b_blocks) {
        acc += block_sum(std::array::from_fn(|i| term(x[i], y[i])));
    }
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        acc += term(x, y);
    }
    acc
}

/// `sum(term(v[i]))` in Spark's order.
#[inline(always)]
fn sum1(v: &[f32], term: impl Fn(f32) -> f32) -> f32 {
    let (blocks, tail) = v.as_chunks::<8>();
    let mut acc = 0.0_f32;
    for x in blocks {
        acc += block_sum(std::array::from_fn(|i| term(x[i])));
    }
    for &x in tail {
        acc += term(x);
    }
    acc
}

/// `vector_inner_product`: `0.0` for empty vectors.
#[must_use]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    sum2(a, b, |x, y| x * y)
}

/// `vector_l2_distance`: `0.0` for empty vectors.
#[must_use]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    sum2(a, b, |x, y| {
        let d = x - y;
        d * d
    })
    .sqrt()
}

/// `vector_cosine_similarity`: `None` (NULL) for empty vectors and when the
/// product of the squared norms, rooted, is below `Float.MIN_NORMAL` (a NaN
/// is not below it and yields NaN).
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.is_empty() {
        return None;
    }
    let (a_blocks, a_tail) = a.as_chunks::<8>();
    let (b_blocks, b_tail) = b.as_chunks::<8>();
    let (mut dot, mut norm_a, mut norm_b) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (x, y) in a_blocks.iter().zip(b_blocks) {
        dot += block_sum(std::array::from_fn(|i| x[i] * y[i]));
        norm_a += block_sum(std::array::from_fn(|i| x[i] * x[i]));
        norm_b += block_sum(std::array::from_fn(|i| y[i] * y[i]));
    }
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product < f32::MIN_POSITIVE {
        return None;
    }
    Some(dot / norm_product)
}

/// The L1 norm: `0.0` for an empty vector.
#[must_use]
pub fn l1_norm(v: &[f32]) -> f32 {
    sum1(v, f32::abs)
}

/// The L2 norm: `0.0` for an empty vector.
#[must_use]
pub fn l2_norm(v: &[f32]) -> f32 {
    sum1(v, |x| x * x).sqrt()
}

/// The infinity norm. A NaN element never exceeds the running maximum, so
/// it is skipped (4.2.0).
#[must_use]
pub fn inf_norm(v: &[f32]) -> f32 {
    v.iter().fold(0.0_f32, |m, &x| {
        let a = x.abs();
        if a > m { a } else { m }
    })
}

/// The infinity norm as Spark 4.3.0's `vectorInfNorm` computes it:
/// `Math.max(maxAbs, Math.abs(x))`, which propagates a NaN element.
#[must_use]
pub fn inf_norm_nan(v: &[f32]) -> f32 {
    // `f32::max` ignores a NaN; Java's `Math.max` returns it.
    v.iter().fold(0.0_f32, |m, &x| {
        let a = x.abs();
        if m.is_nan() || a.is_nan() {
            f32::NAN
        } else {
            m.max(a)
        }
    })
}

/// Whether a vector with this norm can be divided by it: a norm below
/// `Float.MIN_NORMAL` (zero or subnormal) gives NULL; a NaN norm is not
/// below it and gives NaN elements.
#[must_use]
pub fn is_normalizable(norm: f32) -> bool {
    norm.is_nan() || norm >= f32::MIN_POSITIVE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plain left-to-right fold: what the engine did before, wrong from 8
    /// elements up.
    fn folded(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0, |s, (x, y)| s + x * y)
    }

    fn pseudo(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                #[expect(clippy::cast_precision_loss, reason = "test data")]
                let u = (s >> 40) as f32 / (1_u64 << 24) as f32;
                u - 0.5
            })
            .collect()
    }

    #[test]
    fn the_block_order_is_not_a_plain_fold() {
        let (a, b) = (pseudo(768, 1), pseudo(768, 2));
        // The two orders disagree at 768 dimensions (the reason for this
        // module); both are within a few ULP of each other.
        assert_ne!(inner_product(&a, &b).to_bits(), folded(&a, &b).to_bits());
        assert!((inner_product(&a, &b) - folded(&a, &b)).abs() < 1e-3);
    }

    #[test]
    fn under_eight_elements_the_order_is_the_plain_fold() {
        let (a, b) = (pseudo(7, 3), pseudo(7, 4));
        assert_eq!(inner_product(&a, &b).to_bits(), folded(&a, &b).to_bits());
    }

    #[test]
    fn nan_is_skipped_by_the_infinity_norm_and_poisons_the_rest() {
        let v = [1.0, f32::NAN, 2.0];
        assert_eq!(inf_norm(&v), 2.0);
        assert!(l2_norm(&v).is_nan() && l1_norm(&v).is_nan());
    }

    #[test]
    fn the_43_infinity_norm_propagates_nan() {
        assert!(inf_norm_nan(&[1.0, f32::NAN, 2.0]).is_nan());
        assert!(inf_norm_nan(&[f32::NAN, 5.0]).is_nan());
        assert_eq!(inf_norm_nan(&[1.0, -3.0, 2.0]), 3.0);
        assert_eq!(inf_norm_nan(&[]), 0.0);
    }

    #[test]
    fn a_subnormal_norm_product_is_null_but_nan_is_not() {
        assert_eq!(cosine_similarity(&[], &[]), None);
        assert_eq!(cosine_similarity(&[1e-20, 1e-20], &[1e-20, 1e-20]), None);
        assert!(cosine_similarity(&[f32::NAN], &[1.0]).is_some_and(f32::is_nan));
        assert!(!is_normalizable(0.0) && !is_normalizable(1e-40));
        assert!(is_normalizable(f32::NAN) && is_normalizable(1e-20));
    }
}
