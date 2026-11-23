// SPDX-License-Identifier: Apache-2.0
//! `hll_sketch_agg`, `hll_union_agg`, `hll_sketch_estimate` and `hll_union`, after Spark's
//! `datasketchesAggregates.scala` and `datasketchesExpressions.scala` (v4.2.0).
//!
//! Update encodings follow `HllSketchAgg.update`: `INT` is widened to `long`, `STRING` and `BINARY` hash
//! their raw bytes, and an empty string or empty binary is skipped.
//!
//! Known divergence: the crate writes the *compact* form, Spark the *updatable* one. Both libraries read
//! both, so sketches stay interchangeable; only `length`/`hex` of the bytes differ (design note, owner question 2).
//!
//! Known gap (Spark 4.3.0): 4.3 exempts an empty sketch from the lgConfigK check in `hll_union_agg` and
//! `hll_union` and takes the non-empty side's lgK; this module applies the 4.2 rules everywhere because
//! `SparkProfile::V4_3` exists now; key both by profile (queue row "P3-T086 HLL union 4.3").

use datasketches::hash::value::raw_bytes;
pub use datasketches::hll::{HllSketch, HllType, HllUnion};
use krishiv_common::error::{
    HllInvalidInputSketchBuffer, HllInvalidLgK, HllUnionDifferentLgK, KrishivError,
};

/// Smallest `lgConfigK` Spark accepts.
pub const MIN_LG_K: i32 = 4;
/// Largest `lgConfigK` Spark accepts.
pub const MAX_LG_K: i32 = 21;
/// `lgConfigK` when the argument is omitted (the library's `DEFAULT_LG_K`).
pub const DEFAULT_LG_K: i32 = 12;

/// The function name Spark puts in `HLL_INVALID_LG_K` (hard-coded in Spark, whatever function raised it).
const LG_K_FUNCTION: &str = "`hll_sketch_agg`";

/// Check `lg_k` is in `4..=21`, else `HLL_INVALID_LG_K`.
///
/// # Errors
/// `HLL_INVALID_LG_K` for a value outside the range.
pub fn check_lg_k(lg_k: i32) -> Result<u8, KrishivError> {
    u8::try_from(lg_k)
        .ok()
        .filter(|k| (MIN_LG_K..=MAX_LG_K).contains(&i32::from(*k)))
        .ok_or_else(|| {
            HllInvalidLgK {
                function: LG_K_FUNCTION.to_owned(),
                min: MIN_LG_K.to_string(),
                max: MAX_LG_K.to_string(),
                value: lg_k.to_string(),
            }
            .into()
        })
}

fn invalid_buffer(function: &str) -> KrishivError {
    HllInvalidInputSketchBuffer {
        function: format!("`{function}`"),
    }
    .into()
}

/// A new `HLL_8` sketch of `lg_k`.
///
/// # Errors
/// `HLL_INVALID_LG_K` when `lg_k` is out of range.
pub fn new_sketch(lg_k: i32) -> Result<HllSketch, KrishivError> {
    let k = check_lg_k(lg_k)?;
    HllSketch::new(k, HllType::Hll8).map_err(|_| {
        HllInvalidLgK {
            function: LG_K_FUNCTION.to_owned(),
            min: MIN_LG_K.to_string(),
            max: MAX_LG_K.to_string(),
            value: lg_k.to_string(),
        }
        .into()
    })
}

/// `INT` and `BIGINT` inputs: both hash as a `long`.
pub fn update_long(sketch: &mut HllSketch, value: i64) {
    sketch.update(value);
}

/// `STRING` input: the UTF-8 bytes; the empty string is ignored.
pub fn update_str(sketch: &mut HllSketch, value: &str) {
    if !value.is_empty() {
        sketch.update(raw_bytes::from_str(value));
    }
}

/// `BINARY` input: the raw bytes; the empty array is ignored.
pub fn update_bytes(sketch: &mut HllSketch, value: &[u8]) {
    if !value.is_empty() {
        sketch.update(raw_bytes::from_slice(value));
    }
}

/// The function name Spark puts in `HLL_UNION_DIFFERENT_LG_K` / `HLL_INVALID_INPUT_SKETCH_BUFFER` for `hll_union_agg`.
pub const UNION_AGG: &str = "hll_union_agg";

/// The sketch of `bytes` as `hll_union_agg` reads it, joined into `union` (created on the first sketch at its lgK,
/// as Spark's `HllUnionAgg.update`).
///
/// # Errors
/// `HLL_INVALID_INPUT_SKETCH_BUFFER` for a bad buffer, `HLL_UNION_DIFFERENT_LG_K` when the lgK values differ
/// and `allow_different` is false.
pub fn union_agg_update(
    union: &mut Option<HllUnion>,
    bytes: &[u8],
    allow_different: bool,
) -> Result<(), KrishivError> {
    let sketch = from_bytes(bytes, UNION_AGG)?;
    if let Some(u) = union.as_ref() {
        // The union's lgK is the one it started with (`HllUnion::lg_max_k`).
        if !allow_different && u.lg_max_k() != sketch.lg_config_k() {
            return Err(HllUnionDifferentLgK {
                left: u.lg_max_k().to_string(),
                right: sketch.lg_config_k().to_string(),
                function: format!("`{UNION_AGG}`"),
            }
            .into());
        }
    }
    if union.is_none() {
        *union = Some(HllUnion::new(sketch.lg_config_k()).map_err(|_| invalid_buffer(UNION_AGG))?);
    }
    if let Some(u) = union.as_mut() {
        u.update(&sketch);
    }
    Ok(())
}

/// The bytes Krishiv stores in a `BINARY` column (compact form, see the module note).
#[must_use]
pub fn to_bytes(sketch: &HllSketch) -> Vec<u8> {
    sketch.serialize()
}

/// Read a sketch buffer, as `function` would.
///
/// # Errors
/// `HLL_INVALID_INPUT_SKETCH_BUFFER` when the bytes are not an HLL sketch.
pub fn from_bytes(bytes: &[u8], function: &str) -> Result<HllSketch, KrishivError> {
    HllSketch::deserialize(bytes).map_err(|_| invalid_buffer(function))
}

/// `hll_sketch_estimate`: `Math.round(getEstimate)`.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "an estimate is far below i64::MAX"
)]
pub fn estimate(sketch: &HllSketch) -> i64 {
    sketch.estimate().round() as i64
}

/// `hll_union(a, b[, allowDifferentLgConfigK])`: the union of two sketches, written as `HLL_8` at the smaller lgK.
///
/// # Errors
/// `HLL_UNION_DIFFERENT_LG_K` when the lgK values differ and `allow_different` is false.
pub fn union(
    a: &HllSketch,
    b: &HllSketch,
    allow_different: bool,
    function: &str,
) -> Result<HllSketch, KrishivError> {
    if !allow_different && a.lg_config_k() != b.lg_config_k() {
        return Err(HllUnionDifferentLgK {
            left: a.lg_config_k().to_string(),
            right: b.lg_config_k().to_string(),
            function: format!("`{function}`"),
        }
        .into());
    }
    let mut u = HllUnion::new(a.lg_config_k().min(b.lg_config_k()))
        .map_err(|_| invalid_buffer(function))?;
    u.update(a);
    u.update(b);
    Ok(u.to_sketch(HllType::Hll8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lg_k_bounds() {
        assert!(check_lg_k(3).is_err());
        assert_eq!(check_lg_k(4).unwrap(), 4);
        assert_eq!(check_lg_k(21).unwrap(), 21);
        assert!(check_lg_k(22).is_err());
        assert!(check_lg_k(-1).is_err());
    }

    #[test]
    fn small_inputs_count_exactly_and_empties_are_skipped() {
        let mut s = new_sketch(DEFAULT_LG_K).unwrap();
        for v in [1_i64, 2, 2, 3] {
            update_long(&mut s, v);
        }
        assert_eq!(estimate(&s), 3);
        let mut t = new_sketch(DEFAULT_LG_K).unwrap();
        update_str(&mut t, "");
        update_bytes(&mut t, b"");
        update_str(&mut t, "a");
        update_bytes(&mut t, b"a");
        assert_eq!(estimate(&t), 1, "a string and the same bytes hash alike");
    }

    /// Buffers are untrusted input: every truncation and a spread of bit flips of valid sketches (list, set and
    /// array modes) must come back as an error or a readable sketch, never a panic.
    #[test]
    fn mutated_and_truncated_buffers_never_panic() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (lg_k, n) in [(4_u8, 3_i64), (8, 40), (8, 5_000), (12, 100_000)] {
            let mut s = new_sketch(i32::from(lg_k)).unwrap();
            for v in 0..n {
                update_long(&mut s, v);
            }
            let bytes = to_bytes(&s);
            for len in 0..bytes.len().min(300) {
                let _ = from_bytes(&bytes[..len], "hll_sketch_estimate");
            }
            for _ in 0..300 {
                let mut m = bytes.clone();
                let flips = 1 + next() % 4;
                for _ in 0..flips {
                    let i = usize::try_from(next()).unwrap() % m.len();
                    m[i] ^= 1 << (next() % 8);
                }
                if let Ok(back) = from_bytes(&m, "hll_sketch_estimate") {
                    let _ = estimate(&back);
                }
            }
        }
    }

    #[test]
    fn bytes_round_trip_and_garbage_is_rejected() {
        let mut s = new_sketch(10).unwrap();
        update_long(&mut s, 7);
        let back = from_bytes(&to_bytes(&s), "hll_sketch_estimate").unwrap();
        assert_eq!(estimate(&back), 1);
        assert!(from_bytes(b"not a sketch", "hll_sketch_estimate").is_err());
        assert!(from_bytes(&[], "hll_sketch_estimate").is_err());
    }

    #[test]
    fn union_checks_lg_k() {
        let mut a = new_sketch(10).unwrap();
        let mut b = new_sketch(12).unwrap();
        update_long(&mut a, 1);
        update_long(&mut b, 2);
        assert!(union(&a, &b, false, "hll_union").is_err());
        let u = union(&a, &b, true, "hll_union").unwrap();
        assert_eq!(u.lg_config_k(), 10);
        assert_eq!(estimate(&u), 2);
    }
}
