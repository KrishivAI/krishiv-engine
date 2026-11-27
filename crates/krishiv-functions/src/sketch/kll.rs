// SPDX-License-Identifier: Apache-2.0
//! The KLL quantile sketches behind `kll_sketch_*` and `kll_merge_*` (Spark 4.1+, `kllExpressions.scala` and
//! `kllAggregates.scala` at v4.3.0), over `datasketches::kll`. Spark's three element types are `BIGINT`
//! (`KllLongsSketch`), `FLOAT` (`KllFloatsSketch`) and `DOUBLE` (`KllDoublesSketch`); a buffer that does not decode as
//! the function's type is `KLL_INVALID_INPUT_SKETCH_BUFFER`, a rank outside `[0, 1]` is
//! `KLL_SKETCH_INVALID_QUANTILE_RANGE`.
//!
//! [`Kll::summary`] is Java's `KllSketch.toString()` in full (the doubled `Kll` in the title is Spark's output).

use datasketches::common::SearchCriteria;
use datasketches::kll::{KllFloat, KllSketch};
use krishiv_common::error::{
    KllInvalidInputSketchBuffer, KllSketchInvalidQuantileRange, KrishivError,
};

/// Spark's default `k` (`KllSketch.DEFAULT_K`).
pub const DEFAULT_K: u16 = 200;
/// Smallest and largest `k` Spark accepts.
pub const MIN_K: i32 = 8;
/// Largest `k` Spark accepts.
pub const MAX_K: i32 = 65_535;

/// The element type of a KLL function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KllType {
    /// `BIGINT`.
    Bigint,
    /// `FLOAT`.
    Float,
    /// `DOUBLE`.
    Double,
}

/// A decoded KLL sketch of one element type.
#[derive(Debug, Clone)]
pub enum Kll {
    /// `kll_*_bigint`.
    Long(KllSketch<i64>),
    /// `kll_*_float`.
    Float(KllSketch<KllFloat<f32>>),
    /// `kll_*_double`.
    Double(KllSketch<KllFloat<f64>>),
}

fn invalid_buffer(function: &str) -> KrishivError {
    KllInvalidInputSketchBuffer {
        function: format!("`{function}`"),
    }
    .into()
}

fn invalid_range(function: &str) -> KrishivError {
    KllSketchInvalidQuantileRange {
        function_name: format!("`{function}`"),
    }
    .into()
}

impl Kll {
    /// An empty sketch of `ty` with `k`.
    ///
    /// # Errors
    /// `k` out of the crate's range (callers check Spark's `8..=65535` first).
    pub fn new(ty: KllType, k: u16, function: &str) -> Result<Self, KrishivError> {
        let bad = |_| invalid_buffer(function);
        Ok(match ty {
            KllType::Bigint => Self::Long(KllSketch::new(k).map_err(bad)?),
            KllType::Float => Self::Float(KllSketch::new(k).map_err(bad)?),
            KllType::Double => Self::Double(KllSketch::new(k).map_err(bad)?),
        })
    }

    /// Decode `bytes` as a sketch of `ty`.
    ///
    /// # Errors
    /// `KLL_INVALID_INPUT_SKETCH_BUFFER` naming `function`.
    pub fn from_bytes(ty: KllType, bytes: &[u8], function: &str) -> Result<Self, KrishivError> {
        let bad = |_| invalid_buffer(function);
        Ok(match ty {
            KllType::Bigint => Self::Long(KllSketch::deserialize(bytes).map_err(bad)?),
            KllType::Float => Self::Float(KllSketch::deserialize(bytes).map_err(bad)?),
            KllType::Double => Self::Double(KllSketch::deserialize(bytes).map_err(bad)?),
        })
    }

    /// The serialized (compact) image, Spark's `toByteArray`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Long(s) => s.serialize(),
            Self::Float(s) => s.serialize(),
            Self::Double(s) => s.serialize(),
        }
    }

    /// The sketch's `k`.
    #[must_use]
    pub fn k(&self) -> u16 {
        match self {
            Self::Long(s) => s.k(),
            Self::Float(s) => s.k(),
            Self::Double(s) => s.k(),
        }
    }

    /// The number of items the sketch keeps.
    #[must_use]
    pub fn retained(&self) -> usize {
        match self {
            Self::Long(s) => s.num_retained(),
            Self::Float(s) => s.num_retained(),
            Self::Double(s) => s.num_retained(),
        }
    }

    /// The number of items fed to the sketch (`getN`).
    #[must_use]
    pub fn n(&self) -> u64 {
        match self {
            Self::Long(s) => s.n(),
            Self::Float(s) => s.n(),
            Self::Double(s) => s.n(),
        }
    }

    /// Merge `other` (same variant) into `self`.
    ///
    /// # Errors
    /// `KLL_INVALID_INPUT_SKETCH_BUFFER` when the variants differ or the merge is refused.
    pub fn merge(&mut self, other: &Self, function: &str) -> Result<(), KrishivError> {
        let r = match (self, other) {
            (Self::Long(a), Self::Long(b)) => a.merge(b),
            (Self::Float(a), Self::Float(b)) => a.merge(b),
            (Self::Double(a), Self::Double(b)) => a.merge(b),
            _ => return Err(invalid_buffer(function)),
        };
        r.map_err(|_| invalid_buffer(function))
    }

    /// Add one non-NULL value (a NaN float is skipped, as Spark's `update` does).
    pub fn update_long(&mut self, v: i64) {
        if let Self::Long(s) = self {
            s.update(v);
        }
    }

    /// Add one `FLOAT` value.
    pub fn update_f32(&mut self, v: f32) {
        if let (Self::Float(s), Ok(x)) = (self, KllFloat::<f32>::new(v)) {
            s.update(x);
        }
    }

    /// Add one `DOUBLE` value.
    pub fn update_f64(&mut self, v: f64) {
        if let (Self::Double(s), Ok(x)) = (self, KllFloat::<f64>::new(v)) {
            s.update(x);
        }
    }

    /// The value at normalized rank `rank` as `f64` bits of the element type: a `BIGINT` as `i64`, floats widened.
    /// Returns `Ok(None)` for an empty sketch.
    ///
    /// # Errors
    /// `KLL_SKETCH_INVALID_QUANTILE_RANGE` for a rank outside `[0, 1]` (or NaN).
    pub fn quantile(&self, rank: f64, function: &str) -> Result<Option<Quantile>, KrishivError> {
        if !(0.0..=1.0).contains(&rank) {
            return Err(invalid_range(function));
        }
        let bad = |_| invalid_buffer(function);
        Ok(match self {
            Self::Long(s) => s
                .quantile(rank, SearchCriteria::Inclusive)
                .map_err(bad)?
                .map(Quantile::Long),
            Self::Float(s) => s
                .quantile(rank, SearchCriteria::Inclusive)
                .map_err(bad)?
                .map(|v| Quantile::Float(v.into_inner())),
            Self::Double(s) => s
                .quantile(rank, SearchCriteria::Inclusive)
                .map_err(bad)?
                .map(|v| Quantile::Double(v.into_inner())),
        })
    }

    /// The normalized rank of `value` (inclusive); `None` for an empty sketch. `value` is converted to the element
    /// type (a NaN float has no rank).
    #[must_use]
    pub fn rank(&self, value: f64, long: i64) -> Option<f64> {
        match self {
            Self::Long(s) => s.rank(&long, SearchCriteria::Inclusive),
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the caller passes a FLOAT widened to f64"
            )]
            Self::Float(s) => KllFloat::<f32>::new(value as f32)
                .ok()
                .and_then(|v| s.rank(&v, SearchCriteria::Inclusive)),
            Self::Double(s) => KllFloat::<f64>::new(value)
                .ok()
                .and_then(|v| s.rank(&v, SearchCriteria::Inclusive)),
        }
    }

    /// Human-readable summary, Java's `KllSketch.toString()` (datasketches-java 6.2, `KllHelper.toStringImpl`): the
    /// 17 fields in Java's order. Levels, the level-zero flag and the item size come from the compact image, the
    /// capacities and the updatable size from Java's formulas. Pinned by an exact-text golden for an exact-mode
    /// `BIGINT` sketch; the empty form and the FLOAT/DOUBLE min and max text follow the Java source (not a Spark run).
    #[must_use]
    pub fn summary(&self) -> String {
        let (name, item_bytes) = match self {
            Self::Long(_) => ("KllKllLongsSketch", 8_u64),
            Self::Float(_) => ("KllKllFloatsSketch", 4),
            Self::Double(_) => ("KllKllDoublesSketch", 8),
        };
        let (k, min_k, n, retained, est, empty) = match self {
            Self::Long(s) => (
                s.k(),
                s.min_k(),
                s.n(),
                s.num_retained(),
                s.is_estimation_mode(),
                s.is_empty(),
            ),
            Self::Float(s) => (
                s.k(),
                s.min_k(),
                s.n(),
                s.num_retained(),
                s.is_estimation_mode(),
                s.is_empty(),
            ),
            Self::Double(s) => (
                s.k(),
                s.min_k(),
                s.n(),
                s.num_retained(),
                s.is_estimation_mode(),
                s.is_empty(),
            ),
        };
        let (rank_error, pmf_error) = match self {
            Self::Long(s) => (s.normalized_rank_error(), s.normalized_pmf_error()),
            Self::Float(s) => (s.normalized_rank_error(), s.normalized_pmf_error()),
            Self::Double(s) => (s.normalized_rank_error(), s.normalized_pmf_error()),
        };
        let bytes = self.to_bytes();
        let m = bytes.get(6).copied().unwrap_or(8);
        // Flags are byte 3: level zero sorted is bit 1. Levels are byte 18 of a full image (1 for the short ones).
        let sorted = bytes.get(3).is_some_and(|f| f & 2 != 0);
        let levels = if empty || n == 1 {
            1
        } else {
            usize::from(bytes.get(18).copied().unwrap_or(1))
        };
        let capacity = u64::from(total_capacity(k, m, levels));
        let free = capacity.saturating_sub(retained as u64);
        let updatable = 20 + (levels as u64 + 1) * 4 + 2 * item_bytes + capacity * item_bytes;
        let (min_item, max_item) = self.extremes();
        format!(
            "\n### {name} Summary:\n   K                      : {k}\n   Dynamic min K          : {min_k}\n   \
             M                      : {m}\n   N                      : {n}\n   \
             Epsilon                : {:.3}%\n   Epsilon PMF            : {:.3}%\n   \
             Empty                  : {empty}\n   Estimation Mode        : {est}\n   \
             Levels                 : {levels}\n   Level 0 Sorted         : {sorted}\n   \
             Capacity Items         : {capacity}\n   Retained Items         : {retained}\n   \
             Free Space             : {free}\n   ReadOnly               : false\n   \
             Updatable Storage Bytes: {updatable}\n   Compact Storage Bytes  : {}\n   \
             Min Item               : {min_item}\n   Max Item               : {max_item}\n\
             ### End sketch summary\n",
            rank_error * 100.0,
            pmf_error * 100.0,
            bytes.len(),
        )
    }

    /// The minimum and maximum as Java prints them (an empty sketch: `Long.MAX_VALUE`/`MIN_VALUE`, or `NaN`).
    fn extremes(&self) -> (String, String) {
        match self {
            Self::Long(s) => (
                s.min_item().map_or(i64::MAX, |v| *v).to_string(),
                s.max_item().map_or(i64::MIN, |v| *v).to_string(),
            ),
            Self::Float(s) => (
                krishiv_kernels::text::java_float_string32(s.min_item().map_or(f32::NAN, |v| **v)),
                krishiv_kernels::text::java_float_string32(s.max_item().map_or(f32::NAN, |v| **v)),
            ),
            Self::Double(s) => (
                krishiv_kernels::text::java_double_string(s.min_item().map_or(f64::NAN, |v| **v)),
                krishiv_kernels::text::java_double_string(s.max_item().map_or(f64::NAN, |v| **v)),
            ),
        }
    }
}

/// Java's `KllHelper.computeTotalCapacity`: the items the updatable layout holds across `levels`.
fn total_capacity(k: u16, m: u8, levels: usize) -> u32 {
    const POW3: [u128; 31] = {
        let mut a = [1_u128; 31];
        let mut i = 1;
        while i < 31 {
            a[i] = a[i - 1] * 3;
            i += 1;
        }
        a
    };
    fn aux(k: u128, depth: usize) -> u128 {
        let tmp = (k << 1 << depth) / POW3[depth];
        (tmp + 1) >> 1
    }
    let mut total = 0_u32;
    for level in 0..levels {
        let depth = levels - level - 1;
        let cap = if depth <= 30 {
            aux(u128::from(k), depth)
        } else {
            let half = depth / 2;
            aux(aux(u128::from(k), half), depth - half)
        };
        total = total.saturating_add(u32::try_from(cap).unwrap_or(u32::MAX).max(u32::from(m)));
    }
    total
}

/// A quantile in its element type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quantile {
    /// `BIGINT`.
    Long(i64),
    /// `FLOAT`.
    Float(f32),
    /// `DOUBLE`.
    Double(f64),
}

#[cfg(test)]
mod summary_tests {
    use super::*;

    fn bigint(n: i64) -> Kll {
        let mut s = KllSketch::<i64>::new(200).unwrap();
        for i in 0..n {
            s.update(i);
        }
        Kll::Long(s)
    }

    /// Spark 4.3.0: `kll_sketch_to_string_bigint(kll_sketch_agg_bigint(id))` over `range(5)`.
    #[test]
    fn exact_mode_bigint_matches_spark() {
        let want = "\n### KllKllLongsSketch Summary:\n   K                      : 200\n   Dynamic min K          : 200\n   \
M                      : 8\n   N                      : 5\n   Epsilon                : 1.329%\n   \
Epsilon PMF            : 1.652%\n   Empty                  : false\n   Estimation Mode        : false\n   \
Levels                 : 1\n   Level 0 Sorted         : false\n   Capacity Items         : 200\n   \
Retained Items         : 5\n   Free Space             : 195\n   ReadOnly               : false\n   \
Updatable Storage Bytes: 1644\n   Compact Storage Bytes  : 80\n   Min Item               : 0\n   \
Max Item               : 4\n### End sketch summary\n";
        assert_eq!(bigint(5).summary(), want);
    }

    #[test]
    fn empty_and_estimation_forms_carry_every_field() {
        let empty = bigint(0).summary();
        assert!(
            empty.contains("   Empty                  : true\n"),
            "{empty}"
        );
        assert!(empty.contains("   Compact Storage Bytes  : 8\n"), "{empty}");
        assert_eq!(empty.lines().count(), 21, "{empty}");
        let big = bigint(10_000).summary();
        assert!(big.contains("   Estimation Mode        : true\n"), "{big}");
        assert!(!big.contains("   Levels                 : 1\n"), "{big}");
        assert_eq!(big.lines().count(), 21, "{big}");
    }

    #[test]
    fn capacities_follow_java() {
        assert_eq!(total_capacity(200, 8, 1), 200);
        assert_eq!(total_capacity(200, 8, 2), 200 + 133);
    }
}
