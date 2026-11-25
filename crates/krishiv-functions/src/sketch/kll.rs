// SPDX-License-Identifier: Apache-2.0
//! The KLL quantile sketches behind `kll_sketch_*` and `kll_merge_*` (Spark 4.1+, `kllExpressions.scala` and
//! `kllAggregates.scala` at v4.3.0), over `datasketches::kll`. Spark's three element types are `BIGINT`
//! (`KllLongsSketch`), `FLOAT` (`KllFloatsSketch`) and `DOUBLE` (`KllDoublesSketch`); a buffer that does not decode as
//! the function's type is `KLL_INVALID_INPUT_SKETCH_BUFFER`, a rank outside `[0, 1]` is
//! `KLL_SKETCH_INVALID_QUANTILE_RANGE`.
//!
//! Known difference: [`Kll::summary`] follows Java's `KllSketch.toString()` layout; only its first line is pinned by
//! a Spark golden (`### KllKllLongsSketch Summary:`, the doubled `Kll` is Spark's output).

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

/// An upper bound of the bytes a sketch of `k` holds: its levels keep at most about `3k` items of 8 bytes, plus the
/// level bookkeeping (a slack of 512 items) and a header.
#[must_use]
pub fn capacity_bytes(k: u16) -> usize {
    (3 * usize::from(k) + 512) * 8 + 256
}

/// The `k` in the preamble of a serialized sketch (bytes 4 and 5, little endian); `None` when too short.
#[must_use]
pub fn k_of_bytes(bytes: &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes([*bytes.get(4)?, *bytes.get(5)?]))
}

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

    /// Human-readable summary, after Java's `KllSketch.toString()`.
    #[must_use]
    pub fn summary(&self) -> String {
        let (name, k, min_k, n, retained, est) = match self {
            Self::Long(s) => (
                "KllKllLongsSketch",
                s.k(),
                s.min_k(),
                s.n(),
                s.num_retained(),
                s.is_estimation_mode(),
            ),
            Self::Float(s) => (
                "KllKllFloatsSketch",
                s.k(),
                s.min_k(),
                s.n(),
                s.num_retained(),
                s.is_estimation_mode(),
            ),
            Self::Double(s) => (
                "KllKllDoublesSketch",
                s.k(),
                s.min_k(),
                s.n(),
                s.num_retained(),
                s.is_estimation_mode(),
            ),
        };
        let bytes = self.to_bytes().len();
        format!(
            "\n### {name} Summary:\n   K                      : {k}\n   Dynamic min K          : {min_k}\n   \
             N                      : {n}\n   Retained Items         : {retained}\n   \
             Estimation Mode        : {est}\n   Compact Storage Bytes  : {bytes}\n### End sketch summary\n"
        )
    }
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
