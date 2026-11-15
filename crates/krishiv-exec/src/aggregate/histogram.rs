// SPDX-License-Identifier: Apache-2.0
//! `histogram_numeric(col, nb)`: Spark's `NumericHistogram` (Hive's online
//! histogram of Ben-Haim and Tom-Tov, Spark 4.3.0
//! `sql/catalyst/src/main/java/org/apache/spark/sql/util/NumericHistogram.java`)
//! ported one-to-one: `add` (Algorithm 1), `merge` and `trim` (Algorithm 2),
//! with the tie-breaking PRNG `java.util.Random(31183)` of each histogram.
//!
//! The PRNG belongs to one histogram object, as in Spark: a group's histogram
//! starts with a fresh one, and a histogram rebuilt from partial state (what
//! Spark's `deserialize` does) starts with a fresh one too, so results depend
//! on the partitioning exactly where Spark's do (ties between equally close
//! bins). The partial state is Spark's `NumericHistogramSerializer` layout
//! (big-endian `nBins`, `nUsedBins`, then the `(x, y)` doubles).
//!
//! The result is `array<struct<x, y>>` where `x` has the input's type
//! (`spark.sql.legacy.histogramNumericPropagateInputType`, default true) and
//! `y` is `double`; a group without a value is NULL.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, Float32Array, Float64Array, StructArray};
use arrow_schema::{DataType, Fields, IntervalUnit, TimeUnit};

use super::accumulator::{decimal_unscaled_in, narrow};
use super::quantile::finish_lists;
use crate::error::{Result, internal};

/// Whether `t` is an input type of `histogram_numeric` (Spark's numeric,
/// date, timestamp and ANSI interval types; `TIME` is not run here).
pub(crate) fn histogram_supported(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Date32
            | DataType::Timestamp(TimeUnit::Microsecond, _)
            | DataType::Interval(IntervalUnit::YearMonth)
            | DataType::Duration(TimeUnit::Microsecond)
    )
}

/// `java.util.Random`'s 48-bit linear congruential generator.
#[derive(Clone, Debug)]
struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    const MULTIPLIER: u64 = 0x5_DEEC_E66D;
    const MASK: u64 = (1 << 48) - 1;
    /// Spark's fixed seed, "to aid testing".
    const SEED: u64 = 31183;

    fn new() -> Self {
        Self {
            seed: (Self::SEED ^ Self::MULTIPLIER) & Self::MASK,
        }
    }

    /// `next(bits)`: the top `bits` of the next 48-bit state.
    fn next(&mut self, bits: u32) -> u64 {
        self.seed = self.seed.wrapping_mul(Self::MULTIPLIER).wrapping_add(0xB) & Self::MASK;
        self.seed >> (48 - bits)
    }

    /// `nextDouble()`: 53 random bits scaled into `[0, 1)`.
    fn next_double(&mut self) -> f64 {
        let hi = self.next(26) << 27;
        let lo = self.next(27);
        // Both parts are below 2^53, so the sum is exact.
        (hi + lo) as f64 * (1.0 / (1_u64 << 53) as f64)
    }
}

/// `Double.compare`: `-0.0 < 0.0` and every NaN is equal and the greatest.
fn java_compare(a: f64, b: f64) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        let (x, y) = (canonical_bits(a), canonical_bits(b));
        x.cmp(&y)
    }
}

/// `Double.doubleToLongBits`: all NaNs collapse to one pattern; the
/// sign-magnitude order is made monotone for the tie cases `compare` reaches.
fn canonical_bits(v: f64) -> i64 {
    if v.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        v.to_bits() as i64
    }
}

/// One `NumericHistogram`: at most `nbins` `(x, y)` bins, ascending in `x`.
#[derive(Clone, Debug)]
pub(crate) struct NumericHistogram {
    nbins: usize,
    bins: Vec<(f64, f64)>,
    rng: JavaRandom,
}

impl NumericHistogram {
    /// `allocate(nbins)`.
    pub(crate) fn new(nbins: usize) -> Self {
        Self {
            nbins,
            bins: Vec::new(),
            rng: JavaRandom::new(),
        }
    }

    /// The bins in ascending order (`getBin`).
    pub(crate) fn bins(&self) -> &[(f64, f64)] {
        &self.bins
    }

    /// `add(v)`: Algorithm 1 of the paper.
    pub(crate) fn add(&mut self, v: f64) {
        // Binary search for the bin to shift right; `v` greater than every
        // bin gives `len`. Comparisons are the raw `f64` ones, as Java's
        // `>`, `<` and `==` on doubles are (NaN is neither).
        let mut bin = 0;
        let (mut l, mut r) = (0, self.bins.len());
        while l < r {
            bin = (l + r) / 2;
            let x = self.bins[bin].0;
            if x > v {
                r = bin;
            } else if x < v {
                bin += 1;
                l = bin;
            } else {
                break;
            }
        }
        if bin < self.bins.len() && self.bins[bin].0 == v {
            self.bins[bin].1 += 1.0;
        } else {
            self.bins.insert(bin, (v, 1.0));
            if self.bins.len() > self.nbins {
                self.trim();
            }
        }
    }

    /// `merge(other)`: Algorithm 2 of the paper.
    pub(crate) fn merge(&mut self, other: &Self) {
        if self.nbins == 0 || self.bins.is_empty() {
            // Nothing here yet: copy `other`.
            self.nbins = other.nbins;
            self.bins.clone_from(&other.bins);
        } else {
            self.bins.extend_from_slice(&other.bins);
            // A stable sort, as `Collections.sort`.
            self.bins.sort_by(|a, b| java_compare(a.0, b.0));
            self.trim();
        }
    }

    /// `trim()`: merges the closest pair of bins until `nbins` are left.
    /// Equally close pairs are chosen among by the PRNG.
    fn trim(&mut self) {
        while self.bins.len() > self.nbins {
            let mut smallest = self.bins[1].0 - self.bins[0].0;
            let (mut at, mut count) = (0_usize, 1_u32);
            for i in 1..self.bins.len() - 1 {
                let diff = self.bins[i + 1].0 - self.bins[i].0;
                if diff < smallest {
                    smallest = diff;
                    at = i;
                    count = 1;
                } else if diff == smallest {
                    count += 1;
                    if self.rng.next_double() <= 1.0 / f64::from(count) {
                        at = i;
                    }
                }
            }
            let (x0, y0) = self.bins[at];
            let (x1, y1) = self.bins[at + 1];
            let d = y0 + y1;
            let mut x = x0;
            x *= y0 / d;
            x += x1 / d * y1;
            self.bins[at] = (x, d);
            self.bins.remove(at + 1);
        }
    }

    /// Spark's `NumericHistogramSerializer.serialize`.
    fn serialize(&self) -> Result<Vec<u8>> {
        let to_i32 = |n: usize| {
            i32::try_from(n).map_err(|_| internal("a histogram of too many bins".to_owned()))
        };
        let mut out = Vec::with_capacity(8 + 16 * self.bins.len());
        out.extend_from_slice(&to_i32(self.nbins)?.to_be_bytes());
        out.extend_from_slice(&to_i32(self.bins.len())?.to_be_bytes());
        for &(x, y) in &self.bins {
            out.extend_from_slice(&x.to_be_bytes());
            out.extend_from_slice(&y.to_be_bytes());
        }
        Ok(out)
    }

    /// Spark's `NumericHistogramSerializer.deserialize`; the PRNG is fresh.
    fn deserialize(bytes: &[u8]) -> Result<Self> {
        let bad = || internal("a histogram state of a bad layout".to_owned());
        let int = |at: usize| -> Result<i32> {
            let b: [u8; 4] = bytes
                .get(at..at + 4)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(bad)?;
            Ok(i32::from_be_bytes(b))
        };
        let dbl = |at: usize| -> Result<f64> {
            let b: [u8; 8] = bytes
                .get(at..at + 8)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(bad)?;
            Ok(f64::from_be_bytes(b))
        };
        let nbins = usize::try_from(int(0)?).map_err(|_| bad())?;
        let used = usize::try_from(int(4)?).map_err(|_| bad())?;
        if bytes.len() != 8 + 16 * used || used > nbins.saturating_add(1) {
            return Err(bad());
        }
        let mut hist = Self::new(nbins);
        hist.bins.reserve(used);
        for i in 0..used {
            let at = 8 + 16 * i;
            hist.bins.push((dbl(at)?, dbl(at + 8)?));
        }
        Ok(hist)
    }
}

/// The histograms of every group of one aggregate.
#[derive(Debug)]
pub(crate) struct HistAcc {
    nb: usize,
    groups: Vec<NumericHistogram>,
}

impl HistAcc {
    /// A table of histograms of `nb` bins each.
    pub(crate) fn new(nb: i32) -> Self {
        Self {
            nb: usize::try_from(nb).unwrap_or(0),
            groups: Vec::new(),
        }
    }

    pub(crate) fn resize(&mut self, n: usize) {
        if n < self.groups.len() {
            self.groups.truncate(n);
        } else {
            let nb = self.nb;
            self.groups.resize_with(n, || NumericHistogram::new(nb));
        }
    }

    pub(crate) fn reserve(&mut self, new: usize) {
        self.groups.reserve(new);
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        self.groups.capacity() * size_of::<NumericHistogram>()
            + self
                .groups
                .iter()
                .map(|h| h.bins.capacity() * size_of::<(f64, f64)>())
                .sum::<usize>()
    }

    /// The size after `new` more groups and `rows` more values: each value
    /// adds at most one bin.
    pub(crate) fn projected(&self, new: usize, rows: usize) -> usize {
        self.size()
            + new.saturating_mul(size_of::<NumericHistogram>())
            + rows.saturating_mul(size_of::<(f64, f64)>())
    }

    /// Adds `v` to the histogram of group `g`.
    pub(crate) fn add(&mut self, g: usize, v: f64) {
        self.groups[g].add(v);
    }

    /// The partial state of groups `ids`: Spark's serialized histograms (an
    /// empty histogram is NULL, which a merge skips).
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(ids.len());
        for &i in ids {
            let h = &self.groups[i as usize];
            out.push(if h.bins.is_empty() {
                None
            } else {
                Some(h.serialize()?)
            });
        }
        Ok(vec![
            Arc::new(BinaryArray::from_iter(out.iter().map(Option::as_deref))) as ArrayRef,
        ])
    }

    /// Merges serialized histograms into `groups`.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &BinaryArray) -> Result<()> {
        use arrow_array::Array;
        if states.len() != groups.len() {
            return Err(internal(
                "a histogram state of another length than its groups".to_owned(),
            ));
        }
        for (row, &g) in groups.iter().enumerate() {
            if states.is_null(row) {
                continue;
            }
            let other = NumericHistogram::deserialize(states.value(row))?;
            self.groups[g as usize].merge(&other);
        }
        Ok(())
    }

    /// The final values of groups `ids` as `array<struct<x, y>>` of type `t`
    /// (`x` of the input's type); a group with no value is NULL.
    pub(crate) fn evaluate(&self, ids: &[u32], t: &DataType) -> Result<ArrayRef> {
        let bad =
            || internal("a histogram result of a type other than array<struct<x, y>>".to_owned());
        let DataType::List(field) = t else {
            return Err(bad());
        };
        let DataType::Struct(fields) = field.data_type() else {
            return Err(bad());
        };
        if fields.len() != 2 {
            return Err(bad());
        }
        let mut offsets = Vec::with_capacity(ids.len() + 1);
        offsets.push(0_i32);
        let mut xs: Vec<f64> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        let mut present = Vec::with_capacity(ids.len());
        for &i in ids {
            let bins = self.groups[i as usize].bins();
            present.push(!bins.is_empty());
            xs.extend(bins.iter().map(|b| b.0));
            ys.extend(bins.iter().map(|b| b.1));
            offsets.push(
                i32::try_from(xs.len())
                    .map_err(|_| internal("a histogram of too many bins".to_owned()))?,
            );
        }
        let x = x_column(&xs, fields[0].data_type())?;
        let y = Arc::new(Float64Array::from(ys)) as ArrayRef;
        let structs = StructArray::try_new(Fields::clone(fields), vec![x, y], None)
            .map_err(|e| internal(format!("a histogram struct: {e}")))?;
        Ok(finish_lists(
            Arc::new(structs),
            offsets,
            Arc::clone(field),
            Some(&present),
        ))
    }
}

/// The bin centers as a column of the input type `t`, converted as Spark's
/// `HistogramNumeric.eval` does: Java narrowing (`toByte`, `toShort`, `toInt`,
/// `toLong` through saturating `d2i`/`d2l`), `toFloat`, and for decimals
/// `BigDecimal.decimal(x, MathContext(precision))` set to the scale half up.
#[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
fn x_column(xs: &[f64], t: &DataType) -> Result<ArrayRef> {
    Ok(match t {
        DataType::Float64 => Arc::new(Float64Array::from(xs.to_vec())) as ArrayRef,
        DataType::Float32 => Arc::new(Float32Array::from_iter_values(xs.iter().map(|&x| x as f32))),
        // `Double.toByte` is `d2i` then `i2b`: it wraps.
        DataType::Int8 => narrow(xs.iter().map(|&x| Some(i128::from(x as i32 as i8))), t)?,
        DataType::Int16 => narrow(xs.iter().map(|&x| Some(i128::from(x as i32 as i16))), t)?,
        DataType::Int32 | DataType::Date32 | DataType::Interval(IntervalUnit::YearMonth) => {
            narrow(xs.iter().map(|&x| Some(i128::from(x as i32))), t)?
        }
        DataType::Int64 | DataType::Timestamp(..) | DataType::Duration(_) => {
            narrow(xs.iter().map(|&x| Some(i128::from(x as i64))), t)?
        }
        DataType::Decimal64(p, s) | DataType::Decimal128(p, s) => narrow(
            xs.iter()
                .map(|&x| Some(decimal_unscaled_in(x, u32::from(*p), *s))),
            t,
        )?,
        t => return Err(internal(format!("a histogram of {t}"))),
    })
}

#[cfg(test)]
mod tests;
