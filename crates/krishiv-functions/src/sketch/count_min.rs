// SPDX-License-Identifier: Apache-2.0
//! Spark's count-min sketch (`org.apache.spark.util.sketch.CountMinSketchImpl`), byte for byte: the sizing from
//! `eps` and `confidence`, the row hashes, and the version 1 serialization `count_min_sketch` returns.
//!
//! A sketch is `depth` rows of `width` counters. The counters and their total live with the caller (an aggregate
//! keeps one flat buffer for all its groups); [`Shape`] holds what every sketch of one aggregate shares.
//! Integers are placed by a universal hash with per-row coefficients drawn from `java.util.Random(seed)`,
//! strings and binaries by two Murmur3 hashes. All counter arithmetic wraps, as Java's `long` does.
//!
//! The serialized form, big-endian: version `1` (`int`), total count (`long`), depth (`int`), width (`int`),
//! the `depth` coefficients (`long`s), then the counters row by row (`long`s).

use krishiv_kernels::hash::murmur3_bytes;

/// The version number a serialized sketch starts with (`CountMinSketch.Version.V1`).
const VERSION: i32 = 1;

/// The bytes before the coefficients: version, total count, depth and width.
const HEADER: usize = 4 + 8 + 4 + 4;

/// The number of counter rows for a `confidence` in `(0, 1)`: `ceil(-log1p(-confidence) / ln 2)`, with Java's
/// cast (a NaN gives 0).
#[must_use]
pub fn depth(confidence: f64) -> u32 {
    java_int((-(-confidence).ln_1p() / std::f64::consts::LN_2).ceil())
}

/// The number of counters a row for a positive `eps`: `ceil(2 / eps)`, with Java's cast (an infinite or NaN
/// `eps` gives 0, a tiny one saturates).
#[must_use]
pub fn width(eps: f64) -> u32 {
    java_int((2.0 / eps).ceil())
}

/// Java's `(int)` of a non-negative `double`: NaN is 0 and large values saturate at `Integer.MAX_VALUE`.
fn java_int(v: f64) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a saturating float cast is Java's `(int)`"
    )]
    let i = v as i32;
    u32::try_from(i).unwrap_or(0)
}

/// The serialized length of a sketch of this size, `None` when it does not fit the 2 GiB a `binary` value holds.
#[must_use]
pub fn serialized_len(depth: u32, width: u32) -> Option<usize> {
    let depth = u64::from(depth);
    let cells = depth.checked_mul(u64::from(width))?;
    let bytes = cells
        .checked_add(depth)?
        .checked_mul(8)?
        .checked_add(HEADER as u64)?;
    i32::try_from(bytes).ok()?;
    usize::try_from(bytes).ok()
}

/// `java.util.Random`, as far as `nextInt(Integer.MAX_VALUE)`.
struct JavaRandom(u64);

impl JavaRandom {
    const MULTIPLIER: u64 = 0x5_DEEC_E66D;
    const MASK: u64 = (1 << 48) - 1;

    fn new(seed: i32) -> Self {
        Self((i64::from(seed).cast_unsigned() ^ Self::MULTIPLIER) & Self::MASK)
    }

    /// `nextInt(Integer.MAX_VALUE)`: 31 random bits, drawn again in the one case (`2^31 - 1`) Java rejects.
    fn next_below_int_max(&mut self) -> i64 {
        loop {
            self.0 = self.0.wrapping_mul(Self::MULTIPLIER).wrapping_add(0xB) & Self::MASK;
            let bits = (self.0 >> 17).cast_signed();
            if bits < i64::from(i32::MAX) {
                return bits;
            }
        }
    }
}

/// What the sketches of one aggregate share: their size and the per-row hash coefficients.
#[derive(Debug, Clone)]
pub struct Shape {
    width: u32,
    coefficients: Vec<i64>,
}

impl Shape {
    /// The shape of `depth` rows of `width` counters, hashed with `seed`.
    #[must_use]
    pub fn new(depth: u32, width: u32, seed: i32) -> Self {
        let mut random = JavaRandom::new(seed);
        Self {
            width,
            coefficients: (0..depth).map(|_| random.next_below_int_max()).collect(),
        }
    }

    /// The number of rows (hash coefficients) of the sketch.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.coefficients.len()
    }

    /// The number of counters of one sketch.
    #[must_use]
    pub fn cells(&self) -> usize {
        self.coefficients.len().saturating_mul(self.width as usize)
    }

    /// Whether the rows have no counter to place a value in (an infinite or NaN `eps`): Spark fails such a
    /// sketch with a division by zero at the first value.
    #[must_use]
    pub fn has_empty_rows(&self) -> bool {
        self.width == 0 && !self.coefficients.is_empty()
    }

    /// Counts one integer into `counters` (a sketch of this shape): `addLong`.
    pub fn add_long(&self, counters: &mut [i64], item: i64) {
        let width = i64::from(self.width);
        for (row, &a) in counters
            .chunks_exact_mut((self.width as usize).max(1))
            .zip(&self.coefficients)
        {
            let mut hash = a.wrapping_mul(item);
            hash = hash.wrapping_add(hash >> 32);
            hash &= i64::from(i32::MAX);
            // `hash` is non-negative and a chunk exists only when `width > 0`.
            if let Some(c) = usize::try_from(hash % width.max(1))
                .ok()
                .and_then(|at| row.get_mut(at))
            {
                *c = c.wrapping_add(1);
            }
        }
    }

    /// Counts one string or binary into `counters`: `addBinary`.
    pub fn add_bytes(&self, counters: &mut [i64], item: &[u8]) {
        let first = murmur3_bytes(item, 0);
        let second = murmur3_bytes(item, first);
        let mut combined = first;
        for row in counters
            .chunks_exact_mut((self.width as usize).max(1))
            .take(self.coefficients.len())
        {
            // `h1 + i * h2` in Java's `int`, negated when negative. (Java leaves `Integer.MIN_VALUE` negative
            // and fails on it; its magnitude is counted here.)
            let at = combined.unsigned_abs() % self.width.max(1);
            if let Some(c) = row.get_mut(at as usize) {
                *c = c.wrapping_add(1);
            }
            combined = combined.wrapping_add(second);
        }
    }

    /// The serialized sketch of `total` values counted in `counters`.
    #[must_use]
    pub fn serialize(&self, total: i64, counters: &[i64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            HEADER + 8 * (self.coefficients.len().saturating_add(counters.len())),
        );
        out.extend_from_slice(&VERSION.to_be_bytes());
        out.extend_from_slice(&total.to_be_bytes());
        // A depth is at most `i32::MAX` (`java_int`).
        let depth = u32::try_from(self.coefficients.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&depth.to_be_bytes());
        out.extend_from_slice(&self.width.to_be_bytes());
        for v in self.coefficients.iter().chain(counters) {
            out.extend_from_slice(&v.to_be_bytes());
        }
        out
    }

    /// Adds the serialized sketch `bytes` into `total` and `counters` (`mergeInPlace`). `false`, with nothing
    /// changed, when `bytes` is not a sketch of this shape.
    #[must_use]
    pub fn merge(&self, total: &mut i64, counters: &mut [i64], bytes: &[u8]) -> bool {
        let expected = self.serialize(0, &[]);
        let prefix = expected.len();
        let fits = prefix
            .checked_add(counters.len().saturating_mul(8))
            .is_some_and(|n| n == bytes.len());
        // Everything but the total count (bytes 4 to 12) must match.
        if !fits
            || bytes.get(..4) != expected.get(..4)
            || bytes.get(12..prefix) != expected.get(12..)
        {
            return false;
        }
        let Some(other) = bytes.get(4..12).and_then(|b| b.try_into().ok()) else {
            return false;
        };
        *total = total.wrapping_add(i64::from_be_bytes(other));
        let (cells, _) = bytes[prefix..].as_chunks::<8>();
        for (c, b) in counters.iter_mut().zip(cells) {
            *c = c.wrapping_add(i64::from_be_bytes(*b));
        }
        true
    }
}

#[cfg(test)]
mod tests;
