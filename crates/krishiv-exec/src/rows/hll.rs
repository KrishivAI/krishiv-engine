// SPDX-License-Identifier: Apache-2.0
//! [`Hll`]: the cardinality sketch the row tables presize from.

/// A HyperLogLog sketch (Flajolet et al. 2007) with `2^bits` one-byte
/// registers over the bits of `h` below the top 8 (so the radix bits, equal
/// within a partition, do not bias it), with linear counting for small
/// counts. Seedless and mergeable: the sketches of one partition from every
/// table merge into that partition's estimate. Standard error
/// `1.04 / 2^(bits/2)`: 3.3% for the partition sketches (the aggregate's `PART_HLL_BITS`,
/// enough to presize an index), 1.6% for the adaptation's (`ADAPT_HLL_BITS`,
/// which decides at 95% unique).
#[derive(Clone, Debug)]
pub(crate) struct Hll {
    bits: u32,
    registers: Box<[u8]>,
}

/// Top bits of `h` the sketch skips (radix bits, up to 8).
const HLL_SKIP: u32 = 8;

impl Hll {
    pub(crate) fn new(bits: u32) -> Self {
        Self {
            bits,
            registers: vec![0; 1 << bits].into_boxed_slice(),
        }
    }

    #[inline]
    pub(crate) fn add(&mut self, h: u64) {
        let rest = h << HLL_SKIP;
        let i = usize::try_from(rest >> (64 - self.bits)).unwrap_or(0);
        let w = rest << self.bits;
        let max = 64 - self.bits - HLL_SKIP + 1;
        let rank = u8::try_from((w.leading_zeros() + 1).min(max)).unwrap_or(u8::MAX);
        if let Some(r) = self.registers.get_mut(i) {
            *r = (*r).max(rank);
        }
    }

    /// Merges a sketch with as many registers (others are ignored).
    pub(crate) fn merge(&mut self, o: &Self) {
        if o.bits == self.bits {
            for (a, b) in self.registers.iter_mut().zip(o.registers.iter()) {
                *a = (*a).max(*b);
            }
        }
    }

    /// The distinct count estimate.
    #[expect(
        clippy::cast_precision_loss,
        reason = "an estimate: register counts are small"
    )]
    pub(crate) fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let (mut sum, mut zeros) = (0.0, 0usize);
        for &r in self.registers.iter() {
            sum += (-f64::from(r)).exp2();
            zeros += usize::from(r == 0);
        }
        let e = 0.7213 / (1.0 + 1.079 / m) * m * m / sum;
        if e <= 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln()
        } else {
            e
        }
    }

    /// Registers that saw a hash (tests).
    #[cfg(test)]
    pub(crate) fn used(&self) -> usize {
        self.registers.iter().filter(|&&r| r > 0).count()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.registers.len()
    }
}
