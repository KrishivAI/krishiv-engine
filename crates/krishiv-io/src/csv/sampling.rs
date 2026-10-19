// SPDX-License-Identifier: Apache-2.0
//! Spark's row sampling (`Dataset.sample(false, ratio, 1)`), reproduced so
//! that `samplingRatio` picks the same lines: per partition, a
//! `BernoulliCellSampler` over an `XORShiftRandom` seeded with
//! `seed + partitionIndex`, one draw per line.

/// `scala.util.hashing.MurmurHash3.bytesHash`.
fn bytes_hash(data: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mix_last = |h: u32, k: u32| h ^ k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
    let mut h = seed;
    let (words, tail) = data.as_chunks::<4>();
    for c in words {
        let k = u32::from_le_bytes(*c);
        h = mix_last(h, k)
            .rotate_left(13)
            .wrapping_mul(5)
            .wrapping_add(0xe654_6b64);
    }
    if !tail.is_empty() {
        let mut k = 0_u32;
        for (i, b) in tail.iter().enumerate() {
            k ^= u32::from(*b) << (8 * i);
        }
        h = mix_last(h, k);
    }
    h ^= u32::try_from(data.len()).unwrap_or(u32::MAX);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// Spark's `XORShiftRandom`.
#[derive(Debug, Clone)]
pub(crate) struct XorShift {
    seed: u64,
}

impl XorShift {
    /// The generator Spark seeds with `init`.
    pub(crate) fn new(init: i64) -> Self {
        let bytes = init.to_be_bytes();
        let low = bytes_hash(&bytes, 0x3c07_4a61);
        let high = bytes_hash(&bytes, low);
        Self {
            seed: (u64::from(high) << 32) | u64::from(low),
        }
    }

    fn next(&mut self, bits: u32) -> u64 {
        let mut s = self.seed ^ (self.seed << 21);
        s ^= s >> 35;
        s ^= s << 4;
        self.seed = s;
        s & ((1_u64 << bits) - 1)
    }

    /// `java.util.Random.nextDouble`.
    pub(crate) fn next_double(&mut self) -> f64 {
        let hi = self.next(26) << 27;
        let lo = self.next(27);
        (hi + lo) as f64 * (1.0 / (1_u64 << 53) as f64)
    }
}

/// `BernoulliCellSampler(0, ratio)` of one partition.
#[derive(Debug, Clone)]
pub(crate) struct Sampler {
    rng: XorShift,
    ratio: f64,
}

impl Sampler {
    /// The sampler of partition `index` (seed 1, as `CSVUtils.sample`).
    pub(crate) fn new(ratio: f64, index: usize) -> Self {
        Self {
            rng: XorShift::new(1_i64.saturating_add(i64::try_from(index).unwrap_or(i64::MAX))),
            ratio,
        }
    }

    /// Whether the next line is sampled (draws once).
    pub(crate) fn take(&mut self) -> bool {
        let x = self.rng.next_double();
        x >= 0.0 && x < self.ratio
    }
}
