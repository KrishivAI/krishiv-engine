// SPDX-License-Identifier: Apache-2.0
//! [`RuntimeBloom`]: the Parquet split-block Bloom filter (SBBF) algorithm
//! over hashes: 256-bit blocks of eight 32-bit words, one bit set per word,
//! the block chosen from the hash's upper half, the bits from its lower
//! half and the specification's eight salts.
//!
//! Parquet hashes values with XXH64 (seed 0) of their plain encoding; the
//! runtime filter hashes [normalized key bytes](super::key) the same way, so
//! a runtime Bloom and a file's stored Bloom of the same type agree hash by
//! hash. arrow-rs's `Sbbf` keeps its hash-level insert and check private, so
//! the producer could not insert hashes it collected; this type can, and is
//! sized exactly once the number of hashes is known.

/// The specification's salts.
const SALT: [u32; 8] = [
    0x47b6_137b,
    0x4497_4d91,
    0x8824_ad5b,
    0xa2b7_289d,
    0x7054_95c7,
    0x2df1_424b,
    0x9efc_4947,
    0x5c6b_fb31,
];

/// Bytes of one block.
pub const BLOCK_BYTES: usize = 32;

/// XXH64 with seed 0, Parquet's Bloom-filter hash.
#[must_use]
#[inline]
pub fn xxh64(bytes: &[u8]) -> u64 {
    twox_hash::XxHash64::oneshot(0, bytes)
}

/// A split-block Bloom filter over 64-bit hashes (see the module docs).
#[derive(Clone, PartialEq, Eq)]
pub struct RuntimeBloom {
    blocks: Box<[[u32; 8]]>,
}

impl std::fmt::Debug for RuntimeBloom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeBloom")
            .field("bytes", &self.num_bytes())
            .finish()
    }
}

#[inline]
fn mask(key: u32) -> [u32; 8] {
    let mut m = [0u32; 8];
    for (w, salt) in m.iter_mut().zip(SALT) {
        *w = 1 << (key.wrapping_mul(salt) >> 27);
    }
    m
}

impl RuntimeBloom {
    /// Bytes that hold `items` distinct hashes at false-positive rate `fpp`
    /// (Parquet's sizing: `-8 n / ln(1 - fpp^(1/8))` bits), in whole blocks,
    /// at least one.
    #[must_use]
    pub fn bytes_for(items: u64, fpp: f64) -> usize {
        let fpp = fpp.clamp(1e-9, 0.5);
        #[expect(clippy::cast_precision_loss, reason = "a size estimate")]
        let bits = -8.0 * items as f64 / (1.0 - fpp.powf(1.0 / 8.0)).ln();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "non-negative and far below usize::MAX for any accepted item count"
        )]
        let bytes = (bits / 8.0).ceil() as usize;
        bytes
            .div_ceil(BLOCK_BYTES)
            .max(1)
            .saturating_mul(BLOCK_BYTES)
    }

    /// An empty filter of `bytes` bytes, rounded up to whole blocks (at
    /// least one).
    #[must_use]
    pub fn with_bytes(bytes: usize) -> Self {
        let blocks = bytes.div_ceil(BLOCK_BYTES).max(1);
        Self {
            blocks: vec![[0u32; 8]; blocks].into_boxed_slice(),
        }
    }

    /// The filter's size in bytes.
    #[must_use]
    pub fn num_bytes(&self) -> usize {
        self.blocks.len() * BLOCK_BYTES
    }

    #[inline]
    fn block(&self, hash: u64) -> usize {
        // `(upper 32 bits * blocks) >> 32` is below `blocks`.
        usize::try_from(((hash >> 32) * self.blocks.len() as u64) >> 32).unwrap_or(usize::MAX)
    }

    /// Adds `hash`.
    #[inline]
    pub fn insert_hash(&mut self, hash: u64) {
        let i = self.block(hash);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the SBBF uses the low half"
        )]
        let m = mask(hash as u32);
        if let Some(b) = self.blocks.get_mut(i) {
            for (w, bit) in b.iter_mut().zip(m) {
                *w |= bit;
            }
        }
    }

    /// Whether `hash` may have been added (never `false` for one that was).
    #[inline]
    #[must_use]
    pub fn check_hash(&self, hash: u64) -> bool {
        let i = self.block(hash);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the SBBF uses the low half"
        )]
        let m = mask(hash as u32);
        self.blocks
            .get(i)
            .is_some_and(|b| b.iter().zip(m).all(|(w, bit)| w & bit == bit))
    }

    /// The expected false-positive rate after `items` distinct hashes:
    /// each block receives a Poisson number of them, and a probe passes
    /// when the eight bits it tests are set in its block.
    #[must_use]
    pub fn expected_fpp(&self, items: u64) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "an estimate")]
        let lambda = items as f64 / self.blocks.len() as f64;
        if lambda == 0.0 {
            return 0.0;
        }
        if lambda > 1024.0 {
            // Every word of every block is all but full.
            return (1.0 - (31.0f64 / 32.0).powf(lambda)).powi(8);
        }
        // Sum the Poisson(lambda) weights of block loads (in log space: the
        // weights underflow for large loads); beyond a few standard
        // deviations the tails are negligible.
        let spread = 10.0 * lambda.sqrt() + 20.0;
        let (lo, hi) = ((lambda - spread).max(0.0).floor(), lambda + spread);
        let mut ln_p = -lambda;
        let mut k = 1.0;
        while k <= lo {
            ln_p += lambda.ln() - f64::ln(k);
            k += 1.0;
        }
        let mut total = 0.0;
        let mut k = lo;
        while k <= hi {
            // A word holds a given bit unless all k inserts missed it.
            let word = 1.0 - (31.0f64 / 32.0).powf(k);
            total += ln_p.exp() * word.powi(8);
            k += 1.0;
            ln_p += lambda.ln() - f64::ln(k);
        }
        total.min(1.0)
    }

    /// The bitset in the Parquet layout (little-endian words, block by
    /// block): what a file's Bloom filter stores after its header.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.num_bytes());
        for b in self.blocks.iter() {
            for w in b {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use parquet::bloom_filter::Sbbf;

    use super::*;

    #[test]
    fn agrees_bit_for_bit_with_parquet() {
        // Same size, same values: the same bitset as arrow-rs's SBBF.
        let mut ours = RuntimeBloom::with_bytes(1024);
        let mut theirs = Sbbf::new_with_num_of_bytes(1024);
        for v in (0..500i64).map(|i| i * 7919 - 3) {
            ours.insert_hash(xxh64(&v.to_le_bytes()));
            theirs.insert(&v);
        }
        let mut bits = Vec::new();
        theirs
            .write_bitset(&mut bits)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(ours.to_bytes(), bits);
        // And a stored filter checks our hashes.
        let stored = Sbbf::new(&ours.to_bytes());
        assert!(stored.check(&(7919i64 * 3 - 3)));
        assert!(ours.check_hash(xxh64(&(7919i64 * 3 - 3).to_le_bytes())));
    }

    #[test]
    fn sizing_and_false_positives() {
        let n = 100_000u64;
        let bytes = RuntimeBloom::bytes_for(n, 0.01);
        // About 10.5 bits per value at 1%.
        assert!((120_000..=140_000).contains(&bytes), "{bytes}");
        let mut f = RuntimeBloom::with_bytes(bytes);
        for i in 0..n {
            f.insert_hash(xxh64(&i.to_le_bytes()));
        }
        for i in 0..n {
            assert!(f.check_hash(xxh64(&i.to_le_bytes())), "no false negative");
        }
        let fp = (n..n + 100_000)
            .filter(|i| f.check_hash(xxh64(&i.to_le_bytes())))
            .count();
        assert!(fp < 1_600, "observed {fp} false positives in 100000");
        let est = f.expected_fpp(n);
        assert!((0.005..0.016).contains(&est), "{est}");
        assert!(
            f.expected_fpp(n * 20) > 0.05,
            "an overfull filter is detected"
        );
        assert!(f.expected_fpp(0).abs() < f64::EPSILON);
        assert!(
            RuntimeBloom::with_bytes(32).expected_fpp(5000) > 0.99,
            "saturated"
        );
        assert_eq!(RuntimeBloom::with_bytes(0).num_bytes(), BLOCK_BYTES);
        assert!(format!("{f:?}").contains("bytes"));
    }
}
