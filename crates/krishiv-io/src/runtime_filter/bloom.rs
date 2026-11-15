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

/// A filter up to this size probes one hash at a time: it stays in the
/// caches, where a prefetch only adds instructions (study 18: filters
/// under 30 KB probe in 2-5 ns a row).
pub(crate) const PREFETCH_ABOVE_BYTES: usize = 256 * 1024;

/// How many hashes ahead [`RuntimeBloom::check_batch`] prefetches a
/// block: enough loads in flight to cover a memory miss at a few
/// nanoseconds a probe (the prototype's 32, study 18 §7.1).
pub(crate) const PREFETCH_DISTANCE: usize = 32;

/// XXH64 with seed 0, Parquet's Bloom-filter hash.
#[must_use]
#[inline]
pub fn xxh64(bytes: &[u8]) -> u64 {
    twox_hash::XxHash64::oneshot(0, bytes)
}

/// One 256-bit block, aligned to its size: a block never straddles two
/// cache lines, so a probe (and its prefetch) touches one line, not two.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(32))]
struct Block([u32; 8]);

impl std::ops::Deref for Block {
    type Target = [u32; 8];
    fn deref(&self) -> &[u32; 8] {
        &self.0
    }
}

impl std::ops::DerefMut for Block {
    fn deref_mut(&mut self) -> &mut [u32; 8] {
        &mut self.0
    }
}

/// A split-block Bloom filter over 64-bit hashes (see the module docs).
#[derive(Clone, PartialEq, Eq)]
pub struct RuntimeBloom {
    blocks: Box<[Block]>,
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
            blocks: vec![Block([0u32; 8]); blocks].into_boxed_slice(),
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

    /// Adds every hash `other` holds: a bitwise OR. `false`, and nothing
    /// changed, when the two differ in size (their hashes land in different
    /// blocks).
    pub fn union_with(&mut self, other: &Self) -> bool {
        if self.blocks.len() != other.blocks.len() {
            return false;
        }
        for (a, b) in self.blocks.iter_mut().zip(&other.blocks) {
            for (w, x) in a.iter_mut().zip(b.iter()) {
                *w |= x;
            }
        }
        true
    }

    /// The share of random hashes that pass, measured from the bits set:
    /// the mean over blocks of the product of each word's fill. Unlike
    /// [`Self::expected_fpp`] it needs no count of distinct hashes, so it
    /// also holds for a union of filters whose keys overlap.
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        let block = |b: &Block| {
            b.iter()
                .map(|w| f64::from(w.count_ones()) / 32.0)
                .product::<f64>()
        };
        #[expect(clippy::cast_precision_loss, reason = "a rate")]
        let n = self.blocks.len() as f64;
        self.blocks.iter().map(block).sum::<f64>() / n
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
        // Every tested bit set: OR the missing bits of the eight words
        // (no early exit, so the eight lanes vectorize).
        self.blocks
            .get(i)
            .is_some_and(|b| b.iter().zip(m).fold(0, |miss, (w, bit)| miss | (bit & !w)) == 0)
    }

    /// Checks every hash of `hashes`, packed 64 answers a word: answer
    /// `i` is whether `hashes[i]` may have been added. The eight words of a
    /// block are tested together (a SIMD compare where the target has one),
    /// as the Parquet and Impala SBBF probes do.
    ///
    /// A filter larger than the caches makes each probe a memory miss,
    /// one block load waiting on the next (study 18 §R4: 16-22 ns a row
    /// on 1-4 MB filters). Past `PREFETCH_ABOVE_BYTES` the block index of
    /// every hash is computed once, and the block `PREFETCH_DISTANCE`
    /// hashes ahead is prefetched while the current one is tested, so many
    /// loads are in flight at once (Impala's and Kudu's batched SBBF
    /// `Find`). The answers are the same either way.
    #[must_use]
    pub fn check_batch(&self, hashes: &[u64]) -> arrow_buffer::BooleanBuffer {
        // Block indices are kept as `u32`: a filter of 2^32 blocks or more
        // (128 GiB, never planned) keeps the one-at-a-time loop.
        if self.num_bytes() <= PREFETCH_ABOVE_BYTES || u32::try_from(self.blocks.len()).is_err() {
            return super::member::pack(hashes, |h| self.check_hash(h));
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "below the block count, which fits in u32 (checked above)"
        )]
        let blocks: Vec<u32> = hashes.iter().map(|&h| self.block(h) as u32).collect();
        let mut words = Vec::with_capacity(hashes.len().div_ceil(64));
        for (c, chunk) in hashes.chunks(64).enumerate() {
            let base = c * 64;
            let mut w = 0u64;
            for (j, &h) in chunk.iter().enumerate() {
                let i = base + j;
                if let Some(&ahead) = blocks.get(i + PREFETCH_DISTANCE) {
                    krishiv_kernels::prefetch_read(&self.blocks, ahead as usize);
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the SBBF uses the low half"
                )]
                let m = mask(h as u32);
                let hit = blocks
                    .get(i)
                    .and_then(|&b| self.blocks.get(b as usize))
                    .is_some_and(|b| {
                        b.iter().zip(m).fold(0, |miss, (w, bit)| miss | (bit & !w)) == 0
                    });
                w |= u64::from(hit) << j;
            }
            words.push(w);
        }
        arrow_buffer::BooleanBuffer::new(arrow_buffer::Buffer::from_vec(words), 0, hashes.len())
    }

    /// A filter over the Parquet-layout bitset `bytes` (see
    /// [`Self::to_bytes`]); `None` unless it is a non-empty whole number of
    /// blocks.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let (chunks, rest) = bytes.as_chunks::<BLOCK_BYTES>();
        if chunks.is_empty() || !rest.is_empty() {
            return None;
        }
        let blocks = chunks
            .iter()
            .map(|b| {
                let mut block = [0u32; 8];
                for (w, c) in block.iter_mut().zip(b.as_chunks::<4>().0) {
                    *w = u32::from_le_bytes(*c);
                }
                Block(block)
            })
            .collect();
        Some(Self { blocks })
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
            for w in b.iter() {
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

    /// A block is one aligned 32-byte unit, so it never spans two lines.
    #[test]
    fn blocks_never_straddle_a_cache_line() {
        assert_eq!(std::mem::size_of::<Block>(), BLOCK_BYTES);
        assert_eq!(std::mem::align_of::<Block>(), BLOCK_BYTES);
        let f = RuntimeBloom::with_bytes(1 << 20);
        assert_eq!(f.blocks.as_ptr().addr() % BLOCK_BYTES, 0);
    }

    /// The batched probe answers as the one-hash probe does, on both sides
    /// of the prefetch threshold and for batch lengths around the 64-answer
    /// words and the prefetch distance (study 18 §R4).
    #[test]
    fn batched_probes_answer_as_single_probes() {
        for bytes in [
            4 * 1024,
            PREFETCH_ABOVE_BYTES,
            PREFETCH_ABOVE_BYTES + BLOCK_BYTES,
            4 << 20,
        ] {
            let mut f = RuntimeBloom::with_bytes(bytes);
            let keys = u64::try_from(bytes / 4).unwrap_or(u64::MAX);
            for i in 0..keys {
                f.insert_hash(xxh64(&(i * 3).to_le_bytes()));
            }
            let probes: Vec<u64> = (0..5_000u64).map(|i| xxh64(&i.to_le_bytes())).collect();
            for n in [
                0,
                1,
                PREFETCH_DISTANCE - 1,
                PREFETCH_DISTANCE,
                63,
                64,
                65,
                129,
                5_000,
            ] {
                let hashes = &probes[..n];
                let got = f.check_batch(hashes);
                assert_eq!(got.len(), n);
                for (i, &h) in hashes.iter().enumerate() {
                    assert_eq!(
                        got.value(i),
                        f.check_hash(h),
                        "{bytes} bytes, {n} hashes, at {i}"
                    );
                }
            }
            // Every inserted key passes (no false negative on the batched path).
            let added: Vec<u64> = (0..keys.min(3_000))
                .map(|i| xxh64(&(i * 3).to_le_bytes()))
                .collect();
            assert_eq!(
                f.check_batch(&added).count_set_bits(),
                added.len(),
                "{bytes} bytes"
            );
        }
    }
}
