// SPDX-License-Identifier: Apache-2.0
//! [`SharedBloom`]: one Bloom bit array that every driver of a large build
//! sets bits in (design P2b-R1 §2.2), so no driver collects its keys'
//! hashes. Its size is fixed by the plan before the first key; the array
//! is allocated only once the build has reserved it
//! ([`SharedBloom::allocate`]), and frozen into a [`RuntimeBloom`] when the
//! build finishes.
//!
//! Bits only go up, so the drivers' relaxed `fetch_or`s commute: the frozen
//! bits are the same whatever the insertion order. [`SharedBloom::freeze`]
//! takes the array by value, after the last other holder dropped its `Arc`
//! (that drop orders every holder's stores before the freeze).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use super::bloom::{BLOCK_BYTES, RuntimeBloom, block_of, mask};

type Blocks = Box<[[AtomicU32; 8]]>;

/// A split-block Bloom filter whose bits several threads set (see the
/// module docs).
pub struct SharedBloom {
    blocks: usize,
    /// `None` until allocated; `Some(None)` once abandoned (the build could
    /// not reserve it).
    array: OnceLock<Option<Blocks>>,
}

impl std::fmt::Debug for SharedBloom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedBloom")
            .field("bytes", &self.num_bytes())
            .field("allocated", &self.is_allocated())
            .finish()
    }
}

impl SharedBloom {
    /// A filter of `bytes` bytes (whole blocks, at least one), not yet
    /// allocated.
    #[must_use]
    pub fn with_bytes(bytes: usize) -> Self {
        Self {
            blocks: bytes.div_ceil(BLOCK_BYTES).max(1),
            array: OnceLock::new(),
        }
    }

    /// The filter's size in bytes (what its array takes once allocated).
    #[must_use]
    pub fn num_bytes(&self) -> usize {
        self.blocks * BLOCK_BYTES
    }

    /// Allocates the zeroed array (once the build reserved
    /// [`Self::num_bytes`]); `false` when it was abandoned first.
    pub fn allocate(&self) -> bool {
        self.array
            .get_or_init(|| {
                Some(
                    (0..self.blocks)
                        .map(|_| std::array::from_fn(|_| AtomicU32::new(0)))
                        .collect(),
                )
            })
            .is_some()
    }

    /// Gives the filter up (the build could not reserve it): it is never
    /// allocated, and builders over it publish no Bloom form. `false` when
    /// it was allocated first.
    pub fn abandon(&self) -> bool {
        let _ = self.array.set(None);
        self.array.get().is_some_and(Option::is_none)
    }

    /// Whether the array is allocated.
    #[must_use]
    pub fn is_allocated(&self) -> bool {
        self.array.get().is_some_and(Option::is_some)
    }

    /// Whether the filter was given up ([`Self::abandon`]).
    #[must_use]
    pub fn is_abandoned(&self) -> bool {
        self.array.get().is_some_and(Option::is_none)
    }

    /// Sets the bits of every hash of `hashes`; `false`, and nothing set,
    /// when the array is not allocated.
    pub fn insert_hashes(&self, hashes: &[u64]) -> bool {
        let Some(Some(blocks)) = self.array.get() else {
            return false;
        };
        for &h in hashes {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the SBBF uses the low half"
            )]
            let m = mask(h as u32);
            if let Some(b) = blocks.get(block_of(h, blocks.len())) {
                for (w, bit) in b.iter().zip(m) {
                    // Skip the store when the bit is set already: a set
                    // bit's line can stay shared between cores.
                    if w.load(Ordering::Relaxed) & bit == 0 {
                        w.fetch_or(bit, Ordering::Relaxed);
                    }
                }
            }
        }
        true
    }

    /// The bits as a [`RuntimeBloom`] (the array's memory reused, no
    /// copy); `None` when the array was never allocated.
    #[must_use]
    pub fn freeze(self) -> Option<RuntimeBloom> {
        let blocks = self.array.into_inner().flatten()?;
        // An in-place collect: same size and alignment, one pass.
        let plain: Box<[[u32; 8]]> = blocks
            .into_vec()
            .into_iter()
            .map(|b| b.map(AtomicU32::into_inner))
            .collect();
        Some(RuntimeBloom::from_blocks(plain))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::runtime_filter::bloom::xxh64;

    fn hashes(range: std::ops::Range<u64>) -> Vec<u64> {
        range.map(|i| xxh64(&i.to_le_bytes())).collect()
    }

    #[test]
    fn same_bits_as_a_runtime_bloom_in_any_order() {
        let all = hashes(0..20_000);
        let mut plain = RuntimeBloom::with_bytes(8192);
        for &h in &all {
            plain.insert_hash(h);
        }
        let shared = Arc::new(SharedBloom::with_bytes(8192));
        assert!(!shared.insert_hashes(&all), "not allocated yet");
        assert!(shared.allocate() && shared.is_allocated());
        // Four threads, interleaved chunks, reversed order.
        std::thread::scope(|s| {
            for t in 0..4 {
                let (shared, all) = (Arc::clone(&shared), &all);
                s.spawn(move || {
                    let mine: Vec<u64> = all.iter().rev().skip(t).step_by(4).copied().collect();
                    for chunk in mine.chunks(97) {
                        assert!(shared.insert_hashes(chunk));
                    }
                });
            }
        });
        let frozen = Arc::try_unwrap(shared)
            .unwrap_or_else(|_| panic!("one holder"))
            .freeze()
            .unwrap_or_else(|| panic!("allocated"));
        assert_eq!(frozen, plain);
        assert!(all.iter().all(|&h| frozen.check_hash(h)));
    }

    /// The freeze reuses the array's allocation (std's in-place collect):
    /// a toolchain that copied it would hold the array twice, unreserved.
    #[test]
    fn freeze_copies_nothing() {
        let shared = SharedBloom::with_bytes(1 << 20);
        assert!(shared.allocate());
        let before = match shared.array.get() {
            Some(Some(blocks)) => blocks.as_ptr().addr(),
            _ => panic!("allocated"),
        };
        let frozen = shared.freeze().unwrap_or_else(|| panic!("allocated"));
        assert_eq!(frozen.blocks_addr(), before);
    }

    #[test]
    fn abandoned_or_unallocated_freezes_to_nothing() {
        let a = SharedBloom::with_bytes(0);
        assert_eq!(a.num_bytes(), BLOCK_BYTES);
        assert!(a.freeze().is_none());
        let b = SharedBloom::with_bytes(64);
        assert!(b.abandon() && b.is_abandoned());
        assert!(!b.allocate(), "abandoned first");
        assert!(!b.insert_hashes(&[1, 2, 3]));
        assert!(format!("{b:?}").contains("allocated"));
        assert!(b.freeze().is_none());
        let c = SharedBloom::with_bytes(64);
        assert!(c.allocate());
        assert!(!c.abandon(), "allocated first");
        assert!(c.freeze().is_some_and(|f| f.num_bytes() == 64));
    }
}
