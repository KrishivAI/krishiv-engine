// SPDX-License-Identifier: Apache-2.0
//! The row hash `h` of one key column: XXH64 with seed 0 of the key's
//! [normalized bytes](super::key) — the Parquet Bloom-filter hash, and the
//! single-column routing hash of design P2-T171 §5.1.
//!
//! It is computed **once per batch, column-wise** ([`row_hashes`]) and
//! every form that needs a hash (the exact hash set and the Bloom filter)
//! reads that one vector: a probe row is never hashed twice. Fixed-width
//! keys take straight-line specializations of XXH64 for their width
//! ([`xxh64_fixed`]), which agree bit for bit with the general function
//! (tested against `twox-hash`).
//!
//! The function is named and versioned ([`RowHash`]); a serialized filter
//! carries the version, and a filter built under another version is never
//! applied. It is the one-column case of the routing hash `H_kg`
//! (`krishiv_kernels::hash::routing`, P2-T175), whose XXH64 this module
//! uses. Only hashes produced by this routing-hash implementation, such as
//! `row_hashes` or [`xxh64_fixed`], are valid for
//! [`KeyFilterBuilder::add_hashed`](super::KeyFilterBuilder::add_hashed). A
//! row-layout table hash is not valid input.

pub use krishiv_kernels::hash::routing::xxh64_fixed;

use super::key::{Norm, NormalizedKeys, for_each_bytes};

/// The named, versioned row hash a filter's hashed forms are built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RowHash {
    /// XXH64, seed 0, of the normalized key bytes (version 1).
    Xxh64Seed0,
}

impl RowHash {
    /// The hash runtime filters use today.
    pub const CURRENT: Self = Self::Xxh64Seed0;

    /// The stable id written to a serialized filter.
    #[must_use]
    pub fn id(self) -> u8 {
        match self {
            Self::Xxh64Seed0 => 1,
        }
    }

    /// The hash with stable id `id`.
    #[must_use]
    pub fn from_id(id: u8) -> Option<Self> {
        (id == 1).then_some(Self::Xxh64Seed0)
    }
}

/// The hash of an integer key stored in `width` bytes (the key type's).
#[inline]
#[must_use]
pub(crate) fn int_hash(v: i64, width: usize) -> u64 {
    #[expect(clippy::cast_sign_loss, reason = "the two's-complement bytes")]
    let bits = u128::from(v as u64);
    match width {
        1 => xxh64_fixed::<1>(bits & 0xFF),
        2 => xxh64_fixed::<2>(bits & 0xFFFF),
        4 => xxh64_fixed::<4>(bits & 0xFFFF_FFFF),
        _ => xxh64_fixed::<8>(bits),
    }
}

/// The hash of a decimal key (16 little-endian bytes).
#[inline]
#[must_use]
pub(crate) fn decimal_hash(v: i128) -> u64 {
    #[expect(clippy::cast_sign_loss, reason = "the two's-complement bytes")]
    xxh64_fixed::<16>(v as u128)
}

/// The hash of a normalized float key (four bytes for a `float` key).
#[inline]
#[must_use]
pub(crate) fn float_hash(v: f64, single: bool) -> u64 {
    if single {
        #[expect(clippy::cast_possible_truncation, reason = "a widened float32")]
        let f = v as f32;
        xxh64_fixed::<4>(u128::from(f.to_bits()))
    } else {
        xxh64_fixed::<8>(u128::from(v.to_bits()))
    }
}

/// Writes `h` of the rows `rows` of `keys` (every row when `None`) to
/// `out`, in that order (`out` is cleared first). NULL rows hash their
/// placeholder value; callers skip them.
pub(crate) fn row_hashes(keys: &NormalizedKeys, rows: Option<&[u32]>, out: &mut Vec<u64>) {
    out.clear();
    macro_rules! hash_all {
        ($values:expr, $f:expr) => {{
            let values = $values;
            let f = $f;
            match rows {
                None => out.extend(values.iter().map(|&v| f(v))),
                Some(r) => out.extend(
                    r.iter()
                        .map(|&i| values.get(i as usize).map_or(0, |&v| f(v))),
                ),
            }
        }};
    }
    match &keys.norm {
        Norm::Int { values, width } => match width {
            1 => hash_all!(values, |v| int_hash(v, 1)),
            2 => hash_all!(values, |v| int_hash(v, 2)),
            4 => hash_all!(values, |v| int_hash(v, 4)),
            _ => hash_all!(values, |v| int_hash(v, 8)),
        },
        Norm::Decimal(values) => hash_all!(values, decimal_hash),
        Norm::Float { values, single } => {
            let single = *single;
            hash_all!(values, |v| float_hash(v, single));
        }
        Norm::Bytes(a) => {
            let mut all = vec![0u64; keys.len()];
            for_each_bytes(a.as_ref(), |i, b| {
                if let Some(slot) = all.get_mut(i) {
                    *slot = super::bloom::xxh64(b);
                }
            });
            match rows {
                None => *out = all,
                Some(r) => out.extend(r.iter().map(|&i| all.get(i as usize).copied().unwrap_or(0))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::runtime_filter::bloom::xxh64;

    proptest! {
        /// The unrolled widths (the routing hash's) agree with an
        /// independent XXH64 (`twox-hash`) bit for bit.
        #[test]
        fn fixed_widths_match_xxh64(v in any::<u128>()) {
            let b = v.to_le_bytes();
            prop_assert_eq!(xxh64_fixed::<1>(v & 0xFF), xxh64(&b[..1]));
            prop_assert_eq!(xxh64_fixed::<2>(v & 0xFFFF), xxh64(&b[..2]));
            prop_assert_eq!(xxh64_fixed::<4>(v & 0xFFFF_FFFF), xxh64(&b[..4]));
            prop_assert_eq!(xxh64_fixed::<8>(v & u128::from(u64::MAX)), xxh64(&b[..8]));
            prop_assert_eq!(xxh64_fixed::<16>(v), xxh64(&b));
        }
    }

    #[test]
    fn ids_round_trip() {
        assert_eq!(
            RowHash::from_id(RowHash::CURRENT.id()),
            Some(RowHash::CURRENT)
        );
        assert_eq!(RowHash::from_id(0), None);
    }
}
