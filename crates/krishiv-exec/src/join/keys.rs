// SPDX-License-Identifier: Apache-2.0
//! Join keys on the shared hash blocks ([`crate::hash`]): Spark-normalized
//! arrow-row key rows, the seeded hash and the spill partitions.
//!
//! [`KeyEncoder`] encodes NULLs as equal, which gives null-safe equality
//! (`<=>`); a row with a NULL in a key compared with `=` can never match, so
//! the join marks it in [`Encoded::valid`] and never probes or stores it.
//!
//! The hash seed is per query: the deterministic-mode seed when set, else
//! random, so an attacker who controls the key values but not the seed
//! cannot aim many keys at one bucket (hash flooding).

use arrow_array::{Array, ArrayRef};
use arrow_buffer::NullBuffer;
use arrow_row::Rows;
use arrow_schema::DataType;

use crate::error::Result;
use crate::hash::{HashSeed, KeyEncoder, MAX_LEVEL, spill_partition};

pub(crate) use crate::hash::PARTITIONS;

/// Repartitioning rounds below level 0 before a partition that still does
/// not fit fails the query (`8^6` ways in all).
pub(crate) const MAX_DEPTH: u32 = if MAX_LEVEL < 6 { MAX_LEVEL } else { 6 };

/// The partition of `hash` at repartitioning level `depth`.
pub(crate) fn partition(hash: u64, depth: u32) -> usize {
    spill_partition(hash, depth)
}

/// The join's hash seed, shared by its build and probe: `seed`
/// (deterministic mode) or random ([`HashSeed::new`]).
pub(crate) fn query_seed(seed: Option<u64>) -> HashSeed {
    HashSeed::new(seed)
}

/// One batch's keys: row-format bytes, whether each row can match, and
/// each row's hash.
#[derive(Debug)]
pub(crate) struct Encoded {
    pub(crate) rows: Rows,
    /// `None`: every row can match.
    pub(crate) valid: Option<NullBuffer>,
    pub(crate) hashes: Vec<u64>,
}

impl Encoded {
    pub(crate) fn is_valid(&self, i: usize) -> bool {
        self.valid.as_ref().is_none_or(|v| v.is_valid(i))
    }
}

/// Encodes the key columns of either side (same types on both).
#[derive(Debug)]
pub(crate) struct KeyCodec {
    encoder: KeyEncoder,
    null_safe: Vec<bool>,
}

impl KeyCodec {
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for key types the encoder does not
    /// normalize (floats inside nested types).
    pub(crate) fn new(types: &[DataType], null_safe: Vec<bool>) -> Result<Self> {
        Ok(Self {
            encoder: KeyEncoder::new(types)?,
            null_safe,
        })
    }

    /// Encodes and hashes `columns` (the key columns, in key order).
    pub(crate) fn encode(&self, columns: &[ArrayRef], seed: &HashSeed) -> Result<Encoded> {
        let rows = self.encoder.encode(columns)?;
        let valid = columns
            .iter()
            .zip(&self.null_safe)
            .filter(|(_, safe)| !**safe)
            .fold(None, |acc: Option<NullBuffer>, (c, _)| {
                NullBuffer::union(acc.as_ref(), c.logical_nulls().as_ref())
            });
        let hashes = rows.iter().map(|r| seed.hash(r.as_ref())).collect();
        Ok(Encoded {
            rows,
            valid,
            hashes,
        })
    }
}

/// Whether some row cannot match.
pub(crate) fn any_invalid(valid: Option<&NullBuffer>) -> bool {
    valid.is_some_and(|v| v.null_count() > 0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, Int32Array};

    use super::*;

    #[test]
    fn floats_are_normalized_and_nulls_marked() {
        let codec = KeyCodec::new(&[DataType::Float64, DataType::Int32], vec![false, true]);
        let codec = codec.unwrap_or_else(|e| panic!("{e}"));
        let f: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(-0.0),
            Some(0.0),
            Some(f64::NAN),
            Some(-f64::NAN),
            None,
        ]));
        let i: ArrayRef = Arc::new(Int32Array::from(vec![
            None,
            None,
            Some(1),
            Some(1),
            Some(2),
        ]));
        let e = codec.encode(&[f, i], &query_seed(Some(7)));
        let e = e.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(e.rows.row(0), e.rows.row(1));
        assert_eq!(e.rows.row(2), e.rows.row(3));
        assert_eq!(e.hashes[0], e.hashes[1]);
        assert!(e.is_valid(0) && !e.is_valid(4));
        assert!(any_invalid(e.valid.as_ref()));
        let f: ArrayRef = Arc::new(Float64Array::from(vec![0.0]));
        let i: ArrayRef = Arc::new(Int32Array::from(vec![None::<i32>]));
        let other = codec.encode(&[f, i], &query_seed(Some(8)));
        let other = other.unwrap_or_else(|e| panic!("{e}"));
        assert_ne!(other.hashes[0], e.hashes[0], "the seed changes the hash");
    }

    #[test]
    fn partitions_and_seeds() {
        const { assert!(MAX_DEPTH <= MAX_LEVEL) };
        assert!(partition(u64::MAX, MAX_DEPTH) < PARTITIONS);
        let rows = |s: &HashSeed| s.hash(b"key");
        assert_eq!(rows(&query_seed(Some(1))), rows(&query_seed(Some(1))));
        assert_ne!(rows(&query_seed(Some(1))), rows(&query_seed(Some(2))));
    }
}
