// SPDX-License-Identifier: Apache-2.0
//! Join keys in one of two formats, chosen per join instance ([`Hashing`]):
//!
//! - **arrow-row** (the default): Spark-normalized arrow-row key rows on the
//!   shared hash blocks ([`crate::hash`]), hashed with the seeded hash, which
//!   also picks the spill partitions;
//! - **row layout** (`krishiv.exec.join.rowLayout`, P2-T158): fixed-width
//!   [`RowLayout`] rows with a stored hash `h`, seeded per join instance.
//!   Level-0 partitions are the top bits of `h`; the index and the grace
//!   sub-partitions use `g = fin(h ^ seed)` ([`finalize`]; P2-T157
//!   Appendix C, with its fallback: until P2-T175 fixes the seedless
//!   routing hash, tables keep a seeded hash). Keys without a row encoding (nested,
//!   dictionary) and, until their compare is tuned, string and binary
//!   keys keep the arrow-row format.
//!
//! Both encode NULLs as equal, which gives null-safe equality (`<=>`); a
//! row with a NULL in a key compared with `=` can never match, so the join
//! marks it in [`Encoded::valid`] and never probes or stores it.
//!
//! The seed is per join instance: the deterministic-mode seed when set,
//! else random, so an attacker who controls the key values but not the
//! seed cannot make keys share `h` (hence `g`) and aim them at one bucket,
//! slot run or grace sub-partition (hash flooding). The row layout's
//! column-wise mix is easy to collide when its starting state is known, so
//! `h` is seeded too: it never leaves the join instance (spill files are
//! local to it).

use std::hash::BuildHasher;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeBinaryArray, LargeBinaryArray};
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_row::Rows;
use arrow_schema::DataType;

use crate::error::{Result, internal};
use crate::hash::{HashSeed, KeyEncoder, MAX_LEVEL, PARTITION_BITS, spill_partition};
use crate::rows::{EncodedKeys, RowLayout, finalize};

pub(crate) use crate::hash::PARTITIONS;

/// The top bit (exclusive) of the grace sub-partition field of `g`: level
/// `d` uses bits `48 - 3d .. 51 - 3d`, below the 16-bit salt and above
/// the 26-bit directory index (P2-T157 §4.3).
const GRACE_TOP: u32 = 48;

/// Repartitioning rounds below level 0 before a partition that still does
/// not fit fails the query (`8^6` ways in all).
pub(crate) const MAX_DEPTH: u32 = if MAX_LEVEL < 6 { MAX_LEVEL } else { 6 };

/// How one join instance encodes, hashes and partitions its keys; shared
/// by its build and probe.
#[derive(Debug, Clone)]
pub(crate) struct Hashing {
    /// Seeds the arrow-row hash.
    seed: HashSeed,
    /// Seeds `g` in the row layout.
    raw: u64,
    rows: bool,
}

impl Hashing {
    /// `seed` (deterministic mode) or a random one; the row layout if
    /// `rows`.
    pub(crate) fn new(seed: Option<u64>, rows: bool) -> Self {
        let raw =
            seed.unwrap_or_else(|| std::collections::hash_map::RandomState::new().hash_one(0u8));
        Self {
            seed: HashSeed::new(Some(raw)),
            raw,
            rows,
        }
    }

    /// Whether keys are in the row layout.
    pub(crate) fn rows(&self) -> bool {
        self.rows
    }

    /// The seed of the table-local hash `g` (row layout).
    pub(crate) fn g_seed(&self) -> u64 {
        self.raw
    }

    /// The partition of a row with stored hash `hash` at repartitioning
    /// level `depth` (`depth <= MAX_DEPTH`).
    pub(crate) fn partition(&self, hash: u64, depth: u32) -> usize {
        if !self.rows {
            return spill_partition(hash, depth);
        }
        let bits = if depth == 0 {
            hash >> (64 - PARTITION_BITS)
        } else {
            let shift = GRACE_TOP.saturating_sub(PARTITION_BITS * depth.min(MAX_DEPTH));
            finalize(hash, self.raw) >> shift
        };
        usize::try_from(bits & (PARTITIONS as u64 - 1)).unwrap_or(0)
    }
}

/// One batch's keys in either format.
#[derive(Debug)]
pub(crate) enum KeyRows {
    /// Arrow-row bytes, with the seeded hash of each row.
    Arrow(Rows, Vec<u64>),
    /// Row-layout rows, with their seeded hashes.
    Layout(EncodedKeys),
}

/// One batch's keys, whether each row can match, and each row's hash.
#[derive(Debug)]
pub(crate) struct Encoded {
    pub(crate) rows: KeyRows,
    /// `None`: every row can match.
    pub(crate) valid: Option<NullBuffer>,
}

impl Encoded {
    pub(crate) fn is_valid(&self, i: usize) -> bool {
        self.valid.as_ref().is_none_or(|v| v.is_valid(i))
    }

    /// Each row's stored hash.
    pub(crate) fn hashes(&self) -> &[u64] {
        match &self.rows {
            KeyRows::Arrow(_, h) => h,
            KeyRows::Layout(k) => k.hashes(),
        }
    }

    /// The key columns of a row arena chunk holding rows `idx`: arrow-row
    /// bytes (`LargeBinary`), or row-layout rows (`FixedSizeBinary`) plus,
    /// when the layout has string views, each row's long values
    /// (`LargeBinary`).
    pub(crate) fn arena_columns(
        &self,
        layout: Option<&RowLayout>,
        idx: &[usize],
    ) -> Result<Vec<ArrayRef>> {
        match (&self.rows, layout) {
            (KeyRows::Arrow(rows, _), _) => Ok(vec![Arc::new(LargeBinaryArray::from_iter_values(
                idx.iter().map(|&i| rows.row(i)),
            ))]),
            (KeyRows::Layout(keys), Some(layout)) => {
                let g = keys.gather(layout, idx)?;
                let width = i32::try_from(layout.stride())
                    .map_err(|_| internal("join key row too wide".to_owned()))?;
                let fixed = FixedSizeBinaryArray::try_new(width, Buffer::from_vec(g.fixed), None)
                    .map_err(|e| internal(e.to_string()))?;
                let mut out: Vec<ArrayRef> = vec![Arc::new(fixed)];
                if let Some((offsets, bytes)) = g.heap {
                    let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));
                    let heap = LargeBinaryArray::try_new(offsets, Buffer::from_vec(bytes), None)
                        .map_err(|e| internal(e.to_string()))?;
                    out.push(Arc::new(heap));
                }
                Ok(out)
            }
            (KeyRows::Layout(_), None) => {
                Err(internal("join key rows without a layout".to_owned()))
            }
        }
    }
}

/// Encodes the key columns of either side (same types on both).
#[derive(Debug)]
pub(crate) struct KeyCodec {
    encoder: KeyEncoder,
    /// `None`: a key type has no row encoding; the join keeps arrow rows.
    layout: Option<Arc<RowLayout>>,
    null_safe: Vec<bool>,
}

impl KeyCodec {
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for key types the encoder does not
    /// normalize (floats inside nested types).
    pub(crate) fn new(types: &[DataType], null_safe: Vec<bool>) -> Result<Self> {
        Ok(Self {
            encoder: KeyEncoder::new(types)?,
            // String and binary keys keep arrow rows for now: on the
            // P2-T158 sweep (24-byte strings) the row layout's view compare
            // and per-row heap lookups probed up to 1.75x slower than the
            // arrow-row `KeyTable` below 1M keys and only level with it
            // beyond.
            layout: RowLayout::for_join(types, &null_safe)
                .ok()
                .filter(|l| !l.has_views())
                .map(Arc::new),
            null_safe,
        })
    }

    /// As [`new`](Self::new), but string keys take the row layout too
    /// (their arena and index paths stay tested until they are enabled).
    #[cfg(test)]
    pub(crate) fn with_string_rows(types: &[DataType], null_safe: Vec<bool>) -> Result<Self> {
        Ok(Self {
            encoder: KeyEncoder::new(types)?,
            layout: Some(Arc::new(RowLayout::for_join(types, &null_safe)?)),
            null_safe,
        })
    }

    /// The row layout of these keys, if they have one.
    pub(crate) fn layout(&self) -> Option<&Arc<RowLayout>> {
        self.layout.as_ref()
    }

    /// Encodes and hashes `columns` (the key columns, in key order) in the
    /// format `hashing` chose.
    pub(crate) fn encode(&self, columns: &[ArrayRef], hashing: &Hashing) -> Result<Encoded> {
        let valid = columns
            .iter()
            .zip(&self.null_safe)
            .filter(|(_, safe)| !**safe)
            .fold(None, |acc: Option<NullBuffer>, (c, _)| {
                NullBuffer::union(acc.as_ref(), c.logical_nulls().as_ref())
            });
        let rows = match (&self.layout, hashing.rows) {
            (Some(layout), true) => KeyRows::Layout(layout.encode(columns, hashing.raw)?),
            (None, true) => return Err(internal("join keys have no row layout".to_owned())),
            (_, false) => {
                let rows = self.encoder.encode(columns)?;
                let hashes = rows.iter().map(|r| hashing.seed.hash(r.as_ref())).collect();
                KeyRows::Arrow(rows, hashes)
            }
        };
        Ok(Encoded { rows, valid })
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
        let e = codec.encode(&[f, i], &Hashing::new(Some(7), false));
        let e = e.unwrap_or_else(|e| panic!("{e}"));
        let KeyRows::Arrow(rows, _) = &e.rows else {
            unreachable!("arrow rows")
        };
        assert_eq!(rows.row(0), rows.row(1));
        assert_eq!(rows.row(2), rows.row(3));
        assert_eq!(e.hashes()[0], e.hashes()[1]);
        assert!(e.is_valid(0) && !e.is_valid(4));
        assert!(any_invalid(e.valid.as_ref()));
        let f: ArrayRef = Arc::new(Float64Array::from(vec![0.0]));
        let i: ArrayRef = Arc::new(Int32Array::from(vec![None::<i32>]));
        let other = codec.encode(
            &[Arc::clone(&f), Arc::clone(&i)],
            &Hashing::new(Some(8), false),
        );
        let other = other.unwrap_or_else(|e| panic!("{e}"));
        assert_ne!(
            other.hashes()[0],
            e.hashes()[0],
            "the seed changes the hash"
        );
        // The row layout's stored hash is seeded too: crafted keys cannot
        // share it without the seed.
        let a = codec.encode(
            &[Arc::clone(&f), Arc::clone(&i)],
            &Hashing::new(Some(7), true),
        );
        let b = codec.encode(&[f, i], &Hashing::new(Some(8), true));
        let (a, b) = (
            a.unwrap_or_else(|e| panic!("{e}")),
            b.unwrap_or_else(|e| panic!("{e}")),
        );
        assert_ne!(a.hashes(), b.hashes());
    }

    #[test]
    fn partitions_and_seeds() {
        const { assert!(MAX_DEPTH <= MAX_LEVEL) };
        const { assert!(GRACE_TOP >= PARTITION_BITS * MAX_DEPTH + 26) };
        for rows in [false, true] {
            let h = Hashing::new(Some(1), rows);
            for depth in 0..=MAX_DEPTH + 1 {
                assert!(h.partition(u64::MAX, depth) < PARTITIONS);
            }
        }
        // Level 0 of the row layout is the top bits of the stored hash;
        // deeper levels also depend on the seed through `g`.
        let (a, b) = (Hashing::new(Some(1), true), Hashing::new(Some(2), true));
        assert_eq!(a.partition(5 << 61, 0), 5);
        assert_eq!(b.partition(5 << 61, 0), 5);
        let differs = (0..64u64).any(|x| a.partition(x << 20, 1) != b.partition(x << 20, 1));
        assert!(differs);
        assert_eq!(a.g_seed(), 1);
        assert!(a.rows() && !Hashing::new(None, false).rows());
    }
}
