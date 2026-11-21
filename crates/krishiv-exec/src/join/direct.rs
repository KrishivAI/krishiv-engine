// SPDX-License-Identifier: Apache-2.0
//! Direct-index probe of a tiny build (perf lever J1, TPC-DS q62/q99).
//!
//! A build of at most [`MAX_ROWS`] rows whose one key is an integer column
//! with values in a span of at most [`SPAN_PER_ROW`] slots a row (and at
//! most [`MAX_SLOTS`]) is probed through an array indexed by `key - min`:
//! each slot holds the partition and the newest build row of that key, the
//! row [`JoinTable::find_heads`] would return. A probe row then costs one
//! subtraction and one load: no key encoding, no hash, no index walk. Older
//! rows with the same key stay chained in their table
//! ([`JoinTable::next_of`]), so duplicates, marks and the emission of build
//! rows are those of the hashed path.
//!
//! Built per probe driver from the level-0 tables it holds, only when none
//! of them spilled; a NULL probe key or one outside the span finds nothing,
//! as in the hashed path. Null-safe (`<=>`) keys and null-aware anti joins
//! keep the hashed path. `krishiv.exec.join.directIndex` switches it off.
//!
//! Memory: the slot array is at most [`MAX_SLOTS`] `u64`s (128 KiB) per
//! probe driver and the build-time vectors at most [`MAX_ROWS`] entries
//! each, small and bounded, so they are not reserved from the pool. The
//! span limit is four slots a row with a floor of 64 slots (a build of
//! fewer than 16 rows may span 64 values).
//!
//! Build payload (perf lever P3): a direct build whose level-0 tables hold
//! at most [`MAX_PAYLOAD_BYTES`] also has its payload decoded once into
//! Arrow columns ([`DirectPayload`]), every row of every table in table
//! then row order, with the decoder the probe would use per pair. A probe
//! then fills its build columns with an Arrow `take` by row instead of
//! decoding one payload row per output row: the same values, since they
//! come from the same decoder (a string view `take` copies the views and
//! shares the buffers). The columns are of at most 4,096 rows and the
//! bytes are capped, so they are not reserved either.

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, UInt32Array};
use arrow_schema::DataType;

use super::keys::KeyCodec;
use super::table::{JoinTable, NONE, RowArena};
use crate::error::Result;
use crate::rows::RowSource;

/// The most build rows a direct index covers.
pub(crate) const MAX_ROWS: usize = 4096;

/// The most slots a direct index holds per build row.
pub(crate) const SPAN_PER_ROW: usize = 4;

/// The most slots of a direct index (a small build with a sparse key keeps
/// the hashed path).
pub(crate) const MAX_SLOTS: usize = 16 * 1024;

/// The most bytes the level-0 tables of a direct build may hold for its
/// payload to be decoded once ([`DirectPayload`]).
pub(crate) const MAX_PAYLOAD_BYTES: usize = 256 * 1024;

/// An empty slot.
const EMPTY: u64 = u64::MAX;

/// The direct index of one build: for key `k`, slot `k - min` holds
/// `partition << 32 | row`, or [`EMPTY`].
#[derive(Debug)]
pub(crate) struct Direct {
    min: i64,
    slots: Vec<u64>,
    /// The build columns decoded once, when the tables are small enough.
    payload: Option<Arc<DirectPayload>>,
}

impl Direct {
    /// The direct index of `tables` (the level-0 partitions; `spilled` the
    /// mask of those spilled), when the build qualifies (module docs).
    pub(crate) fn of(tables: &[Option<JoinTable>], spilled: u64, codec: &KeyCodec) -> Option<Self> {
        if spilled != 0 || codec.null_safe().iter().any(|&s| s) {
            return None;
        }
        let layout = codec.layout()?;
        let rows: usize = tables.iter().flatten().map(|t| t.arena().len()).sum();
        if rows == 0 || rows > MAX_ROWS {
            return None;
        }
        // Every indexed row's key bytes, with its partition and row.
        let mut at: Vec<(u32, u32)> = Vec::with_capacity(rows);
        let mut keys: Vec<&[u8]> = Vec::with_capacity(rows);
        for (p, t) in tables.iter().enumerate() {
            let Some(t) = t else { continue };
            let p = u32::try_from(p).ok()?;
            let arena = t.arena();
            for r in 0..u32::try_from(arena.len()).ok()? {
                if !arena.indexed(r) {
                    continue;
                }
                // Only row-layout key rows: arrow-row bytes (row layout
                // off, or a build column with no row arena) keep the
                // hashed path rather than being decoded as a layout row.
                keys.push(arena.layout_key_bytes(r)?);
                at.push((p, r));
            }
        }
        let cols = layout.decode(&keys).ok()?;
        let [col] = cols.as_slice() else {
            return None;
        };
        let values = int_values(col)?;
        let (mut min, mut max) = (i64::MAX, i64::MIN);
        for v in values.iter().flatten() {
            min = min.min(*v);
            max = max.max(*v);
        }
        if min > max {
            return None;
        }
        let span = usize::try_from(max.checked_sub(min)?)
            .ok()?
            .checked_add(1)?;
        if span > MAX_SLOTS || span > SPAN_PER_ROW.saturating_mul(rows.max(16)) {
            return None;
        }
        let mut slots = vec![EMPTY; span];
        // Rows in table order and, within a table, in row order: the last
        // row written to a slot is the newest of its key (a key lives in one
        // partition).
        for ((p, r), v) in at.iter().zip(&values) {
            let Some(v) = v else { continue };
            let i = usize::try_from(v - min).ok()?;
            *slots.get_mut(i)? = (u64::from(*p) << 32) | u64::from(*r);
        }
        Some(Self {
            min,
            slots,
            payload: None,
        })
    }

    /// This index with the build's payload decoded once (`payload`).
    pub(crate) fn with_payload(mut self, payload: Option<DirectPayload>) -> Self {
        self.payload = payload.map(Arc::new);
        self
    }

    /// The build's decoded payload, when it was decoded.
    pub(crate) fn payload(&self) -> Option<&Arc<DirectPayload>> {
        self.payload.as_ref()
    }

    /// For each row of `col` (the probe's key column, in the build key's
    /// type), its build partition and newest matching row, or [`NONE`];
    /// `None` when `col` is not of an integer type this index reads.
    pub(crate) fn find(&self, col: &ArrayRef, heads: &mut [u32], parts: &mut [u8]) -> Option<()> {
        let n = col.len();
        if heads.len() != n || parts.len() != n {
            return None;
        }
        let mut put = |i: usize, v: i64| {
            let slot = v
                .checked_sub(self.min)
                .and_then(|d| usize::try_from(d).ok())
                .and_then(|d| self.slots.get(d))
                .copied()
                .unwrap_or(EMPTY);
            if slot != EMPTY {
                // The partition fits in a byte: at most 2^MAX_RADIX_BITS.
                parts[i] = u8::try_from(slot >> 32).unwrap_or(0);
                heads[i] = u32::try_from(slot & u64::from(u32::MAX)).unwrap_or(NONE);
            }
        };
        match col.data_type() {
            DataType::Int32 => {
                let a = col.as_primitive_opt::<Int32Type>()?;
                probe_each(a.values(), a.nulls(), |i, v| put(i, i64::from(v)));
            }
            DataType::Int64 => {
                let a = col.as_primitive_opt::<Int64Type>()?;
                probe_each(a.values(), a.nulls(), put);
            }
            _ => return None,
        }
        Some(())
    }
}

/// The build columns of every row of a direct build's level-0 tables,
/// decoded once (module docs, perf lever P3): row `r` of partition `p` is
/// row `base[p] + r` of each column.
#[derive(Debug)]
pub(crate) struct DirectPayload {
    /// Each partition's first row in `columns`; `None` for a partition
    /// without a table.
    base: Vec<Option<u32>>,
    /// Each partition's row count.
    len: Vec<u32>,
    columns: Vec<ArrayRef>,
}

impl DirectPayload {
    /// The payload of `tables`, decoded by `decode` from the pairs
    /// `(partition, row)` of every row in table then row order; `None`
    /// when the tables hold more than [`MAX_PAYLOAD_BYTES`].
    pub(crate) fn of(
        tables: &[Option<JoinTable>],
        decode: impl FnOnce(&[(usize, u32)]) -> Result<Vec<ArrayRef>>,
    ) -> Result<Option<Self>> {
        let held: usize = tables
            .iter()
            .flatten()
            .map(|t| t.arena().held_bytes())
            .sum();
        if held > MAX_PAYLOAD_BYTES {
            return Ok(None);
        }
        let mut base = Vec::with_capacity(tables.len());
        let mut len = Vec::with_capacity(tables.len());
        let mut pairs = Vec::new();
        for (p, t) in tables.iter().enumerate() {
            let Some(t) = t else {
                base.push(None);
                len.push(0);
                continue;
            };
            let (Ok(start), Ok(n)) = (u32::try_from(pairs.len()), u32::try_from(t.arena().len()))
            else {
                return Ok(None);
            };
            base.push(Some(start));
            len.push(n);
            pairs.extend((0..n).map(|r| (p, r)));
        }
        let columns = decode(&pairs)?;
        Ok(Some(Self { base, len, columns }))
    }

    /// The build columns of the pairs `build` (partition, row), taken from
    /// the decoded columns; `None` when a pair is not one of its rows.
    pub(crate) fn columns(&self, build: &[(usize, u32)]) -> Result<Option<Vec<ArrayRef>>> {
        let mut ids = Vec::with_capacity(build.len());
        for &(p, r) in build {
            match (self.base.get(p).copied().flatten(), self.len.get(p)) {
                (Some(start), Some(&n)) if r < n => ids.push(start + r),
                _ => return Ok(None),
            }
        }
        let ids = UInt32Array::from(ids);
        let columns = self
            .columns
            .iter()
            .map(|c| {
                arrow_select::take::take(c.as_ref(), &ids, None)
                    .map_err(|e| crate::error::internal(e.to_string()))
            })
            .collect::<Result<_>>()?;
        Ok(Some(columns))
    }
}

/// The arena of partition `p` of `tables`, for a payload decoder.
pub(crate) fn arena_of(tables: &[Option<JoinTable>], p: usize) -> Option<&RowArena> {
    tables.get(p).and_then(Option::as_ref).map(JoinTable::arena)
}

/// Calls `f(i, value)` for every non-NULL row `i`.
fn probe_each<T: Copy>(
    values: &[T],
    nulls: Option<&arrow_buffer::NullBuffer>,
    mut f: impl FnMut(usize, T),
) {
    match nulls {
        None => values.iter().enumerate().for_each(|(i, &v)| f(i, v)),
        Some(n) => values
            .iter()
            .enumerate()
            .filter(|(i, _)| n.is_valid(*i))
            .for_each(|(i, &v)| f(i, v)),
    }
}

/// The values of an integer key column as `i64`, NULL as `None`.
fn int_values(col: &ArrayRef) -> Option<Vec<Option<i64>>> {
    match col.data_type() {
        DataType::Int32 => Some(
            col.as_primitive_opt::<Int32Type>()?
                .iter()
                .map(|v| v.map(i64::from))
                .collect(),
        ),
        DataType::Int64 => Some(col.as_primitive_opt::<Int64Type>()?.iter().collect()),
        _ => None,
    }
}
