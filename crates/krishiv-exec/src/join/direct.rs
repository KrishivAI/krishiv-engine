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
//! Memory: the slot array (at most [`MAX_SLOTS`] `u64`s, 128 KiB per probe
//! driver) is reserved from the driver's consumer for the index's life, and
//! the build-time vectors (at most [`MAX_ROWS`] entries each) while it is
//! built; a refused reservation keeps the hashed path. The span limit is four slots a row with a floor of 64 slots (a build of
//! fewer than 16 rows may span 64 values).

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use krishiv_memory::Reservation;

use super::keys::KeyCodec;
use super::table::{JoinTable, NONE};
use crate::rows::RowSource;

/// The most build rows a direct index covers.
pub(crate) const MAX_ROWS: usize = 4096;

/// The most slots a direct index holds per build row.
pub(crate) const SPAN_PER_ROW: usize = 4;

/// The most slots of a direct index (a small build with a sparse key keeps
/// the hashed path).
pub(crate) const MAX_SLOTS: usize = 16 * 1024;

/// An empty slot.
const EMPTY: u64 = u64::MAX;

/// The bytes a build row takes while the index is built: its partition and
/// row, its key bytes' slice, its decoded key and value.
const BUILD_BYTES_PER_ROW: usize = 64;

/// The direct index of one build: for key `k`, slot `k - min` holds
/// `partition << 32 | row`, or [`EMPTY`].
#[derive(Debug)]
pub(crate) struct Direct {
    min: i64,
    slots: Vec<u64>,
    /// The slot array's bytes, released when the index drops.
    _memory: Reservation,
}

impl Direct {
    /// The direct index of `tables` (the level-0 partitions; `spilled` the
    /// mask of those spilled), when the build qualifies (module docs) and
    /// `memory` grants its bytes.
    pub(crate) fn of(
        tables: &[Option<JoinTable>],
        spilled: u64,
        codec: &KeyCodec,
        mut memory: Reservation,
    ) -> Option<Self> {
        if spilled != 0 || codec.null_safe().iter().any(|&s| s) {
            return None;
        }
        let layout = codec.layout()?;
        let rows: usize = tables.iter().flatten().map(|t| t.arena().len()).sum();
        if rows == 0 || rows > MAX_ROWS {
            return None;
        }
        let building = rows.saturating_mul(BUILD_BYTES_PER_ROW);
        memory.try_grow(building).ok()?;
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
        let slot_bytes = span.saturating_mul(std::mem::size_of::<u64>());
        memory.try_grow(slot_bytes).ok()?;
        let mut slots = vec![EMPTY; span];
        // Rows in table order and, within a table, in row order: the last
        // row written to a slot is the newest of its key (a key lives in one
        // partition).
        for ((p, r), v) in at.iter().zip(&values) {
            let Some(v) = v else { continue };
            let i = usize::try_from(v - min).ok()?;
            *slots.get_mut(i)? = (u64::from(*p) << 32) | u64::from(*r);
        }
        drop((at, keys, values, cols));
        memory.shrink(building);
        Some(Self {
            min,
            slots,
            _memory: memory,
        })
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
