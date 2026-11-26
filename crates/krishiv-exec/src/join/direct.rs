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
//! Built once per build, by the first probe driver that reads the level-0
//! tables (`build.rs` `Level0::direct`, shared by every probe driver and
//! dropped with the tables), only when none of them spilled; a NULL probe key or one outside the span finds nothing,
//! as in the hashed path. Null-safe (`<=>`) keys and null-aware anti joins
//! keep the hashed path. `krishiv.exec.join.directIndex` switches it off.
//!
//! Memory: the slot array (at most [`MAX_SLOTS`] `u64`s, 128 KiB per
//! build) is reserved from the build's consumer for the index's life
//! (`Level0::direct_memory`), and the build-time vectors (at most
//! [`MAX_ROWS`] entries each) while it is built; a refused reservation keeps
//! the hashed path. The span limit is four slots a row with a floor of 64 slots (a build of
//! fewer than 16 rows may span 64 values).
//!
//! Build payload (perf lever P3): a direct build whose level-0 tables hold
//! at most [`MAX_PAYLOAD_BYTES`] also has its payload decoded once into
//! Arrow columns ([`DirectPayload`]), every row of every table in table
//! then row order, with the decoder the probe would use per pair. A probe
//! then fills its build columns with an Arrow `take` by row instead of
//! decoding one payload row per output row: the same values, since they
//! come from the same decoder (a string view `take` copies the views and
//! shares the buffers). A taken view column whose buffers hold more than
//! 16 times the bytes its views use is compacted (`gc`), as the probe
//! side's selections are (`compact_sparse_views`), so a small output batch
//! is not charged downstream for the whole payload. The columns (one copy
//! per build, at most 4,096 rows and 256 KiB plus builder slack) are
//! reserved from the index's reservation for its life: the cap before the
//! decode, trued up to the columns' size after it; a refused reservation
//! probes without them, decoding per pair.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, UInt32Array};
use arrow_schema::DataType;
use krishiv_memory::Reservation;

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

/// The bytes a build row takes while the index is built: its partition and
/// row, its key bytes' slice, its decoded key and value.
const BUILD_BYTES_PER_ROW: usize = 64;

/// The direct index of one build: for key `k`, slot `k - min` holds
/// `partition << 32 | row`, or [`EMPTY`].
#[derive(Debug)]
pub(crate) struct Direct {
    min: i64,
    slots: Vec<u64>,
    /// The build columns decoded once, when the tables are small enough.
    payload: Option<Arc<DirectPayload>>,
    /// The slot array's and the payload's bytes, released when the index
    /// drops.
    memory: Reservation,
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
            payload: None,
            memory,
        })
    }

    /// This index with the build's payload decoded once by `decode`, when
    /// its reservation grants the payload's bytes. [`MAX_PAYLOAD_BYTES`] is
    /// reserved before the decode, so the decode never holds bytes the pool
    /// has not granted, then trued up to the payload's size; a refusal
    /// before (`decode` does not run) or after keeps no payload.
    pub(crate) fn with_payload_of(
        mut self,
        decode: impl FnOnce() -> Option<DirectPayload>,
    ) -> Self {
        if self.memory.try_grow(MAX_PAYLOAD_BYTES).is_err() {
            return self;
        }
        let payload = decode();
        let size = payload.as_ref().map_or(0, DirectPayload::memory_size);
        let fits = match size.checked_sub(MAX_PAYLOAD_BYTES) {
            Some(extra) => self.memory.try_grow(extra).is_ok(),
            None => {
                self.memory.shrink(MAX_PAYLOAD_BYTES - size);
                true
            }
        };
        if fits {
            self.payload = payload.map(Arc::new);
        } else {
            self.memory.shrink(MAX_PAYLOAD_BYTES);
        }
        self
    }

    /// The bytes this index holds in its reservation.
    #[cfg(test)]
    pub(crate) fn reserved(&self) -> usize {
        self.memory.size()
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
        let lens: Vec<Option<usize>> = tables
            .iter()
            .map(|t| t.as_ref().map(|t| t.arena().len()))
            .collect();
        Self::of_lens(held, &lens, decode)
    }

    /// [`of`](Self::of) over the tables' held bytes and each partition's
    /// row count (`None`: no table).
    fn of_lens(
        held: usize,
        lens: &[Option<usize>],
        decode: impl FnOnce(&[(usize, u32)]) -> Result<Vec<ArrayRef>>,
    ) -> Result<Option<Self>> {
        if held > MAX_PAYLOAD_BYTES {
            return Ok(None);
        }
        let mut base = Vec::with_capacity(lens.len());
        let mut len = Vec::with_capacity(lens.len());
        let mut pairs = Vec::new();
        for (p, n) in lens.iter().enumerate() {
            let Some(n) = *n else {
                base.push(None);
                len.push(0);
                continue;
            };
            let (Ok(start), Ok(n)) = (u32::try_from(pairs.len()), u32::try_from(n)) else {
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
            match self.id(p, r) {
                Some(id) => ids.push(id),
                None => return Ok(None),
            }
        }
        self.take(&UInt32Array::from(ids)).map(Some)
    }

    /// The decoded row of build row `r` of partition `p`, when it is one.
    #[inline]
    pub(crate) fn id(&self, p: usize, r: u32) -> Option<u32> {
        match (self.base.get(p).copied().flatten(), self.len.get(p)) {
            (Some(start), Some(&n)) if r < n => Some(start + r),
            _ => None,
        }
    }

    /// The build columns of the decoded rows `ids` (each one from
    /// [`id`](Self::id)).
    pub(crate) fn take(&self, ids: &UInt32Array) -> Result<Vec<ArrayRef>> {
        self.columns
            .iter()
            .map(|c| {
                arrow_select::take::take(c.as_ref(), ids, None)
                    .map(compact_sparse)
                    .map_err(|e| crate::error::internal(e.to_string()))
            })
            .collect()
    }

    /// The number of decoded rows (every row of every table).
    pub(crate) fn rows(&self) -> usize {
        self.columns.first().map_or(0, Array::len)
    }

    /// The bytes the decoded columns and the partition offsets hold.
    pub(crate) fn memory_size(&self) -> usize {
        let columns: usize = self.columns.iter().map(Array::get_array_memory_size).sum();
        columns
            + self.base.capacity() * std::mem::size_of::<Option<u32>>()
            + self.len.capacity() * std::mem::size_of::<u32>()
    }
}

/// `col` compacted (`gc`) when it is a view column whose data buffers hold
/// more than 16 times the bytes its views use (module docs).
fn compact_sparse(col: ArrayRef) -> ArrayRef {
    fn sparse(held: usize, used: usize) -> bool {
        held > used.saturating_mul(16)
    }
    match col.data_type() {
        DataType::Utf8View => {
            let a = col.as_string_view();
            let held = a.data_buffers().iter().map(arrow_buffer::Buffer::len).sum();
            if sparse(held, a.total_buffer_bytes_used()) {
                return Arc::new(a.gc());
            }
        }
        DataType::BinaryView => {
            let a = col.as_binary_view();
            let held = a.data_buffers().iter().map(arrow_buffer::Buffer::len).sum();
            if sparse(held, a.total_buffer_bytes_used()) {
                return Arc::new(a.gc());
            }
        }
        _ => {}
    }
    col
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, Int64Array, StringViewArray};

    use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};

    use super::{Direct, DirectPayload, MAX_PAYLOAD_BYTES, compact_sparse};
    use crate::error::Result;

    /// Column 0: the pair's partition * 100 + row; column 1: a view of
    /// 40-byte strings (not inlined).
    fn decode(pairs: &[(usize, u32)]) -> Result<Vec<ArrayRef>> {
        let ids: Vec<i64> = pairs
            .iter()
            .map(|&(p, r)| i64::try_from(p * 100).unwrap_or(0) + i64::from(r))
            .collect();
        let text: StringViewArray = ids.iter().map(|v| Some(format!("{v:0>40}"))).collect();
        Ok(vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(text) as ArrayRef,
        ])
    }

    fn ids(cols: &[ArrayRef]) -> Vec<i64> {
        cols[0]
            .as_primitive::<arrow_array::types::Int64Type>()
            .values()
            .to_vec()
    }

    #[test]
    fn pairs_map_to_their_rows_across_missing_partitions() -> Result<()> {
        let lens = [Some(3), None, Some(2), Some(0), Some(4)];
        let Some(payload) = DirectPayload::of_lens(0, &lens, decode)? else {
            panic!("a payload under the cap is decoded");
        };
        assert_eq!(payload.rows(), 9);
        let build = [(4, 3), (0, 0), (2, 1), (4, 0), (0, 2)];
        let Some(cols) = payload.columns(&build)? else {
            panic!("every pair is a row of the payload");
        };
        assert_eq!(ids(&cols), vec![403, 0, 201, 400, 2]);
        assert_eq!(cols[1].as_string_view().value(2), format!("{:0>40}", 201));
        Ok(())
    }

    #[test]
    fn a_pair_outside_the_payload_falls_back() -> Result<()> {
        let lens = [Some(3), None, Some(2)];
        let Some(payload) = DirectPayload::of_lens(0, &lens, decode)? else {
            panic!("a payload under the cap is decoded");
        };
        // A row past its partition, a partition without a table, one past
        // the last partition.
        for pair in [(0, 3), (1, 0), (3, 0)] {
            assert!(payload.columns(&[(0, 0), pair])?.is_none(), "{pair:?}");
        }
        Ok(())
    }

    #[test]
    fn a_payload_over_the_cap_is_not_decoded() -> Result<()> {
        let mut called = false;
        let over = DirectPayload::of_lens(MAX_PAYLOAD_BYTES + 1, &[Some(1)], |pairs| {
            called = true;
            decode(pairs)
        })?;
        assert!(over.is_none());
        assert!(!called, "an over-cap payload is never decoded");
        assert!(DirectPayload::of_lens(MAX_PAYLOAD_BYTES, &[Some(1)], decode)?.is_some());
        Ok(())
    }

    #[test]
    fn a_payload_without_columns_takes_none() -> Result<()> {
        let Some(payload) = DirectPayload::of_lens(0, &[Some(2)], |_| Ok(Vec::new()))? else {
            panic!("a payload under the cap is decoded");
        };
        assert_eq!(payload.columns(&[(0, 1)])?.map(|c| c.len()), Some(0));
        Ok(())
    }

    #[test]
    fn a_decode_error_is_returned() {
        let err = DirectPayload::of_lens(0, &[Some(1)], |_| {
            Err(crate::error::internal("bad payload".to_owned()))
        });
        assert!(err.is_err());
    }

    fn held(col: &ArrayRef) -> usize {
        col.as_string_view()
            .data_buffers()
            .iter()
            .map(arrow_buffer::Buffer::len)
            .sum()
    }

    #[test]
    fn a_small_take_of_views_does_not_pin_the_payload() -> Result<()> {
        let lens = [Some(4_000)];
        let Some(payload) = DirectPayload::of_lens(0, &lens, decode)? else {
            panic!("a payload under the cap is decoded");
        };
        let whole = held(&payload.columns[1]);
        let Some(one) = payload.columns(&[(0, 7)])? else {
            panic!("the pair is a row of the payload");
        };
        assert!(
            held(&one[1]) <= 64,
            "one row holds {} of {whole}",
            held(&one[1])
        );
        assert_eq!(one[1].as_string_view().value(0), format!("{:0>40}", 7));
        // A dense take keeps sharing the buffers.
        let all: Vec<(usize, u32)> = (0..4_000).map(|r| (0, r)).collect();
        let Some(dense) = payload.columns(&all)? else {
            panic!("the pairs are rows of the payload");
        };
        assert_eq!(held(&dense[1]), whole);
        Ok(())
    }

    #[test]
    fn other_columns_are_not_compacted() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        assert!(Arc::ptr_eq(&compact_sparse(Arc::clone(&col)), &col));
        let bin: ArrayRef = Arc::new(arrow_array::BinaryViewArray::from_iter_values(
            (0..1_000).map(|i| format!("{i:0>40}").into_bytes()),
        ));
        let one =
            arrow_select::take::take(bin.as_ref(), &arrow_array::UInt32Array::from(vec![3]), None)
                .map(compact_sparse);
        let Ok(one) = one else { panic!("take") };
        assert_eq!(
            one.as_binary_view().value(0),
            format!("{:0>40}", 3).as_bytes()
        );
        assert!(
            one.as_binary_view()
                .data_buffers()
                .iter()
                .map(arrow_buffer::Buffer::len)
                .sum::<usize>()
                <= 64
        );
    }

    /// An empty index holding a reservation of a pool limited to `limit`.
    fn index(limit: Option<usize>) -> Direct {
        let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        let query = root.child("q", limit);
        let id = query.register("direct", None);
        Direct {
            min: 0,
            slots: Vec::new(),
            payload: None,
            memory: Reservation::new(query, id),
        }
    }

    #[test]
    fn the_payload_is_reserved_for_the_index_life_or_left_out() -> Result<()> {
        let lens = [Some(40), Some(24)];
        let Some(payload) = DirectPayload::of_lens(0, &lens, decode)? else {
            panic!("a payload under the cap is decoded");
        };
        let bytes = payload.memory_size();
        assert!(bytes > 64 * 40, "the payload holds its strings: {bytes}");
        let kept = index(None).with_payload_of(|| Some(payload));
        assert!(kept.payload().is_some());
        assert_eq!(kept.reserved(), bytes);

        // The cap is reserved before the decode: a pool that grants the
        // payload's bytes but not the cap decodes nothing.
        let mut ran = false;
        let refused = index(Some(MAX_PAYLOAD_BYTES - 1)).with_payload_of(|| {
            ran = true;
            None
        });
        assert!(!ran, "refused before the decode");
        assert!(
            refused.payload().is_none(),
            "a refused reservation probes per pair"
        );
        assert_eq!(refused.reserved(), 0);

        // A decode that gives no payload keeps nothing reserved.
        let none = index(None).with_payload_of(|| None);
        assert!(none.payload().is_none());
        assert_eq!(none.reserved(), 0);
        Ok(())
    }

    #[test]
    fn a_payload_past_the_cap_is_trued_up_or_refused_after_the_decode() -> Result<()> {
        // Long strings: the decoded columns hold more than the cap.
        let wide = |pairs: &[(usize, u32)]| -> Result<Vec<ArrayRef>> {
            let text: StringViewArray = pairs
                .iter()
                .map(|&(_, r)| Some(format!("{r:0>200}")))
                .collect();
            Ok(vec![Arc::new(text) as ArrayRef])
        };
        let lens = [Some(2_000)];
        let payload = || DirectPayload::of_lens(0, &lens, wide).ok().flatten();
        let Some(bytes) = payload().map(|p| p.memory_size()) else {
            panic!("decoded");
        };
        assert!(bytes > MAX_PAYLOAD_BYTES, "{bytes}");
        let kept = index(None).with_payload_of(payload);
        assert!(kept.payload().is_some());
        assert_eq!(kept.reserved(), bytes);
        let refused = index(Some(bytes - 1)).with_payload_of(payload);
        assert!(refused.payload().is_none());
        assert_eq!(refused.reserved(), 0);
        Ok(())
    }
}
