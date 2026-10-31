// SPDX-License-Identifier: Apache-2.0
//! The join hash table: rows of one partition in a row arena, indexed by
//! the shared [`KeyTable`] (E02: `hashbrown::HashTable` over a key arena)
//! or, in the row layout, by a [`JoinIndex`] over the arena's key rows.
//!
//! Build rows live in a **row arena**: Arrow batches of `hash: UInt64`, the
//! key and, when the join needs build columns, `payload: LargeBinary`
//! (arrow-row bytes of the whole build row), with the build weights. The
//! key is `key: LargeBinary` (arrow-row key bytes), or in the row layout
//! `key: FixedSizeBinary(stride)` (the [`RowLayout`] rows: P2-T157's page
//! frame, so a probe reads one contiguous row) plus, for string keys,
//! `key_heap: LargeBinary` (each row's long values). The same batches are
//! what a spilled partition writes, so spilling and reloading never
//! re-encode. Each distinct key points to its newest row; older rows with
//! the same key are chained through [`JoinTable::next`], so duplicate keys
//! never lengthen a probe sequence.
//!
//! A join that returns its build rows by their matches (right semi, right
//! anti, right outer; design P2-T166) gives the table a match bitmap
//! ([`JoinTable::with_marks`]): probe drivers set a row's bit when a pair
//! with it matches, and once every probe driver's input ended (the flush
//! barrier orders those writes before the reads) the rows are emitted by
//! their bits.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, LargeBinaryArray, UInt64Array};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_types::batch::{Batch, Weights};

use super::keys::{Encoded, KeyRows};
use crate::error::{Result, internal};
use crate::hash::{KeyTable, ProbeCount};
use crate::rows::{IndexChoice, JoinIndex, RowLayout, RowSource};

/// End of a chain / no match.
pub(crate) const NONE: u32 = u32::MAX;
const _: () = assert!(NONE == crate::rows::NO_MATCH);

/// How an arena stores its keys and payload.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ArenaFormat<'a> {
    /// Arrow-row keys, and arrow-row payload when `payload`.
    Arrow { payload: bool },
    /// Rows of `keys`, and the build columns as rows of `payload` or, when
    /// `payload` is `None` and `arrow_payload`, as arrow rows.
    Rows {
        keys: &'a RowLayout,
        payload: Option<&'a RowLayout>,
        arrow_payload: bool,
    },
}

/// The fields holding rows of `layout` as `name` (and `name_heap`).
fn row_fields(name: &str, layout: &RowLayout, fields: &mut Vec<Field>) -> Result<()> {
    let width =
        i32::try_from(layout.stride()).map_err(|_| internal("join row too wide".to_owned()))?;
    fields.push(Field::new(name, DataType::FixedSizeBinary(width), false));
    if layout.has_views() {
        fields.push(Field::new(
            format!("{name}_heap"),
            DataType::LargeBinary,
            false,
        ));
    }
    Ok(())
}

/// The schema of the row arena's batches in `format`.
pub(crate) fn arena_schema(format: ArenaFormat<'_>) -> Result<SchemaRef> {
    let mut fields = vec![Field::new("hash", DataType::UInt64, false)];
    match format {
        ArenaFormat::Arrow { payload } => {
            fields.push(Field::new("key", DataType::LargeBinary, false));
            if payload {
                fields.push(Field::new("payload", DataType::LargeBinary, false));
            }
        }
        ArenaFormat::Rows {
            keys,
            payload,
            arrow_payload,
        } => {
            row_fields("key", keys, &mut fields)?;
            match payload {
                Some(p) => row_fields("payload", p, &mut fields)?,
                None if arrow_payload => {
                    fields.push(Field::new("payload", DataType::LargeBinary, false));
                }
                None => {}
            }
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// The arena column flagging rows with a NULL `=` key (right anti and
/// right outer joins: stored, emitted as unmatched, never indexed).
pub(crate) const NULL_KEY: &str = "null_key";

/// `schema` with the [`NULL_KEY`] column appended.
pub(crate) fn with_null_key(schema: &SchemaRef) -> SchemaRef {
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(NULL_KEY, DataType::Boolean, false));
    Arc::new(Schema::new(fields))
}

/// Bytes of the arrow-row key rows in arena chunks (0 for row-layout
/// chunks: their index copies no key bytes).
pub(crate) fn key_bytes(chunks: &[Batch]) -> usize {
    let one = |c: &Batch| {
        let keys = c
            .data()
            .columns()
            .get(1)
            .and_then(|k| k.as_binary_opt::<i64>());
        keys.map_or(0, |k| k.values().len())
    };
    chunks.iter().map(one).sum()
}

/// Row-layout rows of `stride` bytes back to back, and each row's long
/// values when the layout has views.
#[derive(Debug)]
struct RowCols {
    fixed: Buffer,
    stride: usize,
    heap: Option<LargeBinaryArray>,
}

impl RowCols {
    /// The rows of column `name` (and `name_heap`) of `batch`, if that
    /// column holds rows.
    fn of(batch: &Batch, name: &str) -> Result<Option<Self>> {
        let bad = || internal("malformed join row arena".to_owned());
        let Some(col) = batch.data().column_by_name(name) else {
            return Ok(None);
        };
        let Some(fixed) = col.as_fixed_size_binary_opt() else {
            return Ok(None);
        };
        let stride = usize::try_from(fixed.value_length()).map_err(|_| bad())?;
        let data = fixed.to_data();
        let values = data.buffers().first().cloned().ok_or_else(bad)?;
        let start = data.offset().saturating_mul(stride);
        let len = fixed.len().saturating_mul(stride);
        if stride == 0 || start.saturating_add(len) > values.len() {
            return Err(bad());
        }
        let heap = match batch.data().column_by_name(&format!("{name}_heap")) {
            Some(h) => Some(h.as_binary_opt::<i64>().cloned().ok_or_else(bad)?),
            None => None,
        };
        if heap.as_ref().is_some_and(|h| h.len() != fixed.len()) {
            return Err(bad());
        }
        Ok(Some(Self {
            fixed: values.slice_with_length(start, len),
            stride,
            heap,
        }))
    }

    /// Row `id` and its heap (empty if out of range).
    #[inline]
    fn prefetch(&self, id: u32) {
        krishiv_kernels::prefetch_read(&self.fixed, (id as usize).saturating_mul(self.stride));
    }

    #[inline]
    fn row(&self, id: u32) -> (&[u8], &[u8]) {
        let at = (id as usize).saturating_mul(self.stride);
        let row = self
            .fixed
            .get(at..at.saturating_add(self.stride))
            .unwrap_or_default();
        let heap = self
            .heap
            .as_ref()
            .filter(|h| (id as usize) < h.len())
            .map_or(&[][..], |h| h.value(id as usize));
        (row, heap)
    }
}

/// An arena's keys or payload.
#[derive(Debug)]
enum Cols {
    /// Arrow-row bytes.
    Arrow(LargeBinaryArray),
    /// Row-layout rows.
    Rows(RowCols),
}

impl Cols {
    fn of(batch: &Batch, name: &str) -> Result<Option<Self>> {
        if let Some(rows) = RowCols::of(batch, name)? {
            return Ok(Some(Self::Rows(rows)));
        }
        match batch.data().column_by_name(name) {
            Some(c) => c
                .as_binary_opt::<i64>()
                .cloned()
                .map(|b| Some(Self::Arrow(b)))
                .ok_or_else(|| internal("malformed join row arena".to_owned())),
            None => Ok(None),
        }
    }
}

/// One partition's build rows, coalesced into a single batch.
#[derive(Debug)]
pub(crate) struct RowArena {
    batch: Batch,
    hashes: UInt64Array,
    keys: Cols,
    payload: Option<Cols>,
    /// Rows with a NULL `=` key that a right anti or right outer join
    /// keeps: stored, never indexed (column `null_key`, when present).
    null_key: Option<arrow_array::BooleanArray>,
}

impl RowArena {
    /// Concatenates arena chunks (same schema).
    pub(crate) fn concat(schema: &SchemaRef, chunks: &[Batch]) -> Result<Self> {
        let batch = match chunks {
            [one] => one.clone(),
            _ => Batch::concat(schema, chunks).map_err(|e| internal(e.to_string()))?,
        };
        let bad = || internal("malformed join row arena".to_owned());
        let hashes = batch.data().column_by_name("hash").ok_or_else(bad)?;
        let hashes = hashes
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(bad)?
            .clone();
        let keys = Cols::of(&batch, "key")?.ok_or_else(bad)?;
        let payload = Cols::of(&batch, "payload")?;
        let null_key = match batch.data().column_by_name(NULL_KEY) {
            Some(c) => Some(c.as_boolean_opt().ok_or_else(bad)?.clone()),
            None => None,
        };
        if u32::try_from(batch.num_rows()).is_err() {
            return Err(internal("a join partition has over 2^32 rows".to_owned()));
        }
        Ok(Self {
            batch,
            hashes,
            keys,
            payload,
            null_key,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.batch.num_rows()
    }

    pub(crate) fn hash(&self, row: u32) -> u64 {
        self.hashes.value(row as usize)
    }

    /// The arrow-row key bytes, if the keys are in that format.
    fn arrow_keys(&self) -> Option<&LargeBinaryArray> {
        match &self.keys {
            Cols::Arrow(k) => Some(k),
            Cols::Rows(_) => None,
        }
    }

    /// The row's key bytes, comparable byte for byte with another row's of
    /// the same format (equal keys have equal bytes): arrow-row bytes, or a
    /// row-layout row. `None` for a layout with string views, whose rows
    /// point into a heap.
    pub(crate) fn key_bytes(&self, row: u32) -> Option<&[u8]> {
        let row = row as usize;
        match &self.keys {
            Cols::Arrow(k) => Some(k.value(row)),
            Cols::Rows(RowCols {
                fixed,
                stride,
                heap: None,
            }) => fixed.get(row * stride..(row + 1) * stride),
            Cols::Rows(RowCols { heap: Some(_), .. }) => None,
        }
    }

    /// The row's arrow-row payload bytes (empty if the arena has none or
    /// holds payload rows).
    pub(crate) fn payload(&self, row: u32) -> &[u8] {
        match &self.payload {
            Some(Cols::Arrow(p)) if (row as usize) < p.len() => p.value(row as usize),
            _ => &[],
        }
    }

    /// The row's payload row and its heap (empty if the arena has none or
    /// holds arrow-row payload).
    pub(crate) fn payload_row(&self, row: u32) -> (&[u8], &[u8]) {
        match &self.payload {
            Some(Cols::Rows(p)) => p.row(row),
            _ => (&[], &[]),
        }
    }

    /// The row's weight.
    pub(crate) fn weight(&self, row: u32) -> i64 {
        match self.batch.weights() {
            Weights::Const(w) => *w,
            Weights::Column(c) => c.value(row as usize),
        }
    }

    /// Whether every weight is 1.
    pub(crate) fn is_unweighted(&self) -> bool {
        self.batch.is_unweighted()
    }
}

impl RowSource for RowArena {
    #[inline]
    fn prefetch(&self, id: u32) {
        if let Cols::Rows(k) = &self.keys {
            k.prefetch(id);
        }
    }

    #[inline]
    fn row(&self, id: u32) -> (&[u8], &[u8]) {
        match &self.keys {
            Cols::Rows(k) => k.row(id),
            Cols::Arrow(_) => (&[], &[]),
        }
    }

    #[inline]
    fn hash(&self, id: u32) -> u64 {
        self.hashes.values().get(id as usize).copied().unwrap_or(0)
    }

    #[inline]
    fn indexed(&self, id: u32) -> bool {
        self.null_key
            .as_ref()
            .is_none_or(|f| (id as usize) >= f.len() || !f.value(id as usize))
    }
}

/// The kind of a table's index, for `EXPLAIN ANALYZE` (a bit each, so a
/// join's tables can report every kind they used).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IndexKind {
    /// [`KeyTable`] over arrow-row keys, duplicates chained.
    Swiss = 1,
    /// The row layout's flat [`JoinIndex`], duplicates chained.
    RowFlat = 2,
    /// The row layout's salted [`JoinIndex`], duplicates chained.
    RowSalted = 4,
}

impl IndexKind {
    /// The kinds a bit set names, as `EXPLAIN ANALYZE` prints them.
    pub(crate) fn names(bits: u64) -> String {
        [
            (Self::Swiss, "swiss+chain"),
            (Self::RowFlat, "row-flat+chain"),
            (Self::RowSalted, "row-salted+chain"),
        ]
        .iter()
        .filter(|(k, _)| bits & (*k as u64) != 0)
        .map(|(_, n)| *n)
        .collect::<Vec<_>>()
        .join("; ")
    }
}

/// A table's key index.
#[derive(Debug)]
enum Index {
    /// The shared [`KeyTable`] over arrow-row keys and each key's newest
    /// row.
    Arrow { keys: KeyTable, heads: Vec<u32> },
    /// A [`JoinIndex`] over row-layout keys (it holds the newest rows).
    Rows {
        index: JoinIndex,
        layout: Arc<RowLayout>,
    },
}

/// An immutable join table over one partition's rows: the shared
/// [`KeyTable`] (hashbrown `HashTable<u32>`, E02 decision B), or a
/// [`JoinIndex`] in the row layout, maps each distinct key to its newest
/// row, and older rows with the same key are chained through
/// [`next`](Self::next).
#[derive(Debug)]
pub(crate) struct JoinTable {
    arena: RowArena,
    index: Index,
    next: Vec<u32>,
    /// One bit per row: a pair with the row matched (empty unless
    /// [`with_marks`](Self::with_marks)).
    marks: Vec<AtomicU64>,
}

impl JoinTable {
    /// Bytes the index over `rows` rows with `key_bytes` bytes of arrow-row
    /// keys takes (reserved before [`build`](Self::build)). Arrow rows: the
    /// key table at worst one key per row, a chain head per key and a link
    /// per row. Row layout: the [`JoinIndex`] sized for one key per row and
    /// a link per row (it copies no keys and holds the heads itself).
    pub(crate) fn index_bytes(rows: usize, key_bytes: usize, row_layout: bool) -> usize {
        if row_layout {
            IndexChoice::Adaptive
                .bytes(rows)
                .saturating_add(rows * size_of::<u32>())
        } else {
            KeyTable::new().projected_size(rows, key_bytes) + rows * 2 * size_of::<u32>()
        }
    }

    /// [`index_bytes`](Self::index_bytes) of an index sized from a distinct-key
    /// estimate ([`build_expecting`](Self::build_expecting)): an estimate
    /// that was low makes the index grow, and while it grows it holds the
    /// old table beside the new one, at most half the final size again.
    /// The caller shrinks the reservation to [`index_memory`](Self::index_memory)
    /// once the table is built.
    pub(crate) fn index_bytes_estimated(rows: usize, key_bytes: usize, row_layout: bool) -> usize {
        let base = Self::index_bytes(rows, key_bytes, row_layout);
        if row_layout {
            base.saturating_add(IndexChoice::Adaptive.bytes(rows) / 2)
        } else {
            base
        }
    }

    /// Indexes `arena`; row-layout keys are indexed under `layout` with the
    /// table-local seed `g_seed`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the index cannot be allocated, or row-layout keys
    /// come without their layout.
    pub(crate) fn build(
        arena: RowArena,
        layout: Option<&Arc<RowLayout>>,
        g_seed: u64,
    ) -> Result<Self> {
        Self::build_with(arena, layout, g_seed, IndexChoice::Adaptive, None)
    }

    /// [`build`](Self::build) of a row-layout arena sized for about `keys`
    /// distinct keys (a cardinality estimate): see
    /// [`JoinIndex::build_expecting`].
    pub(crate) fn build_expecting(
        arena: RowArena,
        layout: Option<&Arc<RowLayout>>,
        g_seed: u64,
        keys: Option<usize>,
    ) -> Result<Self> {
        Self::build_with(arena, layout, g_seed, IndexChoice::Adaptive, keys)
    }

    /// Bytes the index and the chains hold (the arena not included): what a
    /// build's [`index_bytes`](Self::index_bytes) reservation shrinks to.
    pub(crate) fn index_memory(&self) -> usize {
        let index = match &self.index {
            Index::Arrow { keys, heads } => {
                keys.memory_size() + heads.capacity() * size_of::<u32>()
            }
            Index::Rows { index, .. } => index.memory_size(),
        };
        index + self.next.capacity() * size_of::<u32>() + self.marks.capacity() * size_of::<u64>()
    }

    /// Bytes of the match bitmap of a table of `rows` rows
    /// ([`with_marks`](Self::with_marks)).
    pub(crate) fn marks_bytes(rows: usize) -> usize {
        rows.div_ceil(64) * size_of::<u64>()
    }

    /// This table with a match bitmap, every bit clear.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the bitmap cannot be allocated.
    pub(crate) fn with_marks(mut self) -> Result<Self> {
        let words = self.arena.len().div_ceil(64);
        let mut marks = Vec::new();
        marks
            .try_reserve_exact(words)
            .map_err(|e| internal(e.to_string()))?;
        marks.resize_with(words, || AtomicU64::new(0));
        self.marks = marks;
        Ok(self)
    }

    /// Whether a pair with `row` matched (`false` without a bitmap).
    #[inline]
    pub(crate) fn is_marked(&self, row: u32) -> bool {
        self.marks
            .get(row as usize / 64)
            .is_some_and(|w| w.load(Ordering::Relaxed) & (1 << (row % 64)) != 0)
    }

    /// Records that a pair with `row` matched. Probe drivers mark
    /// concurrently; a bit already set is not written again, so rows many
    /// probes match keep their cache line shared.
    #[inline]
    pub(crate) fn mark(&self, row: u32) {
        if let Some(w) = self.marks.get(row as usize / 64) {
            let bit = 1 << (row % 64);
            if w.load(Ordering::Relaxed) & bit == 0 {
                w.fetch_or(bit, Ordering::Relaxed);
            }
        }
    }

    /// Marks `head` and every older row with its key (no residual
    /// condition: every row of the chain matches). A chain whose head is
    /// marked was marked whole already.
    pub(crate) fn mark_chain(&self, head: u32) {
        if head == NONE || self.is_marked(head) {
            return;
        }
        let mut r = head;
        while r != NONE {
            self.mark(r);
            r = self.next.get(r as usize).copied().unwrap_or(NONE);
        }
    }

    /// Up to `limit` rows in `*from..end` whose mark is `marked`, appended
    /// to `out`; `*from` moves past the rows looked at.
    pub(crate) fn rows_marked(
        &self,
        marked: bool,
        from: &mut u32,
        end: u32,
        limit: usize,
        out: &mut Vec<u32>,
    ) {
        let n = u32::try_from(self.arena.len()).unwrap_or(u32::MAX).min(end);
        while *from < n && out.len() < limit {
            if self.is_marked(*from) == marked {
                out.push(*from);
            }
            *from += 1;
        }
    }

    /// [`build`](Self::build) with the row-layout index forced (tests).
    pub(crate) fn build_with(
        arena: RowArena,
        layout: Option<&Arc<RowLayout>>,
        g_seed: u64,
        choice: IndexChoice,
        keys: Option<usize>,
    ) -> Result<Self> {
        let n = arena.len();
        let mut next: Vec<u32> = Vec::new();
        let oom = |e: std::collections::TryReserveError| internal(e.to_string());
        next.try_reserve_exact(n).map_err(oom)?;
        next.resize(n, NONE);
        let index = match (arena.arrow_keys(), layout) {
            (Some(arrow), _) => {
                let mut keys = KeyTable::new();
                keys.reserve(n, arrow.values().len());
                let mut heads: Vec<u32> = Vec::new();
                heads.try_reserve_exact(n).map_err(oom)?;
                for row in (0..u32::try_from(n).unwrap_or(0)).filter(|&r| arena.indexed(r)) {
                    let (id, new) =
                        keys.insert_hashed(arrow.value(row as usize), arena.hash(row))?;
                    if new {
                        heads.push(row);
                    } else if let Some(head) = heads.get_mut(id as usize) {
                        // Newest first: the older head follows this row.
                        next[row as usize] = *head;
                        *head = row;
                    }
                }
                Index::Arrow { keys, heads }
            }
            (None, Some(layout)) => Index::Rows {
                index: JoinIndex::build_expecting(
                    layout,
                    &arena,
                    n,
                    keys.unwrap_or(n),
                    g_seed,
                    choice,
                    &mut next,
                )?,
                layout: Arc::clone(layout),
            },
            (None, None) => return Err(internal("join key rows without a layout".to_owned())),
        };
        Ok(Self {
            arena,
            index,
            next,
            marks: Vec::new(),
        })
    }

    /// For each probe row `rows[j]` of `keys` (encoded in this table's key
    /// format), the newest build row with an equal key, or [`NONE`].
    pub(crate) fn find_heads(&self, keys: &Encoded, rows: &[u32], heads: &mut Vec<u32>) {
        match (&self.index, &keys.rows) {
            (
                Index::Arrow {
                    keys: table,
                    heads: h,
                },
                KeyRows::Arrow(encoded, hashes),
            ) => {
                heads.extend(rows.iter().map(|&i| {
                    let key = encoded.row(i as usize);
                    let id = table.find(key.as_ref(), hashes[i as usize]);
                    id.and_then(|id| h.get(id as usize).copied())
                        .unwrap_or(NONE)
                }));
            }
            (Index::Rows { index, layout }, KeyRows::Layout(encoded)) => {
                index.find(layout, encoded, rows, &self.arena, heads);
            }
            // Both sides of one join instance use one format.
            _ => heads.extend(rows.iter().map(|_| NONE)),
        }
    }

    /// Whether the row-layout index of this table is the flat one (`None`
    /// for arrow-row keys).
    #[cfg(test)]
    pub(crate) fn flat_index(&self) -> Option<bool> {
        match &self.index {
            Index::Arrow { .. } => None,
            Index::Rows { index, .. } => Some(index.is_flat()),
        }
    }

    /// As [`find_heads`](Self::find_heads), counting probe lengths (keys
    /// compared per lookup) into `count`.
    pub(crate) fn find_heads_counted(
        &self,
        keys: &Encoded,
        rows: &[u32],
        heads: &mut Vec<u32>,
        count: &mut ProbeCount,
    ) {
        match (&self.index, &keys.rows) {
            (
                Index::Arrow {
                    keys: table,
                    heads: h,
                },
                KeyRows::Arrow(encoded, hashes),
            ) => heads.extend(rows.iter().map(|&i| {
                let key = encoded.row(i as usize);
                let id = table.find_counted(key.as_ref(), hashes[i as usize], count);
                id.and_then(|id| h.get(id as usize).copied())
                    .unwrap_or(NONE)
            })),
            (Index::Rows { index, layout }, KeyRows::Layout(encoded)) => {
                index.find_counted(layout, encoded, rows, &self.arena, heads, count);
            }
            _ => self.find_heads(keys, rows, heads),
        }
    }

    /// Distinct keys and buckets of the index, and its kind.
    pub(crate) fn index_size(&self) -> (usize, usize, IndexKind) {
        match &self.index {
            Index::Arrow { keys, .. } => (keys.len(), keys.buckets(), IndexKind::Swiss),
            Index::Rows { index, .. } => {
                let kind = if index.is_flat() {
                    IndexKind::RowFlat
                } else {
                    IndexKind::RowSalted
                };
                (index.len(), index.buckets(), kind)
            }
        }
    }

    /// `next()[r]`: the next build row with the same key as `r`, or [`NONE`].
    pub(crate) fn next(&self) -> &[u32] {
        &self.next
    }

    /// The rows.
    pub(crate) fn arena(&self) -> &RowArena {
        &self.arena
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatch;

    fn arena(hashes: Vec<u64>, keys: Vec<&[u8]>) -> RowArena {
        let schema =
            arena_schema(ArenaFormat::Arrow { payload: false }).unwrap_or_else(|e| panic!("{e}"));
        let data = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(UInt64Array::from(hashes)),
                Arc::new(LargeBinaryArray::from(keys)),
            ],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        RowArena::concat(&schema, &[Batch::new(data)]).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn duplicates_chain_and_collisions_compare_keys() {
        // Rows 0 and 2 share a key; row 1 collides on the hash only.
        let t = JoinTable::build(arena(vec![7, 7, 7], vec![b"a", b"b", b"a"]), None, 0);
        let t = t.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t.next(), &[NONE, NONE, 0]);
        assert_eq!(t.arena().len(), 3);
        assert!(JoinTable::index_bytes(3, 3, false) > 0);
        assert!(JoinTable::index_bytes(3, 0, true) > 0);
        assert_eq!(t.flat_index(), None);
        assert_eq!(t.arena().weight(0), 1);
        assert!(t.arena().payload(0).is_empty());
    }

    /// Row-layout arenas (several chunks concatenated, spill-shaped
    /// columns) give the same heads and chains with the flat and the
    /// salted index, and with arrow-row keys.
    #[test]
    fn row_layout_tables_agree_with_arrow_rows_for_both_index_kinds() {
        use super::super::keys::{Hashing, KeyCodec};
        use arrow_array::{ArrayRef, Int64Array, StringArray};

        let n = 3000usize;
        let ints: ArrayRef = Arc::new(Int64Array::from_iter(
            (0..n).map(|i| (i % 7 != 3).then_some((i % 500) as i64)),
        ));
        let strs: ArrayRef = Arc::new(StringArray::from_iter(
            (0..n).map(|i| Some(format!("{}-a-long-enough-string", i % 3))),
        ));
        let cols = [ints, strs];
        let types = [DataType::Int64, DataType::Utf8];
        let codec =
            KeyCodec::with_string_rows(&types, vec![false, true]).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            KeyCodec::new(&types, vec![false, true])
                .unwrap_or_else(|e| panic!("{e}"))
                .layout()
                .is_none()
        );
        let layout = codec.layout().cloned();
        let tables = |rows: bool, choice: IndexChoice| {
            let hashing = Hashing::new(Some(3), rows);
            let format = match layout.as_deref().filter(|_| rows) {
                Some(keys) => ArenaFormat::Rows {
                    keys,
                    payload: None,
                    arrow_payload: false,
                },
                None => ArenaFormat::Arrow { payload: false },
            };
            let schema = arena_schema(format).unwrap_or_else(|e| panic!("{e}"));
            let keys = codec
                .encode(&cols, &hashing)
                .unwrap_or_else(|e| panic!("{e}"));
            let valid: Vec<usize> = (0..n).filter(|&i| keys.is_valid(i)).collect();
            let chunk = |idx: &[usize]| {
                let mut c: Vec<arrow_array::ArrayRef> = vec![Arc::new(
                    UInt64Array::from_iter_values(idx.iter().map(|&i| keys.hashes()[i])),
                )];
                let l = layout.as_deref().filter(|_| rows);
                c.extend(keys.arena_columns(l, idx).unwrap_or_else(|e| panic!("{e}")));
                Batch::new(
                    RecordBatch::try_new(Arc::clone(&schema), c).unwrap_or_else(|e| panic!("{e}")),
                )
            };
            let (a, b) = valid.split_at(valid.len() / 3);
            let arena =
                RowArena::concat(&schema, &[chunk(a), chunk(b)]).unwrap_or_else(|e| panic!("{e}"));
            let l = layout.as_ref().filter(|_| rows);
            let t = JoinTable::build_with(arena, l, hashing.g_seed(), choice, None)
                .unwrap_or_else(|e| panic!("{e}"));
            let probe: Vec<u32> = (0..u32::try_from(n).unwrap_or(0))
                .filter(|&i| keys.is_valid(i as usize))
                .collect();
            let mut heads = Vec::new();
            t.find_heads(&keys, &probe, &mut heads);
            // A probe in the other format finds nothing.
            let other = codec
                .encode(&cols, &Hashing::new(Some(3), !rows))
                .unwrap_or_else(|e| panic!("{e}"));
            let mut none = Vec::new();
            t.find_heads(&other, &probe[..3], &mut none);
            assert_eq!(none, [NONE; 3]);
            (heads, t.next().to_vec(), t.flat_index())
        };
        let (arrow_heads, arrow_next, kind) = tables(false, IndexChoice::Adaptive);
        assert_eq!(kind, None);
        assert!(arrow_heads.iter().all(|&h| h != NONE));
        for choice in [IndexChoice::Flat, IndexChoice::Salted] {
            let (heads, next, kind) = tables(true, choice);
            assert_eq!(kind, Some(choice == IndexChoice::Flat));
            assert_eq!(heads, arrow_heads, "{choice:?}");
            assert_eq!(next, arrow_next, "{choice:?}");
        }
    }

    #[test]
    fn malformed_arenas_are_internal_errors() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hash", DataType::UInt64, false),
            Field::new("key", DataType::Utf8, false),
        ]));
        let data = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(UInt64Array::from(vec![1u64])),
                Arc::new(arrow_array::StringArray::from(vec!["x"])),
            ],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(RowArena::concat(&schema, &[Batch::new(data)]).is_err());
        let rows = arena(vec![1], vec![b"a"]);
        assert_eq!(RowSource::row(&rows, 0), (&[][..], &[][..]));
        assert_eq!(RowSource::hash(&rows, 9), 0);
        let t = JoinTable::build(rows, None, 0).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t.flat_index(), None);
    }
}
