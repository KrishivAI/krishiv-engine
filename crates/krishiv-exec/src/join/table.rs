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

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, LargeBinaryArray, UInt64Array};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_types::batch::{Batch, Weights};

use super::keys::{Encoded, KeyRows};
use crate::error::{Result, internal};
use crate::hash::KeyTable;
use crate::rows::{IndexChoice, JoinIndex, RowLayout, RowSource};

/// End of a chain / no match.
pub(crate) const NONE: u32 = u32::MAX;
const _: () = assert!(NONE == crate::rows::NO_MATCH);

/// The schema of the row arena's batches: arrow-row keys, or the rows of
/// `layout`.
pub(crate) fn arena_schema(payload: bool, layout: Option<&RowLayout>) -> Result<SchemaRef> {
    let mut fields = vec![Field::new("hash", DataType::UInt64, false)];
    match layout {
        None => fields.push(Field::new("key", DataType::LargeBinary, false)),
        Some(l) => {
            let width = i32::try_from(l.stride())
                .map_err(|_| internal("join key row too wide".to_owned()))?;
            fields.push(Field::new("key", DataType::FixedSizeBinary(width), false));
            if l.has_views() {
                fields.push(Field::new("key_heap", DataType::LargeBinary, false));
            }
        }
    }
    if payload {
        fields.push(Field::new("payload", DataType::LargeBinary, false));
    }
    Ok(Arc::new(Schema::new(fields)))
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

/// An arena's keys.
#[derive(Debug)]
enum ArenaKeys {
    /// Arrow-row key bytes.
    Arrow(LargeBinaryArray),
    /// Row-layout rows of `stride` bytes back to back, and each row's long
    /// values when the layout has views.
    Rows {
        fixed: Buffer,
        stride: usize,
        heap: Option<LargeBinaryArray>,
    },
}

/// One partition's build rows, coalesced into a single batch.
#[derive(Debug)]
pub(crate) struct RowArena {
    batch: Batch,
    hashes: UInt64Array,
    keys: ArenaKeys,
    payload: Option<LargeBinaryArray>,
}

impl RowArena {
    /// Concatenates arena chunks (same schema).
    pub(crate) fn concat(schema: &SchemaRef, chunks: &[Batch]) -> Result<Self> {
        let batch = match chunks {
            [one] => one.clone(),
            _ => Batch::concat(schema, chunks).map_err(|e| internal(e.to_string()))?,
        };
        let named = |name: &str| batch.data().column_by_name(name).cloned();
        let bad = || internal("malformed join row arena".to_owned());
        let binary = |a: Option<arrow_array::ArrayRef>| match a {
            Some(a) => a.as_binary_opt::<i64>().cloned().ok_or_else(bad).map(Some),
            None => Ok(None),
        };
        let hashes = named("hash").ok_or_else(bad)?;
        let hashes = hashes
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(bad)?
            .clone();
        let key = named("key").ok_or_else(bad)?;
        let keys = match key.data_type() {
            DataType::LargeBinary => ArenaKeys::Arrow(binary(Some(key))?.ok_or_else(bad)?),
            DataType::FixedSizeBinary(_) => {
                let fixed = key.as_fixed_size_binary_opt().ok_or_else(bad)?;
                let stride = usize::try_from(fixed.value_length()).map_err(|_| bad())?;
                // `value_data` starts at the array's first value.
                let values = fixed.to_data().buffers().first().cloned().ok_or_else(bad)?;
                let start = fixed.to_data().offset().saturating_mul(stride);
                let len = fixed.len().saturating_mul(stride);
                if stride == 0 || start.saturating_add(len) > values.len() {
                    return Err(bad());
                }
                ArenaKeys::Rows {
                    fixed: values.slice_with_length(start, len),
                    stride,
                    heap: binary(named("key_heap"))?,
                }
            }
            _ => return Err(bad()),
        };
        let payload = binary(named("payload"))?;
        if u32::try_from(batch.num_rows()).is_err() {
            return Err(internal("a join partition has over 2^32 rows".to_owned()));
        }
        Ok(Self {
            batch,
            hashes,
            keys,
            payload,
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
            ArenaKeys::Arrow(k) => Some(k),
            ArenaKeys::Rows { .. } => None,
        }
    }

    /// The row's payload bytes (empty if the arena has none).
    pub(crate) fn payload(&self, row: u32) -> &[u8] {
        self.payload.as_ref().map_or(&[], |p| p.value(row as usize))
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
    fn row(&self, id: u32) -> (&[u8], &[u8]) {
        let ArenaKeys::Rows {
            fixed,
            stride,
            heap,
        } = &self.keys
        else {
            return (&[], &[]);
        };
        let at = (id as usize).saturating_mul(*stride);
        let row = fixed
            .get(at..at.saturating_add(*stride))
            .unwrap_or_default();
        let heap = heap
            .as_ref()
            .filter(|h| (id as usize) < h.len())
            .map_or(&[][..], |h| h.value(id as usize));
        (row, heap)
    }

    #[inline]
    fn hash(&self, id: u32) -> u64 {
        self.hashes.values().get(id as usize).copied().unwrap_or(0)
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
        Self::build_with(arena, layout, g_seed, IndexChoice::Adaptive)
    }

    /// [`build`](Self::build) with the row-layout index forced (tests).
    pub(crate) fn build_with(
        arena: RowArena,
        layout: Option<&Arc<RowLayout>>,
        g_seed: u64,
        choice: IndexChoice,
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
                for row in 0..u32::try_from(n).unwrap_or(0) {
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
                index: JoinIndex::build(layout, &arena, n, g_seed, choice, &mut next)?,
                layout: Arc::clone(layout),
            },
            (None, None) => return Err(internal("join key rows without a layout".to_owned())),
        };
        Ok(Self { arena, index, next })
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
        let schema = arena_schema(false, None).unwrap_or_else(|e| panic!("{e}"));
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
            let schema = arena_schema(false, layout.as_deref().filter(|_| rows))
                .unwrap_or_else(|e| panic!("{e}"));
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
            let t = JoinTable::build_with(arena, l, hashing.g_seed(), choice)
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
