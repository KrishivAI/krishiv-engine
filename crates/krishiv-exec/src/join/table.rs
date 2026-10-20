// SPDX-License-Identifier: Apache-2.0
//! The join hash table: rows of one partition in a row arena, indexed by
//! the shared [`KeyTable`] (E02: `hashbrown::HashTable` over a key arena).
//!
//! Build rows live in a **row arena**: Arrow batches of `hash: UInt64`,
//! `key: LargeBinary` (arrow-row key bytes) and, when the join needs build
//! columns, `payload: LargeBinary` (arrow-row bytes of the whole build row),
//! with the build weights. The same batches are what a spilled partition
//! writes, so spilling and reloading never re-encode. Each distinct key
//! points to its newest row; older rows with the same key are chained
//! through [`JoinTable::next`], so duplicate keys never lengthen a
//! probe sequence.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{LargeBinaryArray, UInt64Array};
use arrow_row::Rows;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_types::batch::{Batch, Weights};

use crate::error::{Result, internal};
use crate::hash::{KeyTable, ProbeCount};

/// End of a chain / no match.
pub(crate) const NONE: u32 = u32::MAX;

/// The schema of the row arena's batches.
pub(crate) fn arena_schema(payload: bool) -> SchemaRef {
    let mut fields = vec![
        Field::new("hash", DataType::UInt64, false),
        Field::new("key", DataType::LargeBinary, false),
    ];
    if payload {
        fields.push(Field::new("payload", DataType::LargeBinary, false));
    }
    Arc::new(Schema::new(fields))
}

/// Bytes of the key rows in arena chunks.
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

/// One partition's build rows, coalesced into a single batch.
#[derive(Debug)]
pub(crate) struct RowArena {
    batch: Batch,
    hashes: UInt64Array,
    keys: LargeBinaryArray,
    payload: Option<LargeBinaryArray>,
}

impl RowArena {
    /// Concatenates arena chunks (same schema).
    pub(crate) fn concat(schema: &SchemaRef, chunks: &[Batch]) -> Result<Self> {
        let batch = match chunks {
            [one] => one.clone(),
            _ => Batch::concat(schema, chunks).map_err(|e| internal(e.to_string()))?,
        };
        let column = |i: usize| batch.data().columns().get(i).cloned();
        let bad = || internal("malformed join row arena".to_owned());
        let hashes = column(0).ok_or_else(bad)?;
        let hashes = hashes
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(bad)?
            .clone();
        let keys = column(1).ok_or_else(bad)?;
        let keys = keys.as_binary_opt::<i64>().ok_or_else(bad)?.clone();
        let payload = match column(2) {
            Some(p) => Some(p.as_binary_opt::<i64>().ok_or_else(bad)?.clone()),
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
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.batch.num_rows()
    }

    pub(crate) fn hash(&self, row: u32) -> u64 {
        self.hashes.value(row as usize)
    }

    /// Bytes of all key rows.
    pub(crate) fn key_bytes(&self) -> usize {
        self.keys.values().len()
    }

    pub(crate) fn key(&self, row: u32) -> &[u8] {
        self.keys.value(row as usize)
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

/// An immutable join table over one partition's rows: the shared
/// [`KeyTable`] (hashbrown `HashTable<u32>`, E02 decision B) maps each
/// distinct key to its newest row, and older rows with the same key are
/// chained through [`next`](Self::next).
#[derive(Debug)]
pub(crate) struct JoinTable {
    arena: RowArena,
    keys: KeyTable,
    heads: Vec<u32>,
    next: Vec<u32>,
}

impl JoinTable {
    /// Bytes the index over `rows` rows with `key_bytes` bytes of keys
    /// takes (reserved before [`build`](Self::build)): the key table at
    /// worst one key per row, a chain head per key and a link per row.
    pub(crate) fn index_bytes(rows: usize, key_bytes: usize) -> usize {
        KeyTable::new().projected_size(rows, key_bytes) + rows * 2 * size_of::<u32>()
    }

    /// Indexes `arena`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the index cannot be allocated.
    pub(crate) fn build(arena: RowArena) -> Result<Self> {
        let n = arena.len();
        let mut keys = KeyTable::new();
        keys.reserve(n, arena.key_bytes());
        let mut heads: Vec<u32> = Vec::new();
        let mut next: Vec<u32> = Vec::new();
        let oom = |e: std::collections::TryReserveError| internal(e.to_string());
        heads.try_reserve_exact(n).map_err(oom)?;
        next.try_reserve_exact(n).map_err(oom)?;
        next.resize(n, NONE);
        for row in 0..u32::try_from(n).unwrap_or(0) {
            let (id, new) = keys.insert_hashed(arena.key(row), arena.hash(row))?;
            if new {
                heads.push(row);
            } else if let Some(head) = heads.get_mut(id as usize) {
                // Newest first: the older head follows this row.
                next[row as usize] = *head;
                *head = row;
            }
        }
        Ok(Self {
            arena,
            keys,
            heads,
            next,
        })
    }

    /// For each probe row `rows[j]` (with key bytes in `keys` and hash in
    /// `hashes`), the newest build row with an equal key, or [`NONE`].
    pub(crate) fn find_heads(
        &self,
        keys: &Rows,
        hashes: &[u64],
        rows: &[u32],
        heads: &mut Vec<u32>,
    ) {
        heads.extend(rows.iter().map(|&i| {
            let key = keys.row(i as usize);
            let id = self.keys.find(key.as_ref(), hashes[i as usize]);
            id.and_then(|id| self.heads.get(id as usize).copied())
                .unwrap_or(NONE)
        }));
    }

    /// As [`find_heads`](Self::find_heads), counting probe lengths.
    pub(crate) fn find_heads_counted(
        &self,
        keys: &Rows,
        hashes: &[u64],
        rows: &[u32],
        heads: &mut Vec<u32>,
        count: &mut ProbeCount,
    ) {
        heads.extend(rows.iter().map(|&i| {
            let key = keys.row(i as usize);
            let id = self
                .keys
                .find_counted(key.as_ref(), hashes[i as usize], count);
            id.and_then(|id| self.heads.get(id as usize).copied())
                .unwrap_or(NONE)
        }));
    }

    /// Distinct keys and buckets of the index.
    pub(crate) fn index_size(&self) -> (usize, usize) {
        (self.keys.len(), self.keys.buckets())
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
        let schema = arena_schema(false);
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
        let t = JoinTable::build(arena(vec![7, 7, 7], vec![b"a", b"b", b"a"]));
        let t = t.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t.next(), &[NONE, NONE, 0]);
        assert_eq!(t.arena().len(), 3);
        assert!(JoinTable::index_bytes(3, 3) > 0);
        assert_eq!(t.arena().weight(0), 1);
        assert!(t.arena().payload(0).is_empty());
    }
}
