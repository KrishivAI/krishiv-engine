// SPDX-License-Identifier: Apache-2.0
//! The two indexes over [`RowPages`]: [`SaltedTable`] (P2-T157 §5.1) and
//! [`HashbrownTable`], behind one [`RowIndex`] interface so they can be
//! measured on the same rows.

use std::hint::black_box;

use hashbrown::HashTable;
use hashbrown::hash_table::Entry;

use super::layout::{EncodedKeys, RowLayout};
use super::pages::RowPages;
use crate::error::{Result, internal};

/// "No row" in a lookup result.
pub const NO_ROW: u32 = u32::MAX;

/// Rows per lookup chunk: the salted table touches a chunk's slots, then
/// its candidate rows, then resolves it, so a chunk's misses overlap.
const CHUNK: usize = 256;
/// Below this many slots (2 MiB) the table stays in the L2/L3 caches and
/// the touch passes only add work (measured; Polars starts prefetching at
/// 2^15 hashbrown buckets, which are 4 bytes each).
const TOUCH_MIN_SLOTS: usize = 1 << 18;
/// Fewest slots in a salted table.
const MIN_SLOTS: usize = 1 << 10;
/// Salt bits: the top 16 bits of the hash, kept in the slot.
const SALT_MASK: u64 = 0xFFFF << 48;
/// The `row id + 1` part of a slot (0 = empty).
const ID_MASK: u64 = u32::MAX as u64;

/// A key index over [`RowPages`].
pub trait RowIndex: std::fmt::Debug {
    /// The row id of each key of `keys`, appending a row to `rows` for each
    /// key not seen before; returns how many keys were new.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` once row ids are exhausted.
    fn find_or_insert(
        &mut self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        rows: &mut RowPages,
        ids: &mut Vec<u32>,
    ) -> Result<usize>;

    /// The row id of each key of `keys`, or [`NO_ROW`].
    fn find(&self, layout: &RowLayout, keys: &EncodedKeys, rows: &RowPages, out: &mut Vec<u32>);

    /// Number of keys indexed.
    fn len(&self) -> usize;

    /// Whether no key is indexed.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes held by the index (not the rows).
    fn memory_size(&self) -> usize;
}

/// Power-of-two `u64` slots of `salt:16 | unused:16 | row id + 1:32`,
/// probed linearly from the low bits of the hash (DuckDB's `ht_entry_t`,
/// with row ids instead of pointers).
#[derive(Debug)]
pub struct SaltedTable {
    slots: Vec<u64>,
    len: usize,
    /// Most keys per 8 slots (4 = load 1/2, a join; 5 ≈ 2/3, an aggregate).
    load_eighths: usize,
}

impl SaltedTable {
    /// An empty table that grows past `load_eighths / 8` full (4 for a
    /// join build, 5 for an aggregate; clamped to 1..=7).
    pub fn new(load_eighths: usize) -> Self {
        Self {
            slots: vec![0; MIN_SLOTS],
            len: 0,
            load_eighths: load_eighths.clamp(1, 7),
        }
    }

    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// Grows (re-inserting from the rows' stored hashes) so `extra` more
    /// keys fit under the load limit.
    fn reserve(&mut self, extra: usize, rows: &RowPages) {
        let need = self.len + extra;
        let mut n = self.slots.len();
        while need * 8 > n * self.load_eighths {
            n *= 2;
        }
        if n == self.slots.len() {
            return;
        }
        let old = std::mem::replace(&mut self.slots, vec![0; n]);
        let mask = n - 1;
        for slot in old.into_iter().filter(|&s| s != 0) {
            let id = slot_id(slot);
            let mut i = index(rows.hash(id), mask);
            while self.slots[i] != 0 {
                i = (i + 1) & mask;
            }
            self.slots[i] = slot;
        }
    }

    /// Touches the slots, then the first salt-matching row, of each key of a
    /// chunk, so their cache misses are in flight together before the
    /// chunk is resolved in order.
    #[inline]
    fn touch(&self, hashes: &[u64], rows: &RowPages) {
        if self.slots.len() < TOUCH_MIN_SLOTS {
            return;
        }
        let mask = self.mask();
        for &h in hashes {
            black_box(self.slots[index(h, mask)]);
        }
        for &h in hashes {
            let salt = h & SALT_MASK;
            let mut i = index(h, mask);
            loop {
                let s = self.slots[i];
                if s == 0 {
                    break;
                }
                if s & SALT_MASK == salt {
                    black_box(rows.row(slot_id(s)).0.first());
                    break;
                }
                i = (i + 1) & mask;
            }
        }
    }

    /// The slot index holding key `k` of `keys`, or the empty slot where
    /// it belongs.
    #[inline]
    fn probe(
        &self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        k: usize,
        rows: &RowPages,
    ) -> (usize, Option<u32>) {
        let h = keys.hashes()[k];
        let salt = h & SALT_MASK;
        let mask = self.mask();
        let key = keys.row(k);
        let mut i = index(h, mask);
        loop {
            let s = self.slots[i];
            if s == 0 {
                return (i, None);
            }
            if s & SALT_MASK == salt {
                let id = slot_id(s);
                let (row, heap) = rows.row(id);
                if layout.eq(key, keys.heap(), row, heap) {
                    return (i, Some(id));
                }
            }
            i = (i + 1) & mask;
        }
    }
}

impl RowIndex for SaltedTable {
    fn find_or_insert(
        &mut self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        rows: &mut RowPages,
        ids: &mut Vec<u32>,
    ) -> Result<usize> {
        ids.clear();
        let before = self.len;
        // Worst case every key is new: grow once, so no chunk sees a resize.
        self.reserve(keys.len(), rows);
        for start in (0..keys.len()).step_by(CHUNK) {
            let end = (start + CHUNK).min(keys.len());
            self.touch(&keys.hashes()[start..end], rows);
            for k in start..end {
                let (slot, found) = self.probe(layout, keys, k, rows);
                let id = match found {
                    Some(id) => id,
                    None => {
                        let id = rows.push(keys, k)?;
                        if id == NO_ROW {
                            return Err(internal("row ids exhausted".to_owned()));
                        }
                        self.slots[slot] = (keys.hashes()[k] & SALT_MASK) | (u64::from(id) + 1);
                        self.len += 1;
                        id
                    }
                };
                ids.push(id);
            }
        }
        Ok(self.len - before)
    }

    fn find(&self, layout: &RowLayout, keys: &EncodedKeys, rows: &RowPages, out: &mut Vec<u32>) {
        out.clear();
        for start in (0..keys.len()).step_by(CHUNK) {
            let end = (start + CHUNK).min(keys.len());
            self.touch(&keys.hashes()[start..end], rows);
            out.extend((start..end).map(|k| self.probe(layout, keys, k, rows).1.unwrap_or(NO_ROW)));
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn memory_size(&self) -> usize {
        self.slots.capacity() * size_of::<u64>()
    }
}

/// `hashbrown::HashTable<u32>` of row ids over the same rows (the
/// alternative T157 measures the salted table against).
#[derive(Debug, Default)]
pub struct HashbrownTable {
    table: HashTable<u32>,
}

impl HashbrownTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }
}

impl RowIndex for HashbrownTable {
    fn find_or_insert(
        &mut self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        rows: &mut RowPages,
        ids: &mut Vec<u32>,
    ) -> Result<usize> {
        ids.clear();
        let before = self.table.len();
        {
            let rows = &*rows;
            self.table.reserve(keys.len(), |&id| rows.hash(id));
        }
        for (k, &h) in keys.hashes().iter().enumerate() {
            let key = keys.row(k);
            let entry = {
                let rows = &*rows;
                self.table.entry(
                    h,
                    |&id| {
                        let (row, heap) = rows.row(id);
                        layout.eq(key, keys.heap(), row, heap)
                    },
                    |&id| rows.hash(id),
                )
            };
            let id = match entry {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let id = rows.push(keys, k)?;
                    v.insert(id);
                    id
                }
            };
            ids.push(id);
        }
        Ok(self.table.len() - before)
    }

    fn find(&self, layout: &RowLayout, keys: &EncodedKeys, rows: &RowPages, out: &mut Vec<u32>) {
        out.clear();
        out.extend(keys.hashes().iter().enumerate().map(|(k, &h)| {
            let key = keys.row(k);
            self.table
                .find(h, |&id| {
                    let (row, heap) = rows.row(id);
                    layout.eq(key, keys.heap(), row, heap)
                })
                .copied()
                .unwrap_or(NO_ROW)
        }));
    }

    fn len(&self) -> usize {
        self.table.len()
    }

    fn memory_size(&self) -> usize {
        // A `u32` and a control byte per bucket.
        self.table.capacity().div_ceil(7) * 8 * (size_of::<u32>() + 1)
    }
}

#[inline]
fn index(hash: u64, mask: usize) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "only the low bits below the mask are used"
    )]
    let low = hash as usize;
    low & mask
}

#[inline]
fn slot_id(slot: u64) -> u32 {
    u32::try_from((slot & ID_MASK).saturating_sub(1)).unwrap_or(NO_ROW)
}
