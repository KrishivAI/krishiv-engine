// SPDX-License-Identifier: Apache-2.0
//! The batch-local key map (E2 steps 2 and 3,
//! `design/P2b-E2-agg-batch-protocol.md`): a batch with few distinct keys
//! is grouped from its columns, before any row is encoded. Rows of one key
//! share a batch-local id, only each id's first row is encoded and looked
//! up in the table (in first-occurrence order), and every row then takes its
//! id's table row. The grouping uses the stored hash `h`
//! ([`RowLayout::hash_only`], equal to the encoded one) and the table's key
//! equality ([`RowLayout::key_eq`]), so a row gets the row a per-row probe
//! would give it, except that a fixed index that clears between two rows of
//! one key no longer splits that key in two (one partial duplicate less,
//! merged anyway).

use std::sync::atomic::Ordering;

use arrow_array::{ArrayRef, UInt32Array};
use arrow_select::take::take;
use hashbrown::HashTable;

use super::{Mode, ProbeCount, RowTable, fin};
use crate::error::{Result, internal};
use crate::rows::{EncodedKeys, RowLayout};

/// The most distinct keys a batch may hold for the batch-local key map:
/// beyond it the batch takes the per-row probe.
pub(super) const LOCAL_MAX: usize = 256;

/// Batches the per-row probe runs for after a batch exceeded [`LOCAL_MAX`],
/// before the local map is tried again.
pub(super) const LOCAL_BACKOFF: u8 = 16;

/// A batch grouped by key.
pub(super) struct Local {
    /// The first row of each id, encoded.
    pub(super) keys: EncodedKeys,
    /// The id of every row of the batch.
    pub(super) ids: Vec<u32>,
    /// The stored hash of every row of the batch.
    pub(super) hashes: Vec<u64>,
}

impl RowTable {
    /// Groups the key columns `cols` of a batch, or `None` when the batch is
    /// too small, the table is not in a mode that probes, the backoff is
    /// running, or the batch holds more than [`LOCAL_MAX`] distinct keys
    /// (after which the next [`LOCAL_BACKOFF`] batches skip the attempt).
    pub(super) fn group_local(
        &self,
        layout: &RowLayout,
        cols: &[ArrayRef],
    ) -> Result<Option<Local>> {
        let n = cols.first().map_or(0, |c| c.len());
        if n < 2 * LOCAL_MAX || matches!(self.next_mode(), Mode::Append) {
            return Ok(None);
        }
        let skip = self.local_skip.load(Ordering::Relaxed);
        if skip > 0 {
            self.local_skip.store(skip - 1, Ordering::Relaxed);
            return Ok(None);
        }
        let hashes = layout.hash_only(cols)?;
        let same = layout.key_eq(cols)?;
        let gseed = self.gseed;
        // Hashed by the seeded `g`, so crafted keys cannot pile up in it.
        let mut map: HashTable<u32> = HashTable::with_capacity(2 * LOCAL_MAX);
        let mut firsts: Vec<u32> = Vec::with_capacity(LOCAL_MAX);
        let mut ids: Vec<u32> = Vec::with_capacity(n);
        for (k, &h) in hashes.iter().enumerate() {
            let g = fin(h, gseed);
            // A comparison error splits a key (never merges two), and the
            // encode below reports the same error.
            let found = map
                .find(g, |&j| {
                    same.eq(firsts[j as usize] as usize, k).unwrap_or(false)
                })
                .copied();
            let id = if let Some(j) = found {
                j
            } else {
                if firsts.len() == LOCAL_MAX {
                    self.local_skip.store(LOCAL_BACKOFF, Ordering::Relaxed);
                    return Ok(None);
                }
                let j = u32::try_from(firsts.len()).map_err(|e| internal(e.to_string()))?;
                let at = u32::try_from(k).map_err(|e| internal(e.to_string()))?;
                firsts.push(at);
                map.insert_unique(g, j, |&j| fin(hashes[firsts[j as usize] as usize], gseed));
                j
            };
            ids.push(id);
        }
        let idx = UInt32Array::from(firsts);
        let first_cols = cols
            .iter()
            .map(|c| take(c.as_ref(), &idx, None).map_err(|e| internal(e.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let keys = layout.encode(&first_cols)?;
        Ok(Some(Local { keys, ids, hashes }))
    }

    /// The table row of each row of a grouped batch, inserting new groups:
    /// each id's first row is looked up (or pushed, in append mode), every
    /// row takes its id's row.
    pub(super) fn probe_local(
        &mut self,
        keys: &EncodedKeys,
        ids: &[u32],
        refs: &mut Vec<u32>,
    ) -> Result<()> {
        let limit = match self.mode {
            Mode::Fixed { limit } => limit,
            _ => usize::MAX,
        };
        let (heap, hashes) = (keys.heap(), keys.hashes());
        let gseed = self.gseed;
        let mut count = ProbeCount::default();
        let mut found = Vec::with_capacity(keys.len());
        for (k, &h) in hashes.iter().enumerate() {
            let row = keys.row(k);
            let hg = (h, fin(h, gseed));
            found.push(
                match (self.mode, self.views.is_some(), self.probes.is_some()) {
                    (Mode::Append, ..) => {
                        self.unique = false;
                        self.push(row, heap, h)?
                    }
                    (_, true, false) => {
                        self.find_or_insert::<true, false>(row, heap, hg, limit, &mut count)?
                    }
                    (_, false, false) => {
                        self.find_or_insert::<false, false>(row, heap, hg, limit, &mut count)?
                    }
                    (_, true, true) => {
                        self.find_or_insert::<true, true>(row, heap, hg, limit, &mut count)?
                    }
                    (_, false, true) => {
                        self.find_or_insert::<false, true>(row, heap, hg, limit, &mut count)?
                    }
                },
            );
        }
        if let Some(totals) = &self.probes
            && count.lookups > 0
        {
            totals.add(&count);
        }
        refs.extend(ids.iter().map(|&j| found[j as usize]));
        Ok(())
    }
}
