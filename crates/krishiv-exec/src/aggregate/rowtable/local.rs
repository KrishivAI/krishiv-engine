// SPDX-License-Identifier: Apache-2.0
//! The batch-local key map of a row table (E2 step 2,
//! `design/P2b-E2-agg-batch-protocol.md`): a batch with few distinct keys
//! looks up only each key's first row.

use hashbrown::HashTable;

use super::{LOCAL_BACKOFF, LOCAL_MAX, RowTable, fin};
use crate::error::{Result, internal};
use crate::hash::ProbeCount;
use crate::rows::EncodedKeys;

impl RowTable {
    /// The rows of a batch with few distinct keys (E2 step 2,
    /// `design/P2b-E2-agg-batch-protocol.md`): rows whose encoded keys are
    /// byte-equal share a batch-local id, and only each id's first row is
    /// looked up in the table, in first-occurrence order; every row then takes
    /// its id's row. Byte equality is never coarser than the table's key
    /// equality, so a row gets the row a per-row probe would give it, except
    /// that a fixed index that clears between two rows of one key no longer
    /// splits that key in two (one partial duplicate less, merged anyway).
    /// Returns `false`, with `refs` untouched, for a batch with long (heap)
    /// key values or more than [`LOCAL_MAX`] distinct keys; after the latter
    /// the next [`LOCAL_BACKOFF`] batches skip the attempt.
    pub(super) fn probe_local(
        &mut self,
        keys: &EncodedKeys,
        refs: &mut Vec<u32>,
        limit: usize,
    ) -> Result<bool> {
        if self.local_skip > 0 {
            self.local_skip -= 1;
            return Ok(false);
        }
        if !keys.heap().is_empty() || keys.len() < 2 * LOCAL_MAX {
            return Ok(false);
        }
        let (hashes, gseed) = (keys.hashes(), self.gseed);
        // Hashed by the seeded `g`, so crafted keys cannot pile up in it.
        let mut map: HashTable<u32> = HashTable::with_capacity(2 * LOCAL_MAX);
        let mut firsts: Vec<usize> = Vec::with_capacity(LOCAL_MAX);
        let mut ids: Vec<u32> = Vec::with_capacity(keys.len());
        for (k, &h) in hashes.iter().enumerate() {
            let row = keys.row(k);
            let g = fin(h, gseed);
            let found = map
                .find(g, |&j| keys.row(firsts[j as usize]) == row)
                .copied();
            let id = match found {
                Some(j) => j,
                None => {
                    if firsts.len() == LOCAL_MAX {
                        self.local_skip = LOCAL_BACKOFF;
                        return Ok(false);
                    }
                    let j = u32::try_from(firsts.len()).map_err(|e| internal(e.to_string()))?;
                    firsts.push(k);
                    map.insert_unique(g, j, |&j| fin(hashes[firsts[j as usize]], gseed));
                    j
                }
            };
            ids.push(id);
        }
        let heap = keys.heap();
        let mut count = ProbeCount::default();
        let mut found = Vec::with_capacity(firsts.len());
        for &k in &firsts {
            let h = hashes[k];
            let hg = (h, fin(h, gseed));
            let row = keys.row(k);
            found.push(match (self.views.is_some(), self.probes.is_some()) {
                (true, false) => {
                    self.find_or_insert::<true, false>(row, heap, hg, limit, &mut count)?
                }
                (false, false) => {
                    self.find_or_insert::<false, false>(row, heap, hg, limit, &mut count)?
                }
                (true, true) => {
                    self.find_or_insert::<true, true>(row, heap, hg, limit, &mut count)?
                }
                (false, true) => {
                    self.find_or_insert::<false, true>(row, heap, hg, limit, &mut count)?
                }
            });
        }
        if let Some(totals) = &self.probes
            && count.lookups > 0
        {
            totals.add(&count);
        }
        refs.extend(ids.iter().map(|&j| found[j as usize]));
        Ok(true)
    }
}
