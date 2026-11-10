// SPDX-License-Identifier: Apache-2.0
//! Looking rows up: the local and all-partition probes and find-or-insert.

use super::*;

impl RowTable {
    /// The row of each row of `keys`, inserting new groups.
    pub(super) fn lookup(&mut self, keys: &EncodedKeys, refs: &mut Vec<u32>) -> Result<()> {
        let (hashes, heap) = (keys.hashes(), keys.heap());
        match self.mode {
            Mode::Append => {
                self.unique &= keys.is_empty();
                for (k, &h) in hashes.iter().enumerate() {
                    refs.push(self.push(keys.row(k), heap, h)?);
                }
            }
            Mode::Fixed { limit } => {
                if !self.probe_local(keys, refs, limit)? {
                    self.probe_all(keys, refs, limit)?;
                }
            }
            Mode::Grow => {
                if !self.probe_local(keys, refs, usize::MAX)? {
                    self.probe_all(keys, refs, usize::MAX)?;
                }
            }
        }
        Ok(())
    }

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

    /// [`find_or_insert`](Self::find_or_insert) for every row of `keys`, in
    /// one loop per key shape: the probe's compare of fixed-width keys has
    /// no view branch, so it stays small enough to inline.
    pub(super) fn probe_all(
        &mut self,
        keys: &EncodedKeys,
        refs: &mut Vec<u32>,
        limit: usize,
    ) -> Result<()> {
        let mut count = ProbeCount::default();
        match (self.views.is_some(), self.probes.is_some()) {
            (true, false) => self.probe_rows::<true, false>(keys, refs, limit, &mut count)?,
            (false, false) => self.probe_rows::<false, false>(keys, refs, limit, &mut count)?,
            (true, true) => self.probe_rows::<true, true>(keys, refs, limit, &mut count)?,
            (false, true) => self.probe_rows::<false, true>(keys, refs, limit, &mut count)?,
        }
        if let Some(totals) = &self.probes
            && count.lookups > 0
        {
            totals.add(&count);
        }
        Ok(())
    }

    /// The loop of [`probe_all`](Self::probe_all) for one key shape;
    /// `COUNT` adds each lookup's key comparisons to `count`.
    ///
    /// The index's control bytes are mostly cached, the rows it points to
    /// are not (TPC-DS q22: 29% of the probe in the key compare). So each
    /// [`PREFETCH_ROWS`] rows' first tag-matching rows are prefetched
    /// before the rows are resolved in order; a candidate that an insert or
    /// a clear in between made stale is only a wasted prefetch.
    pub(super) fn probe_rows<const VIEWS: bool, const COUNT: bool>(
        &mut self,
        keys: &EncodedKeys,
        refs: &mut Vec<u32>,
        limit: usize,
        count: &mut ProbeCount,
    ) -> Result<()> {
        let (hashes, heap) = (keys.hashes(), keys.heap());
        let mut gs = [0_u64; PREFETCH_ROWS];
        for (c, chunk) in hashes.chunks(PREFETCH_ROWS).enumerate() {
            for (g, &h) in gs.iter_mut().zip(chunk) {
                *g = fin(h, self.gseed);
            }
            let gs = &gs[..chunk.len()];
            if self.index.len() > PREFETCH_ROWS {
                for &g in gs {
                    if let Some(&r) = self.index.find(g, |_| true) {
                        prefetch_read(row_at(&self.parts, self.stride, self.refs, r), 0);
                    }
                }
            }
            for (i, (&h, &g)) in chunk.iter().zip(gs).enumerate() {
                let k = c * PREFETCH_ROWS + i;
                refs.push(self.find_or_insert::<VIEWS, COUNT>(
                    keys.row(k),
                    heap,
                    (h, g),
                    limit,
                    count,
                )?);
            }
        }
        Ok(())
    }

    /// The row of key row `key` (with hash `h` and index hash
    /// `g = fin(h, gseed)`; its long values in
    /// `key_heap`), inserting it if new; an index holding `limit` keys is
    /// cleared before it takes another (the rows stay: a key seen again
    /// gets a new row). `VIEWS`: the layout has string or binary views;
    /// `COUNT`: the lookup's key comparisons are added to `count`.
    #[inline(always)]
    pub(super) fn find_or_insert<const VIEWS: bool, const COUNT: bool>(
        &mut self,
        key: &[u8],
        key_heap: &[u8],
        (h, g): (u64, u64),
        limit: usize,
        count: &mut ProbeCount,
    ) -> Result<u32> {
        let Self {
            parts,
            sketches,
            groups,
            index,
            unique,
            abandons,
            stride,
            gseed,
            bits,
            refs,
            views,
            adapt,
            ..
        } = self;
        let (stride, gseed) = (*stride, *gseed);
        // A lookup first (hashbrown inlines `find`; most rows are hits),
        // then an insert on a miss.
        let mut compares = 0_u64;
        let found = index
            .find(g, |&r| {
                if COUNT {
                    compares += 1;
                }
                match views {
                    Some(l) if VIEWS => {
                        let (row, heap) = row_heap_at(parts, stride, *refs, r);
                        !row.is_empty() && l.eq(row, heap, key, key_heap)
                    }
                    _ => same_key(row_at(parts, stride, *refs, r), key, stride),
                }
            })
            .copied();
        if COUNT {
            count.add(compares);
        }
        if let Some(r) = found {
            return Ok(r);
        }
        let shape = (stride, *bits, *refs, views.as_ref());
        let r = push_row(parts, sketches, groups, shape, (key, key_heap), h)?;
        if index.len() >= limit {
            index.clear();
            *unique = false;
            if let Some(a) = adapt {
                a.filled = true;
            }
            if let Some(c) = abandons {
                c.fetch_add(1, Ordering::Relaxed);
            }
        }
        let refs = *refs;
        index.insert_unique(g, r, |&o| {
            fin(word(row_at(parts, stride, refs, o), 0), gseed)
        });
        Ok(r)
    }

    pub(super) fn push(&mut self, key: &[u8], key_heap: &[u8], h: u64) -> Result<u32> {
        push_row(
            &mut self.parts,
            &mut self.sketches,
            &mut self.groups,
            (self.stride, self.bits, self.refs, self.views.as_ref()),
            (key, key_heap),
            h,
        )
    }
}
