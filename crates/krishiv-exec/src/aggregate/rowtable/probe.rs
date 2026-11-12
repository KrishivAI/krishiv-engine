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
