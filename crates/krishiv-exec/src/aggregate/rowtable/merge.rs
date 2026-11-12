// SPDX-License-Identifier: Apache-2.0
//! Merging one row table into another: chunks, their projected size and the salted index.

use super::*;

impl RowTable {
    /// Rows `rows` of partition `p` of this table, with their states, to
    /// merge into another table of the same aggregate.
    pub(crate) fn merge_chunk(&self, p: usize, rows: Range<usize>) -> Result<MergeChunk> {
        let refs = self.part_ids(p, rows.clone());
        let part = self
            .parts
            .get(p)
            .ok_or_else(|| internal(format!("radix partition {p} out of range")))?;
        if rows.end > self.part_len(p) {
            return Err(internal(format!(
                "rows {rows:?} of partition {p} out of range"
            )));
        }
        let ids: Vec<u32> = refs.iter().map(|&r| self.refs.row(r)).collect();
        let mut states = Vec::new();
        for a in &part.accs {
            states.extend(a.state(&ids)?);
        }
        let heap = match &self.views {
            Some(l) => refs
                .iter()
                .map(|&r| l.long_bytes(row_at(&self.parts, self.stride, self.refs, r)))
                .sum(),
            None => 0,
        };
        Ok(MergeChunk {
            part: p,
            refs,
            states,
            heap,
        })
    }

    /// [`memory_size`](Self::memory_size) after
    /// [`apply_merge`](Self::apply_merge) of `c` from `src`, plus the
    /// merge's scratch (an upper bound for the merge to reserve).
    /// The partition of this (merge) table that chunk `c` lands in: its
    /// own partition, or the only one of a table with one partition (which
    /// merges any input, whatever its width).
    pub(super) fn merge_part(&self, c: &MergeChunk) -> usize {
        if self.bits == 0 { 0 } else { c.part }
    }

    pub(crate) fn projected_merge(&self, c: &MergeChunk) -> usize {
        let n = c.refs.len();
        let part = self.merge_part(c);
        let rows: usize = self
            .parts
            .iter()
            .enumerate()
            .map(|(p, v)| {
                if p == part {
                    grown(&v.rows, n * self.stride) + grown(&v.heap, c.heap)
                } else {
                    v.rows.capacity() + v.heap.capacity()
                }
            })
            .sum();
        let mut new = vec![0; self.parts.len()];
        if let Some(x) = new.get_mut(part) {
            *x = n;
        }
        let k = self.core.spec.group_by.len();
        rows + self.index_bytes(self.mode, n)
            + self.projected_accs(&new, &c.states, Fold::States { offset: k })
            + Self::scratch(n)
            + self.sketch_bytes()
    }

    /// Merges a chunk of `src`'s rows (rows are copied, states merged).
    /// Call after reserving [`projected_merge`](Self::projected_merge). Both
    /// tables have the same radix bits, so the rows stay in partition
    /// `c.part`.
    pub(crate) fn apply_merge(&mut self, src: &Self, c: &MergeChunk) -> Result<()> {
        if self.mode != Mode::Grow {
            return Err(internal(
                "merging into a table that is not growing".to_owned(),
            ));
        }
        if self.bits != 0 && self.bits != src.bits {
            return Err(internal(
                "merging tables of different radix bits".to_owned(),
            ));
        }
        let part = self.merge_part(c);
        let mut refs = Vec::with_capacity(c.refs.len());
        self.reserve_index(c.refs.len());
        self.reserve_merge_index(c.refs.len())?;
        if let Some(v) = self.parts.get_mut(part) {
            v.rows.reserve(c.refs.len() * self.stride);
            v.heap.reserve(c.heap);
        }
        let mut result = Ok(());
        let mut count = ProbeCount::default();
        let counted = self.probes.is_some();
        if self.merge_index.is_some() {
            result = self.merge_rows_salted(src, c, &mut refs, &mut count);
        } else {
            for &r in &c.refs {
                let (row, heap) = row_heap_at(&src.parts, src.stride, src.refs, r);
                let h = word(row, 0);
                let h = (h, fin(h, self.gseed));
                let found = match (self.views.is_some(), counted) {
                    (true, false) => {
                        self.find_or_insert::<true, false>(row, heap, h, usize::MAX, &mut count)
                    }
                    (false, false) => {
                        self.find_or_insert::<false, false>(row, heap, h, usize::MAX, &mut count)
                    }
                    (true, true) => {
                        self.find_or_insert::<true, true>(row, heap, h, usize::MAX, &mut count)
                    }
                    (false, true) => {
                        self.find_or_insert::<false, true>(row, heap, h, usize::MAX, &mut count)
                    }
                };
                match found {
                    Ok(id) => refs.push(id),
                    Err(e) => {
                        result = Err(e);
                        break;
                    }
                }
            }
        }
        if let Some(totals) = &self.probes
            && count.lookups > 0
        {
            totals.add(&count);
        }
        let mut counts = vec![0; self.parts.len()];
        if let Some(x) = counts.get_mut(part) {
            *x = refs.len();
        }
        let k = self.core.spec.group_by.len();
        result.and_then(|()| {
            self.fold(
                &refs,
                &counts,
                &c.states,
                &Weights::Const(1),
                Fold::States { offset: k },
            )
        })
    }

    /// Refuses what would look keys up in the empty `hashbrown` index of a
    /// table holding a [`MergeIndex`], and so duplicate its groups: such a
    /// table only merges.
    pub(super) fn only_hashbrown(&self) -> Result<()> {
        if self.merge_index.is_some() {
            return Err(internal("a merge table only takes merges".to_owned()));
        }
        Ok(())
    }

    /// Grows a [`MergeIndex`] for `new` more keys, as
    /// [`index_bytes`](Self::index_bytes) assumed.
    pub(super) fn reserve_merge_index(&mut self, new: usize) -> Result<()> {
        let (parts, stride, gseed, refs) = (&self.parts, self.stride, self.gseed, self.refs);
        if let Some(index) = &mut self.merge_index
            && index.reserve(new, |r| fin(word(row_at(parts, stride, refs, r), 0), gseed))?
            && let Some(c) = &self.resizes
        {
            c.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// The lookups of [`apply_merge`](Self::apply_merge) over a
    /// [`MergeIndex`]: each [`PREFETCH_ROWS`] rows' slots, then their first
    /// salt-matching rows, are prefetched before they are resolved in order.
    /// New keys get rows in input order, as with `hashbrown`.
    pub(super) fn merge_rows_salted(
        &mut self,
        src: &Self,
        c: &MergeChunk,
        out: &mut Vec<u32>,
        count: &mut ProbeCount,
    ) -> Result<()> {
        let counted = self.probes.is_some();
        let Self {
            parts,
            sketches,
            groups,
            merge_index,
            stride,
            gseed,
            bits,
            refs,
            views,
            ..
        } = self;
        let Some(index) = merge_index else {
            return Err(internal("no merge index".to_owned()));
        };
        let (stride, gseed, bits, refs) = (*stride, *gseed, *bits, *refs);
        let mut gs = [0_u64; PREFETCH_ROWS];
        for rows in c.refs.chunks(PREFETCH_ROWS) {
            for (g, &r) in gs.iter_mut().zip(rows) {
                *g = fin(word(row_at(&src.parts, src.stride, src.refs, r), 0), gseed);
                index.prefetch_slot(*g);
            }
            let gs = &gs[..rows.len()];
            for &g in gs {
                if let Slot::Candidate(_, r) = index.walk(g, index.start(g)) {
                    prefetch_read(row_at(parts, stride, refs, r), 0);
                }
            }
            for (&from, &g) in rows.iter().zip(gs) {
                let (key, key_heap) = row_heap_at(&src.parts, src.stride, src.refs, from);
                let mut at = index.start(g);
                let mut compares = 0_u64;
                let id = loop {
                    match index.walk(g, at) {
                        Slot::Empty(free) => {
                            let shape = (stride, bits, refs, views.as_ref());
                            let h = word(key, 0);
                            let r = push_row(parts, sketches, groups, shape, (key, key_heap), h)?;
                            index.insert(free, g, r);
                            break r;
                        }
                        Slot::Candidate(found, r) => {
                            if counted {
                                compares += 1;
                            }
                            let same = match views {
                                Some(l) => {
                                    let (row, heap) = row_heap_at(parts, stride, refs, r);
                                    !row.is_empty() && l.eq(row, heap, key, key_heap)
                                }
                                None => same_key(row_at(parts, stride, refs, r), key, stride),
                            };
                            if same {
                                break r;
                            }
                            at = index.after(found);
                        }
                    }
                };
                if counted {
                    count.add(compares);
                }
                out.push(id);
            }
        }
        Ok(())
    }
}
