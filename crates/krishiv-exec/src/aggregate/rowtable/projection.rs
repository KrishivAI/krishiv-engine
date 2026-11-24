// SPDX-License-Identifier: Apache-2.0
//! The memory a batch will add to a [`RowTable`], per partition: the accumulators' bytes (sketch aggregates read the
//! batch's sketch columns) and the radix partition of each input row.

use super::*;

impl RowTable {
    /// The accumulators' bytes after `new[p]` more groups in each partition
    /// `p` and a fold of `cols` (`fold` says which columns each aggregate
    /// reads; a string `min`/`max` may keep all their bytes, once).
    pub(super) fn projected_accs(
        &self,
        new: &[usize],
        cols: &[ArrayRef],
        fold: Fold,
        row_part: Option<&[u8]>,
    ) -> usize {
        let states = matches!(fold, Fold::States { .. });
        // The sketch bytes of each partition's own rows, per argument column (when the rows' partitions are known).
        let mut by_part: HashMap<usize, Vec<usize>> = HashMap::new();
        if let Some(rp) = row_part {
            for (a, r) in self
                .parts
                .first()
                .iter()
                .flat_map(|p| p.accs.iter().zip(&self.core.ranges))
            {
                if !a.theta_reads_sketches(states) {
                    continue;
                }
                let at = match fold {
                    Fold::States { offset } => r.start.checked_sub(offset),
                    Fold::Raw => a.args().first().copied(),
                };
                if let Some(i) = at
                    && let Some(c) = cols.get(i)
                    && !by_part.contains_key(&i)
                {
                    let mut v = vec![0_usize; PARTS];
                    for (row, &q) in rp.iter().enumerate() {
                        v[usize::from(q)] += sketch_row_bytes(c.as_ref(), row);
                    }
                    by_part.insert(i, v);
                }
            }
        }
        let parts: usize = self
            .parts
            .iter()
            .zip(new.iter().copied().chain(std::iter::repeat(0)))
            .enumerate()
            .map(|(pi, (part, n))| {
                part.accs
                    .iter()
                    .zip(&self.core.ranges)
                    .map(|(a, r)| {
                        // A partition with no row reads none of the column; the others each take it whole
                        // (a bound: the fold of a batch adds at most its bytes times a constant in all).
                        let state_bytes = if n > 0 && a.theta_reads_sketches(states) {
                            let at = match fold {
                                Fold::States { offset } => r.start.checked_sub(offset),
                                Fold::Raw => a.args().first().copied(),
                            };
                            at.and_then(|i| {
                                by_part
                                    .get(&i)
                                    .and_then(|v| v.get(pi).copied())
                                    .or_else(|| cols.get(i).map(|c| sketch_bytes(c.as_ref())))
                            })
                            .unwrap_or(0)
                        } else if n > 0 && a.theta_raw_values(states) {
                            // The one buffer an array value is copied into while it hashes.
                            a.args()
                                .first()
                                .and_then(|&i| cols.get(i))
                                .map_or(0, |c| largest_array_bytes(c.as_ref()))
                        } else {
                            0
                        };
                        a.projected_part(n, states, state_bytes)
                    })
                    .sum::<usize>()
            })
            .sum();
        let kept: usize = self.parts.first().map_or(0, |p| {
            p.accs
                .iter()
                .zip(&self.core.ranges)
                .map(|(a, r)| {
                    // The HLL sketches are bounded per partition above (their rows are the partition's).
                    if a.projects_by_partition() {
                        return 0;
                    }
                    // The value column, then (`max_by`/`min_by`) the key.
                    let at = |n: usize| {
                        let i = match fold {
                            Fold::States { offset } => r.start.checked_sub(offset).map(|i| i + n),
                            Fold::Raw => a.args().get(n).copied(),
                        };
                        i.and_then(|i| cols.get(i))
                    };
                    a.projected(0, at(0), at(1), states)
                        .saturating_sub(a.projected(0, None, None, false))
                })
                .sum()
        });
        parts + kept
    }

    /// The radix partition of each input row, when a batch of sketches spreads over several partitions (so each
    /// is charged its own rows' sketch bytes); `None` for one partition, the direct path, or no sketch aggregate.
    pub(super) fn row_parts(&self, p: &RowPrepared) -> Option<Vec<u8>> {
        let spread = p.counts.iter().filter(|&&c| c > 0).count() > 1;
        if p.local.is_some() || self.bits == 0 || !spread || p.keys.hashes().len() != p.rows() {
            return None;
        }
        let reads = self.parts.first().is_some_and(|v| {
            v.accs
                .iter()
                .any(|a| a.theta_reads_sketches(matches!(Self::fold_of(p), Fold::States { .. })))
        });
        reads.then(|| {
            p.keys
                .hashes()
                .iter()
                .map(|&h| u8::try_from(part_of(h, self.bits)).unwrap_or(0))
                .collect()
        })
    }
}
