// SPDX-License-Identifier: Apache-2.0
//! The partition index an operator of a `UNION ALL` group sees (Spark's
//! `partitionIndex` of the operator's input RDD).
//!
//! Every driver of a pipeline group has a globally unique partition id;
//! the members' ids follow one another (`span` ids each). An operator
//! numbers its input's partitions from its [`Origin`]:
//! - below a union, in a branch: the branch's own partitions from 0
//!   (`UnionRDD` iterates the child's own partition);
//! - above a plain union: branch `i` after the partitions of branches
//!   `0..i` (`UnionRDD`'s numbering);
//! - above a coalesced union (every input Spark's `SinglePartition`, under
//!   `spark.sql.unionOutputPartitioning`): every branch at index 0, as
//!   `SQLPartitioningAwareUnionRDD` zips one partition of each child.
//!
//! The index of id `p` for operator `k` of member `m` is
//! `p - numbering_base(m, k)`; this module computes the bases.
//!
//! Coalescing applies only to a group whose members run one after another
//! (`runner.rs` `members_run_in_turn`). Known gap: when a shared operator
//! needs a flush (a hash-join probe above the union), the members run
//! together and the union keeps the plain numbering, deterministic but not
//! Spark's one stream (Spark 4.2.0 `bhj_above`: mono 0..7 across both
//! branches).

use std::cell::RefCell;
use std::collections::HashMap;

use crate::pipeline::Origin;

/// What the numbering needs of one member of a group.
pub(crate) struct MemberShape<'a> {
    /// The member's pipeline index.
    pub(crate) pipeline: usize,
    /// Its first partition id.
    pub(crate) base: u64,
    /// The partition ids it takes (its partition count, at least its drivers).
    pub(crate) span: u64,
    /// Whether its source is Spark's `SinglePartition`.
    pub(crate) single_source: bool,
    /// Per operator: its origin.
    pub(crate) origins: &'a [Origin],
    /// Per operator: whether it keeps its input's partitioning.
    pub(crate) keeps: Vec<bool>,
    /// The member's fan-ins (`Pipeline::fans`).
    pub(crate) fans: &'a [Vec<Origin>],
}

/// The numbering of one group.
pub(crate) struct Numbering<'a> {
    members: &'a [MemberShape<'a>],
    by_pipeline: HashMap<usize, usize>,
    /// `spark.sql.unionOutputPartitioning`.
    coalesce: bool,
    coalesced: RefCell<HashMap<Origin, bool>>,
    parts: RefCell<HashMap<Origin, u64>>,
}

impl<'a> Numbering<'a> {
    pub(crate) fn new(members: &'a [MemberShape<'a>], coalesce: bool) -> Self {
        let by_pipeline = members
            .iter()
            .enumerate()
            .map(|(i, m)| (m.pipeline, i))
            .collect();
        Self {
            members,
            by_pipeline,
            coalesce,
            coalesced: RefCell::new(HashMap::new()),
            parts: RefCell::new(HashMap::new()),
        }
    }

    fn member(&self, pipeline: usize) -> Option<&MemberShape<'a>> {
        self.by_pipeline
            .get(&pipeline)
            .and_then(|&i| self.members.get(i))
    }

    /// The fan-in that `o` sits right above, as `(first branch, other branches)`.
    fn fan(&self, o: Origin) -> Option<(Origin, &'a [Origin])> {
        let level = o.level.checked_sub(1)?;
        let index = *self.by_pipeline.get(&o.pipeline)?;
        let members: &'a [MemberShape<'a>] = self.members;
        let rest = members.get(index)?.fans.get(level)?;
        Some((
            Origin {
                pipeline: o.pipeline,
                level,
            },
            rest.as_slice(),
        ))
    }

    /// Whether every operator numbering from `o` keeps its partitioning.
    fn operators_keep(&self, o: Origin) -> bool {
        self.member(o.pipeline).is_some_and(|m| {
            m.origins
                .iter()
                .zip(&m.keeps)
                .filter(|(origin, _)| **origin == o)
                .all(|(_, keeps)| *keeps)
        })
    }

    /// Whether the output at `o` is Spark's `SinglePartition`.
    fn single(&self, o: Origin) -> bool {
        let below = if o.level == 0 {
            self.member(o.pipeline).is_some_and(|m| m.single_source)
        } else {
            self.coalesced(o)
        };
        below && self.operators_keep(o)
    }

    /// Whether the fan-in right below `o` is one partition.
    pub(crate) fn coalesced(&self, o: Origin) -> bool {
        if !self.coalesce {
            return false;
        }
        if let Some(&c) = self.coalesced.borrow().get(&o) {
            return c;
        }
        let c = self.fan(o).is_some_and(|(first, rest)| {
            self.single(first) && rest.iter().all(|b| self.single(*b))
        });
        self.coalesced.borrow_mut().insert(o, c);
        c
    }

    /// The partitions of the output at `o` in Spark's numbering.
    fn parts(&self, o: Origin) -> u64 {
        if let Some(&p) = self.parts.borrow().get(&o) {
            return p;
        }
        let p = match self.fan(o) {
            None => self.member(o.pipeline).map_or(0, |m| m.span),
            Some(_) if self.coalesced(o) => 1,
            Some((first, rest)) => rest.iter().fold(self.parts(first), |acc, b| {
                acc.saturating_add(self.parts(*b))
            }),
        };
        self.parts.borrow_mut().insert(o, p);
        p
    }

    /// The index of member `pipeline`'s first partition at `o`, or `None`
    /// when the member is not below `o`.
    fn offset(&self, pipeline: usize, o: Origin) -> Option<u64> {
        let Some((first, rest)) = self.fan(o) else {
            return (o.pipeline == pipeline).then_some(0);
        };
        let one = self.coalesced(o);
        if let Some(x) = self.offset(pipeline, first) {
            return Some(if one { 0 } else { x });
        }
        let mut before = self.parts(first);
        for b in rest {
            if let Some(x) = self.offset(pipeline, *b) {
                return Some(if one { 0 } else { before.saturating_add(x) });
            }
            before = before.saturating_add(self.parts(*b));
        }
        None
    }

    /// Per operator of member `m`, then for the group's sink (above every
    /// fan-in, `top`): the first partition id it numbers from
    /// (`base - offset`, the member's own base when its origin is not
    /// above it, which the builder never produces).
    pub(crate) fn bases(&self, m: &MemberShape<'_>, top: Origin) -> Vec<u32> {
        m.origins
            .iter()
            .chain(std::iter::once(&top))
            .map(|o| {
                let offset = self.offset(m.pipeline, *o).unwrap_or(0);
                u32::try_from(m.base.saturating_sub(offset)).unwrap_or(u32::MAX)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(pipeline: usize, level: usize) -> Origin {
        Origin { pipeline, level }
    }

    /// `A UNION ALL B`, a project in each branch and one above.
    #[test]
    fn plain_and_coalesced_unions() {
        let fans_a = vec![vec![o(1, 0)]];
        let origins_a = [o(0, 0), o(0, 1)];
        let origins_b = [o(1, 0), o(0, 1)];
        let shape = |single: bool, parts: u64| {
            vec![
                MemberShape {
                    pipeline: 0,
                    base: 0,
                    span: parts,
                    single_source: single,
                    origins: &origins_a,
                    keeps: vec![true, true],
                    fans: &fans_a,
                },
                MemberShape {
                    pipeline: 1,
                    base: parts,
                    span: 2,
                    single_source: single,
                    origins: &origins_b,
                    keeps: vec![true, true],
                    fans: &[],
                },
            ]
        };
        // 3 + 2 partitions: branch 2 is 0.. in its branch, 3.. above.
        let members = shape(false, 3);
        let top = o(0, 1);
        let n = Numbering::new(&members, true);
        assert_eq!(n.bases(&members[0], top), [0, 0, 0]);
        assert_eq!(n.bases(&members[1], top), [3, 0, 0]);
        assert!(!n.coalesced(top));
        // Two single partitions: index 0 everywhere.
        let members = shape(true, 1);
        let n = Numbering::new(&members, true);
        assert_eq!(n.bases(&members[1], top), [1, 1, 1]);
        assert!(n.coalesced(top));
        // ... unless the conf is off.
        let n = Numbering::new(&members, false);
        assert_eq!(n.bases(&members[1], top), [1, 0, 0]);
    }
}
