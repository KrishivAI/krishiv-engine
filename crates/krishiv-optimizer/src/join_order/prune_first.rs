// SPDX-License-Identifier: Apache-2.0
//! Prune first (perf lever D1a, design `P3-0.3.1-fd-ff-step0.md` §2): a bounded-regret choice
//! between the cheapest join order and the cheapest one that plans a *block* of inputs first and
//! joins the tree's largest input to it later.
//!
//! The gate: the largest input `L` (by estimated rows) joins some input `P` on two or more
//! columns (a composite key, [`super::composite_keys`]), and one of `L`'s columns of that key is
//! clustered in `L`'s files: the mean row group spans at most [`SPREAD_MAX`] of the column's
//! range ([`Statistics::unit_spread`]). The block is `P` with every input it reaches without
//! passing `L`; the blocked order plans the block on its own and joins `L` to it before anything
//! else, so `L` (the larger side) probes the block's table. A composite-key estimate cannot tell
//! a foreign key (TPC-DS `store_sales` to `store_returns` on item and ticket) from a many-to-many
//! match on the same statistics, and when
//! the block is planned first, the runtime filter its keys publish skips most of `L`'s row groups
//! through the clustered column, which plain `C_out` does not see.
//!
//! The choice: the blocked order wins when its estimated cost is at most [`REGRET`] times the
//! cheapest order's (bounded regret: a wrong pick costs the block's extra rows, a right one the
//! scan of most of `L`). `EXPLAIN VERBOSE` names it on the tree's note (`prune_first=`), with
//! `ratio` the blocked cost over `max(best, 1)` (a cheapest cost under 1 is clamped to 1). Only
//! under the rows cost (`krishiv.optimizer.joinCost=rows`, the default).

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_plan::{AttrId, LogicalPlan, PlanNode};

use super::{Graph, Set, selection};
use crate::stats::Statistics;

/// The most a clustered column's mean row group may span of the column's range.
pub(super) const SPREAD_MAX: f64 = 1.0 / 16.0;

/// The most times the cheapest order's estimated cost the blocked order may cost.
pub(super) const REGRET: f64 = 4.0;

/// The columns of `leaf` (through filters and column selections down to a scan) clustered in the
/// scan's files.
pub(super) fn clustered(leaf: &Arc<LogicalPlan>, stats: &dyn Statistics) -> HashSet<AttrId> {
    let mut p = leaf;
    loop {
        match &p.node {
            PlanNode::Filter { input, .. } => p = input,
            PlanNode::Scan(scan) => {
                return scan
                    .output
                    .iter()
                    .filter(|c| {
                        stats
                            .unit_spread(scan, &c.name)
                            .is_some_and(|s| s <= SPREAD_MAX)
                    })
                    .map(|c| c.id)
                    .collect();
            }
            _ => match selection(p) {
                Some(input) => p = input,
                None => return HashSet::new(),
            },
        }
    }
}

/// What the choice did on one tree: the block's inputs and the two estimated costs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Choice {
    block: Set,
    best: f64,
    /// The blocked order's cost (`None`: no blocked order has a plan).
    blocked: Option<f64>,
    pub(super) taken: bool,
}

impl Choice {
    pub(super) fn new(block: Set, best: f64, blocked: Option<f64>) -> Self {
        let taken = blocked.is_some_and(|b| b <= REGRET * best.max(1.0));
        Self {
            block,
            best,
            blocked,
            taken,
        }
    }
}

impl std::fmt::Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inputs: Vec<String> = (0..Set::BITS)
            .filter(|i| self.block & (1 << i) != 0)
            .map(|i| i.to_string())
            .collect();
        let verdict = if self.taken { "taken" } else { "kept" };
        match self.blocked {
            Some(b) => write!(
                f,
                "{verdict}(block={} ratio={:.3})",
                inputs.join(","),
                b / self.best.max(1.0)
            ),
            None => write!(f, "{verdict}(block={} no-plan)", inputs.join(",")),
        }
    }
}

impl Graph {
    /// The block to plan first when the gate holds (module docs), and the block with the largest
    /// input, as sets of inputs.
    ///
    /// Only under `krishiv.optimizer.joinCost=rows`: [`REGRET`] was measured on `C_out`, so under
    /// the bytes cost the choice stays off. `stats` says which of the largest input's columns are
    /// clustered (asked only here, for that one input).
    pub(super) fn prune_block(&self, stats: &dyn Statistics) -> Option<(Set, Set)> {
        let n = self.n();
        if self.by_bytes || n < 3 || self.leaves.len() != n {
            return None;
        }
        let (l, _) = self
            .rows
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))?;
        let lc = &clustered(&self.leaves[l], stats);
        if lc.is_empty() {
            return None;
        }
        // The partner of a composite-key edge of `l` on a clustered column of `l`, the lowest
        // such input (canonical).
        let partner = self
            .preds
            .iter()
            .filter(|p| p.carrier.is_some() && p.edge && p.on.contains(&l))
            .filter(|p| {
                p.class
                    .is_some_and(|[a, b]| lc.contains(&a) || lc.contains(&b))
            })
            .filter_map(|p| match p.on.as_slice() {
                [a, b] if *a == l => Some(*b),
                [a, b] if *b == l => Some(*a),
                _ => None,
            })
            .min()?;
        // Every input the partner reaches through join conditions that do not read `l`.
        let lbit: Set = 1 << l;
        let mut block: Set = 1 << partner;
        loop {
            let grown = self
                .preds
                .iter()
                .filter(|p| p.edge)
                .map(|p| p.on.iter().fold(0, |m: Set, &x| m | (1 << x)))
                .filter(|&m| m & lbit == 0 && m & block != 0)
                .fold(block, |acc, m| acc | m);
            if grown == block {
                break;
            }
            block = grown;
        }
        (block.count_ones() >= 2).then_some((block, block | lbit))
    }
}

#[cfg(test)]
mod tests {
    use super::{Choice, REGRET};

    #[test]
    fn the_blocked_order_wins_within_the_regret_bound() {
        assert!(Choice::new(0b110, 100.0, Some(100.0 * REGRET)).taken);
        assert!(!Choice::new(0b110, 100.0, Some(100.0 * REGRET + 1.0)).taken);
        assert!(!Choice::new(0b110, 100.0, None).taken);
        assert_eq!(
            Choice::new(0b110, 100.0, Some(250.0)).to_string(),
            "taken(block=1,2 ratio=2.500)"
        );
    }
}
