// SPDX-License-Identifier: Apache-2.0
//! The exact join enumerator over a query hypergraph: inputs are the bits
//! of a [`Set`], each join condition is the mask of the inputs it reads, and
//! `rows_of` estimates the rows of a set's join. [`dpsub`] fills the table
//! of the cheapest plan of every set that has one, under `C_out` (the rows
//! of every join in a plan, summed), visiting every subset and every split
//! of it, about 3^n splits.
//!
//! A split of a set into halves `a` and `b` is *connected* when some
//! condition reads only inputs of the set and at least one of each half;
//! a set has a plan when some connected split has planned halves. A
//! condition over three or more inputs (a hyperedge) thus joins two halves
//! only once its inputs are all in their union.
//!
//! P2-T113 compared it with DPhyp (connected subgraph and complement pairs
//! only), which fills the same table: on DSB, TPC-DS and TPC-H it changed
//! two plans and planned no faster, so it did not ship (the experiment's
//! `decision.md`).

use std::collections::HashMap;

use super::{Best, Set, rows};

/// What an enumeration did: the connected pairs of planned halves it
/// costed (csg-cmp pairs) and the candidate splits it looked at to find
/// them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Work {
    pub(super) pairs: u64,
    pub(super) considered: u64,
}

/// Every input as a plan of its own.
fn singletons(n: usize) -> impl Iterator<Item = (Set, Best)> {
    (0..n).map(|i| {
        (
            1 << i,
            Best {
                cost: 0.0,
                left: 0,
                right: 0,
            },
        )
    })
}

/// Whether some condition joins `a` and `b`: it reads only inputs of
/// `a ∪ b`, and some of each.
fn connected(masks: &[Set], a: Set, b: Set) -> bool {
    let s = a | b;
    masks
        .iter()
        .any(|&m| m & !s == 0 && m & a != 0 && m & b != 0)
}

/// Dynamic programming over every split of every subset of `n` inputs:
/// the table of cheapest plans and the work done. `n` is below 64. A
/// set's join costs `rows_of` of the set plus `split_of` of its two halves
/// (the build of a hash join; zero for `C_out` by rows).
pub(super) fn dpsub(
    n: usize,
    masks: &[Set],
    rows_of: &dyn Fn(Set) -> f64,
    split_of: &dyn Fn(Set, Set) -> f64,
) -> (HashMap<Set, Best>, Work) {
    let full: Set = (1 << n) - 1;
    let mut best: HashMap<Set, Best> = singletons(n).collect();
    let mut work = Work::default();
    for s in 1..=full {
        if s.count_ones() < 2 {
            continue;
        }
        let mut found: Option<Best> = None;
        let low = s.isolate_lowest_one();
        let mut a = (s - 1) & s;
        while a > 0 {
            let b = s & !a;
            work.considered += u64::from(a & low != 0);
            // Each split once (the half holding the lowest input is `a`),
            // both halves planned, and joined by some conjunct.
            if a & low != 0
                && let (Some(x), Some(y)) = (best.get(&a), best.get(&b))
                && connected(masks, a, b)
            {
                work.pairs += 1;
                let cost = x.cost + y.cost + split_of(a, b);
                // The first of equal splits wins: they come largest `a` first.
                if found.is_none_or(|f| cost < f.cost) {
                    found = Some(Best {
                        cost,
                        left: a,
                        right: b,
                    });
                }
            }
            a = (a - 1) & s;
        }
        if let Some(mut f) = found {
            f.cost = rows(f.cost + rows_of(s));
            best.insert(s, f);
        }
    }
    (best, work)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows of a set: the product of the inputs' rows and of the
    /// selectivities of the conditions it holds, as the optimizer estimates.
    fn estimator(input_rows: Vec<f64>, conds: Vec<(Set, f64)>) -> impl Fn(Set) -> f64 {
        move |s| {
            let base: f64 = (0..input_rows.len())
                .filter(|&i| s & (1 << i) != 0)
                .map(|i| input_rows[i])
                .product();
            let sel: f64 = conds
                .iter()
                .filter(|(m, _)| m & !s == 0)
                .map(|(_, sel)| sel)
                .product();
            rows(base * sel)
        }
    }

    #[test]
    fn a_chain_costs_its_connected_splits() {
        let n = 10;
        let masks: Vec<Set> = (1..n).map(|i| (1 << (i - 1)) | (1 << i)).collect();
        let rows_of = estimator(vec![100.0; n], masks.iter().map(|&m| (m, 0.01)).collect());
        let (best, work) = dpsub(n, &masks, &rows_of, &|_, _| 0.0);
        // (n³ − n) / 6 connected pairs of a chain, among every split of
        // every subset: (3^n + 1) / 2 − 2^n.
        assert_eq!(work.pairs, 165);
        assert_eq!(work.considered, 3u64.pow(10).div_ceil(2) - 2u64.pow(10));
        // Only connected sets (the 55 intervals) have plans.
        assert_eq!(best.len(), 55);
    }

    #[test]
    fn a_part_held_by_a_hyperedge_alone_has_no_plan() {
        // Inputs 0, 1 and 2 meet only in one condition over all three.
        let masks = vec![0b111];
        let rows_of = estimator(vec![10.0; 3], vec![(0b111, 0.5)]);
        let (best, work) = dpsub(3, &masks, &rows_of, &|_, _| 0.0);
        assert!(!best.contains_key(&0b111));
        assert_eq!(work.pairs, 0);
    }

    #[test]
    fn a_hyperedge_joins_halves_that_hold_its_inputs() {
        // 0-1 joined; the hyperedge {0, 1, 2} then joins {0, 1} and {2}.
        let masks = vec![0b011, 0b111];
        let rows_of = estimator(vec![10.0; 3], vec![(0b011, 0.1), (0b111, 0.5)]);
        let (best, _) = dpsub(3, &masks, &rows_of, &|_, _| 0.0);
        assert_eq!(
            best.get(&0b111).map(|b| (b.left, b.right)),
            Some((0b011, 0b100))
        );
    }

    #[test]
    fn equal_splits_keep_the_first() {
        // A triangle of equal inputs: every split of {0, 1, 2} costs the
        // same, and the largest half holding input 0 wins.
        let masks = vec![0b011, 0b110, 0b101];
        let rows_of = estimator(vec![10.0; 3], masks.iter().map(|&m| (m, 0.1)).collect());
        let (best, work) = dpsub(3, &masks, &rows_of, &|_, _| 0.0);
        assert_eq!(work.pairs, 6);
        assert_eq!(
            best.get(&0b111).map(|b| (b.left, b.right)),
            Some((0b101, 0b010))
        );
    }
}
