// SPDX-License-Identifier: Apache-2.0
//! The exact join enumerators over a query hypergraph (P2-T113): inputs are
//! the bits of a [`Set`], each join condition is the mask of the inputs it
//! reads, and `rows_of` estimates the rows of a set's join. Both fill the
//! same table, the cheapest plan of every set that has one, under `C_out`
//! (the rows of every join in a plan, summed), and break ties the same way,
//! so they return the same table; they differ in how many pairs of halves
//! they look at.
//!
//! A split of a set into halves `a` and `b` is *connected* when some
//! condition reads only inputs of the set and at least one of each half;
//! a set has a plan when some connected split has planned halves. A
//! condition over three or more inputs (a hyperedge) thus joins two halves
//! only once its inputs are all in their union.
//!
//! - [`dpsub`] visits every subset and every split of it, about 3^n splits.
//! - [`dphyp`] (Moerkotte and Neumann, "Dynamic programming strikes back",
//!   SIGMOD 2008) visits only connected subgraphs and, for each, the
//!   connected complements that a condition joins to it (csg-cmp pairs):
//!   (n³ − n) / 6 on a chain of n inputs instead of 3^n. Its neighbourhood
//!   of a set takes, for each condition reading some of the set and some
//!   inputs outside it, the lowest outside input; a hyperedge's other inputs
//!   join as the subgraph grows, and a set is used only once its plan
//!   exists. It stops after a budget of pairs.

use std::collections::HashMap;

use super::{Best, Set, rows};

/// What an enumeration did: the connected pairs of planned halves it
/// costed (csg-cmp pairs; the same for every exact enumerator) and the
/// candidate pairs it looked at to find them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Work {
    pub(super) pairs: u64,
    pub(super) considered: u64,
}

/// The cheapest plan of a set while it is being filled: the cost of its
/// best split (the halves' costs summed, before its own rows), kept apart
/// from the plan's cost so that splits compare as [`dpsub`] compares them.
struct Entry {
    split: f64,
    best: Best,
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
/// the table of cheapest plans and the work done. `n` is below 64.
pub(super) fn dpsub(
    n: usize,
    masks: &[Set],
    rows_of: &dyn Fn(Set) -> f64,
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
                let cost = x.cost + y.cost;
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

/// The non-empty subsets of `mask`, in increasing order.
fn subsets(mask: Set) -> impl Iterator<Item = Set> {
    let mut sub: Set = 0;
    std::iter::from_fn(move || {
        sub = sub.wrapping_sub(mask) & mask;
        (sub != 0).then_some(sub)
    })
}

/// The inputs `0..=i` of input bit `v` (`B_v` in the paper).
fn up_to(v: Set) -> Set {
    v | (v - 1)
}

/// The pair budget ran out.
struct OverBudget;

struct DpHyp<'a> {
    masks: &'a [Set],
    rows_of: &'a dyn Fn(Set) -> f64,
    table: HashMap<Set, Entry>,
    work: Work,
    budget: u64,
}

/// DPhyp over `n` inputs (below 64): the table of cheapest plans (the same
/// as [`dpsub`]'s) and the work done, or only the work once more than
/// `budget` pairs were costed.
pub(super) fn dphyp(
    n: usize,
    masks: &[Set],
    rows_of: &dyn Fn(Set) -> f64,
    budget: u64,
) -> Result<(HashMap<Set, Best>, Work), Work> {
    let mut d = DpHyp {
        masks,
        rows_of,
        table: singletons(n)
            .map(|(s, best)| (s, Entry { split: 0.0, best }))
            .collect(),
        work: Work::default(),
        budget,
    };
    if d.solve(n).is_err() {
        return Err(d.work);
    }
    let table = d.table.into_iter().map(|(s, e)| (s, e.best)).collect();
    Ok((table, d.work))
}

impl DpHyp<'_> {
    /// `Solve`: from the highest input down, each input's connected
    /// subgraphs that hold no lower input.
    fn solve(&mut self, n: usize) -> Result<(), OverBudget> {
        for i in (0..n).rev() {
            let v: Set = 1 << i;
            self.emit_csg(v)?;
            self.enumerate_csg(v, up_to(v))?;
        }
        Ok(())
    }

    /// `N(S, X)`: for each condition reading some of `s` and inputs outside
    /// it, none of them in `x`, the lowest of those outside inputs.
    fn neighbours(&self, s: Set, x: Set) -> Set {
        let mut n = 0;
        for &m in self.masks {
            let rest = m & !s;
            if m & s != 0 && rest != 0 && rest & x == 0 {
                n |= rest.isolate_lowest_one();
            }
        }
        n
    }

    /// `EnumerateCsgRec`: the connected supersets of `s` grown through its
    /// neighbours outside `x`.
    fn enumerate_csg(&mut self, s: Set, x: Set) -> Result<(), OverBudget> {
        let nb = self.neighbours(s, x);
        if nb == 0 {
            return Ok(());
        }
        for sub in subsets(nb) {
            self.work.considered += 1;
            if self.table.contains_key(&(s | sub)) {
                self.emit_csg(s | sub)?;
            }
        }
        for sub in subsets(nb) {
            self.enumerate_csg(s | sub, x | nb)?;
        }
        Ok(())
    }

    /// `EmitCsg`: the complements of the connected subgraph `s1` that hold
    /// no input below its lowest one, each started from one neighbour.
    fn emit_csg(&mut self, s1: Set) -> Result<(), OverBudget> {
        let x = s1 | up_to(s1.isolate_lowest_one());
        let nb = self.neighbours(s1, x);
        let mut rest = nb;
        while rest != 0 {
            let v: Set = 1 << (63 - rest.leading_zeros());
            rest &= !v;
            self.work.considered += 1;
            if connected(self.masks, s1, v) {
                self.emit_pair(s1, v)?;
            }
            self.enumerate_cmp(s1, v, x | (nb & up_to(v)))?;
        }
        Ok(())
    }

    /// `EnumerateCmpRec`: the complements grown from `s2` through its
    /// neighbours outside `x`.
    fn enumerate_cmp(&mut self, s1: Set, s2: Set, x: Set) -> Result<(), OverBudget> {
        let nb = self.neighbours(s2, x);
        if nb == 0 {
            return Ok(());
        }
        for sub in subsets(nb) {
            let t = s2 | sub;
            self.work.considered += 1;
            if self.table.contains_key(&t) && connected(self.masks, s1, t) {
                self.emit_pair(s1, t)?;
            }
        }
        for sub in subsets(nb) {
            self.enumerate_cmp(s1, s2 | sub, x | nb)?;
        }
        Ok(())
    }

    /// `EmitCsgCmp`: costs joining the planned halves `s1` and `s2`. Splits
    /// compare as in [`dpsub`]: by the halves' summed cost, an equal one won
    /// by the larger half that holds the lowest input.
    fn emit_pair(&mut self, s1: Set, s2: Set) -> Result<(), OverBudget> {
        self.work.pairs += 1;
        if self.work.pairs > self.budget {
            return Err(OverBudget);
        }
        let s = s1 | s2;
        let (a, b) = if s1 & s.isolate_lowest_one() != 0 {
            (s1, s2)
        } else {
            (s2, s1)
        };
        let (Some(x), Some(y)) = (self.table.get(&a), self.table.get(&b)) else {
            return Ok(());
        };
        let split = x.best.cost + y.best.cost;
        if let Some(e) = self.table.get(&s)
            && !(split < e.split || (split == e.split && a > e.best.left))
        {
            return Ok(());
        }
        let cost = rows(split + (self.rows_of)(s));
        self.table.insert(
            s,
            Entry {
                split,
                best: Best {
                    cost,
                    left: a,
                    right: b,
                },
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

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

    /// A random query graph: `n` inputs with rows, a spanning tree of
    /// binary conditions over some of them (so parts stay connected or
    /// not at random), extra binary conditions and some hyperedges.
    fn graph() -> impl Strategy<Value = (usize, Vec<f64>, Vec<(Set, f64)>)> {
        (2usize..=10).prop_flat_map(|n| {
            (
                Just(n),
                proptest::collection::vec(1.0f64..1e7, n),
                proptest::collection::vec(
                    (
                        proptest::collection::btree_set(0..n, 2..=3.min(n)),
                        prop_oneof![Just(1.0), 1e-7f64..1.0],
                    ),
                    1..=(2 * n),
                ),
            )
                .prop_map(|(n, rows, conds)| {
                    let conds = conds
                        .into_iter()
                        .map(|(on, sel)| (on.iter().fold(0, |m, &x| m | (1 << x)), sel))
                        .collect();
                    (n, rows, conds)
                })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// DPhyp fills exactly DPsub's table: every set planned, at the same
        /// cost, from the same halves (so the same plan).
        #[test]
        fn dphyp_finds_dpsubs_plans((n, rows, conds) in graph()) {
            let masks: Vec<Set> = conds.iter().map(|(m, _)| *m).collect();
            let rows_of = estimator(rows, conds);
            let (sub, sub_work) = dpsub(n, &masks, &rows_of);
            let (hyp, hyp_work) = dphyp(n, &masks, &rows_of, u64::MAX).unwrap();
            prop_assert_eq!(&hyp, &sub);
            // Each csg-cmp pair costed exactly once by both.
            prop_assert_eq!(hyp_work.pairs, sub_work.pairs);
        }
    }

    /// Binary conditions form a chain: each pair of adjacent inputs.
    fn chain(n: usize) -> Vec<Set> {
        (1..n).map(|i| (1 << (i - 1)) | (1 << i)).collect()
    }

    #[test]
    fn a_chain_costs_its_csg_cmp_pairs_only() {
        let n = 10;
        let masks = chain(n);
        let rows_of = estimator(vec![100.0; n], masks.iter().map(|&m| (m, 0.01)).collect());
        let (sub, sub_work) = dpsub(n, &masks, &rows_of);
        let (hyp, hyp_work) = dphyp(n, &masks, &rows_of, u64::MAX).unwrap();
        assert_eq!(hyp, sub);
        // (n³ − n) / 6 csg-cmp pairs on a chain, each costed once; DPsub
        // looks at every split of every subset to find them.
        assert_eq!((hyp_work.pairs, sub_work.pairs), (165, 165));
        assert_eq!(sub_work.considered, 3u64.pow(10).div_ceil(2) - 2u64.pow(10));
        assert!(hyp_work.considered < 1_000, "{hyp_work:?}");
    }

    #[test]
    fn a_star_costs_its_csg_cmp_pairs_only() {
        let n = 8;
        let masks: Vec<Set> = (1..n).map(|i| 1 | (1 << i)).collect();
        let rows_of = estimator(vec![1000.0; n], masks.iter().map(|&m| (m, 0.001)).collect());
        let (sub, _) = dpsub(n, &masks, &rows_of);
        let (hyp, work) = dphyp(n, &masks, &rows_of, u64::MAX).unwrap();
        assert_eq!(hyp, sub);
        // (n − 1) · 2^(n − 2) on a star.
        assert_eq!(work.pairs, 7 * 64);
    }

    #[test]
    fn the_budget_stops_the_enumeration() {
        let n = 12;
        let masks: Vec<Set> = (1..n).map(|i| 1 | (1 << i)).collect();
        let rows_of = estimator(vec![1000.0; n], Vec::new());
        assert_eq!(
            dphyp(n, &masks, &rows_of, 100)
                .map(|_| ())
                .map_err(|w| w.pairs),
            Err(101)
        );
        assert!(dphyp(n, &masks, &rows_of, 100_000).is_ok());
    }

    #[test]
    fn a_part_held_by_a_hyperedge_alone_has_no_plan() {
        // Inputs 0, 1 and 2 meet only in one condition over all three.
        let masks = vec![0b111];
        let rows_of = estimator(vec![10.0; 3], vec![(0b111, 0.5)]);
        let (sub, _) = dpsub(3, &masks, &rows_of);
        let (hyp, work) = dphyp(3, &masks, &rows_of, u64::MAX).unwrap();
        assert_eq!(hyp, sub);
        assert!(!hyp.contains_key(&0b111));
        assert_eq!(work.pairs, 0);
    }

    #[test]
    fn a_hyperedge_joins_halves_that_hold_its_inputs() {
        // 0-1 joined; the hyperedge {0, 1, 2} then joins {0, 1} and {2}.
        let masks = vec![0b011, 0b111];
        let rows_of = estimator(vec![10.0; 3], vec![(0b011, 0.1), (0b111, 0.5)]);
        let (sub, _) = dpsub(3, &masks, &rows_of);
        let (hyp, _) = dphyp(3, &masks, &rows_of, u64::MAX).unwrap();
        assert_eq!(hyp, sub);
        assert_eq!(
            hyp.get(&0b111).map(|b| (b.left, b.right)),
            Some((0b011, 0b100))
        );
    }
}
