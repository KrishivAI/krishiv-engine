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
                tie: 0.0,
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

/// Whether costs `(cost, tie)` are below `(other, other_tie)`: the cost
/// decides unless the two are equal, up to the rounding of summing the same
/// numbers in another order (a relative 1e-9, three orders under the
/// smallest gap that matters: TPC-H q09's two shapes differ by 1e-6); then
/// the tie (bytes) does.
pub(super) fn cheaper(cost: (f64, f64), other: (f64, f64)) -> bool {
    if same_cost(cost.0, other.0) {
        cost.1 < other.1
    } else {
        cost.0 < other.0
    }
}

/// Whether two costs are equal up to the rounding of [`cheaper`].
fn same_cost(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

/// What [`dpsub`] asks of the graph. A set's rows, width and the bytes of a join input of it are worked
/// out once, when the set is planned, into tables indexed by the set's bits: a split reads its halves
/// from those, which is what it does 3^n times.
pub(super) struct Shape<'a> {
    /// The rows of the join of a set's inputs.
    pub(super) rows_of: &'a dyn Fn(Set) -> f64,
    /// The bytes per row of that join (0 when the reads are unknown).
    pub(super) width_of: &'a dyn Fn(Set) -> f64,
    /// What a set's join result costs, from its rows (`C_out`: the rows; bytes: a row's work).
    pub(super) result_cost: &'a dyn Fn(f64) -> f64,
    /// Whether plans are costed by bytes: a split then also costs its build input.
    pub(super) by_bytes: bool,
    /// A split's build cost from the rows and width of its halves (by bytes only).
    pub(super) build_cost: &'a dyn Fn((f64, f64), (f64, f64)) -> f64,
    /// Whether, costed by rows, plans of equal rows are told apart by their bytes.
    pub(super) ties: bool,
    /// The bytes of producing `rows` rows (the tie of a set).
    pub(super) tie_result: &'a dyn Fn(f64) -> f64,
    /// The bytes of building a join input of `rows` rows and `width` bytes per row.
    pub(super) tie_bytes: &'a dyn Fn(f64, f64) -> f64,
}

/// A set that has a plan: its rows, its width and the bytes of building it.
#[derive(Clone, Copy, Default)]
struct Planned {
    rows: f64,
    width: f64,
    bytes: f64,
}

/// Dynamic programming over every split of every subset of `n` inputs:
/// the table of cheapest plans and the work done. `n` is below 64 (and the tables below 2^n entries, so
/// at most `MAX_DP_LIMIT` inputs). A set's join costs its result plus, by bytes, the build of its two halves.
pub(super) fn dpsub(n: usize, masks: &[Set], shape: &Shape<'_>) -> (HashMap<Set, Best>, Work) {
    let full: Set = (1 << n) - 1;
    let size = usize::try_from(full).map_or(usize::MAX, |f| f + 1);
    let mut table: Vec<Option<Best>> = vec![None; size];
    let mut planned = vec![Planned::default(); size];
    let at = |s: Set| usize::try_from(s).unwrap_or(usize::MAX);
    let stats = |s: Set| {
        let (rows, width) = ((shape.rows_of)(s), (shape.width_of)(s));
        Planned {
            rows,
            width,
            bytes: if shape.ties {
                (shape.tie_bytes)(rows, width)
            } else {
                0.0
            },
        }
    };
    for (s, b) in singletons(n) {
        table[at(s)] = Some(b);
        planned[at(s)] = stats(s);
    }
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
                && let (Some(x), Some(y)) = (table[at(a)], table[at(b)])
                && connected(masks, a, b)
            {
                work.pairs += 1;
                let build = if shape.by_bytes {
                    let (pa, pb) = (planned[at(a)], planned[at(b)]);
                    (shape.build_cost)((pa.rows, pa.width), (pb.rows, pb.width))
                } else {
                    0.0
                };
                let cost = x.cost + y.cost + build;
                // The tie (the bytes of the split: the build input is the one with fewer rows,
                // `b` on a tie) is worked out only when it can matter: for a split that is cheaper
                // in rows or equal to the best so far, not for the many that are dearer. The first
                // of equal splits wins: they come largest `a` first.
                // (The halves' stored ties count even with ties off: a combined set's is clamped to
                // at least 1.0, so of equal splits the one with fewer combined halves wins.)
                let tie_of_split = || {
                    let (pa, pb) = (planned[at(a)], planned[at(b)]);
                    x.tie
                        + y.tie
                        + if !shape.ties {
                            0.0
                        } else if pb.rows <= pa.rows {
                            pb.bytes
                        } else {
                            pa.bytes
                        }
                };
                let tie = match found {
                    None => Some(tie_of_split()),
                    // Dearer by more than the tolerance (the common case): skip.
                    Some(f) if cost - f.cost > 1e-9 * cost => None,
                    Some(f) if same_cost(cost, f.cost) => {
                        Some(tie_of_split()).filter(|&t| t < f.tie)
                    }
                    Some(f) => (cost < f.cost).then(tie_of_split),
                };
                if let Some(tie) = tie {
                    found = Some(Best {
                        cost,
                        tie,
                        left: a,
                        right: b,
                    });
                }
            }
            a = (a - 1) & s;
        }
        if let Some(mut f) = found {
            let p = stats(s);
            f.cost = rows(f.cost + (shape.result_cost)(p.rows));
            f.tie = if shape.ties {
                rows(f.tie + (shape.tie_result)(p.rows))
            } else {
                rows(f.tie)
            };
            table[at(s)] = Some(f);
            planned[at(s)] = p;
        }
    }
    let best = table
        .into_iter()
        .enumerate()
        .filter_map(|(s, b)| Some((Set::try_from(s).ok()?, b?)))
        .collect();
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

    /// A shape costed by `C_out` alone: no widths, no ties.
    fn by_rows(rows_of: &dyn Fn(Set) -> f64) -> Shape<'_> {
        Shape {
            rows_of,
            width_of: &|_| 0.0,
            result_cost: &|r| r,
            by_bytes: false,
            build_cost: &|_, _| 0.0,
            ties: false,
            tie_result: &|_| 0.0,
            tie_bytes: &|_, _| 0.0,
        }
    }

    #[test]
    fn a_chain_costs_its_connected_splits() {
        let n = 10;
        let masks: Vec<Set> = (1..n).map(|i| (1 << (i - 1)) | (1 << i)).collect();
        let rows_of = estimator(vec![100.0; n], masks.iter().map(|&m| (m, 0.01)).collect());
        let (best, work) = dpsub(n, &masks, &by_rows(&rows_of));
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
        let (best, work) = dpsub(3, &masks, &by_rows(&rows_of));
        assert!(!best.contains_key(&0b111));
        assert_eq!(work.pairs, 0);
    }

    #[test]
    fn a_hyperedge_joins_halves_that_hold_its_inputs() {
        // 0-1 joined; the hyperedge {0, 1, 2} then joins {0, 1} and {2}.
        let masks = vec![0b011, 0b111];
        let rows_of = estimator(vec![10.0; 3], vec![(0b011, 0.1), (0b111, 0.5)]);
        let (best, _) = dpsub(3, &masks, &by_rows(&rows_of));
        assert_eq!(
            best.get(&0b111).map(|b| (b.left, b.right)),
            Some((0b011, 0b100))
        );
    }

    #[test]
    fn equal_rows_are_broken_by_bytes() {
        // Two inputs joined last: {0, 1} then 2, or {1, 2} then 0 cost the
        // same rows; the tie says the second is the smaller in bytes.
        let masks = vec![0b011, 0b110];
        let rows_of = estimator(vec![10.0; 3], masks.iter().map(|&m| (m, 0.1)).collect());
        // The join of {1, 2} is the narrowest, so building it (rows equal) is the smaller build.
        let width_of = |s: Set| if s == 0b110 { 0.0 } else { 1.0 };
        let shape = Shape {
            width_of: &width_of,
            ties: true,
            tie_bytes: &|r, w| r * (48.0 + w),
            ..by_rows(&rows_of)
        };
        let (best, _) = dpsub(3, &masks, &shape);
        assert_eq!(
            best.get(&0b111).map(|b| (b.left, b.right)),
            Some((0b001, 0b110))
        );
        assert!(cheaper((1.0, 1.0), (1.0 + 1e-12, 2.0)));
        assert!(!cheaper((1.0, 3.0), (1.0, 2.0)));
        assert!(!cheaper((1.001, 1.0), (1.0, 2.0)));
    }

    #[test]
    fn equal_splits_keep_the_first() {
        // A triangle of equal inputs: every split of {0, 1, 2} costs the
        // same, and the largest half holding input 0 wins.
        let masks = vec![0b011, 0b110, 0b101];
        let rows_of = estimator(vec![10.0; 3], masks.iter().map(|&m| (m, 0.1)).collect());
        let (best, work) = dpsub(3, &masks, &by_rows(&rows_of));
        assert_eq!(work.pairs, 6);
        assert_eq!(
            best.get(&0b111).map(|b| (b.left, b.right)),
            Some((0b101, 0b010))
        );
    }

    /// The enumeration as it was before the tables: a map, and closures for every size. The split
    /// loop and tie rules are the specification the fast version must keep.
    fn reference(n: usize, masks: &[Set], shape: &Shape<'_>) -> HashMap<Set, Best> {
        let full: Set = (1 << n) - 1;
        let mut best: HashMap<Set, Best> = singletons(n).collect();
        let size = |s: Set| ((shape.rows_of)(s), (shape.width_of)(s));
        for s in 1..=full {
            if s.count_ones() < 2 {
                continue;
            }
            let mut found: Option<Best> = None;
            let low = s.isolate_lowest_one();
            let mut a = (s - 1) & s;
            while a > 0 {
                let b = s & !a;
                if a & low != 0
                    && let (Some(x), Some(y)) = (best.get(&a), best.get(&b))
                    && connected(masks, a, b)
                {
                    let (sa, sb) = (size(a), size(b));
                    let build = if shape.by_bytes {
                        (shape.build_cost)(sa, sb)
                    } else {
                        0.0
                    };
                    let cost = x.cost + y.cost + build;
                    let tie_split = if shape.ties {
                        let build = if sb.0 <= sa.0 { sb } else { sa };
                        (shape.tie_bytes)(build.0, build.1)
                    } else {
                        0.0
                    };
                    let tie = x.tie + y.tie + tie_split;
                    if found.is_none_or(|f| cheaper((cost, tie), (f.cost, f.tie))) {
                        found = Some(Best {
                            cost,
                            tie,
                            left: a,
                            right: b,
                        });
                    }
                }
                a = (a - 1) & s;
            }
            if let Some(mut f) = found {
                let r = (shape.rows_of)(s);
                f.cost = rows(f.cost + (shape.result_cost)(r));
                f.tie = rows(
                    f.tie
                        + if shape.ties {
                            (shape.tie_result)(r)
                        } else {
                            0.0
                        },
                );
                best.insert(s, f);
            }
        }
        best
    }

    #[test]
    fn the_tables_plan_exactly_as_the_map_and_closures_did() {
        // A small deterministic generator: random graphs of up to 8 inputs, costed by rows with ties
        // off and on, and by bytes.
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let pick = |v: &[f64], i: u64| v[usize::try_from(i).unwrap()];
        for _ in 0..3000 {
            let n = 3 + usize::try_from(next(6)).unwrap();
            let mut masks: Vec<Set> = Vec::new();
            for i in 1..n {
                // A spanning tree, then a few more edges and sometimes a hyperedge.
                masks.push((1 << i) | (1 << next(i as u64)));
            }
            for _ in 0..next(3) {
                masks.push((1 << next(n as u64)) | (1 << next(n as u64)));
            }
            if next(4) == 0 {
                masks.push((1 << next(n as u64)) | (1 << next(n as u64)) | (1 << next(n as u64)));
            }
            masks.retain(|m| m.count_ones() >= 2);
            let input_rows: Vec<f64> = (0..n).map(|_| pick(&[1.0, 10.0, 100.0], next(3))).collect();
            let conds: Vec<(Set, f64)> = masks
                .iter()
                .map(|&m| (m, pick(&[0.1, 0.01, 1.0, 0.001], next(4))))
                .collect();
            let rows_of = estimator(input_rows, conds);
            let widths: Vec<f64> = (0..n).map(|_| next(40) as f64).collect();
            let width_of = move |s: Set| {
                (0..n)
                    .filter(|&i| s & (1 << i) != 0)
                    .map(|i| widths[i])
                    .sum::<f64>()
            };
            for (by_bytes, ties) in [(false, false), (false, true), (true, false)] {
                let shape = Shape {
                    width_of: &width_of,
                    by_bytes,
                    build_cost: &|a, b| {
                        let build = if b.0 <= a.0 { b } else { a };
                        build.0 * (48.0 + build.1)
                    },
                    ties,
                    tie_result: &|r| r * 48.0,
                    tie_bytes: &|r, w| rows(r * (48.0 + w)),
                    ..by_rows(&rows_of)
                };
                let (fast, _) = dpsub(n, &masks, &shape);
                let slow = reference(n, &masks, &shape);
                assert_eq!(fast.len(), slow.len(), "sets with plans: {masks:?}");
                for (s, b) in &slow {
                    let f = fast
                        .get(s)
                        .unwrap_or_else(|| panic!("{s:b} missing: {masks:?}"));
                    assert_eq!(
                        (f.left, f.right),
                        (b.left, b.right),
                        "{s:b} {masks:?} bytes={by_bytes} ties={ties}"
                    );
                    assert_eq!(
                        (f.cost.to_bits(), f.tie.to_bits()),
                        (b.cost.to_bits(), b.tie.to_bits()),
                        "{s:b}"
                    );
                }
            }
        }
    }
}
