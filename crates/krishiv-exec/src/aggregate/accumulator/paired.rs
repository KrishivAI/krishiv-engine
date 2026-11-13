// SPDX-License-Identifier: Apache-2.0
//! Decimal `sum(x)` and `avg(x)` of one aggregate folded in one pass (study
//! 19 A4).
//!
//! TPC-H q01 computes `sum(l_quantity)` and `avg(l_quantity)` (and the same
//! for `l_extendedprice`) in one aggregate. Both keep the sum in the same
//! `decimal(min(38, p + 10), s)`, so on a batch both would take their proof
//! path for (raw rows, unit weights, no NULL, no `FILTER`, no sum can leave
//! the bound), [`update_all`] reads the values and group ids once and
//! updates both states in one loop. Each accumulator keeps its own state, so
//! partial states, spill files, merges and results, overflow and its error
//! included, are exactly what two separate folds give; any other batch is
//! folded by each accumulator on its own.

use super::small_fold::{SMALL_GROUPS, small_totals};
use super::{Accumulator, AggInput, AggregateExpr, AggregateFunction, DataType, Decimals, Kind};
use super::{arg, decimals, fits64_moves, proven_moves, refresh_max};
use crate::error::Result;
use arrow_array::{Array, ArrayRef};
use krishiv_kernels::decimal::DecimalNative;
use krishiv_types::batch::Weights;

/// The `(avg, sum)` index pairs of `exprs` that [`update_all`] may fold
/// together: a decimal `avg` and `sum` (not `try_`) of the same raw column,
/// both without a `FILTER`. Each aggregate is in at most one pair.
pub(crate) fn fold_pairs(exprs: &[AggregateExpr]) -> Vec<(usize, usize)> {
    let pairable = |e: &AggregateExpr, f: &AggregateFunction| {
        e.function == *f
            && e.input == AggInput::Raw
            && e.mask.is_none()
            && e.args.len() == 1
            && matches!(
                e.arg_type,
                DataType::Decimal64(..) | DataType::Decimal128(..)
            )
    };
    let mut taken = vec![false; exprs.len()];
    let mut pairs = Vec::new();
    for (a, avg) in exprs.iter().enumerate() {
        if !pairable(avg, &AggregateFunction::Avg) {
            continue;
        }
        let sum = exprs.iter().enumerate().position(|(s, e)| {
            !taken[s]
                && pairable(e, &AggregateFunction::Sum)
                && e.args == avg.args
                && e.arg_type == avg.arg_type
        });
        if let Some(s) = sum {
            taken[s] = true;
            pairs.push((a, s));
        }
    }
    pairs
}

/// [`Accumulator::update`] of every accumulator of `accs`, in order, but a
/// pair of [`fold_pairs`] folds in one pass where both would take their
/// proof path.
///
/// # Errors
/// As [`Accumulator::update`].
pub(crate) fn update_all(
    accs: &mut [Accumulator],
    pairs: &[(usize, usize)],
    groups: &[u32],
    cols: &[ArrayRef],
    w: &Weights,
) -> Result<()> {
    if pairs.is_empty() {
        return accs.iter_mut().try_for_each(|a| a.update(groups, cols, w));
    }
    let mut done = vec![false; accs.len()];
    for i in 0..accs.len() {
        if done[i] {
            continue;
        }
        let pair = pairs.iter().find(|&&(a, s)| a.min(s) == i);
        if let Some(&(a, s)) = pair
            && fold_pair(accs, a, s, groups, cols, w)?
        {
            done[a] = true;
            done[s] = true;
            continue;
        }
        if let Some(acc) = accs.get_mut(i) {
            acc.update(groups, cols, w)?;
        }
    }
    Ok(())
}

/// Folds a batch into the `avg` at `a` and the `sum` at `s` in one pass if
/// both would take their proof path on it; `false` (nothing folded)
/// otherwise.
fn fold_pair(
    accs: &mut [Accumulator],
    a: usize,
    s: usize,
    groups: &[u32],
    cols: &[ArrayRef],
    w: &Weights,
) -> Result<bool> {
    let Some((avg, sum)) = two_mut(accs, a, s) else {
        return Ok(false);
    };
    if !matches!(w, Weights::Const(1))
        || !(avg.proofs && sum.proofs)
        || avg.sum_bound != sum.sum_bound
    {
        return Ok(false);
    }
    // `fold_pairs` paired them on one and the same argument column.
    let Some(&at) = avg.expr.args.first() else {
        return Ok(false);
    };
    let col = arg(cols, at)?;
    if col.len() != groups.len() || col.logical_nulls().is_some_and(|n| n.null_count() > 0) {
        return Ok(false);
    }
    let max = avg.max_sum.max(sum.max_sum);
    let (
        Kind::AvgDec {
            sum: avg_sum,
            count,
            ..
        },
        Kind::SumDec {
            sum: sum_sum,
            state,
        },
    ) = (&mut avg.kind, &mut sum.kind)
    else {
        return Ok(false);
    };
    let folded = match decimals(col)? {
        Decimals::I64(v) => fold(
            avg.sum_bound,
            max,
            v,
            groups,
            avg_sum,
            count,
            sum_sum,
            state,
        ),
        Decimals::I128(v) => fold(
            avg.sum_bound,
            max,
            v,
            groups,
            avg_sum,
            count,
            sum_sum,
            state,
        ),
    };
    if folded {
        avg.max_sum = refresh_max(avg.max_sum, avg_sum, groups);
        sum.max_sum = refresh_max(sum.max_sum, sum_sum, groups);
    }
    Ok(folded)
}

/// The two folds' proof-path loops as one (`DecFold::sum` and `DecFold::avg`
/// on dense unit-weight rows), if the batch is proven for both sums.
#[expect(clippy::too_many_arguments, reason = "both states, split by field")]
fn fold<N: DecimalNative>(
    bound: Option<u128>,
    max_sum: u128,
    v: &[N],
    groups: &[u32],
    avg_sum: &mut [i128],
    count: &mut [i64],
    sum: &mut [i128],
    state: &mut [u8],
) -> bool {
    let Some(moves) = proven_moves(true, bound, max_sum, v, None, Some(&Weights::Const(1))) else {
        return false;
    };
    if avg_sum.len() <= SMALL_GROUPS && sum.len() == avg_sum.len() {
        debug_assert!(
            groups.iter().all(|&g| (g as usize) < avg_sum.len()),
            "a group id past the table"
        );
        let (acc, rows) = small_totals(groups, v, fits64_moves(moves));
        for (g, ((a, c), (s, st))) in avg_sum
            .iter_mut()
            .zip(count.iter_mut())
            .zip(sum.iter_mut().zip(state.iter_mut()))
            .enumerate()
        {
            *a = a.wrapping_add(acc[g]);
            *c = c.wrapping_add(rows[g]);
            // An overflowed sum stays NULL, as in `DecFold::sum`.
            if rows[g] > 0 && *st != 2 {
                *s = s.wrapping_add(acc[g]);
                *st = 1;
            }
        }
        return true;
    }
    for (&g, x) in groups.iter().zip(v) {
        let (g, x) = (g as usize, x.wide());
        avg_sum[g] = avg_sum[g].wrapping_add(x);
        count[g] = count[g].wrapping_add(1);
        // An overflowed sum stays NULL, as in `DecFold::sum`.
        if state[g] != 2 {
            sum[g] = sum[g].wrapping_add(x);
            state[g] = 1;
        }
    }
    true
}

/// Mutable references to `accs[a]` and `accs[s]` (`a != s`).
fn two_mut(
    accs: &mut [Accumulator],
    a: usize,
    s: usize,
) -> Option<(&mut Accumulator, &mut Accumulator)> {
    if a == s || a.max(s) >= accs.len() {
        return None;
    }
    let (lo, hi) = accs.split_at_mut(a.max(s));
    let (x, y) = (lo.get_mut(a.min(s))?, hi.first_mut()?);
    Some(if a < s { (x, y) } else { (y, x) })
}

/// [`fold_pair`] of the only pair of `accs`, for tests.
#[cfg(test)]
pub(super) fn fold_pair_for_test(
    accs: &mut [Accumulator],
    groups: &[u32],
    cols: &[ArrayRef],
    w: &Weights,
) -> bool {
    let exprs: Vec<AggregateExpr> = accs.iter().map(|a| a.expr.clone()).collect();
    let pairs = fold_pairs(&exprs);
    let Some(&(a, s)) = pairs.first() else {
        return false;
    };
    fold_pair(accs, a, s, groups, cols, w).is_ok_and(|done| done)
}
