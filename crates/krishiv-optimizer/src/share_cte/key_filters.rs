// SPDX-License-Identifier: Apache-2.0
//! Aggregate cores equal modulo null tests on their group keys and pruned
//! outputs (L4, TPC-DS q01).
//!
//! An aggregate grouped by plain columns gives every row of a group the same
//! key, so whether a key column is NULL is the same for the whole group:
//! `Agg(... Filter[isnotnull(k)](X) ...) = Filter[isnotnull(k')](Agg(... X ...))`,
//! `k'` the aggregate's output of `k`, when the path from the filter up to the
//! aggregate passes `k` on unchanged (filters, projections, inner joins and
//! the preserved sides of outer, semi and anti joins; a NULL-supplying side
//! could turn a key NULL). The null tests inferred from a join above one
//! instance (q01's `isnotnull(sr_customer_sk)`) are the usual case. Pruning
//! also drops the outputs one instance does not read (q01's average reads no
//! customer key), so instances of one aggregate may list different outputs.
//!
//! [`merge`] finds the one aggregate two such cores read: the unfiltered one,
//! with the outputs only the other lists added. The filtered side may add
//! only bare group keys, never a call, so the shared aggregate evaluates no
//! call on rows the query did not already evaluate it on, and raises nothing
//! the query would not raise.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_plan::{
    AttrId, Builtin, Expr, Grouping, JoinCondition, JoinKind, LogicalPlan, PlanNode,
};

use crate::dag;
use crate::expr::{and_all, builtin, column, conjuncts, output_of, strip_alias, substitute};
use crate::late_payload::{bare_column, with_child};

/// The most nodes one walk below an aggregate visits.
const MAX_WALK: usize = 256;

/// The aggregate two cores share, and the null tests each evaluates over it
/// (over the aggregate's output columns; empty for the unfiltered side).
pub(super) struct Merged {
    pub(super) def: Arc<LogicalPlan>,
    pub(super) tests_a: Vec<Arc<Expr>>,
    pub(super) tests_b: Vec<Arc<Expr>>,
}

/// The aggregate cores `a` and `b` both read, if they are plain `GROUP BY`
/// aggregates on equal keys whose inputs are equal once the null tests on
/// their group keys are removed, and one side's tests include the other's:
/// that side evaluates the rest of its tests over the shared aggregate.
pub(super) fn merge(a: &Arc<LogicalPlan>, b: &Arc<LogicalPlan>) -> Option<Merged> {
    let (ia, keys, aggs_a) = parts(a)?;
    let (ib, keys_b, aggs_b) = parts(b)?;
    if keys != keys_b {
        return None;
    }
    let (sa, ta) = strip(ia, keys);
    let (sb, tb) = strip(ib, keys);
    if !(Arc::ptr_eq(&sa, &sb) || *sa == *sb) {
        return None;
    }
    // The side with fewer tests is the shared aggregate, as written.
    let (a_filtered, rest) = if tb.iter().all(|t| ta.contains(t)) {
        (true, minus(&ta, &tb))
    } else if ta.iter().all(|t| tb.contains(t)) {
        (false, minus(&tb, &ta))
    } else {
        return None;
    };
    let filtered = !rest.is_empty();
    let (base, base_aggs, other_aggs) = if a_filtered && filtered {
        (b, aggs_b, aggs_a)
    } else {
        (a, aggs_a, aggs_b)
    };
    let mut extras: Vec<Arc<Expr>> = Vec::new();
    for e in other_aggs {
        if base_aggs.contains(e) {
            continue;
        }
        let id = output_of(e)?.id;
        // One id, two expressions: not copies of one aggregate.
        if base_aggs
            .iter()
            .any(|x| output_of(x).is_some_and(|o| o.id == id))
        {
            return None;
        }
        if filtered && !is_key_output(e, keys) {
            return None;
        }
        extras.push(Arc::clone(e));
    }
    let mut all = base_aggs.to_vec();
    all.extend(extras.iter().cloned());
    // The tests over the shared aggregate's outputs of their keys.
    let mut outputs: HashMap<AttrId, Arc<Expr>> = HashMap::new();
    for e in &all {
        if let (Some(c), Some(o)) = (bare_column(strip_alias(e)), output_of(e)) {
            outputs.entry(c.id).or_insert_with(|| column(o));
        }
    }
    let mut tests = Vec::with_capacity(rest.len());
    for t in &rest {
        let Some((_, [k], _)) = builtin(t) else {
            return None;
        };
        if !bare_column(k).is_some_and(|c| outputs.contains_key(&c.id)) {
            return None;
        }
        tests.push(substitute(t, &outputs));
    }
    let def = if extras.is_empty() {
        Arc::clone(base)
    } else {
        let mut p = LogicalPlan::clone(base);
        if let PlanNode::Aggregate { aggregates, .. } = &mut p.node {
            *aggregates = all;
        }
        Arc::new(p)
    };
    let (tests_a, tests_b) = match (filtered, a_filtered) {
        (false, _) => (Vec::new(), Vec::new()),
        (true, true) => (tests, Vec::new()),
        (true, false) => (Vec::new(), tests),
    };
    Some(Merged {
        def,
        tests_a,
        tests_b,
    })
}

/// The tests of `a` not in `b`.
fn minus(a: &[Arc<Expr>], b: &[Arc<Expr>]) -> Vec<Arc<Expr>> {
    a.iter().filter(|t| !b.contains(t)).cloned().collect()
}

/// A plain `GROUP BY` aggregate's input, keys and outputs.
type Parts<'a> = (&'a Arc<LogicalPlan>, &'a [Arc<Expr>], &'a [Arc<Expr>]);

/// The [`Parts`] of `p`, when it is a plain `GROUP BY` aggregate with keys.
fn parts(p: &LogicalPlan) -> Option<Parts<'_>> {
    match &p.node {
        PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            aggregates,
        } if !keys.is_empty() => Some((input, keys, aggregates)),
        _ => None,
    }
}

/// Whether the output entry `e` is a bare group key (aliased or not).
fn is_key_output(e: &Arc<Expr>, keys: &[Arc<Expr>]) -> bool {
    bare_column(strip_alias(e)).is_some_and(|c| {
        keys.iter()
            .any(|k| bare_column(k).is_some_and(|k| k.id == c.id))
    })
}

/// `input` without the null tests on its aggregate's bare group keys, and
/// those tests (each once, over the input's columns).
fn strip(input: &Arc<LogicalPlan>, keys: &[Arc<Expr>]) -> (Arc<LogicalPlan>, Vec<Arc<Expr>>) {
    let ids: HashSet<AttrId> = keys
        .iter()
        .filter_map(|k| bare_column(k).map(|c| c.id))
        .collect();
    let mut removed: Vec<Arc<Expr>> = Vec::new();
    let mut budget = MAX_WALK;
    let stripped = below(input, &ids, &mut removed, &mut budget);
    let mut once: Vec<Arc<Expr>> = Vec::new();
    for e in removed {
        if !once.contains(&e) {
            once.push(e);
        }
    }
    (stripped, once)
}

/// `p` with the null tests on the columns of `keys` removed, through the
/// nodes of the module docs; the removed tests go to `removed`.
fn below(
    p: &Arc<LogicalPlan>,
    keys: &HashSet<AttrId>,
    removed: &mut Vec<Arc<Expr>>,
    budget: &mut usize,
) -> Arc<LogicalPlan> {
    if *budget == 0 {
        return Arc::clone(p);
    }
    *budget -= 1;
    match &p.node {
        PlanNode::Filter { input, predicate } => {
            let (gone, kept): (Vec<_>, Vec<_>) = conjuncts(predicate)
                .into_iter()
                .partition(|c| null_test_of(c, keys));
            let inner = below(input, keys, removed, budget);
            if gone.is_empty() && Arc::ptr_eq(&inner, input) {
                return Arc::clone(p);
            }
            removed.extend(gone);
            match and_all(kept) {
                None => inner,
                Some(kept) => Arc::new(
                    LogicalPlan::new(PlanNode::Filter {
                        input: inner,
                        predicate: kept,
                    })
                    .with_origin(p.origin),
                ),
            }
        }
        PlanNode::Project { input, .. } => {
            let inner = below(input, keys, removed, budget);
            if Arc::ptr_eq(&inner, input) {
                Arc::clone(p)
            } else {
                dag::with_input(p, inner)
            }
        }
        PlanNode::Join {
            kind,
            left,
            right,
            condition: JoinCondition::On(_),
        } => {
            let sides: &[usize] = match kind {
                JoinKind::Inner => &[0, 1],
                JoinKind::LeftOuter
                | JoinKind::LeftSemi
                | JoinKind::LeftAnti
                | JoinKind::Existence(_) => &[0],
                JoinKind::RightOuter => &[1],
                _ => &[],
            };
            let mut at = Arc::clone(p);
            for &i in sides {
                let kid = if i == 0 { left } else { right };
                let inner = below(kid, keys, removed, budget);
                if !Arc::ptr_eq(&inner, kid) {
                    at = with_child(&at, i, inner);
                }
            }
            at
        }
        _ => Arc::clone(p),
    }
}

/// Whether `e` is `isnotnull(k)` or `isnull(k)` of a key column.
fn null_test_of(e: &Arc<Expr>, keys: &HashSet<AttrId>) -> bool {
    matches!(builtin(e), Some((Builtin::IsNotNull | Builtin::IsNull, [a], _))
        if bare_column(a).is_some_and(|c| keys.contains(&c.id)))
}
