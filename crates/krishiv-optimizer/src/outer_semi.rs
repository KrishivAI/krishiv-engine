// SPDX-License-Identifier: Apache-2.0
//! `SemiBelowOuterJoin`: a filtered relation that an inner join above an
//! outer join joins to the outer join's preserved side also reduces that side
//! below the outer join, as a key-only semi join (TPC-DS q40, q80).
//!
//! ```text
//! Path(A ⟕ B) ⋈[a = c] C   =>   Path((A ⋉[a = c'] Shift(π_c C)) ⟕ B) ⋈[a = c] C
//! ```
//!
//! The inner join keeps a row only when some `C` row equals it on every
//! `a = c`; a row of `A` with no such `C` row gives outer-join rows that carry
//! the same `a` (a preserved column is never NULL-extended), through filters,
//! projections passing `a` on, inner joins and the preserved sides of further
//! outer, semi and anti joins, and the inner join drops them all. So the semi
//! join removes only rows the inner join removes, whatever the multiplicity,
//! and the answer is the same. A NULL `a` matches nothing in either. The
//! semi join tests a subset of the inner join's conjuncts (its plain column
//! equalities), which every surviving row satisfies.
//!
//! Join ordering never moves `C` below the outer join, so without this the
//! outer join's preserved side keeps every row of `A` and the physical
//! planner builds the outer join's other side even when `C`'s filter leaves
//! few of `A`'s rows (q40: `catalog_sales ⟕ catalog_returns`, joined to
//! `item` and `date_dim` filtered to about 2% of the sales). With the semi
//! join the preserved side's estimate includes `C`'s filter, and P2-T166
//! orientation builds it.
//!
//! # When it applies
//!
//! - The inner join's condition has at least one plain `=` of a column of
//!   each side, of one type whose equality and grouping agree, the column on
//!   the path's side a column of `A`;
//! - the path from the inner join down to the outer join holds only
//!   deterministic filters, projections, and joins with deterministic
//!   conditions entered from a side whose rows they keep (inner either side;
//!   left outer, semi, anti and existence on the left; right outer on the
//!   right), each read by no other parent (rebuilding them would copy a
//!   shared subtree); `a` keeps its attribute id along it, so it is the same
//!   column;
//! - `C` is `restrict::repeatable` (computed twice, the semi join's instance
//!   under a [`PlanNode::Shift`] as in `restrict`);
//! - `A` holds no semi join on those columns already (the rule's own, or
//!   the one `LatePayloadJoin` leaves): the plan stays a fixed point;
//! - gates, from statistics only (never default row counts): the semi join
//!   is estimated to keep at most [`MAX_KEPT`] of `A`'s rows, and `C`'s
//!   scans read no more rows than `A` holds.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, JoinCondition, JoinKind, LogicalPlan, PlanNode,
};

use crate::dag;
use crate::estimate::Estimator;
use crate::expr::{
    Attrs, and_all, builtin, column, conjuncts, is_deterministic, recall, references,
};
use crate::late_payload::{
    bare_column, children, plain_column, scanned_rows, shared_nodes, with_child,
};
use crate::restrict::{MAX_ID, key_type, max_id, repeatable, shift};
use crate::stats::Statistics;

/// The largest estimated fraction of the preserved side's rows the semi join
/// may keep: a semi join that keeps more costs about what it saves.
pub(crate) const MAX_KEPT: f64 = 0.25;

/// The most semi joins added below one inner join.
const MAX_MOVES: usize = 8;

/// The most plan nodes one walk down from an inner join visits.
const MAX_WALK: usize = 256;

/// Adds key-only semi joins below outer joins, entering subquery plans. Also
/// returns one line per semi join added, and per refused candidate the gate
/// that refused it (`EXPLAIN VERBOSE`).
pub(crate) fn rewrite(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
) -> Result<(Arc<LogicalPlan>, Vec<String>), KrishivError> {
    let mut ctx = Context {
        next: max_id(&root, &mut crate::expr::AddrMap::default()).saturating_add(1),
        notes: Vec::new(),
        shared: shared_nodes(&root),
    };
    let mut est = Estimator::with_propagation(stats, propagate);
    let out = dag::rewrite_up(root, &mut |p| {
        let mut at = p;
        for _ in 0..MAX_MOVES {
            match at_join(&at, &mut est, &mut ctx) {
                Some(added) => at = added,
                None => break,
            }
        }
        Ok(at)
    })?;
    Ok((out, ctx.notes))
}

struct Context {
    /// The next free attribute id.
    next: u64,
    /// `EXPLAIN VERBOSE` lines, each refusal once.
    notes: Vec<String>,
    /// Nodes read by more than one parent in the input plan.
    shared: HashSet<*const LogicalPlan>,
}

fn note(notes: &mut Vec<String>, line: String) {
    if line.starts_with("added") || !notes.contains(&line) {
        notes.push(line);
    }
}

/// A plain column equality of the inner join: the path side's column `a`,
/// the other side's `c`, and the conjunct as written.
#[derive(Clone)]
struct KeyEq {
    a: BoundColumn,
    c: BoundColumn,
    e: Arc<Expr>,
}

/// One step of a path: a node and the index of the child the path goes to.
#[derive(Clone)]
struct Step {
    node: Arc<LogicalPlan>,
    child: usize,
}

/// An outer join below the inner join: the path from the inner join's child
/// down to it (top first, the outer join last) and its preserved side.
struct Candidate {
    path: Vec<Step>,
}

/// `join` with one semi join added below an outer join under it, if any
/// qualifies.
fn at_join(
    join: &Arc<LogicalPlan>,
    est: &mut Estimator<'_>,
    ctx: &mut Context,
) -> Option<Arc<LogicalPlan>> {
    let PlanNode::Join {
        kind: JoinKind::Inner,
        left,
        right,
        condition: JoinCondition::On(on),
    } = &join.node
    else {
        return None;
    };
    let conj = conjuncts(on);
    for (side, (x, c)) in [(0usize, (left, right)), (1, (right, left))] {
        if ctx.shared.contains(&Arc::as_ptr(x)) {
            continue;
        }
        let Some(eqs) = equalities(&conj, x, c) else {
            continue;
        };
        if !repeatable(c) {
            continue;
        }
        let mut found = Vec::new();
        let mut budget = MAX_WALK;
        walk(x, &mut Vec::new(), &mut found, &mut budget);
        for cand in found {
            if cand
                .path
                .iter()
                .any(|s| ctx.shared.contains(&Arc::as_ptr(&s.node)))
            {
                continue;
            }
            let Some(last) = cand.path.last() else {
                continue;
            };
            let kids = children(&last.node);
            let Some(preserved) = kids.get(last.child) else {
                continue;
            };
            let out = preserved.resolved_output().unwrap_or_default();
            let mine: Vec<KeyEq> = eqs
                .iter()
                .filter(|q| out.iter().any(|o| o.id == q.a.id))
                .cloned()
                .collect();
            if mine.is_empty() || !keeps_ids(&cand.path, &mine) {
                continue;
            }
            let ids: Attrs = mine.iter().map(|q| q.a.id).collect();
            if has_semi_on(preserved, &ids) {
                continue;
            }
            let Some(semi) = semi_join(preserved, c, &mine, &mut ctx.next) else {
                note(
                    &mut ctx.notes,
                    "not applied: the plan has no attribute ids left".to_owned(),
                );
                return None;
            };
            if let Err(why) = gates(preserved, &semi, c, est) {
                note(
                    &mut ctx.notes,
                    format!("not applied on [{}]: {why}", keys_text(&mine)),
                );
                continue;
            }
            // The path rebuilt bottom up over the reduced preserved side.
            let mut below = semi;
            for s in cand.path.iter().rev() {
                below = with_child(&s.node, s.child, below);
            }
            note(
                &mut ctx.notes,
                format!(
                    "added a semi join on [{}] below a {} join",
                    keys_text(&mine),
                    kind_text(&last.node)
                ),
            );
            return Some(with_child(join, side, below));
        }
    }
    None
}

/// The plain column equalities `(x, c)` of `conj` between `x`'s and `c`'s
/// outputs, of one key type; `None` when there are none.
fn equalities(
    conj: &[Arc<Expr>],
    x: &Arc<LogicalPlan>,
    c: &Arc<LogicalPlan>,
) -> Option<Vec<KeyEq>> {
    let xs = x.resolved_output()?;
    let cs = c.resolved_output()?;
    let in_x = |id: AttrId| xs.iter().any(|o| o.id == id);
    let in_c = |id: AttrId| cs.iter().any(|o| o.id == id);
    let mut eqs = Vec::new();
    for e in conj {
        let Some((Builtin::Eq, [a, b], _)) = builtin(e) else {
            continue;
        };
        let (Some(a), Some(b)) = (bare_column(a), bare_column(b)) else {
            continue;
        };
        if a.data_type != b.data_type || !key_type(&a.data_type) {
            continue;
        }
        if in_x(a.id) && in_c(b.id) {
            eqs.push(KeyEq {
                a: a.clone(),
                c: b.clone(),
                e: Arc::clone(e),
            });
        } else if in_c(a.id) && in_x(b.id) {
            eqs.push(KeyEq {
                a: b.clone(),
                c: a.clone(),
                e: Arc::clone(e),
            });
        }
    }
    (!eqs.is_empty()).then_some(eqs)
}

/// Every outer join reachable from `at` through the path nodes of the module
/// docs, with the path to it.
fn walk(
    at: &Arc<LogicalPlan>,
    path: &mut Vec<Step>,
    found: &mut Vec<Candidate>,
    budget: &mut usize,
) {
    if *budget == 0 {
        return;
    }
    *budget -= 1;
    if let PlanNode::Join {
        kind, condition, ..
    } = &at.node
    {
        let det = matches!(condition, JoinCondition::On(on) if is_deterministic(on));
        let preserved = match kind {
            JoinKind::LeftOuter => Some(0),
            JoinKind::RightOuter => Some(1),
            _ => None,
        };
        if let (true, Some(i)) = (det, preserved) {
            path.push(Step {
                node: Arc::clone(at),
                child: i,
            });
            found.push(Candidate { path: path.clone() });
            path.pop();
        }
    }
    let kids = children(at);
    for (i, kid) in kids.iter().enumerate() {
        if may_pass(at, i) {
            path.push(Step {
                node: Arc::clone(at),
                child: i,
            });
            walk(kid, path, found, budget);
            path.pop();
        }
    }
}

/// Whether the path may go down through `node` into child `i`: every row the
/// node outputs carries the child's values of the columns it passes on.
fn may_pass(node: &LogicalPlan, i: usize) -> bool {
    match &node.node {
        PlanNode::Project { exprs, .. } => i == 0 && exprs.iter().all(|e| is_deterministic(e)),
        PlanNode::Filter { predicate, .. } => i == 0 && is_deterministic(predicate),
        PlanNode::Join {
            kind, condition, ..
        } => {
            let det = matches!(condition, JoinCondition::On(on) if is_deterministic(on));
            det && match kind {
                JoinKind::Inner => true,
                JoinKind::LeftSemi
                | JoinKind::LeftAnti
                | JoinKind::LeftOuter
                | JoinKind::Existence(_) => i == 0,
                JoinKind::RightOuter => i == 1,
                _ => false,
            }
        }
        _ => false,
    }
}

/// Whether every projection on the path passes each `a` on as a bare column
/// (an id is defined once, so an unchanged id is the same column).
fn keeps_ids(path: &[Step], eqs: &[KeyEq]) -> bool {
    path.iter().all(|s| match &s.node.node {
        PlanNode::Project { exprs, .. } => eqs.iter().all(|q| {
            exprs
                .iter()
                .any(|e| plain_column(e) && bare_column(e).is_some_and(|b| b.id == q.a.id))
        }),
        _ => true,
    })
}

/// Whether `p`, through filters, projections and the left input of semi
/// joins, already holds a semi join whose left-side columns include `ids`.
fn has_semi_on(p: &Arc<LogicalPlan>, ids: &Attrs) -> bool {
    let mut at = Arc::clone(p);
    loop {
        let next = match &at.node {
            PlanNode::Filter { input, .. } | PlanNode::Project { input, .. } => Arc::clone(input),
            PlanNode::Join {
                kind: JoinKind::LeftSemi,
                left,
                condition: JoinCondition::On(on),
                ..
            } => {
                if ids.is_subset(&references(on)) {
                    return true;
                }
                Arc::clone(left)
            }
            _ => return false,
        };
        at = next;
    }
}

/// `A ⋉[a = c'] Shift(π_c C)`; `None` when attribute ids run out.
fn semi_join(
    a: &Arc<LogicalPlan>,
    c: &Arc<LogicalPlan>,
    eqs: &[KeyEq],
    next: &mut u64,
) -> Option<Arc<LogicalPlan>> {
    let offset = *next;
    let high = max_id(c, &mut crate::expr::AddrMap::default());
    *next = offset
        .checked_add(high)?
        .checked_add(1)
        .filter(|n| *n <= MAX_ID)?;
    let mut keys: Vec<BoundColumn> = Vec::new();
    for q in eqs {
        if !keys.iter().any(|x| x.id == q.c.id) {
            keys.push(q.c.clone());
        }
    }
    let shifted = Arc::new(LogicalPlan::new(PlanNode::Shift {
        input: Arc::new(LogicalPlan::new(PlanNode::Project {
            input: Arc::clone(c),
            exprs: keys.iter().map(column).collect(),
        })),
        offset,
    }));
    let cond = eqs
        .iter()
        .map(|q| {
            let (_, _, f) = builtin(&q.e)?;
            Some(recall(
                f,
                Builtin::Eq,
                vec![column(&q.a), shift(&column(&q.c), offset)],
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Arc::new(
        LogicalPlan::new(PlanNode::Join {
            left: Arc::clone(a),
            right: shifted,
            kind: JoinKind::LeftSemi,
            condition: JoinCondition::On(and_all(cond)?),
        })
        .with_origin(a.origin),
    ))
}

/// The statistics gates, or why the semi join is not worth it.
fn gates(
    a: &Arc<LogicalPlan>,
    semi: &Arc<LogicalPlan>,
    c: &Arc<LogicalPlan>,
    est: &mut Estimator<'_>,
) -> Result<(), String> {
    let (ae, ce) = (est.plan(a), est.plan(c));
    if !ae.from_statistics || !ce.from_statistics {
        return Err("the estimates rest on default row counts".to_owned());
    }
    let kept = est.plan(semi).rows;
    if kept > MAX_KEPT * ae.rows {
        return Err(format!(
            "the semi join keeps an estimated {kept:.0} of {:.0} rows, over {MAX_KEPT}",
            ae.rows
        ));
    }
    let scanned = scanned_rows(c, est);
    if scanned > ae.rows {
        return Err(format!(
            "the other side scans more rows ({scanned:.0}) than the preserved side holds ({:.0})",
            ae.rows
        ));
    }
    Ok(())
}

fn keys_text(eqs: &[KeyEq]) -> String {
    let v: Vec<String> = eqs
        .iter()
        .map(|q| format!("{} = {}", q.a.name, q.c.name))
        .collect();
    v.join(", ")
}

fn kind_text(p: &LogicalPlan) -> &'static str {
    match &p.node {
        PlanNode::Join {
            kind: JoinKind::RightOuter,
            ..
        } => "right outer",
        _ => "left outer",
    }
}
