// SPDX-License-Identifier: Apache-2.0
//! `ShareFilteredScan`: a table scan read only by filters, two or more, is
//! read once (perf-1 lever Q28-S, TPC-DS q28).
//!
//! ```text
//! ... Filter p1 (S) ... Filter p2 (S) ...  S one relation (one `Arc`)
//!   =>  WITH s := Filter (p1 or p2 ...) (S)
//!       ... Filter p1 (CteRef s) ... Filter p2 (CteRef s) ...
//! ```
//!
//! TPC-DS q28 cross-joins six global aggregates, each over its own filter on
//! one `store_sales` scan; the analyzer reads the table once (one `Arc`), but
//! each filter's pipeline scanned it again, so the query decoded the table six
//! times to keep 7% of its rows. `ShareCte` leaves such a scan alone (no
//! pipeline breaker: rescanning beats buffering the whole table), and the
//! filters differ, so no subtree above it is shared either. Here the shared
//! definition is the scan under the disjunction of its readers' predicates,
//! so only the rows some reader keeps are buffered, and each reader applies
//! its own predicate to them.
//!
//! Exact: `p_i` implies the disjunction, so `p_i` over the kept rows keeps
//! exactly the rows `p_i` keeps over the scan, in the same order within a
//! partition of the scan (a filter keeps its input's order).
//!
//! # When it applies
//!
//! - The relation is a [`PlanNode::Scan`] of a table or of files, read by two
//!   or more distinct parents (subquery plans included), every one of them a
//!   [`PlanNode::Filter`] directly over it, with at least two distinct
//!   predicates among them (equal ones are one chain read twice, which
//!   `ShareCte` leaves to rescanning);
//! - each predicate runs on any row (`runs_on_any_row`, ANSI): the
//!   disjunction evaluates every reader's predicate on every row, where the
//!   plan as written may never evaluate one (an inner join that skips its
//!   other side once one side is empty), so none may raise, be
//!   nondeterministic or hold a subquery; and none reads an outer reference;
//! - statistics say it pays: the scan's estimate comes from statistics, is at
//!   least [`MIN_ROWS`] rows, and the disjunction keeps at most [`MAX_KEPT`] of its rows (the buffer
//!   replaces scans, it should be small next to them).
//!
//! Runs once, after `ShareCte`, only with a session configuration and
//! `krishiv.optimizer.shareFilteredScans` on. The scan is then read by one
//! filter only (the definition's), so a second run finds nothing: the plan
//! stays a fixed point. Whether a reference really reads the buffer is
//! decided by the physical planner, as for every shared definition (copies
//! are kept when a runtime filter reaches one of them).

use crate::expr::{AddrMap, AddrSet};
use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{AttrId, Cte, CteRef, Expr, LogicalPlan, PlanNode, ReadSource, runs_on_any_row};

use crate::estimate::Estimator;
use crate::expr::{has_outer_reference, or_all};
use crate::restrict::subquery_plans;
use crate::share_cte::Sharer;
use crate::stats::Statistics;

/// The name prefix of the definitions this rule adds.
pub(crate) const SHARED_SCAN_PREFIX: &str = "__shared_scan_";

/// The fewest estimated rows of a scan worth storing instead of rescanning:
/// a small table's rescan costs less than the store's bookkeeping.
pub(crate) const MIN_ROWS: f64 = 1_000_000.0;

/// The largest fraction of the scan's estimated rows the disjunction may keep.
pub(crate) const MAX_KEPT: f64 = 0.25;

/// How a relation is read by one of its parents.
enum Parent {
    /// A filter directly over it.
    Filter(Arc<LogicalPlan>),
    /// Anything else: another operator, a definition, a subquery expression.
    Other,
}

/// `root` with each scan read only by filters read once.
pub(crate) fn share(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let (order, parents) = parents(&root);
    let mut est = Estimator::new(stats);
    let mut replace: AddrMap<LogicalPlan, Arc<LogicalPlan>> = Default::default();
    let mut ctes = Vec::new();
    for scan in &order {
        let PlanNode::Scan(s) = &scan.node else {
            continue;
        };
        if !matches!(
            s.source,
            ReadSource::Table { .. } | ReadSource::Files { .. }
        ) {
            continue;
        }
        let Some(readers) = parents.get(&Arc::as_ptr(scan)) else {
            continue;
        };
        let Some(filters) = filters_only(readers) else {
            continue;
        };
        if filters.len() < 2 {
            continue;
        }
        let predicates: Vec<_> = filters
            .iter()
            .filter_map(|f| match &f.node {
                PlanNode::Filter { predicate, .. } => Some(Arc::clone(predicate)),
                _ => None,
            })
            .collect();
        if predicates.len() != filters.len()
            || !predicates
                .iter()
                .all(|p| runs_on_any_row(p, true) && !has_outer_reference(p))
        {
            continue;
        }
        // Readers with equal predicates are one chain read twice, which
        // rescanning serves as well (`ShareCte` leaves it, keeping its
        // runtime filters): only distinct predicates make the store pay.
        let mut distinct: Vec<Arc<Expr>> = Vec::new();
        for p in &predicates {
            if !distinct.iter().any(|d| d == p) {
                distinct.push(Arc::clone(p));
            }
        }
        if distinct.len() < 2 {
            continue;
        }
        let Some(any) = or_all(distinct) else {
            continue;
        };
        let def = Arc::new(LogicalPlan::new(PlanNode::Filter {
            input: Arc::clone(scan),
            predicate: any,
        }));
        let whole = est.plan(scan);
        let kept = est.plan(&def);
        if !whole.from_statistics
            || whole.rows_for_costing() < MIN_ROWS
            || kept.rows > MAX_KEPT * whole.rows_for_costing()
        {
            continue;
        }
        let name = format!("{SHARED_SCAN_PREFIX}{}", ctes.len());
        let source: Vec<AttrId> = s.output.iter().map(|c| c.id).collect();
        let reference = Arc::new(LogicalPlan::new(PlanNode::CteRef(CteRef {
            name: name.clone(),
            source,
            output: s.output.clone(),
        })));
        for (f, predicate) in filters.iter().zip(predicates) {
            replace.insert(
                Arc::as_ptr(f),
                Arc::new(LogicalPlan::new(PlanNode::Filter {
                    input: Arc::clone(&reference),
                    predicate,
                })),
            );
        }
        ctes.push(Cte { name, plan: def });
    }
    if ctes.is_empty() {
        return Ok(root);
    }
    let mut sharer = Sharer {
        replace: &replace,
        done: Default::default(),
    };
    // Into the root's `WITH` when `ShareCte` added one: `InlineCTE` keeps the
    // generated definitions it finds there (a second optimization finds them
    // where the first left them). These read no definition, so they go first.
    if let PlanNode::WithCte {
        input,
        ctes: theirs,
    } = &root.node
    {
        let input = sharer.go(Arc::clone(input))?;
        let mut all = ctes;
        for c in theirs {
            all.push(Cte {
                name: c.name.clone(),
                plan: sharer.go(Arc::clone(&c.plan))?,
            });
        }
        return Ok(Arc::new(LogicalPlan::new(PlanNode::WithCte {
            input,
            ctes: all,
        })));
    }
    let input = sharer.go(root)?;
    Ok(Arc::new(LogicalPlan::new(PlanNode::WithCte {
        input,
        ctes,
    })))
}

/// The filters among `readers` when every reader is one, in a stable order.
fn filters_only(readers: &[Parent]) -> Option<Vec<Arc<LogicalPlan>>> {
    readers
        .iter()
        .map(|r| match r {
            Parent::Filter(f) => Some(Arc::clone(f)),
            Parent::Other => None,
        })
        .collect()
}

/// Every relation under `root` (subquery plans included) in a stable
/// pre-order, and for each one its distinct parents in the order first met.
fn parents(root: &Arc<LogicalPlan>) -> (Vec<Arc<LogicalPlan>>, AddrMap<LogicalPlan, Vec<Parent>>) {
    let mut order = Vec::new();
    let mut parents: AddrMap<LogicalPlan, Vec<Parent>> = Default::default();
    let mut edges: HashSet<(*const LogicalPlan, *const LogicalPlan)> = HashSet::new();
    let mut seen: AddrSet<LogicalPlan> = Default::default();
    let mut stack = vec![Arc::clone(root)];
    while let Some(p) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        let mut children: Vec<(Arc<LogicalPlan>, bool)> = Vec::new();
        p.for_each_child(&mut |c| children.push((Arc::clone(c), true)));
        p.for_each_expr(&mut |e| {
            children.extend(subquery_plans(e).into_iter().map(|c| (c, false)));
        });
        for (c, as_child) in children.iter().rev() {
            if edges.insert((Arc::as_ptr(&p), Arc::as_ptr(c))) {
                let filter_over = *as_child
                    && matches!(&p.node, PlanNode::Filter { input, .. } if Arc::ptr_eq(input, c));
                parents
                    .entry(Arc::as_ptr(c))
                    .or_default()
                    .push(if filter_over {
                        Parent::Filter(Arc::clone(&p))
                    } else {
                        Parent::Other
                    });
            }
            stack.push(Arc::clone(c));
        }
        order.push(p);
    }
    // The root itself is read by the query.
    parents
        .entry(Arc::as_ptr(root))
        .or_default()
        .push(Parent::Other);
    (order, parents)
}
