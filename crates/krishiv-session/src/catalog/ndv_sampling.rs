// SPDX-License-Identifier: Apache-2.0
//! Plan-time distinct-value samples of the file columns a query compares
//! with values (P2-T164 unit U6): which columns ([`compared_columns`]) and
//! the call into each opened [`FileTable`](krishiv_exec::planner::FileTable).

use std::sync::Arc;

use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode};

use super::{FileScanOf, QueryCatalog, run_on};

impl QueryCatalog {
    /// Samples the distinct values of the scanned file columns an equality or
    /// `IN` in `plan` compares with values (P2-T164 U6), so the optimizer
    /// reads a distinct count where the files store none. Off with
    /// `krishiv.optimizer.ndvSampling.enabled`; a failed or slow sample keeps
    /// the default estimates and never fails the query. The tables share one
    /// budget, `krishiv.optimizer.ndvSampling.timeout` per query: each gets
    /// what the tables before it left ([`Self::sample_notes`] reports it).
    pub(super) async fn sample_distinct(&self, plan: &LogicalPlan, scans: &[FileScanOf]) {
        use krishiv_common::config::{
            NDV_SAMPLING_ENABLED, NDV_SAMPLING_ROWS, NDV_SAMPLING_TIMEOUT,
        };
        if scans.is_empty() {
            return;
        }
        let config = &self.scans.config;
        if config
            .get(&NDV_SAMPLING_ENABLED)
            .and_then(krishiv_common::config::ConfigValue::as_bool)
            != Some(true)
        {
            return;
        }
        let mut options = krishiv_exec::planner::SampleOptions {
            rows: config
                .get(&NDV_SAMPLING_ROWS)
                .and_then(krishiv_common::config::ConfigValue::as_int)
                .and_then(|n| u64::try_from(n).ok())
                .unwrap_or(0),
            timeout: config
                .get(&NDV_SAMPLING_TIMEOUT)
                .and_then(krishiv_common::config::ConfigValue::as_duration)
                .unwrap_or_default(),
        };
        let deadline = std::time::Instant::now() + options.timeout;
        let mut compared = std::collections::HashSet::new();
        compared_columns(plan, &mut compared);
        if compared.is_empty() {
            return;
        }
        for (key, _, _, ids, _) in scans {
            let names: Vec<String> = ids
                .iter()
                .filter(|(id, _)| compared.contains(id))
                .map(|(_, name)| name.clone())
                .collect();
            let Some(table) = (!names.is_empty()).then(|| self.opened(key)).flatten() else {
                continue;
            };
            options.timeout = deadline.saturating_duration_since(std::time::Instant::now());
            let sampled = run_on(&self.scans.runtime, async move {
                table.sample_distinct(&names, options).await
            })
            .await;
            if let Err(e) = sampled {
                tracing::debug!("distinct-value sample skipped: {e}");
            }
        }
    }
}

impl QueryCatalog {
    /// One line per sampled table of the query, in sampling order:
    /// `<table>: <columns>: <what happened>` ([`SampleOutcome`]'s text, with
    /// the read's time when `timed`: `EXPLAIN VERBOSE`; without it for
    /// `EXPLAIN COST`, which then reads the same from run to run). A line
    /// repeated (a table scanned twice) is kept once. Empty when nothing was
    /// sampled. A read by path names its paths as `Display` prints them
    /// (credentials redacted).
    ///
    /// [`SampleOutcome`]: krishiv_exec::planner::SampleOutcome
    pub(crate) fn sample_notes(&self, timed: bool) -> Vec<String> {
        let opened = self
            .opened
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let mut notes = Vec::new();
        for (key, table) in opened {
            let name = match &key {
                super::FilesKey::View(n) | super::FilesKey::Table(n) => n.clone(),
                super::FilesKey::Read(krishiv_plan::ReadSource::Files {
                    format, paths, ..
                }) => {
                    let paths: Vec<String> = paths.iter().map(ToString::to_string).collect();
                    format!("{format}.`{}`", paths.join("`, `"))
                }
                super::FilesKey::Read(krishiv_plan::ReadSource::Table { name, .. }) => {
                    name.join(".")
                }
                super::FilesKey::Read(_) => "a read".to_owned(),
            };
            for outcome in table.sample_outcomes() {
                let text = if timed {
                    outcome.with_time()
                } else {
                    outcome.to_string()
                };
                let note = format!("{name}: {text}");
                if !notes.contains(&note) {
                    notes.push(note);
                }
            }
        }
        notes
    }
}

/// The attribute ids an equality or `IN` in a `Filter` condition of `plan`
/// (subquery plans included) compares with a deterministic expression that
/// reads no column: the columns whose distinct count decides the
/// comparison's selectivity (P2-T164 U6).
fn compared_columns(plan: &LogicalPlan, out: &mut std::collections::HashSet<krishiv_plan::AttrId>) {
    // Plans and expressions are DAGs: every node is visited once.
    walk(plan, out, &mut std::collections::HashSet::new());
}

/// [`compared_columns`] over the plan nodes not in `seen`.
fn walk(
    plan: &LogicalPlan,
    out: &mut std::collections::HashSet<krishiv_plan::AttrId>,
    seen: &mut std::collections::HashSet<*const LogicalPlan>,
) {
    if !seen.insert(std::ptr::from_ref(plan)) {
        return;
    }
    use krishiv_plan::{Builtin, FunctionId, ResolvedExpr, ResolvedFunction};
    fn column(e: &Expr) -> Option<krishiv_plan::AttrId> {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
            ExprNode::Cast { expr, .. } => column(expr),
            _ => None,
        }
    }
    /// A value the comparison is with: no column, no subquery, every call
    /// deterministic.
    fn is_value(e: &Arc<Expr>) -> bool {
        let mut stack = vec![Arc::clone(e)];
        let mut seen = std::collections::HashSet::new();
        while let Some(x) = stack.pop() {
            if !seen.insert(Arc::as_ptr(&x)) {
                continue;
            }
            match &x.node {
                ExprNode::Resolved(ResolvedExpr::Column(_) | ResolvedExpr::OuterColumn(_))
                | ExprNode::Subquery { .. } => return false,
                ExprNode::Resolved(ResolvedExpr::Function(f)) if !f.deterministic => return false,
                _ => x.for_each_child(&mut |c| stack.push(Arc::clone(c))),
            }
        }
        true
    }
    /// The compared columns of a filter's `predicate`.
    fn in_predicate(e: &Arc<Expr>, out: &mut std::collections::HashSet<krishiv_plan::AttrId>) {
        let mut stack = vec![Arc::clone(e)];
        let mut seen = std::collections::HashSet::new();
        while let Some(x) = stack.pop() {
            if !seen.insert(Arc::as_ptr(&x)) {
                continue;
            }
            if let ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                id: FunctionId::Builtin(b),
                args,
                ..
            })) = &x.node
            {
                match (b, args.as_slice()) {
                    (Builtin::Eq | Builtin::EqNullSafe, [a, c]) => {
                        for (col, value) in [(a, c), (c, a)] {
                            if let Some(id) = column(col).filter(|_| is_value(value)) {
                                out.insert(id);
                            }
                        }
                    }
                    (Builtin::In, [x, list @ ..]) if list.iter().all(is_value) => {
                        if let Some(id) = column(x) {
                            out.insert(id);
                        }
                    }
                    _ => {}
                }
            }
            x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
        }
    }
    /// Subquery plans inside any expression of a node.
    fn subqueries(
        e: &Arc<Expr>,
        out: &mut std::collections::HashSet<krishiv_plan::AttrId>,
        plans: &mut std::collections::HashSet<*const LogicalPlan>,
    ) {
        // Expressions are DAGs: a shared subtree is walked once.
        let mut stack = vec![Arc::clone(e)];
        let mut seen = std::collections::HashSet::new();
        while let Some(x) = stack.pop() {
            if !seen.insert(Arc::as_ptr(&x)) {
                continue;
            }
            if let ExprNode::Subquery { plan, .. } = &x.node {
                walk(plan, out, plans);
            }
            x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
        }
    }
    if let PlanNode::Filter { predicate, .. } = &plan.node {
        in_predicate(predicate, out);
    }
    plan.for_each_expr(&mut |e| subqueries(e, out, seen));
    plan.for_each_child(&mut |c| walk(c, out, seen));
}
