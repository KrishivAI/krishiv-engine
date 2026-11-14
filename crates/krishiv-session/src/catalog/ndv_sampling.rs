// SPDX-License-Identifier: Apache-2.0
//! Plan-time distinct-value samples of the file columns a query compares
//! with values (P2-T164 unit U6), and of its group keys while
//! `krishiv.optimizer.groupKeyCode` is on (design P2b D2): which columns
//! ([`compared_columns`], [`grouped_columns`]) and the call into each opened
//! [`FileTable`](krishiv_exec::planner::FileTable).

use std::sync::Arc;

use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode};

use krishiv_exec::planner::TableProvider as _;

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
        // `GroupKeyCode` (design P2b D2) codes an aggregate's dimension keys
        // only on known distinct counts: while it is on, the group keys it
        // could code are sampled too.
        if group_key_code(config) {
            let rows = |id: krishiv_plan::AttrId| {
                let scan = scans.iter().find(|s| s.3.iter().any(|(i, _)| *i == id))?;
                Some(self.opened(&scan.0)?.statistics()?.rows)
            };
            for g in grouped_columns(plan) {
                if dimension_may_pay(rows(g.scan), g.others.iter().map(|id| rows(*id))) {
                    compared.extend(g.keys);
                }
            }
        }
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
                // A reference with read options of its own is named by its table, as a
                // reference without options is; the options never reach the text.
                super::FilesKey::View(n)
                | super::FilesKey::Table(n)
                | super::FilesKey::Reference { key: n, .. } => n.clone(),
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

/// Whether the optimizer may code group keys (`krishiv.optimizer.groupKeyCode`,
/// or `krishiv.experimental.groupKeyCodeForce`, which also switches it on).
fn group_key_code(config: &krishiv_common::config::Config) -> bool {
    use krishiv_common::config::{ConfigValue, GROUP_KEY_CODE};
    config.get(&GROUP_KEY_CODE).and_then(ConfigValue::as_bool) == Some(true)
        || krishiv_common::experimental::is_enabled(config, "groupKeyCodeForce")
}

/// The column group keys of one aggregate that come from one scan below it
/// and that `GroupKeyCode` could code by their types
/// ([`krishiv_optimizer::could_code_group_keys`]).
struct GroupKeys {
    /// The keys.
    keys: Vec<krishiv_plan::AttrId>,
    /// A column of their scan.
    scan: krishiv_plan::AttrId,
    /// A column of each other scan below the aggregate.
    others: Vec<krishiv_plan::AttrId>,
}

/// The rule's `rows(I) >= 2 rows(D)` test on the tables' row counts, the
/// largest other table standing for the aggregate's input `I`: `true` unless
/// every count is known and the dimension is too large. An unknown count
/// keeps the sample.
fn dimension_may_pay(
    dimension: Option<u64>,
    mut others: impl Iterator<Item = Option<u64>>,
) -> bool {
    let largest = others.try_fold(0_u64, |m, r| r.map(|r| m.max(r)));
    match (dimension, largest) {
        (Some(d), Some(i)) => i >= d.saturating_mul(2),
        _ => true,
    }
}

/// Per `GROUP BY` aggregate of `plan` (subquery plans included) with two or
/// more scans below it, the column group keys of each scan that
/// `GroupKeyCode` could code: the columns whose distinct counts its
/// statistics test reads. An aggregate over one table (TPC-H q1) has none.
fn grouped_columns(plan: &LogicalPlan) -> Vec<GroupKeys> {
    // Plans are DAGs, subquery plans included: every node is visited once.
    let mut seen = std::collections::HashSet::new();
    let mut stack: Vec<Arc<LogicalPlan>> = Vec::new();
    let mut out = Vec::new();
    let mut visit = |p: &LogicalPlan, stack: &mut Vec<Arc<LogicalPlan>>| {
        if !seen.insert(std::ptr::from_ref(p)) {
            return;
        }
        if let PlanNode::Aggregate {
            input,
            grouping: krishiv_plan::Grouping::GroupBy(keys),
            ..
        } = &p.node
        {
            aggregate_keys(input, keys, &mut out);
        }
        p.for_each_expr(&mut |e| subquery_plans(e, stack));
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    };
    visit(plan, &mut stack);
    while let Some(p) = stack.pop() {
        visit(&p, &mut stack);
    }
    out
}

/// [`grouped_columns`] of one aggregate over `input` grouped by `keys`.
fn aggregate_keys(input: &LogicalPlan, keys: &[Arc<Expr>], out: &mut Vec<GroupKeys>) {
    let scans = scans_below(input);
    if scans.len() < 2 {
        return;
    }
    let keys: Vec<_> = keys.iter().filter_map(|k| column_of(k)).collect();
    for (n, scan) in scans.iter().enumerate() {
        let (ids, types): (Vec<_>, Vec<_>) = scan
            .output
            .iter()
            .filter(|c| keys.contains(&c.id))
            .map(|c| (c.id, &c.data_type))
            .unzip();
        let Some(first) = scan.output.first() else {
            continue;
        };
        if ids.is_empty() || !krishiv_optimizer::could_code_group_keys(types) {
            continue;
        }
        out.push(GroupKeys {
            keys: ids,
            scan: first.id,
            others: scans
                .iter()
                .enumerate()
                .filter(|(m, _)| *m != n)
                .filter_map(|(_, s)| s.output.first().map(|c| c.id))
                .collect(),
        });
    }
}

/// The scans in `plan`, subquery plans left out (each is an aggregate's
/// input of its own), each shared node once.
fn scans_below(plan: &LogicalPlan) -> Vec<&krishiv_plan::Scan> {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![plan];
    let mut out = Vec::new();
    while let Some(p) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(p)) {
            continue;
        }
        if let PlanNode::Scan(scan) = &p.node {
            out.push(scan);
        }
        p.for_each_child(&mut |c| stack.push(&**c));
    }
    out
}

/// The column `e` is, under casts.
fn column_of(e: &Expr) -> Option<krishiv_plan::AttrId> {
    use krishiv_plan::ResolvedExpr;
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        ExprNode::Cast { expr, .. } => column_of(expr),
        _ => None,
    }
}

/// The plans of the subqueries inside `e`.
fn subquery_plans(e: &Arc<Expr>, out: &mut Vec<Arc<LogicalPlan>>) {
    let mut stack = vec![Arc::clone(e)];
    let mut seen = std::collections::HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&x)) {
            continue;
        }
        if let ExprNode::Subquery { plan, .. } = &x.node {
            out.push(Arc::clone(plan));
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
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
