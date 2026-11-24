// SPDX-License-Identifier: Apache-2.0
//! `input_file_name()`, `input_file_block_start()` and `input_file_block_length()` (Spark's `InputFileName`,
//! `InputFileBlockStart`, `InputFileBlockLength`; goldens `tests/fixtures/spark-4.2.0/input-file-goldens.json`).
//!
//! Spark reads a per-thread value its file scan sets while it reads a block, so a call sees the file of the row
//! only while it runs in the scan's pipeline; elsewhere it gets the no-file values `''`, `-1`, `-1`. Krishiv
//! makes that explicit in two steps:
//!
//! - [`check`] is Spark's `PreReadCheck`, at analysis: an operator holding a call must not read more than one
//!   source (a file read counts 1, any other leaf 0, a union at most 1, anything else the sum of its inputs;
//!   subquery plans are checked on their own and never count for the query around them), else
//!   `MULTI_SOURCES_UNSUPPORTED_FOR_EXPRESSION`. It walks the unresolved plan, whose operators are Spark's
//!   analyzed ones: the resolved plan has rewritten `IN`/`EXISTS` subqueries and `INTERSECT` into joins.
//! - [`bind_input_files`] runs on the plan that is executed (after the session substituted cached relations, so
//!   a cached read has no file, as in Spark): each call becomes a read of the hidden `_metadata` column of the
//!   one file scan its operator's input reaches through operators that keep Spark's pipeline (projections,
//!   filters, aliases, generators, windows, every branch of a `UNION ALL`, the side of a join that holds the
//!   file); a call over anything else (an aggregate's output, a sort, a limit, a repartition, a distinct, a set
//!   operation, a scan with no file) is the no-file constant. Only existing nodes are produced.
//!
//! A join passes the file on only from a side it keeps whole (inner and cross: either; left outer, semi, anti,
//! existence and single: the left; right outer: the right; full outer: neither), so the three values are never
//! NULL, as in Spark, whose padded side is the broadcast build side (or the join is sort-merge).
//!
//! Known differences from Spark:
//! - Physical-plan dependent there: a sort-merge join (no broadcast) gives Spark's `''` on its probe side, and
//!   an inner join whose file side is the broadcast build side (chosen from statistics) gives `''` on every row;
//!   Krishiv gives the file in both. A window whose result is read gives Spark's `''` (Spark prunes an unread
//!   window, then the file shows, as here).
//! - Iceberg and Delta scans: counted as a source by the check, but a call over one gives the constants (their
//!   providers have no `_metadata` column; Spark's readers set the data file). Recorded divergence, tested by
//!   `krishiv-plan` `tests/input_file.rs` `versioned_table_scans_count_but_give_the_constants`.
//! - A table with a data column named `_metadata` hides the hidden column, so a call over its scan gives the
//!   constants; Spark's value does not depend on `_metadata` and is the file.
//! - EXPLAIN shows the rewrite (`regexp_replace(_metadata.file_path, ..) AS input_file_name()` over a scan with
//!   `_metadata`) where Spark prints `input_file_name()`; output names are the call's.
//! - A reference to a `WITH` definition that is nondeterministic after binding (`plan_is_deterministic`, the
//!   optimizer's own test: a nondeterministic call or Python relation) is a pipeline break (Krishiv computes such a
//!   definition once for all its references; Spark inlines it and shows the file). Any other is read through a
//!   fresh instance of the bound definition, as the optimizer inlines it.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{KrishivError, MultiSourcesUnsupportedForExpression};
use krishiv_types::DataType;

use super::bind::{Bindings, BoundTable, addr};
use super::{AnalysisError, FileMetadata};
use crate::{
    AttrId, BoundColumn, Expr, ExprNode, FunctionId, Grouping, JoinKind, Literal, LogicalPlan,
    Origin, Plan, PlanNode, ReadSource, ResolvedExpr, ResolvedFunction, Scan,
};

/// The functions, in the order of the columns [`bind_input_files`] provides.
const NAMES: [&str; 3] = [
    "input_file_name",
    "input_file_block_start",
    "input_file_block_length",
];

/// The function that prints `file_path` as Spark's `urlEncodedPath`.
const REPLACE: &str = "regexp_replace";

/// The `_metadata` field each function reads.
const FIELDS: [&str; 3] = ["file_path", "file_block_start", "file_block_length"];

/// Which of [`NAMES`] an unresolved call is (`builtin.` qualifiers included).
fn unresolved_call(e: &Expr) -> Option<usize> {
    let ExprNode::Function(f) = &e.node else {
        return None;
    };
    let builtin = match f.qualifier.as_slice() {
        [] => true,
        [b] => b.eq_ignore_ascii_case("builtin"),
        [s, b] => s.eq_ignore_ascii_case("system") && b.eq_ignore_ascii_case("builtin"),
        _ => false,
    };
    if !builtin || !f.args.is_empty() {
        return None;
    }
    NAMES.iter().position(|n| f.name.eq_ignore_ascii_case(n))
}

/// Which of [`NAMES`] a resolved call is.
fn resolved_call(e: &Expr) -> Option<usize> {
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
        return None;
    };
    match &f.id {
        FunctionId::Registry { name, .. } if f.args.is_empty() => {
            NAMES.iter().position(|n| name == n)
        }
        _ => None,
    }
}

/// The first call `is_call` finds in `e` (not inside subquery plans), and every subquery plan `e` holds.
fn calls_in<'a>(
    e: &'a Expr,
    is_call: fn(&Expr) -> Option<usize>,
    plans: &mut Vec<&'a Arc<LogicalPlan>>,
) -> Option<(usize, Origin)> {
    let mut found = None;
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        if !seen.insert(addr(x)) {
            continue;
        }
        if found.is_none()
            && let Some(k) = is_call(x)
        {
            found = Some((k, x.origin));
        }
        if let ExprNode::Subquery { plan, .. } = &x.node {
            plans.push(plan);
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
    found
}

/// Spark's `PreReadCheck` over the unresolved plan `root` (see the module docs), as Spark runs it after inlining:
/// a CTE reference counts its definition's sources, a SQL table function its body's, and an operator calling a
/// SQL function whose body calls one of the functions holds that call.
///
/// # Errors
/// `MULTI_SOURCES_UNSUPPORTED_FOR_EXPRESSION` at the call of an operator that reads more than one source.
pub(super) fn check(
    root: &Arc<LogicalPlan>,
    b: &Bindings,
    ctx: &super::AnalysisContext,
) -> Result<(), AnalysisError> {
    let mut c = Checker {
        b,
        ctx,
        seen: std::collections::HashSet::new(),
        memo: HashMap::new(),
        counting: std::collections::HashSet::new(),
        cuts: 0,
        kept: Vec::new(),
        bodies: std::collections::HashSet::new(),
    };
    c.walk(root)
}

/// The `WITH` definitions in scope, innermost last.
type Ctes = [(String, Arc<LogicalPlan>)];

/// The identity of a scope of [`Ctes`]: each definition's name and plan address, outermost first.
type ScopeKey = Vec<(String, usize)>;

fn scope_key(ctes: &Ctes) -> ScopeKey {
    ctes.iter().map(|(n, d)| (n.clone(), addr(&**d))).collect()
}

/// A scope of [`Ctes`] shared by the relations under it.
type Scope = Arc<Vec<(String, Arc<LogicalPlan>)>>;

struct Checker<'a> {
    b: &'a Bindings,
    ctx: &'a super::AnalysisContext,
    /// Relations walked, by (address, CTE scope depth).
    seen: std::collections::HashSet<(usize, usize)>,
    /// Source counts of relations, by (address, the CTE scope they were counted in): every reference to a
    /// definition counts it under the same scope, so chained CTEs that read the previous one twice stay linear.
    memo: HashMap<(usize, ScopeKey), u32>,
    /// The CTE definitions being counted (a recursive reference counts 0).
    counting: std::collections::HashSet<usize>,
    /// Recursive references cut so far: a count that cut one depends on what was being counted, so it is not kept.
    /// A count with no cut never depends on `counting`: a definition is counted in the scope before it, so no
    /// relation it reaches is counted while the definition itself is.
    cuts: u64,
    /// The SQL table function bodies counted: kept alive so that the memo's addresses of their plans stay theirs.
    kept: Vec<Arc<LogicalPlan>>,
    /// The SQL functions whose bodies were walked or are being searched, by routine name.
    bodies: std::collections::HashSet<String>,
}

impl Checker<'_> {
    /// Checks every relation under `root` (with an explicit stack: plans nest up to `MAX_DEPTH`).
    fn walk(&mut self, root: &Arc<LogicalPlan>) -> Result<(), AnalysisError> {
        let none: Scope = Arc::new(Vec::new());
        let mut stack = vec![(Arc::clone(root), Arc::clone(&none))];
        while let Some((p, ctes)) = stack.pop() {
            if !self.seen.insert((addr(&*p), ctes.len())) {
                continue;
            }
            let mut plans = Vec::new();
            let mut call = None;
            let mut exprs = Vec::new();
            p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
            for e in &exprs {
                let here = self.call_in(e, &mut plans, &mut std::collections::HashSet::new());
                call = call.or(here);
            }
            if let Some((k, origin)) = call
                && self.sources(&p, &ctes) > 1
            {
                return Err(AnalysisError::new(
                    MultiSourcesUnsupportedForExpression {
                        expr: format!("\"{}()\"", NAMES[k]),
                    },
                    origin,
                ));
            }
            let mut next: Vec<(Arc<LogicalPlan>, Scope)> = plans
                .into_iter()
                .map(|(plan, scoped)| (plan, Arc::clone(if scoped { &ctes } else { &none })))
                .collect();
            match &p.node {
                PlanNode::WithCte { input, ctes: defs } => {
                    let mut scope = (*ctes).clone();
                    for d in defs {
                        next.push((Arc::clone(&d.plan), Arc::new(scope.clone())));
                        scope.push((d.name.clone(), Arc::clone(&d.plan)));
                    }
                    next.push((Arc::clone(input), Arc::new(scope)));
                }
                _ => {
                    if let PlanNode::TableFunction { name, .. } = &p.node
                        && let Some(def) = super::bind::sql_table_function(self.ctx, name)
                        && self.bodies.insert(def.routine_name())
                    {
                        let mut body = Vec::new();
                        let _ = calls_in(&def.body, unresolved_call, &mut body);
                        next.extend(body.into_iter().map(|b| (Arc::clone(b), Arc::clone(&none))));
                    }
                    p.for_each_child(&mut |c| next.push((Arc::clone(c), Arc::clone(&ctes))));
                    if let Some(lowered) = self.b.sql.get(&addr(&*p)) {
                        next.push((Arc::clone(lowered), Arc::clone(&ctes)));
                    }
                }
            }
            stack.extend(next.into_iter().rev());
        }
        Ok(())
    }

    /// The first call `e` holds (directly, or through the body of a SQL function it calls), and the subquery
    /// plans to walk (`true`: in the current CTE scope; a function body's: in none).
    fn call_in(
        &mut self,
        e: &Expr,
        plans: &mut Vec<(Arc<LogicalPlan>, bool)>,
        open: &mut std::collections::HashSet<String>,
    ) -> Option<(usize, Origin)> {
        let mut found = None;
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![e];
        while let Some(x) = stack.pop() {
            if !seen.insert(addr(x)) {
                continue;
            }
            if found.is_none()
                && let Some(k) = unresolved_call(x)
            {
                found = Some((k, x.origin));
            }
            match &x.node {
                ExprNode::Subquery { plan, .. } => plans.push((Arc::clone(plan), true)),
                ExprNode::Function(f) => {
                    if let Some(def) = super::bind::sql_call(self.ctx, f) {
                        let name = def.routine_name();
                        if open.insert(name.clone()) {
                            let mut inner = Vec::new();
                            let held = self.call_in(&def.body, &mut inner, open);
                            open.remove(&name);
                            // The body's subquery plans are checked on their own, once.
                            if self.bodies.insert(name) {
                                plans.extend(inner.into_iter().map(|(p, _)| (p, false)));
                            }
                            if found.is_none() {
                                found = held.map(|(k, _)| (k, x.origin));
                            }
                        }
                    }
                }
                _ => {}
            }
            x.for_each_child(&mut |c| stack.push(c));
        }
        found
    }

    /// The sources Spark's check counts under `p` (`checkNumInputFileBlockSources`) after inlining.
    fn sources(&mut self, p: &LogicalPlan, ctes: &Ctes) -> u32 {
        let key = (addr(p), scope_key(ctes));
        if let Some(n) = self.memo.get(&key) {
            return *n;
        }
        let cuts = self.cuts;
        let n = match &p.node {
            PlanNode::Read(ReadSource::Table { name, .. }) => self.table(name, ctes),
            PlanNode::Read(ReadSource::Files { .. })
            | PlanNode::Scan(Scan {
                source: ReadSource::Files { .. },
                ..
            }) => 1,
            PlanNode::Sql { .. } => match self.b.sql.get(&addr(p)) {
                Some(l) => {
                    let l = Arc::clone(l);
                    self.sources(&l, ctes)
                }
                None => 0,
            },
            PlanNode::WithCte { input, ctes: defs } => {
                let mut scope = ctes.to_vec();
                scope.extend(defs.iter().map(|d| (d.name.clone(), Arc::clone(&d.plan))));
                self.sources(input, &scope)
            }
            PlanNode::TableFunction { name, .. } => {
                match super::bind::sql_table_function(self.ctx, name) {
                    Some(def) => {
                        let mut plans = Vec::new();
                        let _ = calls_in(&def.body, unresolved_call, &mut plans);
                        let plans: Vec<Arc<LogicalPlan>> = plans.into_iter().cloned().collect();
                        let n = plans
                            .iter()
                            .fold(0_u32, |n, q| n.saturating_add(self.sources(q, &[])));
                        self.kept.extend(plans);
                        n
                    }
                    None => 0,
                }
            }
            PlanNode::Union { inputs, .. } => inputs
                .iter()
                .fold(0_u32, |n, i| n.saturating_add(self.sources(i, ctes)))
                .min(1),
            _ => {
                let mut children = Vec::new();
                p.for_each_child(&mut |c| children.push(Arc::clone(c)));
                children
                    .iter()
                    .fold(0_u32, |n, c| n.saturating_add(self.sources(c, ctes)))
            }
        };
        if self.cuts == cuts {
            self.memo.insert(key, n);
        }
        n
    }

    /// The sources of a read of `name`: a CTE in scope counts its definition's (definitions shadow tables), a
    /// file table, or a versioned table format (Iceberg, Delta: Spark's run counts its V2 file scans), 1.
    fn table(&mut self, name: &[String], ctes: &Ctes) -> u32 {
        if let [one] = name
            && let Some(i) = ctes.iter().rposition(|(n, _)| {
                if self.ctx.conf.case_sensitive {
                    n == one
                } else {
                    n.eq_ignore_ascii_case(one)
                }
            })
        {
            let def = Arc::clone(&ctes[i].1);
            // A recursive reference reads the rows of the definition being counted.
            if !self.counting.insert(addr(&*def)) {
                self.cuts += 1;
                return 0;
            }
            let n = self.sources(&def, &ctes[..i]);
            self.counting.remove(&addr(&*def));
            return n;
        }
        match self.b.tables.get(name) {
            Some(BoundTable::Found { meta, .. }) => {
                u32::from(meta.file_metadata != FileMetadata::None || meta.version.is_some())
            }
            Some(BoundTable::Files(_)) => 1,
            Some(BoundTable::Missing) | None => 0,
        }
    }
}

/// `plan` with every `input_file_*` call bound to the file its rows come from (see the module docs), or `None`
/// when it holds no call. `metadata` says which `_metadata` column a scan's source can give
/// ([`FileMetadata::None`]: no file, such as a cached relation's snapshot).
///
/// # Errors
/// `FAILED_TO_PARSE_TOO_COMPLEX` when the plan's attribute ids leave no room for new ones (`first_free_id`), and
/// the depth check of the rebuilt plan (`Plan::try_new`). Ids are taken saturating after that: a plan within the
/// id limit has room for the three columns per scan.
pub fn bind_input_files(
    plan: &Plan,
    metadata: &dyn Fn(&Scan) -> FileMetadata,
) -> Result<Option<Plan>, KrishivError> {
    let root = plan.root();
    let mut any = false;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(root), |n| {
        if let crate::walk::Node::Expr(e) = n
            && resolved_call(e).is_some()
        {
            any = true;
            return std::ops::ControlFlow::Break(());
        }
        std::ops::ControlFlow::Continue(())
    });
    if !any {
        return Ok(None);
    }
    let next = super::resolve::first_free_id(root).map_err(|e| e.error)?;
    let mut defs = HashMap::new();
    let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(root), |n| {
        if let crate::walk::Node::Plan(p) = n
            && let PlanNode::WithCte { ctes, .. } = &p.node
        {
            for c in ctes {
                if let Some(out) = c.plan.resolved_output() {
                    defs.insert(
                        out.iter().map(|o| o.id).collect::<Vec<_>>(),
                        Arc::clone(&c.plan),
                    );
                }
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    let mut r = Binder {
        next,
        metadata,
        done: HashMap::new(),
        provided: HashMap::new(),
        defs,
        spans: HashMap::new(),
    };
    let out = r.rewrite(root);
    Plan::try_new(out).map(Some)
}

/// The three columns a widened relation adds: name, block start, block length.
type Provided = [BoundColumn; 3];

/// A relation widened to output the [`Provided`] columns, and those columns.
type Widened = Option<(Arc<LogicalPlan>, Provided)>;

struct Binder<'a> {
    next: u64,
    metadata: &'a dyn Fn(&Scan) -> FileMetadata,
    /// Rewritten relations by address (the original kept alive with it).
    done: HashMap<usize, (Arc<LogicalPlan>, Arc<LogicalPlan>)>,
    /// Widened relations by address of the (rewritten) relation.
    provided: HashMap<usize, (Arc<LogicalPlan>, Widened)>,
    /// The `WITH` definitions, by their output ids (a `CteRef`'s `source`).
    defs: HashMap<Vec<AttrId>, Arc<LogicalPlan>>,
    /// The id span of each widened definition (`first_free_id`), measured once.
    spans: HashMap<Vec<AttrId>, u64>,
}

fn col(c: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        c.clone(),
    ))))
}

fn alias(expr: Arc<Expr>, output: BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
        expr,
        output,
    })))
}

/// The no-file value of function `k`.
fn constant(k: usize) -> Expr {
    if k == 0 {
        Expr::literal(Literal::String(String::new()))
    } else {
        Expr::literal(Literal::Long(-1))
    }
}

fn data_type(k: usize) -> DataType {
    if k == 0 {
        DataType::String
    } else {
        DataType::Long
    }
}

/// Whether `p`'s own expressions (not its subquery plans) call one of the functions.
fn holds_call(p: &LogicalPlan) -> bool {
    let mut found = false;
    let mut plans = Vec::new();
    p.for_each_expr(&mut |e| {
        found |= calls_in(e, resolved_call, &mut plans).is_some();
    });
    found
}

/// `e` with origins cleared, for comparing a select-list expression with a grouping one.
fn plain(e: &Arc<Expr>) -> Arc<Expr> {
    Arc::clone(e)
        .transform_up(&mut |x| {
            if x.origin == Origin::UNKNOWN {
                return Ok::<_, std::convert::Infallible>(x);
            }
            let mut y = (*x).clone();
            y.origin = Origin::UNKNOWN;
            Ok(Arc::new(y))
        })
        .unwrap_or_else(|never| match never {})
}

/// `e` with each call read from `cols` (the no-file constant without them).
fn replace(e: Arc<Expr>, cols: Option<&Provided>) -> Arc<Expr> {
    e.transform_up(&mut |x| {
        let Some(k) = resolved_call(&x) else {
            return Ok::<_, std::convert::Infallible>(x);
        };
        let value = match cols {
            Some(c) => Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c[k].clone()))),
            None => constant(k),
        };
        Ok(Arc::new(value.with_origin(x.origin)))
    })
    .unwrap_or_else(|never| match never {})
}

impl Binder<'_> {
    fn fresh(&mut self, name: &str, data_type: DataType, nullable: bool) -> BoundColumn {
        let id = AttrId(self.next);
        self.next = self.next.saturating_add(1);
        BoundColumn {
            id,
            name: name.to_owned(),
            data_type,
            nullable,
        }
    }

    /// `p` with every call below bound, children first.
    fn rewrite(&mut self, p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        if let Some((_, out)) = self.done.get(&addr(&**p)) {
            return Arc::clone(out);
        }
        let rebuilt = Arc::clone(p)
            .map_children(|c| Ok::<_, std::convert::Infallible>(self.rewrite(&c)))
            .unwrap_or_else(|never| match never {});
        let rebuilt = rebuilt
            .map_expressions(|e| Ok::<_, std::convert::Infallible>(self.subqueries(e)))
            .unwrap_or_else(|never| match never {});
        let out = if holds_call(&rebuilt) {
            self.bind(rebuilt)
        } else {
            rebuilt
        };
        self.done
            .insert(addr(&**p), (Arc::clone(p), Arc::clone(&out)));
        out
    }

    /// `e` with its subquery plans rewritten (each on its own).
    fn subqueries(&mut self, e: Arc<Expr>) -> Arc<Expr> {
        e.transform_up(&mut |x| {
            let ExprNode::Subquery { kind, plan } = &x.node else {
                return Ok::<_, std::convert::Infallible>(x);
            };
            let next = self.rewrite(plan);
            if Arc::ptr_eq(&next, plan) {
                return Ok(x);
            }
            Ok(Arc::new(Expr {
                node: ExprNode::Subquery {
                    kind: kind.clone(),
                    plan: next,
                },
                origin: x.origin,
            }))
        })
        .unwrap_or_else(|never| match never {})
    }

    /// `x`, which holds calls, with them bound to its input's file (or the constants).
    fn bind(&mut self, x: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        let Some(output) = x.resolved_output() else {
            return Self::constants(x);
        };
        // An aggregate's output is evaluated over the groups: only a call in a key reads the input's file.
        if let PlanNode::Aggregate { grouping, .. } = &x.node {
            let keys = match grouping {
                Grouping::GroupBy(k) | Grouping::Rollup(k) | Grouping::Cube(k) => k,
                Grouping::GroupingSets { exprs, .. } => exprs,
            };
            let mut plans = Vec::new();
            if !keys
                .iter()
                .any(|k| calls_in(k, resolved_call, &mut plans).is_some())
            {
                return Self::constants(x);
            }
        }
        let Some((wide, cols)) = self.widen_input(&x) else {
            return Self::constants(x);
        };
        let bound = if let PlanNode::Aggregate {
            input,
            grouping,
            aggregates,
        } = &wide.node
        {
            // Keys evaluate on the input; the output over the groups, where only a key's value is the file.
            let keys: Vec<Arc<Expr>> = match grouping {
                Grouping::GroupBy(k) | Grouping::Rollup(k) | Grouping::Cube(k) => k.clone(),
                Grouping::GroupingSets { exprs, .. } => exprs.clone(),
            };
            let bound_keys: Vec<(Arc<Expr>, Arc<Expr>)> = keys
                .iter()
                .map(|k| (plain(k), replace(Arc::clone(k), Some(&cols))))
                .collect();
            let map_keys = |v: &[Arc<Expr>]| -> Vec<Arc<Expr>> {
                v.iter()
                    .map(|k| replace(Arc::clone(k), Some(&cols)))
                    .collect()
            };
            let grouping = match grouping {
                Grouping::GroupBy(k) => Grouping::GroupBy(map_keys(k)),
                Grouping::Rollup(k) => Grouping::Rollup(map_keys(k)),
                Grouping::Cube(k) => Grouping::Cube(map_keys(k)),
                Grouping::GroupingSets { exprs, sets } => Grouping::GroupingSets {
                    exprs: map_keys(exprs),
                    sets: sets.iter().map(|s| map_keys(s)).collect(),
                },
            };
            let aggregates = aggregates
                .iter()
                .map(|a| {
                    let keyed = Arc::clone(a)
                        .transform_down(&mut |e| {
                            let p = plain(&e);
                            Ok::<_, std::convert::Infallible>(
                                bound_keys
                                    .iter()
                                    .find(|(k, _)| *k == p)
                                    .map_or(e, |(_, b)| Arc::clone(b)),
                            )
                        })
                        .unwrap_or_else(|never| match never {});
                    replace(keyed, None)
                })
                .collect();
            Arc::new(
                LogicalPlan::new(PlanNode::Aggregate {
                    input: Arc::clone(input),
                    grouping,
                    aggregates,
                })
                .with_origin(wide.origin),
            )
        } else {
            wide.map_expressions(|e| Ok::<_, std::convert::Infallible>(replace(e, Some(&cols))))
                .unwrap_or_else(|never| match never {})
        };
        if matches!(
            bound.node,
            PlanNode::Project { .. } | PlanNode::Aggregate { .. }
        ) {
            return bound;
        }
        // The operator passes its input's columns on: keep its own output.
        Arc::new(LogicalPlan::new(PlanNode::Project {
            input: bound,
            exprs: output.iter().map(col).collect(),
        }))
    }

    /// `x` with every call the no-file constant.
    fn constants(x: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        x.map_expressions(|e| Ok::<_, std::convert::Infallible>(replace(e, None)))
            .unwrap_or_else(|never| match never {})
    }

    /// `x` over a widened input, and the columns its expressions read.
    fn widen_input(&mut self, x: &Arc<LogicalPlan>) -> Option<(Arc<LogicalPlan>, Provided)> {
        if let PlanNode::Join { .. } = &x.node {
            return self.join(x);
        }
        let mut children = Vec::new();
        x.for_each_child(&mut |c| children.push(Arc::clone(c)));
        let [child] = children.as_slice() else {
            return None;
        };
        let (wide, cols) = self.provide(child)?;
        let out = Arc::clone(x)
            .map_children(|_| Ok::<_, std::convert::Infallible>(Arc::clone(&wide)))
            .unwrap_or_else(|never| match never {});
        Some((out, cols))
    }

    /// `p` outputting the three columns too, when its rows come from one file scan in Spark's pipeline.
    fn provide(&mut self, p: &Arc<LogicalPlan>) -> Option<(Arc<LogicalPlan>, Provided)> {
        if let Some((_, out)) = self.provided.get(&addr(&**p)) {
            return out.clone();
        }
        let out = self.provide_new(p);
        self.provided
            .insert(addr(&**p), (Arc::clone(p), out.clone()));
        out
    }

    fn provide_new(&mut self, p: &Arc<LogicalPlan>) -> Option<(Arc<LogicalPlan>, Provided)> {
        match &p.node {
            PlanNode::Scan(s) => self.scan(p, s),
            PlanNode::Project { input, exprs } => {
                let (wide, cols) = self.provide(input)?;
                let mut exprs = exprs.clone();
                exprs.extend(cols.iter().map(col));
                let node = PlanNode::Project { input: wide, exprs };
                Some((Arc::new(LogicalPlan::new(node).with_origin(p.origin)), cols))
            }
            PlanNode::SubqueryAlias { columns, .. } if !columns.is_empty() => None,
            PlanNode::Filter { .. }
            | PlanNode::SubqueryAlias { .. }
            | PlanNode::Window { .. }
            | PlanNode::Generate { .. }
            | PlanNode::SecurityBarrier { .. }
            | PlanNode::JoinHint { .. }
            | PlanNode::Sample { .. }
            | PlanNode::CollectMetrics { .. } => self.widen_input(p),
            PlanNode::Join { .. } => self.join(p),
            PlanNode::Union {
                inputs,
                by_name,
                allow_missing_columns,
            } => self.union(p, inputs, *by_name, *allow_missing_columns),
            PlanNode::Shift { input, offset } => {
                let (wide, cols) = self.provide(input)?;
                let mut shifted = cols;
                for c in &mut shifted {
                    c.id = AttrId(c.id.0.checked_add(*offset)?);
                    self.next = self.next.max(c.id.0.saturating_add(1));
                }
                let node = PlanNode::Shift {
                    input: wide,
                    offset: *offset,
                };
                Some((
                    Arc::new(LogicalPlan::new(node).with_origin(p.origin)),
                    shifted,
                ))
            }
            PlanNode::CteRef(r) => self.cte_instance(r),
            // Pipeline breaks (aggregates, sorts, limits, exchanges, set operations) and leaves with no file.
            _ => None,
        }
    }

    /// A reference to a `WITH` definition, read through a fresh instance of the widened definition, as the
    /// optimizer's `InlineCTE` instantiates it (`Project[source + offset AS output]` over `Shift`): Spark inlines
    /// the definition, so its scan is in the reference's pipeline. A definition with a nondeterministic call stays
    /// shared (computed once for all its references), so a reference to it is a break.
    fn cte_instance(&mut self, r: &crate::CteRef) -> Option<(Arc<LogicalPlan>, Provided)> {
        let def = Arc::clone(self.defs.get(&r.source)?);
        // Tested after the rewrite, as the optimizer tests what it inlines: a definition that calls
        // `input_file_name()` is deterministic once bound.
        let def = self.rewrite(&def);
        if !crate::plan_is_deterministic(&def) {
            return None;
        }
        let (wide, cols) = self.provide(&def)?;
        let span = match self.spans.get(&r.source) {
            Some(n) => *n,
            None => {
                let n = super::resolve::first_free_id(&wide).ok()?;
                self.spans.insert(r.source.clone(), n);
                n
            }
        };
        let offset = self.next;
        self.next = offset.checked_add(span)?.checked_add(1)?;
        let moved = |c: &BoundColumn, as_: &BoundColumn| -> Option<Arc<Expr>> {
            let inner = BoundColumn {
                id: AttrId(c.id.0.checked_add(offset)?),
                ..as_.clone()
            };
            Some(col(&inner))
        };
        let mut exprs = Vec::with_capacity(r.output.len() + 3);
        for (s, o) in r.source.iter().zip(&r.output) {
            let at = BoundColumn {
                id: *s,
                ..o.clone()
            };
            exprs.push(alias(moved(&at, o)?, o.clone()));
        }
        let mut out = cols.clone();
        for (c, o) in cols.iter().zip(&mut out) {
            o.id = self.fresh(&c.name, c.data_type.clone(), c.nullable).id;
            exprs.push(alias(moved(c, c)?, o.clone()));
        }
        let shifted = Arc::new(LogicalPlan::new(PlanNode::Shift {
            input: wide,
            offset,
        }));
        let node = PlanNode::Project {
            input: shifted,
            exprs,
        };
        Some((Arc::new(LogicalPlan::new(node)), out))
    }

    /// A join over the widened side that holds the file.
    fn join(&mut self, p: &Arc<LogicalPlan>) -> Option<(Arc<LogicalPlan>, Provided)> {
        let PlanNode::Join {
            left,
            right,
            kind,
            condition,
        } = &p.node
        else {
            return None;
        };
        // Only a side the join keeps whole (no row of it padded with NULLs, none of its columns nullable): the
        // three functions never return NULL. In Spark's default plans the padded side is the broadcast build side,
        // or the join is sort-merge, so its rows get the no-file values; a full outer join gives them on both.
        let (from_left, from_right) = match kind {
            JoinKind::Inner | JoinKind::Cross => (true, true),
            JoinKind::RightOuter => (false, true),
            JoinKind::FullOuter => (false, false),
            _ => (true, false),
        };
        let widened_left = if from_left { self.provide(left) } else { None };
        let (left, right, cols) = if let Some((l, c)) = widened_left {
            (l, Arc::clone(right), c)
        } else if from_right && let Some((r, c)) = self.provide(right) {
            (Arc::clone(left), r, c)
        } else {
            return None;
        };
        let node = PlanNode::Join {
            left,
            right,
            kind: *kind,
            condition: condition.clone(),
        };
        Some((Arc::new(LogicalPlan::new(node).with_origin(p.origin)), cols))
    }

    /// A union of widened inputs; an input with no file gives the constants.
    fn union(
        &mut self,
        p: &Arc<LogicalPlan>,
        inputs: &[Arc<LogicalPlan>],
        by_name: bool,
        allow_missing_columns: bool,
    ) -> Option<(Arc<LogicalPlan>, Provided)> {
        let widened: Vec<Option<(Arc<LogicalPlan>, Provided)>> =
            inputs.iter().map(|i| self.provide(i)).collect();
        if widened.iter().all(Option::is_none) {
            return None;
        }
        let mut out = Vec::with_capacity(inputs.len());
        let mut first: Option<Provided> = None;
        for (input, w) in inputs.iter().zip(widened) {
            let (plan, cols) = match w {
                Some(w) => w,
                None => {
                    let output = input.resolved_output()?;
                    let cols: Provided =
                        [0, 1, 2].map(|k| self.fresh(NAMES[k], data_type(k), false));
                    let mut exprs: Vec<Arc<Expr>> = output.iter().map(col).collect();
                    exprs.extend(
                        cols.iter()
                            .enumerate()
                            .map(|(k, c)| alias(Arc::new(constant(k)), c.clone())),
                    );
                    let node = PlanNode::Project {
                        input: Arc::clone(input),
                        exprs,
                    };
                    (Arc::new(LogicalPlan::new(node)), cols)
                }
            };
            first.get_or_insert(cols);
            out.push(plan);
        }
        let node = PlanNode::Union {
            inputs: out,
            by_name,
            allow_missing_columns,
        };
        Some((
            Arc::new(LogicalPlan::new(node).with_origin(p.origin)),
            first?,
        ))
    }

    /// A file scan with its `_metadata` column, under a projection computing the three values.
    fn scan(&mut self, p: &Arc<LogicalPlan>, s: &Scan) -> Option<(Arc<LogicalPlan>, Provided)> {
        let kind = (self.metadata)(s);
        if kind == FileMetadata::None {
            return None;
        }
        let mut scan = s.clone();
        let meta = if let Some(m) = s.metadata_column() {
            m.clone()
        } else {
            // A table column of the name hides the hidden column.
            if s.output
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(super::ops::METADATA_COLUMN))
            {
                return None;
            }
            let m = self.fresh(
                super::ops::METADATA_COLUMN,
                super::ops::metadata_type(kind == FileMetadata::WithRowIndex),
                false,
            );
            scan.output.push(m.clone());
            m
        };
        let DataType::Struct(st) = &meta.data_type else {
            return None;
        };
        let field = |k: usize| -> Option<Arc<Expr>> {
            let ordinal = st.fields().iter().position(|f| f.name() == FIELDS[k])?;
            Some(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Field {
                    base: col(&meta),
                    ordinal,
                    name: FIELDS[k].to_owned(),
                    data_type: data_type(k),
                    nullable: false,
                },
            ))))
        };
        // Spark prints a local file as `file:///x` (`urlEncodedPath`) where `_metadata.file_path` is `file:/x`.
        // One match of a compiled literal pattern per row; doing it once per file would need the scan to produce
        // the value (krishiv-exec), left as a follow-up.
        let path = field(0)?;
        let krishiv_functions::registry::Lookup::Found(def) =
            krishiv_functions::registry::builtins().lookup(REPLACE, super::SparkProfile::default())
        else {
            return None;
        };
        let name = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Registry {
                    name: def.name.to_owned(),
                    id: def.id,
                },
                kind: def.kind,
                args: vec![
                    path,
                    Arc::new(Expr::literal(Literal::String("^file:/([^/])".to_owned()))),
                    Arc::new(Expr::literal(Literal::String("file:///$1".to_owned()))),
                ],
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::String,
                nullable: false,
                deterministic: def.deterministic,
            },
        ))));
        let values = [Some(name), field(1), field(2)];
        let cols: Provided = [0, 1, 2].map(|k| self.fresh(NAMES[k], data_type(k), false));
        let mut exprs: Vec<Arc<Expr>> = s.output.iter().map(col).collect();
        for (v, c) in values.into_iter().zip(&cols) {
            exprs.push(alias(v?, c.clone()));
        }
        let scan = Arc::new(LogicalPlan::new(PlanNode::Scan(scan)).with_origin(p.origin));
        let node = PlanNode::Project { input: scan, exprs };
        Some((Arc::new(LogicalPlan::new(node)), cols))
    }
}
