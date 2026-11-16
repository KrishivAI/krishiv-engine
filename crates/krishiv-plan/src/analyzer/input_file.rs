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
//! Known differences from Spark, all physical-plan dependent there: a sort-merge join (no broadcast) gives
//! Spark's `''` on its probe side where Krishiv gives the file; a window whose result is read gives Spark's `''`
//! (Spark prunes an unread window, then the file shows, as here); a call over an Iceberg or Delta scan is counted
//! as a source but gives the constants (those scans have no `_metadata` column).

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{KrishivError, MultiSourcesUnsupportedForExpression};
use krishiv_types::DataType;

use super::bind::{Bindings, BoundTable, addr};
use super::{AnalysisError, FileMetadata};
use crate::{
    AttrId, BoundColumn, Expr, ExprNode, FunctionId, FunctionKind, Grouping, JoinKind, Literal,
    LogicalPlan, Origin, Plan, PlanNode, ReadSource, ResolvedExpr, ResolvedFunction, Scan,
};

/// The functions, in the order of the columns [`bind_input_files`] provides.
const NAMES: [&str; 3] = [
    "input_file_name",
    "input_file_block_start",
    "input_file_block_length",
];

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

/// Spark's `PreReadCheck` over the unresolved plan `root` (see the module docs).
///
/// # Errors
/// `MULTI_SOURCES_UNSUPPORTED_FOR_EXPRESSION` at the call of an operator that reads more than one source.
pub(super) fn check(root: &Arc<LogicalPlan>, b: &Bindings) -> Result<(), AnalysisError> {
    let mut seen = std::collections::HashSet::new();
    let mut counts: HashMap<usize, u32> = HashMap::new();
    let mut stack = vec![Arc::clone(root)];
    while let Some(p) = stack.pop() {
        if !seen.insert(addr(&*p)) {
            continue;
        }
        let mut plans = Vec::new();
        let mut call = None;
        p.for_each_expr(&mut |e| {
            let here = calls_in(e, unresolved_call, &mut plans);
            call = call.or(here);
        });
        if let Some((k, origin)) = call
            && sources(&p, b, &mut counts) > 1
        {
            return Err(AnalysisError::new(
                MultiSourcesUnsupportedForExpression {
                    expr: format!("\"{}()\"", NAMES[k]),
                },
                origin,
            ));
        }
        stack.extend(plans.into_iter().cloned());
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
        if let Some(lowered) = b.sql.get(&addr(&*p)) {
            stack.push(Arc::clone(lowered));
        }
    }
    Ok(())
}

/// The sources Spark's check counts under `p` (`checkNumInputFileBlockSources`).
fn sources(p: &LogicalPlan, b: &Bindings, memo: &mut HashMap<usize, u32>) -> u32 {
    if let Some(n) = memo.get(&addr(p)) {
        return *n;
    }
    let n = match &p.node {
        PlanNode::Read(ReadSource::Table { name, .. }) => match b.tables.get(name) {
            // A file table, or a versioned table format (Iceberg, Delta: Spark's run counts its V2 file scans).
            Some(BoundTable::Found { meta, .. }) => {
                u32::from(meta.file_metadata != FileMetadata::None || meta.version.is_some())
            }
            Some(BoundTable::Files(_)) => 1,
            Some(BoundTable::Missing) | None => 0,
        },
        PlanNode::Read(ReadSource::Files { .. })
        | PlanNode::Scan(Scan {
            source: ReadSource::Files { .. },
            ..
        }) => 1,
        PlanNode::Sql { .. } => b.sql.get(&addr(p)).map_or(0, |l| sources(l, b, memo)),
        PlanNode::Union { inputs, .. } => inputs
            .iter()
            .fold(0_u32, |n, i| n.saturating_add(sources(i, b, memo)))
            .min(1),
        _ => {
            let mut children = Vec::new();
            p.for_each_child(&mut |c| children.push(Arc::clone(c)));
            children
                .iter()
                .fold(0_u32, |n, c| n.saturating_add(sources(c, b, memo)))
        }
    };
    memo.insert(addr(p), n);
    n
}

/// `plan` with every `input_file_*` call bound to the file its rows come from (see the module docs), or `None`
/// when it holds no call. `metadata` says which `_metadata` column a scan's source can give
/// ([`FileMetadata::None`]: no file, such as a cached relation's snapshot).
///
/// # Errors
/// `FAILED_TO_PARSE_TOO_COMPLEX` when attribute ids run out; the depth check of a rebuilt plan.
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
    let mut r = Binder {
        next,
        metadata,
        done: HashMap::new(),
        provided: HashMap::new(),
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
            // Pipeline breaks (aggregates, sorts, limits, exchanges, set operations) and leaves with no file.
            _ => None,
        }
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
        let both = matches!(
            kind,
            JoinKind::Inner
                | JoinKind::Cross
                | JoinKind::LeftOuter
                | JoinKind::RightOuter
                | JoinKind::FullOuter
        );
        let (left, right, mut cols, nullable) = if let Some((l, c)) = self.provide(left) {
            let nullable = matches!(kind, JoinKind::RightOuter | JoinKind::FullOuter);
            (l, Arc::clone(right), c, nullable)
        } else if both && let Some((r, c)) = self.provide(right) {
            let nullable = matches!(kind, JoinKind::LeftOuter | JoinKind::FullOuter);
            (Arc::clone(left), r, c, nullable)
        } else {
            return None;
        };
        if nullable {
            for c in &mut cols {
                c.nullable = true;
            }
        }
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
        let path = field(0)?;
        let name = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Registry {
                    name: "regexp_replace".to_owned(),
                    id: 0,
                },
                kind: FunctionKind::Scalar,
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
                deterministic: true,
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
