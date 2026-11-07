// SPDX-License-Identifier: Apache-2.0
//! Python UDF calls planned as `PythonEval` nodes (design P2-T059 §2.2,
//! Spark's `ExtractPythonUDFs`).
//!
//! A node holding Python UDF calls (a projection, a filter, an aggregate's
//! keys and arguments, an inner join's condition) is planned over a
//! [`PhysNode::PythonEval`] that evaluates every call over **all** its input
//! rows and appends one column per call; the node then reads those columns.
//! So a call is never short-circuited inside `CASE`, `AND` or `OR`: a UDF
//! that raises on rows of a branch not taken still raises, as in Spark.
//!
//! The calls of one eval type whose arguments hold no other Python UDF call
//! share a node (equal calls are evaluated once); a call over another
//! call's result, or calls of another eval type, go in a node above.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::{KrishivError, UnsupportedFeaturePythonUdfInOnClause};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, JoinCondition,
    JoinKind, LogicalPlan, PythonEvalType, ResolvedExpr, ResolvedFunction,
};
use krishiv_types::DataType;

use super::build::{Planned, Planner, has_subquery, internal};
use super::explain::ExplainNode;
use super::runtime_filter::kept;
use crate::error::Result;
use crate::physical::{PhysNode, PythonCallPlan, compile_all};
use crate::python_eval::output_schema;

/// The eval type of `e` if it is a Python UDF call.
fn python_call(e: &Expr) -> Option<PythonEvalType> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => match &f.id {
            FunctionId::PythonUdf { eval_type, .. } => Some(*eval_type),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `e` holds a Python UDF call anywhere (not inside subqueries,
/// which are planned on their own).
pub(crate) fn has_python_udf(e: &Expr) -> bool {
    if python_call(e).is_some() {
        return true;
    }
    let mut found = false;
    e.for_each_child(&mut |c| found = found || has_python_udf(c));
    found
}

/// The Python UDF calls of `e` whose arguments hold none, in first-seen
/// order.
fn innermost(e: &Arc<Expr>, out: &mut Vec<Arc<Expr>>) {
    if python_call(e).is_some() {
        let mut nested = false;
        e.for_each_child(&mut |c| nested = nested || has_python_udf(c));
        if !nested {
            if !out.iter().any(|o| o == e) {
                out.push(Arc::clone(e));
            }
            return;
        }
    }
    e.for_each_child(&mut |c| innermost(c, out));
}

/// `e` with each call of `columns` replaced by a read of its column.
fn replaced(e: &Arc<Expr>, columns: &[(Arc<Expr>, BoundColumn)]) -> Result<Arc<Expr>> {
    Arc::clone(e).transform_up(&mut |n: Arc<Expr>| -> Result<Arc<Expr>> {
        if python_call(&n).is_none() {
            return Ok(n);
        }
        Ok(columns
            .iter()
            .find(|(call, _)| *call == n)
            .map_or(n, |(_, column)| {
                Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                    column.clone(),
                ))))
            }))
    })
}

impl Planner<'_> {
    /// `input` under a `PythonEval` node for each round of the Python UDF
    /// calls in `exprs`, and `exprs` reading the calls' columns instead.
    /// Unchanged (no node) when `exprs` call no Python UDF.
    pub(crate) fn extract_python(
        &mut self,
        input: Planned,
        exprs: &[Arc<Expr>],
    ) -> Result<(Planned, Vec<Arc<Expr>>)> {
        self.extract_python_where(input, exprs, &|_| true)
    }

    /// [`Self::extract_python`] for the calls `wanted` accepts only (a
    /// join side's: the calls reading only its columns); the others stay in
    /// the expressions.
    fn extract_python_where(
        &mut self,
        mut input: Planned,
        exprs: &[Arc<Expr>],
        wanted: &dyn Fn(&Expr) -> bool,
    ) -> Result<(Planned, Vec<Arc<Expr>>)> {
        let mut exprs = exprs.to_vec();
        loop {
            let mut calls = Vec::new();
            for e in &exprs {
                innermost(e, &mut calls);
            }
            calls.retain(|c| wanted(c));
            let Some(first) = calls.first().and_then(|c| python_call(c)) else {
                return Ok((input, exprs));
            };
            calls.retain(|c| python_call(c) == Some(first));
            let mut columns = Vec::with_capacity(calls.len());
            for (i, call) in calls.into_iter().enumerate() {
                let ExprNode::Resolved(ResolvedExpr::Function(f)) = &call.node else {
                    return Err(internal("a Python UDF call is not a function"));
                };
                let column = BoundColumn {
                    id: self.fresh_id(),
                    name: format!("pythonUDF{i}"),
                    data_type: f.data_type.clone(),
                    nullable: true,
                };
                columns.push((call, column));
            }
            input = self.python_eval(input, &columns)?;
            exprs = exprs
                .iter()
                .map(|e| replaced(e, &columns))
                .collect::<Result<_>>()?;
        }
    }

    /// One `PythonEval` over `input` computing `columns`' calls.
    fn python_eval(
        &mut self,
        input: Planned,
        columns: &[(Arc<Expr>, BoundColumn)],
    ) -> Result<Planned> {
        let mut named = Vec::with_capacity(columns.len());
        let prepared = self.prepare(&[])?;
        for (call, column) in columns {
            if has_subquery(call) {
                return Err(super::build::unsupported(
                    "a subquery in the arguments of a Python UDF",
                ));
            }
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &call.node else {
                return Err(internal("a Python UDF call is not a function"));
            };
            // The arguments must compile over the input, as on the worker.
            let args: Vec<&Expr> = f.args.iter().map(AsRef::as_ref).collect();
            compile_all(&args, &input.layout, &input.schema, &self.values(&prepared))?;
            named.push((column.name.clone(), column.data_type.to_arrow()));
        }
        let schema = output_schema(&input.schema, &named);
        let mut layout = input.layout.clone();
        layout.extend(columns.iter().map(|(_, c)| c.id));
        let ordering = input.ordering.clone();
        let details = columns
            .iter()
            .map(|(call, c)| format!("{call} AS {}", c.name))
            .collect::<Vec<_>>()
            .join(", ");
        let node = PhysNode::PythonEval {
            input: Box::new(input.node),
            layout: input.layout,
            calls: columns
                .iter()
                .map(|(call, c)| PythonCallPlan {
                    call: Arc::clone(call),
                    output: c.id,
                })
                .collect(),
        };
        Ok(Planned {
            sites: kept(&input.sites, &layout),
            placement: input.placement.restrict(&layout),
            ordering,
            schema,
            node,
            layout,
            explain: ExplainNode::new("PythonEval", details, vec![input.explain]),
        })
    }

    /// A join whose condition calls Python UDFs (Spark's
    /// `ExtractPythonUDFFromJoinCondition`, `joins.scala` at v4.2.0): for an
    /// inner or cross join the conjuncts calling one become a filter above
    /// the join of the rest (a cross join when none is left). For the other
    /// join types a call reading both sides raises
    /// `UNSUPPORTED_FEATURE.PYTHON_UDF_IN_ON_CLAUSE`; a call reading one
    /// side is evaluated below that side, and the join reads its column
    /// (`research/spark-notes/python-udf-errors.md` §2: Spark accepts
    /// `f(a.id) = b.id` for every join type).
    pub(crate) fn join_with_python_udfs(
        &mut self,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        on: &Arc<Expr>,
    ) -> Result<Planned> {
        if !matches!(kind, JoinKind::Inner | JoinKind::Cross) {
            return self.outer_join_with_python_udfs(left, right, kind, on);
        }
        let mut conjuncts = Vec::new();
        super::join::conjuncts_of(on, &mut conjuncts);
        let (udf, rest): (Vec<_>, Vec<_>) = conjuncts.into_iter().partition(|c| has_python_udf(c));
        let condition = match conjoined(rest) {
            Some(rest) => JoinCondition::On(rest),
            None => JoinCondition::None,
        };
        let joined = self.join(left, right, kind, &condition)?;
        let predicate =
            conjoined(udf).ok_or_else(|| internal("a Python UDF join without its conjuncts"))?;
        let (input, exprs) = self.extract_python(joined, std::slice::from_ref(&predicate))?;
        let predicate = exprs
            .first()
            .ok_or_else(|| internal("a filter without a predicate"))?;
        self.filter(input, predicate)
    }
}

impl Planner<'_> {
    /// The non-inner case of [`Self::join_with_python_udfs`].
    fn outer_join_with_python_udfs(
        &mut self,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        on: &Arc<Expr>,
    ) -> Result<Planned> {
        let hints = (self.side_hint(left), self.side_hint(right));
        let (lp, rp) = (self.plan(left)?, self.plan(right)?);
        let lids: HashSet<AttrId> = lp.layout.iter().copied().collect();
        let rids: HashSet<AttrId> = rp.layout.iter().copied().collect();
        let reads_of = |e: &Expr| {
            let mut ids = HashSet::new();
            super::join::reads(e, &mut ids);
            ids
        };
        // A call reading both sides is Spark's error, whatever else it holds.
        let mut both = false;
        visit_calls(on, &mut |call| {
            let ids = reads_of(call);
            both = both
                || (ids.iter().any(|i| lids.contains(i)) && ids.iter().any(|i| rids.contains(i)));
        });
        if both {
            return Err(KrishivError::from(UnsupportedFeaturePythonUdfInOnClause {
                join_type: spark_join_type(kind).to_owned(),
            }));
        }
        // A call with no column read is evaluated with the left side.
        let on_left = |e: &Expr| reads_of(e).iter().all(|i| lids.contains(i));
        let on_right = |e: &Expr| {
            let ids = reads_of(e);
            !ids.is_empty() && ids.iter().all(|i| rids.contains(i))
        };
        let (lp, exprs) = self.extract_python_where(lp, std::slice::from_ref(on), &on_left)?;
        let (rp, exprs) = self.extract_python_where(rp, &exprs, &on_right)?;
        let on = exprs
            .first()
            .ok_or_else(|| internal("a join condition went missing"))?;
        if has_python_udf(on) {
            return Err(internal(
                "a Python UDF call in a join condition reads no side",
            ));
        }
        self.join_planned((left, right), (lp, rp), hints, kind, Some(on))
    }
}

/// Calls `f` with every Python UDF call in `e`, outer ones first.
fn visit_calls(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    if python_call(e).is_some() {
        f(e);
    }
    e.for_each_child(&mut |c| visit_calls(c, f));
}

/// `conjuncts` joined by `AND`; `None` when there are none.
fn conjoined(conjuncts: Vec<Arc<Expr>>) -> Option<Arc<Expr>> {
    conjuncts.into_iter().reduce(|l, r| {
        let nullable = nullable_of(&l) || nullable_of(&r);
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Builtin(Builtin::And),
                kind: FunctionKind::Scalar,
                args: vec![l, r],
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Boolean,
                nullable,
                deterministic: true,
            },
        ))))
    })
}

/// Whether a boolean conjunct may be null (unknown: assume so).
fn nullable_of(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => f.nullable,
        ExprNode::Resolved(ResolvedExpr::Column(c)) => c.nullable,
        _ => true,
    }
}

/// Spark's `JoinType.sql` of `kind`, as `PYTHON_UDF_IN_ON_CLAUSE` names it.
fn spark_join_type(kind: JoinKind) -> &'static str {
    match kind {
        JoinKind::Inner => "INNER",
        JoinKind::Cross => "CROSS",
        JoinKind::LeftOuter | JoinKind::Single => "LEFT OUTER",
        JoinKind::RightOuter => "RIGHT OUTER",
        JoinKind::FullOuter => "FULL OUTER",
        JoinKind::LeftSemi => "LEFT SEMI",
        JoinKind::LeftAnti => "LEFT ANTI",
        _ => "EXISTENCE",
    }
}
