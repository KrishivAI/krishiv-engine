// SPDX-License-Identifier: Apache-2.0
//! The types of a parameterized statement's positional markers (`?`), which an Arrow Flight SQL
//! server advertises as the prepared statement's `parameter_schema` (P3-T101 note §4.4, amended
//! 2026-11-20: the JDBC driver binds each value into a vector of the advertised type, so a `Null`
//! or a `Utf8` for every marker refuses `setDate`/`setInt`).
//!
//! A query's markers are typed from its analysis: the statement is bound with every marker
//! `NULL` ([`crate::Session::bind_parameters_spanned`] gives where each went), analyzed, and each
//! marker gets the type the analyzer cast that `NULL` to (coercion, or a written `CAST`), as
//! Postgres types a parameter from its context. A marker nothing types (`SELECT ?`, `? AS s`)
//! has none. An `INSERT ... VALUES` cell that is a marker gets its target column's type
//! ([`crate::Session::insert_parameter_types`]).

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::{Expr, ExprNode, Literal, LogicalPlan, SourceId};
use krishiv_types::DataType;

use crate::dataframe::DataFrame;

impl DataFrame {
    /// [`marker_types`] of this frame's analysis, for a statement whose markers were bound to
    /// `bound` at `spans`; every marker untyped when the principal may not be shown the plan (the
    /// inlined queries of `SQL SECURITY DEFINER` views it may not read as invoker). That is
    /// asked quietly, as `system.operators` asks it: no audit record and no denial count, so
    /// preparing a statement over such a view is not a denial (ADR 0046 §8).
    ///
    /// # Errors
    /// Spark's analysis errors.
    pub async fn marker_types(
        &self,
        spans: &[Range<usize>],
        bound: BoundAs,
    ) -> Result<Vec<Option<DataType>>, KrishivError> {
        let df = self.clone();
        let spans = spans.to_vec();
        self.session
            .on_executor(async move {
                let analysis = df.analysis().await?;
                let analyzed = &analysis.analyzed;
                if !df.session.may_see_definer_reads(&analyzed.views).await {
                    return Ok(vec![None; spans.len()]);
                }
                Ok(marker_types(analyzed, &spans, bound))
            })
            .await
    }
}

/// What every marker was bound to before analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundAs {
    /// `NULL` (the first classification).
    Null,
    /// The `INT` `1` (a statement a `NULL` does not analyze: `LIMIT ?`).
    One,
}

/// The type of each marker of a statement whose markers were bound to `bound` at `spans` (byte
/// ranges of the bound text, in marker order) and analyzed as `plan`: the type a cast of that
/// literal (implicit or written) gives it, `None` where none does. Only the statement's own text
/// (`SourceId(0)`) is matched, never a view's or a function body's, and only a literal of the
/// kind bound: should anything shift the text between binding and parsing (`${var}`
/// substitution before a marker), a marker's span may start at another literal, which then
/// types nothing unless it is the same `NULL` or `1`.
#[must_use]
pub fn marker_types(
    plan: &AnalyzedPlan,
    spans: &[Range<usize>],
    bound: BoundAs,
) -> Vec<Option<DataType>> {
    let mut out: Vec<Option<DataType>> = vec![None; spans.len()];
    let at: HashMap<u32, usize> = spans
        .iter()
        .enumerate()
        .filter_map(|(i, r)| u32::try_from(r.start).ok().map(|s| (s, i)))
        .collect();
    if at.is_empty() {
        return out;
    }
    // Plans and expressions are DAGs: each node is visited once.
    let mut seen_plans: HashSet<*const LogicalPlan> = HashSet::new();
    let mut seen_exprs: HashSet<*const Expr> = HashSet::new();
    let mut plans: Vec<&Arc<LogicalPlan>> = vec![plan.plan.root()];
    let mut exprs: Vec<&Arc<Expr>> = Vec::new();
    loop {
        if let Some(p) = plans.pop() {
            if seen_plans.insert(Arc::as_ptr(p)) {
                p.for_each_child(&mut |c| plans.push(c));
                p.for_each_expr(&mut |e| exprs.push(e));
            }
            continue;
        }
        let Some(e) = exprs.pop() else {
            break;
        };
        if !seen_exprs.insert(Arc::as_ptr(e)) {
            continue;
        }
        match &e.node {
            ExprNode::Cast { expr, to, .. } => {
                if let Some(i) = literal_marker(expr, &at, bound) {
                    settle(&mut out, i, to);
                }
            }
            // A cast the analyzer folded into a typed `NULL` keeps the literal's origin.
            ExprNode::Literal(Literal::Null(Some(t))) if bound == BoundAs::Null => {
                if let Some(i) = literal_marker(e, &at, bound) {
                    settle(&mut out, i, t);
                }
            }
            ExprNode::Subquery { plan, .. } => plans.push(plan),
            _ => {}
        }
        e.for_each_child(&mut |c| exprs.push(c));
    }
    out
}

/// The marker whose bound value `e` is: a literal of the kind bound, of the statement's own
/// text, starting where a marker's value was put.
fn literal_marker(e: &Expr, at: &HashMap<u32, usize>, bound: BoundAs) -> Option<usize> {
    let fits = matches!(
        (&e.node, bound),
        (ExprNode::Literal(Literal::Null(_)), BoundAs::Null)
            | (ExprNode::Literal(Literal::Integer(1)), BoundAs::One)
    );
    if !fits {
        return None;
    }
    match e.origin.sql {
        Some((SourceId(0), span)) => at.get(&span.start).copied(),
        _ => None,
    }
}

/// Records `t` for marker `i` unless it has a type already; `NULL`'s own type types nothing.
fn settle(out: &mut [Option<DataType>], i: usize, t: &DataType) {
    if let Some(slot) = out.get_mut(i)
        && slot.is_none()
        && !matches!(t, DataType::Null)
    {
        *slot = Some(t.clone());
    }
}

/// The Arrow schema of markers typed `types`, each as the session's results show a column of that
/// type: `parameter_<n>` (from 1), nullable; a marker with no type is a string (a driver binds
/// text into it).
#[must_use]
pub fn parameter_schema(types: &[Option<DataType>]) -> arrow_schema::SchemaRef {
    let fields: Vec<arrow_schema::Field> = types
        .iter()
        .enumerate()
        .map(|(i, t)| {
            t.as_ref()
                .unwrap_or(&DataType::String)
                .to_arrow_field(format!("parameter_{}", i + 1), true)
        })
        .collect();
    crate::carrier::client_schema(&arrow_schema::Schema::new(fields))
}
