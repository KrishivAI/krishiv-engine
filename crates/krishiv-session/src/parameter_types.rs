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

use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::{Expr, ExprNode, Literal, LogicalPlan, SourceId};
use krishiv_types::DataType;

/// The type of each marker of a statement whose markers were bound to literals at `spans` (byte
/// ranges of the bound text, in marker order: `NULL`s, or the `1`s of a statement a `NULL` does
/// not analyze) and analyzed as `plan`: the type a cast of that literal (implicit or written)
/// gives it, `None` where none does. Only the statement's own text (`SourceId(0)`) is matched,
/// never a view's or a function body's.
#[must_use]
pub fn marker_types(plan: &AnalyzedPlan, spans: &[Range<usize>]) -> Vec<Option<DataType>> {
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
                if let Some(i) = literal_marker(expr, &at, false) {
                    settle(&mut out, i, to);
                }
            }
            // A cast the analyzer folded into a typed `NULL` keeps the literal's origin.
            ExprNode::Literal(Literal::Null(Some(t))) => {
                if let Some(i) = literal_marker(e, &at, true) {
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

/// The marker whose bound value `e` is: a literal (a `NULL` only when `null`) of the
/// statement's own text starting where a marker's value was put.
fn literal_marker(e: &Expr, at: &HashMap<u32, usize>, null: bool) -> Option<usize> {
    let fits = match &e.node {
        ExprNode::Literal(Literal::Null(_)) => true,
        ExprNode::Literal(_) => !null,
        _ => false,
    };
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
