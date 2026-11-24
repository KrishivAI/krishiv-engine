// SPDX-License-Identifier: Apache-2.0
//! SQL `window(...)` and `session_window(...)`: Spark's `TimeWindowing` and `SessionWindowing` rules
//! (`ResolveTimeWindows.scala`, `TimeWindowResolution.scala`), as a rewrite of the lowered SQL plan before
//! Bind checks its functions.
//!
//! A single-input node (project, filter, aggregate, sort) whose expressions hold one distinct window call
//! gets a new input and reads the window as the column `window` (`session_window`):
//!
//! * `window(ts, window[, slide[, start]])`: `Generate[explode(__time_windows(ts, ..))] AS window` over the input. The
//!   call (see `krishiv-functions` `scalar/time_window.rs`) holds the windows of `ts` that contain it, so a
//!   NULL time and a time in no window leave no row, as Spark's `IS NOT NULL` / `start <= ts < end` filters
//!   do. Durations are parsed here, as `TimeWindow.parseExpression` does, and checked as
//!   `TimeWindow.checkInputDataTypes` does.
//! * `session_window(ts, gap)` as a grouping key of a plain `GROUP BY`: Spark merges the rows whose
//!   `[ts, ts + gap)` ranges touch (`MergingSessionsExec`). Here: the rows with a NULL time or a gap of zero
//!   or less are filtered; a running maximum of the earlier ranges' ends, per the other grouping keys,
//!   numbers the sessions (a new session starts when `ts` is past that end); the session's start and end
//!   are the minimum and maximum over its rows; the aggregate groups by that struct.
//! * `session_window` anywhere else (a `WHERE`, a select item, inside a grouping expression): one range per
//!   row, `named_struct('start', ts, 'end', ts + gap)`, with the same filter.
//!
//! More than one distinct window call in one node is `_LEGACY_ERROR_TEMP_1039`; a window call in a
//! `HAVING` or `ORDER BY` over an aggregate cannot find its input there
//! (`MISSING_ATTRIBUTES.RESOLVED_ATTRIBUTE_MISSING_FROM_INPUT`). A window call whose time column is itself
//! a window (a nested call, or a column a derived table took from an inner `window(..)`), `window_time(..)` over
//! anything else and the DataFrame form stay unsupported.

use std::ops::ControlFlow;
use std::sync::Arc;

use krishiv_common::error::{
    CannotParseInterval, DatatypeMismatchParameterConstraintViolation,
    DatatypeMismatchValueOutOfRange, InternalError, LegacyErrorTemp1006, LegacyErrorTemp1039,
    LegacyErrorTemp1207, LegacyErrorTemp3231, MissingAggregation,
    MissingAttributesResolvedAttributeMissingFromInput, UnresolvedColumnWithSuggestion,
    WrongNumArgsWithoutSuggestion,
};
use krishiv_types::DataType;
use krishiv_types::interval_parse::parse_string;

use super::AnalysisError;
use super::bind::unsupported;
use super::exprs::SPARK_DOC_ROOT;
use crate::walk::{Node, visit_distinct};
use crate::{
    Builtin, CastMode, Expr, ExprNode, FrameBound, FrameUnits, FunctionCall, GeneratorKind,
    Grouping, Literal, LogicalPlan, NullOrdering, Plan, PlanNode, SortDirection, SortOrder, Star,
    WindowExpr, WindowFrame, WindowSpec, WindowSpecRef,
};

const WINDOW: &str = "window";
const SESSION: &str = "session_window";
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// Whether `e` is an unqualified call of `name` (`window` or `session_window`).
fn call_of<'a>(e: &'a Expr, name: &str) -> Option<&'a FunctionCall> {
    match &e.node {
        ExprNode::Function(f) if f.qualifier.is_empty() && f.name.eq_ignore_ascii_case(name) => {
            Some(f)
        }
        _ => None,
    }
}

fn is_window_call(e: &Expr) -> bool {
    call_of(e, WINDOW).is_some() || call_of(e, SESSION).is_some()
}

/// Spark's `window_time` is registered as the scalar `__window_time` (the catalog's row is an aggregate).
const WINDOW_TIME: &str = "window_time";
const WINDOW_TIME_SCALAR: &str = "__window_time";

/// Whether anything under `root` (subqueries included) calls `window` or `session_window`.
fn has_calls(root: &Arc<LogicalPlan>) -> bool {
    visit_distinct(Node::Plan(root), |n| match n {
        Node::Expr(e) if is_window_call(e) || call_of(e, WINDOW_TIME).is_some() => {
            ControlFlow::Break(())
        }
        _ => ControlFlow::Continue(()),
    })
    .is_break()
}

/// The window calls in `e`, outside subqueries, outermost first.
fn collect(e: &Arc<Expr>, out: &mut Vec<Arc<Expr>>) {
    if is_window_call(e) {
        out.push(Arc::clone(e));
    }
    e.for_each_child(&mut |c| collect(c, out));
}

/// The structural identity of an expression (origins excluded).
fn key(e: &Arc<Expr>, origin: crate::Origin) -> Result<crate::StructuralHash, AnalysisError> {
    let plan = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![(**e).clone()]);
    Plan::try_new(plan)
        .map(|p| p.structural_hash())
        .map_err(|err| AnalysisError::new(err, origin))
}

/// Rewrites the lowered plan `root` (see the module documentation); a plan with no window call is returned
/// as it is.
pub(super) fn rewrite(root: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, AnalysisError> {
    if !has_calls(&root) {
        return Ok(root);
    }
    rewrite_plan(root)
}

fn rewrite_plan(root: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, AnalysisError> {
    // Checked before the rewrite: it replaces the nested window call the check looks for.
    let mut refused = Ok(());
    let _ = visit_distinct(Node::Plan(&root), |n| {
        if let Node::Plan(p) = n
            && let Err(e) = refuse_inner_window_above_nested(p)
        {
            refused = Err(e);
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    refused?;
    root.transform_up(&mut |p| {
        let p = p.map_expressions(rewrite_subqueries)?;
        let p = rename_window_time(p)?;
        node(p)
    })
}

/// `p` with its `window_time(w)` calls renamed to the scalar, a select item that is one named as Spark
/// names it (`window_time(window)`).
fn rename_window_time(p: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, AnalysisError> {
    let mut any = false;
    p.for_each_expr(&mut |e| any = any || has_window_time(e));
    if !any {
        return Ok(p);
    }
    let named = matches!(
        p.node,
        PlanNode::Project { .. } | PlanNode::Aggregate { .. }
    );
    p.map_expressions(|e| {
        let item = if named && call_of(&e, WINDOW_TIME).is_some() {
            let text = match call_of(&e, WINDOW_TIME).and_then(|f| f.args.first()) {
                Some(a) => match &a.node {
                    ExprNode::Column(c) => Some(c.name.join(".")),
                    _ => None,
                },
                None => None,
            };
            text.map(|t| format!("{WINDOW_TIME}({t})"))
        } else {
            None
        };
        let origin = e.origin;
        let renamed = e.transform_up(&mut |x| match &x.node {
            ExprNode::Function(f)
                if f.qualifier.is_empty() && f.name.eq_ignore_ascii_case(WINDOW_TIME) =>
            {
                let mut f = f.clone();
                WINDOW_TIME_SCALAR.clone_into(&mut f.name);
                Ok::<_, AnalysisError>(Arc::new(
                    Expr::new(ExprNode::Function(f)).with_origin(x.origin),
                ))
            }
            _ => Ok(x),
        })?;
        Ok(match item {
            Some(name) => alias(Arc::unwrap_or_clone(renamed), &name, origin),
            None => renamed,
        })
    })
}

fn has_window_time(e: &Arc<Expr>) -> bool {
    call_of(e, WINDOW_TIME).is_some() || {
        let mut any = false;
        e.for_each_child(&mut |c| any = any || has_window_time(c));
        any
    }
}

/// `e` with the plans of its subqueries rewritten.
fn rewrite_subqueries(e: Arc<Expr>) -> Result<Arc<Expr>, AnalysisError> {
    e.transform_up(&mut |x| match &x.node {
        ExprNode::Subquery { kind, plan } if has_calls(plan) => {
            let plan = rewrite_plan(Arc::clone(plan))?;
            Ok(Arc::new(
                Expr::new(ExprNode::Subquery {
                    kind: kind.clone(),
                    plan,
                })
                .with_origin(x.origin),
            ))
        }
        _ => Ok(x),
    })
}

fn col(name: &str, origin: crate::Origin) -> Arc<Expr> {
    Arc::new(Expr::column([name]).with_origin(origin))
}

fn lit_long(v: i64) -> Expr {
    Expr::literal(Literal::Long(v))
}

fn alias(e: Expr, name: &str, origin: crate::Origin) -> Arc<Expr> {
    Arc::new(
        Expr::new(ExprNode::Alias {
            expr: Arc::new(e),
            name: name.to_owned(),
        })
        .with_origin(origin),
    )
}

fn star(origin: crate::Origin) -> Arc<Expr> {
    Arc::new(
        Expr::new(ExprNode::Star(Star {
            qualifier: Vec::new(),
            except: Vec::new(),
            replace: Vec::new(),
            position: None,
            regex: None,
            plan_id: None,
        }))
        .with_origin(origin),
    )
}

fn project(
    input: Arc<LogicalPlan>,
    exprs: Vec<Arc<Expr>>,
    origin: crate::Origin,
) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Project { input, exprs }).with_origin(origin))
}

/// Whether `p` is the `QUALIFY` filter lowering builds: the wrapped predicate over the select list's
/// projection.
fn qualify_over_project(p: &LogicalPlan) -> Option<(&Arc<LogicalPlan>, &Arc<Expr>)> {
    match &p.node {
        PlanNode::Filter { input, predicate }
            if call_of(predicate, "qualify").is_some()
                && matches!(input.node, PlanNode::Project { .. }) =>
        {
            Some((input, predicate))
        }
        _ => None,
    }
}

/// Whether `e` calls `window_time` (renamed) outside any aggregate function (outside subqueries): inside
/// `max(window_time(window))` the call reads the group's window, so it is no error.
fn calls_window_time(e: &Arc<Expr>) -> bool {
    if let ExprNode::Function(f) = &e.node
        && f.qualifier.is_empty()
        && super::BuiltinFunctions::is_aggregate(&f.name)
    {
        return false;
    }
    call_of(e, WINDOW_TIME_SCALAR).is_some_and(|f| f.args.len() == 1) || {
        let mut any = false;
        e.for_each_child(&mut |c| any = any || calls_window_time(c));
        any
    }
}

/// The inner window's column as a reference names it: the bare `window`, or one of its fields
/// (`window.start`, `window.end`). A table qualifier called `window` (`window.ts`) is not the column, and with a
/// relation aliased `window` in scope the two-part forms are that relation's columns.
fn inner_window_ref(parts: &[String], alias_window: bool) -> Option<&[String]> {
    let is_window = |p: &String| p.eq_ignore_ascii_case(WINDOW);
    match parts {
        [w] if is_window(w) => Some(parts),
        [w, field]
            if !alias_window
                && is_window(w)
                && (field.eq_ignore_ascii_case("start") || field.eq_ignore_ascii_case("end")) =>
        {
            Some(parts)
        }
        _ => None,
    }
}

/// Whether `e` names the inner window's `window` column outside any aggregate call: an aggregate's
/// arguments are covered (`count(window)`, `any_value(window)`), as in `calls_window_time`. `seen` holds the
/// expressions already scanned without a match, so a shared one is walked once.
fn bare_window(
    e: &Arc<Expr>,
    alias_window: bool,
    seen: &mut std::collections::HashSet<*const Expr>,
) -> bool {
    if let ExprNode::Function(f) = &e.node
        && f.qualifier.is_empty()
        && super::BuiltinFunctions::is_aggregate(&f.name)
    {
        return false;
    }
    if matches!(&e.node, ExprNode::Column(c) if inner_window_ref(&c.name, alias_window).is_some()) {
        return true;
    }
    // The pointer is the key: `e` stays alive for the whole scan, so no address is reused.
    if !seen.insert(Arc::as_ptr(e)) {
        return false;
    }
    let mut any = false;
    e.for_each_child(&mut |c| any = any || bare_window(c, alias_window, seen));
    any
}

/// In a grouped `window(window(ts, ..), ..)`, a `window` (or `window.start` / `window.end`) inside an aggregate's
/// arguments is the inner window (Spark's inner expansion), not the outer window column that grouping adds:
/// `any_value(window)` is the inner struct and `max(window.start)` its latest start. The reference is read from
/// `__tw_inner`; the aggregate keeps the name Spark gives it (`any_value(window)`).
fn inner_window_in_aggregates(
    e: &Arc<Expr>,
    alias_window: bool,
) -> Result<Arc<Expr>, AnalysisError> {
    let rewritten = Arc::clone(e).transform_up(&mut |x| {
        let is_aggregate = matches!(&x.node, ExprNode::Function(f)
            if f.qualifier.is_empty() && super::BuiltinFunctions::is_aggregate(&f.name));
        if !is_aggregate {
            return Ok(x);
        }
        x.transform_up(&mut |y| match &y.node {
            ExprNode::Column(c) if inner_window_ref(&c.name, alias_window).is_some() => {
                let mut parts = c.name.clone();
                parts[0] = "__tw_inner".to_owned();
                Ok(Arc::new(Expr::column(parts).with_origin(y.origin)))
            }
            _ => Ok(y),
        })
    })?;
    if matches!(&e.node, ExprNode::Alias { .. }) || *rewritten == **e {
        return Ok(rewritten);
    }
    Ok(alias(
        (*rewritten).clone(),
        &crate::display::identity_string(e),
        e.origin,
    ))
}

/// The distinct window calls of `found` that are not the time column of another: the operator's windows.
fn outermost(
    found: &[Arc<Expr>],
    origin: crate::Origin,
) -> Result<(Vec<Arc<Expr>>, Vec<crate::StructuralHash>), AnalysisError> {
    let mut inner = Vec::new();
    for c in found {
        if let Some(f) = call_of(c, WINDOW)
            && let Some(a) = f.args.first()
            && call_of(a, WINDOW).is_some()
        {
            inner.push(key(a, origin)?);
        }
    }
    let mut keys = Vec::new();
    let mut calls = Vec::new();
    for c in found {
        let k = key(c, origin)?;
        if !keys.contains(&k) && !inner.contains(&k) {
            keys.push(k);
            calls.push(Arc::clone(c));
        }
    }
    Ok((calls, keys))
}

/// Whether `agg` is an aggregate that groups by `window(window(ts, ..), ..)`.
fn is_nested_grouped_window(agg: &LogicalPlan) -> bool {
    if !matches!(agg.node, PlanNode::Aggregate { .. }) {
        return false;
    }
    let mut found = Vec::new();
    agg.for_each_expr(&mut |e| collect(e, &mut found));
    found.iter().any(|c| {
        call_of(c, WINDOW)
            .and_then(|f| f.args.first())
            .is_some_and(|a| call_of(a, WINDOW).is_some())
    })
}

/// In a `HAVING` or `ORDER BY` over an aggregate grouped by a nested window, `window`, `window.start` and
/// `window.end` inside an aggregate do not resolve (Spark 4.3: `UNRESOLVED_COLUMN.WITH_SUGGESTION`, where the
/// select list reads the inner window). The proposal is the aggregate's output names, closest first. An aggregate
/// whose own list reads the window outside an aggregate is left to the `MISSING_AGGREGATION` check, which Spark
/// raises first.
///
/// The plan shapes the analyzer produces over the aggregate are `Filter` (HAVING), `Sort` and `Sort -> Filter`
/// (ORDER BY with HAVING); a `Project` between them is not produced.
fn refuse_inner_window_above_nested(p: &LogicalPlan) -> Result<(), AnalysisError> {
    fn through_filter(input: &Arc<LogicalPlan>) -> &LogicalPlan {
        match &input.node {
            PlanNode::Filter { input, .. } => input,
            _ => input,
        }
    }
    let below: &LogicalPlan = match &p.node {
        PlanNode::Filter { input, .. } => input,
        PlanNode::Sort { input, .. } => through_filter(input),
        _ => return Ok(()),
    };
    let PlanNode::Aggregate {
        input, aggregates, ..
    } = &below.node
    else {
        return Ok(());
    };
    if !is_nested_grouped_window(below) {
        return Ok(());
    }
    let alias_window = matches!(&input.node, PlanNode::SubqueryAlias { alias, .. }
        if alias.eq_ignore_ascii_case(WINDOW));
    let mut seen = std::collections::HashSet::new();
    if aggregates
        .iter()
        .any(|e| bare_window(e, alias_window, &mut seen))
    {
        return Ok(());
    }
    let mut hit: Option<(Vec<String>, crate::Origin)> = None;
    p.for_each_expr(&mut |e| {
        visit_aggregate_args(e, &mut |c, o| {
            if hit.is_none() && inner_window_ref(c, alias_window).is_some() {
                hit = Some((c.to_vec(), o));
            }
        });
    });
    let Some((name, origin)) = hit else {
        return Ok(());
    };
    let outputs = aggregates
        .iter()
        .map(|e| match &e.node {
            ExprNode::Alias { name, .. } => vec![name.clone()],
            ExprNode::Column(c) => vec![c.name.last().cloned().unwrap_or_default()],
            _ => vec![super::exprs::pretty(&count_star_as_one(e), None)],
        })
        .collect();
    let proposal = super::suggest::closest(&name.join("."), outputs)
        .iter()
        .map(|n| super::catalog::quoted(n))
        .collect::<Vec<_>>()
        .join(", ");
    Err(AnalysisError::new(
        UnresolvedColumnWithSuggestion {
            object_name: super::catalog::quoted(&name),
            proposal,
        },
        origin,
    ))
}

/// `e` with a plain `count(*)` as `count(1)`, which is how Spark names it (the star is still unresolved here).
fn count_star_as_one(e: &Arc<Expr>) -> Arc<Expr> {
    match &e.node {
        ExprNode::Function(f)
            if !f.distinct
                && call_of(e, "count").is_some()
                && matches!(&f.args[..], [a] if matches!(&a.node, ExprNode::Star(s) if s.qualifier.is_empty())) =>
        {
            let mut f = f.clone();
            f.args = vec![Arc::new(Expr::literal(Literal::Integer(1)))];
            Arc::new(Expr {
                node: ExprNode::Function(f),
                ..(**e).clone()
            })
        }
        _ => Arc::clone(e),
    }
}

/// Calls `f` with every column reference inside an aggregate call of `e`.
fn visit_aggregate_args(e: &Arc<Expr>, f: &mut dyn FnMut(&[String], crate::Origin)) {
    let mut seen = std::collections::HashSet::new();
    fn go(
        e: &Arc<Expr>,
        in_agg: bool,
        seen: &mut std::collections::HashSet<(*const Expr, bool)>,
        f: &mut dyn FnMut(&[String], crate::Origin),
    ) {
        if !seen.insert((Arc::as_ptr(e), in_agg)) {
            return;
        }
        let agg = in_agg
            || matches!(&e.node, ExprNode::Function(c)
                if c.qualifier.is_empty() && super::BuiltinFunctions::is_aggregate(&c.name));
        if agg && let ExprNode::Column(c) = &e.node {
            f(&c.name, e.origin);
        }
        e.for_each_child(&mut |c| go(c, agg, seen, f));
    }
    go(e, false, &mut seen, f);
}

/// `p` with its window calls read from a new input (see the module documentation).
fn node(p: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, AnalysisError> {
    if !matches!(
        p.node,
        PlanNode::Project { .. }
            | PlanNode::Filter { .. }
            | PlanNode::Aggregate { .. }
            | PlanNode::Sort { .. }
            | PlanNode::Pivot { .. }
    ) {
        return Ok(p);
    }
    let mut found = Vec::new();
    p.for_each_expr(&mut |e| collect(e, &mut found));
    if found.is_empty() {
        return Ok(p);
    }
    let origin = p.origin;
    // A window call as a pivot's aggregate is no aggregate.
    if matches!(p.node, PlanNode::Pivot { .. }) {
        let name = call_of(&found[0], WINDOW).map_or(SESSION, |_| WINDOW);
        return Err(AnalysisError::new(
            LegacyErrorTemp1006 {
                sql: name.to_owned(),
            },
            origin,
        ));
    }
    let (calls, keys) = outermost(&found, origin)?;
    if calls.len() > 1 {
        return Err(AnalysisError::new(LegacyErrorTemp1039 {}, origin));
    }
    let call = Arc::clone(&calls[0]);
    let wkey = keys[0];
    let Some(f) = call_of(&call, WINDOW).or_else(|| call_of(&call, SESSION)) else {
        return Ok(p);
    };
    if f.distinct || f.filter.is_some() || !f.within_group.is_empty() || !f.arg_names.is_empty() {
        return Err(unsupported("a modified window call", origin));
    }
    // `window_time(window)` of a grouped window is not a grouping key or an aggregate.
    let mut grouped_window_time = false;
    if let PlanNode::Aggregate { aggregates, .. } = &p.node {
        grouped_window_time = aggregates.iter().any(calls_window_time);
    }
    if grouped_window_time {
        return Err(AnalysisError::new(
            MissingAggregation {
                expression: "\"window_time(window)\"".to_owned(),
                expression_any_value: "\"any_value(window_time(window))\"".to_owned(),
            },
            origin,
        ));
    }
    // A window call in a `HAVING` or `ORDER BY` over an aggregate: the rewrite adds the window below the
    // operator, where the aggregate's input is not visible.
    let over_aggregate = |input: &Arc<LogicalPlan>| match &input.node {
        PlanNode::Aggregate { .. } => true,
        PlanNode::Filter { input, .. } => matches!(input.node, PlanNode::Aggregate { .. }),
        _ => false,
    };
    let reads_aggregate = match &p.node {
        PlanNode::Filter { input, .. } => matches!(input.node, PlanNode::Aggregate { .. }),
        PlanNode::Sort { input, .. } => over_aggregate(input),
        _ => false,
    };
    if reads_aggregate {
        let time = f.args.first().map_or_else(String::new, |a| match &a.node {
            ExprNode::Column(c) => c.name.join("."),
            _ => String::new(),
        });
        return Err(AnalysisError::new(
            MissingAttributesResolvedAttributeMissingFromInput {
                missing_attributes: format!("\"{time}\""),
                input: String::new(),
                operator: "Project".to_owned(),
            },
            origin,
        ));
    }
    let is_time = call_of(&call, WINDOW).is_some();
    let name = if is_time { WINDOW } else { SESSION };
    let replace = |e: Arc<Expr>| -> Result<Arc<Expr>, AnalysisError> {
        e.transform_up(&mut |x| {
            if is_window_call(&x) && key(&x, origin)? == wkey {
                Ok(col(name, x.origin))
            } else {
                Ok(x)
            }
        })
    };
    // The relation the window is added to: the `QUALIFY` filter's window goes below the select list's
    // projection (Spark adds it below the filter, and the projection's outputs follow it).
    let qualify = qualify_over_project(&p).map(|(project, _)| Arc::clone(project));
    let base = match &qualify {
        Some(project) => only_input(project),
        None => only_input(&p),
    }
    .ok_or_else(|| unsupported("a window over several inputs", origin))?;
    let input = if is_time {
        time_window_input(base, f, origin)?
    } else {
        session_input(&p, base, f, wkey, origin)?
    };
    // `window(window(ts, ..), ..)` grouped: a bare `window` in the select list names the inner window's
    // column (Spark's inner expansion), which the outer grouping does not cover. Inside an aggregate it reads the
    // inner window.
    let alias_window = matches!(&base.node, PlanNode::SubqueryAlias { alias, .. }
        if alias.eq_ignore_ascii_case(WINDOW));
    let mut p = p;
    if let PlanNode::Aggregate { aggregates, .. } = &p.node
        && call_of(&call, WINDOW)
            .is_some_and(|w| w.args.first().is_some_and(|a| call_of(a, WINDOW).is_some()))
    {
        let mut seen = std::collections::HashSet::new();
        if aggregates
            .iter()
            .any(|a| bare_window(a, alias_window, &mut seen))
        {
            return Err(AnalysisError::new(
                MissingAggregation {
                    expression: "\"window\"".to_owned(),
                    expression_any_value: "\"any_value(window)\"".to_owned(),
                },
                origin,
            ));
        }
        if let PlanNode::Aggregate {
            input: agg_input,
            grouping,
            aggregates,
        } = &p.node
        {
            let rewritten = aggregates
                .iter()
                .map(|a| inner_window_in_aggregates(a, alias_window))
                .collect::<Result<Vec<_>, _>>()?;
            if rewritten
                .iter()
                .zip(aggregates)
                .any(|(r, a)| !Arc::ptr_eq(r, a))
            {
                p = Arc::new(
                    LogicalPlan::new(PlanNode::Aggregate {
                        input: Arc::clone(agg_input),
                        grouping: grouping.clone(),
                        aggregates: rewritten,
                    })
                    .with_origin(p.origin),
                );
            }
        }
    }
    let Some(project) = qualify else {
        let with_input = p.map_children(|_| Ok::<_, AnalysisError>(Arc::clone(&input)))?;
        let with_input = except_added_column(with_input, name);
        return with_input.map_expressions(replace);
    };
    let PlanNode::Project { exprs, .. } = &project.node else {
        return Ok(p);
    };
    let mut selected = vec![col(name, origin)];
    selected.extend(exprs.iter().map(|e| except_star(e, name)));
    let widened = Arc::new(
        LogicalPlan::new(PlanNode::Project {
            input,
            exprs: selected,
        })
        .with_origin(project.origin),
    );
    let with_input = p.map_children(|_| Ok::<_, AnalysisError>(Arc::clone(&widened)))?;
    with_input.map_expressions(replace)
}

/// A select list's bare `*` stands for the input's columns as the query wrote them: Spark expands it before
/// it adds the window column, so `SELECT *, window(ts, ..)` has `window` once. A table that has a column of
/// the same name makes the reference ambiguous.
fn except_added_column(p: Arc<LogicalPlan>, name: &str) -> Arc<LogicalPlan> {
    let PlanNode::Project { input, exprs } = &p.node else {
        return p;
    };
    let exprs = exprs.iter().map(|e| except_star(e, name)).collect();
    Arc::new(
        LogicalPlan::new(PlanNode::Project {
            input: Arc::clone(input),
            exprs,
        })
        .with_origin(p.origin),
    )
}

/// `e`, a bare `*`, without the column `name` the rewrite added; any other expression as it is.
fn except_star(e: &Arc<Expr>, name: &str) -> Arc<Expr> {
    match &e.node {
        ExprNode::Star(s)
            if s.qualifier.is_empty() && s.regex.is_none() && s.position.is_none() =>
        {
            let mut s = s.clone();
            s.except.push(vec![name.to_owned()]);
            Arc::new(Expr::new(ExprNode::Star(s)).with_origin(e.origin))
        }
        _ => Arc::clone(e),
    }
}

/// The only input of `p`.
fn only_input(p: &LogicalPlan) -> Option<&Arc<LogicalPlan>> {
    let mut kids = Vec::new();
    p.for_each_child(&mut |c| kids.push(c));
    match kids.as_slice() {
        [one] => Some(one),
        _ => None,
    }
}

/// The text of an argument for an error's `sqlExpr`.
fn shown(f: &FunctionCall) -> String {
    format!("\"{}(..)\"", f.name)
}

/// One duration argument as microseconds (`TimeWindow.parseExpression`).
fn duration(e: &Expr, origin: crate::Origin) -> Result<i64, AnalysisError> {
    match &e.node {
        ExprNode::Literal(Literal::String(s)) => {
            let (cal, _) = parse_string(s).map_err(|_| {
                AnalysisError::new(
                    CannotParseInterval {
                        interval_string: format!("'{s}'"),
                    },
                    origin,
                )
            })?;
            if cal.months != 0 {
                return Err(AnalysisError::new(
                    LegacyErrorTemp3231 {
                        interval: s.clone(),
                    },
                    origin,
                ));
            }
            i64::from(cal.days)
                .checked_mul(MICROS_PER_DAY)
                .and_then(|d| d.checked_add(cal.micros))
                .ok_or_else(|| {
                    AnalysisError::new(
                        InternalError {
                            message: "arithmetic overflow".to_owned(),
                        },
                        origin,
                    )
                })
        }
        ExprNode::Literal(Literal::Integer(i)) => Ok(i64::from(*i)),
        ExprNode::Literal(Literal::Long(l)) => Ok(*l),
        _ => Err(AnalysisError::new(LegacyErrorTemp1207 {}, origin)),
    }
}

/// `LATERAL VIEW explode(__time_windows(ts, window, slide, start)) t AS window` over `base`, for the
/// window call `f`.
fn time_window_input(
    base: &Arc<LogicalPlan>,
    f: &FunctionCall,
    origin: crate::Origin,
) -> Result<Arc<LogicalPlan>, AnalysisError> {
    if !(2..=4).contains(&f.args.len()) {
        return Err(AnalysisError::new(
            WrongNumArgsWithoutSuggestion {
                function_name: "`window`".to_owned(),
                expected_num: "[2, 3, 4]".to_owned(),
                actual_num: f.args.len().to_string(),
                docroot: SPARK_DOC_ROOT.to_owned(),
            },
            origin,
        ));
    }
    // `window(window(ts, ..), ..)`: the outer window reads the inner one's `window_time`.
    if let Some(inner) = call_of(&f.args[0], WINDOW) {
        // An inner `window()` with no arguments gets the arity error of `single_window`.
        let inner_time = match inner.args.first() {
            Some(t) => (**t).clone(),
            None => {
                return Err(AnalysisError::new(
                    WrongNumArgsWithoutSuggestion {
                        function_name: "`window`".to_owned(),
                        expected_num: "[2, 3, 4]".to_owned(),
                        actual_num: "0".to_owned(),
                        docroot: SPARK_DOC_ROOT.to_owned(),
                    },
                    origin,
                ));
            }
        };
        let inner_input = single_window(base, inner, "__tw_inner", inner_time, origin)?;
        let time = Expr::function(WINDOW_TIME_SCALAR, vec![Expr::column(["__tw_inner"])]);
        return single_window(&inner_input, f, WINDOW, time, origin);
    }
    // `window(w, ..)` where `w` is a column a derived table took from an inner `window(..)`: Spark reads the
    // window's `window_time`, as for the nested call above.
    if let ExprNode::Column(c) = &f.args[0].node
        && let Some(name) = derived_column_name(base, &c.name)
        && is_window_column(base, name)
    {
        // The column is read under the name the `window_time` check accepts for a window column.
        let renamed = project(
            Arc::clone(base),
            vec![
                star(origin),
                alias(Expr::column([name]), "__tw_inner", origin),
            ],
            origin,
        );
        let time = Expr::function(WINDOW_TIME_SCALAR, vec![Expr::column(["__tw_inner"])]);
        return single_window(&renamed, f, WINDOW, time, origin);
    }
    single_window(base, f, WINDOW, (*f.args[0]).clone(), origin)
}

/// The column name of a one-part reference, or of a two-part one whose qualifier is the derived table's alias.
fn derived_column_name<'a>(base: &Arc<LogicalPlan>, parts: &'a [String]) -> Option<&'a str> {
    match parts {
        [name] => Some(name),
        [qualifier, name] if matches!(&base.node, PlanNode::SubqueryAlias { alias, .. } if alias.eq_ignore_ascii_case(qualifier)) => {
            Some(name)
        }
        _ => None,
    }
}

/// Whether the output column `name` of `p` is the window column an inner `window(..)` added: an alias of
/// `window` (or `window` itself) over the time-window generator, seen through derived-table aliases.
fn is_window_column(p: &Arc<LogicalPlan>, name: &str) -> bool {
    match &p.node {
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => {
            is_window_column(input, name)
        }
        // These keep their input's rows and output columns.
        PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Offset { input, .. } => is_window_column(input, name),
        // A renamed column list (`AS q(w, n)`): `name` is the list's entry at some position, and the input's
        // output column at that position is the one to look at.
        PlanNode::SubqueryAlias { input, columns, .. } => columns
            .iter()
            .position(|c| c.eq_ignore_ascii_case(name))
            .and_then(|i| output_names(input).into_iter().nth(i))
            .is_some_and(|inner| is_window_column(input, &inner)),
        PlanNode::Project { input, exprs }
        | PlanNode::Aggregate {
            input,
            aggregates: exprs,
            ..
        } => {
            let over_generator = matches!(
                &input.node,
                PlanNode::Generate { names, arg, .. }
                    if names.as_slice() == [WINDOW.to_owned()]
                        && call_of(arg, "__time_windows").is_some()
            );
            if over_generator {
                return exprs.iter().any(|e| match &e.node {
                    ExprNode::Alias { expr, name: n } => {
                        n.eq_ignore_ascii_case(name)
                            && matches!(&expr.node, ExprNode::Column(c) if c.name == [WINDOW])
                    }
                    ExprNode::Column(c) => {
                        c.name.len() == 1
                            && c.name[0].eq_ignore_ascii_case(name)
                            && name.eq_ignore_ascii_case(WINDOW)
                    }
                    _ => false,
                });
            }
            // A projection that forwards a column (`SELECT a.iw AS w FROM (..) a`): the output `name` is the
            // window column when the input column it reads is.
            exprs.iter().any(|e| {
                let (out, read) = match &e.node {
                    ExprNode::Alias { expr, name: n } => match &expr.node {
                        ExprNode::Column(c) => (n.as_str(), c),
                        _ => return false,
                    },
                    ExprNode::Column(c) => (c.name.last().map_or("", String::as_str), c),
                    _ => return false,
                };
                out.eq_ignore_ascii_case(name)
                    && read
                        .name
                        .last()
                        .is_some_and(|inner| is_window_column(input, inner))
            })
        }
        _ => false,
    }
}

/// The output column names of a projection or aggregate (an alias, or a bare column's own name; `""` for an
/// expression with neither), seen through derived-table aliases; empty for any other node.
fn output_names(p: &Arc<LogicalPlan>) -> Vec<String> {
    match &p.node {
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => output_names(input),
        PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. }
        | PlanNode::Offset { input, .. } => output_names(input),
        PlanNode::SubqueryAlias { columns, .. } => columns.clone(),
        PlanNode::Project { exprs, .. }
        | PlanNode::Aggregate {
            aggregates: exprs, ..
        } => exprs
            .iter()
            .map(|e| match &e.node {
                ExprNode::Alias { name, .. } => name.clone(),
                ExprNode::Column(c) => c.name.last().cloned().unwrap_or_default(),
                _ => String::new(),
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// One window column `name` over `base`: the windows of `time` under the durations of `f`.
fn single_window(
    base: &Arc<LogicalPlan>,
    f: &FunctionCall,
    name: &str,
    time: Expr,
    origin: crate::Origin,
) -> Result<Arc<LogicalPlan>, AnalysisError> {
    if !(2..=4).contains(&f.args.len()) {
        return Err(AnalysisError::new(
            WrongNumArgsWithoutSuggestion {
                function_name: "`window`".to_owned(),
                expected_num: "[2, 3, 4]".to_owned(),
                actual_num: f.args.len().to_string(),
                docroot: SPARK_DOC_ROOT.to_owned(),
            },
            origin,
        ));
    }
    if call_of(&time, WINDOW).is_some() {
        return Err(unsupported("windows nested more than one level", origin));
    }
    let window = duration(&f.args[1], origin)?;
    let slide = match f.args.get(2) {
        Some(a) => duration(a, origin)?,
        None => window,
    };
    let start = match f.args.get(3) {
        Some(a) => duration(a, origin)?,
        None => 0,
    };
    let expr = shown(f);
    let range = |name: &str, value: i64| {
        AnalysisError::new(
            DatatypeMismatchValueOutOfRange {
                expr_name: format!("`{name}`"),
                value_range: format!("(0, {}]", i64::MAX),
                current_value: format!("{value}L"),
                sql_expr: expr.clone(),
            },
            origin,
        )
    };
    let constraint = |l: (&str, i64), op: &str, r: (&str, i64)| {
        AnalysisError::new(
            DatatypeMismatchParameterConstraintViolation {
                left_expr_name: format!("`{}`", l.0),
                left_expr_value: format!("{}L", l.1),
                constraint: op.to_owned(),
                right_expr_name: format!("`{}`", r.0),
                right_expr_value: format!("{}L", r.1),
                sql_expr: expr.clone(),
            },
            origin,
        )
    };
    if window <= 0 {
        return Err(range("window_duration", window));
    }
    if slide <= 0 {
        return Err(range("slide_duration", slide));
    }
    if slide > window {
        return Err(constraint(
            ("slide_duration", slide),
            "<=",
            ("window_duration", window),
        ));
    }
    if start.unsigned_abs() >= slide.unsigned_abs() {
        return Err(constraint(
            ("abs(start_time)", start.saturating_abs()),
            "<",
            ("slide_duration", slide),
        ));
    }
    let windows = Expr::function(
        "__time_windows",
        vec![time, lit_long(window), lit_long(slide), lit_long(start)],
    )
    .with_origin(origin);
    Ok(Arc::new(
        LogicalPlan::new(PlanNode::Generate {
            input: Arc::clone(base),
            kind: GeneratorKind::Explode,
            outer: false,
            arg: Arc::new(windows),
            output: Vec::new(),
            names: vec![name.to_owned()],
            qualifier: None,
        })
        .with_origin(origin),
    ))
}

/// The input of `p` with the session window: see the module documentation.
fn session_input(
    p: &Arc<LogicalPlan>,
    input: &Arc<LogicalPlan>,
    f: &FunctionCall,
    wkey: crate::StructuralHash,
    origin: crate::Origin,
) -> Result<Arc<LogicalPlan>, AnalysisError> {
    if f.args.len() != 2 {
        return Err(AnalysisError::new(
            WrongNumArgsWithoutSuggestion {
                function_name: "`session_window`".to_owned(),
                expected_num: "2".to_owned(),
                actual_num: f.args.len().to_string(),
                docroot: SPARK_DOC_ROOT.to_owned(),
            },
            origin,
        ));
    }
    let time = (*f.args[0]).clone();
    let gap = Expr::new(ExprNode::Cast {
        expr: Arc::clone(&f.args[1]),
        to: DataType::CalendarInterval,
        mode: CastMode::Default,
    })
    .with_origin(origin);
    let end = Expr::function(Builtin::Add.name(), vec![time.clone(), gap]).with_origin(origin);
    // The grouping key that is the session window itself merges the rows' ranges.
    let direct = match &p.node {
        PlanNode::Aggregate { grouping, .. } => {
            let exprs: &[Arc<Expr>] = match grouping {
                Grouping::GroupBy(e) | Grouping::Rollup(e) | Grouping::Cube(e) => e,
                Grouping::GroupingSets { exprs, .. } => exprs,
            };
            let at = exprs
                .iter()
                .position(|e| is_window_call(e) && key(e, origin).is_ok_and(|k| k == wkey));
            at.map(|i| (i, grouping))
        }
        _ => None,
    };
    let Some((at, grouping)) = direct else {
        // One range per row.
        let range = Expr::function(
            "named_struct",
            vec![
                Expr::literal(Literal::String("start".to_owned())),
                time.clone(),
                Expr::literal(Literal::String("end".to_owned())),
                end,
            ],
        )
        .with_origin(origin);
        let with = project(
            Arc::clone(input),
            vec![alias(range, SESSION, origin), star(origin)],
            origin,
        );
        let s = |f: &str| Expr::column([SESSION, f]);
        let keep = Expr::function(
            Builtin::And.name(),
            vec![
                Expr::function(Builtin::IsNotNull.name(), vec![time]),
                Expr::function(Builtin::Gt.name(), vec![s("end"), s("start")]),
            ],
        )
        .with_origin(origin);
        return Ok(Arc::new(
            LogicalPlan::new(PlanNode::Filter {
                input: with,
                predicate: Arc::new(keep),
            })
            .with_origin(origin),
        ));
    };
    let Grouping::GroupBy(exprs) = grouping else {
        return Err(AnalysisError::new(
            InternalError {
                message: "Grouping Session should not be null.".to_owned(),
            },
            origin,
        ));
    };
    let others: Vec<Arc<Expr>> = exprs
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != at)
        .map(|(_, e)| Arc::clone(e))
        .collect();
    if others.iter().any(|e| match &e.node {
        ExprNode::Literal(_) => true,
        ExprNode::Column(c) => c.name.len() == 1 && c.name[0].eq_ignore_ascii_case("all"),
        _ => false,
    }) {
        return Err(unsupported(
            "session_window with a positional or ALL grouping key",
            origin,
        ));
    }
    let (start, end_col, prev, id) = ("__sw_start", "__sw_end", "__sw_prev", "__sw_id");
    let ordered = vec![SortOrder {
        expr: col(start, origin),
        direction: SortDirection::Ascending,
        nulls: NullOrdering::NullsFirst,
    }];
    let over = |function: Expr, partition: Vec<Arc<Expr>>, order: Vec<SortOrder>, frame| {
        Expr::new(ExprNode::Window(WindowExpr {
            function: Arc::new(function),
            window: WindowSpecRef::Spec(WindowSpec {
                partition_by: partition,
                order_by: order,
                frame,
            }),
        }))
        .with_origin(origin)
    };
    let seen = Some(WindowFrame {
        units: FrameUnits::Rows,
        lower: FrameBound::UnboundedPreceding,
        upper: FrameBound::Offset(Arc::new(Expr::function(
            Builtin::Negative.name(),
            vec![Expr::literal(Literal::Integer(1))],
        ))),
    });
    let running = Some(WindowFrame::running(FrameUnits::Rows));
    // Rows with a time and a positive range.
    let ranged = project(
        Arc::clone(input),
        vec![
            star(origin),
            alias(time, start, origin),
            alias(end, end_col, origin),
        ],
        origin,
    );
    let keep = Expr::function(
        Builtin::And.name(),
        vec![
            Expr::function(Builtin::IsNotNull.name(), vec![Expr::column([start])]),
            Expr::function(
                Builtin::Gt.name(),
                vec![Expr::column([end_col]), Expr::column([start])],
            ),
        ],
    )
    .with_origin(origin);
    let kept = Arc::new(
        LogicalPlan::new(PlanNode::Filter {
            input: ranged,
            predicate: Arc::new(keep),
        })
        .with_origin(origin),
    );
    // The latest end of the earlier rows of the partition.
    let before = project(
        kept,
        vec![
            star(origin),
            alias(
                over(
                    Expr::function("max", vec![Expr::column([end_col])]),
                    others.clone(),
                    ordered.clone(),
                    seen,
                ),
                prev,
                origin,
            ),
        ],
        origin,
    );
    // The session number: the sessions started so far.
    let fresh = Expr::function(
        "when",
        vec![
            Expr::function(
                Builtin::Or.name(),
                vec![
                    Expr::function(Builtin::IsNull.name(), vec![Expr::column([prev])]),
                    Expr::function(
                        Builtin::Gt.name(),
                        vec![Expr::column([start]), Expr::column([prev])],
                    ),
                ],
            ),
            Expr::literal(Literal::Integer(1)),
            Expr::literal(Literal::Integer(0)),
        ],
    );
    let numbered = project(
        before,
        vec![
            star(origin),
            alias(
                over(
                    Expr::function("sum", vec![fresh]),
                    others.clone(),
                    ordered,
                    running,
                ),
                id,
                origin,
            ),
        ],
        origin,
    );
    // The session's range: its first start and its last end.
    let mut by_session = others;
    by_session.push(col(id, origin));
    let range = Expr::function(
        "named_struct",
        vec![
            Expr::literal(Literal::String("start".to_owned())),
            over(
                Expr::function("min", vec![Expr::column([start])]),
                by_session.clone(),
                Vec::new(),
                None,
            ),
            Expr::literal(Literal::String("end".to_owned())),
            over(
                Expr::function("max", vec![Expr::column([end_col])]),
                by_session,
                Vec::new(),
                None,
            ),
        ],
    )
    .with_origin(origin);
    Ok(project(
        numbered,
        vec![alias(range, SESSION, origin), star(origin)],
        origin,
    ))
}
