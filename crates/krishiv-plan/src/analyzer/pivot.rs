// SPDX-License-Identifier: Apache-2.0
//! `ResolvePivot`: an unresolved PIVOT node becomes an aggregate
//! of filtered aggregates, as Spark's `PivotTransformer` does (always its
//! non-`PivotFirst` path, chosen to match its results).
//!
//! For every pivot value and every aggregate the rule builds
//! `agg(if(pivot_col <=> value, expr, NULL))` over the unresolved aggregate
//! (the children of every aggregate call wrapped, as Spark's
//! `transformDown` does; `first` / `last` also flip to ignoring nulls, for
//! the nulls the `if` introduces) and binds it with the existing aggregate
//! machinery: no second aggregate binder. The grouping is every input
//! column neither the pivot columns nor the aggregates reference, in input
//! order; the aggregate's output names are Spark's (`value` for one
//! aggregate, `value_aggalias` or `value_<aggregate sql>` otherwise).
//!
//! Several pivot columns are one struct column, `struct(c1, c2)`, and a value
//! of several parts (`(2012, 'Java')`) one struct value, as Spark's parser
//! builds them (`CreateStruct`); the rule then treats them like any other
//! column and value. The pivot column must be orderable
//! (`INCOMPARABLE_PIVOT_COLUMN`: no map anywhere in its type), its values
//! foldable (`NON_LITERAL_PIVOT_VALUES`; arrays, structs and maps of
//! constants count) and castable to the pivot column type
//! (`PIVOT_VALUE_DATA_TYPE_MISMATCH`). A value names its outputs as Spark's
//! cast of the value to `STRING` prints it (`{2012, Java}`, `[1, 1]`).

use super::SessionZone;

use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::Arc;

use krishiv_common::error::{
    IncomparablePivotColumn, LegacyErrorTemp1006, NonLiteralPivotValues, PivotValueDataTypeMismatch,
};
use krishiv_types::DataType;

use super::bind::{addr, unsupported};
use super::coerce::{can_cast, orderable};
use super::exprs::{Bound, WindowUse, pretty};
use super::fold;
use super::ops::column_bound;
use super::resolve::{Attr, Env, Rel, Resolver};
use super::{AnalysisError, Constant};
use crate::{
    AttrId, Builtin, CastMode, Expr, ExprNode, FunctionCall, FunctionKind, Literal, LogicalPlan,
    Origin, PivotValue, ResolvedExpr, SortOrder,
};

impl Resolver<'_> {
    /// Resolves `p`: a `PIVOT` over `input` with `pivot_columns`, `values`
    /// and `aggregates`.
    pub(super) fn pivot(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        pivot_columns: &[Arc<Expr>],
        values: &[PivotValue],
        aggregates: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let (pivot_columns, values) = as_structs(pivot_columns, values);
        let (pivot_columns, values) = (pivot_columns.as_slice(), values.as_slice());
        let rels = [&c];
        let mut cx = Self::cx(&c.out, &rels, env, "Pivot");
        cx.windows = WindowUse::Allowed;
        if pivot_columns.is_empty() {
            return Err(unsupported("PIVOT without pivot columns", p.origin));
        }
        let mut pivots = Vec::with_capacity(pivot_columns.len());
        for col in pivot_columns {
            pivots.push(self.bind(col, &cx)?);
        }
        // Spark's `RowOrdering.isOrderable`, checked before the values.
        if let Some(t) = &pivots[0].ty.data_type
            && !orderable(t)
        {
            return Err(AnalysisError::new(
                IncomparablePivotColumn {
                    column_name: incomparable_name(&pivots[0].expr, &c.out),
                },
                p.origin,
            ));
        }
        let mut bound_values: Vec<Vec<Bound>> = Vec::with_capacity(values.len());
        for v in values {
            let mut parts = Vec::with_capacity(v.exprs.len());
            for e in &v.exprs {
                parts.push(self.bind(e, &cx)?);
            }
            bound_values.push(parts);
        }
        // Spark checks the values trimmed of aliases.
        let trimmed_values: Vec<Vec<Bound>> = bound_values
            .iter()
            .map(|parts| {
                parts
                    .iter()
                    .map(|b| Bound {
                        expr: trim_value_alias(&b.expr),
                        ..b.clone()
                    })
                    .collect()
            })
            .collect();
        // Spark's `PivotTransformer` checks every value is literal before
        // checking its type.
        let ansi = self.ctx.conf.ansi;
        for (i, v) in values.iter().enumerate() {
            for (b, t) in bound_values[i].iter().zip(&trimmed_values[i]) {
                let folded = self.fold(&t.expr, ansi);
                if folded.is_none() && !self.foldable(&t.expr) {
                    return Err(AnalysisError::new(
                        NonLiteralPivotValues {
                            expression: value_sql(&b.expr, &v.alias, self.ctx.conf.session_zone),
                        },
                        b.expr.origin,
                    ));
                }
                if let Some(Constant::Error(e)) = folded {
                    return Err(AnalysisError::new(e, b.expr.origin));
                }
            }
        }
        let pivot_type = pivots[0].ty.data_type.clone().unwrap_or(DataType::Null);
        // Spark's `Cast.canCast` check: the value type must cast to the
        // pivot column type, in the session's mode (the mode the `if`'s
        // cast below binds with, so a check that passes cannot fail
        // there).
        let mode = if self.ctx.conf.ansi {
            CastMode::Ansi
        } else {
            CastMode::Legacy
        };
        let mut casts: Vec<(String, Arc<Expr>)> = Vec::with_capacity(values.len());
        for (i, v) in values.iter().enumerate() {
            let (parts, trimmed) = (&bound_values[i], &trimmed_values[i]);
            let ([b], [t]) = (parts.as_slice(), trimmed.as_slice()) else {
                let (value, value_type) = value_render(v, parts, trimmed, self);
                return Err(pivot_type_mismatch(
                    value,
                    value_type,
                    pivot_type.catalog_string(),
                    p.origin,
                ));
            };
            let from = t.ty.data_type.as_ref();
            if can_cast(from, &pivot_type, mode).is_err() {
                let (value, value_type) = value_render(v, parts, trimmed, self);
                return Err(pivot_type_mismatch(
                    value,
                    value_type,
                    pivot_type.catalog_string(),
                    p.origin,
                ));
            }
            // A value of the pivot column's own type needs no cast (and
            // shows as the value, as Spark's implicit cast does).
            let value = if from == Some(&pivot_type) {
                trim_value_alias(&v.exprs[0])
            } else {
                Arc::new(
                    Expr::new(ExprNode::Cast {
                        expr: trim_value_alias(&v.exprs[0]),
                        to: pivot_type.clone(),
                        mode: CastMode::Default,
                    })
                    .with_origin(b.expr.origin),
                )
            };
            casts.push((value_name(v, b, t, self), value));
        }
        // The aggregates, bound by the existing machinery (which rejects a
        // nested aggregate, as Spark's `CheckAnalysis` does), then checked
        // to be aggregates at all, as Spark's
        // `checkValidAggregateExpression` checks the resolved expressions.
        let items = self.select_items(&c, aggregates, env, "Pivot")?;
        for b in &items {
            check_aggregate(&b.expr, &c.out)?;
        }
        let mut refs = HashSet::new();
        for b in pivots.iter().chain(items.iter()) {
            column_refs(&b.expr, &mut refs);
        }
        let group: Vec<Bound> = c
            .out
            .iter()
            .filter(|a| !refs.contains(&a.col.id))
            .map(|a| column_bound(a, p.origin))
            .collect();
        let global = group.is_empty();
        let pivot_ref = Arc::clone(&pivot_columns[0]);
        let zone = self.ctx.conf.session_zone;
        let single = items.len() == 1;
        let mut combined: Vec<Bound> = group.clone();
        for (value_name, cast) in &casts {
            let cond = Cond {
                pivot: Arc::clone(&pivot_ref),
                value: Arc::clone(cast),
            };
            for (original, unresolved) in items.iter().zip(aggregates.iter()) {
                let filtered = self.filter_aggregate(unresolved, &cond);
                let bound = self.bind(&filtered, &cx)?;
                let name = if single {
                    value_name.clone()
                } else {
                    format!("{value_name}_{}", agg_suffix(&original.expr, zone))
                };
                let output = self.attr(&name, &bound.ty);
                let expr = Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: bound.expr,
                        output,
                    }))
                    .with_origin(p.origin),
                );
                combined.push(Bound { expr, ..bound });
            }
        }
        // The grouping columns ride as items, as in `SELECT year,
        // sum(...) GROUP BY year`: the aggregate's output is the grouping
        // columns, then one aggregate per (value, aggregate) pair.
        self.build_aggregate(p.origin, &c, group, &combined, global, None, env)
    }

    /// Filters one unresolved aggregate: every aggregate call's children
    /// wrapped in `if(pivot_col <=> value, child, NULL)`, as Spark's
    /// `transformDown` wraps them. `first` / `last` (and their `_value`
    /// aliases, which the registry also declares as aggregates) also flip
    /// to ignoring nulls, for the nulls the `if` introduces; `count(*)`
    /// becomes `count(if(pivot_col <=> value, 1, NULL))`, counting matching
    /// rows. The tree stays fully unresolved so
    /// the re-bind types everything (the binder keeps resolved parents as
    /// they are).
    fn filter_aggregate(&self, e: &Arc<Expr>, cond: &Cond) -> Arc<Expr> {
        Arc::clone(e)
            .transform_down(&mut |x: Arc<Expr>| -> Result<Arc<Expr>, Infallible> {
                let ExprNode::Function(f) = &x.node else {
                    return Ok(x);
                };
                if f.name.eq_ignore_ascii_case("count")
                    && matches!(&f.args[..], [a] if is_bare_star(a))
                {
                    let one =
                        Expr::new(ExprNode::Literal(Literal::Integer(1))).with_origin(x.origin);
                    return Ok(Arc::new(
                        Expr::new(ExprNode::Function(FunctionCall {
                            name: f.name.clone(),
                            qualifier: f.qualifier.clone(),
                            args: vec![if_expr(cond, Arc::new(one), x.origin)],
                            arg_names: Vec::new(),
                            distinct: f.distinct,
                            filter: f.filter.clone(),
                            within_group: Vec::new(),
                        }))
                        .with_origin(x.origin),
                    ));
                }
                if !self.is_aggregate(&x) {
                    return Ok(x);
                }
                let (args, arg_names) = if matches!(
                    f.name.to_ascii_lowercase().as_str(),
                    "first" | "last" | "first_value" | "last_value"
                ) {
                    let mut args = Vec::with_capacity(2);
                    if let Some((head, _)) = f.args.split_first() {
                        args.push(if_expr(cond, Arc::clone(head), x.origin));
                        args.push(Arc::new(
                            Expr::new(ExprNode::Literal(Literal::Boolean(true)))
                                .with_origin(x.origin),
                        ));
                    }
                    let arg_names = if f.arg_names.is_empty() {
                        Vec::new()
                    } else {
                        let mut names = vec![f.arg_names.first().cloned().flatten(), None];
                        names.truncate(args.len());
                        if names.iter().all(Option::is_none) {
                            Vec::new()
                        } else {
                            names
                        }
                    };
                    (args, arg_names)
                } else {
                    // Every argument takes the condition, as Spark's
                    // `transformDown` wraps them: a `listagg` delimiter or
                    // a `mode` flag stops being foldable and is refused by
                    // the re-bind, as Spark refuses it.
                    let args: Vec<Arc<Expr>> = f
                        .args
                        .iter()
                        .map(|a| if_expr(cond, Arc::clone(a), x.origin))
                        .collect();
                    let arg_names = if f.arg_names.is_empty() {
                        Vec::new()
                    } else {
                        f.arg_names.clone()
                    };
                    (args, arg_names)
                };
                Ok(Arc::new(
                    Expr::new(ExprNode::Function(FunctionCall {
                        name: f.name.clone(),
                        qualifier: f.qualifier.clone(),
                        args,
                        arg_names,
                        distinct: f.distinct,
                        filter: f.filter.clone(),
                        // The ordering keys are children of the call, so
                        // they take the same condition as its arguments.
                        within_group: f
                            .within_group
                            .iter()
                            .map(|o| SortOrder {
                                expr: if_expr(cond, Arc::clone(&o.expr), x.origin),
                                direction: o.direction,
                                nulls: o.nulls,
                            })
                            .collect(),
                    }))
                    .with_origin(x.origin),
                ))
            })
            .unwrap_or_else(|never| match never {})
    }
}

impl Resolver<'_> {
    /// Spark's `foldable` for a pivot value: a literal, or a deterministic
    /// scalar call or cast over foldable arguments (`array(1, 1)`,
    /// `struct(1, 'a')`, `map('1', 1)`), which the constant folder may not
    /// value.
    fn foldable(&self, e: &Arc<Expr>) -> bool {
        match &e.node {
            ExprNode::Literal(_) => true,
            ExprNode::Cast { expr, .. }
            | ExprNode::Alias { expr, .. }
            | ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => self.foldable(expr),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                f.kind == FunctionKind::Scalar
                    && f.deterministic
                    && f.filter.is_none()
                    && f.args.iter().all(|a| self.foldable(a))
            }
            _ => fold::is_literal(e) || self.fold(e, self.ctx.conf.ansi).is_some(),
        }
    }
}

/// Spark's parser's pivot shapes: several pivot columns are one
/// `struct(c1, c2, ...)` column and a value of several parts one
/// `struct(v1, v2, ...)` value (`CreateStruct`).
fn as_structs(columns: &[Arc<Expr>], values: &[PivotValue]) -> (Vec<Arc<Expr>>, Vec<PivotValue>) {
    let call = |name: &str, args: &[Arc<Expr>]| {
        let origin = args.first().map_or(Origin::UNKNOWN, |a| a.origin);
        Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                name: name.to_owned(),
                qualifier: Vec::new(),
                args: args.to_vec(),
                arg_names: Vec::new(),
                distinct: false,
                filter: None,
                within_group: Vec::new(),
            }))
            .with_origin(origin),
        )
    };
    let columns = if columns.len() > 1 {
        vec![call("struct", columns)]
    } else {
        columns.to_vec()
    };
    let values = values
        .iter()
        .map(|v| {
            if v.exprs.len() > 1 {
                // Spark's row constructor (`CreateStruct`): a column part
                // is named as the column (an alias as itself), any other
                // `col<i>`; the type `struct(a, ...)` has, and its text.
                let args: Vec<Arc<Expr>> = v
                    .exprs
                    .iter()
                    .enumerate()
                    .flat_map(|(i, e)| {
                        let field = match &e.node {
                            ExprNode::Column(c) if let Some(last) = c.name.last() => last.clone(),
                            ExprNode::Alias { name, .. } => name.clone(),
                            _ => format!("col{}", i + 1),
                        };
                        let name = Expr::new(ExprNode::Literal(Literal::String(field)))
                            .with_origin(e.origin);
                        [Arc::new(name), Arc::clone(e)]
                    })
                    .collect();
                PivotValue {
                    exprs: vec![call("named_struct", &args)],
                    alias: v.alias.clone(),
                }
            } else {
                v.clone()
            }
        })
        .collect();
    (columns, values)
}

/// The `columnName` of `INCOMPARABLE_PIVOT_COLUMN`: Spark's `toSQLId` of
/// the pivot column's `sql`. A column prints as its name; the struct of
/// several columns as `named_struct('a', q.a, ...)`, which `toSQLId` splits
/// at its dots and quotes part by part.
fn incomparable_name(e: &Arc<Expr>, scope: &[Attr]) -> String {
    let quote = |s: &str| format!("`{}`", s.replace('`', "``"));
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => quote(&c.name),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            // `named_struct` arguments come in (name literal, value) pairs.
            let fields: Vec<String> = f
                .args
                .chunks(2)
                .map(|pair| match pair {
                    [_, v] if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &v.node => {
                        let sql = scope
                            .iter()
                            .find(|x| x.col.id == c.id)
                            .map_or_else(|| c.name.clone(), attr_sql);
                        format!("'{}', {sql}", c.name)
                    }
                    [k, v] => format!("{}, {}", pretty(k, None), pretty(v, None)),
                    other => other
                        .iter()
                        .map(|a| pretty(a, None))
                        .collect::<Vec<_>>()
                        .join(", "),
                })
                .collect();
            let sql = format!("named_struct({})", fields.join(", "));
            sql.split('.').map(quote).collect::<Vec<_>>().join(".")
        }
        _ => quote(&pretty(e, None)),
    }
}

/// A bare `*`, as `count(*)` writes it (see the binder's `star_args`).
fn is_bare_star(e: &Expr) -> bool {
    matches!(&e.node, ExprNode::Star(s) if s.qualifier.is_empty() && s.plan_id.is_none())
}

/// A pivot value trimmed of aliases, as Spark's `trimAliases` trims
/// them before checking and evaluating it.
fn trim_value_alias(e: &Arc<Expr>) -> Arc<Expr> {
    let mut x = Arc::clone(e);
    loop {
        let inner = match &x.node {
            ExprNode::Alias { expr, .. } | ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => {
                Arc::clone(expr)
            }
            _ => return x,
        };
        x = inner;
    }
}

/// The pivot column and the cast value one `if` compares.
struct Cond {
    pivot: Arc<Expr>,
    value: Arc<Expr>,
}

/// `if(pivot_col <=> value, then, NULL)`.
fn if_expr(cond: &Cond, then: Arc<Expr>, origin: Origin) -> Arc<Expr> {
    let eq = Expr::new(ExprNode::Function(FunctionCall {
        name: Builtin::EqNullSafe.name().to_owned(),
        qualifier: Vec::new(),
        args: vec![Arc::clone(&cond.pivot), Arc::clone(&cond.value)],
        arg_names: Vec::new(),
        distinct: false,
        filter: None,
        within_group: Vec::new(),
    }))
    .with_origin(origin);
    Arc::new(
        Expr::new(ExprNode::Function(FunctionCall {
            name: "if".to_owned(),
            qualifier: Vec::new(),
            args: vec![
                Arc::new(eq),
                then,
                Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None))).with_origin(origin)),
            ],
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
            within_group: Vec::new(),
        }))
        .with_origin(origin),
    )
}

/// Spark's `checkValidAggregateExpression` on a bound aggregate: an
/// aggregate call is fine (its arguments are not visited, so a nested
/// aggregate stays for the aggregate machinery to reject); a bare column
/// is `aggregateExpressionRequiredForPivotError`, with the attribute as
/// Spark's `sql` prints it; anything else is visited.
fn check_aggregate(e: &Arc<Expr>, scope: &[Attr]) -> Result<(), AnalysisError> {
    let mut stack = vec![Arc::clone(e)];
    let mut seen = HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(addr(&*x)) {
            continue;
        }
        match &x.node {
            ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {}
            ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
                let sql = scope
                    .iter()
                    .find(|a| a.col.id == c.id)
                    .map_or_else(|| c.name.clone(), attr_sql);
                return Err(AnalysisError::new(LegacyErrorTemp1006 { sql }, x.origin));
            }
            _ => x.for_each_child(&mut |k| stack.push(Arc::clone(k))),
        }
    }
    Ok(())
}

/// The attribute as Spark's `Attribute.sql` prints it: the qualifier and
/// the name. The qualifier is the last part (a table read carries the
/// catalog and namespace too, which Spark's temporary-view qualifier has
/// not).
fn attr_sql(a: &Attr) -> String {
    match a.qual.last() {
        Some(qual) => format!("{qual}.{}", a.col.name),
        None => a.col.name.clone(),
    }
}

/// The attribute ids `e` references.
fn column_refs(e: &Arc<Expr>, refs: &mut HashSet<AttrId>) {
    let mut stack = vec![Arc::clone(e)];
    let mut seen = HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) = &x.node
        {
            refs.insert(c.id);
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
}

/// The `expression` parameter of `NON_LITERAL_PIVOT_VALUES`: the value as
/// Spark's `toSQLExpr` prints it (double-quoted), keeping an alias as
/// Spark's `Alias.sql` does (the alias field, or an `Alias` node).
fn value_sql(e: &Arc<Expr>, alias: &Option<String>, zone: Option<SessionZone>) -> String {
    // A resolved alias prints its own `AS name`.
    let name = match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { .. }) => None,
        _ => alias.clone(),
    };
    let inner = match name {
        Some(name) => format!("{} AS {}", pretty(e, zone), quote_if_needed(&name)),
        None => pretty(e, zone),
    };
    format!("\"{inner}\"")
}

/// Spark's `quoteIfNeeded`: a name of letters, digits and `_` (not all
/// digits) stays bare.
pub(super) fn quote_if_needed(name: &str) -> String {
    let bare = !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.chars().all(|c| c.is_ascii_digit());
    if bare {
        name.to_owned()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}

/// The output name of one pivot value: the alias field, an `Alias` node
/// (as Spark's parser builds aliased values), or the single trimmed part
/// rendered as Spark's `Cast(value, StringType)` renders it.
fn value_name(v: &PivotValue, b: &Bound, t: &Bound, r: &Resolver<'_>) -> String {
    if let Some(alias) = &v.alias {
        return alias.clone();
    }
    if let ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) = &b.expr.node {
        return output.name.clone();
    }
    part_name(t, r)
}

/// One value (or part) rendered: a literal prints as Spark's cast to
/// string prints it (dates and timestamps without their keywords); any
/// other foldable expression prints its folded constant, or as written
/// when it has no exact constant here.
fn part_name(b: &Bound, r: &Resolver<'_>) -> String {
    if let Some(text) = nested_string(&b.expr, r) {
        return text;
    }
    let zone = r.ctx.conf.session_zone;
    if let ExprNode::Literal(v) = &b.expr.node {
        return match v {
            Literal::Date(days) => crate::spark_fmt::date_value(*days),
            Literal::Timestamp(us) => {
                crate::spark_fmt::timestamp_value(SessionZone::local_micros(zone, *us))
            }
            Literal::TimestampNtz(us) => crate::spark_fmt::timestamp_value(*us),
            // Spark's cast of a binary to a string: the bytes as UTF-8.
            Literal::Binary(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            _ => pretty(&b.expr, zone),
        };
    }
    match r.fold(&b.expr, r.ctx.conf.ansi) {
        Some(Constant::Null) => "null".to_owned(),
        Some(Constant::Boolean(v)) => v.to_string(),
        Some(Constant::Integral(v)) => v.to_string(),
        Some(Constant::String(v)) => v,
        // Anything else (a folding error is raised by the validation
        // above) prints as written, before implicit casts.
        _ => pretty(&b.shown(), zone),
    }
}

/// An array, struct or map of constants as Spark's cast to `STRING` prints
/// it (`[1, 2]`, `{1, a}`, `{1 -> a}`; a null element is `null`); `None`
/// for anything else.
fn nested_string(e: &Arc<Expr>, r: &Resolver<'_>) -> Option<String> {
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
        return None;
    };
    let element = |a: &Arc<Expr>| -> String {
        if let ExprNode::Literal(Literal::Null(_)) = &a.node {
            return "null".to_owned();
        }
        let bound = Bound {
            expr: Arc::clone(a),
            ty: super::catalog::ArgType {
                data_type: None,
                nullable: true,
            },
            agg: None,
            win: None,
            outer: None,
            lateral: Vec::new(),
            shown: None,
        };
        part_name(&bound, r)
    };
    let joined = |items: &[Arc<Expr>]| items.iter().map(element).collect::<Vec<_>>().join(", ");
    match f.id.name() {
        "array" => Some(format!("[{}]", joined(&f.args))),
        "struct" => Some(format!("{{{}}}", joined(&f.args))),
        "named_struct" => {
            let values: Vec<Arc<Expr>> = f.args.iter().skip(1).step_by(2).cloned().collect();
            Some(format!("{{{}}}", joined(&values)))
        }
        "map" => {
            let pairs: Vec<String> = f
                .args
                .chunks(2)
                .map(|kv| match kv {
                    [k, v] => format!("{} -> {}", element(k), element(v)),
                    _ => String::new(),
                })
                .collect();
            Some(format!("{{{}}}", pairs.join(", ")))
        }
        _ => None,
    }
}

/// A value as Spark's `Expression.toString` prints it in
/// `PIVOT_VALUE_DATA_TYPE_MISMATCH` (checked against 4.2.0): a scalar as its
/// output name does, `array(1, 2)`, `map(1, 2)`, `named_struct(p, 1)` and
/// `struct(col1, 1, col2, q)`, strings without quotes.
fn value_text(e: &Arc<Expr>, r: &Resolver<'_>) -> String {
    if let ExprNode::Cast { expr, to, mode } = &e.node
        && *mode != CastMode::Try
    {
        return format!("cast({} as {})", value_text(expr, r), to.simple_string());
    }
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
        if let ExprNode::Literal(Literal::Null(_)) = &e.node {
            return "null".to_owned();
        }
        let bound = Bound {
            expr: Arc::clone(e),
            ty: super::catalog::ArgType {
                data_type: None,
                nullable: true,
            },
            agg: None,
            win: None,
            outer: None,
            lateral: Vec::new(),
            shown: None,
        };
        return part_name(&bound, r);
    };
    let args: Vec<String> = match f.id.name() {
        "array" | "map" | "named_struct" => f.args.iter().map(|a| value_text(a, r)).collect(),
        "struct" => f
            .args
            .iter()
            .enumerate()
            .flat_map(|(i, a)| [format!("col{}", i + 1), value_text(a, r)])
            .collect(),
        _ => return pretty(e, r.ctx.conf.session_zone),
    };
    format!("{}({})", f.id.name(), args.join(", "))
}

/// A whole value and its type for `PIVOT_VALUE_DATA_TYPE_MISMATCH`: one
/// part prints as itself, several as `named_struct(col1, .., col2, ..)`.
///
/// A value tuple `(a, b)` arrives as Spark's row constructor
/// (`as_structs`), which prints as `named_struct(col1, a, col2, b)`; a
/// written `struct(a, b)` prints as `struct(col1, a, col2, b)` (checked
/// against 4.2.0).
fn value_render(
    v: &PivotValue,
    parts: &[Bound],
    trimmed: &[Bound],
    r: &Resolver<'_>,
) -> (String, String) {
    if let ([b], [t]) = (parts, trimmed) {
        let collection = |e: &Expr| {
            matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Function(f))
                if matches!(f.id.name(), "array" | "map" | "struct" | "named_struct"))
        };
        let value = v.alias.clone().unwrap_or_else(|| match &b.expr.node {
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.name.clone(),
            _ if collection(&t.expr) => value_text(&t.expr, r),
            // An explicit cast of one prints as Spark's `Cast.toString`
            // does: `cast(array(1) as array<int>)`.
            _ if matches!(&t.expr.node, ExprNode::Cast { expr, .. } if collection(expr)) => {
                value_text(&t.expr, r)
            }
            // A scalar prints as its output name does (as written, before
            // implicit casts, when it does not fold).
            _ => part_name(t, r),
        });
        let ty =
            t.ty.data_type
                .as_ref()
                .map(DataType::simple_string)
                .unwrap_or_else(|| "void".to_owned());
        return (value, ty);
    }
    let value = v.alias.clone().unwrap_or_else(|| {
        format!(
            "named_struct({})",
            trimmed
                .iter()
                .enumerate()
                .map(|(i, b)| format!("col{}, {}", i + 1, value_text(&b.expr, r)))
                .collect::<Vec<_>>()
                .join(", ")
        )
    });
    let ty = format!(
        "struct<{}>",
        trimmed
            .iter()
            .enumerate()
            .map(|(i, b)| format!(
                "col{}:{}",
                i + 1,
                b.ty.data_type
                    .as_ref()
                    .map(DataType::simple_string)
                    .unwrap_or_else(|| "void".to_owned())
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    (value, ty)
}

/// `PIVOT_VALUE_DATA_TYPE_MISMATCH` with its message parameters.
fn pivot_type_mismatch(
    value: String,
    value_type: String,
    pivot_type: String,
    origin: Origin,
) -> AnalysisError {
    AnalysisError::new(
        PivotValueDataTypeMismatch {
            pivot_type,
            value,
            value_type,
        },
        origin,
    )
}

/// The suffix of a pivot output with several aggregates: the aggregate's
/// alias, or the aggregate as Spark's `toPrettySQL` prints it (which is
/// what the aggregate machinery names it).
fn agg_suffix(e: &Arc<Expr>, zone: Option<SessionZone>) -> String {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.name.clone(),
        _ => pretty(e, zone),
    }
}
