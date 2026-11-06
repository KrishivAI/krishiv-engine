// SPDX-License-Identifier: Apache-2.0
//! `ResolvePivot`: an unresolved PIVOT node becomes aggregates, along
//! Spark's two paths (`PivotTransformer`, `ResolvePivot` before 4.2):
//!
//! - When every aggregate's type is one `PivotFirst` takes (numeric,
//!   boolean, decimal), Spark's `PivotFirst` plan: the aggregates grouped by
//!   the grouping columns and the pivot column, then one cell per pivot
//!   value picked from that value's row. A cell with no matching row is
//!   NULL, whatever the aggregate (`count` too).
//! - Otherwise the `If` fallback: for every pivot value and every
//!   aggregate, `agg(if(pivot_col <=> value, expr, NULL))` over the
//!   unresolved aggregate (the children of every aggregate call wrapped, as
//!   Spark's `transformDown` does; `first` / `last` also flip to ignoring
//!   nulls, for the nulls the `if` introduces), so `count` of no matching
//!   row is 0.
//!
//! Both bind with the existing aggregate machinery: no second aggregate
//! binder. The grouping is every input column neither the pivot columns
//! nor the aggregates reference, in input order; the output names are
//! Spark's (`value` for one aggregate, `value_aggalias` or
//! `value_<aggregate sql>` otherwise).
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
use super::exprs::{Bound, Cx, WindowUse, pretty};
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
                // The exact fold: a fractional value is not truncated to
                // `INT` here (`CAST('NaN' AS DOUBLE)` is a valid value).
                let folded = self.fold_array(&t.expr, ansi);
                if folded.is_none() && !self.foldable(&t.expr) {
                    return Err(AnalysisError::new(
                        NonLiteralPivotValues {
                            expression: value_sql(&b.expr, &v.alias, self.ctx.conf.session_zone),
                        },
                        b.expr.origin,
                    ));
                }
                if let Some(Err(e)) = folded {
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
        // Spark evaluates every value cast to the pivot type at analysis, on
        // both paths: a cast that raises fails the query on any input.
        for (_, value) in &casts {
            let bound = self.bind(value, &cx)?;
            if let Some(Err(e)) = self.fold_array(&bound.expr, ansi) {
                return Err(AnalysisError::new(e, value.origin));
            }
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
        let zone = self.ctx.conf.session_zone;
        let single = items.len() == 1;
        if items
            .iter()
            .all(|b| pivot_first_type(b.ty.data_type.as_ref()))
        {
            return self.pivot_first(p, &c, &cx, &pivots[0], &group, &items, &casts, env);
        }
        let pivot_ref = Arc::clone(&pivot_columns[0]);
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

    /// Spark's `PivotFirst` path, taken when every aggregate's type is one
    /// `PivotFirst.supportsDataType` accepts: the aggregates run grouped by
    /// the grouping columns and the pivot column, then a second aggregate
    /// over the grouping columns picks, for each pivot value, the row of
    /// that value. A cell no input row matches is NULL (Spark's buffer slot
    /// is never written), every cell is nullable, and the aggregates'
    /// arguments are evaluated on every input row (stage one).
    ///
    /// The cells follow Spark's slot map (`PivotFirst.pivotIndex`: each
    /// distinct value keyed to its last index, `indexSize` slots, output
    /// `i` reading slot `i`): a NaN value is its own key and never matches
    /// (Scala's numeric `==`); `0.0` and `-0.0` are one key; duplicate
    /// values leave fewer slots than outputs, so an output past the slots
    /// raises `INVALID_ARRAY_INDEX` when a row is produced with ANSI on and
    /// is NULL with ANSI off, and a slot no value's last index names is
    /// NULL. Before 4.2 a binary pivot column is keyed by array identity, so
    /// no row matches and every cell is NULL. Refused, never answered
    /// wrong: several values the analyzer cannot compare (a value of a
    /// type it does not evaluate that is not a literal of the pivot
    /// column's type), a collated pivot column before 4.2 (Spark keys it by
    /// bytes, not by collation), and, with ANSI off and several aggregates,
    /// duplicates whose last index is past the slots (Spark writes that
    /// value into another aggregate's slots).
    #[expect(clippy::too_many_arguments, reason = "the resolved parts of one PIVOT")]
    fn pivot_first(
        &mut self,
        p: &LogicalPlan,
        c: &Rel,
        cx: &Cx<'_>,
        pivot: &Bound,
        group: &[Bound],
        items: &[Bound],
        casts: &[(String, Arc<Expr>)],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let origin = p.origin;
        let ansi = self.ctx.conf.ansi;
        let pivot_type = pivot.ty.data_type.clone().unwrap_or(DataType::Null);
        let profile = self.ctx.profile;
        let before_4_2 = profile < super::SparkProfile::V4_2;
        if before_4_2 && matches!(pivot_type, DataType::CollatedString(_)) {
            return Err(unsupported(
                "PIVOT on a collated column before Spark 4.2",
                origin,
            ));
        }
        // Before 4.2 a binary key is an array compared by identity: no value
        // equals another, and no row matches.
        let by_identity = before_4_2 && matches!(pivot_type, DataType::Binary);
        // A NULL value of a binary or collated pivot column: before 4.2 the
        // `HashMap` finds the NULL rows, in 4.2 the `TreeMap` compares the
        // NULL key with the others (and fails); neither is reproduced.
        if matches!(pivot_type, DataType::Binary | DataType::CollatedString(_)) {
            for (_, value) in casts {
                let bound = self.bind(value, cx)?;
                let null = matches!(bound.expr.node, ExprNode::Literal(Literal::Null(_)))
                    || matches!(
                        self.fold_array(&bound.expr, ansi),
                        Some(Ok(a)) if arrow_array::Array::null_count(&a) > 0
                    );
                if null {
                    return Err(unsupported(
                        format!(
                            "a NULL PIVOT value for a {} pivot column",
                            pivot_type.catalog_string()
                        ),
                        value.origin,
                    ));
                }
            }
        }
        let mut keys: Vec<SlotKey> = Vec::with_capacity(casts.len());
        for (_, value) in casts {
            let bound = self.bind(value, cx)?;
            // A value whose cast raises was refused in `pivot`.
            let key = match self.fold_array(&bound.expr, ansi) {
                Some(Err(e)) => return Err(AnalysisError::new(e, value.origin)),
                _ if by_identity => SlotKey::Never,
                Some(Ok(a)) => SlotKey::of(&a),
                None => self.struct_key(&bound.expr, &pivot_type, ansi),
            };
            keys.push(key);
        }
        if keys.len() > 1 && keys.iter().any(|k| matches!(k, SlotKey::Opaque)) {
            return Err(unsupported(
                format!(
                    "PIVOT values of type {} that are not literals of that type",
                    pivot_type.catalog_string()
                ),
                origin,
            ));
        }
        // The slot of each value (its last index among equal values) and
        // the value each slot holds.
        let last: Vec<usize> = (0..keys.len())
            .map(|k| {
                (k..keys.len())
                    .rev()
                    .find(|&j| keys[j].same(&keys[k]))
                    .unwrap_or(k)
            })
            .collect();
        let slots = (0..keys.len()).filter(|&k| last[k] == k).count();
        let mut owner: Vec<Option<usize>> = vec![None; keys.len()];
        for (k, &l) in last.iter().enumerate() {
            if l < slots {
                owner[l] = Some(k);
            }
        }
        if !ansi && items.len() > 1 && last.iter().any(|&l| l >= slots) {
            return Err(unsupported(
                "duplicate PIVOT values with several aggregates and spark.sql.ansi.enabled=false",
                origin,
            ));
        }
        // Stage one: the aggregates per grouping columns and pivot value.
        let pivot_attr = self.attr("__pivot", &pivot.ty);
        let mut first_items = group.to_vec();
        first_items.push(Bound {
            expr: Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                    expr: Arc::clone(&pivot.expr),
                    output: pivot_attr,
                }))
                .with_origin(origin),
            ),
            ..pivot.clone()
        });
        for (k, b) in items.iter().enumerate() {
            let output = self.attr(&format!("__agg{k}"), &b.ty);
            first_items.push(Bound {
                expr: Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: Arc::clone(&b.expr),
                        output,
                    }))
                    .with_origin(origin),
                ),
                ..b.clone()
            });
        }
        let mut first_group = group.to_vec();
        first_group.push(pivot.clone());
        let n = group.len();
        let stage = self.build_aggregate(origin, c, first_group, &first_items, false, None, env)?;
        // Stage two: one cell per (value, aggregate), as Spark orders them.
        let out = Arc::clone(&stage.out);
        let rels = [&stage];
        let cx2 = Self::cx(&out, &rels, env, "Pivot");
        let second_group: Vec<Bound> = out[..n].iter().map(|a| column_bound(a, origin)).collect();
        let pivot_col = column_bound(&out[n], origin).expr;
        let zone = self.ctx.conf.session_zone;
        let single = items.len() == 1;
        let mut combined = second_group.clone();
        let never =
            || Arc::new(Expr::new(ExprNode::Literal(Literal::Boolean(false))).with_origin(origin));
        for (i, (value_name, _)) in casts.iter().enumerate() {
            // The value slot `i` holds, and whether its rows match.
            let read = owner.get(i).copied().flatten();
            let matches = match read {
                Some(k) if keys[k].matches() => call(
                    Builtin::EqNullSafe.name(),
                    vec![Arc::clone(&pivot_col), Arc::clone(&casts[k].1)],
                    origin,
                ),
                _ => never(),
            };
            for (k, original) in items.iter().enumerate() {
                let agg = column_bound(&out[n + 1 + k], origin).expr;
                let null =
                    Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None))).with_origin(origin));
                let picked = call("if", vec![Arc::clone(&matches), agg, null], origin);
                let yes = Arc::new(
                    Expr::new(ExprNode::Literal(Literal::Boolean(true))).with_origin(origin),
                );
                let mut cell = call("first", vec![picked, yes], origin);
                if i >= slots && ansi {
                    // Spark reads slot `i` of an array of `slots` entries:
                    // out of range, which ANSI raises (and is NULL without).
                    let entries = vec![Arc::clone(&cell); slots.max(1)];
                    let index = i32::try_from(i).unwrap_or(i32::MAX);
                    let index = Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Integer(index))).with_origin(origin),
                    );
                    cell = call(
                        "__subscript",
                        vec![call("array", entries, origin), index],
                        origin,
                    );
                }
                let bound = self.bind(&cell, &cx2)?;
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
                    .with_origin(origin),
                );
                combined.push(Bound { expr, ..bound });
            }
        }
        let global = second_group.is_empty();
        self.build_aggregate(origin, &stage, second_group, &combined, global, None, env)
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

/// A pivot value as Spark's `PivotFirst` slot map keys it: its exact
/// value, with `0.0` and `-0.0` one key (Scala's numeric `==`).
enum SlotKey {
    /// A NaN value: never equal to another key (numeric `==`), never found.
    Nan,
    /// A key compared by identity (a binary array before 4.2): equal to
    /// no other, never found.
    Never,
    /// A floating value, `-0.0` as `0.0`.
    Float(u64),
    /// Any other value, as its one-row array.
    Value(arrow_array::ArrayRef),
    /// A struct of constants (several pivot columns), field by field.
    Parts(Vec<SlotKey>),
    /// A literal of a type the analyzer does not evaluate (timestamp,
    /// binary), compared by value.
    Literal(Literal),
    /// A value the analyzer cannot compare.
    Opaque,
}

impl SlotKey {
    fn of(a: &arrow_array::ArrayRef) -> Self {
        use arrow_array::Array as _;
        use arrow_array::cast::AsArray as _;
        use arrow_array::types::{Float32Type, Float64Type};
        let float = match a.data_type() {
            _ if a.is_empty() || a.null_count() > 0 => None,
            arrow_schema::DataType::Float64 => Some(a.as_primitive::<Float64Type>().value(0)),
            arrow_schema::DataType::Float32 => {
                Some(f64::from(a.as_primitive::<Float32Type>().value(0)))
            }
            _ => None,
        };
        match float {
            Some(v) if v.is_nan() => Self::Nan,
            Some(v) => Self::Float((v + 0.0).to_bits()),
            None => Self::Value(Arc::clone(a)),
        }
    }

    /// The key of a value the analyzer does not evaluate: a timestamp or
    /// binary literal by its value, anything else opaque.
    fn literal(e: &Expr) -> Self {
        match &e.node {
            ExprNode::Literal(
                l @ (Literal::Timestamp(_) | Literal::TimestampNtz(_) | Literal::Binary(_)),
            ) => Self::Literal(l.clone()),
            _ => Self::Opaque,
        }
    }

    /// Whether two keys are one slot-map key.
    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => a == b,
            (Self::Value(a), Self::Value(b)) => a.to_data() == b.to_data(),
            (Self::Literal(a), Self::Literal(b)) => literal_eq(a, b),
            (Self::Parts(a), Self::Parts(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same(y))
            }
            _ => false,
        }
    }

    /// Whether rows of this value can be found in the map.
    const fn matches(&self) -> bool {
        !matches!(self, Self::Nan | Self::Never)
    }
}

impl Resolver<'_> {
    /// The key of a pivot value the folder does not value whole: a
    /// constant array or struct (`array(1, 2)`, several pivot columns'
    /// `('Java', 2013)`), cast to the pivot column's type. Keyed part by
    /// part, each part cast to its type in the pivot column's type and
    /// folded; opaque when a part cannot be compared exactly (a NaN, a part
    /// the folder does not value), so such values are refused.
    fn struct_key(&self, e: &Arc<Expr>, pivot_type: &DataType, ansi: bool) -> SlotKey {
        match &e.node {
            ExprNode::Cast { expr, to, mode } => self.part_key(expr, to, *mode, ansi),
            // A value of the pivot column's own type is not cast.
            _ => {
                let mode = if ansi {
                    CastMode::Ansi
                } else {
                    CastMode::Legacy
                };
                self.part_key(e, pivot_type, mode, ansi)
            }
        }
    }

    fn part_key(&self, e: &Arc<Expr>, to: &DataType, mode: CastMode, ansi: bool) -> SlotKey {
        let inner = trim_value_alias(e);
        let inner = match &inner.node {
            ExprNode::Cast { expr, .. } => Arc::clone(expr),
            _ => inner,
        };
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &inner.node {
            let parts: Option<Vec<(&Arc<Expr>, DataType)>> = match (f.id.name(), to) {
                ("named_struct", DataType::Struct(st)) => {
                    let values: Vec<&Arc<Expr>> = f.args.iter().skip(1).step_by(2).collect();
                    (values.len() == st.fields().len()).then(|| {
                        values
                            .into_iter()
                            .zip(st.fields())
                            .map(|(v, fl)| (v, fl.data_type().clone()))
                            .collect()
                    })
                }
                ("struct", DataType::Struct(st)) => {
                    (f.args.len() == st.fields().len()).then(|| {
                        f.args
                            .iter()
                            .zip(st.fields())
                            .map(|(v, fl)| (v, fl.data_type().clone()))
                            .collect()
                    })
                }
                ("array", DataType::Array(at)) => Some(
                    f.args
                        .iter()
                        .map(|v| (v, at.element_type().clone()))
                        .collect(),
                ),
                _ => None,
            };
            if let Some(parts) = parts {
                let mut keys = Vec::with_capacity(parts.len());
                for (part, ty) in parts {
                    let key = self.part_key(part, &ty, mode, ansi);
                    if !matches!(
                        key,
                        SlotKey::Float(_)
                            | SlotKey::Value(_)
                            | SlotKey::Literal(_)
                            | SlotKey::Parts(_)
                    ) {
                        return SlotKey::Opaque;
                    }
                    keys.push(key);
                }
                return SlotKey::Parts(keys);
            }
        }
        let cast = Arc::new(Expr::new(ExprNode::Cast {
            expr: Arc::clone(&inner),
            to: to.clone(),
            mode,
        }));
        match self.fold_array(&cast, ansi) {
            Some(Ok(a)) => SlotKey::of(&a),
            Some(Err(_)) => SlotKey::Opaque,
            None => SlotKey::literal(&inner),
        }
    }
}

/// Equality of two timestamp or binary literals by value.
fn literal_eq(a: &Literal, b: &Literal) -> bool {
    match (a, b) {
        (Literal::Timestamp(x), Literal::Timestamp(y))
        | (Literal::TimestampNtz(x), Literal::TimestampNtz(y)) => x == y,
        (Literal::Binary(x), Literal::Binary(y)) => x == y,
        _ => false,
    }
}

/// `PivotFirst.supportsDataType`: the aggregate result types Spark's
/// `PivotFirst` path takes (the same in 3.5, 4.0, 4.1 and 4.2).
fn pivot_first_type(t: Option<&DataType>) -> bool {
    matches!(
        t,
        Some(
            DataType::Boolean
                | DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Float
                | DataType::Double
                | DataType::Decimal(_)
        )
    )
}

/// An unresolved call of `name` on `args`.
fn call(name: &str, args: Vec<Arc<Expr>>, origin: Origin) -> Arc<Expr> {
    Arc::new(
        Expr::new(ExprNode::Function(FunctionCall {
            name: name.to_owned(),
            qualifier: Vec::new(),
            args,
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
            within_group: Vec::new(),
        }))
        .with_origin(origin),
    )
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
    float_name(t, r).unwrap_or_else(|| part_name(t, r))
}

/// A non-literal floating value named as Spark's cast to string prints its
/// exact value (`CAST('NaN' AS DOUBLE)` is `NaN`); `None` for any other.
fn float_name(t: &Bound, r: &Resolver<'_>) -> Option<String> {
    use arrow_array::Array as _;
    use arrow_array::cast::AsArray as _;
    use arrow_array::types::{Float32Type, Float64Type};
    if matches!(t.expr.node, ExprNode::Literal(_)) {
        return None;
    }
    let Some(Ok(a)) = r.fold_array(&t.expr, r.ctx.conf.ansi) else {
        return None;
    };
    if a.is_empty() || a.null_count() > 0 {
        return None;
    }
    let java = |v: f64, finite: &dyn Fn() -> String| {
        if v.is_nan() {
            "NaN".to_owned()
        } else if v.is_infinite() {
            if v > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
        } else {
            finite()
        }
    };
    match a.data_type() {
        arrow_schema::DataType::Float64 => {
            let v = a.as_primitive::<Float64Type>().value(0);
            Some(java(v, &|| crate::spark_fmt::double(v)))
        }
        arrow_schema::DataType::Float32 => {
            let v = a.as_primitive::<Float32Type>().value(0);
            Some(java(f64::from(v), &|| crate::spark_fmt::float(v)))
        }
        _ => None,
    }
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
            Literal::Null(_) => "null".to_owned(),
            // Spark's cast of binary to string takes the bytes as UTF-8.
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
