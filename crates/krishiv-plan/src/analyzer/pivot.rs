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
//! Pivot values must be foldable (`NON_LITERAL_PIVOT_VALUES`) and castable
//! to the pivot column type (`PIVOT_VALUE_DATA_TYPE_MISMATCH`). Several
//! pivot columns stand for a struct, which the engine cannot value yet:
//! they fail with `KRISHIV_UNSUPPORTED_FEATURE`, never a wrong result.

use super::SessionZone;

use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::Arc;

use krishiv_common::error::{
    LegacyErrorTemp1006, NonLiteralPivotValues, PivotValueDataTypeMismatch,
};
use krishiv_types::DataType;

use super::bind::{addr, unsupported};
use super::coerce::can_cast;
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
                if folded.is_none() && !fold::is_literal(&t.expr) {
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
        if pivot_columns.len() > 1 {
            return self.multi_pivot(p, &pivots, values, &bound_values, &trimmed_values);
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
        let mut keys: Vec<SlotKey> = Vec::with_capacity(casts.len());
        for (_, value) in casts {
            let bound = self.bind(value, cx)?;
            // Spark evaluates every value cast to the pivot type at
            // analysis: a cast that raises fails the query on any input.
            let key = match self.fold_array(&bound.expr, ansi) {
                Some(Err(e)) => return Err(AnalysisError::new(e, value.origin)),
                _ if by_identity => SlotKey::Never,
                Some(Ok(a)) => SlotKey::of(&a),
                None => SlotKey::literal(&bound.expr),
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

    /// Several pivot columns: Spark validates the values against the struct
    /// pivot type, which the engine cannot value; well-typed struct values
    /// fail as unsupported, never wrong. Values whose arity or per-part
    /// type fails Spark's check fail with `PIVOT_VALUE_DATA_TYPE_MISMATCH`
    /// first, as Spark does (a `STRING` part casts to anything, so a
    /// well-typed string-only mismatch still reaches the unsupported path;
    /// both are errors, never a wrong result).
    fn multi_pivot(
        &self,
        p: &LogicalPlan,
        pivots: &[Bound],
        values: &[PivotValue],
        bound: &[Vec<Bound>],
        trimmed: &[Vec<Bound>],
    ) -> Result<Rel, AnalysisError> {
        let pivot_type = format!(
            "struct<{}>",
            pivots
                .iter()
                .map(|b| {
                    let ty =
                        b.ty.data_type
                            .as_ref()
                            .map(DataType::simple_string)
                            .unwrap_or_else(|| "void".to_owned());
                    format!(
                        "{}:{ty}",
                        pivot_part_name(&b.expr, self.ctx.conf.session_zone)
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        );
        let mode = if self.ctx.conf.ansi {
            CastMode::Ansi
        } else {
            CastMode::Legacy
        };
        for ((v, parts), trimmed) in values.iter().zip(bound).zip(trimmed) {
            if parts.len() != pivots.len() {
                let (value, value_type) = value_render(v, parts, trimmed, self);
                return Err(pivot_type_mismatch(value, value_type, pivot_type, p.origin));
            }
            for (pivot, t) in pivots.iter().zip(trimmed.iter()) {
                let to = pivot.ty.data_type.clone().unwrap_or(DataType::Null);
                if can_cast(t.ty.data_type.as_ref(), &to, mode).is_err() {
                    let (value, value_type) = value_render(v, parts, trimmed, self);
                    return Err(pivot_type_mismatch(value, value_type, pivot_type, p.origin));
                }
            }
        }
        Err(unsupported("multi-column PIVOT", p.origin))
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
            _ => false,
        }
    }

    /// Whether rows of this value can be found in the map.
    const fn matches(&self) -> bool {
        !matches!(self, Self::Nan | Self::Never)
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

/// A whole value and its type for `PIVOT_VALUE_DATA_TYPE_MISMATCH`: one
/// part prints as itself, several as a struct.
fn value_render(
    v: &PivotValue,
    parts: &[Bound],
    trimmed: &[Bound],
    r: &Resolver<'_>,
) -> (String, String) {
    if let ([b], [t]) = (parts, trimmed) {
        let value = v.alias.clone().unwrap_or_else(|| match &b.expr.node {
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.name.clone(),
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
            "{{{}}}",
            trimmed
                .iter()
                .map(|b| part_name(b, r))
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

/// The name of one pivot column part, for the struct pivot type.
fn pivot_part_name(e: &Arc<Expr>, zone: Option<SessionZone>) -> String {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
            c.name.clone()
        }
        _ => pretty(e, zone),
    }
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
