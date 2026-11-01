// SPDX-License-Identifier: Apache-2.0
//! Operators (`+`, `=`, `<`, ...): implicit casts, datetime arithmetic and comparison strength reduction.

use super::*;

/// What [`strength_reduce_comparison`] decided for an integral expression
/// compared with a decimal literal.
pub(super) enum ComparisonReduction {
    /// Not an integral-vs-decimal-literal comparison; bind as usual.
    Keep,
    /// The decimal literal at `index` becomes the `BIGINT` literal `value`.
    Narrow { index: usize, value: i64 },
    /// The literal is outside `BIGINT` range and decides the comparison.
    Decided(bool),
}

/// Spark's strength reduction for an integral expression compared with a
/// decimal literal (`DecimalPrecisionTypeCoercion.integralAndDecimalLiteral`
/// in Spark 4.2, which runs before the implicit casts are inserted). With
/// the literal on the right, `i > d` is `i > floor(d)`, `i >= d` is
/// `i >= ceil(d)`, `i < d` is `i < ceil(d)` and `i <= d` is `i <= floor(d)`;
/// with the literal on the left the bounds swap sides (`d > i` is
/// `ceil(d) > i`, `d >= i` is `floor(d) >= i`, `d < i` is `floor(d) < i` and
/// `d <= i` is `ceil(d) <= i`). A literal below `BIGINT` range or above it
/// decides the comparison outright. The output name shows the rewritten
/// comparison (`select 1 > 0.00001` names its column `(1 > 0)`).
pub(super) fn strength_reduce_comparison(
    op: Builtin,
    left: &Bound,
    right: &Bound,
) -> ComparisonReduction {
    /// The value of a bound decimal literal, if the bound expression is one:
    /// its unscaled value and scale.
    fn decimal(b: &Bound) -> Option<(i128, u32)> {
        // Only a written decimal literal: a session variable binds its value
        // as a literal too but keeps its name (`shown`), and Spark's rule
        // matches `DecimalLiteral`, not `VariableReference`.
        if b.shown.is_some() {
            return None;
        }
        match &b.expr.node {
            ExprNode::Literal(Literal::Decimal {
                unscaled,
                data_type,
            }) => Some((*unscaled, u32::from(data_type.scale()))),
            _ => None,
        }
    }
    /// Whether the bound expression is integral, as Spark's
    /// `IntegralTypeExpression` matches (`BYTE`/`SHORT`/`INT`/`LONG`).
    fn integral(b: &Bound) -> bool {
        matches!(
            b.ty.data_type,
            Some(DataType::Byte | DataType::Short | DataType::Integer | DataType::Long)
        )
    }
    // The decimal literal's side, when the other side is integral. Two
    // decimal literals never match (Spark widens those instead).
    let (index, (unscaled, scale)) = match (decimal(left), decimal(right)) {
        (Some(d), None) if integral(right) => (0, d),
        (None, Some(d)) if integral(left) => (1, d),
        _ => return ComparisonReduction::Keep,
    };
    // `floor` and `ceil` of the decimal value, exactly: the scale is at most
    // 38, so `10^scale` fits an `i128`, and the divisions are exact. Any
    // overflow (unreachable for a decimal literal) keeps the comparison.
    let Some(pow) = 10i128.checked_pow(scale) else {
        return ComparisonReduction::Keep;
    };
    let floor = unscaled.div_euclid(pow);
    let Some(ceil) = unscaled
        .checked_neg()
        .map(|n| n.div_euclid(pow))
        .and_then(i128::checked_neg)
    else {
        return ComparisonReduction::Keep;
    };
    // Spark's `DecimalLiteral.largerThanLargestLong` /
    // `smallerThanSmallestLong`: exact, as both bounds are integers. A
    // literal above `BIGINT` range exceeds every integral value, so only
    // the comparisons that the other side cannot reach fail.
    let long_max = i128::from(i64::MAX);
    let long_min = i128::from(i64::MIN);
    if ceil > long_max {
        return ComparisonReduction::Decided(matches!(
            (op, index),
            (Builtin::Gt | Builtin::GtEq, 0) | (Builtin::Lt | Builtin::LtEq, 1)
        ));
    }
    // A literal below `BIGINT` range is exceeded by every integral value.
    if floor < long_min {
        return ComparisonReduction::Decided(matches!(
            (op, index),
            (Builtin::Gt | Builtin::GtEq, 1) | (Builtin::Lt | Builtin::LtEq, 0)
        ));
    }
    // The narrowed bound fits a `BIGINT`: `floor`/`ceil` lie between two
    // consecutive integers around a value within `BIGINT` range.
    let narrowed = match (op, index) {
        (Builtin::Gt | Builtin::LtEq, 1) | (Builtin::GtEq | Builtin::Lt, 0) => floor,
        _ => ceil,
    };
    let Ok(value) = i64::try_from(narrowed) else {
        return ComparisonReduction::Keep;
    };
    ComparisonReduction::Narrow { index, value }
}

/// The function Spark rewrites `+`/`-` to when an operand is a date and the
/// other no interval (`date_add`, `date_sub`, whose input check then refuses
/// a type it does not take), and whether the operands swap. A string never
/// swaps: `'1' + DATE'...'` is `date_add('1', DATE'...')`.
pub(super) fn date_with_days(
    op: &str,
    left: &ArgType,
    right: &ArgType,
) -> Option<(&'static str, bool)> {
    let date = |t: &ArgType| matches!(t.data_type, Some(DataType::Date));
    let typed = |t: &ArgType| !matches!(t.data_type, None | Some(DataType::Null));
    let string = |t: &ArgType| matches!(t.data_type, Some(DataType::String));
    match op {
        "+" if date(left) && typed(right) => Some(("date_add", false)),
        "+" if date(right) && string(left) => Some(("date_add", false)),
        "+" if date(right) && typed(left) => Some(("date_add", true)),
        "-" if date(left) && typed(right) && !string(right) && !date(right) => {
            Some(("date_sub", false))
        }
        _ => None,
    }
}

impl Resolver<'_> {
    /// A call of operator `op` on `kids`, typed as Spark types it.
    /// `BETWEEN` is Spark's replacement, `>=` and `<=` joined by `AND`,
    /// shown as written.
    pub(in crate::analyzer) fn operator_call(
        &mut self,
        op: Builtin,
        origin: Origin,
        kids: &[Bound],
    ) -> Result<Bound, AnalysisError> {
        if let (Builtin::Between, [v, lo, hi]) = (op, kids) {
            // The value is shared, as Spark's errors show it.
            let shared = Expr::function("commonexpressionref", Vec::new());
            let v = Bound {
                shown: Some(Arc::new(shared)),
                ..v.clone()
            };
            let ge = self.operator_call(Builtin::GtEq, origin, &[v.clone(), lo.clone()])?;
            let le = self.operator_call(Builtin::LtEq, origin, &[v, hi.clone()])?;
            let both = self.operator_call(Builtin::And, origin, &[ge, le])?;
            let shown = ResolvedFunction {
                id: FunctionId::Builtin(op),
                kind: FunctionKind::Scalar,
                args: kids.iter().map(Bound::shown).collect(),
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Boolean,
                nullable: both.ty.nullable,
                deterministic: true,
            };
            let shown = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(shown)));
            return Ok(Bound {
                shown: Some(Arc::new(shown.with_origin(origin))),
                ..both
            });
        }
        if matches!(
            op,
            Builtin::Gt | Builtin::GtEq | Builtin::Lt | Builtin::LtEq
        ) && let [a, b] = kids
        {
            match strength_reduce_comparison(op, a, b) {
                ComparisonReduction::Keep => {}
                ComparisonReduction::Decided(v) => {
                    let mut lateral: Vec<AttrId> = Vec::new();
                    for id in kids.iter().flat_map(|k| k.lateral.iter()) {
                        if !lateral.contains(id) {
                            lateral.push(*id);
                        }
                    }
                    return Ok(Bound {
                        shown: None,
                        expr: Arc::new(
                            Expr::new(ExprNode::Literal(Literal::Boolean(v))).with_origin(origin),
                        ),
                        ty: arg_type(DataType::Boolean, false),
                        agg: kids.iter().find_map(|k| k.agg),
                        win: kids.iter().find_map(|k| k.win),
                        outer: kids.iter().filter_map(|k| k.outer).min(),
                        lateral,
                    });
                }
                ComparisonReduction::Narrow { index, value } => {
                    let mut rewritten = kids.to_vec();
                    let literal_origin = rewritten[index].expr.origin;
                    rewritten[index] = Bound {
                        shown: None,
                        expr: Arc::new(
                            Expr::new(ExprNode::Literal(Literal::Long(value)))
                                .with_origin(literal_origin),
                        ),
                        ty: arg_type(DataType::Long, false),
                        agg: None,
                        win: None,
                        outer: None,
                        lateral: Vec::new(),
                    };
                    return self.operator_call(op, origin, &rewritten);
                }
            }
        }
        let args: Vec<Arg<'_>> = kids.iter().map(arg_of).collect();
        let conf = self.ctx.conf;
        let rules = coerce::Rules {
            ansi: conf.ansi,
            case_sensitive: conf.case_sensitive,
        };
        let typed = coerce::operator(op, &args, rules, conf.decimal_precision_loss);
        let call = |args: Vec<Arc<Expr>>, data_type: DataType, nullable: bool| {
            let f = ResolvedFunction {
                id: FunctionId::Builtin(op),
                kind: FunctionKind::Scalar,
                args,
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type,
                nullable,
                deterministic: true,
            };
            Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))).with_origin(origin))
        };
        let typed = typed.map_err(|m| {
            let shown = call(
                kids.iter().map(Bound::shown).collect(),
                DataType::Boolean,
                true,
            );
            self.mismatch(m, &shown, kids, op.name(), origin)
        })?;
        let cast: Vec<Bound> = kids
            .iter()
            .zip(typed.casts)
            .map(|(k, to)| self.implicit_cast(k, to))
            .collect();
        let expr = call(
            cast.iter().map(|k| Arc::clone(&k.expr)).collect(),
            typed.data_type.clone(),
            typed.nullable,
        );
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        Ok(Bound {
            shown: shown_call(&expr, kids, &cast),
            expr,
            ty: arg_type(typed.data_type, typed.nullable),
            agg: kids.iter().find_map(|k| k.agg),
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
        })
    }

    pub(in crate::analyzer) fn implicit_cast(&self, b: &Bound, to: Option<DataType>) -> Bound {
        self.implicit_cast_as(b, to, false)
    }

    /// [`Self::implicit_cast`], as a `TRY_CAST` when `try_cast`.
    pub(in crate::analyzer) fn implicit_cast_as(
        &self,
        b: &Bound,
        to: Option<DataType>,
        try_cast: bool,
    ) -> Bound {
        let Some(to) = to else {
            return b.clone();
        };
        let mode = match (try_cast, self.ctx.conf.ansi) {
            (true, _) => CastMode::Try,
            (false, true) => CastMode::Ansi,
            (false, false) => CastMode::Legacy,
        };
        let may_null = |f: &DataType| {
            if try_cast {
                coerce::try_nullable(f, &to)
            } else {
                coerce::force_nullable(f, &to)
            }
        };
        let cast = ExprNode::Cast {
            expr: Arc::clone(&b.expr),
            to: to.clone(),
            mode,
        };
        Bound {
            expr: Arc::new(Expr::new(cast).with_origin(b.expr.origin)),
            ty: ArgType {
                nullable: b.ty.nullable || b.ty.data_type.as_ref().is_some_and(may_null),
                data_type: Some(to.clone()),
            },
            shown: Some(b.shown()),
            ..b.clone()
        }
    }
}
