// SPDX-License-Identifier: Apache-2.0
//! The match of an as-of join, bound as Spark 4.3's `ResolveAsOfJoin` binds it: each operand over
//! both inputs, checked in Spark's order (an operand reading both sides, an invalid expression, the
//! operand types), put left side first (`normalizeMatchOperands`), then coerced to one type.

use krishiv_common::error::{
    AsofJoinMatchConditionInvalidExpression, AsofJoinMatchConditionInvalidType,
    AsofJoinMatchConditionTableReference, InternalError, KrishivUnsupportedFeature,
};

use std::collections::HashMap;

use super::*;
use crate::function::{Builtin, FunctionId};

impl Resolver<'_> {
    /// The match `m` of an as-of join of `l` and `r` at `origin`, resolved: `left` reads only
    /// `l`, `right` only `r`, both of one orderable type. `on` is the join's `ON` condition as
    /// written, which decides how a table-reference error names the operands.
    pub(in crate::analyzer) fn as_of_match(
        &mut self,
        origin: Origin,
        m: &AsOfMatch,
        l: &Rel,
        r: &Rel,
        on: Option<&Arc<Expr>>,
        env: &Arc<Env>,
    ) -> Result<AsOfMatch, AnalysisError> {
        let scope: Vec<Attr> = l.out.iter().chain(r.out.iter()).cloned().collect();
        let rels = [l, r];
        let zone = self.ctx.conf.session_zone;
        let operand = |this: &mut Self, e: &Arc<Expr>| {
            this.bind(e, &Self::cx(&scope, &rels, env, "Join"))
                .map_err(|err| invalid_expression_of(err, origin))
        };
        let first = operand(self, &m.left)?;
        let second = operand(self, &m.right)?;
        let shown = |b: &Bound| format!("\"{}\"", pretty(&b.shown(), zone));
        // A side's columns include what an earlier `USING` or `NATURAL` join merged away (its
        // hidden output, reachable by qualified name): Spark's `AddMetadataColumns` adds those to
        // the side's output before `validateMatchConditionTableReferences` reads `outputSet`.
        let ids = |rel: &Rel| -> HashSet<AttrId> {
            rel.out
                .iter()
                .chain(rel.hidden.iter())
                .map(|a| a.col.id)
                .collect()
        };
        let (left_ids, right_ids) = (ids(l), ids(r));
        let (refs1, refs2) = (column_ids(&first.expr), column_ids(&second.expr));
        let both = |refs: &HashSet<AttrId>| {
            refs.iter().any(|i| left_ids.contains(i)) && refs.iter().any(|i| right_ids.contains(i))
        };
        if both(&refs1) || both(&refs2) {
            // Spark runs this check in the first pass that resolves the `ON` condition, before
            // the operands resolve, so it names them as they stand then (`as_checked_in`).
            let pass = match on {
                Some(c) => self
                    .bind(c, &Self::cx(&scope, &rels, env, "Join"))
                    .map_or(1, |b| resolved_pass(&b.shown())),
                None => 1,
            };
            let checked =
                |b: &Bound| format!("\"{}\"", pretty(&as_checked_in(&b.shown(), pass), zone));
            return Err(AnalysisError::new(
                AsofJoinMatchConditionTableReference {
                    refs1: checked(&first),
                    refs2: checked(&second),
                },
                origin,
            ));
        }
        for b in [&first, &second] {
            if let Some(expr) = invalid_expression(b, zone) {
                return Err(AnalysisError::new(
                    AsofJoinMatchConditionInvalidExpression { expr },
                    origin,
                ));
            }
        }
        let (type1, type2) = (
            quoted_type(first.ty.data_type.as_ref()),
            quoted_type(second.ty.data_type.as_ref()),
        );
        let type_error = || {
            AnalysisError::new(
                AsofJoinMatchConditionInvalidType {
                    type1: type1.clone(),
                    type2: type2.clone(),
                },
                origin,
            )
        };
        // Spark's `areOperandsCompatible` alone decides this error; an untyped `NULL` is
        // `NullType`, which meets any type.
        let case_sensitive = self.ctx.conf.case_sensitive;
        let (t1, t2) = (first.ty.data_type.as_ref(), second.ty.data_type.as_ref());
        if !operands_compatible(t1, t2, case_sensitive) {
            return Err(type_error());
        }
        // Spark's `normalizeMatchOperands`: the operand that reads the left input goes first (a
        // constant takes its written side). Connect's operands are per side already, but Spark
        // takes `joinAsOf(right, right.ts, left.ts)` too (its condition is an ordinary predicate),
        // which is the swapped match in the flipped direction.
        let side = |refs: &HashSet<AttrId>, written_left: bool| {
            if refs.is_empty() {
                Some(written_left)
            } else if refs.iter().all(|i| left_ids.contains(i)) {
                Some(true)
            } else if refs.iter().all(|i| right_ids.contains(i)) {
                Some(false)
            } else {
                None
            }
        };
        let (written_first, written_second) = (Arc::clone(&first.expr), Arc::clone(&second.expr));
        let (left, right, direction) = match (side(&refs1, true), side(&refs2, false)) {
            (Some(true), Some(false)) => (first, second, m.direction),
            (Some(false), Some(true)) => (second, first, m.direction.flip()),
            _ if m.as_written => {
                // Any other placement has no query context in Spark.
                return Err(AnalysisError::new(
                    AsofJoinMatchConditionTableReference {
                        refs1: shown(&first),
                        refs2: shown(&second),
                    },
                    Origin::UNKNOWN,
                ));
            }
            _ => {
                return Err(AnalysisError::new(
                    KrishivUnsupportedFeature {
                        feature: format!(
                            "an as-of join whose values {} and {} read the same input",
                            shown(&first),
                            shown(&second)
                        ),
                    },
                    origin,
                ));
            }
        };
        // Spark's distance (`buildLeafOrderExpression`, `makeOrderingExpr`) over the operands as
        // normalized, where the left one's type subtracts (`supportsSubtract`): `R - L` forward,
        // `L - R` backward. Spark builds it in every form, so `CheckAnalysis` checks it in every
        // form, but evaluates it only forward and on the Connect form (`RewriteAsOfJoin`); the
        // executor evaluates it where Spark does. Tuple and array values are not subtracted yet
        // (Spark subtracts their leaves), see the design note.
        let every_pair = m.distance_on_every_pair;
        let forward = matches!(direction, crate::AsOfDirection::Forward);
        let operands = if forward {
            (&right.expr, &left.expr)
        } else {
            (&left.expr, &right.expr)
        };
        // Spark's `CheckAnalysis` then checks the comparison as written: a pair the as-of rule
        // accepts and the operator does not (an interval against a STRING) is the operator's
        // own error, `"(dti <= s)"`.
        let written = match (m.direction, m.allow_exact) {
            // Nearest is Connect only, which writes no operator.
            (crate::AsOfDirection::Backward, true) | (crate::AsOfDirection::Nearest, _) => ">=",
            (crate::AsOfDirection::Backward, false) => ">",
            (crate::AsOfDirection::Forward, true) => "<=",
            (crate::AsOfDirection::Forward, false) => "<",
        };
        self.call(
            written,
            vec![Arc::clone(&written_first), Arc::clone(&written_second)],
            env,
            origin,
        )?;
        // The two values meet at the type a comparison of them casts both to.
        let compared = self.call(
            ">=",
            vec![Arc::clone(&left.expr), Arc::clone(&right.expr)],
            env,
            origin,
        )?;
        // Spark's `CheckAnalysis` reaches the distance after the comparison, so a comparison
        // type error wins; a subtraction that does not resolve is reported as Spark reports it.
        let distance = if subtracts(left.ty.data_type.as_ref()) {
            let (a, b) = operands;
            // Resolved whether or not it is kept, so its error is reported either way.
            let subtraction = self.call("-", vec![Arc::clone(a), Arc::clone(b)], env, origin)?;
            let kept =
                forward || (every_pair && matches!(direction, crate::AsOfDirection::Backward));
            kept.then_some(subtraction)
        } else {
            None
        };
        let ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction { args, .. })) =
            &compared.node
        else {
            return Err(AnalysisError::new(
                InternalError {
                    message: "an as-of match's comparison did not resolve to a call".to_owned(),
                },
                origin,
            ));
        };
        let [left, right] = args.as_slice() else {
            return Err(AnalysisError::new(
                InternalError {
                    message: "an as-of match's comparison does not take two values".to_owned(),
                },
                origin,
            ));
        };
        let (left, right) = (Arc::clone(left), Arc::clone(right));
        let tolerance = match &m.tolerance {
            None => None,
            Some(t) => Some(self.bind(t, &Self::cx(&[], &[], env, "Join"))?.expr),
        };
        Ok(AsOfMatch {
            left,
            right,
            direction,
            allow_exact: m.allow_exact,
            tolerance,
            distance,
            distance_on_every_pair: every_pair,
            as_written: false,
        })
    }
}

/// An operand as Spark's table-reference check names it. `ResolveAsOfJoin` checks the operands
/// in the first analyzer pass that resolves both inputs and the `ON` condition
/// (`condition.forall(_.resolved)`), pass `pass` (`resolved_pass` of the condition, 1 without
/// one). Each node resolves in some pass: a leaf in pass 1; any other node is built (by the one
/// bottom-up `ResolveBinaryArithmetic` of a pass) in the pass its children are all resolved, and
/// resolves in that pass, or one later for the datetime nodes that need a time zone
/// (`TimeAdd`, `TimestampAddYMInterval`, `DateAddInterval`, `DatetimeSub`, `SubtractTimestamps`,
/// a comparison that casts a side to TIMESTAMP): `ResolveTimeZone` runs before the rewrite. A node
/// built by `pass` shows as rewritten, `trade_time + (quote_time - trade_time)`; a later one as
/// the operator written, `(trade_time + (quote_time - trade_time))`. Explicit casts already have
/// their zone (Spark 4.3.0 runs, `opus1-asof-ref`, `rev-f2-asof-ref`, `opus1-f5`).
fn as_checked_in(e: &Arc<Expr>, pass: u32) -> Arc<Expr> {
    spark_passes(e, Some(pass)).0
}

/// The analyzer pass in which Spark resolves the shown expression `e` (see `as_checked_in`).
fn resolved_pass(e: &Arc<Expr>) -> u32 {
    spark_passes(e, None).1
}

/// `e` with each node built after `pass` shown as the operator written (unchanged without a
/// pass), and the pass in which `e` resolves.
fn spark_passes(e: &Arc<Expr>, pass: Option<u32>) -> (Arc<Expr>, u32) {
    // The pass each output resolves in, by address; `kept` holds each alive so no address is
    // reused while the map is read.
    let mut resolved: HashMap<*const Expr, u32> = HashMap::new();
    let mut kept: Vec<Arc<Expr>> = Vec::new();
    let out = Arc::clone(e).transform_up(&mut |x: Arc<Expr>| {
        let mut built = 1;
        x.for_each_child(&mut |c| {
            built = built.max(resolved.get(&Arc::as_ptr(c)).copied().unwrap_or(1));
        });
        let call = match &x.node {
            ExprNode::Resolved(ResolvedExpr::Function(f)) => match f.id {
                FunctionId::Builtin(id) => Some((id, f)),
                _ => None,
            },
            _ => None,
        };
        let (zoned, plain) = call.map_or((false, None), |(id, f)| {
            (zoned_rewrite(id, f), plain_operator(id))
        });
        let x = match (call, plain) {
            // Not rewritten: Spark's `Add`/`Subtract` over the operands, `(l + r)`.
            (Some((_, f)), Some(op)) if pass.is_some_and(|p| built > p) => {
                let mut g = f.clone();
                g.id = FunctionId::Builtin(op);
                Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Function(g))).with_origin(x.origin),
                )
            }
            _ => x,
        };
        resolved.insert(Arc::as_ptr(&x), built + u32::from(zoned));
        kept.push(Arc::clone(&x));
        Ok::<_, std::convert::Infallible>(x)
    });
    let out = match out {
        Ok(x) => x,
        Err(never) => match never {},
    };
    let pass = resolved.get(&Arc::as_ptr(&out)).copied().unwrap_or(1);
    (out, pass)
}

/// Whether Spark's node for the shown call `f` needs a time zone, so it resolves one pass after
/// it is built.
fn zoned_rewrite(id: Builtin, f: &ResolvedFunction) -> bool {
    let arg = |i: usize| f.args.get(i).and_then(|a| shown_type(a));
    match id {
        Builtin::TimestampAddInterval
        | Builtin::TimestampAddYmInterval
        | Builtin::DateAddInterval => true,
        // `date - <year-month interval>` is Spark's `DateAddYMInterval(l, -r)`, which has no
        // zone: it resolves at once and an operator over it is rewritten. The interval is
        // often a literal, whose type only `interval_type` knows.
        Builtin::DatetimeSub => !matches!(
            (arg(0), arg(1)),
            (
                Some(DataType::Date),
                Some(DataType::YearMonthInterval { .. })
            )
        ),
        // `SubtractTimestamps` shows as `-` already; a timestamp on either side makes one.
        Builtin::Subtract => (0..f.args.len())
            .any(|i| matches!(arg(i), Some(DataType::Timestamp | DataType::TimestampNtz))),
        // A comparison whose coercion casts a side to TIMESTAMP (`Cast.needsTimeZone`): the
        // implicit cast is added after `ResolveTimeZone` of its pass.
        Builtin::Eq
        | Builtin::EqNullSafe
        | Builtin::Lt
        | Builtin::LtEq
        | Builtin::Gt
        | Builtin::GtEq => {
            let zoned_with_timestamp = |t: Option<DataType>| {
                matches!(
                    t,
                    Some(DataType::String | DataType::Date | DataType::TimestampNtz)
                )
            };
            match (arg(0), arg(1)) {
                (Some(DataType::Timestamp), other) | (other, Some(DataType::Timestamp)) => {
                    zoned_with_timestamp(other)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// The operator a datetime call is written as.
fn plain_operator(id: Builtin) -> Option<Builtin> {
    match id {
        Builtin::TimestampAddInterval
        | Builtin::TimestampAddYmInterval
        | Builtin::DateAddInterval
        | Builtin::DateAddYmInterval => Some(Builtin::Add),
        Builtin::DatetimeSub => Some(Builtin::Subtract),
        _ => None,
    }
}

/// The type of a shown expression, literals of datetime and interval types included.
fn shown_type(e: &Expr) -> Option<DataType> {
    match &e.node {
        ExprNode::Literal(Literal::Date(_)) => Some(DataType::Date),
        ExprNode::Literal(Literal::Timestamp(_)) => Some(DataType::Timestamp),
        ExprNode::Literal(Literal::TimestampNtz(_)) => Some(DataType::TimestampNtz),
        _ => e.interval_type().or_else(|| e.resolved_type()),
    }
}

/// Whether Spark subtracts values of `t` for an as-of distance (`AsOfJoin.supportsSubtract`).
fn subtracts(t: Option<&DataType>) -> bool {
    matches!(
        t,
        Some(
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Float
                | DataType::Double
                | DataType::Decimal(_)
                | DataType::YearMonthInterval { .. }
                | DataType::DayTimeInterval { .. }
                | DataType::CalendarInterval
                | DataType::Timestamp
                | DataType::TimestampNtz
                | DataType::Date
        )
    )
}

/// The ids of the input columns `e` reads (outside subqueries).
fn column_ids(e: &Arc<Expr>) -> HashSet<AttrId> {
    let mut ids = HashSet::new();
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        if let crate::walk::Node::Expr(x) = n
            && let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
        {
            ids.insert(c.id);
        }
        std::ops::ControlFlow::<()>::Continue(())
    });
    ids
}

/// The error of an operand Spark's `findInvalidMatchConditionExpression` refuses, as Spark shows
/// it: a subquery, an aggregate, a window, a generator or a nondeterministic call.
fn invalid_expression(b: &Bound, zone: Option<super::super::SessionZone>) -> Option<String> {
    if b.agg.is_some()
        && let Some(call) = super::super::aggregate_text::aggregate_calls(&b.shown())
            .into_iter()
            .next()
    {
        return Some(format!("\"{call}\""));
    }
    let mut found = None;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(&b.expr), |n| {
        let crate::walk::Node::Expr(x) = n else {
            return std::ops::ControlFlow::Continue(());
        };
        let text = match &x.node {
            ExprNode::Subquery {
                kind: crate::SubqueryKind::Scalar,
                ..
            } => Some("scalarsubquery()".to_owned()),
            ExprNode::Subquery { .. }
            | ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    kind: FunctionKind::Generator,
                    ..
                }
                | ResolvedFunction {
                    deterministic: false,
                    ..
                },
            )) => Some(pretty(x, zone)),
            _ => None,
        };
        match text {
            Some(t) => {
                found = Some(format!("\"{t}\""));
                std::ops::ControlFlow::Break(())
            }
            None => std::ops::ControlFlow::Continue(()),
        }
    });
    found
}

/// A bind error of an operand that Spark reports as an invalid match expression instead: the
/// operator-level refusals of windows, aggregates and generators.
fn invalid_expression_of(err: AnalysisError, origin: Origin) -> AnalysisError {
    let class = err.error.error_class().name();
    if class == "UNSUPPORTED_EXPR_FOR_OPERATOR" || class.starts_with("UNSUPPORTED_GENERATOR") {
        let expr = err
            .error
            .message_parameters()
            .iter()
            .find(|(k, _)| *k == "invalidExprSqls" || *k == "invalid_expr_sqls")
            .map(|(_, v)| v.clone());
        if let Some(expr) = expr {
            return AnalysisError::new(AsofJoinMatchConditionInvalidExpression { expr }, origin);
        }
    }
    err
}

/// Spark's `MatchConditionTypes.areOperandsCompatible` (`None` is an untyped `NULL`, Spark's
/// `NullType`): both orderable without an empty struct, never a string against a datetime, arrays
/// of one orderable element type or positionally compatible structs, otherwise a wider type of
/// the two under the legacy rules (`TypeCoercion.findWiderTypeForTwo`, whatever the ANSI mode).
fn operands_compatible(a: Option<&DataType>, b: Option<&DataType>, case_sensitive: bool) -> bool {
    if !a.is_none_or(valid_operand) || !b.is_none_or(valid_operand) {
        return false;
    }
    let datetime = |t: Option<&DataType>| {
        matches!(
            t,
            Some(DataType::Date | DataType::Timestamp | DataType::TimestampNtz)
        )
    };
    let string =
        |t: Option<&DataType>| matches!(t, Some(DataType::String | DataType::CollatedString(_)));
    if (datetime(a) && string(b)) || (string(a) && datetime(b)) {
        return false;
    }
    match (a, b) {
        (Some(DataType::Array(x)), Some(DataType::Array(y))) => {
            let (x, y) = (x.element_type(), y.element_type());
            if x.equals_structurally(y, true) {
                super::super::coerce::orderable(x)
            } else {
                positional_structs(x, y)
            }
        }
        (Some(x), Some(y)) if positional_structs(x, y) => true,
        _ => {
            let (x, y) = (
                super::super::coerce::Arg {
                    ty: a,
                    nullable: true,
                    literal: None,
                },
                super::super::coerce::Arg {
                    ty: b,
                    nullable: true,
                    literal: None,
                },
            );
            let rules = super::super::coerce::Rules::new(false, case_sensitive);
            super::super::coerce::wider_type(&x, &y, rules).is_some()
        }
    }
}

/// Structs of one field count, each field pair compatible (names may differ).
fn positional_structs(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::Struct(x), DataType::Struct(y)) => {
            !x.fields().is_empty()
                && x.fields().len() == y.fields().len()
                && x.fields().iter().zip(y.fields()).all(|(f, g)| {
                    operands_compatible(Some(f.data_type()), Some(g.data_type()), true)
                })
        }
        _ => false,
    }
}

/// Orderable and holding no empty struct.
fn valid_operand(t: &DataType) -> bool {
    fn empty_struct(t: &DataType) -> bool {
        match t {
            DataType::Struct(s) => {
                s.fields().is_empty() || s.fields().iter().any(|f| empty_struct(f.data_type()))
            }
            DataType::Array(a) => empty_struct(a.element_type()),
            _ => false,
        }
    }
    super::super::coerce::orderable(t) && !empty_struct(t)
}
