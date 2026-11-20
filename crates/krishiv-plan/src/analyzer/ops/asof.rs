// SPDX-License-Identifier: Apache-2.0
//! The match of an as-of join, bound as Spark 4.3's `ResolveAsOfJoin` binds it: each operand over
//! both inputs, checked in Spark's order (an operand reading both sides, an invalid expression, the
//! operand types), put left side first (`normalizeMatchOperands`), then coerced to one type.

use krishiv_common::error::{
    AsofJoinMatchConditionInvalidExpression, AsofJoinMatchConditionInvalidType,
    AsofJoinMatchConditionTableReference,
};

use super::*;

impl Resolver<'_> {
    /// The match `m` of an as-of join of `l` and `r` at `origin`, resolved: `left` reads only
    /// `l`, `right` only `r`, both of one orderable type.
    pub(in crate::analyzer) fn as_of_match(
        &mut self,
        origin: Origin,
        m: &AsOfMatch,
        l: &Rel,
        r: &Rel,
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
        let left_ids: HashSet<AttrId> = l.out.iter().map(|a| a.col.id).collect();
        let right_ids: HashSet<AttrId> = r.out.iter().map(|a| a.col.id).collect();
        let (refs1, refs2) = (column_ids(&first.expr), column_ids(&second.expr));
        let both = |refs: &HashSet<AttrId>| {
            refs.iter().any(|i| left_ids.contains(i)) && refs.iter().any(|i| right_ids.contains(i))
        };
        if both(&refs1) || both(&refs2) {
            return Err(AnalysisError::new(
                AsofJoinMatchConditionTableReference {
                    refs1: shown(&first),
                    refs2: shown(&second),
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
        let (Some(t1), Some(t2)) = (&first.ty.data_type, &second.ty.data_type) else {
            return Err(type_error());
        };
        if !operands_compatible(t1, t2) {
            return Err(type_error());
        }
        // Spark's `normalizeMatchOperands`: the operand that reads the left input goes first (a
        // constant takes its written side); any other placement has no query context.
        let (left, right, direction) = if m.as_written {
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
            match (side(&refs1, true), side(&refs2, false)) {
                (Some(true), Some(false)) => (first, second, m.direction),
                (Some(false), Some(true)) => (second, first, m.direction.flip()),
                _ => {
                    return Err(AnalysisError::new(
                        AsofJoinMatchConditionTableReference {
                            refs1: shown(&first),
                            refs2: shown(&second),
                        },
                        Origin::UNKNOWN,
                    ));
                }
            }
        } else {
            (first, second, m.direction)
        };
        // The two values meet at the type a comparison of them casts both to.
        let compared = self
            .call(
                ">=",
                vec![Arc::clone(&left.expr), Arc::clone(&right.expr)],
                env,
                origin,
            )
            .map_err(|_| type_error())?;
        let (left, right) = match &compared.node {
            ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction { args, .. }))
                if args.len() == 2 =>
            {
                (Arc::clone(&args[0]), Arc::clone(&args[1]))
            }
            _ => (left.expr, right.expr),
        };
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
            as_written: false,
        })
    }
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

/// Spark's `MatchConditionTypes.areOperandsCompatible`, short of the wider type (the comparison's
/// coercion decides that): both orderable without an empty struct, never a string against a
/// datetime, arrays of one orderable element type or positionally compatible structs.
fn operands_compatible(a: &DataType, b: &DataType) -> bool {
    if !valid_operand(a) || !valid_operand(b) {
        return false;
    }
    let datetime = |t: &DataType| {
        matches!(
            t,
            DataType::Date | DataType::Timestamp | DataType::TimestampNtz
        )
    };
    let string = |t: &DataType| matches!(t, DataType::String | DataType::CollatedString(_));
    if (datetime(a) && string(b)) || (string(a) && datetime(b)) {
        return false;
    }
    match (a, b) {
        (DataType::Array(x), DataType::Array(y)) => {
            let (x, y) = (x.element_type(), y.element_type());
            if x.equals_structurally(y, true) {
                super::super::coerce::orderable(x)
            } else {
                positional_structs(x, y)
            }
        }
        (DataType::Struct(_), DataType::Struct(_)) => {
            a.equals_structurally(b, true) || positional_structs(a, b)
        }
        _ => true,
    }
}

/// Structs of one field count, each field pair compatible (names may differ).
fn positional_structs(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::Struct(x), DataType::Struct(y)) => {
            !x.fields().is_empty()
                && x.fields().len() == y.fields().len()
                && x.fields()
                    .iter()
                    .zip(y.fields())
                    .all(|(f, g)| operands_compatible(f.data_type(), g.data_type()))
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
