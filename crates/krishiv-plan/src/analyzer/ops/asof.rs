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
        // Spark's `ResolveAsOfJoin` waits for the `ON` condition to resolve
        // (`condition.forall(_.resolved)`), so a condition that does not resolve (an unknown
        // name or function, a type mismatch) is reported before any as-of check.
        let on = match on {
            Some(c) => Some(self.bind(c, &Self::cx(&scope, &rels, env, "Join"))?),
            None => None,
        };
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
            let pass = on.as_ref().map_or(1, resolved_pass);
            let checked = |b: &Bound| format!("\"{}\"", pretty(&as_checked_in(b, pass), zone));
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
        // Spark's `CheckAnalysis` then checks the as-of condition `buildMatchExpressions` built
        // (`makeAsOfCond` on Connect): the operator of the normalized direction over the
        // normalized operands, structs rebuilt field by field (`alignOperandsForComparison`). A
        // pair the as-of rule accepts and the operator does not (an interval against a STRING) is
        // the operator's own error, `"(dti <= s)"` for `s >= dti` written right side first.
        let condition = match (direction, m.allow_exact) {
            (crate::AsOfDirection::Backward, true) => Some(">="),
            (crate::AsOfDirection::Backward, false) => Some(">"),
            (crate::AsOfDirection::Forward, true) => Some("<="),
            (crate::AsOfDirection::Forward, false) => Some("<"),
            // Connect's nearest match: `true` with exact matches, else `NOT (l = r)`.
            (crate::AsOfDirection::Nearest, true) => None,
            (crate::AsOfDirection::Nearest, false) => Some("="),
        };
        if let Some(op) = condition {
            let (l, r) = self.aligned(&left.expr, &right.expr, env, origin)?;
            self.call(op, vec![l, r], env, origin)?;
        }
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
        // An untyped `NULL` value stays `NullType` in the comparison and matches no row: it
        // takes the other value's type, so both sides of the merge have one type. Against an
        // array or a struct, which the merge does not sort yet (design unit 1), both values are
        // an `INT` NULL instead: the rows the join returns are the same, none matched.
        let (left, right) = match (null_type(left), null_type(right)) {
            (true, false) | (false, true) => {
                let (null, other) = if null_type(left) {
                    (left, right)
                } else {
                    (right, left)
                };
                match other.resolved_type() {
                    Some(t) if !t.is_nested() => {
                        let cast = cast_null(null, t);
                        if null_type(left) {
                            (cast, Arc::clone(right))
                        } else {
                            (Arc::clone(left), cast)
                        }
                    }
                    _ => (
                        cast_null(left, DataType::Integer),
                        cast_null(right, DataType::Integer),
                    ),
                }
            }
            _ => (Arc::clone(left), Arc::clone(right)),
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
            distance,
            distance_on_every_pair: every_pair,
            as_written: false,
        })
    }

    /// Spark's `alignOperandsForComparison`: two structs of one field count are compared as
    /// structs rebuilt from their fields (`CreateStruct`, field pairs aligned the same way), so
    /// the comparison and its error name them `named_struct(d, d)` whatever names were written.
    fn aligned(
        &mut self,
        l: &Arc<Expr>,
        r: &Arc<Expr>,
        env: &Arc<Env>,
        origin: Origin,
    ) -> Result<(Arc<Expr>, Arc<Expr>), AnalysisError> {
        let (Some(DataType::Struct(a)), Some(DataType::Struct(b))) =
            (l.resolved_type(), r.resolved_type())
        else {
            return Ok((Arc::clone(l), Arc::clone(r)));
        };
        if a.fields().is_empty() || a.fields().len() != b.fields().len() {
            return Ok((Arc::clone(l), Arc::clone(r)));
        }
        let (left_fields, right_fields) = (struct_fields(l, &a), struct_fields(r, &b));
        let (mut ls, mut rs) = (Vec::new(), Vec::new());
        for (i, (x, y)) in left_fields.iter().zip(&right_fields).enumerate() {
            let (x, y) = self.aligned(x, y, env, origin)?;
            ls.extend([field_name(&x, i), x]);
            rs.extend([field_name(&y, i), y]);
        }
        Ok((
            self.call("named_struct", ls, env, origin)?,
            self.call("named_struct", rs, env, origin)?,
        ))
    }
}

/// The field values of the struct `e` of type `t` (Spark's `structFieldExprs`): a
/// `named_struct`'s values, else each field read from `e` (`GetStructField`).
fn struct_fields(e: &Arc<Expr>, t: &krishiv_types::StructType) -> Vec<Arc<Expr>> {
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && matches!(f.id, FunctionId::Registry { .. })
        && f.id.name() == "named_struct"
        && f.args.len() == 2 * t.fields().len()
    {
        return f.args.iter().skip(1).step_by(2).cloned().collect();
    }
    let base_nullable = nullable(e);
    t.fields()
        .iter()
        .enumerate()
        .map(|(ordinal, field)| {
            Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                    base: Arc::clone(e),
                    ordinal,
                    name: field.name().to_owned(),
                    data_type: field.data_type().clone(),
                    nullable: base_nullable || field.nullable(),
                }))
                .with_origin(e.origin),
            )
        })
        .collect()
}

/// The name `CreateStruct` gives the value `e` at `i`: a column's, an alias's or a field's
/// name, else `col<i + 1>`.
fn field_name(e: &Arc<Expr>, i: usize) -> Arc<Expr> {
    let name = match &e.node {
        ExprNode::Resolved(
            ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c) | ResolvedExpr::LambdaVar(c),
        ) => c.name.clone(),
        ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.name.clone(),
        ExprNode::Resolved(ResolvedExpr::Field { name, .. }) => name.clone(),
        _ => format!("col{}", i + 1),
    };
    Arc::new(Expr::new(ExprNode::Literal(Literal::String(name))).with_origin(e.origin))
}

/// Whether the resolved value `e` may be null (true when it cannot tell).
fn nullable(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Resolved(
            ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c) | ResolvedExpr::LambdaVar(c),
        ) => c.nullable,
        ExprNode::Resolved(ResolvedExpr::Function(f)) => f.nullable,
        ExprNode::Resolved(ResolvedExpr::Field { nullable, .. }) => *nullable,
        ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.nullable,
        ExprNode::Literal(l) => matches!(l, Literal::Null(_)),
        _ => true,
    }
}

/// Whether the resolved value `e` is an untyped `NULL` (Spark's `NullType`).
fn null_type(e: &Expr) -> bool {
    matches!(e.resolved_type(), None | Some(DataType::Null))
}

/// A `NULL` of type `to`, at `at`'s origin (a `NULL` of any type is the same value).
fn cast_null(at: &Arc<Expr>, to: DataType) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Literal(Literal::Null(Some(to)))).with_origin(at.origin))
}

/// An operand as Spark's table-reference check names it. `ResolveAsOfJoin` checks the operands
/// in the first analyzer pass that resolves both inputs and the `ON` condition
/// (`condition.forall(_.resolved)`), pass `pass` (`resolved_pass` of the condition, 1 without
/// one). Each node resolves in some pass: a leaf in pass 1; any other node is built (by the one
/// bottom-up `ResolveBinaryArithmetic` of a pass) in the pass its children are all resolved, and
/// resolves in that pass, or one later when it needs a time zone: the datetime nodes
/// (`TimeAdd`, `TimestampAddYMInterval`, `DateAddInterval`, `DatetimeSub`, `SubtractTimestamps`)
/// and any node whose coercion adds an implicit cast that needs one (`Cast.needsTimeZone`: a
/// comparison, `IN`, `coalesce`, `hour(string)`, ...), since `ResolveTimeZone` runs before the
/// rewrite and the coercion rules. A node built by `pass` shows as rewritten,
/// `trade_time + (quote_time - trade_time)`; a later one as the operator written,
/// `(trade_time + (quote_time - trade_time))`. Explicit casts already have their zone (Spark
/// 4.3.0 runs, `opus1-asof-ref`, `rev-f2-asof-ref`, `opus1-f5`, `rev-f1-spark/delta8c9`).
fn as_checked_in(b: &Bound, pass: u32) -> Arc<Expr> {
    spark_passes(b, Some(pass)).0
}

/// The analyzer pass in which Spark resolves `b` (see `as_checked_in`).
fn resolved_pass(b: &Bound) -> u32 {
    spark_passes(b, None).1
}

/// `b` as shown, with each node built after `pass` shown as the operator written (unchanged
/// without a pass), and the pass in which `b` resolves.
fn spark_passes(b: &Bound, pass: Option<u32>) -> (Arc<Expr>, u32) {
    let shown = b.shown();
    let zoned_casts = zoned_implicit_casts(&b.expr, &shown);
    // Keyed by the address of a node of `shown`, which stays alive for the walk.
    let mut memo = HashMap::new();
    passes(&shown, pass, &zoned_casts, &mut memo)
}

/// One node of `spark_passes`: `e` rewritten and the pass it resolves in. Recurses once per
/// level, as `Expr::transform_up` does (operands are within `MAX_DEPTH`).
fn passes(
    e: &Arc<Expr>,
    pass: Option<u32>,
    zoned_casts: &HashSet<*const Expr>,
    memo: &mut HashMap<*const Expr, (Arc<Expr>, u32)>,
) -> (Arc<Expr>, u32) {
    if let Some((x, p)) = memo.get(&Arc::as_ptr(e)) {
        return (Arc::clone(x), *p);
    }
    let mut built = 1;
    let rebuilt = Arc::clone(e).map_children(|c| {
        let (x, p) = passes(&c, pass, zoned_casts, memo);
        built = built.max(p);
        Ok::<_, std::convert::Infallible>(x)
    });
    let rebuilt = match rebuilt {
        Ok(x) => x,
        Err(never) => match never {},
    };
    let call = match &rebuilt.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => match f.id {
            FunctionId::Builtin(id) => Some((id, f)),
            _ => None,
        },
        _ => None,
    };
    let zoned = zoned_casts.contains(&Arc::as_ptr(e))
        || call.is_some_and(|(id, f)| zoned_rewrite(id, f))
        || matches!(
            &rebuilt.node,
            ExprNode::Resolved(ResolvedExpr::Function(f)) if casts_to_timestamp(f)
        );
    let x = match call.and_then(|(id, f)| plain_operator(id).map(|op| (op, f))) {
        // Not rewritten: Spark's `Add`/`Subtract` over the operands, `(l + r)`.
        Some((op, f)) if pass.is_some_and(|p| built > p) => {
            let mut g = f.clone();
            g.id = FunctionId::Builtin(op);
            Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(g)))
                    .with_origin(rebuilt.origin),
            )
        }
        _ => Arc::clone(&rebuilt),
    };
    let p = built + u32::from(zoned);
    memo.insert(Arc::as_ptr(e), (Arc::clone(&x), p));
    (x, p)
}

/// The nodes of `shown` whose bound form `bound` casts a child implicitly to or from a type that
/// needs a time zone (`Cast.needsTimeZone`). The shown form is the bound one without the implicit
/// casts, so the two are walked side by side: a bound cast the shown node has not (with its target
/// type) is implicit. Where the two differ in shape (a shown rewrite) the walk stops.
fn zoned_implicit_casts(bound: &Arc<Expr>, shown: &Arc<Expr>) -> HashSet<*const Expr> {
    let mut zoned = HashSet::new();
    let mut seen = HashSet::new();
    let mut stack = vec![(Arc::clone(bound), Arc::clone(shown))];
    while let Some((b, s)) = stack.pop() {
        if !seen.insert((Arc::as_ptr(&b), Arc::as_ptr(&s))) {
            continue;
        }
        let mut bound_kids = Vec::new();
        b.for_each_child(&mut |c| bound_kids.push(Arc::clone(c)));
        let mut shown_kids = Vec::new();
        s.for_each_child(&mut |c| shown_kids.push(Arc::clone(c)));
        if bound_kids.len() != shown_kids.len() {
            // A shown call over another shape (`between(x, lo, hi)` bound as two comparisons):
            // a zoned cast of the bound form the shown form does not write is implicit.
            if zoned_casts(&b) > zoned_casts(&s) {
                zoned.insert(Arc::as_ptr(&s));
            }
            continue;
        }
        for (mut bk, sk) in bound_kids.into_iter().zip(shown_kids) {
            // A bound cast the shown node has not (with its target type) is implicit.
            while let ExprNode::Cast { expr, to, .. } = &bk.node
                && !matches!(&sk.node, ExprNode::Cast { to: t, .. } if t == to)
            {
                if shown_type(expr).is_some_and(|from| needs_time_zone(&from, to)) {
                    zoned.insert(Arc::as_ptr(&s));
                }
                let inner = Arc::clone(expr);
                bk = inner;
            }
            stack.push((bk, sk));
        }
    }
    zoned
}

/// How many casts that need a time zone `e` holds (outside subqueries).
fn zoned_casts(e: &Arc<Expr>) -> usize {
    let mut n = 0;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |node| {
        if let crate::walk::Node::Expr(x) = node
            && let ExprNode::Cast { expr, to, .. } = &x.node
            && shown_type(expr).is_some_and(|from| needs_time_zone(&from, to))
        {
            n += 1;
        }
        std::ops::ControlFlow::<()>::Continue(())
    });
    n
}

/// Spark's `Cast.needsTimeZone(from, to)` over Krishiv's types.
fn needs_time_zone(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    match (from, to) {
        (T::Variant, _)
        | (T::String | T::CollatedString(_) | T::Date | T::TimestampNtz, T::Timestamp)
        | (
            T::Timestamp,
            T::String | T::CollatedString(_) | T::Date | T::TimestampNtz | T::Time { .. },
        )
        | (T::Time { .. }, T::TimestampNtz | T::Timestamp) => true,
        (T::Array(a), T::Array(b)) => needs_time_zone(a.element_type(), b.element_type()),
        (T::Map(a), T::Map(b)) => {
            needs_time_zone(a.key_type(), b.key_type())
                || needs_time_zone(a.value_type(), b.value_type())
        }
        (T::Struct(a), T::Struct(b)) => {
            a.fields().len() == b.fields().len()
                && a.fields()
                    .iter()
                    .zip(b.fields())
                    .any(|(f, g)| needs_time_zone(f.data_type(), g.data_type()))
        }
        _ => false,
    }
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
        _ => false,
    }
}

/// Whether Spark casts an argument of `f` to TIMESTAMP that Krishiv's kernel reads as written,
/// so the bound form has no cast to find: Spark's `inputTypes` take TIMESTAMP there
/// (`DateFormatClass`, `TruncTimestamp`; `GetTimeField`'s `AnyTimestampType` defaults a DATE or
/// STRING to TIMESTAMP), and that implicit cast needs a time zone (Spark 4.3.0 source,
/// `datetimeExpressions.scala`; `date_format(t.d, 'yyyy')` run in `opus1-r3`).
fn casts_to_timestamp(f: &ResolvedFunction) -> bool {
    let (at, ntz) = match f.id.name() {
        "date_format" => (0, true),
        "date_trunc" => (1, true),
        "hour" | "minute" | "second" => (0, false),
        _ => return false,
    };
    let t = f.args.get(at).and_then(|a| shown_type(a));
    matches!(
        t,
        Some(DataType::Date | DataType::String | DataType::CollatedString(_))
    ) || (ntz && matches!(t, Some(DataType::TimestampNtz)))
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
                positional_structs(x, y, case_sensitive)
            }
        }
        (Some(x), Some(y)) if positional_structs(x, y, case_sensitive) => true,
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

/// Structs of one field count, each field pair compatible (names may differ; a wider type of a
/// nested pair matches its names under `case_sensitive`, as Spark's session resolver does).
fn positional_structs(a: &DataType, b: &DataType, case_sensitive: bool) -> bool {
    match (a, b) {
        (DataType::Struct(x), DataType::Struct(y)) => {
            !x.fields().is_empty()
                && x.fields().len() == y.fields().len()
                && x.fields().iter().zip(y.fields()).all(|(f, g)| {
                    operands_compatible(Some(f.data_type()), Some(g.data_type()), case_sensitive)
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
