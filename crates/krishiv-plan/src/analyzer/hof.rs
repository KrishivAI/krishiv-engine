// SPDX-License-Identifier: Apache-2.0
//! Higher-order functions over arrays and maps: `transform`, `filter`,
//! `exists`, `forall`, `aggregate`/`reduce`, `zip_with`, `map_filter`,
//! `map_zip_with`, `transform_keys`, `transform_values` and `array_sort`
//! (Spark's `ArrayHigherOrderFunction`s with lambda bodies).
//!
//! A lambda (`x -> body`) is bound where it is used: the higher-order call
//! binds its value arguments first, pushes the lambda variables (typed with
//! the arguments' element types) for the body, then pops them (Spark's
//! `NamedLambdaVariable`). The variables shadow the enclosing query's
//! columns, innermost lambda first, case-sensitively per the session.
//!
//! An argument that is not a lambda is a constant lambda: the body is the
//! argument itself over hidden variables (as Spark analyzes
//! `transform(arr, 42)` to `[42, 42, 42]` and rejects `filter(arr, 1)` with
//! `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` quoting
//! `lambdafunction(1, namedlambdavariable())`). The wrapped node carries one
//! hidden variable per parameter the position takes at minimum, so names
//! and errors render as Spark's.
//!
//! The call itself is typed by the generic path once its children are
//! bound: the lambda child's type is its body's, and the registry's result
//! rules (see `krishiv-functions`) check the array and body types. When
//! `finish` is absent (`aggregate`/`reduce` of three arguments) the
//! analyzer uses the identity lambda over the accumulator, as Spark's
//! `ArrayAggregate` constructor does.

use std::sync::Arc;

use krishiv_common::error::{
    InternalError, InvalidLambdaFunctionCallDuplicateArgNames,
    InvalidLambdaFunctionCallNonHigherOrderFunction, InvalidLambdaFunctionCallNumArgsMismatch,
    WrongNumArgsWithoutSuggestion,
};
use krishiv_types::DataType;

use super::AnalysisError;
use super::bind::{addr, same, unsupported};
use super::catalog::ArgType;
use super::exprs::{Bound, Ctx, Cx, SPARK_DOC_ROOT};
use super::resolve::Resolver;
use crate::{BoundColumn, Expr, ExprNode, FunctionCall, Origin, ResolvedExpr};

/// One pushed lambda scope: its variables, innermost last.
#[derive(Debug, Clone, Default)]
pub(super) struct LambdaFrame {
    /// The variables, in parameter order.
    pub(super) vars: Vec<BoundColumn>,
}

/// A higher-order function over arrays and maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Hof {
    /// `transform(arr, x -> ...)` or `transform(arr, (x, i) -> ...)`.
    Transform,
    /// `filter(arr, x -> ...)`.
    Filter,
    /// `exists(arr, x -> ...)`.
    Exists,
    /// `forall(arr, x -> ...)`.
    ForAll,
    /// `aggregate(arr, zero, (acc, x) -> ...[, finish])`.
    Aggregate,
    /// `reduce(arr, zero, (acc, x) -> ...[, finish])`.
    Reduce,
    /// `zip_with(a1, a2, (x, y) -> ...)`.
    ZipWith,
    /// `map_filter(m, (k, v) -> ...)`.
    MapFilter,
    /// `map_zip_with(m1, m2, (k, v1, v2) -> ...)`.
    MapZipWith,
    /// `transform_keys(m, (k, v) -> ...)`.
    TransformKeys,
    /// `transform_values(m, (k, v) -> ...)`.
    TransformValues,
    /// `array_sort(arr[, (l, r) -> ...])`.
    ArraySort,
}

impl Hof {
    /// The higher-order function `name` calls, if one (ASCII
    /// case-insensitive, as function names are).
    #[must_use]
    pub(super) fn of(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "transform" => Self::Transform,
            "filter" => Self::Filter,
            "exists" => Self::Exists,
            "forall" => Self::ForAll,
            "aggregate" => Self::Aggregate,
            "reduce" => Self::Reduce,
            "zip_with" => Self::ZipWith,
            "map_filter" => Self::MapFilter,
            "map_zip_with" => Self::MapZipWith,
            "transform_keys" => Self::TransformKeys,
            "transform_values" => Self::TransformValues,
            "array_sort" => Self::ArraySort,
            _ => return None,
        })
    }

    /// The static shape of each call: the registry name, the argument-count
    /// range, its display, and the value and lambda argument positions.
    const fn info(
        self,
    ) -> (
        &'static str,
        (usize, usize),
        &'static str,
        &'static [usize],
        &'static [usize],
    ) {
        match self {
            Self::Transform => ("transform", (2, 2), "2", &[0], &[1]),
            Self::Filter => ("filter", (2, 2), "2", &[0], &[1]),
            Self::Exists => ("exists", (2, 2), "2", &[0], &[1]),
            Self::ForAll => ("forall", (2, 2), "2", &[0], &[1]),
            Self::Aggregate => ("aggregate", (3, 4), "[3, 4]", &[0, 1], &[2, 3]),
            Self::Reduce => ("reduce", (3, 4), "[3, 4]", &[0, 1], &[2, 3]),
            Self::ZipWith => ("zip_with", (3, 3), "3", &[0, 1], &[2]),
            Self::MapFilter => ("map_filter", (2, 2), "2", &[0], &[1]),
            Self::MapZipWith => ("map_zip_with", (3, 3), "3", &[0, 1], &[2]),
            Self::TransformKeys => ("transform_keys", (2, 2), "2", &[0], &[1]),
            Self::TransformValues => ("transform_values", (2, 2), "2", &[0], &[1]),
            Self::ArraySort => ("array_sort", (1, 2), "[1, 2]", &[0], &[1]),
        }
    }

    /// The name, lower case (as the registry lists it).
    const fn name(self) -> &'static str {
        self.info().0
    }

    /// The argument counts the call takes, as Spark's arity check sees
    /// them.
    const fn arity(self) -> (usize, usize) {
        self.info().1
    }

    /// The display of the expected argument count for a call of the wrong
    /// shape.
    const fn expected_args(self) -> &'static str {
        self.info().2
    }

    /// The value arguments, bound before the lambdas (the array or maps,
    /// and `aggregate`'s zero).
    fn values(self) -> &'static [usize] {
        self.info().3
    }

    /// The lambda arguments, in order (an absent `finish` is filled with
    /// the identity lambda).
    pub(super) fn lambdas(self) -> &'static [usize] {
        self.info().4
    }

    /// The lambda's parameter counts `(min, max)` at `position`.
    const fn params(self, position: usize) -> (usize, usize) {
        // The merge takes the accumulator and the element, the finish takes
        // the accumulator, and `transform`'s index is optional.
        if position == 2 && matches!(self, Self::Aggregate | Self::Reduce) {
            return (2, 2);
        }
        if position == 1 && matches!(self, Self::Transform) {
            return (1, 2);
        }
        match self {
            Self::MapZipWith => (3, 3),
            Self::Transform
            | Self::Filter
            | Self::Exists
            | Self::ForAll
            | Self::Aggregate
            | Self::Reduce => (1, 1),
            Self::ZipWith
            | Self::MapFilter
            | Self::TransformKeys
            | Self::TransformValues
            | Self::ArraySort => (2, 2),
        }
    }

    /// The display of `expectedNumArgs` for a lambda of the wrong shape.
    const fn expected_params(self, position: usize) -> &'static str {
        match self.params(position) {
            (1, 2) => "1 to 2",
            (1, 1) => "1",
            (2, 2) => "2",
            (3, 3) => "3",
            _ => "?",
        }
    }
}

/// Whether `f` takes a lambda as a direct argument: only a higher-order
/// call may.
pub(super) fn has_direct_lambda(f: &FunctionCall) -> bool {
    f.args
        .iter()
        .any(|a| matches!(a.node, ExprNode::Lambda { .. }))
}

/// The error for a lambda in `name`'s argument list when `name` is no
/// higher-order function (Spark's `INVALID_LAMBDA_FUNCTION_CALL` names the
/// enclosing expression's class).
pub(super) fn not_higher_order(name: &str, origin: Origin) -> AnalysisError {
    AnalysisError::new(
        InvalidLambdaFunctionCallNonHigherOrderFunction {
            class: catalyst_class(name),
        },
        origin,
    )
}

/// The Catalyst class Spark's message names for a function called with a
/// lambda. Pinned by the goldens for `upper` and `ceil`; anything else is
/// the usual class shape (`Upper` for `upper`), which is best-effort.
fn catalyst_class(name: &str) -> String {
    if name.eq_ignore_ascii_case("ceil") || name.eq_ignore_ascii_case("ceiling") {
        return "org.apache.spark.sql.catalyst.expressions.CeilExpressionBuilder$".to_owned();
    }
    let mut class = name.to_ascii_lowercase();
    if let Some(first) = class.get_mut(..1) {
        first.make_ascii_uppercase();
    }
    format!("org.apache.spark.sql.catalyst.expressions.{class}")
}

impl Resolver<'_> {
    /// The innermost lambda variable `name` names, if any.
    pub(super) fn lambda_var(&self, name: &str) -> Option<BoundColumn> {
        let cs = self.ctx.conf.case_sensitive;
        self.lambdas
            .iter()
            .rev()
            .flat_map(|f| f.vars.iter().rev())
            .find(|v| same(v.name.as_str(), name, cs))
            .cloned()
    }

    /// Binds the higher-order call's value children first, then its lambda
    /// children (an absent `finish` is the identity lambda over the
    /// accumulator): the values first, so the lambda variables take their
    /// types. Returns each child with the address the generic path binds
    /// it under; the caller records them and leaves the rest of the call
    /// to the generic path.
    ///
    /// # Errors
    /// A lambda error, as Spark reports it.
    pub(super) fn hof_children(
        &mut self,
        f: &FunctionCall,
        h: Hof,
        kids: &[(Arc<Expr>, Ctx)],
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Vec<(usize, Bound)>, AnalysisError> {
        // Arity first, as Spark checks it.
        let (min, max) = h.arity();
        if f.args.len() < min || f.args.len() > max {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: format!("`{}`", h.name()),
                    expected_num: h.expected_args().to_owned(),
                    actual_num: f.args.len().to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                origin,
            ));
        }
        let at = |i: usize| {
            kids.iter()
                .find(|(c, _)| addr(&**c) == addr(&*f.args[i]))
                .map(|(_, k)| *k)
        };
        let ctx_of = |i: usize| {
            at(i).unwrap_or(Ctx {
                agg: false,
                window_fn: false,
            })
        };
        let key = |i: usize| addr(&*f.args[i]);
        let mut done: Vec<(usize, Bound)> = Vec::with_capacity(f.args.len() + 1);
        for &i in h.values() {
            if i < f.args.len() {
                let bound = self.expr_in(ctx_of(i), &f.args[i], cx)?;
                done.push((key(i), bound));
            }
        }
        let values: Vec<Option<DataType>> = h
            .values()
            .iter()
            .map(|&i| {
                done.iter()
                    .find(|(k, _)| *k == key(i))
                    .map(|(_, b)| b.ty.data_type.clone())
            })
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default();
        for &i in h.lambdas() {
            if i < f.args.len() {
                let bound = self.hof_lambda(h, ctx_of(i), i, &values, &f.args[i], cx)?;
                done.push((key(i), bound));
            } else if matches!(h, Hof::Aggregate | Hof::Reduce) {
                // An absent `finish` is the identity lambda over the
                // accumulator (Spark's `ArrayAggregate` default): one
                // hidden variable whose body is itself.
                let Some(zero) = values.get(1) else {
                    return Err(AnalysisError::new(
                        InternalError {
                            message: format!("{} typed without its zero argument", h.name()),
                        },
                        origin,
                    ));
                };
                let var = BoundColumn {
                    id: self.fresh(),
                    name: "namedlambdavariable()".to_owned(),
                    data_type: zero.clone().unwrap_or(DataType::Null),
                    nullable: true,
                };
                let body = Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::LambdaVar(var.clone())))
                        .with_origin(origin),
                );
                done.push((
                    usize::MAX - i,
                    Bound {
                        expr: Arc::new(
                            Expr::new(ExprNode::Lambda {
                                params: vec![var.clone()],
                                body: Arc::clone(&body),
                            })
                            .with_origin(origin),
                        ),
                        ty: ArgType::new(var.data_type.clone(), true),
                        agg: None,
                        win: None,
                        outer: None,
                        lateral: Vec::new(),
                        shown: None,
                    },
                ));
            }
        }
        Ok(done)
    }

    /// Binds the lambda `node` at `position` against the value arguments
    /// `values` (already bound above): its variables, then its body. A node
    /// that is no lambda is a constant lambda over as many hidden variables
    /// as the position takes at minimum (see the module doc).
    ///
    /// # Errors
    /// A lambda error, as Spark reports it.
    fn hof_lambda(
        &mut self,
        h: Hof,
        step: Ctx,
        position: usize,
        values: &[Option<DataType>],
        node: &Arc<Expr>,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let hidden = !matches!(node.node, ExprNode::Lambda { .. });
        let (min, max) = h.params(position);
        let (params, inner) = match &node.node {
            ExprNode::Lambda { params, body } => (
                params.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
                Arc::clone(body),
            ),
            // A constant lambda: the argument is the body.
            _ => (
                (0..min)
                    .map(|_| "namedlambdavariable()".to_owned())
                    .collect(),
                Arc::clone(node),
            ),
        };
        if params.len() < min || params.len() > max {
            return Err(AnalysisError::new(
                InvalidLambdaFunctionCallNumArgsMismatch {
                    expected_num_args: h.expected_params(position).to_owned(),
                    actual_num_args: params.len().to_string(),
                },
                node.origin,
            ));
        }
        if !hidden && let Some(dup) = duplicate_param(&params, self.ctx.conf.case_sensitive) {
            return Err(AnalysisError::new(
                InvalidLambdaFunctionCallDuplicateArgNames {
                    args: format!("[{}]", dup.join(", ")),
                    case_sensitive_config: format!(
                        "spark.sql.caseSensitive is set to {}",
                        self.ctx.conf.case_sensitive
                    ),
                },
                node.origin,
            ));
        }
        let vars = self.lambda_params(h, position, values, &params);
        self.lambdas.push(LambdaFrame { vars: vars.clone() });
        let body = self.expr_in(step, &inner, cx);
        self.lambdas.pop();
        let body = body?;
        // An aggregate or window over the lambda's elements is no outer
        // query aggregate: a constant lambda's own aggregates stay outer
        // aggregates, as the generic path resolves them.
        if !hidden && body.agg.is_some() {
            return Err(unsupported(
                "aggregate functions in lambda functions",
                node.origin,
            ));
        }
        if !hidden && body.win.is_some() {
            return Err(unsupported(
                "window functions in lambda functions",
                node.origin,
            ));
        }
        let ty = body.ty.clone();
        Ok(Bound {
            expr: Arc::new(
                Expr::new(ExprNode::Lambda {
                    params: vars,
                    body: Arc::clone(&body.expr),
                })
                .with_origin(node.origin),
            ),
            ty,
            agg: hidden.then_some(body.agg).flatten(),
            win: hidden.then_some(body.win).flatten(),
            outer: body.outer,
            lateral: body.lateral,
            shown: (!hidden).then(|| Arc::clone(node)),
        })
    }

    /// The lambda variables at `position` of `h` over `params`, typed from
    /// the value arguments `values` (element, key, value, accumulator and
    /// index types, as each function binds them).
    fn lambda_params(
        &mut self,
        h: Hof,
        position: usize,
        values: &[Option<DataType>],
        params: &[String],
    ) -> Vec<BoundColumn> {
        // The array argument's element type and `containsNull`, or `NULL`
        // when the collection check fails: the result rule reports the
        // collection, so the lambda still binds (names resolve before
        // types, as in Spark).
        let array_element = |i: usize| match values.get(i).and_then(|v| v.as_ref()) {
            Some(DataType::Array(e)) => (e.element_type().clone(), e.contains_null()),
            _ => (DataType::Null, true),
        };
        // A map argument's key type, value type and `valueContainsNull`.
        let map_parts = |i: usize| match values.get(i).and_then(|v| v.as_ref()) {
            Some(DataType::Map(m)) => (
                m.key_type().clone(),
                m.value_type().clone(),
                m.value_contains_null(),
            ),
            _ => (DataType::Null, DataType::Null, true),
        };
        let mut vars = Vec::with_capacity(params.len());
        let mut var = |name: String, data_type: DataType, nullable: bool| {
            vars.push(BoundColumn {
                id: self.fresh(),
                name,
                data_type,
                nullable,
            });
        };
        match h {
            Hof::Transform => {
                let (element, contains) = array_element(0);
                var(params[0].clone(), element, contains);
                if params.len() == 2 {
                    var(params[1].clone(), DataType::Integer, false);
                }
            }
            Hof::Filter | Hof::Exists | Hof::ForAll => {
                let (element, contains) = array_element(0);
                var(params[0].clone(), element, contains);
            }
            Hof::Aggregate | Hof::Reduce if position == 2 => {
                let zero = values.get(1).cloned().flatten().unwrap_or(DataType::Null);
                let (element, contains) = array_element(0);
                var(params[0].clone(), zero, true);
                var(params[1].clone(), element, contains);
            }
            Hof::Aggregate | Hof::Reduce => {
                let zero = values.get(1).cloned().flatten().unwrap_or(DataType::Null);
                var(params[0].clone(), zero, true);
            }
            Hof::ZipWith => {
                let (first, _) = array_element(0);
                let (second, _) = array_element(1);
                // The shorter array is padded with NULLs, so both
                // variables are always nullable.
                var(params[0].clone(), first, true);
                var(params[1].clone(), second, true);
            }
            Hof::MapFilter | Hof::TransformKeys | Hof::TransformValues => {
                let (key, value, contains) = map_parts(0);
                var(params[0].clone(), key, false);
                var(params[1].clone(), value, contains);
            }
            Hof::MapZipWith => {
                let (key, first, _) = map_parts(0);
                let (_, second, _) = map_parts(1);
                var(params[0].clone(), key, false);
                // A missing key reads NULL, so both values are always
                // nullable.
                var(params[1].clone(), first, true);
                var(params[2].clone(), second, true);
            }
            Hof::ArraySort => {
                let (element, contains) = array_element(0);
                var(params[0].clone(), element.clone(), contains);
                var(params[1].clone(), element, contains);
            }
        }
        vars
    }
}

/// The first parameter `params` repeats (case-sensitively per `cs`), if
/// any.
fn duplicate_param(params: &[String], cs: bool) -> Option<Vec<String>> {
    for (i, a) in params.iter().enumerate() {
        for b in &params[..i] {
            if same(a, b, cs) {
                return Some(vec![a.clone()]);
            }
        }
    }
    None
}
