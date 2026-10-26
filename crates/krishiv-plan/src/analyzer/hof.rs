// SPDX-License-Identifier: Apache-2.0
//! Higher-order functions over arrays: `transform`, `filter`, `exists` and
//! `forall` (Spark's `ArrayHigherOrderFunction`s with lambda bodies).
//!
//! A lambda (`x -> body`) is bound where it is used: the higher-order call
//! binds its array argument first, pushes the lambda variables (typed with
//! the array's element type) for the body, then pops them (Spark's
//! `NamedLambdaVariable`). The variables shadow the enclosing query's
//! columns, innermost lambda first, case-sensitively per the session.
//!
//! An argument that is not a lambda is a constant lambda: the body is the
//! argument itself over hidden variables, as Spark analyzes
//! `transform(arr, 42)` to `[42, 42, 42]` and rejects `filter(arr, 1)` with
//! `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` quoting
//! `lambdafunction(1, namedlambdavariable())`. The wrapped node carries the
//! hidden variables so names and errors render as Spark's.
//!
//! The call itself is typed by the generic path once its children are
//! bound: the lambda child's type is its body's, and the registry's result
//! rules (see `krishiv-functions`) check the array and body types.

use std::sync::Arc;

use krishiv_common::error::{
    InvalidLambdaFunctionCallDuplicateArgNames, InvalidLambdaFunctionCallNonHigherOrderFunction,
    InvalidLambdaFunctionCallNumArgsMismatch, WrongNumArgsWithoutSuggestion,
};
use krishiv_types::{ArrayType, DataType};

use super::AnalysisError;
use super::bind::{addr, same, unsupported};
use super::catalog::ArgType;
use super::exprs::{Bound, Ctx, Cx, SPARK_DOC_ROOT};
use super::resolve::Resolver;
use crate::{BoundColumn, Expr, ExprNode, FunctionCall, Origin};

/// One pushed lambda scope: its variables, innermost last.
#[derive(Debug, Clone, Default)]
pub(super) struct LambdaFrame {
    /// The variables, in parameter order.
    pub(super) vars: Vec<BoundColumn>,
}

/// A higher-order function over arrays.
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
            _ => return None,
        })
    }

    /// The name, lower case (as the registry lists it).
    const fn name(self) -> &'static str {
        match self {
            Self::Transform => "transform",
            Self::Filter => "filter",
            Self::Exists => "exists",
            Self::ForAll => "forall",
        }
    }

    /// The argument holding the array.
    const fn array(self) -> usize {
        0
    }

    /// The argument holding the lambda.
    const fn lambda(self) -> usize {
        1
    }

    /// The lambda's parameter counts `(min, max)`.
    const fn params(self) -> (usize, usize) {
        match self {
            // The index is the optional second variable.
            Self::Transform => (1, 2),
            Self::Filter | Self::Exists | Self::ForAll => (1, 1),
        }
    }

    /// The display of `expectedNumArgs` for a lambda of the wrong shape.
    const fn expected_params(self) -> &'static str {
        match self {
            Self::Transform => "1 to 2",
            Self::Filter | Self::Exists | Self::ForAll => "1",
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
            .find(|v| same(&v.name, name, cs))
            .cloned()
    }

    /// Binds the higher-order call's array child, then its lambda child:
    /// the array first, so the lambda variables take its element type.
    /// Returns each child with the address the generic path binds it under;
    /// the caller records both and leaves the rest of the call to the
    /// generic path.
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
    ) -> Result<[(usize, Bound); 2], AnalysisError> {
        // Arity first, as Spark checks it: every function here takes the
        // array and the lambda.
        if f.args.len() != 2 {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: format!("`{}`", h.name()),
                    expected_num: "2".to_owned(),
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
        let array_ctx = at(h.array()).unwrap_or(Ctx {
            agg: false,
            window_fn: false,
        });
        let lambda_ctx = at(h.lambda()).unwrap_or(array_ctx);
        let array = self.expr_in(array_ctx, &f.args[h.array()], cx)?;
        let lambda = self.hof_lambda(h, lambda_ctx, &array.ty, &f.args[h.lambda()], cx)?;
        Ok([
            (addr(&*f.args[h.array()]), array),
            (addr(&*f.args[h.lambda()]), lambda),
        ])
    }

    /// Binds the lambda `node` against `array_ty`: its variables, then its
    /// body. A node that is no lambda is a constant lambda over one hidden
    /// variable (see the module doc).
    ///
    /// # Errors
    /// A lambda error, as Spark reports it.
    fn hof_lambda(
        &mut self,
        h: Hof,
        step: Ctx,
        array_ty: &ArgType,
        node: &Arc<Expr>,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let hidden = !matches!(node.node, ExprNode::Lambda { .. });
        let (params, inner) = match &node.node {
            ExprNode::Lambda { params, body } => (
                params.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
                Arc::clone(body),
            ),
            // A constant lambda: the argument is the body.
            _ => (vec!["namedlambdavariable()".to_owned()], Arc::clone(node)),
        };
        let (min, max) = h.params();
        if params.len() < min || params.len() > max {
            return Err(AnalysisError::new(
                InvalidLambdaFunctionCallNumArgsMismatch {
                    expected_num_args: h.expected_params().to_owned(),
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
        // The element type, or `NULL` when the array check fails: the
        // result rule reports the array, so the lambda still binds (names
        // resolve before types, as in Spark).
        let element = match &array_ty.data_type {
            Some(DataType::Array(element)) => element.as_ref().clone(),
            _ => ArrayType::new(DataType::Null, true),
        };
        let mut vars = vec![BoundColumn {
            id: self.fresh(),
            name: params[0].clone(),
            data_type: element.element_type().clone(),
            nullable: element.contains_null(),
        }];
        if params.len() == 2 {
            vars.push(BoundColumn {
                id: self.fresh(),
                name: params[1].clone(),
                data_type: DataType::Integer,
                nullable: false,
            });
        }
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
