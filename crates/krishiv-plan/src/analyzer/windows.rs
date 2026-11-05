// SPDX-License-Identifier: Apache-2.0
//! Window expressions: binding (Spark's `ResolveWindowFrame`,
//! `ResolveWindowOrder` and the window checks of `CheckAnalysis`) and their
//! extraction into [`PlanNode::Window`] relations (Spark's
//! `ExtractWindowExpressions`).
//!
//! # Binding
//! A window's function is bound as a window call: `sum(x)` in
//! `sum(x) OVER w` is not an aggregate of the operator, so it may nest one
//! (`sum(sum(x)) OVER w` over a `GROUP BY`). Then, in Spark's order:
//! `MISSING_WINDOW_SPECIFICATION` for a name no `WINDOW` clause defines,
//! lateral column aliases (`LATERAL_COLUMN_ALIAS_IN_WINDOW`), the function
//! (`UNSUPPORTED_EXPR_FOR_WINDOW` unless a window or aggregate function),
//! a window function's ordering (`WINDOW_FUNCTION_FRAME_NOT_ORDERED`) and
//! required frame (`_LEGACY_ERROR_TEMP_1036`), Spark's default frame for
//! an aggregate (`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` with
//! an ordering, else `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED
//! FOLLOWING`), the frame's bounds, a `RANGE` frame without ordering, and
//! `DISTINCT`. The bound window always has its frame, so its generated
//! name shows it, as Spark's does.
//!
//! # Extraction
//! Where a select list holds windows ([`Resolver::windowed`]): every
//! input of a window (function arguments, the aggregate filter,
//! partitioning and ordering expressions) that is neither a column of the
//! operator's input nor a literal is computed by a projection below
//! (`_w0`, `_w1`, ...; windows nested in them are extracted the same way,
//! one level further down); then one [`PlanNode::Window`] per distinct
//! partitioning and ordering, stacked in order of first use; the select
//! list reads each window's result column (`_we0`, ...). Over an aggregate
//! the select list was first rewritten over the aggregate's output
//! (`Resolver::lift`), and a `HAVING` filters the aggregate below the
//! windows, as Spark's rule for `Filter(Aggregate)` does.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchRangeFrameInvalidType,
    DatatypeMismatchRangeFrameMultiOrder, DatatypeMismatchRangeFrameWithoutOrder,
    DatatypeMismatchSpecifiedWindowFrameInvalidBound,
    DatatypeMismatchSpecifiedWindowFrameUnacceptedType,
    DatatypeMismatchSpecifiedWindowFrameWrongComparison, DistinctWindowFunctionUnsupported,
    FailedFunctionCall, InvalidAggregateFilterContainsWindowFunction,
    InvalidSqlSyntaxFunctionWithUnsupportedSyntax, InvalidWindowSpecForAggregationFunc,
    KrishivError, LegacyErrorTemp1031, LegacyErrorTemp1035, LegacyErrorTemp1036,
    MissingWindowSpecification, UnsupportedExprForOperator, UnsupportedExprForWindow,
    UnsupportedFeatureLateralColumnAliasInWindow, WindowFunctionFrameNotOrdered,
    WindowFunctionNotAllowedInClause, WindowFunctionWithoutOverClause,
};

use super::bind::unsupported;
use super::exprs::{Bound, Cx, WindowUse, pretty, semantic_key};
use super::ops::col_expr;
use super::resolve::{Attr, Rel, Resolver};
use super::{AnalysisError, ArgType, Constant};
use crate::{
    Expr, ExprNode, FrameBound, FrameUnits, FunctionKind, Literal, Origin, PlanNode, ResolvedExpr,
    ResolvedFunction, SortOrder, WindowExpr, WindowFrame, WindowSpec, WindowSpecRef,
};

/// Spark's documentation root, as its error messages link it.
const SPARK_DOC_ROOT: &str = "https://spark.apache.org/docs/latest";

/// Window functions Spark writes upper case (`RankLike.sql`).
const RANK_LIKE: [&str; 3] = ["rank", "dense_rank", "percent_rank"];

/// A window function's text as Spark's errors show it: `RANK()`,
/// `row_number()`, `ntile(2)`.
pub(super) fn window_function_sql(name: &str, shown: &Expr) -> String {
    if RANK_LIKE.contains(&name) {
        format!("{}()", name.to_uppercase())
    } else {
        pretty(shown, None)
    }
}

/// `WINDOW_FUNCTION_WITHOUT_OVER_CLAUSE` for `name` called as `shown`.
pub(super) fn without_over(name: &str, shown: &Expr, origin: Origin) -> AnalysisError {
    AnalysisError::new(
        WindowFunctionWithoutOverClause {
            func_name: format!("\"{}\"", window_function_sql(name, shown)),
        },
        origin,
    )
}

/// `_LEGACY_ERROR_TEMP_1031`: a window inside an aggregate call.
pub(super) fn window_in_aggregate(origin: Origin) -> AnalysisError {
    AnalysisError::new(LegacyErrorTemp1031 {}, origin)
}

/// `INVALID_AGGREGATE_FILTER.CONTAINS_WINDOW_FUNCTION` for `filter`.
pub(super) fn window_in_filter(filter: &Bound, origin: Origin) -> AnalysisError {
    let window = first_window(&filter.shown()).map_or_else(String::new, |w| pretty(&w, None));
    AnalysisError::new(
        InvalidAggregateFilterContainsWindowFunction {
            filter_expr: format!("\"{}\"", pretty(&filter.shown(), None)),
            window_expr: format!("\"{window}\""),
        },
        origin,
    )
}

/// The error for a window expression where `use_` forbids one; `b` holds
/// it.
pub(super) fn misplaced(use_: WindowUse, b: &Bound, origin: Origin) -> AnalysisError {
    let clause = |name: &str| -> KrishivError {
        WindowFunctionNotAllowedInClause {
            clause_name: name.to_owned(),
        }
        .into()
    };
    let error = match use_ {
        WindowUse::Where => clause("WHERE"),
        WindowUse::Having => clause("HAVING"),
        WindowUse::Allowed | WindowUse::Other => {
            let window = first_window(&b.shown()).map_or_else(String::new, |w| pretty(&w, None));
            UnsupportedExprForOperator {
                invalid_expr_sqls: format!("\"{window}\""),
            }
            .into()
        }
    };
    AnalysisError::new(error, origin)
}

/// The first window expression in `e`, pre-order, outside subqueries.
fn first_window(e: &Arc<Expr>) -> Option<Arc<Expr>> {
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if matches!(x.node, ExprNode::Window(_)) {
            return Some(x);
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| kids.push(Arc::clone(c)));
        stack.extend(kids.into_iter().rev());
    }
    None
}

/// A frame as Spark's `toString` writes it in `_LEGACY_ERROR_TEMP_1036`:
/// `specifiedwindowframe(RowFrame, unboundedpreceding$(), -1)`.
fn frame_string(f: &WindowFrame) -> String {
    let bound = |b: &FrameBound| match b {
        FrameBound::UnboundedPreceding => "unboundedpreceding$()".to_owned(),
        FrameBound::CurrentRow => "currentrow$()".to_owned(),
        FrameBound::UnboundedFollowing => "unboundedfollowing$()".to_owned(),
        FrameBound::Offset(e) => match crate::expr::negated(e) {
            Some(n) => format!("-{}", pretty(n, None)),
            None => pretty(e, None),
        },
    };
    let units = match f.units {
        FrameUnits::Rows => "RowFrame",
        FrameUnits::Range => "RangeFrame",
    };
    format!(
        "specifiedwindowframe({units}, {}, {})",
        bound(&f.lower),
        bound(&f.upper)
    )
}

/// A frame bound as Spark's `sql` writes it, quoted.
fn bound_sql(b: &FrameBound) -> String {
    let text = match b {
        FrameBound::UnboundedPreceding => "UNBOUNDED PRECEDING".to_owned(),
        FrameBound::CurrentRow => "CURRENT ROW".to_owned(),
        FrameBound::UnboundedFollowing => "UNBOUNDED FOLLOWING".to_owned(),
        FrameBound::Offset(e) => match crate::expr::negated(e) {
            Some(n) => format!("{} PRECEDING", pretty(n, None)),
            None => format!("{} FOLLOWING", pretty(e, None)),
        },
    };
    format!("\"{text}\"")
}

/// Whether `f` is a frame `lag`/`lead` makes: `ROWS BETWEEN k FOLLOWING
/// AND k FOLLOWING` with `k` an `int` literal (what the user writes is a
/// negation for `PRECEDING`, and never this frame for these functions).
fn is_offset_frame(f: &WindowFrame) -> bool {
    let literal = |b: &FrameBound| matches!(b, FrameBound::Offset(e) if matches!(e.node, ExprNode::Literal(Literal::Integer(_))));
    f.units == FrameUnits::Rows && literal(&f.lower) && f.lower == f.upper
}

/// Whether a `lag`/`lead` offset is of type `INT`: any foldable `INT`
/// expression counts (`1 + 1`, `abs(-2)`), as in Spark.
fn offset_type(e: &Expr) -> bool {
    expr_type(e).data_type == Some(krishiv_types::DataType::Integer)
}

fn is_null_literal(e: &Expr) -> bool {
    matches!(e.node, ExprNode::Literal(Literal::Null(_)))
}

/// Spark's `expectedType` of a `RANGE` frame bound.
const RANGE_BOUND_TYPES: &str =
    "(\"NUMERIC\" or \"INTERVAL DAY TO SECOND\" or \"INTERVAL YEAR TO MONTH\" or \"INTERVAL\")";

/// A `RANGE` frame bound's value in the units of its ordering key: whole
/// numbers (integers, dates, and decimals unscaled) or floating point.
#[derive(Clone, Debug)]
enum RangeValue {
    Int(i128),
    Float(f64),
}

impl RangeValue {
    fn above(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a > b,
            (Self::Float(a), Self::Float(b)) => a > b,
            _ => false,
        }
    }

    /// The literal of the key's type (`None` past that type's range).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a double bound narrowed to a FLOAT key, as Spark's cast does"
    )]
    fn literal(&self, key: &krishiv_types::DataType) -> Option<Literal> {
        use krishiv_types::DataType as T;
        Some(match (self, key) {
            (Self::Int(v), T::Byte) => Literal::Byte(i8::try_from(*v).ok()?),
            (Self::Int(v), T::Short) => Literal::Short(i16::try_from(*v).ok()?),
            (Self::Int(v), T::Long) => Literal::Long(i64::try_from(*v).ok()?),
            (Self::Int(v), T::Decimal(d)) => Literal::Decimal {
                unscaled: *v,
                data_type: *d,
            },
            (Self::Int(v), _) => Literal::Integer(i32::try_from(*v).ok()?),
            (Self::Float(v), T::Float) => Literal::Float(*v as f32),
            (Self::Float(v), _) => Literal::Double(*v),
        })
    }
}

/// The value of the `RANGE` frame bound `x` (a literal, or the negation of
/// one for `PRECEDING`) cast to the type of the ordering key; `None` for
/// anything but a literal, or a value the key's type cannot hold.
fn range_value(x: &Arc<Expr>, key: &krishiv_types::DataType) -> Option<RangeValue> {
    range_value_signed(x, key, false)
}

/// `range_value` with the sign applied to the literal before the cast, as
/// Spark casts `UnaryMinus(n)` (so `128 PRECEDING` fits a TINYINT key).
#[expect(
    clippy::cast_possible_truncation,
    reason = "a bound is cast to its key's type as Spark's cast does: doubles truncate to whole numbers"
)]
fn range_value_signed(
    x: &Arc<Expr>,
    key: &krishiv_types::DataType,
    negate: bool,
) -> Option<RangeValue> {
    use krishiv_types::DataType as T;
    if let Some(inner) = crate::expr::negated(x) {
        return range_value_signed(inner, key, !negate);
    }
    let ExprNode::Literal(l) = &x.node else {
        return None;
    };
    // The literal as an exact decimal `(unscaled, scale)` or a double.
    enum Lit {
        Exact(i128, u8),
        Double(f64),
    }
    let lit = match l {
        Literal::Byte(v) => Lit::Exact(i128::from(*v), 0),
        Literal::Short(v) => Lit::Exact(i128::from(*v), 0),
        Literal::Integer(v) => Lit::Exact(i128::from(*v), 0),
        Literal::Long(v) => Lit::Exact(i128::from(*v), 0),
        Literal::Decimal {
            unscaled,
            data_type,
        } => Lit::Exact(*unscaled, data_type.scale()),
        Literal::Float(v) => Lit::Double(f64::from(*v)),
        Literal::Double(v) => Lit::Double(*v),
        _ => return None,
    };
    let lit = match (lit, negate) {
        (lit, false) => lit,
        (Lit::Exact(v, s), true) => Lit::Exact(v.checked_neg()?, s),
        (Lit::Double(d), true) => Lit::Double(-d),
    };
    let fits =
        |v: i128, min: i128, max: i128| (min..=max).contains(&v).then_some(RangeValue::Int(v));
    let truncated = |lit: &Lit| match lit {
        Lit::Exact(v, s) => Some(v / 10_i128.checked_pow(u32::from(*s))?),
        Lit::Double(d) if d.is_finite() && d.abs() < 1e30 => Some(d.trunc() as i128),
        Lit::Double(_) => None,
    };
    match key {
        T::Byte => fits(truncated(&lit)?, i8::MIN.into(), i8::MAX.into()),
        T::Short => fits(truncated(&lit)?, i16::MIN.into(), i16::MAX.into()),
        T::Integer | T::Date => fits(truncated(&lit)?, i32::MIN.into(), i32::MAX.into()),
        T::Long => fits(truncated(&lit)?, i64::MIN.into(), i64::MAX.into()),
        T::Float | T::Double => Some(match lit {
            // Parsed from decimal text so the result is correctly rounded, as
            // BigDecimal.doubleValue is (no double rounding for wide unscaled values).
            Lit::Exact(v, s) => RangeValue::Float(format!("{v}e-{s}").parse::<f64>().ok()?),
            Lit::Double(d) => RangeValue::Float(d),
        })
        // A FLOAT key's bounds are single precision before they are ordered.
        .map(|v| match (key, v) {
            (T::Float, RangeValue::Float(d)) => RangeValue::Float(f64::from(d as f32)),
            (_, v) => v,
        }),
        T::Decimal(d) => match lit {
            Lit::Exact(v, s) if s <= d.scale() => {
                let v = v.checked_mul(10_i128.checked_pow(u32::from(d.scale() - s))?)?;
                // The bound is cast to the key's DECIMAL(p, s) and must fit it.
                let max = 10_i128.checked_pow(u32::from(d.precision()))? - 1;
                (-max..=max).contains(&v).then_some(RangeValue::Int(v))
            }
            _ => None,
        },
        _ => None,
    }
}

/// The value of a frame offset: a constant, or the negation of one (`n
/// PRECEDING`).
fn offset_value(e: &Arc<Expr>, fold: &dyn Fn(&Arc<Expr>) -> Option<Constant>) -> Option<Constant> {
    match crate::expr::negated(e) {
        Some(x) => match offset_value(x, fold)? {
            Constant::Integral(v) => {
                Some(v.checked_neg().map_or(Constant::Null, Constant::Integral))
            }
            other => Some(other),
        },
        None => fold(e),
    }
}

/// Spark's text of `function OVER (spec)`, as its errors quote it.
fn window_text(function: &Arc<Expr>, spec: &WindowSpec) -> String {
    pretty(
        &Expr::new(ExprNode::Window(WindowExpr {
            function: Arc::clone(function),
            window: WindowSpecRef::Spec(spec.clone()),
        })),
        None,
    )
}

impl Resolver<'_> {
    /// Select `items` (some holding windows) over `base`, then the
    /// attributes `pass`: the windows over `base`, then the select list
    /// over their columns (with projections for lateral aliases).
    #[inline(never)]
    pub(super) fn window_project(
        &mut self,
        origin: Origin,
        base: &Rel,
        items: &[Bound],
        pass: &[Attr],
    ) -> Result<Rel, AnalysisError> {
        let exprs: Vec<Arc<Expr>> = items.iter().map(|b| Arc::clone(&b.expr)).collect();
        let (windows, exprs) = self.windowed(base, &exprs)?;
        let items: Vec<Bound> = items
            .iter()
            .zip(exprs)
            .map(|(b, expr)| Bound { expr, ..b.clone() })
            .collect();
        Ok(self.lateral_project(origin, &windows, &items, pass))
    }

    /// A filter over an aggregate whose select list holds windows (Spark's
    /// `ExtractWindowExpressions` on `Filter(Aggregate)`): it filters the
    /// aggregate below the windows and reads the aggregate's output and
    /// grouping attributes, not the windows' columns. Aggregate calls it
    /// uses are added to the aggregate.
    #[inline(never)]
    pub(super) fn having_below_windows(
        &mut self,
        p: &crate::LogicalPlan,
        info: &Arc<super::resolve::AggInfo>,
        predicate: &Arc<Expr>,
        env: &Arc<super::resolve::Env>,
    ) -> Result<Rel, AnalysisError> {
        let top = info.top.clone().unwrap_or_else(|| Arc::from(Vec::new()));
        if let Some(b) = top.iter().find(|b| !b.lateral.is_empty()) {
            let lca = b
                .lateral
                .first()
                .and_then(|id| info.out.iter().find(|a| a.col.id == *id))
                .map_or_else(String::new, |a| a.col.name.clone());
            return Err(AnalysisError::new(
                krishiv_common::error::UnsupportedFeatureLateralColumnAliasInAggregateWithWindowAndHaving {
                    lca: format!("`{lca}`"),
                },
                p.origin,
            ));
        }
        let bare = super::resolve::AggInfo {
            top: None,
            ..(**info).clone()
        };
        let below = self.aggregate_rel(Origin::UNKNOWN, bare, &[])?;
        let (bounds, grown, _) =
            self.over_agg(&below, std::slice::from_ref(predicate), env, "Filter")?;
        let b = bounds
            .into_iter()
            .next()
            .ok_or_else(|| unsupported("a filter shape", p.origin))?;
        super::ops::boolean_filter(&b, p.origin)?;
        let grown = grown
            .agg
            .ok_or_else(|| unsupported("a filter shape", p.origin))?;
        let info = super::resolve::AggInfo {
            top: Some(top),
            having: Some(b.expr),
            ..(*grown).clone()
        };
        self.aggregate_rel(p.origin, info, &[])
    }

    /// A resolved [`PlanNode::Window`] analysed again: its expressions
    /// bound over its input, its output the input's and the windows'.
    #[inline(never)]
    pub(super) fn window_rel(
        &mut self,
        p: &crate::LogicalPlan,
        input: &Arc<crate::LogicalPlan>,
        exprs: &[Arc<Expr>],
        partition_by: &[Arc<Expr>],
        order_by: &[SortOrder],
        env: &Arc<super::resolve::Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let rels = [&c];
        let mut cx = Self::cx(&c.out, &rels, env, "Window");
        cx.windows = WindowUse::Allowed;
        let mut out = c.out.to_vec();
        let mut bound = Vec::with_capacity(exprs.len());
        for e in exprs {
            let b = self.bind(e, &cx)?;
            match &b.expr.node {
                ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => out.push(Attr {
                    col: output.clone(),
                    qual: super::ops::no_qual(),
                    source: None,
                }),
                _ => return Err(unsupported("a window relation without aliases", p.origin)),
            }
            bound.push(b.expr);
        }
        let partition_by = partition_by
            .iter()
            .map(|e| self.bind(e, &cx).map(|b| b.expr))
            .collect::<Result<Vec<_>, _>>()?;
        let order_by = order_by
            .iter()
            .map(|o| {
                self.bind(&o.expr, &cx).map(|b| SortOrder {
                    expr: b.expr,
                    ..o.clone()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let node = PlanNode::Window {
            input: Arc::clone(&c.plan),
            exprs: bound,
            partition_by,
            order_by,
        };
        Ok(self.make(p.origin, node, out, &[&c]))
    }

    /// Binds the window expression `e` (`w`) from its bound children
    /// `kids` (in [`Expr::for_each_child`] order). See the module docs.
    #[inline(never)]
    pub(super) fn window(
        &mut self,
        e: &Arc<Expr>,
        w: &WindowExpr,
        kids: &[Bound],
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let spec = match &w.window {
            WindowSpecRef::Named(name) => {
                return Err(AnalysisError::new(
                    MissingWindowSpecification {
                        window_name: name.clone(),
                        docroot: SPARK_DOC_ROOT.to_owned(),
                    },
                    e.origin,
                ));
            }
            WindowSpecRef::Spec(spec) => spec,
        };
        let shown_of = |exprs: Vec<Arc<Expr>>| -> WindowSpec {
            let mut it = exprs.into_iter();
            let mut s = WindowSpec {
                partition_by: Vec::new(),
                order_by: Vec::new(),
                frame: spec.frame.clone(),
            };
            for _ in &spec.partition_by {
                s.partition_by.extend(it.next());
            }
            for o in &spec.order_by {
                if let Some(expr) = it.next() {
                    s.order_by.push(SortOrder { expr, ..o.clone() });
                }
            }
            if let Some(f) = &mut s.frame {
                for b in [&mut f.lower, &mut f.upper] {
                    if let FrameBound::Offset(x) = b
                        && let Some(y) = it.next()
                    {
                        *x = y;
                    }
                }
            }
            s
        };
        let (function, rest) = kids
            .split_first()
            .ok_or_else(|| unsupported("a window without a function", e.origin))?;
        if let Some(id) = kids.iter().flat_map(|k| k.lateral.iter()).next() {
            let lca = cx
                .lateral
                .and_then(|l| l.iter().find(|a| a.col.id == *id))
                .map_or_else(String::new, |a| a.col.name.clone());
            let shown = shown_of(rest.iter().map(Bound::shown).collect());
            let ids: Vec<crate::AttrId> = kids.iter().flat_map(|k| k.lateral.clone()).collect();
            let text = pretty(
                &super::ops::lateral_refs(
                    &Arc::new(Expr::new(ExprNode::Window(WindowExpr {
                        function: function.shown(),
                        window: WindowSpecRef::Spec(shown),
                    }))),
                    &ids,
                ),
                None,
            );
            return Err(AnalysisError::new(
                UnsupportedFeatureLateralColumnAliasInWindow {
                    lca: format!("`{lca}`"),
                    window_expr: format!("\"{text}\""),
                },
                e.origin,
            ));
        }
        let call = match &function.expr.node {
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if matches!(f.kind, FunctionKind::Window | FunctionKind::Aggregate) =>
            {
                f.clone()
            }
            _ => {
                return Err(AnalysisError::new(
                    UnsupportedExprForWindow {
                        sql_expr: format!("\"{}\"", pretty(&function.shown(), None)),
                    },
                    e.origin,
                ));
            }
        };
        // A window function (not an aggregate) takes no DISTINCT: Spark's
        // function lookup refuses it.
        if call.kind == FunctionKind::Window && call.distinct {
            return Err(AnalysisError::new(
                InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                    pretty_name: format!("`{}`", call.id.name()),
                    syntax: "DISTINCT".to_owned(),
                },
                e.origin,
            ));
        }
        let mut bound = shown_of(rest.iter().map(|k| Arc::clone(&k.expr)).collect());
        let mut shown = shown_of(rest.iter().map(Bound::shown).collect());
        let name = call.id.name();
        // A window function's own arguments: Spark replaces a rank's by the
        // ordering, `ntile()` is `ntile(1)`.
        let (mut function_expr, function_shown) = if call.kind == FunctionKind::Window {
            let args = if RANK_LIKE.contains(&name) {
                Some(Vec::new())
            } else if name == "ntile" && call.args.is_empty() {
                Some(vec![Arc::new(Expr::literal(Literal::Integer(1)))])
            } else {
                None
            };
            match args {
                Some(args) => {
                    let f = Arc::new(
                        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                            ResolvedFunction {
                                args,
                                ..call.clone()
                            },
                        )))
                        .with_origin(function.expr.origin),
                    );
                    (Arc::clone(&f), f)
                }
                None => (Arc::clone(&function.expr), function.shown()),
            }
        } else {
            (Arc::clone(&function.expr), function.shown())
        };
        if call.kind == FunctionKind::Window {
            if bound.order_by.is_empty() {
                return Err(AnalysisError::new(
                    WindowFunctionFrameNotOrdered {
                        wf_name: name.to_owned(),
                        wf_expr: window_function_sql(
                            name,
                            &super::ops::qualified(&function_shown, cx.scope),
                        ),
                    },
                    e.origin,
                ));
            }
            if matches!(name, "lag" | "lead") {
                let frame = self.offset_frame(&call, &function_shown, bound.frame.as_ref(), e)?;
                function_expr = frame.1;
                bound.frame = Some(frame.0.clone());
                shown.frame = Some(frame.0);
            }
            // `nth_value`'s offset (checked by its typing) as an `int`
            // literal for the planner.
            let offset = call
                .args
                .get(1)
                .and_then(|o| self.fold(o, self.ctx.conf.ansi));
            let offset = match offset {
                Some(Constant::Integral(v) | Constant::Fractional(v)) => Some(v),
                Some(Constant::String(t)) => t.trim().parse::<i64>().ok(),
                _ => None,
            };
            if name == "nth_value"
                && let Some(v) = offset
                && let Ok(v) = i32::try_from(v)
            {
                let mut args = call.args.clone();
                args[1] = Arc::new(Expr::literal(Literal::Integer(v)));
                function_expr = Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                        ResolvedFunction {
                            args,
                            ..call.clone()
                        },
                    )))
                    .with_origin(function.expr.origin),
                );
            }
        }
        if call.kind == FunctionKind::Window && !matches!(name, "lag" | "lead" | "nth_value") {
            let required = WindowFrame::running(if name == "cume_dist" {
                FrameUnits::Range
            } else {
                FrameUnits::Rows
            });
            match &bound.frame {
                Some(f) if *f != required => {
                    return Err(AnalysisError::new(
                        LegacyErrorTemp1036 {
                            wf: frame_string(f),
                            required: frame_string(&required),
                        },
                        e.origin,
                    ));
                }
                Some(_) => {}
                None => {
                    bound.frame = Some(required.clone());
                    shown.frame = Some(required);
                }
            }
        } else if bound.frame.is_none() {
            let default = if bound.order_by.is_empty() {
                WindowFrame::unbounded(FrameUnits::Rows)
            } else {
                WindowFrame::running(FrameUnits::Range)
            };
            bound.frame = Some(default.clone());
            shown.frame = Some(default);
        }
        let frame = bound
            .frame
            .clone()
            .ok_or_else(|| unsupported("a window without a frame", e.origin))?;
        let invalid = matches!(frame.lower, FrameBound::UnboundedFollowing)
            || matches!(frame.upper, FrameBound::UnboundedPreceding);
        if invalid {
            let shown_frame = shown.frame.as_ref().unwrap_or(&frame);
            let text = window_text(
                &function_shown,
                &WindowSpec {
                    partition_by: Vec::new(),
                    order_by: Vec::new(),
                    frame: Some(shown_frame.clone()),
                },
            );
            // The frame's own text: what follows `OVER (` in the window's.
            let frame_sql = text
                .split_once(" OVER (")
                .map_or(String::new(), |(_, f)| f.trim_end_matches(')').to_owned());
            return Err(AnalysisError::new(
                DatatypeMismatchSpecifiedWindowFrameInvalidBound {
                    sql_expr: format!("\"{frame_sql}\""),
                    lower: bound_sql(&shown_frame.lower),
                    upper: bound_sql(&shown_frame.upper),
                },
                e.origin,
            ));
        }
        if !matches!(name, "lag" | "lead") {
            self.frame_bounds(
                &mut bound,
                &mut shown,
                rest,
                spec,
                &function_shown,
                e.origin,
            )?;
        }
        let unbounded = matches!(
            (&frame.lower, &frame.upper),
            (
                FrameBound::UnboundedPreceding,
                FrameBound::UnboundedFollowing
            )
        );
        if frame.units == FrameUnits::Range && bound.order_by.is_empty() && !unbounded {
            let text = window_text(&function_shown, &shown);
            let spec_sql = text
                .split_once(" OVER ")
                .map_or(String::new(), |(_, s)| s.to_owned());
            return Err(AnalysisError::new(
                DatatypeMismatchRangeFrameWithoutOrder {
                    sql_expr: format!("\"{spec_sql}\""),
                },
                e.origin,
            ));
        }
        if call.distinct {
            return Err(AnalysisError::new(
                DistinctWindowFunctionUnsupported {
                    window_expr: format!("\"{}\"", window_text(&function_shown, &shown)),
                },
                e.origin,
            ));
        }
        // `listagg(...) WITHIN GROUP (ORDER BY ...) OVER (ORDER BY ...)`
        // is unsupported (Spark only allows the whole-partition frame
        // without an ordering), and so are `median` and the percentiles
        // over an ordered or framed window; the error quotes the
        // aggregate with its ordering as further arguments, as Spark's
        // `ListAgg.sql` does (the percentiles show the ordered value
        // first, then the percentage).
        let name = call.id.name();
        let whole_only = match name {
            "listagg" => !call.within_group.is_empty(),
            "median" | "percentile_cont" | "percentile_disc" => true,
            _ => false,
        };
        if whole_only
            && (!bound.order_by.is_empty()
                || frame.units != FrameUnits::Rows
                || !matches!(frame.lower, FrameBound::UnboundedPreceding)
                || !matches!(frame.upper, FrameBound::UnboundedFollowing))
        {
            let zone = self.ctx.conf.session_zone;
            let mut parts = Vec::new();
            let mut orders = Vec::new();
            // The shown call is resolved or still a plain call, by the
            // function.
            let shown_call = match &function.shown().node {
                ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                    Some((f.args.clone(), f.within_group.clone()))
                }
                ExprNode::Function(f) => Some((f.args.clone(), f.within_group.clone())),
                _ => None,
            };
            if let Some((shown_args, shown_group)) = &shown_call {
                let order_text = |o: &crate::SortOrder| {
                    format!(
                        "{}{}",
                        pretty(&o.expr, zone),
                        crate::display::order_suffix(o)
                    )
                };
                if name == "listagg" {
                    parts.extend(shown_args.iter().map(|a| pretty(a, zone)));
                    orders.extend(shown_group.iter().map(order_text));
                } else {
                    parts.extend(shown_group.iter().map(|o| pretty(&o.expr, zone)));
                    orders.extend(shown_args.iter().map(|a| pretty(a, zone)));
                }
            }
            parts.extend(orders);
            return Err(AnalysisError::new(
                InvalidWindowSpecForAggregationFunc {
                    agg_func: format!("\"{name}({})\"", parts.join(", ")),
                },
                e.origin,
            ));
        }
        let rebuilt = |function: Arc<Expr>, spec: WindowSpec| {
            Arc::new(
                Expr::new(ExprNode::Window(WindowExpr {
                    function,
                    window: WindowSpecRef::Spec(spec),
                }))
                .with_origin(e.origin),
            )
        };
        let shown_expr = rebuilt(Arc::clone(&function_shown), shown);
        let expr = rebuilt(function_expr, bound);
        let outer = kids.iter().filter_map(|k| k.outer).min();
        let b = Bound {
            expr,
            ty: ArgType {
                data_type: Some(call.data_type.clone()),
                nullable: call.nullable,
            },
            agg: kids.iter().find_map(|k| k.agg),
            win: Some(e.origin),
            outer,
            lateral: Vec::new(),
            shown: Some(shown_expr),
        };
        if cx.windows != WindowUse::Allowed {
            return Err(misplaced(cx.windows, &b, e.origin));
        }
        Ok(b)
    }

    /// `lag`/`lead`'s frame, `ROWS BETWEEN k FOLLOWING AND k FOLLOWING` (`k`
    /// the offset, negated for `lag`), and its call with the default cast to
    /// the input's type. Spark's checks: no frame of the user's
    /// (`_LEGACY_ERROR_TEMP_1035`), the offset foldable
    /// (`DATATYPE_MISMATCH.NON_FOLDABLE_INPUT`, shown as the frame bound)
    /// and an `INT` (a string fails the call, `FAILED_FUNCTION_CALL`; any
    /// other type, or NULL, makes a frame Spark refuses as the user's).
    fn offset_frame(
        &self,
        call: &ResolvedFunction,
        shown: &Arc<Expr>,
        framed: Option<&WindowFrame>,
        e: &Arc<Expr>,
    ) -> Result<(WindowFrame, Arc<Expr>), AnalysisError> {
        let name = call.id.name();
        let refuse = || {
            AnalysisError::new(
                LegacyErrorTemp1035 {
                    pretty_name: name.to_owned(),
                },
                e.origin,
            )
        };
        // A frame of the user's is refused; the frame this made (an
        // analyzed plan analyzed again) is kept.
        if framed.is_some_and(|f| !is_offset_frame(f)) {
            return Err(refuse());
        }
        let [input, offset, default, rest @ ..] = &call.args[..] else {
            return Err(unsupported(
                "lag and lead without their offset and default",
                e.origin,
            ));
        };
        let value = match self.fold(offset, self.ctx.conf.ansi) {
            None => {
                let text = pretty(offset, self.ctx.conf.session_zone);
                return Err(AnalysisError::new(
                    DatatypeMismatchNonFoldableInput {
                        sql_expr: format!("\"{}\"", pretty(shown, self.ctx.conf.session_zone)),
                        input_name: "`offset`".to_owned(),
                        input_type: "\"INT\"".to_owned(),
                        input_expr: if name == "lag" {
                            format!("\"(- {text})\"")
                        } else {
                            format!("\"{text}\"")
                        },
                    },
                    e.origin,
                ));
            }
            Some(_) if matches!(offset.node, ExprNode::Literal(Literal::String(_))) => {
                return Err(AnalysisError::new(
                    FailedFunctionCall {
                        func_name: format!("`{name}`"),
                    },
                    e.origin,
                ));
            }
            Some(Constant::Integral(v)) if offset_type(offset) => v,
            Some(Constant::Error(err)) => return Err(AnalysisError::new(err, e.origin)),
            Some(_) => return Err(refuse()),
        };
        let k = if name == "lag" {
            value.checked_neg()
        } else {
            Some(value)
        };
        let k = k.and_then(|k| i32::try_from(k).ok()).ok_or_else(refuse)?;
        let bound = || FrameBound::Offset(Arc::new(Expr::literal(Literal::Integer(k))));
        let frame = WindowFrame {
            units: FrameUnits::Rows,
            lower: bound(),
            upper: bound(),
        };
        if framed.is_some_and(|f| *f != frame) {
            return Err(refuse());
        }
        // The default as a value of the input's type (an implicit cast).
        let mut args = vec![Arc::clone(input), Arc::clone(offset)];
        let to = call.data_type.clone();
        let default = if is_null_literal(default) || to == krishiv_types::DataType::Null {
            Arc::new(Expr::literal(Literal::Null(Some(to))))
        } else {
            Arc::new(
                Expr::new(ExprNode::Cast {
                    expr: Arc::clone(default),
                    to,
                    mode: if self.ctx.conf.ansi {
                        crate::CastMode::Ansi
                    } else {
                        crate::CastMode::Legacy
                    },
                })
                .with_origin(default.origin),
            )
        };
        args.push(default);
        args.extend(rest.iter().cloned());
        let f = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    args,
                    ..call.clone()
                },
            )))
            .with_origin(e.origin),
        );
        Ok((frame, f))
    }

    /// Spark's checks of a frame's value bounds (`SpecifiedWindowFrame`, in
    /// its order): each bound an `INT`
    /// (`SPECIFIED_WINDOW_FRAME_UNACCEPTED_TYPE`), the lower not above the
    /// upper (`SPECIFIED_WINDOW_FRAME_WRONG_COMPARISON`). The bound
    /// window's offsets become signed `int` literals (negative: rows
    /// before). `RANGE` frames with value bounds are not supported yet.
    fn frame_bounds(
        &self,
        bound: &mut WindowSpec,
        shown: &mut WindowSpec,
        rest: &[Bound],
        spec: &WindowSpec,
        function: &Arc<Expr>,
        origin: Origin,
    ) -> Result<(), AnalysisError> {
        let Some(frame) = bound.frame.as_mut() else {
            return Ok(());
        };
        let offset = |b: &FrameBound| matches!(b, FrameBound::Offset(_));
        if !offset(&frame.lower) && !offset(&frame.upper) {
            return Ok(());
        }
        if frame.units == FrameUnits::Range {
            return self.range_frame_bounds(frame, shown, rest, spec, function, origin);
        }
        let shown_frame = shown.frame.clone().unwrap_or_else(|| frame.clone());
        let text = window_text(
            function,
            &WindowSpec {
                partition_by: Vec::new(),
                order_by: Vec::new(),
                frame: Some(shown_frame),
            },
        );
        let sql_expr = format!(
            "\"{}\"",
            text.split_once(" OVER (")
                .map_or(String::new(), |(_, f)| f.trim_end_matches(')').to_owned())
        );
        // The bounds' expressions follow the partitioning and ordering.
        let mut kids = rest
            .iter()
            .skip(spec.partition_by.len() + spec.order_by.len());
        let mut values = [None, None];
        for (k, (b, location)) in [(&frame.lower, "lower"), (&frame.upper, "upper")]
            .into_iter()
            .enumerate()
        {
            let FrameBound::Offset(x) = b else { continue };
            let ty = kids.next().and_then(|kid| kid.ty.data_type.clone());
            if ty != Some(krishiv_types::DataType::Integer) {
                return Err(AnalysisError::new(
                    DatatypeMismatchSpecifiedWindowFrameUnacceptedType {
                        sql_expr,
                        location: location.to_owned(),
                        expr_type: ty
                            .map_or_else(|| "\"VOID\"".to_owned(), |t| format!("\"{}\"", t.sql())),
                        expected_type: "\"INT\"".to_owned(),
                    },
                    origin,
                ));
            }
            values[k] = Some(x);
        }
        let mut ints = [None, None];
        for (k, x) in values.iter().enumerate() {
            let Some(x) = x else { continue };
            ints[k] = Some(
                match offset_value(x, &|c| self.fold(c, self.ctx.conf.ansi)) {
                    Some(Constant::Integral(v)) => v,
                    Some(Constant::Error(e)) => return Err(AnalysisError::new(e, origin)),
                    Some(Constant::Null) => {
                        return Err(unsupported("a NULL window frame bound", origin));
                    }
                    _ => {
                        return Err(unsupported(
                            "window frame bounds other than literals and casts of literals",
                            origin,
                        ));
                    }
                },
            );
        }
        let at = |b: &FrameBound, v: Option<i64>| match b {
            FrameBound::CurrentRow => Some(0),
            _ => v,
        };
        if let (Some(l), Some(u)) = (at(&frame.lower, ints[0]), at(&frame.upper, ints[1]))
            && offset(&frame.lower)
            && offset(&frame.upper)
            && l > u
        {
            return Err(AnalysisError::new(
                DatatypeMismatchSpecifiedWindowFrameWrongComparison {
                    sql_expr,
                    comparison: "less than or equal".to_owned(),
                },
                origin,
            ));
        }
        for (b, v) in [(&mut frame.lower, ints[0]), (&mut frame.upper, ints[1])] {
            if let Some(v) = v.and_then(|v| i32::try_from(v).ok()) {
                *b = FrameBound::Offset(Arc::new(Expr::literal(Literal::Integer(v))));
            }
        }
        Ok(())
    }

    /// The value bounds of a `RANGE` frame, checked as Spark does
    /// (`WindowSpecDefinition`, then `SpecifiedWindowFrame`): one ordering
    /// key (`RANGE_FRAME_MULTI_ORDER`) of a type the bounds are taken over
    /// (`RANGE_FRAME_INVALID_TYPE`; a bound is first cast to a numeric,
    /// string or boolean key's type), numeric bounds
    /// (`SPECIFIED_WINDOW_FRAME_UNACCEPTED_TYPE`), the lower not above the
    /// upper. The bounds become signed literals of the key's type (an `int`
    /// for a date). Interval bounds (`DATE`, `TIMESTAMP` keys) are not
    /// supported yet.
    fn range_frame_bounds(
        &self,
        frame: &mut WindowFrame,
        shown: &mut WindowSpec,
        rest: &[Bound],
        spec: &WindowSpec,
        function: &Arc<Expr>,
        origin: Origin,
    ) -> Result<(), AnalysisError> {
        use krishiv_types::DataType as T;
        // A frame without an ordering is `RANGE_FRAME_WITHOUT_ORDER`,
        // raised by the caller.
        if spec.order_by.is_empty() {
            return Ok(());
        }
        let quoted = |t: &T| format!("\"{}\"", t.sql());
        let spec_sql = |shown: &WindowSpec| {
            format!(
                "\"{}\"",
                window_text(function, shown)
                    .split_once(" OVER ")
                    .map_or(String::new(), |(_, s)| s.to_owned())
            )
        };
        let frame_sql = |shown: &WindowSpec| {
            let text = window_text(
                function,
                &WindowSpec {
                    partition_by: Vec::new(),
                    order_by: Vec::new(),
                    frame: Some(shown.frame.clone().unwrap_or_else(|| frame.clone())),
                },
            );
            format!(
                "\"{}\"",
                text.split_once(" OVER (")
                    .map_or(String::new(), |(_, f)| f.trim_end_matches(')').to_owned())
            )
        };
        let keys = &rest[spec.partition_by.len().min(rest.len())..];
        if spec.order_by.len() > 1 {
            let order_spec = spec
                .order_by
                .iter()
                .zip(keys)
                .map(|(o, k)| {
                    let name = match &k.expr.node {
                        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                            format!("{}#{}", c.name, c.id.0)
                        }
                        _ => pretty(&k.expr, self.ctx.conf.session_zone),
                    };
                    format!("{name}{}", crate::display::order_suffix(o))
                })
                .collect::<Vec<_>>()
                .join(",");
            return Err(AnalysisError::new(
                DatatypeMismatchRangeFrameMultiOrder {
                    sql_expr: spec_sql(shown),
                    order_spec,
                },
                origin,
            ));
        }
        let Some(key) = keys.first().and_then(|k| k.ty.data_type.clone()) else {
            return Err(unsupported("a RANGE frame over an untyped key", origin));
        };
        let mut kids = keys.iter().skip(spec.order_by.len());
        let mut values: [Option<RangeValue>; 2] = [None, None];
        for (k, (b, location)) in [(&frame.lower, "lower"), (&frame.upper, "upper")]
            .into_iter()
            .enumerate()
        {
            let FrameBound::Offset(x) = b else { continue };
            let bt = kids.next().and_then(|kid| kid.ty.data_type.clone());
            let Some(bt) = bt else {
                return Err(unsupported("a NULL window frame bound", origin));
            };
            let numeric = |t: &T| {
                matches!(
                    t,
                    T::Byte
                        | T::Short
                        | T::Integer
                        | T::Long
                        | T::Float
                        | T::Double
                        | T::Decimal(_)
                )
            };
            match &key {
                t if numeric(t) => {
                    if !numeric(&bt) {
                        return Err(unsupported(
                            "a non-numeric RANGE window frame bound",
                            origin,
                        ));
                    }
                }
                T::String | T::Boolean => {
                    // The bound is cast to the key's type, which is not a
                    // numeric one.
                    return Err(AnalysisError::new(
                        DatatypeMismatchSpecifiedWindowFrameUnacceptedType {
                            sql_expr: frame_sql(shown),
                            location: location.to_owned(),
                            expr_type: quoted(&key),
                            expected_type: RANGE_BOUND_TYPES.to_owned(),
                        },
                        origin,
                    ));
                }
                T::Date | T::Timestamp | T::TimestampNtz
                    if !matches!(bt, T::Integer) || key != T::Date =>
                {
                    if !numeric(&bt) {
                        return Err(unsupported("interval RANGE window frame bounds", origin));
                    }
                    return Err(AnalysisError::new(
                        DatatypeMismatchRangeFrameInvalidType {
                            sql_expr: spec_sql(shown),
                            order_spec_type: quoted(&key),
                            value_boundary_type: quoted(&bt),
                        },
                        origin,
                    ));
                }
                T::Date => {}
                _ => {
                    return Err(AnalysisError::new(
                        DatatypeMismatchRangeFrameInvalidType {
                            sql_expr: spec_sql(shown),
                            order_spec_type: quoted(&key),
                            value_boundary_type: quoted(&bt),
                        },
                        origin,
                    ));
                }
            }
            values[k] = Some(range_value(x, &key).ok_or_else(|| {
                unsupported("RANGE window frame bounds other than literals", origin)
            })?);
        }
        // A bound cast to the key's type loses its `PRECEDING` in Spark's
        // text: `(- n) FOLLOWING`.
        if let Some(sf) = shown.frame.as_mut() {
            let mut kids = keys.iter().skip(spec.order_by.len());
            for b in [&mut sf.lower, &mut sf.upper] {
                let FrameBound::Offset(x) = b else { continue };
                let cast = kids
                    .next()
                    .is_some_and(|kid| kid.ty.data_type.as_ref() != Some(&key));
                if cast && let Some(n) = crate::expr::negated(x) {
                    *x = Arc::new(Expr::function("-", vec![(**n).clone()]));
                }
            }
        }
        let at = |b: &FrameBound, v: &Option<RangeValue>| match (b, v) {
            (FrameBound::CurrentRow, _) => Some(RangeValue::Int(0)),
            (_, v) => v.clone(),
        };
        if matches!(frame.lower, FrameBound::Offset(_))
            && matches!(frame.upper, FrameBound::Offset(_))
            && let (Some(l), Some(u)) = (at(&frame.lower, &values[0]), at(&frame.upper, &values[1]))
            && l.above(&u)
        {
            return Err(AnalysisError::new(
                DatatypeMismatchSpecifiedWindowFrameWrongComparison {
                    sql_expr: frame_sql(shown),
                    comparison: "less than or equal".to_owned(),
                },
                origin,
            ));
        }
        for (b, v) in [
            (&mut frame.lower, &values[0]),
            (&mut frame.upper, &values[1]),
        ] {
            if let Some(v) = v {
                let literal = v.literal(&key).ok_or_else(|| {
                    unsupported("RANGE window frame bounds past the key's type", origin)
                })?;
                *b = FrameBound::Offset(Arc::new(Expr::literal(literal)));
            }
        }
        Ok(())
    }

    /// The windows `exprs` (expressions over `base`) hold, computed over
    /// `base`: the relation computing them (`base`'s attributes, then
    /// helper and window columns) and `exprs` reading their columns
    /// instead. `base` itself when `exprs` hold no window.
    #[inline(never)]
    pub(super) fn windowed(
        &mut self,
        base: &Rel,
        exprs: &[Arc<Expr>],
    ) -> Result<(Rel, Vec<Arc<Expr>>), AnalysisError> {
        let mut memo = HashMap::new();
        // The windows, each once.
        let mut windows: Vec<(u128, Arc<Expr>)> = Vec::new();
        for e in exprs {
            let mut stack = vec![Arc::clone(e)];
            while let Some(x) = stack.pop() {
                if let ExprNode::Window(_) = &x.node {
                    let k = semantic_key(&x, &mut memo);
                    if !windows.iter().any(|(kk, _)| *kk == k) {
                        windows.push((k, Arc::clone(&x)));
                    }
                    continue;
                }
                let mut kids = Vec::new();
                x.for_each_child(&mut |c| kids.push(Arc::clone(c)));
                stack.extend(kids.into_iter().rev());
            }
        }
        if windows.is_empty() {
            return Ok((base.clone(), exprs.to_vec()));
        }
        // Inputs that are neither columns of `base` nor literals: computed
        // below, as `_w0`, `_w1`, ...
        let mut extracted: Vec<(u128, Arc<Expr>, Attr)> = Vec::new();
        let mut rewritten: Vec<(u128, WindowExpr)> = Vec::new();
        for (k, x) in &windows {
            let ExprNode::Window(w) = &x.node else {
                continue;
            };
            let WindowSpecRef::Spec(spec) = &w.window else {
                return Err(unsupported("an unresolved window", x.origin));
            };
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &w.function.node else {
                return Err(unsupported("a window over an expression", x.origin));
            };
            let mut input = |e: &Arc<Expr>, this: &mut Self| -> Arc<Expr> {
                this.window_input(e, base, &mut extracted, &mut memo)
            };
            let args = f.args.iter().map(|a| input(a, self)).collect();
            let filter = f.filter.as_ref().map(|a| input(a, self));
            let function = Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        args,
                        filter,
                        ..f.clone()
                    },
                )))
                .with_origin(w.function.origin),
            );
            let partition_by = spec.partition_by.iter().map(|p| input(p, self)).collect();
            let order_by = spec
                .order_by
                .iter()
                .map(|o| SortOrder {
                    expr: input(&o.expr, self),
                    ..o.clone()
                })
                .collect();
            rewritten.push((
                *k,
                WindowExpr {
                    function,
                    window: WindowSpecRef::Spec(WindowSpec {
                        partition_by,
                        order_by,
                        frame: spec.frame.clone(),
                    }),
                },
            ));
        }
        // The projection below, its own windows computed first.
        let mut cur = base.clone();
        if !extracted.is_empty() {
            let aliases: Vec<Arc<Expr>> = extracted
                .iter()
                .map(|(_, e, a)| {
                    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: Arc::clone(e),
                        output: a.col.clone(),
                    })))
                })
                .collect();
            let (inner, aliases) = self.windowed(base, &aliases)?;
            let mut items: Vec<Arc<Expr>> = base
                .out
                .iter()
                .map(|a| col_expr(&a.col, Origin::UNKNOWN))
                .collect();
            items.extend(aliases);
            let mut out = base.out.to_vec();
            out.extend(extracted.iter().map(|(_, _, a)| a.clone()));
            cur = self.projection(Origin::UNKNOWN, &inner, items, out);
        }
        // One window relation per partitioning and ordering.
        let mut groups: Vec<(u128, Vec<usize>)> = Vec::new();
        for (i, (_, w)) in rewritten.iter().enumerate() {
            let WindowSpecRef::Spec(spec) = &w.window else {
                continue;
            };
            let key = spec_key(spec, &mut memo);
            match groups.iter_mut().find(|(g, _)| *g == key) {
                Some((_, members)) => members.push(i),
                None => groups.push((key, vec![i])),
            }
        }
        let mut columns: Vec<(u128, Attr)> = Vec::new();
        for (_, members) in &groups {
            let mut window_exprs = Vec::new();
            let mut out = cur.out.to_vec();
            let mut partition_by = Vec::new();
            let mut order_by = Vec::new();
            for &i in members {
                let (k, w) = &rewritten[i];
                if let WindowSpecRef::Spec(spec) = &w.window {
                    partition_by.clone_from(&spec.partition_by);
                    order_by.clone_from(&spec.order_by);
                }
                let ty = window_type(w);
                let name = format!("_we{}", columns.len());
                let output = self.attr(&name, &ty);
                window_exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::Alias {
                        expr: Arc::new(Expr::new(ExprNode::Window(w.clone()))),
                        output: output.clone(),
                    },
                ))));
                let a = Attr {
                    col: output,
                    qual: super::ops::no_qual(),
                    source: None,
                };
                out.push(a.clone());
                columns.push((*k, a));
            }
            let node = PlanNode::Window {
                input: Arc::clone(&cur.plan),
                exprs: window_exprs,
                partition_by,
                order_by,
            };
            // No widening through a window: the select list above it sees
            // every column of its input already.
            cur = self.make(Origin::UNKNOWN, node, out, &[&cur]);
        }
        // The select list over the windows' columns.
        let mut out = Vec::with_capacity(exprs.len());
        for e in exprs {
            let replaced = Arc::clone(e).transform_down(&mut |x: Arc<Expr>| {
                if let ExprNode::Window(_) = &x.node {
                    let k = semantic_key(&x, &mut memo);
                    if let Some((_, a)) = columns.iter().find(|(kk, _)| *kk == k) {
                        return Ok::<_, AnalysisError>(col_expr(&a.col, x.origin));
                    }
                }
                Ok(x)
            })?;
            out.push(replaced);
        }
        Ok((cur, out))
    }

    /// One input of a window over `base`: a column of `base` or a literal
    /// as it is, anything else a column of the projection below (added to
    /// `extracted` once per semantic key).
    fn window_input(
        &mut self,
        e: &Arc<Expr>,
        base: &Rel,
        extracted: &mut Vec<(u128, Arc<Expr>, Attr)>,
        memo: &mut HashMap<usize, u128>,
    ) -> Arc<Expr> {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Column(c))
                if base.out.iter().any(|a| a.col.id == c.id) =>
            {
                return Arc::clone(e);
            }
            ExprNode::Literal(_) | ExprNode::Resolved(ResolvedExpr::OuterColumn(_)) => {
                return Arc::clone(e);
            }
            _ => {}
        }
        let k = semantic_key(e, memo);
        if let Some((_, _, a)) = extracted.iter().find(|(kk, _, _)| *kk == k) {
            return col_expr(&a.col, e.origin);
        }
        let ty = expr_type(e);
        let output = self.attr(&format!("_w{}", extracted.len()), &ty);
        let a = Attr {
            col: output,
            qual: super::ops::no_qual(),
            source: None,
        };
        extracted.push((k, Arc::clone(e), a.clone()));
        col_expr(&a.col, e.origin)
    }
}

/// A key equal for windows with the same partitioning and ordering.
fn spec_key(spec: &WindowSpec, memo: &mut HashMap<usize, u128>) -> u128 {
    let mut b = Vec::new();
    for p in &spec.partition_by {
        b.extend_from_slice(&semantic_key(p, memo).to_le_bytes());
    }
    b.push(0xff);
    for o in &spec.order_by {
        b.extend_from_slice(&semantic_key(&o.expr, memo).to_le_bytes());
        b.push(match (o.direction, o.nulls) {
            (crate::SortDirection::Ascending, crate::NullOrdering::NullsFirst) => 0,
            (crate::SortDirection::Ascending, crate::NullOrdering::NullsLast) => 1,
            (crate::SortDirection::Descending, crate::NullOrdering::NullsFirst) => 2,
            (crate::SortDirection::Descending, crate::NullOrdering::NullsLast) => 3,
        });
    }
    twox_hash::XxHash3_128::oneshot(&b)
}

/// The result type of a resolved window.
fn window_type(w: &WindowExpr) -> ArgType {
    match &w.function.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => ArgType {
            data_type: Some(f.data_type.clone()),
            nullable: f.nullable,
        },
        _ => ArgType {
            data_type: None,
            nullable: true,
        },
    }
}

/// The type of a resolved expression, from its node; a cast's result is
/// taken as nullable (a helper column's nullability only needs to be safe).
fn expr_type(e: &Expr) -> ArgType {
    let (data_type, nullable) = super::subquery::typed(e);
    ArgType {
        data_type,
        nullable: nullable || matches!(e.node, ExprNode::Cast { .. }),
    }
}
