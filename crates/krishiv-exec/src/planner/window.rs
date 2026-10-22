// SPDX-License-Identifier: Apache-2.0
//! Window relations: a projection computing the keys and arguments that
//! are not columns, a sort by the partitioning keys then the ordering
//! keys, and the [`Window`](crate::window::Window) operator over it (see
//! [`crate::window`]).

use std::sync::Arc;

use krishiv_plan::{
    Expr, ExprNode, FrameBound, FrameUnits, FunctionKind, NullOrdering, ResolvedExpr,
    SortDirection, SortOrder, WindowSpecRef,
};

use super::aggregate::{non_null_literal, picks};
use super::build::{Planned, Planner, internal, list, orders, unsupported};
use crate::aggregate::{AggInput, AggregateExpr, AggregateFunction};
use crate::error::Result;
use crate::operator::Source;
use crate::operators::SortKey;
use crate::physical::PhysNode;
use crate::window::{
    FrameBound as Bound, FrameUnits as Units, Window, WindowFrame, WindowFunction,
    WindowFunctionPlan, WindowPlan,
};

/// One window expression, read off the plan: its function call, frame and
/// output attribute, and the expressions it reads (arguments, filter).
struct Call<'a> {
    f: &'a krishiv_plan::ResolvedFunction,
    frame: WindowFrame,
    output: &'a krishiv_plan::BoundColumn,
}

impl Planner<'_> {
    pub(crate) fn window(
        &mut self,
        input: Planned,
        exprs: &[Arc<Expr>],
        partition_by: &[Arc<Expr>],
        order_by: &[SortOrder],
    ) -> Result<Planned> {
        let calls = exprs.iter().map(|e| call(e)).collect::<Result<Vec<_>>>()?;
        // The columns the window reads: partitioning, ordering, then each
        // function's arguments and filter.
        let mut wanted: Vec<Arc<Expr>> = partition_by.to_vec();
        wanted.extend(order_by.iter().map(|o| Arc::clone(&o.expr)));
        // Without keys, a constant one: the sort then gathers every row on
        // one driver (a single-partition exchange once the exchange pass,
        // P2-T174, provides one).
        let gather = partition_by.is_empty() && order_by.is_empty();
        if gather {
            wanted.push(Arc::new(Expr::literal(krishiv_plan::Literal::Integer(0))));
        }
        let mut arg_at = Vec::with_capacity(calls.len());
        for c in &calls {
            let args = arguments(c.f);
            let start = wanted.len();
            wanted.extend(args.iter().map(|a| Arc::clone(a)));
            wanted.extend(c.f.filter.iter().cloned());
            arg_at.push((start, args.len()));
        }
        let (input, columns) = self.with_columns(input, &wanted)?;
        let np = partition_by.len();
        let no = order_by.len();
        let mut keys: Vec<SortKey> = columns[..np].iter().map(|&c| SortKey::asc(c)).collect();
        let order_keys: Vec<SortKey> = order_by
            .iter()
            .zip(&columns[np..np + no])
            .map(|(o, &c)| {
                let k = match o.direction {
                    SortDirection::Ascending => SortKey::asc(c),
                    SortDirection::Descending => SortKey::desc(c),
                };
                k.nulls_first(o.nulls == NullOrdering::NullsFirst)
            })
            .collect();
        keys.extend(order_keys.iter().copied());
        if gather {
            keys.push(SortKey::asc(columns[np + no]));
        }
        let mut functions = Vec::with_capacity(calls.len());
        for (c, &(start, n)) in calls.iter().zip(&arg_at) {
            let args = &columns[start..start + n];
            let filter = c.f.filter.as_ref().map(|_| columns[start + n]);
            let function = self.function(c, args, &input)?;
            let plan = WindowFunctionPlan {
                function,
                frame: c.frame,
                filter,
                name: c.output.name.clone(),
            };
            let (t, _) = plan.output()?;
            let want = c.output.data_type.to_arrow();
            if t != want {
                return Err(internal(&format!(
                    "{} computes {t} where the analyzer typed {want}",
                    c.output.name
                )));
            }
            functions.push(plan);
        }
        let window = WindowPlan {
            partition_by: columns[..np].to_vec(),
            order_by: order_keys,
            functions,
        };
        // The sort, then the window over it.
        let sort_schema = crate::physical::sort_op(&input.schema, &keys, None)?.schema();
        let layout = input.layout.clone();
        // The sort's keys as a sort shows them: partitioning keys ascending,
        // then the ordering.
        let mut sorted_by: Vec<SortOrder> = partition_by
            .iter()
            .map(|e| SortOrder::asc(Arc::clone(e)))
            .collect();
        sorted_by.extend(order_by.iter().cloned());
        if gather {
            sorted_by.push(SortOrder::asc(Expr::literal(
                krishiv_plan::Literal::Integer(0),
            )));
        }
        let sort_detail = orders(&sorted_by);
        let sorted = input.wrap(
            |input| PhysNode::Sort { input, keys },
            sort_schema,
            layout,
            ("Sort", sort_detail),
        );
        let op = Window::new(Arc::clone(&sorted.schema), window.clone(), self.env.ansi)?;
        let mut layout = sorted.layout.clone();
        layout.extend(calls.iter().map(|c| c.output.id));
        let names: Vec<String> = calls.iter().map(|c| c.output.name.clone()).collect();
        let detail = format!(
            "[{}], partition={}, order={}",
            names.join(", "),
            list(partition_by),
            orders(order_by)
        );
        let mut planned = sorted.wrap(
            |input| PhysNode::Window { input, window },
            crate::operator::Operator::schema(&op),
            layout,
            ("Window", detail),
        );
        // A runtime filter below a window would drop rows its functions
        // count.
        planned.sites.clear();
        Ok(planned)
    }

    /// The executor's function for call `c`, over argument columns `args`
    /// of `input`.
    fn function(&self, c: &Call<'_>, args: &[usize], input: &Planned) -> Result<WindowFunction> {
        let name = c.f.id.name();
        if c.f.kind == FunctionKind::Window {
            return Ok(match name {
                "row_number" => WindowFunction::RowNumber,
                "rank" => WindowFunction::Rank,
                "dense_rank" => WindowFunction::DenseRank,
                "percent_rank" => WindowFunction::PercentRank,
                "cume_dist" => WindowFunction::CumeDist,
                "ntile" => WindowFunction::Ntile(ntile_buckets(c.f)?),
                other => return Err(unsupported(&format!("the window function {other}"))),
            });
        }
        let function = match name {
            "count" => AggregateFunction::Count,
            "sum" => AggregateFunction::Sum,
            "avg" | "mean" => AggregateFunction::Avg,
            "min" => AggregateFunction::Min,
            "max" => AggregateFunction::Max,
            name if picks(name).is_some() => {
                let ignore_nulls = matches!(
                    c.f.args.get(1).map(|a| &a.node),
                    Some(ExprNode::Literal(krishiv_plan::Literal::Boolean(true)))
                );
                if picks(name) == Some(true) {
                    AggregateFunction::Last { ignore_nulls }
                } else {
                    AggregateFunction::First { ignore_nulls }
                }
            }
            other => return Err(unsupported(&format!("the window aggregate {other}"))),
        };
        if function != AggregateFunction::Count && args.len() != 1 {
            return Err(internal("a window aggregate of several arguments"));
        }
        // Spark's STRING and BINARY are views here whatever carrier the
        // input uses; the accumulators read any carrier.
        let arg_type = args.first().map_or(arrow_schema::DataType::Null, |a| {
            match input.schema.field(*a).data_type() {
                arrow_schema::DataType::Utf8 | arrow_schema::DataType::LargeUtf8 => {
                    arrow_schema::DataType::Utf8View
                }
                arrow_schema::DataType::Binary | arrow_schema::DataType::LargeBinary => {
                    arrow_schema::DataType::BinaryView
                }
                t => t.clone(),
            }
        });
        Ok(WindowFunction::Aggregate(AggregateExpr {
            function,
            args: args.to_vec(),
            input: AggInput::Raw,
            arg_type,
            name: c.output.name.clone(),
        }))
    }
}

/// The window call of a window relation's expression `e` (an alias of a
/// resolved window).
fn call(e: &Expr) -> Result<Call<'_>> {
    let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &e.node else {
        return Err(internal("a window expression without an alias"));
    };
    let ExprNode::Window(w) = &expr.node else {
        return Err(internal("a window relation over a non-window expression"));
    };
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &w.function.node else {
        return Err(internal("a window over an unresolved function"));
    };
    let WindowSpecRef::Spec(spec) = &w.window else {
        return Err(internal("an unresolved window"));
    };
    let frame = spec
        .frame
        .as_ref()
        .ok_or_else(|| internal("a window without a frame"))?;
    let bound = |b: &FrameBound| match b {
        FrameBound::UnboundedPreceding => Ok(Bound::UnboundedPreceding),
        FrameBound::CurrentRow => Ok(Bound::CurrentRow),
        FrameBound::UnboundedFollowing => Ok(Bound::UnboundedFollowing),
        _ => Err(unsupported("window frames with value bounds")),
    };
    let frame = WindowFrame {
        units: match frame.units {
            FrameUnits::Rows => Units::Rows,
            FrameUnits::Range => Units::Range,
        },
        lower: bound(&frame.lower)?,
        upper: bound(&frame.upper)?,
    };
    Ok(Call { f, frame, output })
}

/// The argument expressions an aggregate reads: `first(x, flag)` reads
/// `x`; `count(1)` (a non-NULL literal) reads none. Window functions read
/// none (their arguments are constants).
fn arguments(f: &krishiv_plan::ResolvedFunction) -> Vec<&Arc<Expr>> {
    if f.kind == FunctionKind::Window {
        return Vec::new();
    }
    let name = f.id.name();
    f.args
        .iter()
        .take(if picks(name).is_some() { 1 } else { usize::MAX })
        .filter(|a| !non_null_literal(a) || name != "count")
        .collect()
}

/// `ntile(n)`'s bucket count: the analyzer checked it is an `int` literal
/// above 0.
fn ntile_buckets(f: &krishiv_plan::ResolvedFunction) -> Result<i32> {
    match f.args.first().map(|a| &a.node) {
        Some(ExprNode::Literal(krishiv_plan::Literal::Integer(n))) if *n > 0 => Ok(*n),
        _ => Err(unsupported(
            "ntile with a bucket count that is not an int literal",
        )),
    }
}
