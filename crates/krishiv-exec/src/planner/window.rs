// SPDX-License-Identifier: Apache-2.0
//! Window relations: a projection computing the keys and arguments that
//! are not columns, a sort by the partitioning keys then the ordering
//! keys, and the [`Window`](crate::window::Window) operator over it (see
//! [`crate::window`]).

use std::sync::Arc;

use krishiv_plan::{
    AttrId, Expr, ExprNode, FrameBound, FrameUnits, FunctionKind, NullOrdering, ResolvedExpr,
    SortDirection, SortOrder, WindowSpecRef,
};

use super::aggregate::{aggregate_expr, lower};
use super::build::{Planned, Planner, internal, list, orders, unsupported};
use super::properties::Distribution;
use crate::aggregate::is_moment;
use crate::error::Result;
use crate::operator::Source;
use crate::operators::SortKey;
use crate::operators::TopN;
use crate::physical::PhysNode;
use crate::physical::{group_limit_op, top_n_op};
use crate::window::{
    FrameBound as Bound, FrameUnits as Units, GroupLimitPlan, RankLike, RequiredDistribution,
    Window, WindowFrame, WindowFunction, WindowFunctionPlan, WindowPlan,
};
use krishiv_common::config::{ConfigValue, WINDOW_GROUP_LIMIT_THRESHOLD};
use krishiv_optimizer::{DEFAULT_WINDOW_GROUP_LIMIT_THRESHOLD, RankLikeFunction, WindowGroupLimit};

/// One window expression, read off the plan: its function call, frame and
/// output attribute, and the expressions it reads (arguments, filter).
struct Call<'a> {
    f: &'a krishiv_plan::ResolvedFunction,
    frame: WindowFrame,
    output: &'a krishiv_plan::BoundColumn,
}

impl Planner<'_> {
    /// `spark.sql.optimizer.windowGroupLimitThreshold` (Spark's default
    /// when unset); Spark refuses values below -1, which turn the rule off
    /// here as -1 does.
    pub(crate) fn window_group_limit_threshold(&self) -> i64 {
        self.cx
            .config
            .get(&WINDOW_GROUP_LIMIT_THRESHOLD)
            .and_then(ConfigValue::as_int)
            .unwrap_or(DEFAULT_WINDOW_GROUP_LIMIT_THRESHOLD)
            .max(-1)
    }

    pub(crate) fn window(
        &mut self,
        input: Planned,
        exprs: &[Arc<Expr>],
        partition_by: &[Arc<Expr>],
        order_by: &[SortOrder],
        limit: Option<WindowGroupLimit>,
    ) -> Result<Planned> {
        let calls = exprs.iter().map(|e| call(e)).collect::<Result<Vec<_>>>()?;
        // The columns the window reads: partitioning, ordering, then each
        // function's arguments and filter.
        let mut wanted: Vec<Arc<Expr>> = partition_by.to_vec();
        wanted.extend(order_by.iter().map(|o| Arc::clone(&o.expr)));
        let mut arg_at = Vec::with_capacity(calls.len());
        for c in &calls {
            let args = arguments(c.f);
            let start = wanted.len();
            arg_at.push((start, args.len()));
            wanted.extend(args);
            wanted.extend(c.f.filter.iter().cloned());
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
        let mut sorted_by: Vec<SortOrder> = partition_by
            .iter()
            .map(|e| SortOrder::asc(Arc::clone(e)))
            .collect();
        sorted_by.extend(order_by.iter().cloned());
        let top = match limit {
            Some(WindowGroupLimit::Limit(n)) => usize::try_from(n)
                .is_ok_and(|n| n <= TopN::MAX_ROWS)
                .then_some(n),
            _ => None,
        };
        let group = match limit {
            Some(WindowGroupLimit::Group { function, limit }) => Some((function, limit)),
            // Past the top-N's rows: the group limit of one partition.
            Some(WindowGroupLimit::Limit(n)) if top.is_none() => {
                Some((RankLikeFunction::RowNumber, n))
            }
            _ => None,
        };
        // No row passes the filter: none reaches the sort (Spark plans an
        // empty relation).
        let input = if limit == Some(WindowGroupLimit::Empty) {
            Self::local_limit(input, 0)
        } else {
            input
        };
        // `row_number` without partitioning (Spark's top-K path): each
        // partition keeps its first `n` rows before the gather.
        let input = match top {
            Some(n) if !input.placement.satisfies(Distribution::Single) => {
                let detail = format!("{n}, {}", orders(&sorted_by));
                Self::local_top_n(input, &keys, n, detail)?
            }
            _ => input,
        };
        // Every row of a partition in one partition of the input (the
        // property pass inserts an exchange where the input is not placed
        // so), then the local sort; without keys the sort gathers the rows
        // into one stream.
        let input = match window.required_distribution() {
            RequiredDistribution::Single => self.require(input, Distribution::Single)?,
            RequiredDistribution::Clustered(cols) => {
                let attrs: Vec<AttrId> = cols
                    .iter()
                    .map(|&c| input.column_attr(c))
                    .collect::<Result<_>>()?;
                self.require(input, Distribution::Clustered(&attrs))?
            }
        };
        // The sort (or the top-N), then the window over it.
        let layout = input.layout.clone();
        // The sort's keys as a sort shows them: partitioning keys ascending,
        // then the ordering.
        let sort_detail = orders(&sorted_by);
        let sorted = if let Some(n) = top {
            let schema = top_n_op(&input.schema, &keys, n)?.schema();
            input.wrap(
                |input| PhysNode::TopN {
                    input,
                    keys,
                    limit: n,
                },
                schema,
                layout,
                ("TopN", format!("{n}, {sort_detail}")),
            )
        } else {
            let sort_schema = crate::physical::sort_op(&input.schema, &keys, None)?.schema();
            input.wrap(
                |input| PhysNode::Sort { input, keys },
                sort_schema,
                layout,
                ("Sort", sort_detail),
            )
        };
        let sorted = match group {
            Some((function, limit)) => {
                let plan = GroupLimitPlan {
                    partition_by: window.partition_by.clone(),
                    order_by: window.order_by.clone(),
                    function: match function {
                        RankLikeFunction::RowNumber => RankLike::RowNumber,
                        RankLikeFunction::Rank => RankLike::Rank,
                        RankLikeFunction::DenseRank => RankLike::DenseRank,
                    },
                    limit,
                };
                let schema =
                    crate::operator::Operator::schema(&*group_limit_op(&sorted.schema, &plan)?);
                let layout = sorted.layout.clone();
                let detail = format!(
                    "{}, {}, {}(), {limit}",
                    list(partition_by),
                    orders(order_by),
                    plan.function.name()
                );
                sorted.wrap(
                    |input| PhysNode::WindowGroupLimit { input, limit: plan },
                    schema,
                    layout,
                    ("WindowGroupLimit", detail),
                )
            }
            None => sorted,
        };
        let op = Window::new(
            Arc::clone(&sorted.schema),
            window.clone(),
            self.env.ansi,
            None,
        )?;
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
                "lag" | "lead" => {
                    // The analyzer made the offset the frame: `ROWS BETWEEN
                    // k FOLLOWING AND k FOLLOWING`.
                    let Bound::Offset(offset) = c.frame.lower else {
                        return Err(internal("lag/lead without its offset frame"));
                    };
                    WindowFunction::Offset {
                        input: *args
                            .first()
                            .ok_or_else(|| internal("lag/lead without input"))?,
                        offset,
                        default: args.get(1).copied(),
                        ignore_nulls: flag(c.f, 3),
                        data_type: c.output.data_type.to_arrow(),
                    }
                }
                "nth_value" => WindowFunction::NthValue {
                    input: *args
                        .first()
                        .ok_or_else(|| internal("nth_value without input"))?,
                    n: match c.f.args.get(1).map(|a| &a.node) {
                        Some(ExprNode::Literal(krishiv_plan::Literal::Integer(n))) => i64::from(*n),
                        _ => {
                            return Err(unsupported(
                                "nth_value with an offset that is not an int literal",
                            ));
                        }
                    },
                    ignore_nulls: flag(c.f, 2),
                    data_type: c.output.data_type.to_arrow(),
                },
                other => return Err(unsupported(&format!("the window function {other}"))),
            });
        }
        let lowered =
            lower(c.f).ok_or_else(|| unsupported(&format!("the window aggregate {name}")))?;
        // The moment aggregates have no window form yet: refused, never
        // computed as another function.
        if is_moment(lowered.function) {
            return Err(unsupported(&format!("the window aggregate {name}")));
        }
        let function = lowered.function;
        // The analyzer rejects `count(DISTINCT x) OVER (...)` as Spark does;
        // a plan that still holds one must not be computed as `count(x)`.
        if c.f.distinct {
            return Err(unsupported("DISTINCT window aggregates"));
        }
        Ok(WindowFunction::Aggregate(aggregate_expr(
            function,
            args.to_vec(),
            // A window applies the call's filter itself (as the
            // `WindowFunctionPlan`'s own filter column); the accumulator's
            // mask stays empty.
            None,
            &input.schema,
            c.output.name.clone(),
        )?))
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
        // The analyzer leaves a checked offset as an `int` literal.
        FrameBound::Offset(e) => match &e.node {
            ExprNode::Literal(krishiv_plan::Literal::Integer(v)) => {
                Ok(Bound::Offset(i64::from(*v)))
            }
            _ => Err(unsupported("window frame bounds that are not int literals")),
        },
        _ => Err(unsupported("this window frame bound")),
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

/// The argument expressions an aggregate reads (see [`lower`]). Window
/// functions read none (their arguments are constants).
fn arguments(f: &krishiv_plan::ResolvedFunction) -> Vec<Arc<Expr>> {
    if f.kind == FunctionKind::Window {
        // `lag`/`lead` read their input and default, `nth_value` its input.
        return match f.id.name() {
            "lag" | "lead" => [f.args.first(), f.args.get(2)]
                .into_iter()
                .flatten()
                .cloned()
                .collect(),
            "nth_value" => f.args.first().into_iter().cloned().collect(),
            _ => Vec::new(),
        };
    }
    lower(f).map(|l| l.args).unwrap_or_default()
}

/// Whether argument `i` of `f` is the literal `true` (an `IGNORE NULLS`
/// flag).
fn flag(f: &krishiv_plan::ResolvedFunction, i: usize) -> bool {
    matches!(
        f.args.get(i).map(|a| &a.node),
        Some(ExprNode::Literal(krishiv_plan::Literal::Boolean(true)))
    )
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
