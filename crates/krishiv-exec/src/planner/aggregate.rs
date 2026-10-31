// SPDX-License-Identifier: Apache-2.0
//! `GROUP BY` and aggregates.
//!
//! An `Aggregate` relation becomes three steps, as in Spark's physical
//! plan: a projection computing the grouping keys, the aggregates'
//! arguments and their filters, the hash aggregate (two of them for
//! `DISTINCT`), and a projection computing the output expressions from
//! the keys and the aggregates' results (`sum(a) / count(b)`, a key
//! under an alias). A call with a filter reads only the rows where its
//! mask column is true (Spark's `FILTER (WHERE ...)`); other aggregates
//! read all rows.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_functions::registry::{Lookup, builtins};
use krishiv_functions::signature::SparkProfile;
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, FunctionId, FunctionKind, Grouping, Literal, NullOrdering,
    ResolvedExpr, ResolvedFunction, SortDirection,
};

use super::build::{Planned, Planner, column_of, internal, list, unsupported};
use super::properties::Distribution;
use crate::aggregate::{AggregateExpr, AggregateFunction, AggregateOutput, arity};
use crate::error::Result;
use crate::physical::{AggregatePlan, PhysNode, aggregate_node, aggregate_spec};

/// One aggregate call of the output expressions.
struct Call {
    /// Its text, which identifies equal calls.
    key: String,
    f: ResolvedFunction,
    /// The column of its result.
    id: AttrId,
}

impl Planner<'_> {
    pub(crate) fn aggregate(
        &mut self,
        input: Planned,
        grouping: &Grouping,
        outputs: &[Arc<Expr>],
    ) -> Result<Planned> {
        let Grouping::GroupBy(keys) = grouping else {
            return Err(unsupported("ROLLUP, CUBE and GROUPING SETS"));
        };
        // The aggregate calls, once each.
        let mut calls: Vec<Call> = Vec::new();
        for e in outputs {
            self.calls(e, &mut calls)?;
        }
        // Step 1: keys, then each call's arguments, then each call's
        // filter (a boolean mask column when the call filters).
        let mut wanted: Vec<Arc<Expr>> = keys.clone();
        let mut arg_at = Vec::with_capacity(calls.len());
        let mut lowered = Vec::with_capacity(calls.len());
        for c in &calls {
            let l = lower(&c.f)
                .ok_or_else(|| unsupported(&format!("the aggregate function {}", c.f.id.name())))?;
            arg_at.push(wanted.len()..wanted.len() + l.args.len());
            wanted.extend(l.args.iter().cloned());
            lowered.push(l.function);
        }
        let mut mask_at = Vec::with_capacity(calls.len());
        for c in &calls {
            mask_at.push(c.f.filter.as_ref().map(|f| {
                let at = wanted.len();
                wanted.push(Arc::clone(f));
                at
            }));
        }
        let (input, columns) = self.with_columns(input, &wanted)?;
        let k = keys.len();
        let mut aggregates = Vec::with_capacity(calls.len());
        let mut distinct = Vec::with_capacity(calls.len());
        for (((c, at), function), mask) in calls.iter().zip(&arg_at).zip(lowered).zip(mask_at) {
            let args: Vec<usize> = columns[at.clone()].to_vec();
            let mask = mask.map(|m| columns[m]);
            let agg = aggregate_expr(function, args, mask, &input.schema, c.key.clone())?;
            let want = c.f.data_type.to_arrow();
            if agg.result_type()? != want {
                return Err(internal(&format!(
                    "{} computes {} where the analyzer typed {want}",
                    c.key,
                    agg.result_type()?
                )));
            }
            aggregates.push(agg);
            distinct.push(c.f.distinct);
        }
        // Equal keys must meet in one partition; without keys, every row.
        let key_attrs: Vec<AttrId> = columns[..k]
            .iter()
            .map(|c| input.column_attr(*c))
            .collect::<Result<_>>()?;
        let input = if k == 0 {
            self.require(input, Distribution::Single)?
        } else {
            self.require(input, Distribution::Clustered(&key_attrs))?
        };
        let plan = AggregatePlan {
            group_by: columns[..k].to_vec(),
            aggregates,
            distinct,
            output: AggregateOutput::Final,
        };
        // Validates the aggregate (over an input that is never run) and
        // gives its schema.
        let spec = aggregate_spec(&input.schema, &plan, &self.env, None);
        let schema = spec.output_schema()?;
        aggregate_node(Arc::new(crate::physical::Unbuilt), spec, &plan.distinct)?;
        let mut layout: Vec<AttrId> = columns[..k].iter().map(|c| input.layout[*c]).collect();
        layout.extend(calls.iter().map(|c| c.id));
        let detail = format!(
            "keys={}, functions=[{}]",
            list(keys),
            calls
                .iter()
                .map(|c| c.key.clone())
                .collect::<Vec<_>>()
                .join(", ")
        );
        // Grouping keys keep their attributes: a filter on one drops whole
        // groups, which the producer above would drop anyway.
        let sites = super::runtime_filter::kept(&input.sites, &layout[..k]);
        let placement = input.placement.clone().restrict(&layout);
        let agg = Planned {
            node: PhysNode::Aggregate {
                input: Box::new(input.node),
                aggregate: plan,
            },
            schema,
            layout,
            explain: super::explain::ExplainNode::new("HashAggregate", detail, vec![input.explain]),
            sites,
            placement,
        };
        // Step 3: the output expressions over keys and results.
        let mut key_ids: HashMap<String, BoundColumn> = HashMap::new();
        for (i, e) in keys.iter().enumerate() {
            if column_of(e).is_none() {
                let field = agg.schema.field(i);
                let data_type = krishiv_types::DataType::from_arrow(field.data_type())?;
                key_ids.insert(
                    e.to_string(),
                    BoundColumn {
                        id: agg.layout[i],
                        name: e.to_string(),
                        data_type,
                        nullable: field.is_nullable(),
                    },
                );
            }
        }
        let call_ids: HashMap<&str, BoundColumn> = calls
            .iter()
            .map(|c| {
                (
                    c.key.as_str(),
                    BoundColumn {
                        id: c.id,
                        name: c.key.clone(),
                        data_type: c.f.data_type.clone(),
                        nullable: c.f.nullable,
                    },
                )
            })
            .collect();
        let rewritten = outputs
            .iter()
            .map(|e| replace(e, &key_ids, &call_ids))
            .collect::<Result<Vec<_>>>()?;
        self.project(agg, &rewritten)
    }

    /// Collects the aggregate calls of `e`.
    fn calls(&mut self, e: &Expr, out: &mut Vec<Call>) -> Result<()> {
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
            && f.kind == FunctionKind::Aggregate
        {
            let key = e.to_string();
            if !out.iter().any(|c| c.key == key) {
                if !matches!(f.id, FunctionId::Registry { .. }) {
                    return Err(unsupported(&format!("the aggregate {key}")));
                }
                let id = self.fresh_id();
                out.push(Call {
                    key,
                    f: f.clone(),
                    id,
                });
            }
            return Ok(());
        }
        let mut result = Ok(());
        e.for_each_child(&mut |c| {
            if result.is_ok() {
                result = self.calls(c, out);
            }
        });
        result
    }
}

/// `e` with aggregate calls and computed grouping keys replaced by the
/// columns holding them.
fn replace(
    e: &Arc<Expr>,
    keys: &HashMap<String, BoundColumn>,
    calls: &HashMap<&str, BoundColumn>,
) -> Result<Arc<Expr>> {
    Arc::clone(e).transform_down(&mut |x: Arc<Expr>| {
        if matches!(x.node, ExprNode::Resolved(ResolvedExpr::Alias { .. })) {
            return Ok::<_, krishiv_common::error::KrishivError>(x);
        }
        let text = x.to_string();
        let found = calls.get(text.as_str()).or_else(|| keys.get(&text));
        Ok(match found {
            Some(c) => Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                c.clone(),
            )))),
            None => x,
        })
    })
}

/// An aggregate call as the executor runs it: the function and the
/// argument expressions it reads (`first(x, flag)` reads `x`; `count(1)`
/// reads none). The one place that maps registry functions to executor
/// ones, for aggregates and window aggregates alike. Some aggregates are
/// defined by another, whose accumulator they reuse (the output keeps their
/// own name): `count_if(c)` is `count(nullif(c, false))`, `bool_and` and
/// `every` are `min`, `bool_or`, `some` and `any` are `max` (the registry
/// types their argument as `boolean`).
pub(super) struct Lowered {
    pub(super) function: AggregateFunction,
    pub(super) args: Vec<Arc<Expr>>,
}

/// `None` for a function the executor does not run as an aggregate.
pub(super) fn lower(f: &ResolvedFunction) -> Option<Lowered> {
    // `first(x, ignoreNulls)`: the analyzer checked the flag is a boolean
    // literal.
    let ignore_nulls = matches!(
        f.args.get(1).map(|a| &a.node),
        Some(ExprNode::Literal(krishiv_plan::Literal::Boolean(true)))
    );
    let value: Vec<Arc<Expr>> = f.args.iter().take(1).cloned().collect();
    let (function, args) = match f.id.name() {
        "count" => (
            AggregateFunction::Count,
            f.args
                .iter()
                .filter(|a| !non_null_literal(a))
                .cloned()
                .collect(),
        ),
        "count_if" => (
            AggregateFunction::Count,
            vec![null_if_false(f.args.first()?)?],
        ),
        "sum" => (AggregateFunction::Sum, f.args.clone()),
        "try_sum" if f.args.len() == 1 => (AggregateFunction::TrySum, f.args.clone()),
        "avg" | "mean" => (AggregateFunction::Avg, f.args.clone()),
        "try_avg" if f.args.len() == 1 => (AggregateFunction::TryAvg, f.args.clone()),
        "min" | "bool_and" | "every" => (AggregateFunction::Min, f.args.clone()),
        "max" | "bool_or" | "some" | "any" => (AggregateFunction::Max, f.args.clone()),
        "first" | "first_value" => (AggregateFunction::First { ignore_nulls }, value),
        "last" | "last_value" => (AggregateFunction::Last { ignore_nulls }, value),
        "any_value" => (AggregateFunction::AnyValue { ignore_nulls }, value),
        "max_by" if f.args.len() == 2 => (AggregateFunction::MaxBy, f.args.clone()),
        "min_by" if f.args.len() == 2 => (AggregateFunction::MinBy, f.args.clone()),
        "collect_list" if f.args.len() == 1 => (AggregateFunction::CollectList, f.args.clone()),
        "collect_set" if f.args.len() == 1 => (AggregateFunction::CollectSet, f.args.clone()),
        "vector_sum" if f.args.len() == 1 => (AggregateFunction::VectorSum, f.args.clone()),
        "vector_avg" if f.args.len() == 1 => (AggregateFunction::VectorAvg, f.args.clone()),
        "stddev" | "std" | "stddev_samp" if f.args.len() == 1 => {
            (AggregateFunction::StddevSamp, f.args.clone())
        }
        "stddev_pop" if f.args.len() == 1 => (AggregateFunction::StddevPop, f.args.clone()),
        "variance" | "var_samp" if f.args.len() == 1 => {
            (AggregateFunction::VarSamp, f.args.clone())
        }
        "var_pop" if f.args.len() == 1 => (AggregateFunction::VarPop, f.args.clone()),
        "skewness" if f.args.len() == 1 => (AggregateFunction::Skewness, f.args.clone()),
        "kurtosis" if f.args.len() == 1 => (AggregateFunction::Kurtosis, f.args.clone()),
        "corr" if f.args.len() == 2 => (AggregateFunction::Corr, f.args.clone()),
        "covar_samp" if f.args.len() == 2 => (AggregateFunction::CovarSamp, f.args.clone()),
        "covar_pop" if f.args.len() == 2 => (AggregateFunction::CovarPop, f.args.clone()),
        "regr_count" if f.args.len() == 2 => (AggregateFunction::RegrCount, f.args.clone()),
        "regr_avgx" if f.args.len() == 2 => (AggregateFunction::RegrAvgx, f.args.clone()),
        "regr_avgy" if f.args.len() == 2 => (AggregateFunction::RegrAvgy, f.args.clone()),
        "regr_sxx" if f.args.len() == 2 => (AggregateFunction::RegrSxx, f.args.clone()),
        "regr_syy" if f.args.len() == 2 => (AggregateFunction::RegrSyy, f.args.clone()),
        "regr_sxy" if f.args.len() == 2 => (AggregateFunction::RegrSxy, f.args.clone()),
        "regr_slope" if f.args.len() == 2 => (AggregateFunction::RegrSlope, f.args.clone()),
        "regr_intercept" if f.args.len() == 2 => (AggregateFunction::RegrIntercept, f.args.clone()),
        "regr_r2" if f.args.len() == 2 => (AggregateFunction::RegrR2, f.args.clone()),
        "bit_and" if f.args.len() == 1 => (AggregateFunction::BitAnd, f.args.clone()),
        "bit_or" if f.args.len() == 1 => (AggregateFunction::BitOr, f.args.clone()),
        "bit_xor" if f.args.len() == 1 => (AggregateFunction::BitXor, f.args.clone()),
        "mode" => mode_lower(f)?,
        "listagg" => listagg_lower(f)?,
        _ => return None,
    };
    Some(Lowered { function, args })
}

/// `mode(col[, deterministic])`: the analyzer types the ordered forms
/// and keeps the value with a folded tie-break flag (largest when true)
/// as the second argument; the executor reads the value alone.
fn mode_lower(f: &ResolvedFunction) -> Option<(AggregateFunction, Vec<Arc<Expr>>)> {
    let value = Arc::clone(f.args.first()?);
    Some(match f.args.get(1).map(|a| &a.node) {
        None => (AggregateFunction::Mode, vec![value]),
        Some(ExprNode::Literal(Literal::Boolean(largest))) => (
            if *largest {
                AggregateFunction::ModeMax
            } else {
                AggregateFunction::ModeMin
            },
            vec![value],
        ),
        _ => return None,
    })
}

/// `listagg(expr[, delimiter])` with its `WITHIN GROUP` ordering: the
/// analyzer folds the delimiter to a literal and keeps the value, the
/// delimiter and the ordering keys as the arguments; the executor reads
/// the value and the ordering keys, with the fused delimiter.
fn listagg_lower(f: &ResolvedFunction) -> Option<(AggregateFunction, Vec<Arc<Expr>>)> {
    let value = Arc::clone(f.args.first()?);
    // The ordering keys follow the value and the delimiter.
    let orders = f.within_group.len();
    if f.args.len() != orders + 2 {
        return None;
    }
    let delimiter = match f.args.get(1).map(|a| &a.node) {
        Some(ExprNode::Literal(Literal::String(s))) => s.as_bytes().to_vec(),
        Some(ExprNode::Literal(Literal::Binary(b))) => b.clone(),
        Some(ExprNode::Literal(Literal::Null(_))) => Vec::new(),
        _ => return None,
    };
    let dirs = f
        .within_group
        .iter()
        .map(|o| {
            (
                o.direction == SortDirection::Descending,
                o.nulls == NullOrdering::NullsFirst,
            )
        })
        .collect();
    let mut args = vec![value];
    args.extend(f.args.iter().skip(2).cloned());
    Some((
        AggregateFunction::ListAgg {
            delimiter,
            orders: dirs,
        },
        args,
    ))
}

/// `nullif(c, false)`: `c` where it is true, else NULL, so `count` of it
/// counts the rows where `c` is true.
fn null_if_false(c: &Arc<Expr>) -> Option<Arc<Expr>> {
    let profile = SparkProfile::ALL.into_iter().max()?;
    let Lookup::Found(def) = builtins().lookup("nullif", profile) else {
        return None;
    };
    Some(Arc::new(Expr::new(ExprNode::Resolved(
        ResolvedExpr::Function(ResolvedFunction {
            id: FunctionId::Registry {
                name: def.name.to_owned(),
                id: def.id,
            },
            kind: FunctionKind::Scalar,
            args: vec![
                Arc::clone(c),
                Arc::new(Expr::literal(krishiv_plan::Literal::Boolean(false))),
            ],
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: krishiv_types::DataType::Boolean,
            nullable: true,
            deterministic: true,
        }),
    ))))
}

/// The executor's aggregate `function` over columns `args` with boolean
/// mask column `mask` (`FILTER (WHERE ...)`; `None` reads all rows), of
/// `schema`. Spark's STRING and BINARY are views here whatever carrier a
/// registered table uses; the accumulators read any carrier.
pub(super) fn aggregate_expr(
    function: AggregateFunction,
    args: Vec<usize>,
    mask: Option<usize>,
    schema: &arrow_schema::Schema,
    name: String,
) -> Result<AggregateExpr> {
    let typed = |a: &usize| match schema.field(*a).data_type() {
        arrow_schema::DataType::Utf8 | arrow_schema::DataType::LargeUtf8 => {
            arrow_schema::DataType::Utf8View
        }
        arrow_schema::DataType::Binary | arrow_schema::DataType::LargeBinary => {
            arrow_schema::DataType::BinaryView
        }
        t => t.clone(),
    };
    if matches!(
        function,
        AggregateFunction::MaxBy | AggregateFunction::MinBy
    ) {
        let [value, key] = args[..] else {
            return Err(internal("max_by without a value and a key"));
        };
        let mut expr =
            AggregateExpr::by(function, (value, key), (typed(&value), typed(&key)), name);
        expr.mask = mask;
        return Ok(expr);
    }
    if let AggregateFunction::ListAgg { .. } = function {
        let [value, orders @ ..] = &args[..] else {
            return Err(internal("listagg without a value"));
        };
        let mut expr = AggregateExpr::listagg(
            function,
            *value,
            typed(value),
            orders.to_vec(),
            orders.iter().map(&typed).collect(),
            name,
        );
        expr.mask = mask;
        return Ok(expr);
    }
    if function != AggregateFunction::Count && args.len() != arity(&function) {
        return Err(internal("an aggregate with the wrong number of arguments"));
    }
    let arg_type = args.first().map_or(arrow_schema::DataType::Null, typed);
    Ok(AggregateExpr {
        function,
        args,
        input: crate::aggregate::AggInput::Raw,
        arg_type,
        name,
        mask,
    })
}

fn non_null_literal(e: &Expr) -> bool {
    matches!(&e.node, ExprNode::Literal(l) if !matches!(l, krishiv_plan::Literal::Null(_)))
}
