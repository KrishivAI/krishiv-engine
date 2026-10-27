// SPDX-License-Identifier: Apache-2.0
//! `GROUP BY` and aggregates.
//!
//! An `Aggregate` relation becomes three steps, as in Spark's physical
//! plan: a projection computing the grouping keys and the aggregates'
//! arguments, the hash aggregate (two of them for `DISTINCT`), and a
//! projection computing the output expressions from the keys and the
//! aggregates' results (`sum(a) / count(b)`, a key under an alias).

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_functions::registry::{Lookup, builtins};
use krishiv_functions::signature::SparkProfile;
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, FunctionId, FunctionKind, Grouping, ResolvedExpr,
    ResolvedFunction,
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
        // Step 1: keys, then each call's arguments.
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
        let (input, columns) = self.with_columns(input, &wanted)?;
        let k = keys.len();
        let mut aggregates = Vec::with_capacity(calls.len());
        let mut distinct = Vec::with_capacity(calls.len());
        for ((c, at), function) in calls.iter().zip(&arg_at).zip(lowered) {
            if c.f.filter.is_some() {
                return Err(unsupported("aggregate FILTER (WHERE ...) clauses"));
            }
            let args: Vec<usize> = columns[at.clone()].to_vec();
            let agg = aggregate_expr(function, args, &input.schema, c.key.clone())?;
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
        "avg" | "mean" => (AggregateFunction::Avg, f.args.clone()),
        "min" | "bool_and" | "every" => (AggregateFunction::Min, f.args.clone()),
        "max" | "bool_or" | "some" | "any" => (AggregateFunction::Max, f.args.clone()),
        "first" | "first_value" => (AggregateFunction::First { ignore_nulls }, value),
        "last" | "last_value" => (AggregateFunction::Last { ignore_nulls }, value),
        "any_value" => (AggregateFunction::AnyValue { ignore_nulls }, value),
        "max_by" if f.args.len() == 2 => (AggregateFunction::MaxBy, f.args.clone()),
        "min_by" if f.args.len() == 2 => (AggregateFunction::MinBy, f.args.clone()),
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
        "percentile" if f.args.len() == 3 => {
            let (percentages, array) = doubles_arg(&f.args[1])?;
            (
                AggregateFunction::Percentile {
                    percentages,
                    array,
                    reverse: false,
                },
                vec![Arc::clone(&f.args[0]), Arc::clone(&f.args[2])],
            )
        }
        "median" if f.args.len() == 1 => (
            AggregateFunction::Percentile {
                percentages: vec![0.5],
                array: false,
                reverse: false,
            },
            vec![Arc::clone(&f.args[0]), long_lit(1)],
        ),
        "percentile_approx" | "approx_percentile" if f.args.len() == 3 => {
            let (percentages, array) = doubles_arg(&f.args[1])?;
            let ExprNode::Literal(krishiv_plan::Literal::Long(accuracy)) = &f.args[2].node else {
                return None;
            };
            (
                AggregateFunction::PercentileApprox {
                    percentages,
                    array,
                    accuracy: *accuracy,
                },
                vec![Arc::clone(&f.args[0])],
            )
        }
        "percentile_cont" if f.args.len() == 4 => {
            let (percentages, array) = doubles_arg(&f.args[1])?;
            let ExprNode::Literal(krishiv_plan::Literal::Boolean(reverse)) = &f.args[2].node else {
                return None;
            };
            (
                AggregateFunction::Percentile {
                    percentages,
                    array,
                    reverse: *reverse,
                },
                vec![Arc::clone(&f.args[0]), long_lit(1)],
            )
        }
        "percentile_disc" if f.args.len() == 4 => {
            let (percentages, array) = doubles_arg(&f.args[1])?;
            let ExprNode::Literal(krishiv_plan::Literal::Boolean(reverse)) = &f.args[2].node else {
                return None;
            };
            let ExprNode::Literal(krishiv_plan::Literal::Boolean(legacy)) = &f.args[3].node else {
                return None;
            };
            (
                AggregateFunction::PercentileDisc {
                    percentages,
                    array,
                    reverse: *reverse,
                    legacy: *legacy,
                },
                vec![Arc::clone(&f.args[0])],
            )
        }
        "approx_count_distinct" if f.args.len() == 2 => {
            let rsd = double_arg(&f.args[1])?;
            let p = crate::aggregate::precision_for(rsd)?;
            (
                AggregateFunction::ApproxCountDistinct { p },
                vec![Arc::clone(&f.args[0])],
            )
        }
        _ => return None,
    };
    Some(Lowered { function, args })
}

/// A folded percentage list: `Literal::Double(v)` is one percentage, the
/// analyzer-built array of doubles is several (in order); anything else is
/// not an analyzer-lowered call.
fn doubles_arg(e: &Arc<Expr>) -> Option<(Vec<f64>, bool)> {
    match &e.node {
        ExprNode::Literal(krishiv_plan::Literal::Double(v)) => Some((vec![*v], false)),
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.id.name() == "array"
                && f.args.iter().all(|a| {
                    matches!(&a.node, ExprNode::Literal(krishiv_plan::Literal::Double(_)))
                }) =>
        {
            Some((
                f.args
                    .iter()
                    .map(|a| match &a.node {
                        ExprNode::Literal(krishiv_plan::Literal::Double(v)) => *v,
                        _ => f64::NAN,
                    })
                    .collect(),
                true,
            ))
        }
        _ => None,
    }
}

/// A folded double: `Literal::Double`, or a decimal literal read as
/// correctly rounded (as Spark's `Decimal.toDouble` reads it).
fn double_arg(e: &Arc<Expr>) -> Option<f64> {
    match &e.node {
        ExprNode::Literal(krishiv_plan::Literal::Double(v)) => Some(*v),
        ExprNode::Literal(krishiv_plan::Literal::Decimal {
            unscaled,
            data_type,
        }) => Some(crate::aggregate::decimal_to_double(
            *unscaled,
            i8::try_from(data_type.scale()).unwrap_or(i8::MAX),
        )),
        _ => None,
    }
}

/// `Literal::Long(v)`, for a synthesized frequency of one.
fn long_lit(v: i64) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Literal(krishiv_plan::Literal::Long(v))))
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
            data_type: krishiv_types::DataType::Boolean,
            nullable: true,
            deterministic: true,
        }),
    ))))
}

/// The executor's aggregate `function` over columns `args` of `schema`.
/// Spark's STRING and BINARY are views here whatever carrier a registered
/// table uses; the accumulators read any carrier.
pub(super) fn aggregate_expr(
    function: AggregateFunction,
    args: Vec<usize>,
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
        return Ok(AggregateExpr::by(
            function,
            (value, key),
            (typed(&value), typed(&key)),
            name,
        ));
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
    })
}

fn non_null_literal(e: &Expr) -> bool {
    matches!(&e.node, ExprNode::Literal(l) if !matches!(l, krishiv_plan::Literal::Null(_)))
}
