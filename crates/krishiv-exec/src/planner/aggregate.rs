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
        for c in &calls {
            let args: Vec<&Arc<Expr>> =
                c.f.args
                    .iter()
                    .take(if picks(c.f.id.name()).is_some() {
                        1
                    } else {
                        usize::MAX
                    })
                    .filter(|a| !non_null_literal(a) || c.f.id.name() != "count")
                    .collect();
            arg_at.push(wanted.len()..wanted.len() + args.len());
            wanted.extend(args.into_iter().cloned());
        }
        let (input, columns) = self.with_columns(input, &wanted)?;
        let k = keys.len();
        let mut aggregates = Vec::with_capacity(calls.len());
        let mut distinct = Vec::with_capacity(calls.len());
        for (c, at) in calls.iter().zip(&arg_at) {
            let function = match c.f.id.name() {
                "count" => AggregateFunction::Count,
                "sum" => AggregateFunction::Sum,
                "avg" => AggregateFunction::Avg,
                "min" => AggregateFunction::Min,
                "max" => AggregateFunction::Max,
                "stddev" | "std" | "stddev_samp" => AggregateFunction::StddevSamp,
                "stddev_pop" => AggregateFunction::StddevPop,
                "variance" | "var_samp" => AggregateFunction::VarSamp,
                "var_pop" => AggregateFunction::VarPop,
                "skewness" => AggregateFunction::Skewness,
                "kurtosis" => AggregateFunction::Kurtosis,
                "corr" => AggregateFunction::Corr,
                "covar_samp" => AggregateFunction::CovarSamp,
                "covar_pop" => AggregateFunction::CovarPop,
                // `first(x, ignoreNulls)`: the analyzer checked the flag is a
                // boolean literal.
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
                other => {
                    return Err(unsupported(&format!("the aggregate function {other}")));
                }
            };
            if c.f.filter.is_some() {
                return Err(unsupported("aggregate FILTER (WHERE ...) clauses"));
            }
            let args: Vec<usize> = columns[at.clone()].to_vec();
            if function != AggregateFunction::Count && args.len() != arity(function) {
                return Err(internal("an aggregate with the wrong number of arguments"));
            }
            // Spark's STRING and BINARY are views here whatever carrier a
            // registered table uses; the accumulators read any carrier.
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
            let agg = AggregateExpr {
                function,
                args,
                input: crate::aggregate::AggInput::Raw,
                arg_type,
                name: c.key.clone(),
            };
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

/// `Some(true)` for `last`/`last_value`, `Some(false)` for `first`/
/// `first_value`: the aggregates that pick a row's value.
pub(super) fn picks(name: &str) -> Option<bool> {
    match name {
        "first" | "first_value" => Some(false),
        "last" | "last_value" => Some(true),
        _ => None,
    }
}

pub(super) fn non_null_literal(e: &Expr) -> bool {
    matches!(&e.node, ExprNode::Literal(l) if !matches!(l, krishiv_plan::Literal::Null(_)))
}
