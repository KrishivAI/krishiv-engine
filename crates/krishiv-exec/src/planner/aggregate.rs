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
use crate::aggregate::{
    AggregateExpr, AggregateFunction, AggregateNode, AggregateOutput, AggregateSpec, HashAggregate,
    distinct_plan,
};
use crate::error::Result;

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
            if function != AggregateFunction::Count && args.len() != 1 {
                return Err(internal("an aggregate of several arguments"));
            }
            let arg_type = args.first().map_or(arrow_schema::DataType::Null, |a| {
                input.schema.field(*a).data_type().clone()
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
        let spec = AggregateSpec {
            input: Arc::clone(&input.schema),
            group_by: columns[..k].to_vec(),
            aggregates,
            output: AggregateOutput::Final,
            ansi: self.env.ansi,
            spill: self.cx.spill.clone(),
            require_unit_weights: false,
        };
        let node = if distinct.iter().any(|d| *d) {
            distinct_plan(Arc::clone(&input.node), spec.clone(), &distinct)?
        } else {
            AggregateNode::new(Arc::clone(&input.node), HashAggregate::new(spec.clone())?)
        };
        let schema = spec.output_schema()?;
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
        let agg = Planned {
            node,
            schema,
            layout,
            explain: super::explain::ExplainNode::new("HashAggregate", detail, vec![input.explain]),
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
fn picks(name: &str) -> Option<bool> {
    match name {
        "first" | "first_value" => Some(false),
        "last" | "last_value" => Some(true),
        _ => None,
    }
}

fn non_null_literal(e: &Expr) -> bool {
    matches!(&e.node, ExprNode::Literal(l) if !matches!(l, krishiv_plan::Literal::Null(_)))
}
