// SPDX-License-Identifier: Apache-2.0
//! The Python UDF relation nodes planned (design P3-T085 §3.1): map,
//! grouped map, cogrouped map and grouped aggregate, with the data movement
//! Spark gives them (`FlatMapGroupsInBatchExec.scala` L62-71,
//! `FlatMapCoGroupsInBatchExec.scala` L56-64, `ArrowAggregatePythonExec.scala`
//! L82-95 at v4.2.0): a map runs where its input is; a grouped kind gets a
//! hash exchange on its keys (one partition without keys) and a local sort
//! on them, the engine's spillable sort, so no group is held whole.
//!
//! A cogroup's two inputs go through one stream: each side projected to
//! `keys, side, left columns, right columns` (the other side's columns
//! null), a union, then the exchange on the keys and the sort on the keys
//! and the side. Each key's left rows then come before its right rows on
//! one driver, which is what Spark's two co-partitioned sorted inputs and
//! its `CoGroupedIterator` give the UDF.
//!
//! Barrier mode, window aggregates and UDTFs are refused here
//! (`KRISHIV_UNSUPPORTED_FEATURE`): the window and table kinds run with
//! PR-26c.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_schema::{Field, Schema, SchemaRef};
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, LogicalPlan, PythonUdfKind, PythonUdfRef, ResolvedExpr,
    is_python_udaf,
};
use krishiv_types::DataType;

use super::build::{Planned, Planner, internal, unsupported};
use super::explain::ExplainNode;
use super::properties::{Distribution, Placement};
use crate::error::Result;
use crate::operator::{Operator as _, Source as _};
use crate::operators::SortKey;
use crate::physical::{
    PhysNode, ProjectItemPlan, PythonCallPlan, PythonFrameSide, PythonOutputColumn,
    PythonRelationPlan, compile_all, project_op, sort_op,
};

/// The error of a Python UDF kind that runs only from PR-26c.
pub(crate) fn later_kind(what: &str) -> krishiv_common::error::KrishivError {
    unsupported(&format!(
        "{what} (Python window aggregate and table UDFs run from P3-T014 PR-26c)"
    ))
}

/// Refuses `udf` unless its eval type is of `kind`.
fn check_kind(udf: &PythonUdfRef, kind: PythonUdfKind, node: &str) -> Result<()> {
    if udf.eval_type.kind() == kind {
        Ok(())
    } else {
        Err(internal(&format!(
            "{node} over a Python UDF of eval type {}",
            udf.eval_type.code()
        )))
    }
}

/// The declared columns as plan data.
fn outputs(output: &[BoundColumn]) -> Vec<PythonOutputColumn> {
    output
        .iter()
        .map(|c| PythonOutputColumn {
            name: c.name.clone(),
            data_type: c.data_type.clone(),
            nullable: c.nullable,
        })
        .collect()
}

/// The output schema of declared columns (canonical carriers).
fn output_schema(output: &[BoundColumn]) -> SchemaRef {
    Arc::new(Schema::new(
        output
            .iter()
            .map(|c| Field::new(&c.name, c.data_type.to_arrow(), c.nullable))
            .collect::<Vec<_>>(),
    ))
}

/// A frame side: the key columns `key_at` (computed keys included), then
/// the data columns `columns` of a relation whose declared columns are
/// `cols`. Spark's layout (`PandasGroupUtils.resolveArgOffsets` at v4.2.0):
/// the worker builds `key` from the leading columns and the frame from the
/// rest.
fn side_of(
    keys: &[Arc<Expr>],
    key_at: &[usize],
    columns: &[usize],
    cols: &[BoundColumn],
) -> Result<PythonFrameSide> {
    let mut side = PythonFrameSide {
        columns: key_at.iter().chain(columns).copied().collect(),
        names: Vec::with_capacity(keys.len() + cols.len()),
        types: Vec::with_capacity(keys.len() + cols.len()),
    };
    for k in keys {
        let name = match &k.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => c.name.clone(),
            _ => k.to_string(),
        };
        side.names.push(name);
        side.types.push(
            k.resolved_type()
                .ok_or_else(|| internal("a Python UDF grouping key without a type"))?,
        );
    }
    side.names.extend(cols.iter().map(|c| c.name.clone()));
    side.types.extend(cols.iter().map(|c| c.data_type.clone()));
    Ok(side)
}

/// The columns of `p`'s declared output.
fn columns_of(p: &LogicalPlan) -> Result<Vec<BoundColumn>> {
    p.resolved_output()
        .ok_or_else(|| internal("a Python UDF relation over an unresolved input"))
}

impl Planner<'_> {
    /// `mapInPandas` / `mapInArrow`.
    pub(crate) fn map_partitions(
        &mut self,
        input: &Arc<LogicalPlan>,
        udf: &PythonUdfRef,
        output: &[BoundColumn],
        barrier: bool,
    ) -> Result<Planned> {
        if barrier {
            return Err(unsupported(
                "barrier execution mode (isBarrier) of mapInPandas and mapInArrow",
            ));
        }
        check_kind(udf, PythonUdfKind::Map, "MapPartitions")?;
        let cols = columns_of(input)?;
        let planned = self.plan(input)?;
        let columns = cols
            .iter()
            .map(|c| planned.column(c.id))
            .collect::<Result<Vec<_>>>()?;
        let plan = PythonRelationPlan {
            udf: udf.clone(),
            sides: vec![side_of(&[], &[], &columns, &cols)?],
            output: outputs(output),
        };
        let explain = format!("{}#{}", udf.name, udf.version);
        Ok(relation(
            planned,
            |input| PhysNode::PythonMap { input, udf: plan },
            output,
            ("PythonMap", explain),
        ))
    }

    /// `groupBy(keys).applyInPandas` / `applyInArrow`.
    pub(crate) fn group_map(
        &mut self,
        input: &Arc<LogicalPlan>,
        keys: &[Arc<Expr>],
        udf: &PythonUdfRef,
        output: &[BoundColumn],
    ) -> Result<Planned> {
        check_kind(udf, PythonUdfKind::GroupedMap, "GroupMap")?;
        let cols = columns_of(input)?;
        let planned = self.plan(input)?;
        let (planned, key_columns) = self.with_columns(planned, keys)?;
        let columns = cols
            .iter()
            .map(|c| planned.column(c.id))
            .collect::<Result<Vec<_>>>()?;
        let sorted = self.clustered_and_sorted(planned, &key_columns, &[])?;
        let plan = PythonRelationPlan {
            udf: udf.clone(),
            sides: vec![side_of(keys, &key_columns, &columns, &cols)?],
            output: outputs(output),
        };
        let explain = format!(
            "[{}], {}#{}",
            key_names(&sorted, &key_columns),
            udf.name,
            udf.version
        );
        Ok(relation(
            sorted,
            |input| PhysNode::PythonGroupMap {
                input,
                keys: key_columns,
                udf: plan,
            },
            output,
            ("PythonGroupMap", explain),
        ))
    }

    /// `cogroup(...).applyInPandas` / `applyInArrow`.
    pub(crate) fn cogroup_map(
        &mut self,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        left_keys: &[Arc<Expr>],
        right_keys: &[Arc<Expr>],
        udf: &PythonUdfRef,
        output: &[BoundColumn],
    ) -> Result<Planned> {
        check_kind(udf, PythonUdfKind::CogroupedMap, "CoGroupMap")?;
        if left_keys.len() != right_keys.len() {
            return Err(internal("a cogroup with different key counts"));
        }
        let (left_cols, right_cols) = (columns_of(left)?, columns_of(right)?);
        let l = self.plan(left)?;
        let (l, lk) = self.with_columns(l, left_keys)?;
        let r = self.plan(right)?;
        let (r, rk) = self.with_columns(r, right_keys)?;
        let k = lk.len();
        // One layout for both branches: keys, side, left, right.
        let width = k + 1 + left_cols.len() + right_cols.len();
        let layout: Vec<AttrId> = (0..width).map(|_| self.fresh_id()).collect();
        let nullable = vec![true; width];
        let mut branches = Vec::with_capacity(2);
        for (side, (planned, keys)) in [(l, &lk), (r, &rk)].into_iter().enumerate() {
            let mut items = Vec::with_capacity(width);
            for (i, &c) in keys.iter().enumerate() {
                items.push(ProjectItemPlan::Column {
                    index: c,
                    name: Some(format!("_k{i}")),
                });
            }
            items.push(ProjectItemPlan::Expr {
                name: "_side".to_owned(),
                expr: literal(krishiv_plan::Literal::Integer(i32::from(side == 1))),
            });
            for (own, (cols, prefix)) in [(&left_cols, "_l"), (&right_cols, "_r")]
                .into_iter()
                .enumerate()
            {
                for (i, c) in cols.iter().enumerate() {
                    let name = format!("{prefix}{i}");
                    if own == side {
                        items.push(ProjectItemPlan::Column {
                            index: planned.column(c.id)?,
                            name: Some(name),
                        });
                    } else {
                        items.push(ProjectItemPlan::Expr {
                            name,
                            expr: literal(krishiv_plan::Literal::Null(Some(c.data_type.clone()))),
                        });
                    }
                }
            }
            let op = project_op(
                &planned.schema,
                &planned.layout,
                &items,
                Some(&nullable),
                &self.no_values(),
            )?;
            let input_layout = planned.layout.clone();
            let nullable = nullable.clone();
            branches.push(planned.wrap(
                |input| PhysNode::Project {
                    input,
                    layout: input_layout,
                    items,
                    nullable: Some(nullable),
                    subqueries: Vec::new(),
                },
                op.schema(),
                layout.clone(),
                ("Project", format!("cogroup side {side}")),
            ));
        }
        let (Some(rb), Some(lb)) = (branches.pop(), branches.pop()) else {
            return Err(internal("a cogroup without its two sides"));
        };
        if lb.schema.fields() != rb.schema.fields() {
            return Err(unsupported(
                "a cogroup whose keys or columns are stored in different types on its two sides",
            ));
        }
        let union = Planned {
            ordering: Vec::new(),
            schema: Arc::clone(&lb.schema),
            explain: ExplainNode::new("Union", String::new(), vec![lb.explain, rb.explain]),
            node: PhysNode::Union {
                inputs: vec![lb.node, rb.node],
            },
            layout,
            sites: Vec::new(),
            placement: Placement::Unknown,
        };
        let key_columns: Vec<usize> = (0..k).collect();
        let sorted = self.clustered_and_sorted(union, &key_columns, &[k])?;
        let plan = PythonRelationPlan {
            udf: udf.clone(),
            sides: vec![
                side_of(
                    left_keys,
                    &key_columns,
                    &(k + 1..k + 1 + left_cols.len()).collect::<Vec<_>>(),
                    &left_cols,
                )?,
                side_of(
                    right_keys,
                    &key_columns,
                    &(k + 1 + left_cols.len()..width).collect::<Vec<_>>(),
                    &right_cols,
                )?,
            ],
            output: outputs(output),
        };
        let explain = format!("{k} keys, {}#{}", udf.name, udf.version);
        Ok(relation(
            sorted,
            |input| PhysNode::PythonCoGroupMap {
                input,
                keys: key_columns,
                side: k,
                udf: plan,
            },
            output,
            ("PythonCoGroupMap", explain),
        ))
    }

    /// An `Aggregate` whose aggregate calls are grouped-aggregate Python
    /// UDFs (checked by the caller to be the only aggregates): the keys,
    /// one column per distinct call, then the aggregate's own list over them.
    pub(crate) fn python_aggregate(
        &mut self,
        input: &Arc<LogicalPlan>,
        keys: &[Arc<Expr>],
        aggregates: &[Arc<Expr>],
    ) -> Result<Planned> {
        let mut calls: Vec<Arc<Expr>> = Vec::new();
        let mut seen = HashSet::new();
        for a in aggregates {
            udaf_calls(a, &mut calls, &mut seen);
        }
        let planned = self.plan(input)?;
        let (planned, key_columns) = self.with_columns(planned, keys)?;
        let prepared = self.prepare(&[])?;
        for call in &calls {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &call.node else {
                return Err(internal("a Python UDAF call is not a function"));
            };
            refuse_call_syntax(f)?;
            if f.args.iter().any(|a| super::build::has_subquery(a)) {
                return Err(unsupported("a subquery in the arguments of a Python UDF"));
            }
            let args: Vec<&Expr> = f.args.iter().map(AsRef::as_ref).collect();
            compile_all(
                &args,
                &planned.layout,
                &planned.schema,
                &self.values(&prepared),
            )?;
        }
        let sorted = self.clustered_and_sorted(planned, &key_columns, &[])?;
        let mut layout: Vec<AttrId> = key_columns
            .iter()
            .map(|&c| sorted.column_attr(c))
            .collect::<Result<_>>()?;
        let mut fields: Vec<Field> = key_columns
            .iter()
            .map(|&c| sorted.schema.field(c).clone())
            .collect();
        let mut plans = Vec::with_capacity(calls.len());
        let mut columns: Vec<(Arc<Expr>, BoundColumn)> = Vec::with_capacity(calls.len());
        for (i, call) in calls.iter().enumerate() {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &call.node else {
                return Err(internal("a Python UDAF call is not a function"));
            };
            let column = BoundColumn {
                id: self.fresh_id(),
                name: format!("pythonUDAF{i}"),
                data_type: f.data_type.clone(),
                nullable: true,
            };
            fields.push(Field::new(&column.name, f.data_type.to_arrow(), true));
            layout.push(column.id);
            plans.push(PythonCallPlan {
                call: Arc::clone(call),
                output: column.id,
            });
            columns.push((Arc::clone(call), column));
        }
        let detail = calls
            .iter()
            .zip(&columns)
            .map(|(c, (_, col))| format!("{c} AS {}", col.name))
            .collect::<Vec<_>>()
            .join(", ");
        let input_layout = sorted.layout.clone();
        let keys_at = key_columns.clone();
        let schema = Arc::new(Schema::new(fields));
        let node = sorted.wrap(
            |input| PhysNode::PythonAggregate {
                input,
                layout: input_layout,
                keys: keys_at,
                calls: plans,
            },
            schema,
            layout.clone(),
            ("PythonAggregate", detail),
        );
        // The aggregate's list over the keys and the calls' columns.
        let key_bound: Vec<(Arc<Expr>, BoundColumn)> = keys
            .iter()
            .zip(&layout)
            .filter(|(k, _)| column_of(k).is_none())
            .map(|(k, id)| {
                let t = k.resolved_type().unwrap_or(DataType::Null);
                (
                    Arc::clone(k),
                    BoundColumn {
                        id: *id,
                        name: k.to_string(),
                        data_type: t,
                        nullable: true,
                    },
                )
            })
            .collect();
        let exprs = aggregates
            .iter()
            .map(|e| replaced(e, &columns, &key_bound))
            .collect::<Result<Vec<_>>>()?;
        let mut out = self.project(
            Planned {
                placement: Placement::Unknown,
                ordering: Vec::new(),
                sites: Vec::new(),
                ..node
            },
            &exprs,
        )?;
        out.sites.clear();
        Ok(out)
    }

    /// `input` placed by `keys` (one partition without) and sorted by
    /// `keys` then `then`, ascending.
    fn clustered_and_sorted(
        &mut self,
        input: Planned,
        keys: &[usize],
        then: &[usize],
    ) -> Result<Planned> {
        let input = if keys.is_empty() {
            self.require(input, Distribution::Single)?
        } else {
            let attrs: Vec<AttrId> = keys
                .iter()
                .map(|&c| input.column_attr(c))
                .collect::<Result<_>>()?;
            self.require(input, Distribution::Clustered(&attrs))?
        };
        let sort: Vec<SortKey> = keys.iter().chain(then).map(|&c| SortKey::asc(c)).collect();
        let schema = sort_op(&input.schema, &sort, None)?.schema();
        let layout = input.layout.clone();
        let detail = sort
            .iter()
            .map(|k| input.schema.field(k.column).name().clone())
            .collect::<Vec<_>>()
            .join(", ");
        Ok(input.wrap(
            |input| PhysNode::Sort { input, keys: sort },
            schema,
            layout,
            ("Sort", format!("[{detail}]")),
        ))
    }
}

/// A relation node over `input` declaring `output`.
fn relation(
    input: Planned,
    make: impl FnOnce(Box<PhysNode>) -> PhysNode,
    output: &[BoundColumn],
    explain: (&str, String),
) -> Planned {
    Planned {
        node: make(Box::new(input.node)),
        schema: output_schema(output),
        layout: output.iter().map(|c| c.id).collect(),
        explain: ExplainNode::new(explain.0, explain.1, vec![input.explain]),
        sites: Vec::new(),
        placement: Placement::Unknown,
        ordering: Vec::new(),
    }
}

/// The names of `keys`, columns of `input`, joined.
fn key_names(input: &Planned, keys: &[usize]) -> String {
    keys.iter()
        .map(|&k| input.schema.field(k).name().clone())
        .collect::<Vec<_>>()
        .join(", ")
}

fn literal(l: krishiv_plan::Literal) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Literal(l)))
}

/// Refuses a Python UDAF call written with `DISTINCT`, `FILTER (WHERE ...)`
/// or `WITHIN GROUP`: the operator feeds every row of a group to the UDF,
/// so running such a call would silently ignore the clause. Spark refuses
/// them too (a `PythonUDF` is not an `AggregateFunction`); its class is not
/// verified against a Spark run yet, so the refusal is
/// `KRISHIV_UNSUPPORTED_FEATURE`.
fn refuse_call_syntax(f: &krishiv_plan::ResolvedFunction) -> Result<()> {
    let clause = if f.distinct {
        "DISTINCT"
    } else if f.filter.is_some() {
        "FILTER (WHERE ...)"
    } else if !f.within_group.is_empty() {
        "WITHIN GROUP (ORDER BY ...)"
    } else {
        return Ok(());
    };
    Err(unsupported(&format!(
        "{clause} on a call of the Python aggregate UDF {}",
        f.id.name()
    )))
}

/// The attribute `e` reads if it is a plain column.
fn column_of(e: &Expr) -> Option<AttrId> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        _ => None,
    }
}

/// The distinct Python UDAF calls of `e`, in first-seen order.
fn udaf_calls(e: &Arc<Expr>, out: &mut Vec<Arc<Expr>>, seen: &mut HashSet<usize>) {
    if !seen.insert(std::ptr::from_ref::<Expr>(e).addr()) {
        return;
    }
    if is_python_udaf(e) {
        if !out.iter().any(|o| o == e) {
            out.push(Arc::clone(e));
        }
        return;
    }
    e.for_each_child(&mut |c| udaf_calls(c, out, seen));
}

/// `e` reading each call's column and each computed key's column.
fn replaced(
    e: &Arc<Expr>,
    calls: &[(Arc<Expr>, BoundColumn)],
    keys: &[(Arc<Expr>, BoundColumn)],
) -> Result<Arc<Expr>> {
    Arc::clone(e).transform_down(&mut |n: Arc<Expr>| -> Result<Arc<Expr>> {
        Ok(calls
            .iter()
            .chain(keys)
            .find(|(x, _)| *x == n)
            .map_or(n, |(_, column)| {
                Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                    column.clone(),
                ))))
            }))
    })
}
