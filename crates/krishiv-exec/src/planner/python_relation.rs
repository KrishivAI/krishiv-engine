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
//! A window's Python aggregate calls (Spark's `ArrowWindowPythonExec`,
//! `WindowExecBase` at v4.2.0) run over the window's own placement and sort
//! (`Planner::window_with`): one task per call and driver, each partition a
//! group, one value per row. A bounded frame's rows come with each row's
//! frame start and end, from two `count(*)` window columns below it.
//!
//! Barrier mode and UDTFs are refused here (`KRISHIV_UNSUPPORTED_FEATURE`):
//! the table kinds run with PR-26c.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_schema::{Field, Schema, SchemaRef};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, CastMode, Expr, ExprNode, FunctionId, FunctionKind, LogicalPlan,
    PythonEvalType, PythonUdfKind, PythonUdfRef, ResolvedExpr, ResolvedFunction, SortOrder,
    is_python_udaf,
};
use krishiv_types::DataType;

use super::build::{
    Planned, Planner, carrier_cast_target, internal, is_binary_safe_string, unsupported,
};
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
        "{what} (Python table UDFs run from P3-T014 PR-26c)"
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

/// The Spark types of grouping keys.
fn key_types(keys: &[Arc<Expr>]) -> Result<Vec<DataType>> {
    keys.iter()
        .map(|k| {
            k.resolved_type()
                .ok_or_else(|| internal("a Python UDF grouping key without a type"))
        })
        .collect()
}

/// The Arrow type a column of Spark type `spark`, stored as `stored` in two
/// union branches, has in both: theirs when they agree, else the carrier
/// the union operator uses (binary-safe for a `string` held as bytes).
fn carrier(spark: &DataType, stored: [&arrow_schema::DataType; 2]) -> arrow_schema::DataType {
    let [a, b] = stored;
    if a == b {
        a.clone()
    } else if is_binary_safe_string(spark, stored.into_iter()) {
        arrow_schema::DataType::BinaryView
    } else {
        spark.to_arrow()
    }
}

/// Column `index` of `planned` as `column.name`, cast where it is not
/// already stored as `to`.
fn carried(
    planned: &Planned,
    index: usize,
    column: BoundColumn,
    to: &arrow_schema::DataType,
) -> Result<ProjectItemPlan> {
    if planned.stored(index)? == to {
        return Ok(ProjectItemPlan::Column {
            index,
            name: Some(column.name),
        });
    }
    // The expression reads the input by attribute, under the input's id.
    let id = planned.column_attr(index)?;
    let target = carrier_cast_target(&column.data_type, to);
    let name = column.name.clone();
    let read = Expr::new(ExprNode::Resolved(ResolvedExpr::Column(BoundColumn {
        id,
        ..column
    })));
    Ok(ProjectItemPlan::Expr {
        name,
        expr: Arc::new(Expr::new(ExprNode::Cast {
            expr: Arc::new(read),
            to: target,
            mode: CastMode::Ansi,
        })),
    })
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

    /// `groupBy(keys).applyInPandas` / `applyInArrow`. Scalar Python calls
    /// in the keys run in a `PythonEval` below (Spark computes the keys in
    /// a `Project` under `FlatMapGroupsInPandas`, which `ExtractPythonUDFs`
    /// at v4.2.0 turns into an `ArrowEvalPython`); the frames keep the
    /// keys' own names and types.
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
        let (planned, computed) = self.extract_python(planned, keys)?;
        let (planned, key_columns) = self.with_columns(planned, &computed)?;
        let (planned, compared) = self.compared(planned, &computed, &key_columns)?;
        let columns = cols
            .iter()
            .map(|c| planned.column(c.id))
            .collect::<Result<Vec<_>>>()?;
        let sorted = self.clustered_and_sorted(planned, &compared, &[])?;
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
                keys: compared,
                udf: plan,
            },
            output,
            ("PythonGroupMap", explain),
        ))
    }

    /// `cogroup(...).applyInPandas` / `applyInArrow`; each side's scalar
    /// Python key calls run below its branch, as in [`Self::group_map`].
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
        let (l, left_computed) = self.extract_python(l, left_keys)?;
        let (l, lk) = self.with_columns(l, &left_computed)?;
        let (l, lc) = self.compared(l, &left_computed, &lk)?;
        let r = self.plan(right)?;
        let (r, right_computed) = self.extract_python(r, right_keys)?;
        let (r, rk) = self.with_columns(r, &right_computed)?;
        let (r, rc) = self.compared(r, &right_computed, &rk)?;
        let k = lk.len();
        // One layout for both branches: keys, side, left, right, then (for
        // collated keys) the keys compared.
        let width = k + 1 + left_cols.len() + right_cols.len();
        let keyed = lc != lk || rc != rk;
        let total = if keyed { width + k } else { width };
        let layout: Vec<AttrId> = (0..total).map(|_| self.fresh_id()).collect();
        let nullable = vec![true; total];
        // The two branches meet in one Arrow layout per column, whatever each
        // source stores (`Utf8` beside the `Utf8View` of a null literal).
        let left_types = key_types(left_keys)?;
        let right_types = key_types(right_keys)?;
        let key_carriers = left_types
            .iter()
            .zip(&right_types)
            .zip(lk.iter().zip(&rk))
            .map(|((lt, rt), (&li, &ri))| {
                if lt != rt {
                    return Err(unsupported("a cogroup on keys of different types"));
                }
                Ok(carrier(lt, [l.stored(li)?, r.stored(ri)?]))
            })
            .collect::<Result<Vec<_>>>()?;
        let data_carriers = [(&l, &left_cols), (&r, &right_cols)]
            .into_iter()
            .map(|(planned, cols)| {
                cols.iter()
                    .map(|c| {
                        let stored = planned.stored(planned.column(c.id)?)?;
                        Ok(carrier(&c.data_type, [stored, &c.data_type.to_arrow()]))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut branches = Vec::with_capacity(2);
        for (side, (planned, keys, compared)) in
            [(l, &lk, &lc), (r, &rk, &rc)].into_iter().enumerate()
        {
            let mut items = Vec::with_capacity(total);
            let types = if side == 0 { &left_types } else { &right_types };
            for (i, ((&c, data_type), to)) in keys.iter().zip(types).zip(&key_carriers).enumerate()
            {
                let column = BoundColumn {
                    id: planned.column_attr(c)?,
                    name: format!("_k{i}"),
                    data_type: data_type.clone(),
                    nullable: true,
                };
                items.push(carried(&planned, c, column, to)?);
            }
            items.push(ProjectItemPlan::Expr {
                name: "_side".to_owned(),
                expr: literal(krishiv_plan::Literal::Integer(i32::from(side == 1))),
            });
            for (own, ((cols, prefix), carriers)) in [(&left_cols, "_l"), (&right_cols, "_r")]
                .into_iter()
                .zip(&data_carriers)
                .enumerate()
            {
                for (i, (c, to)) in cols.iter().zip(carriers).enumerate() {
                    let name = format!("{prefix}{i}");
                    if own == side {
                        let column = BoundColumn { name, ..c.clone() };
                        items.push(carried(&planned, planned.column(c.id)?, column, to)?);
                    } else {
                        let null_type = carrier_cast_target(&c.data_type, to);
                        items.push(ProjectItemPlan::Expr {
                            name,
                            expr: literal(krishiv_plan::Literal::Null(Some(null_type))),
                        });
                    }
                }
            }
            if keyed {
                for (i, &c) in compared.iter().enumerate() {
                    items.push(ProjectItemPlan::Column {
                        index: c,
                        name: Some(format!("_c{i}")),
                    });
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
        let compared: Vec<usize> = if keyed {
            (width..total).collect()
        } else {
            key_columns.clone()
        };
        let sorted = self.clustered_and_sorted(union, &compared, &[k])?;
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
                keys: compared,
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
        for call in &calls {
            let f = function_of(call)?;
            refuse_call_syntax(f)?;
            if f.args.iter().any(|a| super::build::has_subquery(a)) {
                return Err(unsupported("a subquery in the arguments of a Python UDF"));
            }
        }
        // Scalar Python calls in the keys and the calls' arguments run below
        // (Spark's `ExtractGroupingPythonUDFFromAggregate` and
        // `ExtractPythonUDFs` at v4.2.0); the node reads their columns.
        let mut wanted: Vec<Arc<Expr>> = keys.to_vec();
        for call in &calls {
            wanted.extend(function_of(call)?.args.iter().cloned());
        }
        let planned = self.plan(input)?;
        let (planned, wanted) = self.extract_python(planned, &wanted)?;
        let (key_exprs, mut arg_exprs) = wanted
            .split_at_checked(keys.len())
            .ok_or_else(|| internal("Python aggregate key arity"))?;
        let (planned, key_columns) = self.with_columns(planned, key_exprs)?;
        // A collated key groups by its comparison key; each group shows its
        // first row's key (Spark's `GroupedIterator` key at v4.2.0). The
        // collation rewrite leaves this aggregate alone for that.
        let (planned, compared) = self.compared(planned, key_exprs, &key_columns)?;
        let prepared = self.prepare(&[])?;
        // Each call as the node runs it: over the arguments' columns.
        let mut run = Vec::with_capacity(calls.len());
        for call in &calls {
            let f = function_of(call)?;
            let (args, rest) = arg_exprs
                .split_at_checked(f.args.len())
                .ok_or_else(|| internal("Python aggregate argument arity"))?;
            arg_exprs = rest;
            let refs: Vec<&Expr> = args.iter().map(AsRef::as_ref).collect();
            compile_all(
                &refs,
                &planned.layout,
                &planned.schema,
                &self.values(&prepared),
            )?;
            run.push(Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    krishiv_plan::ResolvedFunction {
                        args: args.to_vec(),
                        ..f.clone()
                    },
                )))
                .with_origin(call.origin),
            ));
        }
        let sorted = self.clustered_and_sorted(planned, &compared, &[])?;
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
        for (i, (call, ran)) in calls.iter().zip(&run).enumerate() {
            let f = function_of(call)?;
            let column = BoundColumn {
                id: self.fresh_id(),
                name: format!("pythonUDAF{i}"),
                data_type: f.data_type.clone(),
                nullable: true,
            };
            fields.push(Field::new(&column.name, f.data_type.to_arrow(), true));
            layout.push(column.id);
            plans.push(PythonCallPlan {
                call: Arc::clone(ran),
                output: column.id,
            });
            columns.push((Arc::clone(call), column));
        }
        let detail = run
            .iter()
            .zip(&columns)
            .map(|(c, (_, col))| format!("{c} AS {}", col.name))
            .collect::<Vec<_>>()
            .join(", ");
        let input_layout = sorted.layout.clone();
        let key_values = key_columns.clone();
        let schema = Arc::new(Schema::new(fields));
        let node = sorted.wrap(
            |input| PhysNode::PythonAggregate {
                input,
                layout: input_layout,
                keys: compared,
                key_values,
                calls: plans,
                window: None,
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
        // Scalar Python calls over the results run above (Spark's
        // `ExtractPythonUDFFromAggregate` at v4.2.0).
        let (node, exprs) = self.extract_python(
            Planned {
                placement: Placement::Unknown,
                ordering: Vec::new(),
                sites: Vec::new(),
                ..node
            },
            &exprs,
        )?;
        let mut out = self.project(node, &exprs)?;
        out.sites.clear();
        Ok(out)
    }

    /// A window's Python aggregate calls `exprs` (aliases of windows over
    /// grouped-aggregate or window Python UDFs; the caller planned the
    /// window's other calls below) over `input`: the window's placement and
    /// sort, each bounded frame's `count(*)` columns, then one
    /// `PythonAggregate` node in window form whose output is every input
    /// row with one value per call. Scalar Python calls in the partitioning,
    /// ordering and arguments run below (Spark's `ExtractPythonUDFs`).
    pub(crate) fn python_window(
        &mut self,
        input: Planned,
        exprs: &[Arc<Expr>],
        partition_by: &[Arc<Expr>],
        order_by: &[SortOrder],
    ) -> Result<Planned> {
        let mut calls = Vec::with_capacity(exprs.len());
        for e in exprs {
            let c = super::window::call(e)?;
            let FunctionId::PythonUdf { eval_type, .. } = &c.f.id else {
                return Err(unsupported(
                    "a Python aggregate UDF inside another expression over a window",
                ));
            };
            // Spark runs a grouped-aggregate UDF over a window as its window
            // kind (`ArrowWindowPythonExec`'s eval type, v4.2.0).
            let window_type = match eval_type {
                PythonEvalType::GroupedAggPandas | PythonEvalType::WindowAggPandas => {
                    PythonEvalType::WindowAggPandas
                }
                PythonEvalType::GroupedAggArrow | PythonEvalType::WindowAggArrow => {
                    PythonEvalType::WindowAggArrow
                }
                other => {
                    return Err(unsupported(&format!(
                        "a Python UDF of eval type {} over a window",
                        other.code()
                    )));
                }
            };
            refuse_call_syntax(c.f)?;
            if c.f.args.iter().any(|a| super::build::has_subquery(a)) {
                return Err(unsupported("a subquery in the arguments of a Python UDF"));
            }
            calls.push((c, window_type));
        }
        // Scalar Python calls below; the window sorts by the columns of the
        // partitioning and ordering.
        let np = partition_by.len();
        let no = order_by.len();
        let mut wanted: Vec<Arc<Expr>> = partition_by.to_vec();
        wanted.extend(order_by.iter().map(|o| Arc::clone(&o.expr)));
        for (c, _) in &calls {
            wanted.extend(c.f.args.iter().cloned());
        }
        let (planned, wanted) = self.extract_python(input, &wanted)?;
        let (planned, columns) = self.with_columns(planned, &wanted[..np + no])?;
        // A collated partitioning key is compared by its collation, as the
        // window operator compares it.
        let (planned, compared) = self.compared(planned, &wanted[..np], &columns[..np])?;
        let column_expr = |planned: &Planned, at: usize| -> Result<Arc<Expr>> {
            let field = planned.schema.field(at);
            Ok(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Column(BoundColumn {
                    id: planned.column_attr(at)?,
                    name: field.name().clone(),
                    data_type: krishiv_types::DataType::from_arrow_field(field)
                        .map_err(|e| internal(&format!("a window key column: {e}")))?,
                    nullable: field.is_nullable(),
                }),
            ))))
        };
        let partition: Vec<Arc<Expr>> = compared
            .iter()
            .map(|&at| column_expr(&planned, at))
            .collect::<Result<_>>()?;
        let order: Vec<SortOrder> = order_by
            .iter()
            .zip(&columns[np..np + no])
            .map(|(o, &at)| {
                Ok(SortOrder {
                    expr: column_expr(&planned, at)?,
                    direction: o.direction,
                    nulls: o.nulls,
                })
            })
            .collect::<Result<_>>()?;
        let partition_attrs: Vec<AttrId> = compared
            .iter()
            .map(|&at| planned.column_attr(at))
            .collect::<Result<_>>()?;
        // Each bounded frame's end (`count(*)` from the partition's start to
        // the frame's end) and width (`count(*)` over it): a frame is a run
        // of rows in the window's order, so its start is the end less the
        // width, for an empty frame too (Spark's frames give the same rows:
        // `WindowFunctionFrame.currentLowerBound`/`currentUpperBound`).
        let mut counts: Vec<(crate::window::WindowFrame, BoundColumn)> = Vec::new();
        let mut bounds: Vec<Option<(BoundColumn, BoundColumn)>> = Vec::with_capacity(calls.len());
        for (c, _) in &calls {
            let frame = c.frame;
            if frame.lower == crate::window::FrameBound::UnboundedPreceding
                && frame.upper == crate::window::FrameBound::UnboundedFollowing
            {
                bounds.push(None);
                continue;
            }
            // Pairs of (end, width): a frame's own entry is the width's.
            let seen = counts
                .chunks(2)
                .find(|pair| pair.get(1).is_some_and(|(f, _)| *f == frame));
            let (end, width) = match seen {
                Some([(_, end), (_, width)]) => (end.clone(), width.clone()),
                _ => {
                    let n = counts.len() / 2;
                    let end = BoundColumn {
                        id: self.fresh_id(),
                        name: format!("_frame_end{n}"),
                        data_type: krishiv_types::DataType::Long,
                        nullable: false,
                    };
                    let width = BoundColumn {
                        id: self.fresh_id(),
                        name: format!("_frame_rows{n}"),
                        data_type: krishiv_types::DataType::Long,
                        nullable: false,
                    };
                    counts.push((
                        crate::window::WindowFrame {
                            units: frame.units,
                            lower: crate::window::FrameBound::UnboundedPreceding,
                            upper: frame.upper,
                        },
                        end.clone(),
                    ));
                    counts.push((frame, width.clone()));
                    (end, width)
                }
            };
            bounds.push(Some((end, width)));
        }
        let sorted = self.window_with(planned, &[], &partition, &order, None, &counts)?;
        let keys: Vec<usize> = partition_attrs
            .iter()
            .map(|&a| sorted.column(a))
            .collect::<Result<_>>()?;
        let prepared = self.prepare(&[])?;
        let mut arg_exprs = &wanted[np + no..];
        let mut layout = sorted.layout.clone();
        let mut fields: Vec<Field> = sorted
            .schema
            .fields()
            .iter()
            .map(|f| (**f).clone())
            .collect();
        let mut plans = Vec::with_capacity(calls.len());
        let mut details = Vec::with_capacity(calls.len());
        for (i, ((c, window_type), bound)) in calls.iter().zip(&bounds).enumerate() {
            let (args, rest) = arg_exprs
                .split_at_checked(c.f.args.len())
                .ok_or_else(|| internal("Python window argument arity"))?;
            arg_exprs = rest;
            let mut run_args = Vec::with_capacity(args.len() + 2);
            if let Some((end, width)) = bound {
                let read = |b: &BoundColumn| {
                    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                        b.clone(),
                    ))))
                };
                let start = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id: FunctionId::Builtin(Builtin::Subtract),
                        kind: FunctionKind::Scalar,
                        args: vec![read(end), read(width)],
                        distinct: false,
                        filter: None,
                        within_group: Vec::new(),
                        data_type: krishiv_types::DataType::Long,
                        nullable: false,
                        deterministic: true,
                    },
                ))));
                // Row indexes as Spark sends them: `INT` (a partition past
                // 2^31 rows fails the cast rather than wrap).
                for e in [start, read(end)] {
                    run_args.push(Arc::new(Expr::new(ExprNode::Cast {
                        expr: e,
                        to: krishiv_types::DataType::Integer,
                        mode: CastMode::Ansi,
                    })));
                }
            }
            run_args.extend(args.iter().cloned());
            let FunctionId::PythonUdf {
                name,
                version,
                thread_safe,
                ..
            } = &c.f.id
            else {
                return Err(internal("a Python window call is not a Python UDF"));
            };
            let run = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id: FunctionId::PythonUdf {
                        name: name.clone(),
                        version: *version,
                        eval_type: *window_type,
                        thread_safe: *thread_safe,
                    },
                    args: run_args,
                    ..c.f.clone()
                },
            ))));
            let refs: Vec<&Expr> = match &run.node {
                ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                    f.args.iter().map(AsRef::as_ref).collect()
                }
                _ => Vec::new(),
            };
            compile_all(
                &refs,
                &sorted.layout,
                &sorted.schema,
                &self.values(&prepared),
            )?;
            details.push(format!("{} AS {}", e_name(c.f), c.output.name));
            // Named as the node's instantiation names it (the plan reads it
            // by attribute).
            fields.push(Field::new(
                format!("pythonUDAF{i}"),
                c.output.data_type.to_arrow(),
                true,
            ));
            layout.push(c.output.id);
            plans.push(PythonCallPlan {
                call: run,
                output: c.output.id,
            });
        }
        let detail = format!(
            "[{}], partition=[{}]",
            details.join(", "),
            key_names(&sorted, &keys)
        );
        let input_layout = sorted.layout.clone();
        let bounded = bounds.iter().map(Option::is_some).collect();
        let mut out = sorted.wrap(
            |input| PhysNode::PythonAggregate {
                input,
                layout: input_layout,
                keys,
                key_values: Vec::new(),
                calls: plans,
                window: Some(bounded),
            },
            Arc::new(Schema::new(fields)),
            layout,
            ("PythonWindow", detail),
        );
        out.ordering = Vec::new();
        out.placement = Placement::Unknown;
        out.sites.clear();
        Ok(out)
    }

    /// The columns a relation's grouping `keys` (at `columns` of `input`)
    /// are compared by: a collated key's comparison key, appended
    /// (`krishiv_optimizer::collation_key`, the key the aggregates, sorts
    /// and exchanges of collated values use), else the key itself. Spark
    /// groups and cogroups by the collation (`GroupedIterator`'s and
    /// `CoGroupedIterator`'s orderings at v4.2.0); the frames keep each
    /// row's own key values.
    fn compared(
        &mut self,
        input: Planned,
        keys: &[Arc<Expr>],
        columns: &[usize],
    ) -> Result<(Planned, Vec<usize>)> {
        let mut exprs = Vec::with_capacity(keys.len());
        let mut changed = false;
        for (key, &at) in keys.iter().zip(columns) {
            let compared = krishiv_optimizer::collation_key(key)?;
            if Arc::ptr_eq(&compared, key) {
                // The key's column as it is (a computed key reads its own).
                let column = input.schema.field(at).clone();
                exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::Column(BoundColumn {
                        id: input.column_attr(at)?,
                        name: column.name().clone(),
                        data_type: key
                            .resolved_type()
                            .ok_or_else(|| internal("a Python UDF grouping key without a type"))?,
                        nullable: column.is_nullable(),
                    }),
                ))));
            } else {
                changed = true;
                exprs.push(compared);
            }
        }
        if !changed {
            return Ok((input, columns.to_vec()));
        }
        self.with_columns(input, &exprs)
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

/// `name#version` of a Python UDF call.
fn e_name(f: &ResolvedFunction) -> String {
    match &f.id {
        FunctionId::PythonUdf { name, version, .. } => format!("{name}#{version}"),
        other => other.name().to_owned(),
    }
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

/// The function `call` is.
fn function_of(call: &Expr) -> Result<&krishiv_plan::ResolvedFunction> {
    match &call.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => Ok(f),
        _ => Err(internal("a Python UDAF call is not a function")),
    }
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
