// SPDX-License-Identifier: Apache-2.0
//! Python table functions planned (design P3-T085 §3.1; Spark 4.2.0's
//! `FunctionTableSubqueryArgumentExpression.evaluable` and
//! `EvalPythonUDTFExec`): a [`PhysNode::PythonUdtf`] over
//! - one row, for a call in `FROM` with scalar arguments only;
//! - the `TABLE` argument's relation: with `PARTITION BY`, a hash exchange
//!   on the partitioning values and a local sort on them then the
//!   `ORDER BY` keys; with `WITH SINGLE PARTITION`, one partition sorted by
//!   the `ORDER BY` keys; else the relation as it is placed;
//! - the left side of a `LATERAL` call, whose columns each result row
//!   carries.
//!
//! The `TABLE` argument reaches the UDTF as one struct column per row: the
//! table's columns (or `analyze`'s `select`), then each `PARTITION BY` value
//! as a `partition_by_<i>` field the worker compares between rows to find a
//! partition's end and removes before `eval`. Spark projects a
//! `PARTITION BY` value only when it is not a column of the table; Krishiv
//! always does, which gives `eval` the same row (the field is removed) and
//! a fresh instance per partition value, as Spark's runs show.

use std::sync::Arc;

use arrow_schema::{Field, Schema};
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, LogicalPlan, PythonUdfKind, PythonUdfRef, ResolvedExpr,
    SortOrder, TableArg,
};

use super::build::{Planned, Planner, internal, one_row};
use super::explain::ExplainNode;
use super::properties::{Distribution, Placement};
use crate::error::Result;
use crate::operator::Source as _;
use crate::operators::SortKey;
use crate::physical::{
    PhysNode, PythonFrameSide, PythonOutputColumn, PythonRelationPlan, PythonUdtfPlan,
    PythonUdtfTable, sort_op,
};

/// An argument's value and name: a named argument is a resolved alias.
fn named(a: &Arc<Expr>) -> (Arc<Expr>, Option<String>) {
    match &a.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
            (Arc::clone(expr), Some(output.name.clone()))
        }
        _ => (Arc::clone(a), None),
    }
}

/// A column name for a struct field: an alias's or a column's name.
fn field_name(e: &Expr, i: usize) -> String {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.name.clone(),
        ExprNode::Resolved(ResolvedExpr::Column(c)) => c.name.clone(),
        _ => format!("col{}", i + 1),
    }
}

/// `e` without its alias.
fn unaliased(e: &Arc<Expr>) -> Arc<Expr> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
        _ => Arc::clone(e),
    }
}

impl Planner<'_> {
    /// A resolved `PythonTableFunction`.
    pub(crate) fn python_table_function(
        &mut self,
        args: &[Arc<Expr>],
        table_arg: Option<&TableArg>,
        udf: &PythonUdfRef,
        output: &[BoundColumn],
        lateral: Option<&Arc<LogicalPlan>>,
        analyzed: Option<&Arc<[u8]>>,
    ) -> Result<Planned> {
        if udf.eval_type.kind() != PythonUdfKind::Udtf {
            return Err(internal(&format!(
                "PythonTableFunction over a Python UDF of eval type {}",
                udf.eval_type.code()
            )));
        }
        let (planned, carried_cols) = match (lateral, table_arg) {
            (Some(l), None) => {
                let cols = l
                    .resolved_output()
                    .ok_or_else(|| internal("a lateral Python UDTF over an unresolved input"))?;
                (self.plan(l)?, cols)
            }
            (None, Some(t)) => (self.plan(&t.input)?, Vec::new()),
            (None, None) => (one_row(), Vec::new()),
            (Some(_), Some(_)) => {
                return Err(internal("a lateral Python UDTF with a TABLE argument"));
            }
        };
        let (planned, table) = match table_arg {
            Some(t) => {
                let (p, table) = self.udtf_table(planned, t)?;
                (p, Some(table))
            }
            None => (planned, None),
        };
        // The scalar arguments, evaluated over each input row.
        let total = args.len() + usize::from(table.is_some());
        let mut arg_names: Vec<Option<String>> = Vec::with_capacity(total);
        let mut values: Vec<Arc<Expr>> = Vec::with_capacity(args.len());
        let mut scalar = args.iter();
        for i in 0..total {
            if let (Some(t), Some(ta)) = (&table, table_arg)
                && t.position == i
            {
                arg_names.push(ta.name.clone());
                continue;
            }
            let a = scalar
                .next()
                .ok_or_else(|| internal("Python UDTF argument count"))?;
            let (value, name) = named(a);
            arg_names.push(name);
            values.push(value);
        }
        let (planned, columns) = self.with_columns(planned, &values)?;
        let mut side = PythonFrameSide {
            columns,
            names: (0..values.len()).map(|j| format!("_{j}")).collect(),
            types: Vec::with_capacity(values.len()),
        };
        for (v, &c) in values.iter().zip(&side.columns) {
            let ty = match v.resolved_type() {
                Some(t) => t,
                None => planned
                    .schema
                    .fields()
                    .get(c)
                    .and_then(|f| krishiv_types::DataType::from_arrow(f.data_type()).ok())
                    .ok_or_else(|| internal("a Python UDTF argument without a type"))?,
            };
            side.types.push(ty);
        }
        let carried = carried_cols
            .iter()
            .map(|c| planned.column(c.id))
            .collect::<Result<Vec<_>>>()?;
        let result: Vec<&BoundColumn> = carried_cols.iter().chain(output).collect();
        let plan = PythonRelationPlan {
            udf: udf.clone(),
            sides: vec![side],
            output: result
                .iter()
                .map(|c| PythonOutputColumn {
                    name: c.name.clone(),
                    data_type: c.data_type.clone(),
                    nullable: c.nullable,
                })
                .collect(),
        };
        let call = PythonUdtfPlan {
            arg_names,
            table,
            carried,
            analyzed: analyzed.map(|a| a.to_vec()),
        };
        let schema = Arc::new(Schema::new(
            result
                .iter()
                .map(|c| Field::new(&c.name, c.data_type.to_arrow(), c.nullable))
                .collect::<Vec<_>>(),
        ));
        let explain = format!(
            "{}#{}{}",
            udf.name,
            udf.version,
            if lateral.is_some() { ", lateral" } else { "" }
        );
        Ok(Planned {
            node: PhysNode::PythonUdtf {
                input: Box::new(planned.node),
                udf: plan,
                call,
            },
            schema,
            layout: result.iter().map(|c| c.id).collect(),
            explain: ExplainNode::new("PythonUdtf", explain, vec![planned.explain]),
            sites: Vec::new(),
            placement: Placement::Unknown,
            ordering: Vec::new(),
        })
    }

    /// The `TABLE` argument's row struct over `input`, with the
    /// partitioning and order Spark's `evaluable` gives it.
    fn udtf_table(&mut self, input: Planned, t: &TableArg) -> Result<(Planned, PythonUdtfTable)> {
        let fields: Vec<Arc<Expr>> = if t.select.is_empty() {
            t.input
                .resolved_output()
                .ok_or_else(|| internal("a TABLE argument over an unresolved input"))?
                .into_iter()
                .map(|c| Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c)))))
                .collect()
        } else {
            t.select.clone()
        };
        let mut names: Vec<String> = fields
            .iter()
            .enumerate()
            .map(|(i, e)| field_name(e, i))
            .collect();
        let mut exprs: Vec<Arc<Expr>> = fields.iter().map(unaliased).collect();
        let first_partition = exprs.len();
        for (i, p) in t.partition_by.iter().enumerate() {
            names.push(format!("partition_by_{i}"));
            exprs.push(Arc::clone(p));
        }
        let (planned, columns) = self.with_columns(input, &exprs)?;
        let keys: Vec<usize> = columns.get(first_partition..).unwrap_or_default().to_vec();
        let planned = if !keys.is_empty() {
            let attrs: Vec<AttrId> = keys
                .iter()
                .map(|&c| planned.column_attr(c))
                .collect::<Result<_>>()?;
            let planned = self.require(planned, Distribution::Clustered(&attrs))?;
            self.udtf_sort(planned, &keys, &t.order_by)?
        } else if t.single_partition {
            let planned = self.require(planned, Distribution::Single)?;
            if t.order_by.is_empty() {
                planned
            } else {
                self.udtf_sort(planned, &[], &t.order_by)?
            }
        } else {
            planned
        };
        let table = PythonUdtfTable {
            position: t.position,
            columns,
            names,
            partition_child_indexes: (first_partition..first_partition + t.partition_by.len())
                .collect(),
        };
        Ok((planned, table))
    }

    /// `input` sorted within its partitions by `keys` (ascending) then
    /// `order` (Spark's local `Sort` under the table argument).
    fn udtf_sort(
        &mut self,
        input: Planned,
        keys: &[usize],
        order: &[SortOrder],
    ) -> Result<Planned> {
        let (input, order_keys) = self.sort_keys(input, order)?;
        let sort: Vec<SortKey> = keys
            .iter()
            .map(|&c| SortKey::asc(c))
            .chain(order_keys)
            .collect();
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
