// SPDX-License-Identifier: Apache-2.0
//! `SELECT` lists: column selection, reordering, renaming and scalar
//! expressions.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema, SchemaRef};
use krishiv_types::batch::Batch;

use super::expr::ScalarExpr;
use super::no_state;
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{LocalState, Operator, OperatorOutput};

/// One output column of a [`Project`].
#[derive(Debug, Clone)]
pub enum ProjectItem {
    /// An input column, optionally renamed. Zero-copy.
    Column {
        /// Index into the input schema (may repeat).
        index: usize,
        /// The output name; the input's by default.
        name: Option<String>,
    },
    /// A computed column.
    Expr {
        /// The output name.
        name: String,
        /// The evaluator (see [`ScalarExpr`]).
        expr: Arc<dyn ScalarExpr>,
    },
}

impl ProjectItem {
    /// The input column `index` under its own name.
    pub fn column(index: usize) -> Self {
        Self::Column { index, name: None }
    }

    /// The input column `index` renamed to `name`.
    pub fn aliased(index: usize, name: impl Into<String>) -> Self {
        Self::Column {
            index,
            name: Some(name.into()),
        }
    }

    /// A computed column.
    pub fn expr(name: impl Into<String>, expr: Arc<dyn ScalarExpr>) -> Self {
        Self::Expr {
            name: name.into(),
            expr,
        }
    }
}

/// Computes a list of output columns from each input batch. Row count and
/// weights pass through unchanged (a projection neither adds nor removes
/// rows); plain columns share the input's buffers.
#[derive(Debug)]
pub struct Project {
    items: Vec<ProjectItem>,
    schema: SchemaRef,
}

impl Project {
    /// A projection of batches of schema `input`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a column index is out of range.
    pub fn try_new(input: &SchemaRef, items: Vec<ProjectItem>) -> Result<Arc<Self>> {
        let fields = items
            .iter()
            .map(|item| match item {
                ProjectItem::Column { index, name } => {
                    let field = input
                        .fields()
                        .get(*index)
                        .ok_or_else(|| internal(format!("project index {index} out of range")))?;
                    Ok(match name {
                        Some(n) => field.as_ref().clone().with_name(n),
                        None => field.as_ref().clone(),
                    })
                }
                ProjectItem::Expr { name, expr } => {
                    Ok(Field::new(name, expr.data_type(), expr.nullable()))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let schema = Arc::new(Schema::new_with_metadata(fields, input.metadata().clone()));
        Ok(Arc::new(Self { items, schema }))
    }

    /// As [`try_new`](Self::try_new), with output column `i` nullable if
    /// `nullable[i]` (the inputs of a union meet at the nullability of
    /// any of them).
    ///
    /// # Errors
    /// As [`try_new`](Self::try_new); `INTERNAL_ERROR` if `nullable` is
    /// not one flag per item.
    pub fn try_new_nullable(
        input: &SchemaRef,
        items: Vec<ProjectItem>,
        nullable: &[bool],
    ) -> Result<Arc<Self>> {
        if nullable.len() != items.len() {
            return Err(internal(
                "one nullability flag per projected column".to_owned(),
            ));
        }
        let p = Self::try_new(input, items)?;
        let fields: Vec<Field> = p
            .schema
            .fields()
            .iter()
            .zip(nullable)
            .map(|(f, n)| f.as_ref().clone().with_nullable(f.is_nullable() || *n))
            .collect();
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            p.schema.metadata().clone(),
        ));
        Ok(Arc::new(Self {
            items: p.items.clone(),
            schema,
        }))
    }
}

impl Operator for Project {
    fn name(&self) -> &'static str {
        "Project"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(no_state())
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        let rows = input.num_rows();
        if rows == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let data = input.data();
        let columns = self
            .items
            .iter()
            .map(|item| -> Result<ArrayRef> {
                match item {
                    ProjectItem::Column { index, .. } => data
                        .columns()
                        .get(*index)
                        .map(Arc::clone)
                        .ok_or_else(|| internal(format!("project index {index} out of range"))),
                    ProjectItem::Expr { expr, .. } => expr.evaluate(data),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        let out = RecordBatch::try_new_with_options(Arc::clone(&self.schema), columns, &options)
            .map_err(|e| internal(format!("project produced an invalid batch: {e}")))?;
        let out = Batch::with_weights(out, input.weights().clone())?;
        Ok(OperatorOutput::NeedMoreInput(Some(out)))
    }
}
