// SPDX-License-Identifier: Apache-2.0
//! `UNION` / `UNION ALL`: the per-branch part.
//!
//! # Who owns what (ADR 0006 deferral)
//! - **Fan-in of `UNION ALL`** (several child pipelines feeding the same
//!   downstream operators and sink, DuckDB-style) is a plan-building matter
//!   and ships with the physical plan (P2-T074), not here. `UNION ALL` is
//!   not a breaker and holds no state.
//! - **Per branch**, [`UnionAlign`] (this task) maps a branch's batches to
//!   the union's output schema: the first branch's column names, nullability
//!   widened over all branches, rows and weights untouched. Type coercion
//!   (`int` with `bigint`, decimals, ...) is the analyzer's: it inserts casts
//!   into the branches, so a type mismatch here is an engine bug.
//! - **`UNION` (distinct)** is `UNION ALL` followed by a distinct aggregate
//!   over all columns. It is a pipeline breaker and its state lives in the
//!   hash aggregate (executor-owned breaker state and spill), so it belongs
//!   to the aggregate task (P2-T052/T053), not to a second implementation
//!   here.

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema, SchemaRef};
use krishiv_types::batch::Batch;

use super::no_state;
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{LocalState, Operator, OperatorOutput};

/// Re-labels one union branch's batches with the union's output schema.
/// Zero-copy; weights pass through.
#[derive(Debug)]
pub struct UnionAlign {
    output: SchemaRef,
}

impl UnionAlign {
    /// The output schema of a union of `branches`: names from the first,
    /// a column nullable if it is nullable in any branch.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if there is no branch, or branches differ in column
    /// count or in a column's data type (the analyzer coerces types first).
    pub fn unify(branches: &[SchemaRef]) -> Result<SchemaRef> {
        let (first, rest) = branches
            .split_first()
            .ok_or_else(|| internal("a union has no branch".to_owned()))?;
        let mut fields: Vec<Field> = first.fields().iter().map(|f| f.as_ref().clone()).collect();
        for branch in rest {
            if branch.fields().len() != fields.len() {
                return Err(internal(format!(
                    "union branches have {} and {} columns",
                    fields.len(),
                    branch.fields().len()
                )));
            }
            for (field, other) in fields.iter_mut().zip(branch.fields()) {
                if field.data_type() != other.data_type() {
                    return Err(internal(format!(
                        "union column {} has types {} and {}",
                        field.name(),
                        field.data_type(),
                        other.data_type()
                    )));
                }
                if other.is_nullable() {
                    *field = field.clone().with_nullable(true);
                }
            }
        }
        Ok(Arc::new(Schema::new_with_metadata(
            fields,
            first.metadata().clone(),
        )))
    }

    /// Aligns batches of `input` to `output` (from [`unify`](Self::unify)).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the columns differ in count or type, or `input`
    /// has a nullable column that `output` declares non-null.
    pub fn try_new(input: &SchemaRef, output: SchemaRef) -> Result<Arc<Self>> {
        let unified = Self::unify(&[Arc::clone(&output), Arc::clone(input)])?;
        if unified.fields() != output.fields() {
            return Err(internal(
                "a union output column is non-null but its branch is nullable".to_owned(),
            ));
        }
        Ok(Arc::new(Self { output }))
    }
}

impl Operator for UnionAlign {
    fn name(&self) -> &'static str {
        "UnionAlign"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output)
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(no_state())
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        let rows = input.num_rows();
        if rows == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        let data = RecordBatch::try_new_with_options(
            Arc::clone(&self.output),
            input.data().columns().to_vec(),
            &options,
        )
        .map_err(|e| internal(format!("union branch does not fit the union schema: {e}")))?;
        let out = Batch::with_weights(data, input.weights().clone())?;
        Ok(OperatorOutput::NeedMoreInput(Some(out)))
    }
}
