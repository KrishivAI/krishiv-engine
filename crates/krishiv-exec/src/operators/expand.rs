// SPDX-License-Identifier: Apache-2.0
//! Each row once per projection: the first step of `ROLLUP`, `CUBE` and
//! `GROUPING SETS` (Spark's `Expand`).

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, new_null_array};
use arrow_schema::{DataType, SchemaRef};
use krishiv_types::batch::Batch;

use super::{PartitionState, Project, ScalarExpr, local_mut};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{LocalState, Operator, OperatorOutput};

/// Emits, for every input batch, one batch per projection: the batch
/// projected by it. The projections share one output schema. Rows keep their
/// weights in every copy. Output is produced one projection per
/// [`Operator::execute`] call (`HaveMoreOutput` until the last), so only
/// one copy of an input batch exists at a time, and the plain columns of
/// each copy share the input's buffers.
#[derive(Debug)]
pub struct Expand {
    projects: Vec<Arc<Project>>,
    schema: SchemaRef,
}

impl Expand {
    /// Expands batches by `projects`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if there is no projection, or two of them output
    /// different columns.
    pub fn try_new(projects: Vec<Arc<Project>>) -> Result<Arc<Self>> {
        let first = projects
            .first()
            .ok_or_else(|| internal("an expansion without projections".to_owned()))?;
        let schema = first.schema();
        for p in &projects {
            let (a, b) = (p.schema(), &schema);
            let same = a.fields().len() == b.fields().len()
                && a.fields().iter().zip(b.fields()).all(|(x, y)| {
                    x.name() == y.name()
                        && x.data_type() == y.data_type()
                        && x.is_nullable() == y.is_nullable()
                });
            if !same {
                return Err(internal(format!(
                    "the projections of an expansion differ: {:?} and {:?}",
                    a.fields(),
                    b.fields()
                )));
            }
        }
        Ok(Arc::new(Self { projects, schema }))
    }
}

/// The next projection to emit for the current input batch, and the
/// driver's partition: the projections evaluate row expressions, which draw
/// the partition's stream.
#[derive(Debug, Clone, Copy)]
struct State {
    next: usize,
    partition: u32,
}

impl Operator for Expand {
    fn name(&self) -> &'static str {
        "Expand"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(State {
            next: 0,
            partition: ctx.partition(),
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        let state = local_mut::<State>(local, "Expand")?;
        if input.num_rows() == 0 {
            state.next = 0;
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let project = self
            .projects
            .get(state.next)
            .ok_or_else(|| internal("an expansion past its last projection".to_owned()))?;
        let mut partition: LocalState = Box::new(PartitionState {
            partition: state.partition,
        });
        let OperatorOutput::NeedMoreInput(Some(out)) = project.execute(input, &mut partition)?
        else {
            return Err(internal("a projection did not output a batch".to_owned()));
        };
        state.next += 1;
        if state.next < self.projects.len() {
            return Ok(OperatorOutput::HaveMoreOutput(out));
        }
        state.next = 0;
        Ok(OperatorOutput::NeedMoreInput(Some(out)))
    }
}

/// A column of nulls, in the Arrow type of an input column. An expansion
/// nulls a grouping column in the carrier its other copies have: strings
/// reach an operator as `Utf8` from a registered table and as `Utf8View`
/// from a file.
#[derive(Debug)]
pub struct NullOf {
    index: usize,
    data_type: DataType,
}

impl NullOf {
    /// Nulls of input column `index` of batches of `input`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the column is not in `input`.
    pub fn new(input: &SchemaRef, index: usize) -> Result<Self> {
        let field = input
            .fields()
            .get(index)
            .ok_or_else(|| internal(format!("null-of index {index} out of range")))?;
        Ok(Self {
            index,
            data_type: field.data_type().clone(),
        })
    }
}

impl ScalarExpr for NullOf {
    fn data_type(&self) -> DataType {
        self.data_type.clone()
    }

    fn nullable(&self) -> bool {
        true
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let column = batch
            .columns()
            .get(self.index)
            .ok_or_else(|| internal(format!("null-of index {} out of range", self.index)))?;
        Ok(new_null_array(column.data_type(), batch.num_rows()))
    }
}
