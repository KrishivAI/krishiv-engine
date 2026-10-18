// SPDX-License-Identifier: Apache-2.0
//! `WHERE` / `HAVING`: boolean mask, then Arrow's filter kernel.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_types::batch::Batch;

use super::expr::Predicate;
use super::no_state;
use crate::context::TaskContext;
use crate::error::Result;
use crate::operator::{LocalState, Operator, OperatorOutput};

/// Keeps the rows for which the predicate is true (a null does not select
/// its row). Each surviving row keeps its weight; a batch with no
/// surviving row produces no output. A mask that keeps every row shares the
/// input's buffers.
#[derive(Debug)]
pub struct Filter {
    schema: SchemaRef,
    predicate: Arc<dyn Predicate>,
}

impl Filter {
    /// A filter over batches of `schema`.
    pub fn new(schema: SchemaRef, predicate: Arc<dyn Predicate>) -> Arc<Self> {
        Arc::new(Self { schema, predicate })
    }
}

impl Operator for Filter {
    fn name(&self) -> &'static str {
        "Filter"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(no_state())
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        if input.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let mask = self.predicate.evaluate(input.data())?;
        let out = input.filter(&mask)?;
        Ok(OperatorOutput::NeedMoreInput(
            (out.num_rows() > 0).then_some(out),
        ))
    }
}
