// SPDX-License-Identifier: Apache-2.0
//! `WHERE` / `HAVING`: boolean mask, then Arrow's filter kernel.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_types::batch::Batch;

use super::expr::Predicate;
use super::{PartitionState, local_mut};
use crate::context::TaskContext;
use crate::error::Result;
use crate::operator::{LocalState, Operator, OperatorOutput};

/// Keeps the rows for which the predicate is true (a null does not select
/// its row). Each surviving row keeps its weight; a batch with no
/// surviving row produces no output. A mask that keeps every row shares the
/// input's buffers. A string or binary view column is compacted when its
/// kept views use under a sixteenth of the data buffers they share with the
/// input, or under half when the filter dropped most of its bytes (kept
/// views use under a quarter of the input's): otherwise a few rows kept
/// from each batch of a selective filter (or of a scan that filtered rows
/// itself) would hold the scan's whole decoded pages (a collected result, a
/// write queue), unseen by the memory pool. A filter keeping most bytes of
/// a batch that shares a page with its neighbours shares the buffers:
/// copying those would cost it and save little.
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

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(PartitionState {
            partition: ctx.partition(),
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        if input.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let partition = local_mut::<PartitionState>(local, "Filter")?.partition;
        let mask = self.predicate.evaluate_in(input.data(), partition)?;
        let out = input.filter(&mask)?;
        if out.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let out = if out.num_rows() < input.num_rows() {
            compact_sparse_views(input, out)?
        } else {
            out
        };
        Ok(OperatorOutput::NeedMoreInput(Some(out)))
    }
}

/// `batch` (the filtered rows of `input`) with each top-level view column
/// compacted (`gc`) whose data buffers hold more than 16 times the bytes
/// its views use, or more than twice when its views use under a quarter of
/// the bytes `input`'s views of the same column use.
fn compact_sparse_views(input: &Batch, batch: Batch) -> Result<Batch> {
    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
    use arrow_schema::DataType;
    fn sparse(held: usize, used: usize, input_used: impl FnOnce() -> usize) -> bool {
        held > used.saturating_mul(16)
            || (held > used.saturating_mul(2) && used.saturating_mul(4) < input_used())
    }
    let data = batch.data();
    let compacted = |(c, i): (&ArrayRef, &ArrayRef)| -> Option<ArrayRef> {
        match c.data_type() {
            DataType::Utf8View => {
                let a = c.as_string_view();
                let held = a.data_buffers().iter().map(arrow_buffer::Buffer::len).sum();
                let input_used = || i.as_string_view().total_buffer_bytes_used();
                sparse(held, a.total_buffer_bytes_used(), input_used)
                    .then(|| Arc::new(a.gc()) as ArrayRef)
            }
            DataType::BinaryView => {
                let a = c.as_binary_view();
                let held = a.data_buffers().iter().map(arrow_buffer::Buffer::len).sum();
                let input_used = || i.as_binary_view().total_buffer_bytes_used();
                sparse(held, a.total_buffer_bytes_used(), input_used)
                    .then(|| Arc::new(a.gc()) as ArrayRef)
            }
            _ => None,
        }
    };
    let columns: Vec<Option<ArrayRef>> = data
        .columns()
        .iter()
        .zip(input.data().columns())
        .map(compacted)
        .collect();
    if columns.iter().all(Option::is_none) {
        return Ok(batch);
    }
    let columns = columns
        .into_iter()
        .zip(data.columns())
        .map(|(new, old)| new.unwrap_or_else(|| Arc::clone(old)))
        .collect();
    let compacted = RecordBatch::try_new_with_options(
        data.schema(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(data.num_rows())),
    )
    .map_err(|e| crate::error::internal(e.to_string()))?;
    Batch::with_weights(compacted, batch.weights().clone())
        .map_err(|e| crate::error::internal(e.to_string()))
}
