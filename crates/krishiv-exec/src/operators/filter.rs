// SPDX-License-Identifier: Apache-2.0
//! `WHERE` / `HAVING`: boolean mask, then Arrow's filter kernel.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_types::batch::Batch;

use arrow_array::{Array, BooleanArray};

use super::expr::Predicate;
use super::{RowStage, RowStageSettings, local_mut};
use crate::context::TaskContext;
use crate::error::Result;
use crate::operator::{LocalState, Operator, OperatorOutput};
use krishiv_io::ScanFilter;

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
    /// Runtime filters whose row stage this filter runs instead of the scan
    /// below it (design P2-E3 U1), with the stage's settings.
    runtime: Vec<ScanFilter>,
    settings: RowStageSettings,
}

/// A driver's state: its partition and the row stage over this filter's input.
struct FilterLocal {
    partition: u32,
    rows: Option<RowStage>,
}

impl Filter {
    /// A filter over batches of `schema`.
    pub fn new(schema: SchemaRef, predicate: Arc<dyn Predicate>) -> Arc<Self> {
        Self::with_row_stage(schema, predicate, Vec::new(), RowStageSettings::default())
    }

    /// A filter that also runs the row stage of `runtime` (filters on columns
    /// of `schema`): the rows it drops leave in the same copy as the
    /// predicate's.
    pub fn with_row_stage(
        schema: SchemaRef,
        predicate: Arc<dyn Predicate>,
        runtime: Vec<ScanFilter>,
        settings: RowStageSettings,
    ) -> Arc<Self> {
        Arc::new(Self {
            schema,
            predicate,
            runtime,
            settings,
        })
    }
}

/// The rows both masks keep (a null in `predicate` keeps nothing).
fn both(predicate: &BooleanArray, rows: &BooleanArray) -> BooleanArray {
    let mut keep = predicate.values().clone();
    if let Some(nulls) = predicate.nulls() {
        keep = &keep & nulls.inner();
    }
    BooleanArray::new(&keep & rows.values(), None)
}

impl Operator for Filter {
    fn name(&self) -> &'static str {
        "Filter"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(FilterLocal {
            partition: ctx.partition(),
            rows: (!self.runtime.is_empty())
                .then(|| RowStage::new(&self.runtime, &self.schema, self.settings)),
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        if input.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let state = local_mut::<FilterLocal>(local, "Filter")?;
        let mask = self.predicate.evaluate_in(input.data(), state.partition)?;
        let mask = match state.rows.as_mut() {
            Some(rows) => match rows.mask(input.data())? {
                Some(dropped) => both(&mask, &dropped),
                None => mask,
            },
            None => mask,
        };
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
