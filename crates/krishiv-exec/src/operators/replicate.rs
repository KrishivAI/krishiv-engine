// SPDX-License-Identifier: Apache-2.0
//! Each row `count` times: the last step of `INTERSECT ALL` and `EXCEPT
//! ALL` (Spark's `Generate(ReplicateRows)`).

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::Int64Type;
use arrow_array::{Array as _, Int64Array, UInt32Array};
use arrow_schema::SchemaRef;
use arrow_select::take::{take, take_record_batch};
use krishiv_types::batch::{Batch, Weights};

use super::expr::ScalarExpr;
use super::local_mut;
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{LocalState, Operator, OperatorOutput};

/// Repeats each input row as many times as its `count` (a `BIGINT`
/// expression) says; a row whose count is NULL or below 1 is dropped. A
/// row keeps its weight in every copy. Output batches hold at most
/// the query's `ExecConfig::batch_size` rows: a large count spreads
/// over several batches instead of one allocation.
#[derive(Debug)]
pub struct ReplicateRows {
    schema: SchemaRef,
    count: Arc<dyn ScalarExpr>,
}

impl ReplicateRows {
    /// Replicates batches of `schema` by `count`.
    pub fn new(schema: SchemaRef, count: Arc<dyn ScalarExpr>) -> Arc<Self> {
        Arc::new(Self { schema, count })
    }
}

/// Where the operator is in the current input batch.
struct Cursor {
    batch_size: usize,
    /// The driver's partition: `count` evaluates stateful functions in it.
    partition: u32,
    /// The input's counts, while some of its copies are still to emit.
    counts: Option<Int64Array>,
    /// The row being copied.
    row: usize,
    /// The copies of `row` emitted so far.
    emitted: u64,
}

impl Operator for ReplicateRows {
    fn name(&self) -> &'static str {
        "ReplicateRows"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(Cursor {
            batch_size: ctx.query().config().batch_size.max(1),
            partition: ctx.partition(),
            counts: None,
            row: 0,
            emitted: 0,
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        let cur = local_mut::<Cursor>(local, "ReplicateRows")?;
        let counts = match cur.counts.take() {
            Some(c) => c,
            None => {
                if input.num_rows() == 0 {
                    return Ok(OperatorOutput::NeedMoreInput(None));
                }
                let v = self.count.evaluate_in(input.data(), cur.partition)?;
                let c = v
                    .as_primitive_opt::<Int64Type>()
                    .ok_or_else(|| internal("a replication count is not BIGINT".to_owned()))?
                    .clone();
                cur.row = 0;
                cur.emitted = 0;
                c
            }
        };
        let rows = counts.len();
        let mut indices: Vec<u32> = Vec::new();
        while cur.row < rows && indices.len() < cur.batch_size {
            let want = if counts.is_null(cur.row) {
                0
            } else {
                u64::try_from(counts.value(cur.row)).unwrap_or(0)
            };
            let room = u64::try_from(cur.batch_size - indices.len()).unwrap_or(u64::MAX);
            let n = (want - cur.emitted.min(want)).min(room);
            let row = u32::try_from(cur.row)
                .map_err(|_| internal("a batch over 2^32 rows".to_owned()))?;
            // `n` is at most `room`, a `usize`.
            indices.extend(std::iter::repeat_n(row, usize::try_from(n).unwrap_or(0)));
            cur.emitted += n;
            if cur.emitted >= want {
                cur.row += 1;
                cur.emitted = 0;
            }
        }
        let out = if indices.is_empty() {
            None
        } else {
            Some(copies(input, &UInt32Array::from(indices))?)
        };
        match out {
            // A full batch with rows of this input left: same input again.
            Some(out) if cur.row < rows => {
                cur.counts = Some(counts);
                Ok(OperatorOutput::HaveMoreOutput(out))
            }
            out => Ok(OperatorOutput::NeedMoreInput(out)),
        }
    }
}

/// The rows of `input` at `indices`, with their weights.
fn copies(input: &Batch, indices: &UInt32Array) -> Result<Batch> {
    let arrow = |e: arrow_schema::ArrowError| internal(e.to_string());
    let data = take_record_batch(input.data(), indices).map_err(arrow)?;
    let weights = match input.weights() {
        Weights::Const(w) => Weights::Const(*w),
        Weights::Column(c) => Weights::Column(
            take(c, indices, None)
                .map_err(arrow)?
                .as_primitive()
                .clone(),
        ),
    };
    Batch::with_weights(data, weights)
}
