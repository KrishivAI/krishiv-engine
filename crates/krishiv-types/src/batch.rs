// SPDX-License-Identifier: Apache-2.0
//! The engine's unit of data: an Arrow [`RecordBatch`] plus per-row
//! [`Weights`] (ADR 0006, 0009, 0017).
//!
//! A [`Batch`] is plain Arrow. There is no selection vector and no parallel
//! vector-encoding enum: dictionary and run-end encodings are Arrow's own
//! array types. Filtering copies the selected rows into new arrays
//! (copy-on-filter); projecting and slicing share buffers.
//!
//! Weights make a batch a Z-set delta, so one type serves batch, streaming
//! and incremental view maintenance ([`ChangeBatch`]). A batch query is an
//! insert-only stream and carries [`Weights::Const`]`(1)`. Operators must
//! route that case to the same code path as "no weights" (see
//! [`Batch::is_insert_only`]); the operations here never turn a constant
//! weight into a column, except [`Batch::concat`] of batches whose constants
//! differ.
//!
//! ```
//! use std::sync::Arc;
//!
//! use arrow_array::{BooleanArray, Int64Array, RecordBatch};
//! use arrow_schema::{DataType, Field, Schema};
//! use krishiv_types::batch::{Batch, Weights};
//!
//! let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
//! let data = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 3]))])?;
//!
//! // A batch query's rows: weight 1 each, no weight column.
//! let batch = Batch::new(data.clone());
//! assert!(batch.is_insert_only());
//!
//! // A change stream: insert row 0, retract row 1, insert row 2 twice.
//! let change = Batch::with_weights(data, Weights::Column(Int64Array::from(vec![1, -1, 2])))?;
//! let kept = change.filter(&BooleanArray::from(vec![false, true, true]))?;
//! assert_eq!(kept.num_rows(), 2);
//! assert_eq!(kept.weights(), &Weights::Column(Int64Array::from(vec![-1, 2])));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch, RecordBatchOptions};
use arrow_schema::{ArrowError, SchemaRef};
use arrow_select::concat::concat_batches;
use arrow_select::filter::FilterBuilder;
use krishiv_common::error::KrishivError;

use crate::error::{self, Result};

/// The weight (multiplicity) of each row of a [`Batch`].
///
/// A positive weight `w` inserts the row `w` times; a negative one retracts
/// it. A weight of zero is allowed (the row has no effect) and is removed by
/// consolidation, not by the operations here.
///
/// The weight kind is known statically from the source and the plan:
/// operators never re-detect a constant inside a [`Weights::Column`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Weights {
    /// Every row has this weight. `Const(1)` is the "no weights" fast path
    /// that batch queries and insert-only streams use.
    Const(i64),
    /// One weight per row, for mixed-sign change streams. The array has the
    /// batch's row count and no nulls.
    Column(Int64Array),
}

/// A [`RecordBatch`] whose rows carry [`Weights`].
///
/// The row count is the [`RecordBatch`]'s, which Arrow keeps even for a batch
/// with no columns.
///
/// Invariants, upheld by every constructor and operation:
/// - a [`Weights::Column`] has exactly [`Self::num_rows`] values and no nulls.
#[derive(Clone, Debug, PartialEq)]
pub struct Batch {
    data: RecordBatch,
    weights: Weights,
}

/// The change batch of streaming and incremental view maintenance: the same
/// type as [`Batch`], so batch, streaming and IVM share one data path.
pub type ChangeBatch = Batch;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Batch>();
    assert_send_sync::<Weights>();
};

impl From<RecordBatch> for Batch {
    fn from(data: RecordBatch) -> Self {
        Self::new(data)
    }
}

impl Batch {
    /// An insert-only batch: every row has weight 1 ([`Weights::Const`]`(1)`).
    pub fn new(data: RecordBatch) -> Self {
        Self {
            data,
            weights: Weights::Const(1),
        }
    }

    /// A batch with explicit weights.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a [`Weights::Column`] does not have one value per
    /// row or contains a null.
    pub fn with_weights(data: RecordBatch, weights: Weights) -> Result<Self> {
        if let Weights::Column(column) = &weights {
            if column.len() != data.num_rows() {
                return Err(error::internal(format!(
                    "Batch weight column has {} values for {} rows.",
                    column.len(),
                    data.num_rows()
                )));
            }
            if column.null_count() != 0 {
                return Err(error::internal(format!(
                    "Batch weight column has {} null values.",
                    column.null_count()
                )));
            }
        }
        Ok(Self { data, weights })
    }

    /// The number of rows.
    pub fn num_rows(&self) -> usize {
        self.data.num_rows()
    }

    /// The Arrow schema of the data columns (the weights are not a column).
    pub fn schema(&self) -> SchemaRef {
        self.data.schema()
    }

    /// The data columns.
    pub fn data(&self) -> &RecordBatch {
        &self.data
    }

    /// The row weights.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// Whether every row has weight 1 ([`Weights::Const`]`(1)`).
    ///
    /// Operators take the "no weights" code path when this is true; that is
    /// what lets batch queries pay nothing for weights.
    pub fn is_insert_only(&self) -> bool {
        self.weights == Weights::Const(1)
    }

    /// Splits the batch into its data and weights.
    pub fn into_parts(self) -> (RecordBatch, Weights) {
        (self.data, self.weights)
    }

    /// The rows where `mask` is true, with their weights (copy-on-filter).
    ///
    /// A null in `mask` does not select its row, as in a SQL `WHERE`. A
    /// [`Weights::Const`] stays constant; a [`Weights::Column`] is filtered
    /// with the same mask. A mask that selects every row shares the input's
    /// buffers instead of copying.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::{BooleanArray, Int64Array, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
    /// use krishiv_types::batch::{Batch, Weights};
    /// # let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
    /// # let data = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 3]))])?;
    /// let batch = Batch::new(data);
    /// let kept = batch.filter(&BooleanArray::from(vec![Some(true), None, Some(false)]))?;
    /// assert_eq!(kept.num_rows(), 1);
    /// assert_eq!(kept.weights(), &Weights::Const(1));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if `mask` does not have one value per row.
    pub fn filter(&self, mask: &BooleanArray) -> Result<Self> {
        if mask.len() != self.num_rows() {
            return Err(error::internal(format!(
                "Filter mask has {} values for {} rows.",
                mask.len(),
                self.num_rows()
            )));
        }
        let mut builder = FilterBuilder::new(mask);
        // Building the optimized iteration plan pays off once it is reused
        // across arrays, or for a single array of a nested type (as arrow's
        // own `filter_record_batch` decides).
        let schema = self.data.schema_ref();
        let arrays =
            self.data.num_columns() + usize::from(matches!(self.weights, Weights::Column(_)));
        if arrays > 1
            || schema
                .fields()
                .first()
                .is_some_and(|f| FilterBuilder::is_optimize_beneficial(f.data_type()))
        {
            builder = builder.optimize();
        }
        let predicate = builder.build();
        let data = predicate
            .filter_record_batch(&self.data)
            .map_err(|e| arrow_error("filter", &e))?;
        let weights = match &self.weights {
            Weights::Const(c) => Weights::Const(*c),
            Weights::Column(column) => {
                let filtered = predicate
                    .filter(column)
                    .map_err(|e| arrow_error("filter", &e))?;
                let filtered = filtered
                    .as_primitive_opt::<Int64Type>()
                    .ok_or_else(|| error::internal("Filtered weights are not Int64.".to_owned()))?;
                Weights::Column(filtered.clone())
            }
        };
        Ok(Self { data, weights })
    }

    /// The columns at `indices`, in that order, with the same rows and
    /// weights. Zero-copy: the result shares the input's buffers. An index
    /// may repeat; an empty `indices` keeps the row count.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if an index is out of range.
    pub fn project(&self, indices: &[usize]) -> Result<Self> {
        let data = self
            .data
            .project(indices)
            .map_err(|e| arrow_error("project", &e))?;
        Ok(Self {
            data,
            weights: self.weights.clone(),
        })
    }

    /// `len` rows starting at `offset`, with their weights. Zero-copy: the
    /// result shares the input's buffers.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if `offset + len` exceeds [`Self::num_rows`].
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        if offset
            .checked_add(len)
            .is_none_or(|end| end > self.num_rows())
        {
            return Err(error::internal(format!(
                "Slice of {len} rows at offset {offset} is out of range for {} rows.",
                self.num_rows()
            )));
        }
        let weights = match &self.weights {
            Weights::Const(c) => Weights::Const(*c),
            Weights::Column(column) => Weights::Column(column.slice(offset, len)),
        };
        Ok(Self {
            data: self.data.slice(offset, len),
            weights,
        })
    }

    /// The rows of `batches`, in order, as one batch with `schema` (copies,
    /// except that a single batch is returned with shared buffers).
    ///
    /// Weights: if every batch has [`Weights::Const`]`(c)` with the same `c`,
    /// the result is `Const(c)` (an empty `batches` gives `Const(1)`);
    /// otherwise the result is a [`Weights::Column`] holding each batch's
    /// weights in turn.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::{Int64Array, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
    /// use krishiv_types::batch::{Batch, Weights};
    /// # let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
    /// # let data = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![7, 8]))])?;
    /// let inserts = Batch::new(data.clone());
    /// let deletes = Batch::with_weights(data, Weights::Const(-1))?;
    /// let all = Batch::concat(&schema, &[inserts, deletes])?;
    /// assert_eq!(all.weights(), &Weights::Column(Int64Array::from(vec![1, 1, -1, -1])));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a batch's fields differ from `schema`'s (names,
    /// types, nullability or field metadata; schema-level metadata is
    /// ignored), or if Arrow cannot concatenate the columns (for example an
    /// offset overflow).
    pub fn concat(schema: &SchemaRef, batches: &[Self]) -> Result<Self> {
        if let Some(bad) = batches
            .iter()
            .find(|b| b.data.schema_ref().fields() != schema.fields())
        {
            return Err(error::internal(format!(
                "Cannot concatenate a batch with fields {:?} into schema fields {:?}.",
                bad.data.schema_ref().fields(),
                schema.fields()
            )));
        }
        let data = match batches {
            [only] => RecordBatch::try_new_with_options(
                SchemaRef::clone(schema),
                only.data.columns().to_vec(),
                &RecordBatchOptions::new().with_row_count(Some(only.num_rows())),
            )
            .map_err(|e| arrow_error("concat", &e))?,
            _ => concat_batches(schema, batches.iter().map(|b| &b.data))
                .map_err(|e| arrow_error("concat", &e))?,
        };
        Ok(Self {
            weights: concat_weights(batches, data.num_rows()),
            data,
        })
    }
}

/// The weights of `batches` concatenated: one shared constant stays constant,
/// anything else becomes a column of `num_rows` values.
fn concat_weights(batches: &[Batch], num_rows: usize) -> Weights {
    let first = batches
        .first()
        .map_or(Weights::Const(1), |b| b.weights.clone());
    if let Weights::Const(c) = first
        && batches.iter().all(|b| b.weights == Weights::Const(c))
    {
        return Weights::Const(c);
    }
    if let [only] = batches {
        return only.weights.clone();
    }
    let mut values = Vec::with_capacity(num_rows);
    for b in batches {
        match &b.weights {
            Weights::Const(c) => values.extend(std::iter::repeat_n(*c, b.num_rows())),
            Weights::Column(column) => values.extend_from_slice(column.values()),
        }
    }
    Weights::Column(Int64Array::from(values))
}

fn arrow_error(op: &str, e: &ArrowError) -> KrishivError {
    error::internal(format!("Batch {op} failed in Arrow: {e}"))
}

#[cfg(test)]
mod tests;
