// SPDX-License-Identifier: Apache-2.0
//! The row stage's survivors of sparse probe batches, held until a batch is
//! full (design `P3-0.3.1-probe-empty-batches.md` §3 K1).
//!
//! A fact scan's runtime filters often keep a few rows of each 4,096-row
//! batch (TPC-DS q48: 3.5 a batch). The probe's work after its row stage
//! (the fused predicate, the key encoding and lookups, the output) costs
//! about the same per batch whatever the rows, so the probe copies the rows
//! the stage kept of such a batch into one buffer and runs the rest once
//! per `krishiv.exec.batchSize` held rows instead of once per input batch.
//!
//! Like the driver's `InputCoalesce` (`input_coalesce.rs`), the held rows
//! are not charged to the memory pool: they are bounded instead, at about
//! one batch of rows and [`held_max_bytes`](crate::input_coalesce::held_max_bytes)
//! (estimated from the batches they came from, an overestimate for slices
//! and string views). Only unweighted batches are held (the probe checks),
//! and a batch whose schema differs from the held rows' is not.

use arrow_array::{BooleanArray, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_select::coalesce::BatchCoalescer;

use crate::error::{Result, internal};

/// A batch is sparse when its row stage keeps fewer than one row in this
/// many.
const SPARSE: usize = 4;

/// The held rows of one probe driver.
#[derive(Debug)]
pub(super) struct Held {
    /// The rows, copied (string views compacted); created with the first.
    rows: Option<BatchCoalescer>,
    /// A full held batch has this many rows.
    target: usize,
    /// The estimated memory of the held rows.
    bytes: usize,
    /// Held rows reaching this estimate are probed at once; 0 never holds.
    max_bytes: usize,
}

impl Held {
    /// Holds toward `target` rows (at least 1) and at most about
    /// `max_bytes` (0: holds nothing).
    pub(super) fn new(target: usize, max_bytes: usize) -> Self {
        Self {
            rows: None,
            target: target.max(1),
            bytes: 0,
            max_bytes,
        }
    }

    /// Whether the `kept` rows the row stage kept of `data` are held: the
    /// batch is sparse, they are fewer than a full batch (so one push fills
    /// at most one) and `data` has the held rows' schema.
    pub(super) fn takes(&self, data: &RecordBatch, kept: usize) -> bool {
        self.max_bytes > 0
            && kept.saturating_mul(SPARSE) < data.num_rows()
            && kept < self.target
            && self.rows.as_ref().is_none_or(|c| {
                let (a, b) = (c.schema(), data.schema());
                std::sync::Arc::ptr_eq(&a, &b) || a == b
            })
    }

    /// Holds the rows `kept` (`count` of them) of `data`; returns a full
    /// batch of held rows to probe now, if they reached the target rows or
    /// bytes.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when the copy fails.
    pub(super) fn push(
        &mut self,
        data: &RecordBatch,
        kept: &BooleanBuffer,
        count: usize,
    ) -> Result<Option<RecordBatch>> {
        if count == 0 {
            return Ok(None);
        }
        let n = data.num_rows().max(1);
        let rows = self
            .rows
            .get_or_insert_with(|| BatchCoalescer::new(data.schema(), self.target));
        let mask = BooleanArray::new(kept.clone(), None);
        rows.push_batch_with_filter(data.clone(), &mask)
            .map_err(|e| internal(format!("holding probe rows: {e}")))?;
        let per_row = data.get_array_memory_size() / n;
        self.bytes = self.bytes.saturating_add(per_row.saturating_mul(count));
        if let Some(full) = rows.next_completed_batch() {
            // The tail of this push stays held, with its share of the bytes.
            let left = rows.get_buffered_rows();
            let all = left + full.num_rows();
            self.bytes = self.bytes.saturating_mul(left) / all.max(1);
            return Ok(Some(full));
        }
        if self.bytes >= self.max_bytes {
            return self.rest();
        }
        Ok(None)
    }

    /// Every held row, as one batch (`None`: none held).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when the copy fails.
    pub(super) fn rest(&mut self) -> Result<Option<RecordBatch>> {
        let Some(rows) = self.rows.as_mut() else {
            return Ok(None);
        };
        if let Some(done) = rows.next_completed_batch() {
            return Ok(Some(done));
        }
        rows.finish_buffered_batch()
            .map_err(|e| internal(format!("holding probe rows: {e}")))?;
        self.bytes = 0;
        Ok(rows.next_completed_batch())
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array, RecordBatch};
    use arrow_buffer::BooleanBuffer;
    use arrow_schema::{DataType, Field, Schema};

    use super::Held;

    fn data(from: i64, rows: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from_iter_values(from..from + rows))],
        )
        .unwrap_or_else(|e| unreachable!("{e}"))
    }

    fn every(step: usize, rows: usize) -> (BooleanBuffer, usize) {
        let mask = BooleanBuffer::collect_bool(rows, |i| i % step == 0);
        let count = mask.count_set_bits();
        (mask, count)
    }

    fn values(b: &RecordBatch) -> Vec<i64> {
        let v = b.column(0).as_any().downcast_ref::<Int64Array>();
        v.map(|v| v.values().to_vec()).unwrap_or_default()
    }

    #[test]
    fn sparse_rows_are_held_until_a_batch_is_full() {
        let mut held = Held::new(4, 1 << 20);
        let d = data(0, 20);
        let (mask, count) = every(10, 20);
        assert!(held.takes(&d, count));
        // Two of 20 rows: held.
        assert!(held.push(&d, &mask, count).ok().flatten().is_none());
        let d = data(100, 20);
        let (mask, count) = every(7, 20);
        // Three more reach four: one full batch, one row left held.
        let full = held.push(&d, &mask, count).ok().flatten();
        assert_eq!(full.as_ref().map(values), Some(vec![0, 10, 100, 107]));
        let rest = held.rest().ok().flatten();
        assert_eq!(rest.as_ref().map(values), Some(vec![114]));
        assert!(held.rest().ok().flatten().is_none());
    }

    #[test]
    fn dense_wide_or_other_schema_batches_are_not_held() {
        let held = Held::new(8, 1 << 20);
        let d = data(0, 20);
        // Five of 20 is a quarter: not sparse.
        assert!(!held.takes(&d, 5));
        assert!(held.takes(&d, 4));
        // As many as a full batch: not held.
        let d = data(0, 400);
        assert!(!held.takes(&d, 8));
        assert!(!Held::new(8, 0).takes(&d, 1));
        let mut held = Held::new(8, 1 << 20);
        let (mask, count) = every(10, 40);
        assert!(held.push(&data(0, 40), &mask, count).is_ok());
        let other = Arc::new(Schema::new(vec![Field::new("w", DataType::Int64, false)]));
        let col = Arc::new(Int64Array::from(vec![0; 40])) as Arc<dyn Array>;
        let other = RecordBatch::try_new(other, vec![col]).unwrap_or_else(|e| unreachable!("{e}"));
        assert!(!held.takes(&other, 1));
        assert!(held.takes(&data(0, 40), 1));
        // No row kept: nothing held, nothing to probe.
        let none = BooleanBuffer::new_unset(40);
        assert!(held.push(&data(0, 40), &none, 0).ok().flatten().is_none());
    }

    #[test]
    fn held_bytes_past_the_cap_are_probed_at_once() {
        let d = data(0, 64);
        let (mask, count) = every(16, 64);
        // 8 bytes a row over 64 rows, plus the buffers: 4 rows are past 16.
        let mut held = Held::new(1000, 16);
        let full = held.push(&d, &mask, count).ok().flatten();
        assert_eq!(full.as_ref().map(values), Some(vec![0, 16, 32, 48]));
        assert_eq!(held.bytes, 0);
        assert!(held.rest().ok().flatten().is_none());
    }
}
