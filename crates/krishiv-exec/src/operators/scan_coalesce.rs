// SPDX-License-Identifier: Apache-2.0
//! Small scan batches joined toward the query's batch size (E7', study 14).
//!
//! A file scan decodes fixed-size batches and emits each one after its
//! pushed row filter and runtime filter, however few rows survive. Every
//! operator above then pays its per-batch cost for a handful of rows. A
//! [`ScanCoalesce`] copies the small ones into batches of about
//! `krishiv.exec.batchSize` rows. A batch with at least half the target
//! passes through uncopied, after the rows buffered before it, so dense
//! scans stay copy-free. The scan flushes it before it moves to another
//! morsel (the source's position names the rows' morsel) and at its end.

use std::collections::VecDeque;

use arrow_array::RecordBatch;
use arrow_select::coalesce::BatchCoalescer;

use crate::error::{Result, internal};

/// Buffers small batches into batches of about `target` rows, in order.
pub(crate) struct ScanCoalesce {
    target: usize,
    /// Built from the first small batch's schema.
    coalescer: Option<BatchCoalescer>,
    /// Batches ready to emit, in input order.
    ready: VecDeque<RecordBatch>,
}

impl ScanCoalesce {
    /// Joins batches toward `target` rows (at least 1).
    pub(crate) fn new(target: usize) -> Self {
        Self {
            target: target.max(1),
            coalescer: None,
            ready: VecDeque::new(),
        }
    }

    /// Takes one batch. An empty one is dropped; one of at least half the
    /// target is ready as it is, after what is buffered.
    pub(crate) fn push(&mut self, batch: RecordBatch) -> Result<()> {
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(());
        }
        if rows >= self.target.div_ceil(2) {
            self.finish()?;
            self.ready.push_back(batch);
            return Ok(());
        }
        let target = self.target;
        let c = self
            .coalescer
            .get_or_insert_with(|| BatchCoalescer::new(batch.schema(), target));
        c.push_batch(batch)
            .map_err(|e| internal(format!("coalescing scan batches: {e}")))?;
        while let Some(b) = c.next_completed_batch() {
            self.ready.push_back(b);
        }
        Ok(())
    }

    /// Makes the buffered rows ready as one batch.
    pub(crate) fn finish(&mut self) -> Result<()> {
        if let Some(c) = self.coalescer.as_mut() {
            c.finish_buffered_batch()
                .map_err(|e| internal(format!("coalescing scan batches: {e}")))?;
            while let Some(b) = c.next_completed_batch() {
                self.ready.push_back(b);
            }
        }
        Ok(())
    }

    /// The next ready batch.
    pub(crate) fn next(&mut self) -> Option<RecordBatch> {
        self.ready.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    fn batch(from: i64, rows: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from_iter_values(from..from + rows))],
        )
        .unwrap()
    }

    fn drain(c: &mut ScanCoalesce) -> Vec<(usize, Vec<i64>)> {
        let mut out = Vec::new();
        while let Some(b) = c.next() {
            let v = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            out.push((b.num_rows(), v.values().to_vec()));
        }
        out
    }

    #[test]
    fn small_batches_join_and_dense_ones_pass_after_them_in_order() {
        let mut c = ScanCoalesce::new(8);
        for i in 0..5 {
            c.push(batch(i * 3, 3)).unwrap();
        }
        c.push(batch(100, 0)).unwrap();
        // 15 rows: one full batch of 8 is ready, 7 wait.
        let got = drain(&mut c);
        assert_eq!(got, vec![(8, (0..8).collect())]);
        // A dense batch (4 = half of 8) goes out after the 7 buffered rows.
        let dense = batch(15, 4);
        let ptr = dense.column(0).to_data().buffers()[0].as_ptr();
        c.push(dense).unwrap();
        let mut seen = Vec::new();
        while let Some(b) = c.next() {
            seen.push(b);
        }
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].num_rows(), 7);
        assert_eq!(seen[1].num_rows(), 4);
        // Uncopied: the same buffer.
        assert_eq!(seen[1].column(0).to_data().buffers()[0].as_ptr(), ptr);
        c.push(batch(19, 2)).unwrap();
        assert!(c.next().is_none());
        c.finish().unwrap();
        assert_eq!(drain(&mut c), vec![(2, vec![19, 20])]);
        c.finish().unwrap();
        assert!(c.next().is_none());
    }

    #[test]
    fn a_target_of_one_passes_everything_through() {
        let mut c = ScanCoalesce::new(0);
        c.push(batch(0, 1)).unwrap();
        assert_eq!(drain(&mut c), vec![(1, vec![0])]);
    }
}
