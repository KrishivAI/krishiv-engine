// SPDX-License-Identifier: Apache-2.0
//! Small scan batches joined toward the query's batch size (E7', study 14).
//!
//! A file scan decodes batches of at most [`DECODE_ROWS`] rows and emitted
//! each one after its pushed row filter and runtime filter, however few
//! rows survived. Every operator above then paid its per-batch cost for a
//! handful of rows. A [`ScanCoalesce`] copies the small ones into batches of
//! about `krishiv.exec.batchSize` rows. A batch with at least half of
//! `min(target, DECODE_ROWS)` rows passes through uncopied, after the rows
//! buffered before it, so dense scans stay copy-free whatever the target.
//! The buffer is charged to the driver's consumer; when the pool refuses
//! the charge, the buffered rows go out at once. The scan flushes it before
//! it moves to another morsel (the source's position names the rows'
//! morsel), at a pushed limit and at its end. When its time slice ends it hands the
//! driver an empty batch and keeps the buffer (the driver pushes an empty batch
//! to no operator).

use std::collections::VecDeque;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use arrow_select::coalesce::BatchCoalescer;
use krishiv_memory::Reservation;

use crate::error::{Result, internal};

/// The most rows a file format's decoder puts in one batch: a decoded
/// batch is never larger, so the pass-through threshold is capped by it.
/// Keep it equal to krishiv-io's three `pub(crate) BATCH_ROWS` (Parquet
/// `parquet/decode.rs`, Avro `avro/decode.rs`, text `text/mod.rs`); if
/// one changes, change this with it.
pub(crate) const DECODE_ROWS: usize = 4096;

/// Buffers small batches into batches of about `target` rows, in order.
pub(crate) struct ScanCoalesce {
    coalescer: BatchCoalescer,
    /// Batches of at least this many rows pass through uncopied.
    pass: usize,
    /// What the coalescer holds, charged to the driver's consumer.
    charged: Reservation,
    /// Batches ready to emit, in input order.
    ready: VecDeque<RecordBatch>,
}

impl ScanCoalesce {
    /// Joins batches of `schema` toward `target` rows (at least 1), charging
    /// what it buffers to `charged`.
    pub(crate) fn new(schema: SchemaRef, target: usize, charged: Reservation) -> Self {
        let target = target.max(1);
        Self {
            coalescer: BatchCoalescer::new(schema, target),
            pass: target.min(DECODE_ROWS).div_ceil(2),
            charged,
            ready: VecDeque::new(),
        }
    }

    /// Takes one batch. An empty one is dropped; one of at least the
    /// pass-through threshold is ready as it is, after what is buffered.
    pub(crate) fn push(&mut self, batch: RecordBatch) -> Result<()> {
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(());
        }
        if rows >= self.pass {
            self.flush()?;
            self.ready.push_back(batch);
            return Ok(());
        }
        self.coalescer
            .push_batch(batch)
            .map_err(|e| internal(format!("coalescing scan batches: {e}")))?;
        self.take_completed();
        // The buffer grows within the pool or goes out now.
        let held = self.coalescer.size();
        if held > self.charged.size() && self.charged.try_grow(held - self.charged.size()).is_err()
        {
            self.flush()?;
        }
        Ok(())
    }

    /// Makes the buffered rows ready as one batch.
    pub(crate) fn flush(&mut self) -> Result<()> {
        self.coalescer
            .finish_buffered_batch()
            .map_err(|e| internal(format!("coalescing scan batches: {e}")))?;
        self.take_completed();
        Ok(())
    }

    /// Whether rows are buffered (not yet ready).
    #[cfg(test)]
    pub(crate) fn has_buffered(&self) -> bool {
        self.coalescer.get_buffered_rows() > 0
    }

    /// The next ready batch.
    pub(crate) fn pop_ready(&mut self) -> Option<RecordBatch> {
        self.ready.pop_front()
    }

    /// Moves completed batches out of the coalescer: they leave its charge
    /// (a ready batch is the driver's batch in flight).
    fn take_completed(&mut self) {
        while let Some(b) = self.coalescer.next_completed_batch() {
            self.ready.push_back(b);
        }
        let held = self.coalescer.size();
        if held < self.charged.size() {
            self.charged.shrink(self.charged.size() - held);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_memory::{ConsumerHandle, MemoryConfig, MemoryPool, TreePool};

    use super::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]))
    }

    fn batch(from: i64, rows: i64) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int64Array::from_iter_values(from..from + rows))],
        )
        .unwrap()
    }

    fn reservation(limit: Option<usize>) -> (Arc<ConsumerHandle>, Reservation) {
        let pool = TreePool::new(&MemoryConfig {
            limit,
            ..MemoryConfig::default()
        });
        let h = ConsumerHandle::register(pool as Arc<dyn MemoryPool>, "scan", None);
        let r = h.reservation();
        (h, r)
    }

    fn drain(c: &mut ScanCoalesce) -> Vec<(usize, Vec<i64>)> {
        let mut out = Vec::new();
        while let Some(b) = c.pop_ready() {
            let v = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            out.push((b.num_rows(), v.values().to_vec()));
        }
        out
    }

    #[test]
    fn small_batches_join_and_dense_ones_pass_after_them_in_order() {
        let (_h, r) = reservation(None);
        let mut c = ScanCoalesce::new(schema(), 8, r);
        for i in 0..5 {
            c.push(batch(i * 3, 3)).unwrap();
        }
        c.push(batch(100, 0)).unwrap();
        // 15 rows: one full batch of 8 is ready, 7 wait (charged).
        assert_eq!(drain(&mut c), vec![(8, (0..8).collect())]);
        assert!(c.has_buffered() && c.charged.size() > 0);
        // A dense batch (4 = half of 8) goes out after the 7 buffered rows.
        let dense = batch(15, 4);
        let ptr = dense.column(0).to_data().buffers()[0].as_ptr();
        c.push(dense).unwrap();
        let seen: Vec<RecordBatch> = std::iter::from_fn(|| c.pop_ready()).collect();
        assert_eq!(
            seen.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
            [7, 4]
        );
        // Uncopied: the same buffer.
        assert_eq!(seen[1].column(0).to_data().buffers()[0].as_ptr(), ptr);
        c.push(batch(19, 2)).unwrap();
        assert!(c.pop_ready().is_none());
        c.flush().unwrap();
        assert_eq!(drain(&mut c), vec![(2, vec![19, 20])]);
        assert!(!c.has_buffered());
        c.flush().unwrap();
        assert!(c.pop_ready().is_none());
    }

    #[test]
    fn the_pass_through_threshold_is_capped_by_the_decoders_batch() {
        let (_h, r) = reservation(None);
        // A target far above what a decoder emits: a full decoded batch
        // still passes uncopied.
        let mut c = ScanCoalesce::new(schema(), 64 * DECODE_ROWS, r);
        let rows = i64::try_from(DECODE_ROWS / 2).unwrap();
        c.push(batch(0, rows)).unwrap();
        assert_eq!(c.pop_ready().map(|b| b.num_rows()), Some(DECODE_ROWS / 2));
        // A target of 0 is 1: everything passes.
        let (_h, r) = reservation(None);
        let mut c = ScanCoalesce::new(schema(), 0, r);
        c.push(batch(0, 1)).unwrap();
        assert_eq!(drain(&mut c), vec![(1, vec![0])]);
    }

    #[test]
    fn a_refused_charge_sends_the_buffer_out_at_once() {
        // A pool too small for the coalescer's buffer: each small batch goes
        // out as it comes, in order, and nothing stays charged.
        let (_h, r) = reservation(Some(64));
        let mut c = ScanCoalesce::new(schema(), 1024, r);
        c.push(batch(0, 3)).unwrap();
        c.push(batch(3, 3)).unwrap();
        assert_eq!(drain(&mut c), vec![(3, vec![0, 1, 2]), (3, vec![3, 4, 5])]);
        assert!(!c.has_buffered());
        assert_eq!(c.charged.size(), 0);
    }
}
