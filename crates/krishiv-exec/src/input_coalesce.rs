// SPDX-License-Identifier: Apache-2.0
//! Small batches joined before a pipeline breaker (Polars item 2, J1).
//!
//! A selective filter or join hands on batches of a few rows: TPC-H q12's
//! filter over `lineitem` keeps 21 rows a batch, and q05's last join 5. Each
//! operator above then pays its per-batch cost (a hash join build cuts every
//! batch into radix partitions and arena chunks, about 60 us a batch) for a
//! handful of rows. In a pipeline whose sink allows it
//! ([`Sink::coalesce_input`](crate::operator::Sink::coalesce_input)), the
//! driver puts an [`InputCoalesce`] in front of the sink and of each operator
//! that asks for it, which joins such batches into batches of about
//! `krishiv.exec.batchSize` rows.
//!
//! A batch with at least a quarter of `min(target, PASS_CAP)` rows (1,024 at
//! the defaults) passes through uncopied, after the rows held before it, so
//! pipelines of full or nearly full batches stay copy-free: such a batch
//! already spreads the per-batch cost thin, and joining it would only make
//! the operator above grow its state in larger steps. Only unweighted batches of one scan split are joined: a
//! weighted batch or one of another split sends the held rows on first. The
//! held batches are not charged to the memory pool, like any batch in flight
//! between two operators: they are bounded instead, at about one batch of
//! `target` rows and at most [`held_max_bytes`] (counted with the buffers
//! they keep alive, so a slice of a large batch is not held long): per
//! driver, that much for each operator that joins its input. The cap is
//! 1 MiB, or a sixty-fourth of the query's memory limit when that is less:
//! a hash build reserves a whole input batch at once, so under a tight limit
//! (Q18 at 1 MiB) batches of a few tens of KiB pass on alone, as they would
//! without joining. Charging them made an operator under a tight limit
//! fail for memory that its own spill could not free.

use std::collections::VecDeque;

use krishiv_types::batch::Batch;

/// The pass-through threshold is a quarter of at most this many rows: a file
/// scan's decoded batch (krishiv-io's `BATCH_ROWS`) is never larger, so a
/// full one passes whatever the target.
const PASS_CAP: usize = 4096;

/// Held batches whose memory reaches this many bytes are joined and sent
/// on at once, at most (see [`held_max_bytes`]).
pub(crate) const HELD_MAX_BYTES: usize = 1 << 20;

/// The byte cap on held batches under a memory `limit`: [`HELD_MAX_BYTES`],
/// or a sixty-fourth of the limit when that is less. A cap of 0 (a limit
/// under 64 bytes) never joins: each batch is sent on as it comes.
pub(crate) fn held_max_bytes(limit: Option<usize>) -> usize {
    limit.map_or(HELD_MAX_BYTES, |l| HELD_MAX_BYTES.min(l / 64))
}

/// Joins small batches into batches of about `target` rows, in order.
pub(crate) struct InputCoalesce {
    target: usize,
    /// Batches of at least this many rows pass through uncopied.
    pass: usize,
    /// Held batches (unweighted, one split), not yet joined.
    held: Vec<Batch>,
    rows: usize,
    split: Option<u32>,
    /// The memory `held` keeps alive.
    bytes: usize,
    /// Held batches whose memory reaches this are sent on.
    max_bytes: usize,
    /// Batches ready to hand on, in input order.
    ready: VecDeque<Batch>,
}

impl InputCoalesce {
    /// Joins batches toward `target` rows (at least 1), holding at most
    /// about `max_bytes`.
    pub(crate) fn new(target: usize, max_bytes: usize) -> Self {
        let target = target.max(1);
        Self {
            target,
            pass: target.min(PASS_CAP).div_ceil(4),
            held: Vec::new(),
            rows: 0,
            split: None,
            bytes: 0,
            max_bytes,
            ready: VecDeque::new(),
        }
    }

    /// Takes one batch. An empty one is dropped; a dense or weighted one is
    /// ready as it is, after what is held. One of another split or schema
    /// (field names, nullability) sends the held rows on first.
    pub(crate) fn push(&mut self, batch: Batch) {
        let rows = batch.num_rows();
        if rows == 0 {
            return;
        }
        if rows >= self.pass || !batch.is_unweighted() {
            self.flush();
            self.ready.push_back(batch);
            return;
        }
        if self
            .held
            .first()
            .is_some_and(|h| batch.split() != self.split || h.schema() != batch.schema())
        {
            self.flush();
        }
        self.split = batch.split();
        self.rows += rows;
        self.bytes += batch.data().get_array_memory_size();
        self.held.push(batch);
        if self.rows >= self.target || self.bytes >= self.max_bytes {
            self.flush();
        }
    }

    /// Makes the held rows ready, as one batch where they can be joined
    /// (else as they came: joining is only an optimisation).
    pub(crate) fn flush(&mut self) {
        let joined = match self.held.first() {
            Some(first) if self.held.len() > 1 => Batch::concat(&first.schema(), &self.held).ok(),
            _ => None,
        };
        match joined {
            Some(b) => {
                self.held.clear();
                self.ready.push_back(b.with_split(self.split));
            }
            None => self.ready.extend(self.held.drain(..)),
        }
        self.rows = 0;
        self.bytes = 0;
    }

    /// The next ready batch.
    pub(crate) fn pop_ready(&mut self) -> Option<Batch> {
        self.ready.pop_front()
    }

    /// Whether rows are held (not yet ready).
    #[cfg(test)]
    pub(crate) fn has_held(&self) -> bool {
        !self.held.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use krishiv_types::batch::Weights;

    use super::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]))
    }

    fn data(from: i64, rows: i64) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int64Array::from_iter_values(from..from + rows))],
        )
        .unwrap()
    }

    fn batch(from: i64, rows: i64) -> Batch {
        Batch::new(data(from, rows))
    }

    fn values(b: &Batch) -> Vec<i64> {
        let v = b.data().column(0);
        v.as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn drain(c: &mut InputCoalesce) -> Vec<(Option<u32>, Vec<i64>)> {
        std::iter::from_fn(|| c.pop_ready())
            .map(|b| (b.split(), values(&b)))
            .collect()
    }

    #[test]
    fn small_batches_join_in_order_and_dense_ones_pass_uncopied() {
        let mut c = InputCoalesce::new(16, HELD_MAX_BYTES);
        for i in 0..6 {
            c.push(batch(i * 3, 3));
        }
        c.push(batch(100, 0));
        // 18 rows reach the target of 16: one batch, all of them.
        assert_eq!(drain(&mut c), vec![(None, (0..18).collect())]);
        assert!(!c.has_held());
        assert_eq!(c.bytes, 0);
        c.push(batch(18, 2));
        assert!(c.has_held() && c.bytes > 0);
        assert!(c.pop_ready().is_none());
        // A dense batch (4 = a quarter of 16) goes on after the held rows,
        // uncopied.
        let dense = batch(20, 4);
        let ptr = dense.data().column(0).to_data().buffers()[0].as_ptr();
        c.push(dense);
        let seen: Vec<Batch> = std::iter::from_fn(|| c.pop_ready()).collect();
        assert_eq!(seen.iter().map(Batch::num_rows).collect::<Vec<_>>(), [2, 4]);
        assert_eq!(
            seen[1].data().column(0).to_data().buffers()[0].as_ptr(),
            ptr
        );
        c.flush();
        assert!(c.pop_ready().is_none());
    }

    #[test]
    fn rows_of_another_split_or_with_weights_are_not_joined() {
        let mut c = InputCoalesce::new(100, HELD_MAX_BYTES);
        c.push(batch(0, 2).with_split(Some(1)));
        c.push(batch(2, 2).with_split(Some(1)));
        c.push(batch(4, 2).with_split(Some(2)));
        let weighted = Batch::with_weights(data(6, 2), Weights::Const(-1)).unwrap();
        c.push(weighted);
        c.push(batch(8, 1).with_split(Some(2)));
        c.flush();
        let out: Vec<Batch> = std::iter::from_fn(|| c.pop_ready()).collect();
        let shape: Vec<(Option<u32>, Vec<i64>, bool)> = out
            .iter()
            .map(|b| (b.split(), values(b), b.is_unweighted()))
            .collect();
        assert_eq!(
            shape,
            vec![
                (Some(1), vec![0, 1, 2, 3], true),
                (Some(2), vec![4, 5], true),
                (None, vec![6, 7], false),
                (Some(2), vec![8], true),
            ]
        );
    }

    #[test]
    fn a_batch_of_another_schema_is_not_joined() {
        let mut c = InputCoalesce::new(100, HELD_MAX_BYTES);
        c.push(batch(0, 2));
        let other = Arc::new(Schema::new(vec![Field::new("w", DataType::Int64, true)]));
        let renamed = RecordBatch::try_new(other, vec![Arc::clone(data(2, 2).column(0))]).unwrap();
        c.push(Batch::new(renamed));
        c.push(batch(4, 2));
        c.flush();
        let names: Vec<String> = std::iter::from_fn(|| c.pop_ready())
            .map(|b| format!("{}{:?}", b.schema().field(0).name(), values(&b)))
            .collect();
        assert_eq!(names, ["v[0, 1]", "w[2, 3]", "v[4, 5]"]);
    }

    #[test]
    fn the_pass_through_threshold_is_capped() {
        let mut c = InputCoalesce::new(64 * PASS_CAP, HELD_MAX_BYTES);
        let rows = i64::try_from(PASS_CAP / 4).unwrap();
        c.push(batch(0, rows));
        assert_eq!(c.pop_ready().map(|b| b.num_rows()), Some(PASS_CAP / 4));
        // A target of 0 is 1: everything passes.
        let mut c = InputCoalesce::new(0, HELD_MAX_BYTES);
        c.push(batch(0, 1));
        assert_eq!(drain(&mut c), vec![(None, vec![0])]);
    }

    #[test]
    fn a_tight_memory_limit_lowers_the_byte_cap() {
        assert_eq!(held_max_bytes(None), HELD_MAX_BYTES);
        assert_eq!(held_max_bytes(Some(1 << 30)), HELD_MAX_BYTES);
        assert_eq!(held_max_bytes(Some(1 << 26)), HELD_MAX_BYTES);
        assert_eq!(held_max_bytes(Some(1 << 20)), 1 << 14);
        assert_eq!(held_max_bytes(Some(0)), 0);
        // A cap just above one 20-row batch's memory: the second reaches it,
        // so the two go on joined, before the 100-row target.
        let one = batch(0, 20).data().get_array_memory_size();
        let mut c = InputCoalesce::new(100, one + 1);
        c.push(batch(0, 20));
        assert!(c.has_held());
        c.push(batch(20, 20));
        assert!(!c.has_held());
        assert_eq!(drain(&mut c), vec![(None, (0..40).collect())]);
    }

    #[test]
    fn slices_of_a_large_batch_are_not_held_past_the_byte_cap() {
        // Each one-row slice keeps the whole 64K-row buffer (512 KiB) alive:
        // the second reaches the cap, so the two go on as one batch.
        let mut c = InputCoalesce::new(1024, HELD_MAX_BYTES);
        let big = batch(0, 65_536);
        c.push(big.slice(0, 1).unwrap());
        assert!(c.has_held() && c.pop_ready().is_none());
        c.push(big.slice(1, 1).unwrap());
        assert_eq!(drain(&mut c), vec![(None, vec![0, 1])]);
        assert!(!c.has_held());
        assert_eq!(c.bytes, 0);
    }
}
