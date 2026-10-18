// SPDX-License-Identifier: Apache-2.0
//! Sorted runs and their k-way merge.
//!
//! A run is a sorted stream of rows: a buffered input batch with its sorted
//! row order (sorted when it was sunk), or a spill file whose batches are in
//! order. The [`Merger`] keeps a loser tree of run cursors keyed by
//! `(key row, run index)` (one comparison per level and output row), so rows that compare equal come out in run order:
//! the merge is stable when runs are numbered in input order.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, RecordBatchOptions};
use arrow_row::{Row, Rows};
use arrow_schema::SchemaRef;
use arrow_select::interleave::interleave;
use krishiv_memory::{
    ConsumerHandle, MemoryError, Reservation, SpillError, SpillFile, SpillReader, SpillScope,
};
use krishiv_types::batch::{Batch, Weights};

use super::keys::KeyEncoder;
use crate::context::QueryContext;
use crate::error::{Result, internal};

/// A sunk batch with its key rows and its sorted row order (ties in row
/// order). Shared, so a spill can merge it while the sink still owns it.
#[derive(Debug)]
pub(crate) struct SortedBatch {
    pub(crate) batch: Batch,
    pub(crate) rows: Rows,
    pub(crate) order: Vec<u32>,
}

impl SortedBatch {
    /// Encodes and sorts `batch`.
    pub(crate) fn new(batch: Batch, keys: &KeyEncoder) -> Result<Self> {
        let rows = keys.encode(batch.data())?;
        let n = u32::try_from(batch.num_rows())
            .map_err(|_| internal("a batch has more than u32::MAX rows".to_owned()))?;
        let mut keyed: Vec<(u64, u32)> = (0..n)
            .map(|i| (prefix(rows.row(i as usize).data()), i))
            .collect();
        // Unstable sort plus the row index as tie-break: a stable order
        // without the stable sort's scratch buffer. Most comparisons end at
        // the 8-byte key prefix.
        keyed.sort_unstable_by(|&(pa, a), &(pb, b)| {
            pa.cmp(&pb)
                .then_with(|| rows.row(a as usize).cmp(&rows.row(b as usize)))
                .then(a.cmp(&b))
        });
        let order = keyed.into_iter().map(|(_, i)| i).collect();
        Ok(Self { batch, rows, order })
    }

    /// Bytes this batch holds: data, key rows and order.
    pub(crate) fn bytes(&self) -> usize {
        self.batch.data().get_array_memory_size()
            + self.batch.num_rows() * (size_of::<u32>() + weight_bytes(&self.batch))
            + self.rows.size()
    }
}

/// The first 8 bytes of a key row, big-endian (zero-padded): comparing
/// prefixes orders rows as their bytes do, up to prefix ties.
fn prefix(row: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    let n = row.len().min(8);
    bytes[..n].copy_from_slice(&row[..n]);
    u64::from_be_bytes(bytes)
}

/// The bytes of `batch`'s rows (what a copy of them, or a spill frame,
/// holds), without the slack of the buffers it shares: a per-row size that
/// does not depend on the batch's length.
pub(crate) fn data_bytes(batch: &Batch) -> usize {
    let columns: usize = batch
        .data()
        .columns()
        .iter()
        .map(|c| {
            c.to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| c.get_array_memory_size())
        })
        .sum();
    columns + batch.num_rows() * weight_bytes(batch)
}

fn weight_bytes(batch: &Batch) -> usize {
    match batch.weights() {
        Weights::Const(_) => 0,
        Weights::Column(_) => size_of::<i64>(),
    }
}

/// A spilled sorted run.
#[derive(Debug)]
pub(crate) struct SpillRun {
    pub(crate) file: SpillFile,
    /// The largest batch written, in memory bytes: what reading it costs.
    pub(crate) max_batch_bytes: usize,
    /// The most memory bytes per row of a written batch.
    pub(crate) row_bytes: usize,
}

impl SpillRun {
    /// Memory a merge needs per open run: the reader's frame (encoded and
    /// decoded batch) and the batch's key rows.
    pub(crate) fn merge_bytes(&self) -> usize {
        self.max_batch_bytes.saturating_mul(3)
    }
}

/// Where a cursor's rows come from.
enum Input {
    Memory(Arc<SortedBatch>),
    Spill(Box<SpillInput>),
}

/// A spilled run being read: its current batch and key rows.
struct SpillInput {
    reader: SpillReader,
    batch: Batch,
    rows: Rows,
    /// Accounts `rows`.
    keys: Reservation,
}

/// The current position in one run.
struct Cursor {
    input: Input,
    pos: usize,
    len: usize,
    /// Index of the cursor's batch in [`Merger::sources`].
    source: usize,
    /// The current key's [`prefix`].
    prefix: u64,
}

impl Cursor {
    fn row_index(&self) -> usize {
        match &self.input {
            Input::Memory(s) => s.order.get(self.pos).map_or(0, |&i| i as usize),
            Input::Spill(_) => self.pos,
        }
    }

    fn key(&self) -> Row<'_> {
        let i = self.row_index();
        match &self.input {
            Input::Memory(s) => s.rows.row(i),
            Input::Spill(s) => s.rows.row(i),
        }
    }

    /// Caches the current key's prefix (after a move).
    fn refresh(&mut self) {
        if self.pos < self.len {
            self.prefix = prefix(self.key().data());
        }
    }

    fn batch(&self) -> &Batch {
        match &self.input {
            Input::Memory(s) => &s.batch,
            Input::Spill(s) => &s.batch,
        }
    }
}

/// Reads the next non-empty batch of `reader` with its key rows, accounted
/// in `keys`.
fn next_batch(
    reader: &mut SpillReader,
    encoder: &KeyEncoder,
    keys: &mut Reservation,
) -> Result<Option<(Batch, Rows)>> {
    for batch in reader.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let rows = encoder.encode(batch.data())?;
        keys.shrink(keys.size());
        keys.try_grow(rows.size())?;
        return Ok(Some((batch, rows)));
    }
    Ok(None)
}

/// A k-way merge of sorted runs.
pub(crate) struct Merger {
    schema: SchemaRef,
    encoder: Arc<KeyEncoder>,
    cursors: Vec<Cursor>,
    /// Loser tree over the cursors by `(key, cursor index)`: `tree[0]` is
    /// the winner, inner node `n` the loser of its match.
    tree: Vec<usize>,
    /// The batches the pending output rows point into.
    sources: Vec<Batch>,
    /// Whether a cursor moved to a new batch since the last output, so
    /// `sources` holds batches no cursor is on any more.
    retired: bool,
}

impl Merger {
    /// A merge of `memory` runs followed by `spilled` runs (in that run
    /// order). Opening a spill file is blocking I/O: call it from a
    /// blocking closure when `spilled` is not empty. Readers and key rows
    /// are accounted in reservations of `handle`.
    pub(crate) fn new(
        schema: SchemaRef,
        encoder: Arc<KeyEncoder>,
        memory: Vec<Arc<SortedBatch>>,
        spilled: &[SpillRun],
        handle: &Arc<ConsumerHandle>,
    ) -> Result<Self> {
        let mut cursors = Vec::with_capacity(memory.len() + spilled.len());
        for s in memory {
            let len = s.order.len();
            cursors.push(Cursor {
                input: Input::Memory(s),
                pos: 0,
                len,
                source: cursors.len(),
                prefix: 0,
            });
        }
        for run in spilled {
            // Frames and key rows come and go: what they free stays with
            // the consumer as credit for the next ones (see `Sort`).
            let mut reader = run
                .file
                .reader(handle.reservation().releasing_to_credit())?;
            let mut keys = handle.reservation().releasing_to_credit();
            if let Some((batch, rows)) = next_batch(&mut reader, &encoder, &mut keys)? {
                cursors.push(Cursor {
                    pos: 0,
                    len: batch.num_rows(),
                    source: cursors.len(),
                    prefix: 0,
                    input: Input::Spill(Box::new(SpillInput {
                        reader,
                        batch,
                        rows,
                        keys,
                    })),
                });
            }
        }
        cursors.iter_mut().for_each(Cursor::refresh);
        let sources = cursors.iter().map(|c| c.batch().clone()).collect();
        let mut merger = Self {
            schema,
            encoder,
            tree: vec![usize::MAX; cursors.len()],
            cursors,
            sources,
            retired: false,
        };
        for i in 0..merger.cursors.len() {
            // Plays `i` up until it meets an empty node (initial build).
            let (mut node, mut winner) = ((merger.cursors.len() + i) / 2, i);
            while node != 0 && merger.tree[node] != usize::MAX {
                if merger.beats(merger.tree[node], winner) {
                    winner = std::mem::replace(&mut merger.tree[node], winner);
                }
                node /= 2;
            }
            merger.tree[node] = winner;
        }
        Ok(merger)
    }

    /// Whether cursor `a` comes before `b`; an exhausted cursor never does.
    fn beats(&self, a: usize, b: usize) -> bool {
        let (ca, cb) = (&self.cursors[a], &self.cursors[b]);
        match (ca.pos < ca.len, cb.pos < cb.len) {
            (true, true) => ca
                .prefix
                .cmp(&cb.prefix)
                .then_with(|| ca.key().cmp(&cb.key()))
                .then(a.cmp(&b))
                .is_lt(),
            (live, _) => live,
        }
    }

    /// Replays the winner's path after it moved.
    fn replay(&mut self) {
        let mut winner = self.tree[0];
        let mut node = (self.cursors.len() + winner) / 2;
        while node != 0 {
            if self.beats(self.tree[node], winner) {
                winner = std::mem::replace(&mut self.tree[node], winner);
            }
            node /= 2;
        }
        self.tree[0] = winner;
    }

    /// Moves cursor `c` past its current row; `false` once its run ends.
    fn advance(&mut self, c: usize) -> Result<bool> {
        let cursor = &mut self.cursors[c];
        cursor.pos += 1;
        if cursor.pos < cursor.len {
            cursor.refresh();
            return Ok(true);
        }
        let Input::Spill(s) = &mut cursor.input else {
            return Ok(false);
        };
        let Some((next, next_rows)) = next_batch(&mut s.reader, &self.encoder, &mut s.keys)? else {
            return Ok(false);
        };
        cursor.pos = 0;
        cursor.len = next.num_rows();
        cursor.source = self.sources.len();
        self.sources.push(next.clone());
        s.batch = next;
        s.rows = next_rows;
        cursor.refresh();
        self.retired = true;
        Ok(true)
    }

    /// The next merged batch of at most `max_rows` rows (at least one), or
    /// `None` once every run is exhausted. Reading a spilled run is blocking
    /// I/O.
    pub(crate) fn next(&mut self, max_rows: usize) -> Result<Option<Batch>> {
        let mut out = Vec::with_capacity(max_rows.min(1 << 16));
        while out.len() < max_rows.max(1) {
            let Some(&top) = self.tree.first() else {
                break;
            };
            let cursor = &self.cursors[top];
            if cursor.pos >= cursor.len {
                break;
            }
            out.push((cursor.source, cursor.row_index()));
            self.advance(top)?;
            self.replay();
        }
        if out.is_empty() {
            return Ok(None);
        }
        // Interleave only the batches this output uses: an in-memory merge
        // has one source per buffered batch, and passing them all would
        // make every output batch cost O(buffered batches).
        let mut used: Vec<usize> = out.iter().map(|&(s, _)| s).collect();
        used.sort_unstable();
        used.dedup();
        let sources: Vec<Batch> = used.iter().map(|&s| self.sources[s].clone()).collect();
        for o in &mut out {
            o.0 = used
                .binary_search(&o.0)
                .map_err(|_| internal(format!("merge source {} is not in use", o.0)))?;
        }
        let batch = interleave_batches(&self.schema, &sources, &out)?;
        if self.retired {
            // Only the batches cursors are on stay referenced.
            self.sources = self.cursors.iter().map(|c| c.batch().clone()).collect();
            for (i, c) in self.cursors.iter_mut().enumerate() {
                c.source = i;
            }
            self.retired = false;
        }
        Ok(Some(batch))
    }
}

/// The rows at `indices` (`(source batch, row)`) of `sources`, weights
/// included: a constant weight shared by every source stays constant.
pub(crate) fn interleave_batches(
    schema: &SchemaRef,
    sources: &[Batch],
    indices: &[(usize, usize)],
) -> Result<Batch> {
    let columns = (0..schema.fields().len())
        .map(|c| {
            let arrays: Vec<&dyn Array> = sources
                .iter()
                .map(|b| b.data().column(c).as_ref())
                .collect();
            interleave(&arrays, indices)
        })
        .collect::<Result<Vec<ArrayRef>, _>>()
        .map_err(|e| internal(e.to_string()))?;
    let options = RecordBatchOptions::new().with_row_count(Some(indices.len()));
    let data = RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)
        .map_err(|e| internal(e.to_string()))?;
    let first = sources.first().map(Batch::weights);
    let weights = match first {
        Some(Weights::Const(w)) if sources.iter().all(|b| b.weights() == &Weights::Const(*w)) => {
            Weights::Const(*w)
        }
        _ => Weights::Column(
            indices
                .iter()
                .map(|&(b, r)| match sources.get(b).map(Batch::weights) {
                    Some(Weights::Const(w)) => Ok(*w),
                    Some(Weights::Column(c)) if r < c.len() => Ok(c.value(r)),
                    _ => Err(internal(format!("merge index ({b}, {r}) is out of range"))),
                })
                .collect::<Result<Int64Array>>()?,
        ),
    };
    Batch::with_weights(data, weights)
}

/// Merges `merger` into a new spill file, in batches of `batch_rows`,
/// checking cancellation between batches. The writer's scratch memory comes
/// from `scratch`. `Ok(Err(_))` when the memory pool refused the scratch
/// memory: nothing was spilled and the caller keeps its data.
pub(crate) fn write_run(
    merger: &mut Merger,
    scope: &SpillScope,
    scratch: Reservation,
    batch_rows: usize,
    query: &QueryContext,
) -> Result<Result<SpillRun, MemoryError>> {
    let refused = |e: SpillError| match e {
        SpillError::Memory(m) => Ok(Err(m)),
        other => Err(other.into()),
    };
    let mut writer = match scope.writer(Arc::clone(&merger.schema), scratch) {
        Ok(w) => w,
        Err(e) => return refused(e),
    };
    let (mut max_batch_bytes, mut row_bytes) = (0, 1);
    while let Some(batch) = merger.next(batch_rows)? {
        query.check_cancelled()?;
        let bytes = batch.data().get_array_memory_size();
        max_batch_bytes = max_batch_bytes.max(bytes);
        row_bytes = row_bytes.max(data_bytes(&batch).div_ceil(batch.num_rows().max(1)));
        if let Err(e) = writer.write(&batch) {
            return refused(e);
        }
    }
    match writer.finish() {
        Ok(file) => Ok(Ok(SpillRun {
            file,
            max_batch_bytes,
            row_bytes,
        })),
        Err(e) => refused(e),
    }
}
