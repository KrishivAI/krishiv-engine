// SPDX-License-Identifier: Apache-2.0
//! `count(DISTINCT v)` inside a `GROUP BY`, counted where the pairs merge
//! (design `P3-cb-distinct-in-group` §3 and its 2026-11-16 addendum, option
//! (a); `krishiv.exec.agg.distinctCountAtMerge`).
//!
//! Three pipelines, each an ordinary breaker:
//! 1. [`DistinctCount`] takes the input. It feeds two hash aggregations in
//!    the same `sink` call: the **pairs**, grouped by the keys and the
//!    distinct arguments with no aggregates (no payload), and, when there
//!    are plain aggregates, the **key table**, grouped by the keys. Its
//!    `finalize` finishes the pairs only: the key table's drivers have
//!    combined, and its tables wait, spillable, on their own consumer
//!    (`sink/combined.rs`).
//! 2. The count rows: [`DistinctCount`]'s source side emits one row per key
//!    and count, and [`KeyFeed`] sinks them into the same key table. A pair
//!    is distinct only once the drivers' pair tables merged, so it is
//!    counted where the merge finishes: each in-memory final-phase task of
//!    the pairs (`sink/parallel.rs`) holds every copy of its radix
//!    partitions' pairs, so it counts each partition by key and queues the
//!    `(key, count)` rows ([`PairCounts`]). No other task holds those
//!    pairs, so the rows of one key add up. Pairs that reach the output
//!    instead (one driver table, the columnar tables, a spill to disk, or a
//!    task whose count memory was refused) are read and emitted as a row
//!    each, counting 1 when every distinct value is non-NULL, else 0.
//! 3. [`KeyFeed`]'s source side reads the key table's result: its count
//!    column is the sum of the rows' counts (a key whose values were all
//!    NULL counts 0).
//!
//! **One table holds every key.** The key table's input is the input's
//! columns, then a copy of each grouping column (the key table groups by
//! the copies), then `is_raw` (true on an input row, NULL on a count row),
//! then `count` (NULL on an input row). Count rows carry their key only in
//! the copies and NULL in every input column, the grouping columns too, so
//! an aggregate over a grouping column (`count(k)`, a FILTER on `k`) never
//! reads one. An aggregate a NULL argument leaves unchanged
//! (`sum`, `avg`, `min`, `max`, `count(x)`) reads them as written, `count(*)`
//! becomes `count(is_raw)`, and every other one reads only rows whose
//! `is_raw` is true (its mask, or `is_raw` when it has none: a count row's
//! NULL fails any mask). The count is `sum(count)`. So the key table is an
//! ordinary aggregation, spilling as any does, and nothing else holds a
//! key: every memory limit the two aggregations in series meet is met.
//!
//! **Memory.** A merge task counts a partition into a partition-local
//! table and builds its count rows, all or nothing: the local table's
//! growth, each chunk's scratch (up to [`COUNT_ROWS`] decoded pairs and
//! their encoding) and the queued rows are reserved without waiting. A
//! refusal leaves the partition (and every partition after it in this
//! query) in the pairs table, which reaches the output. Queued rows stay
//! reserved until a reader takes them. The queue has no spill hook: it
//! holds only what the pool granted without waiting, and the second
//! pipeline's readers drain it first, while the key table they feed spills
//! as any aggregate does.
//!
//! **Merge order.** The key table's tables merge in driver-partition order,
//! and the second pipeline's partition indices restart at 0, so its tables
//! may merge before the first pipeline's. Their plain aggregates' states
//! are empty (every count row is skipped), so the order changes no result.
//!
//! **Chosen by parallelism.** [`DistinctChoice`] picks this plan when its
//! pipelines are built, and only when the executor has more than one slot
//! (`PipelineBuilder::parallelism`). That is the slot count, not the input
//! pipeline's own driver count (its source's `max_parallelism` is known
//! only once it runs), so a one-driver source under many slots still takes
//! this plan. On one slot it builds the two aggregations in series: every
//! row would go into two tables here, with no parallel merge to win that
//! back. Under `krishiv.deterministicOrder` the parallelism is 1, so the
//! two aggregations run there too: this plan's rows come from the key
//! table in an order that follows which merge task queued a key first.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, new_null_array};
use arrow_buffer::NullBuffer;
use arrow_row::Rows;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_memory::{ConsumerHandle, Reservation};
use krishiv_types::batch::Batch;

use super::rowtable::RowTable;
use super::sink::{HashAggregate, set_pair_counter};
use super::{AggInput, AggregateExpr, AggregateFunction, AggregateOutput, AggregateSpec};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::hash::{HashSeed, KeyEncoder, KeyTable, grown};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::pipeline::{ExecNode, PipelineBuilder};
use crate::sync::{Mutex, lock};

/// Pairs encoded and counted at once.
const COUNT_ROWS: usize = 4096;

/// Test hooks of one [`DistinctCount`] (per operator, so tests running in
/// one process do not see each other's).
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct Hooks {
    /// Partitions the merge hook counted.
    pub(crate) parts: std::sync::atomic::AtomicUsize,
    /// Grants of the counter's reservations left before they are refused
    /// (`usize::MAX`: never refused by the test).
    pub(crate) refuse_after: std::sync::atomic::AtomicUsize,
    /// Chunk scratch reservations refused while counting a partition.
    pub(crate) scratch_refusals: std::sync::atomic::AtomicUsize,
    /// Queued rows refused memory while counting a partition.
    pub(crate) queue_refusals: std::sync::atomic::AtomicUsize,
    /// Cancel the query once a merge task counted a chunk of pairs.
    pub(crate) cancel_in_count: AtomicBool,
    /// Cancel the query once the count rows' source read a batch of pairs.
    pub(crate) cancel_in_read: AtomicBool,
}

#[cfg(test)]
impl Default for Hooks {
    fn default() -> Self {
        Self {
            parts: std::sync::atomic::AtomicUsize::new(0),
            refuse_after: std::sync::atomic::AtomicUsize::new(usize::MAX),
            scratch_refusals: std::sync::atomic::AtomicUsize::new(0),
            queue_refusals: std::sync::atomic::AtomicUsize::new(0),
            cancel_in_count: AtomicBool::new(false),
            cancel_in_read: AtomicBool::new(false),
        }
    }
}

/// Shrinks `r` to `keep` bytes.
fn shrink_to(r: &mut Reservation, keep: usize) {
    r.shrink(r.size().saturating_sub(keep));
}

/// A partition's distinct keys and each key's count.
#[derive(Default)]
struct Local {
    keys: KeyTable,
    counts: Vec<i64>,
}

impl Local {
    fn bytes(&self) -> usize {
        self.keys.memory_size() + self.counts.capacity() * size_of::<i64>()
    }

    /// Bytes after inserting every row of `e` (each a new key at worst).
    fn projected(&self, e: &Encoded) -> usize {
        let n = e.rows.num_rows();
        let bytes = (0..n).map(|i| e.row(i).len()).sum();
        self.keys.projected_size(n, bytes) + grown(&self.counts, n) * size_of::<i64>()
    }

    /// Inserts every row of `e`; a pair whose values are all non-NULL
    /// counts.
    fn insert(&mut self, e: &Encoded) -> Result<()> {
        let n = e.rows.num_rows();
        let bytes = (0..n).map(|i| e.row(i).len()).sum();
        self.keys.reserve(n, bytes);
        self.counts.reserve(n);
        for (i, &h) in e.hashes.iter().enumerate() {
            let (id, new) = self.keys.insert_hashed(e.row(i), h)?;
            if new {
                self.counts.push(0);
            }
            if e.valid.as_ref().is_none_or(|v| v.is_valid(i))
                && let Some(c) = self.counts.get_mut(id as usize)
            {
                *c += 1;
            }
        }
        Ok(())
    }
}

/// A batch of pairs' keys encoded and hashed.
struct Encoded {
    rows: Rows,
    hashes: Vec<u64>,
    /// Rows whose distinct values are all non-NULL (`None`: every row).
    valid: Option<NullBuffer>,
}

impl Encoded {
    fn row(&self, i: usize) -> &[u8] {
        self.rows.row(i).data()
    }

    /// Bytes this encoding holds.
    fn bytes(&self) -> usize {
        self.rows.size()
            + self.hashes.capacity() * size_of::<u64>()
            + self.valid.as_ref().map_or(0, |v| v.buffer().capacity())
    }
}

/// Which rows of a pairs batch have every distinct value non-NULL.
fn all_valid(values: &[ArrayRef]) -> Option<NullBuffer> {
    let mut valid: Option<NullBuffer> = None;
    for v in values {
        valid = NullBuffer::union(valid.as_ref(), v.logical_nulls().as_ref());
    }
    valid
}

/// The shape of the key table's input (see the [module docs](self)).
#[derive(Debug)]
struct KeyInput {
    /// The key table's input schema: the input's columns (all nullable),
    /// a copy of each grouping column (the key table's keys), `is_raw`,
    /// `count`.
    schema: SchemaRef,
    /// Grouping column indices of the input, in key order.
    group_by: Vec<usize>,
    /// Input columns.
    width: usize,
}

impl KeyInput {
    fn new(spec: &AggregateSpec) -> Result<Self> {
        let input = spec.input.fields();
        let mut fields: Vec<Field> = input
            .iter()
            .map(|f| f.as_ref().clone().with_nullable(true))
            .collect();
        let width = fields.len();
        for (j, &k) in spec.group_by.iter().enumerate() {
            let f = input
                .get(k)
                .ok_or_else(|| internal(format!("grouping column {k} out of range")))?;
            fields.push(f.as_ref().clone().with_name(format!("_krishiv_key{j}")));
        }
        fields.push(Field::new("_krishiv_is_raw", DataType::Boolean, true));
        fields.push(Field::new("_krishiv_count", DataType::Int64, true));
        Ok(Self {
            schema: Arc::new(Schema::new(fields)),
            group_by: spec.group_by.clone(),
            width,
        })
    }

    /// The key table's grouping columns: the copies.
    fn keys(&self) -> Vec<usize> {
        (self.width..self.width + self.group_by.len()).collect()
    }

    fn is_raw(&self) -> usize {
        self.width + self.group_by.len()
    }

    fn count(&self) -> usize {
        self.is_raw() + 1
    }

    /// An input batch with its keys copied (shared buffers), `is_raw` true
    /// and `count` NULL appended.
    fn raw(&self, b: &Batch) -> Result<Batch> {
        let n = b.num_rows();
        let mut cols = b.data().columns().to_vec();
        for &k in &self.group_by {
            let c = cols
                .get(k)
                .cloned()
                .ok_or_else(|| internal(format!("grouping column {k} out of range")))?;
            cols.push(c);
        }
        cols.push(Arc::new(BooleanArray::from(vec![true; n])));
        cols.push(new_null_array(&DataType::Int64, n));
        let data = RecordBatch::try_new(Arc::clone(&self.schema), cols)
            .map_err(|e| internal(e.to_string()))?;
        Batch::with_weights(data, b.weights().clone())
    }

    /// Count rows: `keys` (in key order) and their `counts`; every input
    /// column is NULL, the grouping columns too, so no plain aggregate
    /// reads a count row's key (`count(k)`, `sum(k)`, a FILTER on a key).
    fn counts(&self, keys: &[ArrayRef], counts: ArrayRef) -> Result<RecordBatch> {
        let n = counts.len();
        if keys.len() != self.group_by.len() {
            return Err(internal("count rows without their keys".to_owned()));
        }
        let mut cols = Vec::with_capacity(self.width + keys.len() + 2);
        for f in self.schema.fields().iter().take(self.width) {
            cols.push(new_null_array(f.data_type(), n));
        }
        cols.extend(keys.iter().cloned());
        cols.push(new_null_array(&DataType::Boolean, n));
        cols.push(counts);
        RecordBatch::try_new(Arc::clone(&self.schema), cols).map_err(|e| internal(e.to_string()))
    }
}

/// Whether a row whose arguments are all NULL leaves `a`'s state as it was.
fn ignores_null_rows(a: &AggregateExpr) -> bool {
    use AggregateFunction as F;
    !a.args.is_empty() && matches!(a.function, F::Count | F::Sum | F::Avg | F::Min | F::Max)
}

/// The key table's aggregate for `a`, a plain aggregate of the input.
fn plain_on_keys(a: &AggregateExpr, k: &KeyInput) -> AggregateExpr {
    let mut a = a.clone();
    if a.function == AggregateFunction::Count && a.args.is_empty() {
        // `count(*)` counts the input rows: those with `is_raw` set.
        a.args = vec![k.is_raw()];
        a.arg_type = DataType::Boolean;
    } else if !ignores_null_rows(&a) {
        a.mask = Some(a.mask.unwrap_or(k.is_raw()));
    }
    a
}

/// The per-partition counts the final phase's merge tasks queue for the
/// count rows' source.
pub(crate) struct PairCounts {
    /// Key columns at the start of a pairs batch; the rest are the values.
    keys: usize,
    encoder: KeyEncoder,
    seed: HashSeed,
    input: Arc<KeyInput>,
    batch: usize,
    /// Count rows, each with the reservation covering it.
    queue: Mutex<VecDeque<(RecordBatch, Reservation)>>,
    consumer: Arc<ConsumerHandle>,
    /// A merge task was refused memory: the next tasks publish their pairs.
    short: AtomicBool,
    #[cfg(test)]
    hooks: Arc<Hooks>,
}

impl PairCounts {
    /// Grows `r` by `bytes` without waiting; `true` if the pool refuses
    /// (tests also refuse after [`Hooks::refuse_after`] grants).
    fn refused(&self, r: &mut Reservation, bytes: usize) -> bool {
        #[cfg(test)]
        {
            let h = &self.hooks.refuse_after;
            let left = h.load(Ordering::Relaxed);
            if left == 0 {
                return true;
            }
            if left != usize::MAX {
                h.store(left - 1, Ordering::Relaxed);
            }
        }
        r.try_grow(bytes).is_err()
    }

    /// Encodes a pairs batch's keys and which rows have every value
    /// non-NULL.
    fn encode_pairs(&self, b: &RecordBatch) -> Result<Encoded> {
        let cols = b.columns();
        let (keys, values) = cols.split_at(self.keys.min(cols.len()));
        let rows = self.encoder.encode(keys)?;
        let hashes = (0..rows.num_rows())
            .map(|i| self.seed.hash(rows.row(i).data()))
            .collect();
        Ok(Encoded {
            rows,
            hashes,
            valid: all_valid(values),
        })
    }

    /// The count rows of `local`, in batches.
    fn local_rows(&self, local: &Local) -> Result<Vec<RecordBatch>> {
        let n = local.keys.len();
        let mut out = Vec::with_capacity(n.div_ceil(self.batch));
        for start in (0..n).step_by(self.batch) {
            let end = (start + self.batch).min(n);
            let keys = self.encoder.decode(&local.keys, start..end)?;
            let counts = local
                .counts
                .get(start..end)
                .ok_or_else(|| internal("counts out of range".to_owned()))?;
            out.push(
                self.input
                    .counts(&keys, Arc::new(Int64Array::from(counts.to_vec())))?,
            );
        }
        Ok(out)
    }

    /// A pairs batch read from the output as count rows: 1 for a pair
    /// whose values are all non-NULL, else 0.
    fn pair_rows(&self, b: &RecordBatch) -> Result<RecordBatch> {
        let cols = b.columns();
        let (keys, values) = cols.split_at(self.keys.min(cols.len()));
        let counts: Int64Array = match all_valid(values) {
            None => Int64Array::from(vec![1_i64; b.num_rows()]),
            Some(v) => (0..b.num_rows())
                .map(|i| Some(i64::from(v.is_valid(i))))
                .collect(),
        };
        self.input.counts(keys, Arc::new(counts))
    }

    /// Counts a finished final-phase table of distinct pairs (the merge
    /// hook, on the blocking pool), one radix partition at a time, each all
    /// or nothing: the partition is counted into a local table whose growth
    /// is reserved before each insert (a chunk's decoded pairs and their
    /// encoding are reserved too, until the chunk is counted), its count
    /// rows are built and reserved, then queued, and the partition is
    /// freed. A partition refused memory is left in the table, with every
    /// partition after it, for the output.
    ///
    /// # Errors
    /// Cancellation; decoding or encoding errors.
    pub(crate) fn count_table(&self, t: &mut RowTable, query: &QueryContext) -> Result<()> {
        for p in 0..t.part_count() {
            if self.short.load(Ordering::Acquire) {
                return Ok(());
            }
            let len = t.part_len(p);
            if len == 0 {
                continue;
            }
            let Some((rows, r)) = self.count_part(t, p, len, query)? else {
                self.short.store(true, Ordering::Release);
                return Ok(());
            };
            // Counted: the partition's pairs are no longer needed.
            t.release_part(p);
            #[cfg(test)]
            self.hooks.parts.fetch_add(1, Ordering::Relaxed);
            self.enqueue(rows, r);
        }
        Ok(())
    }

    /// Partition `p` (of `len` pairs) as count rows and the reservation
    /// covering them; `None` if memory was refused.
    fn count_part(
        &self,
        t: &RowTable,
        p: usize,
        len: usize,
        query: &QueryContext,
    ) -> Result<Option<(Vec<RecordBatch>, Reservation)>> {
        let mut local = Local::default();
        let mut r = self.consumer.reservation();
        let mut scratch = self.consumer.reservation();
        for start in (0..len).step_by(COUNT_ROWS) {
            query.check_cancelled()?;
            let ids = t.part_ids(p, start..(start + COUNT_ROWS).min(len));
            let batch = t.output_batch(&ids)?;
            if self.refused(&mut scratch, batch.get_array_memory_size()) {
                #[cfg(test)]
                self.hooks.scratch_refusals.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            let e = self.encode_pairs(&batch)?;
            if self.refused(&mut scratch, e.bytes()) {
                #[cfg(test)]
                self.hooks.scratch_refusals.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            drop(batch);
            shrink_to(&mut scratch, e.bytes());
            let grow = local.projected(&e).saturating_sub(local.bytes());
            if self.refused(&mut r, grow) {
                return Ok(None);
            }
            local.insert(&e)?;
            drop(e);
            shrink_to(&mut scratch, 0);
            shrink_to(&mut r, local.bytes());
            #[cfg(test)]
            if self.hooks.cancel_in_count.load(Ordering::Relaxed) {
                query.cancel();
            }
        }
        // The rows are reserved before they are built (their keys decoded,
        // their counts and NULL columns: about the local table's bytes),
        // then to their measured size.
        if self.refused(&mut scratch, local.bytes()) {
            #[cfg(test)]
            self.hooks.queue_refusals.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let rows = self.local_rows(&local)?;
        let bytes: usize = rows.iter().map(RecordBatch::get_array_memory_size).sum();
        let more = bytes.saturating_sub(scratch.size());
        if more > 0 && self.refused(&mut scratch, more) {
            #[cfg(test)]
            self.hooks.queue_refusals.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        shrink_to(&mut scratch, bytes);
        // The rows replace the local table.
        drop(local);
        drop(r);
        Ok(Some((rows, scratch)))
    }

    /// Queues `rows`, each with its share of `r`.
    fn enqueue(&self, rows: Vec<RecordBatch>, mut r: Reservation) {
        let mut q = lock(&self.queue);
        for b in rows {
            let share = r.split(b.get_array_memory_size().min(r.size()));
            q.push_back((b, share));
        }
    }

    /// The next queued count rows (their reservation released).
    fn pop(&self) -> Option<RecordBatch> {
        lock(&self.queue).pop_front().map(|(b, _)| b)
    }

    fn queued(&self) -> usize {
        lock(&self.queue).len()
    }
}

/// The breaker of a `count(DISTINCT)` counted at the merge (see the
/// [module docs](self)): the sink of the input's pipeline and the source of
/// the count rows.
#[derive(Debug)]
pub(crate) struct DistinctCount {
    pairs: Arc<HashAggregate>,
    /// The key table, over [`KeyInput`].
    keys: Arc<HashAggregate>,
    /// The input feeds the key table (there are plain aggregates).
    plain: bool,
    input: Arc<KeyInput>,
    key_types: Vec<DataType>,
    /// The distinct count's place among the aggregates.
    at: usize,
    schema: SchemaRef,
    #[cfg(test)]
    hooks: Arc<Hooks>,
}

/// Whether [`DistinctCount`] covers `spec` with the distinct flags
/// `distinct`: one distinct aggregate, a `count` without a mask, under a
/// `GROUP BY` whose keys the arrow-row encoder takes, every aggregate over
/// raw input.
pub(crate) fn covers(spec: &AggregateSpec, distinct: &[bool]) -> bool {
    let mut flagged = spec.aggregates.iter().zip(distinct).filter(|(_, d)| **d);
    let Some((a, _)) = flagged.next() else {
        return false;
    };
    if flagged.next().is_some()
        || a.function != AggregateFunction::Count
        || a.mask.is_some()
        || a.args.is_empty()
        || spec.group_by.is_empty()
        || spec.aggregates.iter().any(|a| a.input != AggInput::Raw)
    {
        return false;
    }
    let types: Option<Vec<DataType>> = spec
        .group_by
        .iter()
        .map(|&k| spec.input.fields().get(k).map(|f| f.data_type().clone()))
        .collect();
    types.is_some_and(|t| KeyEncoder::new(&t).is_ok())
}

impl DistinctCount {
    /// The breaker for `spec` (raw input, final output) whose distinct
    /// aggregate is flagged in `distinct` ([`covers`] holds).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a spec [`covers`] refuses; typing errors.
    pub(crate) fn new(spec: &AggregateSpec, distinct: &[bool]) -> Result<Arc<Self>> {
        Self::with_pairs(spec, distinct, HashAggregate::new)
    }

    /// [`new`](Self::new) with key-routed pairs (design P3-A 1d): the pairs'
    /// radix partitions come from the hash of the grouping columns alone
    /// ([`HashAggregate::routed`]), so each merge task holds every pair of
    /// its keys and queues one count row per key.
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub(crate) fn routed(spec: &AggregateSpec, distinct: &[bool]) -> Result<Arc<Self>> {
        let k = spec.group_by.len();
        Self::with_pairs(spec, distinct, |s| HashAggregate::routed(s, k))
    }

    /// [`new`](Self::new) whose pairs aggregate `make` builds (tests force
    /// row modes and early widening).
    pub(crate) fn with_pairs(
        spec: &AggregateSpec,
        distinct: &[bool],
        make: impl FnOnce(AggregateSpec) -> Result<Arc<HashAggregate>>,
    ) -> Result<Arc<Self>> {
        if !covers(spec, distinct) || spec.output != AggregateOutput::Final {
            return Err(internal(
                "a distinct count at the merge of an uncovered aggregate".to_owned(),
            ));
        }
        let at = distinct
            .iter()
            .position(|d| *d)
            .ok_or_else(|| internal("no distinct aggregate".to_owned()))?;
        let count = spec
            .aggregates
            .get(at)
            .ok_or_else(|| internal("distinct flag out of range".to_owned()))?;
        let pairs = make(AggregateSpec {
            group_by: spec.group_by.iter().chain(&count.args).copied().collect(),
            aggregates: Vec::new(),
            output: AggregateOutput::Partial,
            require_unit_weights: true,
            ..spec.clone()
        })?;
        let input = Arc::new(KeyInput::new(spec)?);
        let aggregates = spec
            .aggregates
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if i == at {
                    AggregateExpr::new(
                        AggregateFunction::Sum,
                        input.count(),
                        DataType::Int64,
                        a.name.clone(),
                    )
                } else {
                    plain_on_keys(a, &input)
                }
            })
            .collect();
        let keys = HashAggregate::new(AggregateSpec {
            input: Arc::clone(&input.schema),
            group_by: input.keys(),
            aggregates,
            require_unit_weights: true,
            ..spec.clone()
        })?;
        let key_types = spec
            .group_by
            .iter()
            .map(|&k| {
                spec.input
                    .fields()
                    .get(k)
                    .map(|f| f.data_type().clone())
                    .ok_or_else(|| internal(format!("grouping column {k} out of range")))
            })
            .collect::<Result<_>>()?;
        Ok(Arc::new(Self {
            pairs,
            keys,
            plain: spec.aggregates.len() > 1,
            input,
            key_types,
            at,
            schema: spec.output_schema()?,
            #[cfg(test)]
            hooks: Arc::default(),
        }))
    }

    /// The pairs aggregate (tests read its counters).
    #[cfg(test)]
    pub(crate) fn pairs(&self) -> &Arc<HashAggregate> {
        &self.pairs
    }

    /// The key table (tests read its counters).
    #[cfg(test)]
    pub(crate) fn key_table(&self) -> &Arc<HashAggregate> {
        &self.keys
    }

    /// This operator's test hooks.
    #[cfg(test)]
    pub(crate) fn hooks(&self) -> &Arc<Hooks> {
        &self.hooks
    }

    /// The result: the key table's columns with the count's NULLs as 0,
    /// under the aggregate's schema.
    fn result(&self, b: &RecordBatch) -> Result<RecordBatch> {
        let mut cols = b.columns().to_vec();
        let at = self.key_types.len() + self.at;
        let c = cols
            .get_mut(at)
            .ok_or_else(|| internal("key table output without its count".to_owned()))?;
        if c.null_count() > 0 {
            let counts = c
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| internal("a count that is not BIGINT".to_owned()))?;
            *c = Arc::new(
                counts
                    .iter()
                    .map(|v| Some(v.unwrap_or(0)))
                    .collect::<Int64Array>(),
            );
        }
        RecordBatch::try_new(Arc::clone(&self.schema), cols).map_err(|e| internal(e.to_string()))
    }
}

struct Global {
    pairs: GlobalState,
    keys: GlobalState,
    counts: Arc<PairCounts>,
}

fn global(g: &GlobalState) -> Result<&Global> {
    g.downcast_ref::<Global>()
        .ok_or_else(|| internal("wrong global state for DistinctCount".to_owned()))
}

/// What tells a retried batch from another (as the aggregate's own).
type BatchId = (usize, Option<usize>);

fn batch_id(b: &Batch) -> BatchId {
    let first = b
        .data()
        .columns()
        .first()
        .map(|c| Arc::as_ptr(c).cast::<()>() as usize);
    (b.num_rows(), first)
}

struct Driver {
    pairs: LocalState,
    keys: Option<LocalState>,
    /// The batch the key table applied while the pairs table was refused
    /// memory: the retry applies it to the pairs only.
    keys_done: Option<BatchId>,
    /// The key table's input for the batch being applied (a retry passes
    /// the same buffers, as the key table's part-way check needs).
    keys_input: Option<(BatchId, Batch)>,
    /// The driver's spill signal: it gets the sum of both tables'
    /// spillable bytes (each sub-aggregate writes only its own).
    signal: Arc<krishiv_memory::SpillSignal>,
}

fn driver(l: &mut LocalState) -> Result<&mut Driver> {
    l.downcast_mut::<Driver>()
        .ok_or_else(|| internal("wrong local state for DistinctCount".to_owned()))
}

impl DistinctCount {
    /// Reports both tables' spillable bytes, summed, to the driver's signal.
    fn report_local(&self, l: &Driver) {
        let keys = match &l.keys {
            Some(lk) => self.keys.spillable_local(lk),
            None => Some(0),
        };
        if let (Some(a), Some(b)) = (keys, self.pairs.spillable_local(&l.pairs)) {
            l.signal.set_spillable_bytes(a + b);
        }
    }
}

impl Sink for DistinctCount {
    fn name(&self) -> &'static str {
        "DistinctCount"
    }

    fn lane_safe(&self) -> bool {
        true
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let pairs = Sink::global_state(&*self.pairs, ctx)?;
        let keys = Sink::global_state(&*self.keys, ctx)?;
        let query = ctx.query();
        let consumer =
            ConsumerHandle::register(Arc::clone(query.pool()), "DistinctCount counts", None);
        let counts = Arc::new(PairCounts {
            keys: self.key_types.len(),
            encoder: KeyEncoder::new(&self.key_types)?,
            seed: HashSeed::new(query.config().seed),
            input: Arc::clone(&self.input),
            batch: query.config().batch_size.max(1),
            queue: Mutex::new(VecDeque::new()),
            consumer,
            short: AtomicBool::new(false),
            #[cfg(test)]
            hooks: Arc::clone(&self.hooks),
        });
        set_pair_counter(&pairs, Arc::clone(&counts))?;
        Ok(Arc::new(Global {
            pairs,
            keys,
            counts,
        }))
    }

    fn local_state(&self, ctx: &TaskContext, g: &GlobalState) -> Result<LocalState> {
        let g = global(g)?;
        let keys = if self.plain {
            Some(Sink::local_state(&*self.keys, ctx, &g.keys)?)
        } else {
            None
        };
        Ok(Box::new(Driver {
            pairs: Sink::local_state(&*self.pairs, ctx, &g.pairs)?,
            keys,
            keys_done: None,
            keys_input: None,
            signal: Arc::clone(ctx.consumer().signal()),
        }))
    }

    fn sink(&self, input: Batch, l: &mut LocalState, g: &GlobalState) -> Result<SinkStatus> {
        let l = driver(l)?;
        let g = global(g)?;
        let id = batch_id(&input);
        if let Some(lk) = l.keys.as_mut()
            && l.keys_done != Some(id)
        {
            let raw = match &l.keys_input {
                Some((i, b)) if *i == id => b.clone(),
                _ => {
                    let b = self.input.raw(&input)?;
                    l.keys_input = Some((id, b.clone()));
                    b
                }
            };
            match self.keys.sink(raw, lk, &g.keys)? {
                SinkStatus::NeedMoreInput => {
                    l.keys_done = Some(id);
                    l.keys_input = None;
                }
                other => {
                    self.report_local(l);
                    return Ok(other);
                }
            }
        }
        let status = match self.pairs.sink(input, &mut l.pairs, &g.pairs)? {
            SinkStatus::NeedMoreInput => {
                l.keys_done = None;
                SinkStatus::NeedMoreInput
            }
            other => other,
        };
        self.report_local(l);
        Ok(status)
    }

    fn combine(&self, l: LocalState, g: &GlobalState) -> Result<()> {
        let l = l
            .downcast::<Driver>()
            .map_err(|_| internal("wrong local state for DistinctCount".to_owned()))?;
        let g = global(g)?;
        if l.keys_done.is_some() {
            return Err(internal(
                "DistinctCount: input ended with a batch applied to the keys only".to_owned(),
            ));
        }
        let Driver {
            pairs,
            keys,
            signal,
            ..
        } = *l;
        if let Some(lk) = keys {
            self.keys.combine(lk, &g.keys)?;
        }
        self.pairs.combine(pairs, &g.pairs)?;
        if let Some(bytes) = self.pairs.spillable_global(&g.pairs) {
            let keys = self.keys.spillable_global(&g.keys).unwrap_or(0);
            signal.set_spillable_bytes(bytes + keys);
        }
        Ok(())
    }

    /// Finishes the pairs; the key table goes on in [`KeyFeed`].
    fn finalize<'a>(
        &'a self,
        g: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global(g)?;
            if let Some(bytes) = self.pairs.spillable_global(&g.pairs) {
                ctx.consumer().signal().set_spillable_bytes(bytes);
            }
            self.pairs.finalize(&g.pairs, ctx).await
        })
    }

    fn spill<'a>(
        &'a self,
        l: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = driver(l)?;
            let mut freed = 0;
            if let Some(lk) = l.keys.as_mut() {
                freed += self.keys.spill(lk, target, ctx).await?;
            }
            freed += self.pairs.spill(&mut l.pairs, target, ctx).await?;
            self.report_local(l);
            Ok(freed)
        })
    }

    /// The pairs' combined tables (the key table's have their own hook).
    fn spill_global<'a>(
        &'a self,
        g: &'a GlobalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let g = global(g)?;
            let freed = self.pairs.spill_global(&g.pairs, target, ctx).await?;
            if let Some(bytes) = self.pairs.spillable_global(&g.pairs) {
                ctx.consumer().signal().set_spillable_bytes(bytes);
            }
            Ok(freed)
        })
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        Sink::extra_metrics(&*self.pairs)
    }
}

/// A reader of the count rows: queued rows first, then the pairs that
/// reached the output.
struct Reader {
    pairs: Option<LocalState>,
    #[cfg(test)]
    query: Arc<QueryContext>,
}

impl Source for DistinctCount {
    fn name(&self) -> &'static str {
        "DistinctCount"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.input.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let state = ctx
            .dependency()
            .cloned()
            .ok_or_else(|| internal("distinct count read before it was built".to_owned()))?;
        global(&state)?;
        Ok(state)
    }

    fn max_parallelism(&self, g: &GlobalState) -> usize {
        let Ok(g) = global(g) else { return 1 };
        self.pairs
            .max_parallelism(&g.pairs)
            .max(g.counts.queued())
            .max(1)
    }

    fn local_state(&self, ctx: &TaskContext, g: &GlobalState) -> Result<LocalState> {
        let g = global(g)?;
        Ok(Box::new(Reader {
            pairs: Some(Source::local_state(&*self.pairs, ctx, &g.pairs)?),
            #[cfg(test)]
            query: Arc::clone(ctx.query()),
        }))
    }

    fn get_data<'a>(
        &'a self,
        l: &'a mut LocalState,
        g: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let g = global(g)?;
            let reader = l
                .downcast_mut::<Reader>()
                .ok_or_else(|| internal("wrong source state for DistinctCount".to_owned()))?;
            if let Some(b) = g.counts.pop() {
                return Ok(Some(Batch::new(b)));
            }
            let Some(lp) = reader.pairs.as_mut() else {
                return Ok(None);
            };
            match self.pairs.get_data(lp, &g.pairs).await? {
                Some(b) => {
                    #[cfg(test)]
                    if self.hooks.cancel_in_read.load(Ordering::Relaxed) {
                        reader.query.cancel();
                    }
                    Ok(Some(Batch::new(g.counts.pair_rows(b.data())?)))
                }
                None => {
                    reader.pairs = None;
                    Ok(None)
                }
            }
        })
    }
}

/// The sink of the count rows (into the key table [`DistinctCount`]'s
/// input pipeline began) and the source of the result.
#[derive(Debug)]
pub(crate) struct KeyFeed {
    op: Arc<DistinctCount>,
}

impl Sink for KeyFeed {
    fn name(&self) -> &'static str {
        "DistinctCountKeys"
    }

    fn lane_safe(&self) -> bool {
        Sink::lane_safe(&*self.op.keys)
    }

    /// The key table's state, from the [`DistinctCount`] this pipeline's
    /// source reads.
    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let state = ctx
            .dependency()
            .ok_or_else(|| internal("count rows sunk before the pairs were built".to_owned()))?;
        Ok(Arc::clone(&global(state)?.keys))
    }

    fn local_state(&self, ctx: &TaskContext, g: &GlobalState) -> Result<LocalState> {
        Sink::local_state(&*self.op.keys, ctx, g)
    }

    fn sink(&self, input: Batch, l: &mut LocalState, g: &GlobalState) -> Result<SinkStatus> {
        self.op.keys.sink(input, l, g)
    }

    fn combine(&self, l: LocalState, g: &GlobalState) -> Result<()> {
        self.op.keys.combine(l, g)
    }

    fn finalize<'a>(
        &'a self,
        g: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        self.op.keys.finalize(g, ctx)
    }

    fn spill<'a>(
        &'a self,
        l: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        self.op.keys.spill(l, target, ctx)
    }

    fn spill_global<'a>(
        &'a self,
        g: &'a GlobalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        self.op.keys.spill_global(g, target, ctx)
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        Sink::extra_metrics(&*self.op.keys)
    }
}

impl Source for KeyFeed {
    fn name(&self) -> &'static str {
        "DistinctCountKeys"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.op.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        Source::global_state(&*self.op.keys, ctx)
    }

    fn max_parallelism(&self, g: &GlobalState) -> usize {
        self.op.keys.max_parallelism(g)
    }

    fn local_state(&self, ctx: &TaskContext, g: &GlobalState) -> Result<LocalState> {
        Source::local_state(&*self.op.keys, ctx, g)
    }

    fn get_data<'a>(
        &'a self,
        l: &'a mut LocalState,
        g: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            match self.op.keys.get_data(l, g).await? {
                Some(b) => Ok(Some(Batch::new(self.op.result(b.data())?))),
                None => Ok(None),
            }
        })
    }
}

/// The input pipeline and the count rows: [`DistinctCount`] over `input`.
#[derive(Debug)]
struct CountRows {
    input: Arc<dyn ExecNode>,
    op: Arc<DistinctCount>,
}

impl ExecNode for CountRows {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = Arc::clone(&self.op) as Arc<dyn Sink>;
        let id = b.child_pipeline(sink, Arc::clone(&self.input))?;
        b.set_breaker_source(Arc::clone(&self.op) as Arc<dyn Source>, id)
            .map(drop)
    }
}

/// A [`DistinctCount`] over `input`.
#[derive(Debug)]
pub(crate) struct DistinctCountNode {
    input: Arc<dyn ExecNode>,
    op: Arc<DistinctCount>,
}

impl DistinctCountNode {
    /// `op` over `input`.
    pub(crate) fn new(input: Arc<dyn ExecNode>, op: Arc<DistinctCount>) -> Arc<Self> {
        Arc::new(Self { input, op })
    }
}

impl ExecNode for DistinctCountNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let feed = Arc::new(KeyFeed {
            op: Arc::clone(&self.op),
        });
        let rows = Arc::new(CountRows {
            input: Arc::clone(&self.input),
            op: Arc::clone(&self.op),
        });
        let id = b.child_pipeline(Arc::clone(&feed) as Arc<dyn Sink>, rows)?;
        b.set_breaker_source(feed as Arc<dyn Source>, id).map(drop)
    }
}

/// A covered `count(DISTINCT)` plan chosen when its pipelines are built:
/// counted at the merge ([`DistinctCountNode`]) when the builder's
/// parallelism is above 1, else (one slot, or `krishiv.deterministicOrder`)
/// the two aggregations in series. On one driver
/// every input row would go into two tables (pairs and keys) with no
/// parallel merge to win that back (ClickBench q10 1.2x slower on one
/// thread).
#[derive(Debug)]
pub(crate) struct DistinctChoice {
    at_merge: Arc<DistinctCountNode>,
    series: Arc<dyn ExecNode>,
}

impl DistinctChoice {
    /// `at_merge` or `series`, the same aggregation over the same input.
    pub(crate) fn new(at_merge: Arc<DistinctCountNode>, series: Arc<dyn ExecNode>) -> Arc<Self> {
        Arc::new(Self { at_merge, series })
    }
}

impl ExecNode for DistinctChoice {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        if b.parallelism() == Some(1) {
            Arc::clone(&self.series).build_pipelines(b)
        } else {
            Arc::clone(&self.at_merge).build_pipelines(b)
        }
    }
}
