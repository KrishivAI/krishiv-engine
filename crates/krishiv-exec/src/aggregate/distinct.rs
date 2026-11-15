// SPDX-License-Identifier: Apache-2.0
//! `count(DISTINCT v)` inside a `GROUP BY`, counted where the pairs merge
//! (design `P3-cb-distinct-in-group` §3; `krishiv.exec.agg.distinctCountAtMerge`).
//!
//! [`DistinctCount`] is one breaker over two hash aggregations of the same
//! input, fed in the same `sink` call:
//! - the **pairs**: grouped by the keys and the distinct arguments, with no
//!   aggregates (no payload);
//! - the **keys** (only when there are other aggregates): grouped by the
//!   keys, with the plain aggregates' states.
//!
//! A pair is distinct only once the drivers' pair tables merged, so it is
//! counted where the merge finishes: each in-memory final-phase task of the
//! pairs (`sink/parallel.rs`) holds every copy of its radix partitions'
//! pairs, so it counts its merged table into the shared [`PairCounter`] (a
//! sharded `key -> count` map) and frees it. No other task holds those
//! pairs, so the tasks' counts add up. Pairs that reach the output instead
//! (one driver table, the columnar tables, a spill to disk, or a task whose
//! count memory was refused) are read back and counted in `finalize`. Then
//! the key table's rows are the result, each with its key's count (a key
//! whose values were all NULL counts 0); without plain aggregates the
//! counter's keys are the result.
//!
//! No second hash aggregation reads the pairs, and the pairs carry no
//! states. Both tables spill as any aggregate's do.
//!
//! **Chosen by parallelism.** [`DistinctChoice`] picks this plan when its
//! pipelines are built, and only when they run more than one driver
//! (`PipelineBuilder::parallelism`: the executor's slots, or a constant
//! under `krishiv.deterministicOrder`). On one driver it builds the two
//! aggregations in series: every row would go into two tables here, with
//! no parallel merge to win that back.
//!
//! **Memory.** The counter does not spill. Its bound is one entry per
//! distinct key, the key table's bound. Every byte it holds is reserved
//! before it is used: a merge task counts a radix partition all or nothing
//! into partition-local shards, reserving their growth before each insert,
//! then reserves each shared shard's growth before merging into it. A
//! refused count leaves the partition (and the rest of the table) to
//! `finalize`; a refused merge keeps the unmerged local shards, still
//! reserved, for `finalize`, which waits for memory through arbitration.
//! A chunk's scratch (up to [`COUNT_ROWS`] decoded pairs and their
//! encoding) is reserved on the same consumer as it is made and released
//! after the chunk; a refusal there takes the refused-count path.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_buffer::NullBuffer;
use arrow_row::Rows;
use arrow_schema::{DataType, SchemaRef};
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{ConsumerHandle, Reservation, SpillSignal};
use krishiv_types::batch::Batch;

use super::rowtable::RowTable;
use super::sink::{HashAggregate, set_pair_counter};
use super::{AggregateFunction, AggregateOutput, AggregateSpec};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::hash::{HashSeed, KeyEncoder, KeyTable, grown};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::pipeline::{ExecNode, PipelineBuilder};
use crate::sync::{Mutex, lock};

/// Shards of the counter (the bits of the key hash above bit 32 pick one:
/// not the index's low bits, nor hashbrown's top-7-bit tag).
const SHARDS: usize = 64;
/// Pairs encoded and counted at once.
const COUNT_ROWS: usize = 4096;

/// Tests: partitions the merge hook counted (nextest runs a test per process).
#[cfg(test)]
pub(super) static HOOK_PARTS: AtomicUsize = AtomicUsize::new(0);

/// Tests: grants of the counter's non-waiting reservations left before
/// they are refused (`usize::MAX`: never refused by the test).
#[cfg(test)]
pub(super) static REFUSE_AFTER: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Tests: chunk scratch reservations refused while counting a partition.
#[cfg(test)]
pub(super) static SCRATCH_REFUSALS: AtomicUsize = AtomicUsize::new(0);

/// Tests: local shards a refused merge left to `finalize`.
#[cfg(test)]
pub(super) static PENDING_MERGES: AtomicUsize = AtomicUsize::new(0);

/// Grows `r` by `bytes` without waiting; `true` if the pool refuses (tests
/// also refuse after [`REFUSE_AFTER`] grants).
fn refused(r: &mut Reservation, bytes: usize) -> bool {
    #[cfg(test)]
    {
        let left = REFUSE_AFTER.load(Ordering::Relaxed);
        if left == 0 {
            return true;
        }
        if left != usize::MAX {
            REFUSE_AFTER.store(left - 1, Ordering::Relaxed);
        }
    }
    r.try_grow(bytes).is_err()
}

/// Shrinks `r` to `keep` bytes.
fn shrink_to(r: &mut Reservation, keep: usize) {
    r.shrink(r.size().saturating_sub(keep));
}

/// The shard of key hash `h`.
fn shard_of(h: u64) -> usize {
    usize::try_from((h >> 32) & (SHARDS as u64 - 1)).unwrap_or(0)
}

/// One shard of the counter: distinct keys and each key's count.
#[derive(Default)]
struct Shard {
    keys: KeyTable,
    counts: Vec<i64>,
}

impl Shard {
    fn bytes(&self) -> usize {
        self.keys.memory_size() + self.counts.capacity() * size_of::<i64>()
    }

    /// Bytes after inserting `rows` of `e` (each a new key at worst).
    fn projected(&self, e: &Encoded, rows: &[u32]) -> usize {
        let bytes = rows.iter().map(|&i| e.row(i).len()).sum();
        self.keys.projected_size(rows.len(), bytes)
            + grown(&self.counts, rows.len()) * size_of::<i64>()
    }

    /// Bytes after merging `piece` into this shard (each key new at worst).
    fn merge_projected(&self, piece: &Shard) -> usize {
        let n = piece.keys.len();
        let bytes = (0..n).map(|i| piece.keys.row(i).len()).sum();
        self.keys.projected_size(n, bytes) + grown(&self.counts, n) * size_of::<i64>()
    }

    /// Adds `piece`'s counts to this shard's.
    fn merge_from(&mut self, piece: &Shard) -> Result<()> {
        let n = piece.keys.len();
        let bytes = (0..n).map(|i| piece.keys.row(i).len()).sum();
        self.keys.reserve(n, bytes);
        self.counts.reserve(n);
        for (i, (&h, &c)) in piece.keys.hashes().iter().zip(&piece.counts).enumerate() {
            let (id, new) = self.keys.insert_hashed(piece.keys.row(i), h)?;
            if new {
                self.counts.push(0);
            }
            if let Some(d) = self.counts.get_mut(id as usize) {
                *d += c;
            }
        }
        Ok(())
    }

    /// Inserts `rows` of `e`; a pair whose values are all non-NULL counts.
    fn insert(&mut self, e: &Encoded, rows: &[u32]) -> Result<()> {
        let bytes = rows.iter().map(|&i| e.row(i).len()).sum();
        self.keys.reserve(rows.len(), bytes);
        self.counts.reserve(rows.len());
        for &i in rows {
            let at = i as usize;
            let (id, new) = self
                .keys
                .insert_hashed(e.row(i), e.hashes.get(at).copied().unwrap_or(0))?;
            if new {
                self.counts.push(0);
            }
            if e.valid.as_ref().is_none_or(|v| v.is_valid(at))
                && let Some(c) = self.counts.get_mut(id as usize)
            {
                *c += 1;
            }
        }
        Ok(())
    }
}

/// A batch's keys encoded, hashed and grouped by shard.
struct Encoded {
    rows: Rows,
    hashes: Vec<u64>,
    /// Rows whose distinct values are all non-NULL (`None`: every row).
    valid: Option<NullBuffer>,
    by_shard: Vec<Vec<u32>>,
}

impl Encoded {
    fn row(&self, i: u32) -> &[u8] {
        self.rows.row(i as usize).data()
    }

    /// Bytes this encoding holds.
    fn bytes(&self) -> usize {
        self.rows.size()
            + self.hashes.capacity() * size_of::<u64>()
            + self.valid.as_ref().map_or(0, |v| v.buffer().capacity())
            + self
                .by_shard
                .iter()
                .map(|r| size_of::<Vec<u32>>() + r.capacity() * size_of::<u32>())
                .sum::<usize>()
    }
}

/// The per-key counts of distinct pairs, shared by the final phase's tasks.
pub(crate) struct PairCounter {
    /// Key columns at the start of a pairs batch; the rest are the values.
    keys: usize,
    encoder: KeyEncoder,
    seed: HashSeed,
    shards: Vec<Mutex<Shard>>,
    /// Reservations covering exactly what the shards grew by.
    held: Mutex<Vec<Reservation>>,
    /// Counted radix partitions whose merge into the shards was refused
    /// memory: their local shards with the reservation covering them,
    /// merged by `finalize` (which can wait for memory).
    pending: Mutex<Vec<(Vec<Shard>, Reservation)>>,
    consumer: Arc<ConsumerHandle>,
    /// A merge task was refused memory: the next tasks publish their pairs
    /// for `finalize`.
    short: AtomicBool,
}

/// Bytes of `pieces`.
fn pieces_bytes(pieces: &[Shard]) -> usize {
    pieces.iter().map(Shard::bytes).sum()
}

impl PairCounter {
    fn new(types: &[DataType], seed: HashSeed, consumer: Arc<ConsumerHandle>) -> Result<Self> {
        Ok(Self {
            keys: types.len(),
            encoder: KeyEncoder::new(types)?,
            seed,
            shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
            held: Mutex::new(Vec::new()),
            pending: Mutex::new(Vec::new()),
            consumer,
            short: AtomicBool::new(false),
        })
    }

    /// Encodes the key columns `keys` and, when `values` are given, which
    /// rows have all of them non-NULL.
    fn encode(&self, keys: &[ArrayRef], values: &[ArrayRef]) -> Result<Encoded> {
        let rows = self.encoder.encode(keys)?;
        let n = rows.num_rows();
        let mut valid: Option<NullBuffer> = None;
        for v in values {
            valid = NullBuffer::union(valid.as_ref(), v.logical_nulls().as_ref());
        }
        let mut hashes = Vec::with_capacity(n);
        let mut by_shard: Vec<Vec<u32>> = vec![Vec::new(); SHARDS];
        for i in 0..n {
            let h = self.seed.hash(rows.row(i).data());
            hashes.push(h);
            if let Some(s) = by_shard.get_mut(shard_of(h)) {
                s.push(
                    u32::try_from(i).map_err(|_| internal("a batch past 2^32 rows".to_owned()))?,
                );
            }
        }
        Ok(Encoded {
            rows,
            hashes,
            valid,
            by_shard,
        })
    }

    fn encode_pairs(&self, b: &RecordBatch) -> Result<Encoded> {
        let cols = b.columns();
        let (keys, values) = cols.split_at(self.keys.min(cols.len()));
        self.encode(keys, values)
    }

    fn shard(&self, s: usize) -> Result<&Mutex<Shard>> {
        self.shards
            .get(s)
            .ok_or_else(|| internal("counter shard out of range".to_owned()))
    }

    /// Counts a finished final-phase table of distinct pairs (the merge
    /// hook, on the blocking pool), one radix partition at a time, each all
    /// or nothing: the partition is counted into partition-local shards
    /// whose growth is reserved before each insert (a chunk's decoded pairs
    /// and their encoding are reserved too, until the chunk is counted),
    /// then merged into the
    /// shared shards and freed. A partition refused memory while counted is
    /// left in the table, with every partition after it, for `finalize` to
    /// count from the output; one refused while merged keeps its unmerged
    /// local shards (still reserved) for `finalize` to merge.
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
            let mut local: Vec<Shard> = (0..SHARDS).map(|_| Shard::default()).collect();
            let mut r = self.consumer.reservation();
            // One chunk's scratch (its decoded pairs, then their encoding),
            // reserved as it is made and released after the chunk; refused,
            // the partition stays whole, as for the shards' growth.
            let mut scratch = self.consumer.reservation();
            for start in (0..len).step_by(COUNT_ROWS) {
                query.check_cancelled()?;
                let ids = t.part_ids(p, start..(start + COUNT_ROWS).min(len));
                let batch = t.output_batch(&ids)?;
                if refused(&mut scratch, batch.get_array_memory_size()) {
                    #[cfg(test)]
                    SCRATCH_REFUSALS.fetch_add(1, Ordering::Relaxed);
                    self.short.store(true, Ordering::Release);
                    return Ok(());
                }
                let e = self.encode_pairs(&batch)?;
                if refused(&mut scratch, e.bytes()) {
                    #[cfg(test)]
                    SCRATCH_REFUSALS.fetch_add(1, Ordering::Relaxed);
                    self.short.store(true, Ordering::Release);
                    return Ok(());
                }
                drop(batch);
                shrink_to(&mut scratch, e.bytes());
                let mut grow = 0;
                for (piece, rows) in local.iter().zip(&e.by_shard) {
                    if !rows.is_empty() {
                        grow += piece.projected(&e, rows).saturating_sub(piece.bytes());
                    }
                }
                if refused(&mut r, grow) {
                    // Nothing of this partition reached the shards: the
                    // local shards and their reservation go here.
                    self.short.store(true, Ordering::Release);
                    return Ok(());
                }
                for (piece, rows) in local.iter_mut().zip(&e.by_shard) {
                    if !rows.is_empty() {
                        piece.insert(&e, rows)?;
                    }
                }
                drop(e);
                shrink_to(&mut scratch, 0);
                shrink_to(&mut r, pieces_bytes(&local));
            }
            // Counted: the partition's pairs are no longer needed.
            t.release_part(p);
            #[cfg(test)]
            HOOK_PARTS.fetch_add(1, Ordering::Relaxed);
            if let Some(rest) = self.merge_now(local, r)? {
                lock(&self.pending).push(rest);
                self.short.store(true, Ordering::Release);
                return Ok(());
            }
        }
        Ok(())
    }

    /// Merges counted local shards (covered by `r`) into the shared ones,
    /// reserving each shard's growth first (never waiting). Returns the
    /// local shards not merged, with their reservation, if one was refused.
    fn merge_now(
        &self,
        mut local: Vec<Shard>,
        mut r: Reservation,
    ) -> Result<Option<(Vec<Shard>, Reservation)>> {
        for s in 0..local.len() {
            let Some(piece) = local.get_mut(s) else { break };
            if piece.keys.is_empty() {
                continue;
            }
            let mut shard = lock(self.shard(s)?);
            let before = shard.bytes();
            let delta = shard.merge_projected(piece).saturating_sub(before);
            if refused(&mut r, delta) {
                drop(shard);
                // Shards before `s` are merged (and empty): keep the
                // places so each piece still lands in its own shard.
                shrink_to(&mut r, pieces_bytes(&local));
                return Ok(Some((local, r)));
            }
            shard.merge_from(piece)?;
            let grew = shard.bytes().saturating_sub(before);
            drop(shard);
            let moved = r.split(grew.min(r.size()));
            lock(&self.held).push(moved);
            *piece = Shard::default();
            shrink_to(&mut r, pieces_bytes(&local));
        }
        Ok(None)
    }

    /// Merges the local shards the merge tasks could not, waiting for the
    /// memory (in `finalize`: no task counts at the same time).
    async fn merge_pending(&self) -> Result<()> {
        let pending = std::mem::take(&mut *lock(&self.pending));
        #[cfg(test)]
        PENDING_MERGES.fetch_add(pending.len(), Ordering::Relaxed);
        for (mut local, mut r) in pending {
            for s in 0..local.len() {
                let delta = {
                    let Some(piece) = local.get(s) else { break };
                    if piece.keys.is_empty() {
                        continue;
                    }
                    let shard = lock(self.shard(s)?);
                    shard.merge_projected(piece).saturating_sub(shard.bytes())
                };
                r.grow(delta).await.map_err(KrishivError::from)?;
                let Some(piece) = local.get_mut(s) else { break };
                let mut shard = lock(self.shard(s)?);
                let before = shard.bytes();
                shard.merge_from(piece)?;
                let grew = shard.bytes().saturating_sub(before);
                drop(shard);
                let moved = r.split(grew.min(r.size()));
                lock(&self.held).push(moved);
                *piece = Shard::default();
                shrink_to(&mut r, pieces_bytes(&local));
            }
        }
        Ok(())
    }

    /// Counts a batch of pairs read from the output, waiting for each
    /// shard's growth (in `finalize`: no task counts at the same time).
    async fn count_waiting(&self, b: &RecordBatch) -> Result<()> {
        let e = self.encode_pairs(b)?;
        for (s, rows) in e.by_shard.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let delta = {
                let shard = lock(self.shard(s)?);
                shard.projected(&e, rows).saturating_sub(shard.bytes())
            };
            let mut r = self.consumer.reservation();
            r.grow(delta).await.map_err(KrishivError::from)?;
            let mut shard = lock(self.shard(s)?);
            let before = shard.bytes();
            shard.insert(&e, rows)?;
            let grew = shard.bytes().saturating_sub(before);
            drop(shard);
            shrink_to(&mut r, grew);
            lock(&self.held).push(r);
        }
        Ok(())
    }

    /// The count of each row of `keys` (0 for a key never counted).
    fn counts_of(&self, keys: &[ArrayRef]) -> Result<Int64Array> {
        let e = self.encode(keys, &[])?;
        let mut out = vec![0_i64; e.rows.num_rows()];
        for (s, rows) in e.by_shard.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let shard = lock(self.shard(s)?);
            for &i in rows {
                let at = i as usize;
                let h = e.hashes.get(at).copied().unwrap_or(0);
                if let Some(c) = shard
                    .keys
                    .find(e.row(i), h)
                    .and_then(|id| shard.counts.get(id as usize))
                    && let Some(o) = out.get_mut(at)
                {
                    *o = *c;
                }
            }
        }
        Ok(Int64Array::from(out))
    }

    /// Rows `range` of shard `s`: its keys' columns and their counts.
    fn shard_rows(&self, s: usize, range: std::ops::Range<usize>) -> Result<Vec<ArrayRef>> {
        let shard = lock(self.shard(s)?);
        let mut cols = self.encoder.decode(&shard.keys, range.clone())?;
        let counts = shard
            .counts
            .get(range)
            .ok_or_else(|| internal("counter rows out of range".to_owned()))?;
        cols.push(Arc::new(Int64Array::from(counts.to_vec())));
        Ok(cols)
    }

    fn shard_len(&self, s: usize) -> usize {
        self.shards.get(s).map_or(0, |m| lock(m).keys.len())
    }
}

/// The breaker of a `count(DISTINCT)` counted at the merge (see the
/// [module docs](self)).
#[derive(Debug)]
pub(crate) struct DistinctCount {
    pairs: Arc<HashAggregate>,
    /// The plain aggregates by key (`None`: there are none).
    keys: Option<Arc<HashAggregate>>,
    key_types: Vec<DataType>,
    /// The distinct count's place among the aggregates.
    at: usize,
    schema: SchemaRef,
}

/// Whether [`DistinctCount`] covers `spec` with the distinct flags
/// `distinct`: one distinct aggregate, a `count` without a mask, under a
/// `GROUP BY` whose keys the arrow-row encoder takes.
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
        let plain: Vec<_> = spec
            .aggregates
            .iter()
            .zip(distinct)
            .filter(|(_, d)| !**d)
            .map(|(a, _)| a.clone())
            .collect();
        let keys = if plain.is_empty() {
            None
        } else {
            Some(HashAggregate::new(AggregateSpec {
                aggregates: plain,
                require_unit_weights: true,
                ..spec.clone()
            })?)
        };
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
            key_types,
            at,
            schema: spec.output_schema()?,
        }))
    }

    /// The pairs aggregate (tests read its counters).
    #[cfg(test)]
    pub(crate) fn pairs(&self) -> &Arc<HashAggregate> {
        &self.pairs
    }

    /// The result's columns: the key table's (keys, then the plain
    /// aggregates) with the count put in its place.
    fn with_count(&self, b: &RecordBatch, counter: &PairCounter) -> Result<RecordBatch> {
        let nk = self.key_types.len();
        let mut cols = b.columns().to_vec();
        let keys = cols
            .get(..nk)
            .ok_or_else(|| internal("key table output without its keys".to_owned()))?;
        let counts: ArrayRef = Arc::new(counter.counts_of(keys)?);
        let at = (nk + self.at).min(cols.len());
        cols.insert(at, counts);
        RecordBatch::try_new(Arc::clone(&self.schema), cols).map_err(|e| internal(e.to_string()))
    }
}

struct Global {
    pairs: GlobalState,
    keys: Option<GlobalState>,
    counter: Arc<PairCounter>,
    /// The next shard a reader of the counter claims (no key table).
    next: AtomicUsize,
    batch: usize,
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

struct Local {
    pairs: LocalState,
    keys: Option<LocalState>,
    /// The batch the key table applied while the pairs table was refused
    /// memory: the retry applies it to the pairs only.
    keys_done: Option<BatchId>,
    /// The driver's spill signal: it gets the sum of both tables'
    /// spillable bytes (each sub-aggregate writes only its own).
    signal: Arc<SpillSignal>,
}

fn local(l: &mut LocalState) -> Result<&mut Local> {
    l.downcast_mut::<Local>()
        .ok_or_else(|| internal("wrong local state for DistinctCount".to_owned()))
}

impl DistinctCount {
    /// Reports both tables' spillable bytes, summed, to the driver's signal.
    fn report_local(&self, l: &Local) {
        let keys = match (&self.keys, &l.keys) {
            (Some(k), Some(lk)) => k.spillable_local(lk),
            _ => Some(0),
        };
        if let (Some(a), Some(b)) = (keys, self.pairs.spillable_local(&l.pairs)) {
            l.signal.set_spillable_bytes(a + b);
        }
    }

    /// The combined tables' spillable bytes of both aggregates, summed.
    fn spillable_global(&self, g: &Global) -> Option<usize> {
        let keys = match (&self.keys, &g.keys) {
            (Some(k), Some(gk)) => k.spillable_global(gk)?,
            _ => 0,
        };
        Some(keys + self.pairs.spillable_global(&g.pairs)?)
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
        let keys = match &self.keys {
            Some(k) => Some(Sink::global_state(&**k, ctx)?),
            None => None,
        };
        let query = ctx.query();
        let consumer =
            ConsumerHandle::register(Arc::clone(query.pool()), "DistinctCount counts", None);
        let counter = Arc::new(PairCounter::new(
            &self.key_types,
            HashSeed::new(query.config().seed),
            consumer,
        )?);
        set_pair_counter(&pairs, Arc::clone(&counter))?;
        Ok(Arc::new(Global {
            pairs,
            keys,
            counter,
            next: AtomicUsize::new(0),
            batch: query.config().batch_size.max(1),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, g: &GlobalState) -> Result<LocalState> {
        let g = global(g)?;
        let keys = match (&self.keys, &g.keys) {
            (Some(k), Some(gk)) => Some(Sink::local_state(&**k, ctx, gk)?),
            _ => None,
        };
        Ok(Box::new(Local {
            pairs: Sink::local_state(&*self.pairs, ctx, &g.pairs)?,
            keys,
            keys_done: None,
            signal: Arc::clone(ctx.consumer().signal()),
        }))
    }

    fn sink(&self, input: Batch, l: &mut LocalState, g: &GlobalState) -> Result<SinkStatus> {
        let l = local(l)?;
        let g = global(g)?;
        let id = batch_id(&input);
        if let (Some(k), Some(lk), Some(gk)) = (&self.keys, l.keys.as_mut(), &g.keys)
            && l.keys_done != Some(id)
        {
            match k.sink(input.clone(), lk, gk)? {
                SinkStatus::NeedMoreInput => l.keys_done = Some(id),
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
            .downcast::<Local>()
            .map_err(|_| internal("wrong local state for DistinctCount".to_owned()))?;
        let g = global(g)?;
        if l.keys_done.is_some() {
            return Err(internal(
                "DistinctCount: input ended with a batch applied to the keys only".to_owned(),
            ));
        }
        let Local {
            pairs,
            keys,
            signal,
            ..
        } = *l;
        if let (Some(k), Some(lk), Some(gk)) = (&self.keys, keys, &g.keys) {
            k.combine(lk, gk)?;
        }
        self.pairs.combine(pairs, &g.pairs)?;
        if let Some(bytes) = self.spillable_global(g) {
            signal.set_spillable_bytes(bytes);
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        g: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global(g)?;
            if let Some(bytes) = self.spillable_global(g) {
                ctx.consumer().signal().set_spillable_bytes(bytes);
            }
            if let (Some(k), Some(gk)) = (&self.keys, &g.keys) {
                k.finalize(gk, ctx).await?;
            }
            // The pairs' combined tables are what is left to spill.
            if let Some(bytes) = self.pairs.spillable_global(&g.pairs) {
                ctx.consumer().signal().set_spillable_bytes(bytes);
            }
            self.pairs.finalize(&g.pairs, ctx).await?;
            g.counter.merge_pending().await?;
            // What the merge tasks did not count: read and count it here.
            let mut reader = Source::local_state(&*self.pairs, ctx, &g.pairs)?;
            while let Some(b) = self.pairs.get_data(&mut reader, &g.pairs).await? {
                ctx.query().check_cancelled()?;
                g.counter.count_waiting(b.data()).await?;
            }
            Ok(())
        })
    }

    fn spill<'a>(
        &'a self,
        l: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = local(l)?;
            let mut freed = 0;
            if let (Some(k), Some(lk)) = (&self.keys, l.keys.as_mut()) {
                freed += k.spill(lk, target, ctx).await?;
            }
            freed += self.pairs.spill(&mut l.pairs, target, ctx).await?;
            self.report_local(l);
            Ok(freed)
        })
    }

    fn spill_global<'a>(
        &'a self,
        g: &'a GlobalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let g = global(g)?;
            let mut freed = 0;
            if let (Some(k), Some(gk)) = (&self.keys, &g.keys) {
                freed += k.spill_global(gk, target, ctx).await?;
            }
            freed += self.pairs.spill_global(&g.pairs, target, ctx).await?;
            if let Some(bytes) = self.spillable_global(g) {
                ctx.consumer().signal().set_spillable_bytes(bytes);
            }
            Ok(freed)
        })
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        Sink::extra_metrics(&*self.pairs)
    }
}

/// A reader's state: the key table's reader, or a claimed counter shard.
enum Reader {
    Keys(LocalState),
    Shards { shard: usize, at: usize },
}

impl Source for DistinctCount {
    fn name(&self) -> &'static str {
        "DistinctCount"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
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
        match (&self.keys, &g.keys) {
            (Some(k), Some(gk)) => k.max_parallelism(gk),
            _ => (0..SHARDS)
                .filter(|&s| g.counter.shard_len(s) > 0)
                .count()
                .max(1),
        }
    }

    fn local_state(&self, ctx: &TaskContext, g: &GlobalState) -> Result<LocalState> {
        let g = global(g)?;
        Ok(Box::new(match (&self.keys, &g.keys) {
            (Some(k), Some(gk)) => Reader::Keys(Source::local_state(&**k, ctx, gk)?),
            _ => Reader::Shards {
                shard: usize::MAX,
                at: 0,
            },
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
            match reader {
                Reader::Keys(lk) => {
                    let (Some(k), Some(gk)) = (&self.keys, &g.keys) else {
                        return Err(internal("a key reader without a key table".to_owned()));
                    };
                    match k.get_data(lk, gk).await? {
                        Some(b) => Ok(Some(Batch::new(self.with_count(b.data(), &g.counter)?))),
                        None => Ok(None),
                    }
                }
                Reader::Shards { shard, at } => loop {
                    let len = if *shard < SHARDS {
                        g.counter.shard_len(*shard)
                    } else {
                        0
                    };
                    if *at >= len {
                        *shard = g.next.fetch_add(1, Ordering::Relaxed);
                        *at = 0;
                        if *shard >= SHARDS {
                            return Ok(None);
                        }
                        continue;
                    }
                    let end = (*at + g.batch).min(len);
                    let cols = g.counter.shard_rows(*shard, *at..end)?;
                    *at = end;
                    let b = RecordBatch::try_new(Arc::clone(&self.schema), cols)
                        .map_err(|e| internal(e.to_string()))?;
                    return Ok(Some(Batch::new(b)));
                },
            }
        })
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
        let sink = Arc::clone(&self.op) as Arc<dyn Sink>;
        let id = b.child_pipeline(sink, Arc::clone(&self.input))?;
        b.set_breaker_source(Arc::clone(&self.op) as Arc<dyn Source>, id)
            .map(drop)
    }
}

/// A covered `count(DISTINCT)` plan chosen when its pipelines are built:
/// counted at the merge ([`DistinctCountNode`]) when a pipeline runs more
/// than one driver, else the two aggregations in series. On one driver
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
