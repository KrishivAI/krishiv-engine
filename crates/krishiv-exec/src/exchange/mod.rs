// SPDX-License-Identifier: Apache-2.0
//! Exchanges between the stages of a query (spec 13 §6, design P2-T176
//! §2.3): the writer at the root of a map task's fragment and the reader at
//! each [`PhysNode::ExchangeSource`](crate::physical::PhysNode::ExchangeSource)
//! leaf of a reduce task's.
//!
//! An [`Exchange`] is a live resource of the worker
//! ([`WorkerContext::with_exchange`](crate::physical::WorkerContext::with_exchange)),
//! never plan data. Its implementations (`krishiv-shuffle`) own the
//! storage; this module owns what every implementation shares:
//!
//! - **Blocking mode only.** A map task writes its whole output, then
//!   commits it; readers start once the stage ended.
//! - **Commit before read (I10, ADR 0013 §3).** A map output is named by
//!   [`MapOutputId`] (exchange, stage attempt, map partition, task attempt).
//!   Nothing of it is visible before [`ExchangeWriter::commit`]; a writer
//!   dropped without a commit leaves nothing. A reader takes the first
//!   committed attempt of each map partition and ignores the rest.
//! - **Routing.** [`ExchangeSink`] places each row by the plan's
//!   partitioning: `Hash` by its [`HashScheme`] over `n` reduce partitions,
//!   `RoundRobin` by the routing hash of the whole row (design Q4: the same
//!   rows go to the same partitions whatever the driver interleaving),
//!   `Single`, `Broadcast` and `Range` (run as a gather, design Q3) all to
//!   partition 0, which every task reading a broadcast reads.
//! - **Memory (I11).** Rows waiting for a frame are charged to the writing
//!   driver's consumer and spillable: a spill hands them to the writer as
//!   frames, whose storage is the implementation's.
//!
//! Batches carry no weights here: weighted (IVM) frames are P4's.

use std::fmt;
use std::sync::Arc;

use arrow_array::{RecordBatch, UInt32Array};
use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_memory::{Reservation, SpillSignal};
use krishiv_types::batch::Batch;

use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::operators::{global_ref, local_mut};
use crate::physical::{ExchangeFilterPlan, ExchangeId, HashScheme, Partitioning, ValueId, Version};
use crate::pipeline::{ExecNode, PipelineBuilder};
use crate::sync::{Mutex, lock};
use crate::values::{Partial, Producer, Value, ValueService};
use krishiv_io::{DisabledReason, FilterSnapshot, KeyFilterBuilder};

/// Rows buffered per reduce partition before they go out as one frame.
const FRAME_ROWS: usize = 8192;
/// Bytes buffered per reduce partition before they go out as one frame.
/// Emitting one concatenates and encodes it: up to about this much scratch,
/// not charged to the pool.
const FRAME_BYTES: usize = 1 << 20;

/// One map task's output of one exchange: what a commit makes visible and a
/// reader asks for (I10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapOutputId {
    /// The exchange written.
    pub exchange: ExchangeId,
    /// The attempt of the stage that writes it.
    pub stage_attempt: u32,
    /// The map partition (the writing task's partition).
    pub map: u32,
    /// The task attempt.
    pub attempt: u32,
}

/// What an [`ExchangeReader`] yields: rows, then the end of the map output
/// (spec 13 §6; `Barrier` and `Watermark` are P4's).
#[derive(Debug, Clone)]
pub enum ExchangeFrame {
    /// Rows of the reduce partition read.
    Data(RecordBatch),
    /// The map output has no more rows for this reduce partition.
    End,
}

/// Where the map tasks of a stage write their exchange output and the
/// reduce tasks of the next read it (see the module docs).
pub trait Exchange: Send + Sync + fmt::Debug {
    /// A writer of the map output `output`, over `partitions` reduce
    /// partitions of rows of `schema`.
    ///
    /// # Errors
    /// The implementation's: no room for the output, no partitions.
    fn writer(
        &self,
        output: MapOutputId,
        partitions: u32,
        schema: SchemaRef,
    ) -> Result<Box<dyn ExchangeWriter>>;

    /// A reader of reduce partition `reduce` of the first committed attempt
    /// of map partition `map` of `exchange` at `stage_attempt`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when no attempt of that map partition committed.
    fn reader(
        &self,
        exchange: ExchangeId,
        stage_attempt: u32,
        map: u32,
        reduce: u32,
    ) -> Result<Box<dyn ExchangeReader>>;

    /// The frame bytes the committed map outputs of `exchange` at
    /// `stage_attempt` hold for reduce partition `reduce`, summed over the
    /// map partitions, when this exchange knows them without reading
    /// (`None` by default, and for outputs held elsewhere). A broadcast join
    /// reads it to skip a shared build that cannot fit (design
    /// `P2-row30c-m5-shared-skip`).
    fn committed_bytes(
        &self,
        exchange: ExchangeId,
        stage_attempt: u32,
        reduce: u32,
    ) -> Option<u64> {
        let _ = (exchange, stage_attempt, reduce);
        None
    }
}

/// One map task's writer. Writes are sync: they run on the writing driver
/// between batches, as spills do.
pub trait ExchangeWriter: Send {
    /// Appends `batch` to reduce partition `reduce`.
    ///
    /// # Errors
    /// A partition out of range; a failed write.
    fn write(&mut self, reduce: u32, batch: &RecordBatch) -> Result<()>;

    /// Makes the output visible to readers (unless another attempt of the
    /// same map partition committed first, which keeps its own).
    ///
    /// # Errors
    /// A failed write; the output is then not visible.
    fn commit(self: Box<Self>) -> Result<()>;
}

/// One reduce partition of one map output.
pub trait ExchangeReader: Send {
    /// The next frame; [`ExchangeFrame::End`] after the last. A local
    /// implementation may read a file here, on the reading driver, as a
    /// spill read does.
    fn next(&mut self) -> BoxFuture<'_, Result<ExchangeFrame>>;
}

/// The reduce partition of each row of `batch` among `partitions`, as
/// `partitioning` places it (see the module docs), written to `out`.
///
/// # Errors
/// An exchange without a routing function ([`HashScheme::Unpinned`]); a
/// key type the routing hash does not cover (for round-robin, any column
/// of such a type).
pub fn route(
    partitioning: &Partitioning,
    batch: &RecordBatch,
    partitions: u32,
    out: &mut Vec<u32>,
) -> Result<()> {
    out.clear();
    let columns: Vec<&dyn arrow_array::Array> = match partitioning {
        _ if partitions <= 1 => {
            out.resize(batch.num_rows(), 0);
            return Ok(());
        }
        Partitioning::Hash { keys, scheme } => {
            let mut columns = Vec::with_capacity(keys.len());
            for &k in keys {
                columns.push(
                    batch
                        .columns()
                        .get(k)
                        .ok_or_else(|| internal(format!("hash key column {k} out of range")))?
                        .as_ref(),
                );
            }
            if columns.is_empty() {
                out.resize(batch.num_rows(), 0);
                return Ok(());
            }
            return scheme.partitions(&columns, partitions, out);
        }
        Partitioning::RoundRobin => batch.columns().iter().map(AsRef::as_ref).collect(),
        Partitioning::Single | Partitioning::Broadcast | Partitioning::Range { .. } => {
            out.resize(batch.num_rows(), 0);
            return Ok(());
        }
    };
    if columns.is_empty() {
        // Rows without columns (a projection keeping none) have no hash.
        out.resize(batch.num_rows(), 0);
        return Ok(());
    }
    HashScheme::KeyGroupV1.partitions(&columns, partitions, out)
}

/// The number of reduce partitions an exchange of `partitioning` writes
/// when its readers run `tasks` tasks: one for a gather or a broadcast.
#[must_use]
pub fn reduce_partitions(partitioning: &Partitioning, tasks: u32) -> u32 {
    match partitioning {
        Partitioning::Hash { .. } | Partitioning::RoundRobin => tasks.max(1),
        Partitioning::Single | Partitioning::Broadcast | Partitioning::Range { .. } => 1,
    }
}

/// The root of a map task's fragment: routes its input's rows to reduce
/// partitions and writes them as frames through one [`ExchangeWriter`],
/// committed when every driver finished. As a source it yields nothing.
pub struct ExchangeSink {
    exchange: Arc<dyn Exchange>,
    output: MapOutputId,
    partitions: u32,
    partitioning: Partitioning,
    schema: SchemaRef,
    /// The runtime filters collected from the rows written, and where
    /// they are published (design P2-T177, cross-stage filters).
    filters: Vec<ExchangeFilterPlan>,
    publisher: Option<(Arc<dyn ValueService>, Producer)>,
}

impl fmt::Debug for ExchangeSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExchangeSink")
            .field("output", &self.output)
            .field("partitions", &self.partitions)
            .field("partitioning", &self.partitioning)
            .finish_non_exhaustive()
    }
}

struct SinkGlobal {
    /// `None` once committed.
    writer: Mutex<Option<Box<dyn ExchangeWriter>>>,
    /// The drivers' filter partials merged so far (`None`: a filter whose
    /// partial was lost, published as disabled).
    filters: Mutex<Vec<Option<KeyFilterBuilder>>>,
    /// The drivers' reservations of those partials, held until they are
    /// published (I11).
    filter_memory: Mutex<Vec<Reservation>>,
    /// The published filters' memory, held with the sink's state: a final
    /// filter that does not fit is published as its min/max range only, as
    /// a join build's.
    published: Mutex<Option<Reservation>>,
}

struct SinkLocal {
    /// The sink's global state: a spill writes through its writer.
    global: GlobalState,
    /// Rows waiting for a frame, by reduce partition.
    buffers: Vec<Buffered>,
    reservation: Reservation,
    signal: Arc<SpillSignal>,
    routes: Vec<u32>,
    /// This driver's filter partials and the memory they hold.
    filters: Vec<Option<KeyFilterBuilder>>,
    filter_memory: Reservation,
}

#[derive(Default)]
struct Buffered {
    batches: Vec<RecordBatch>,
    rows: usize,
    bytes: usize,
}

impl ExchangeSink {
    /// A sink writing the map output `output` of rows of `schema`, placed
    /// by `partitioning` over `partitions` reduce partitions, into
    /// `exchange`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for no reduce partitions, or several for a
    /// partitioning that writes one.
    pub fn new(
        exchange: Arc<dyn Exchange>,
        output: MapOutputId,
        partitioning: Partitioning,
        partitions: u32,
        schema: SchemaRef,
    ) -> Result<Self> {
        if partitions == 0 || (partitions > 1 && reduce_partitions(&partitioning, 2) == 1) {
            return Err(internal(format!(
                "exchange {} of {partitioning:?} over {partitions} partitions",
                output.exchange.0
            )));
        }
        Ok(Self {
            exchange,
            output,
            partitions,
            partitioning,
            schema,
            filters: Vec::new(),
            publisher: None,
        })
    }

    /// The same sink collecting `filters` from the rows it writes and
    /// publishing them into `values` as `producer`'s partials once its
    /// output committed (design P2-T177, cross-stage filters).
    #[must_use]
    pub fn with_filters(
        mut self,
        filters: Vec<ExchangeFilterPlan>,
        values: Arc<dyn ValueService>,
        producer: Producer,
    ) -> Self {
        self.filters = filters;
        self.publisher = Some((values, producer));
        self
    }

    fn partials(&self) -> Vec<Option<KeyFilterBuilder>> {
        // Every map task is one producer of several: a large build's
        // shared array is not built here (design P2b-R1 M5).
        self.filters
            .iter()
            .map(|f| {
                let options = crate::join::instantiated_filter_options(f.options, u32::MAX);
                KeyFilterBuilder::new(f.key_type.clone(), options)
            })
            .collect()
    }

    /// Adds the keys of `batch` to the driver's partials, reserving what
    /// they grew by; Bloom filters are dropped when that fails (an
    /// optional filter never waits for memory).
    fn collect(&self, l: &mut SinkLocal, batch: &RecordBatch) {
        for (b, f) in l.filters.iter_mut().zip(&self.filters) {
            let ok = match (b.as_mut(), batch.columns().get(f.column)) {
                // As the join's build: a NULL key matches nothing unless
                // the key is null-safe.
                (Some(builder), Some(k)) => {
                    let keep = if f.null_safe { None } else { k.logical_nulls() };
                    builder.add(k.as_ref(), keep.as_ref()).is_ok()
                }
                _ => false,
            };
            if !ok {
                *b = None;
            }
        }
        let want: usize = l
            .filters
            .iter()
            .flatten()
            .map(KeyFilterBuilder::heap_bytes)
            .sum();
        let held = l.filter_memory.size();
        if want > held && l.filter_memory.try_grow(want - held).is_err() {
            for b in l.filters.iter_mut().flatten() {
                b.drop_bloom();
            }
            let want: usize = l
                .filters
                .iter()
                .flatten()
                .map(KeyFilterBuilder::heap_bytes)
                .sum();
            if want < held {
                l.filter_memory.shrink(held - want);
            } else {
                // The few value-set bytes left are not worth failing over.
                let _ = l.filter_memory.try_grow(want - held);
            }
        }
    }

    /// Writes partition `p`'s buffered rows as one frame and frees their
    /// memory; the bytes freed.
    fn emit(&self, l: &mut SinkLocal, p: usize) -> Result<usize> {
        let Some(buffered) = l.buffers.get_mut(p) else {
            return Ok(0);
        };
        if buffered.batches.is_empty() {
            return Ok(0);
        }
        let batches = std::mem::take(&mut buffered.batches);
        let bytes = std::mem::take(&mut buffered.bytes);
        buffered.rows = 0;
        let frame = arrow_select::concat::concat_batches(&self.schema, &batches)
            .map_err(|e| internal(e.to_string()))?;
        drop(batches);
        write(&l.global, p, &frame)?;
        l.reservation.shrink(bytes.min(l.reservation.size()));
        Ok(bytes)
    }

    /// Writes every partition's buffered rows; the bytes freed.
    fn emit_all(&self, l: &mut SinkLocal) -> Result<usize> {
        let mut freed = 0;
        for p in 0..l.buffers.len() {
            freed += self.emit(l, p)?;
        }
        l.signal.set_spillable_bytes(l.reservation.size());
        Ok(freed)
    }
}

impl Sink for ExchangeSink {
    fn name(&self) -> &'static str {
        "ExchangeWrite"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        let writer =
            self.exchange
                .writer(self.output, self.partitions, Arc::clone(&self.schema))?;
        Ok(Arc::new(SinkGlobal {
            writer: Mutex::new(Some(writer)),
            filters: Mutex::new(self.partials()),
            filter_memory: Mutex::default(),
            published: Mutex::default(),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        let partitions = usize::try_from(self.partitions)
            .map_err(|_| internal(format!("{} reduce partitions", self.partitions)))?;
        Ok(Box::new(SinkLocal {
            global: Arc::clone(global),
            buffers: std::iter::repeat_with(Buffered::default)
                .take(partitions)
                .collect(),
            reservation: memory.reservation(),
            signal: Arc::clone(memory.signal()),
            routes: Vec::new(),
            filters: self.partials(),
            filter_memory: memory.reservation(),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        let l = local_mut::<SinkLocal>(local, "ExchangeSink")?;
        let data = input.data();
        if data.num_rows() == 0 {
            return Ok(SinkStatus::NeedMoreInput);
        }
        let bytes = data.get_array_memory_size();
        if l.reservation.try_grow(bytes).is_err() {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        // Taken from here on (a `NeedMemory` retry would count it twice).
        if !self.filters.is_empty() {
            self.collect(l, data);
        }
        let mut routes = std::mem::take(&mut l.routes);
        route(&self.partitioning, data, self.partitions, &mut routes)?;
        let parts = split(data, &routes)?;
        l.routes = routes;
        // The partitions' rows replace the input in memory: charge what
        // they hold.
        let held: usize = parts.iter().map(|(_, b)| b.get_array_memory_size()).sum();
        l.reservation.shrink(bytes);
        if l.reservation.try_grow(held).is_err() {
            // Hand the rows on at once rather than ask for memory with
            // half the input applied.
            for (p, b) in parts {
                write(&l.global, p, &b)?;
            }
            l.signal.set_spillable_bytes(l.reservation.size());
            return Ok(SinkStatus::NeedMoreInput);
        }
        let mut full = Vec::new();
        for (p, b) in parts {
            let buffered = l
                .buffers
                .get_mut(p)
                .ok_or_else(|| internal(format!("reduce partition {p} out of range")))?;
            buffered.rows += b.num_rows();
            buffered.bytes += b.get_array_memory_size();
            buffered.batches.push(b);
            if buffered.rows >= FRAME_ROWS || buffered.bytes >= FRAME_BYTES {
                full.push(p);
            }
        }
        for p in full {
            self.emit(l, p)?;
        }
        l.signal.set_spillable_bytes(l.reservation.size());
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, _global: &GlobalState) -> Result<()> {
        let mut l = local
            .downcast::<SinkLocal>()
            .map_err(|_| internal("wrong local state for ExchangeSink".to_owned()))?;
        self.emit_all(&mut l)?;
        if !self.filters.is_empty() {
            let g = global_ref::<SinkGlobal>(&l.global, "ExchangeSink")?;
            let mut merged = lock(&g.filters);
            for (dst, src) in merged.iter_mut().zip(std::mem::take(&mut l.filters)) {
                *dst = match (dst.take(), src) {
                    (Some(mut a), Some(b)) => {
                        a.merge(b);
                        Some(a)
                    }
                    _ => None,
                };
            }
            drop(merged);
            let memory = l.filter_memory.split(usize::MAX);
            lock(&g.filter_memory).push(memory);
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global_ref::<SinkGlobal>(global, "ExchangeSink")?;
            let writer = lock(&g.writer)
                .take()
                .ok_or_else(|| internal("an exchange committed twice".to_owned()))?;
            writer.commit()?;
            // Published only once the rows are visible: a partial whose
            // output never committed is never merged.
            if let Some((values, producer)) = &self.publisher {
                let partials = std::mem::take(&mut *lock(&g.filters));
                let finished: Vec<_> = partials
                    .into_iter()
                    .map(|b| b.map(KeyFilterBuilder::finish))
                    .collect();
                // The partials are gone: their memory funds the final
                // filters, each reserved before it is published.
                lock(&g.filter_memory).clear();
                let mut held = ctx.consumer().reservation();
                for (f, keys) in self.filters.iter().zip(finished) {
                    let snapshot = keys.map_or(
                        FilterSnapshot::Disabled(DisabledReason::ProducerFailed),
                        |keys| {
                            let keys = if held.try_grow(keys.heap_bytes()).is_ok() {
                                keys
                            } else {
                                keys.range_only()
                            };
                            FilterSnapshot::Final(Arc::new(keys))
                        },
                    );
                    values.publish(Partial {
                        value: ValueId::Filter(f.id),
                        version: Version::BATCH,
                        producer: *producer,
                        payload: Value::Filter(snapshot),
                    })?;
                }
                *lock(&g.published) = Some(held);
            }
            Ok(())
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        // The buffered rows go out as frames: the writer's storage holds
        // them from here on.
        let freed = local_mut::<SinkLocal>(local, "ExchangeSink").and_then(|l| self.emit_all(l));
        Box::pin(std::future::ready(freed))
    }
}

/// Writes `batch` to reduce partition `p` through the writer of the sink
/// whose global state is `global`.
fn write(global: &GlobalState, p: usize, batch: &RecordBatch) -> Result<()> {
    let g = global_ref::<SinkGlobal>(global, "ExchangeSink")?;
    let reduce = u32::try_from(p).map_err(|_| internal(format!("reduce partition {p}")))?;
    lock(&g.writer)
        .as_mut()
        .ok_or_else(|| internal("an exchange written after its commit".to_owned()))?
        .write(reduce, batch)
}

/// `batch`'s rows by reduce partition `routes[i]`, in input order within
/// each partition: `(partition, rows)` for each partition that has rows.
/// Each partition's rows are copied into buffers of their own, so what one
/// holds is what it is charged (a slice would report the whole batch's
/// buffers once per partition).
fn split(batch: &RecordBatch, routes: &[u32]) -> Result<Vec<(usize, RecordBatch)>> {
    let Some(&first) = routes.first() else {
        return Ok(Vec::new());
    };
    if routes.iter().all(|&r| r == first) {
        return Ok(vec![(first as usize, batch.clone())]);
    }
    let parts = routes.iter().map(|&r| r as usize).max().unwrap_or(0) + 1;
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); parts];
    for (row, &r) in routes.iter().enumerate() {
        let row = u32::try_from(row).map_err(|_| internal(format!("row {row} of a batch")))?;
        if let Some(p) = rows.get_mut(r as usize) {
            p.push(row);
        }
    }
    rows.into_iter()
        .enumerate()
        .filter(|(_, r)| !r.is_empty())
        .map(|(p, r)| {
            let taken = arrow_select::take::take_record_batch(batch, &UInt32Array::from(r))
                .map_err(|e| internal(e.to_string()))?;
            Ok((p, taken))
        })
        .collect()
}

impl Source for ExchangeSink {
    fn name(&self) -> &'static str {
        "ExchangeWriteResult"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(()))
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        _global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(std::future::ready(Ok(None)))
    }
}

/// A map task's fragment: `input` written through `sink`.
pub(crate) struct ExchangeWrite {
    pub(crate) input: Arc<dyn ExecNode>,
    pub(crate) sink: Arc<ExchangeSink>,
}

impl ExecNode for ExchangeWrite {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(
            Arc::clone(&self.sink) as Arc<dyn Sink>,
            Arc::clone(&self.input),
        )?;
        b.set_breaker_source(Arc::clone(&self.sink) as Arc<dyn Source>, sink)?;
        Ok(())
    }
}

/// What one task reads from one exchange: reduce partitions `reduce` of map
/// partitions `0..maps`, each pair a morsel.
pub struct ExchangeRead {
    exchange: Arc<dyn Exchange>,
    id: ExchangeId,
    stage_attempt: u32,
    /// `(map, reduce)` in reading order: reduce partition, then map
    /// partition.
    pairs: Vec<(u32, u32)>,
    schema: SchemaRef,
    /// Read by one driver in pair order (a gather keeps its producers'
    /// order).
    ordered: bool,
}

impl fmt::Debug for ExchangeRead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExchangeRead")
            .field("id", &self.id)
            .field("stage_attempt", &self.stage_attempt)
            .field("pairs", &self.pairs.len())
            .field("ordered", &self.ordered)
            .finish_non_exhaustive()
    }
}

struct ReadGlobal {
    next: std::sync::atomic::AtomicUsize,
}

struct ReadLocal {
    current: Option<Box<dyn ExchangeReader>>,
    claimed: u64,
}

impl ExchangeRead {
    /// A source of reduce partitions `reduce` of map partitions `0..maps`
    /// of `id` at `stage_attempt`, as rows of `schema`.
    #[must_use]
    pub fn new(
        exchange: Arc<dyn Exchange>,
        id: ExchangeId,
        stage_attempt: u32,
        reduce: &[u32],
        maps: u32,
        schema: SchemaRef,
        ordered: bool,
    ) -> Self {
        let pairs = reduce
            .iter()
            .flat_map(|&r| (0..maps).map(move |m| (m, r)))
            .collect();
        Self {
            exchange,
            id,
            stage_attempt,
            pairs,
            schema,
            ordered,
        }
    }
}

impl Source for ExchangeRead {
    fn name(&self) -> &'static str {
        "ExchangeRead"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(ReadGlobal {
            next: std::sync::atomic::AtomicUsize::new(0),
        }))
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        if self.ordered {
            1
        } else {
            self.pairs.len().max(1)
        }
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(ReadLocal {
            current: None,
            claimed: 0,
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let g = global_ref::<ReadGlobal>(global, "ExchangeRead")?;
            let l = local_mut::<ReadLocal>(local, "ExchangeRead")?;
            loop {
                if let Some(reader) = l.current.as_mut() {
                    match reader.next().await? {
                        ExchangeFrame::Data(b) if b.num_rows() == 0 => continue,
                        ExchangeFrame::Data(b) => {
                            if b.schema().fields() != self.schema.fields() {
                                return Err(internal(format!(
                                    "exchange {} read {:?} where it planned {:?}",
                                    self.id.0,
                                    b.schema().fields(),
                                    self.schema.fields()
                                )));
                            }
                            return Ok(Some(Batch::new(b)));
                        }
                        ExchangeFrame::End => l.current = None,
                    }
                    continue;
                }
                let i = g.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(map, reduce)) = self.pairs.get(i) else {
                    return Ok(None);
                };
                l.claimed += 1;
                l.current = Some(
                    self.exchange
                        .reader(self.id, self.stage_attempt, map, reduce)?,
                );
            }
        })
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        local.downcast_ref::<ReadLocal>().map_or(0, |l| l.claimed)
    }
}

#[cfg(test)]
mod tests;
