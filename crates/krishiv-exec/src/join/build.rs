// SPDX-License-Identifier: Apache-2.0
//! The build side: a [`Sink`] that partitions build rows into row-arena
//! chunks, spills whole partitions on request, and at `finalize` indexes the
//! partitions still in memory.

use std::sync::{Arc, OnceLock, Weak};

use arrow_array::{ArrayRef, Int64Array, LargeBinaryArray, UInt64Array};
use arrow_row::Rows;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_io::{DisabledReason, FilterId, FilterSnapshot, KeyFilterBuilder};
use krishiv_memory::{ConsumerHandle, Reservation, SpillFile, SpillScope, SpillSignal};
use krishiv_types::batch::{Batch, Weights};

use super::grace::{
    Grace, Part, SPILL_MIN, SPILL_SCRATCH, choose, reserve_chunks, spill_parts, write_sliced,
};
use super::keys::{Encoded, Hashing, any_invalid};
use super::table::{JoinTable, RowArena, key_bytes};
use super::{Counter, FilterPublisher, JoinType, Shared, record, unsupported};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, run_blocking};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::output_spill::{OutputSpill, SpillUnread};
use crate::rows::Hll;
use crate::sync::{Event, Mutex, lock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The build sink.
#[derive(Debug)]
pub(crate) struct BuildSink(pub(crate) Arc<Shared>);

/// The build's global state; after `finalize`, what the probe reads.
#[derive(Debug)]
pub(crate) struct BuildGlobal {
    /// The join instance's key format and seed.
    pub(crate) hashing: Hashing,
    /// Level-0 partitions spilled by any driver (bit per partition).
    spilled: Arc<AtomicU64>,
    /// Build rows, NULL keys included (null-aware anti: is it empty?).
    pub(crate) rows: AtomicU64,
    /// A build key was NULL (null-aware anti).
    pub(crate) null_key: AtomicBool,
    collect: Mutex<Collect>,
    /// The finalized build, once a probe driver read it ([`Self::built`]).
    built: OnceLock<Built>,
    /// Between `finalize` and the first read: the finalized build, its
    /// tables spillable (see [`IdleBuild`]).
    idle: Mutex<Option<Arc<IdleBuild>>>,
    /// Runtime filter partials merged so far, and the memory they hold.
    runtime: Mutex<Partials>,
    /// The runtime filters and where they are published, abandoned if the
    /// build never publishes them.
    filters: Vec<FilterId>,
    publisher: Option<FilterPublisher>,
}

/// Runtime filter partials (one per filter; `None`: the filter is off).
#[derive(Debug, Default)]
struct Partials {
    builders: Vec<Option<KeyFilterBuilder>>,
    memory: Vec<Reservation>,
}

impl Drop for BuildGlobal {
    fn drop(&mut self) {
        // A build that failed or was cancelled before finalize: consumers
        // stop waiting for its filters (a published partial stays: this
        // producer's second partial is ignored).
        if let Some(p) = &self.publisher {
            for &id in &self.filters {
                p.publish(id, FilterSnapshot::Disabled(DisabledReason::ProducerFailed));
            }
        }
    }
}

#[derive(Debug, Default)]
struct Collect {
    parts: Vec<Part>,
    stored: u64,
    file_bytes: u64,
    /// Per partition: the distinct-key sketch of the rows sunk, row layout
    /// only (`None` once any driver had no memory for its sketches).
    sketches: Option<Vec<Hll>>,
    /// The sketches' memory.
    sketch_memory: Vec<Reservation>,
}

/// The finalized build.
#[derive(Debug)]
pub(crate) struct Built {
    /// Per level-0 partition: its table, or `None` if it spilled or is
    /// empty.
    pub(crate) tables: Vec<Option<JoinTable>>,
    /// Level-0 partitions with build rows on disk (bit per partition).
    pub(crate) spilled: u64,
    /// The spilled partitions' work, shared by the probe drivers' flush
    /// (and by a write of an idle build's partition still in progress).
    pub(crate) grace: Arc<Mutex<Grace>>,
    /// Notified whenever a driver's claim on `grace` ends, or a write of an
    /// idle build's partition does.
    pub(crate) grace_changed: Arc<Event>,
    _memory: Vec<Reservation>,
}

impl BuildGlobal {
    /// The finalized build, `None` before `finalize`. The first call after
    /// it ends the build's idle window: from then on its tables stay in
    /// memory, since probe drivers hold references to them (a write already
    /// in progress goes on, and level 0 of the grace phase waits for it).
    pub(crate) fn built(&self) -> Option<&Built> {
        if let Some(b) = self.built.get() {
            return Some(b);
        }
        // Under the lock, so a second reader waits for the first one's seal
        // instead of finding neither the idle build nor the built one.
        let mut idle = lock(&self.idle);
        if self.built.get().is_none()
            && let Some(built) = idle.take().and_then(|i| i.seal())
        {
            let _ = self.built.set(built);
        }
        drop(idle);
        self.built.get()
    }
}

/// A finalized build no probe driver has read yet. Its probe may start
/// long after it (a scan waiting for a runtime filter whose producer is a
/// long aggregate, study 30): until then the tables belong to no running
/// driver, so they are an [`OutputSpill`] of their own. A spill request
/// writes whole partitions, largest first, to the files the grace phase
/// reads (the arena batch is what a build-phase spill writes, so the
/// reload is the usual one) and marks them spilled; the probe defers their
/// rows as for any spilled partition. The hook is [idle](OutputSpill::set_idle)
/// while the build waits, so the arbitrator asks it before a running
/// consumer spills its own state.
struct IdleBuild {
    state: Mutex<IdleState>,
    hook: Arc<OutputSpill>,
    consumer: Arc<ConsumerHandle>,
    shared: Arc<Shared>,
    hashing: Hashing,
    scope: SpillScope,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    grace: Arc<Mutex<Grace>>,
    grace_changed: Arc<Event>,
}

impl std::fmt::Debug for IdleBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // No lock: a `Debug` may run where the state's lock is held.
        f.debug_struct("IdleBuild")
            .field(
                "spillable",
                &krishiv_memory::Spillable::spillable_bytes(&*self.hook),
            )
            .finish_non_exhaustive()
    }
}

struct IdleState {
    /// `None` once sealed.
    built: Option<Built>,
    /// Per level-0 partition: the memory of its table (rows and index), on
    /// the hook's consumer.
    memory: Vec<Vec<Reservation>>,
}

fn held(memory: &[Reservation]) -> usize {
    memory.iter().map(Reservation::size).sum()
}

impl IdleBuild {
    /// Ends the idle window: the build as the probe reads it.
    fn seal(&self) -> Option<Built> {
        let mut st = lock(&self.state);
        let mut built = st.built.take()?;
        for m in &mut st.memory {
            built._memory.append(m);
        }
        drop(st);
        self.hook.set_idle(false);
        self.hook.set_spillable_bytes(0);
        Some(built)
    }
}

/// A write of an idle build's partitions in progress: when it ends, however
/// it ends, the grace phase stops waiting for it; ended any other way than
/// by [`landed`](Self::landed), the partitions' rows are lost.
struct Writing {
    grace: Arc<Mutex<Grace>>,
    changed: Arc<Event>,
    parts: usize,
    landed: bool,
}

impl Writing {
    fn landed(mut self, files: Vec<(usize, SpillFile, u64)>) {
        let mut g = lock(&self.grace);
        for (p, file, rows) in files {
            if let Some(part) = g.build.get_mut(p) {
                part.files.push(file);
                part.rows = rows;
            }
        }
        self.landed = true;
    }
}

impl Drop for Writing {
    fn drop(&mut self) {
        let mut g = lock(&self.grace);
        g.writing = g.writing.saturating_sub(self.parts);
        if !self.landed && g.lost.is_none() {
            g.lost = Some(cancelled());
        }
        drop(g);
        self.changed.notify();
    }
}

impl SpillUnread for IdleBuild {
    fn spill_unread(self: Arc<Self>, target: usize) -> BoxFuture<'static, Result<usize>> {
        Box::pin(async move {
            // The writers' scratch the tables cannot fund comes from the
            // pool now, or the spill frees nothing.
            let mut topup = self.consumer.reservation().for_spill_scratch();
            let (jobs, freed) = {
                let mut st = lock(&self.state);
                let IdleState { built, memory } = &mut *st;
                let Some(built) = built.as_mut() else {
                    return Ok(0);
                };
                let mut order: Vec<usize> = (0..built.tables.len())
                    .filter(|&p| built.tables[p].is_some() && memory.get(p).is_some())
                    .collect();
                order.sort_by_key(|&p| std::cmp::Reverse(held(&memory[p])));
                // One partition covers the request: the smallest that does,
                // so a small request writes no more than it needs.
                let want = target.max(SPILL_MIN);
                if let Some(i) = order.iter().rposition(|&p| held(&memory[p]) >= want) {
                    order.swap(0, i);
                }
                let mut freed = 0usize;
                let chosen: Vec<usize> = order
                    .into_iter()
                    .take_while(|&p| {
                        let more = freed < target.max(SPILL_MIN);
                        if more {
                            freed = freed.saturating_add(held(&memory[p]));
                        }
                        more
                    })
                    .collect();
                if chosen.is_empty() || topup.try_grow(SPILL_SCRATCH.saturating_sub(freed)).is_err()
                {
                    return Ok(0);
                }
                topup.shrink_to_credit(topup.size());
                let mut jobs = Vec::with_capacity(chosen.len());
                for p in chosen {
                    let Some(table) = built.tables[p].take() else {
                        continue;
                    };
                    built.spilled |= 1 << p;
                    // The bytes become the writers' scratch, as our grant
                    // credit (all on the hook's consumer).
                    for mut m in std::mem::take(&mut memory[p]) {
                        let bytes = m.size();
                        m.shrink_to_credit(bytes);
                    }
                    let scratch = self
                        .consumer
                        .reservation()
                        .releasing_to_credit()
                        .for_spill_scratch();
                    jobs.push((p, table, scratch));
                }
                lock(&self.grace).writing += jobs.len();
                let left: usize = memory.iter().map(|m| held(m)).sum();
                self.hook.set_spillable_bytes(left);
                (jobs, freed)
            };
            let writing = Writing {
                grace: Arc::clone(&self.grace),
                changed: Arc::clone(&self.grace_changed),
                parts: jobs.len(),
                landed: false,
            };
            // Until the credit is back in the pool the rows count as about
            // to be released (another driver's round waits for them).
            let spilling = self.consumer.spilling();
            let scope = self.scope.clone();
            let schema = Arc::clone(self.shared.arena_schema(&self.hashing));
            let in_flight = self.query.blocking_started();
            let written = run_blocking(&*self.executor, move || {
                let mut files = Vec::with_capacity(jobs.len());
                for (p, table, scratch) in jobs {
                    let rows = table.arena().len() as u64;
                    let mut w = scope.writer(Arc::clone(&schema), scratch)?;
                    for block in table.arena().blocks() {
                        write_sliced(&mut w, block)?;
                    }
                    drop(table);
                    files.push((p, w.finish()?, rows));
                }
                // The guard travels with the result (see `run_blocking`).
                Ok((files, in_flight))
            })
            .await;
            // Credit the writers did not take returns to the pool.
            self.consumer.release_credit();
            drop(spilling);
            let (files, _in_flight) = match written {
                Ok(w) => w,
                Err(e) => {
                    lock(&self.grace).lost = Some(e.clone());
                    drop(writing);
                    return Err(e);
                }
            };
            let n = &self.shared.counters;
            n.add(Counter::IdleSpills, files.len() as u64);
            n.add(
                Counter::IdleSpilledBytes,
                files.iter().map(|(_, f, _)| f.bytes()).sum(),
            );
            writing.landed(files);
            Ok(freed)
        })
    }
}

struct BuildLocal {
    hashing: Hashing,
    parts: Vec<Part>,
    spilled: Arc<AtomicU64>,
    signal: Arc<SpillSignal>,
    stored: u64,
    file_bytes: u64,
    /// This driver's runtime filter partials and their reservation.
    runtime: Partials,
    /// Per partition: a sketch of this driver's stored hashes (row layout,
    /// when their memory was granted), and that memory.
    sketches: Option<(Vec<Hll>, Reservation)>,
}

/// Register bits of a partition's distinct-key sketch: 1 KiB, 3.3% standard
/// error, enough to presize an index.
const SKETCH_BITS: u32 = 10;

/// A well-mixed value for the `seq`-th stored row with a NULL `=` key
/// (SplitMix64's finalizer): it decides the row's partition at every
/// grace level, nothing else (the row is never looked up).
fn null_spread(seq: u64) -> u64 {
    let mut z = seq.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Builds each filter from its merged partial and publishes it, reserving
/// it in `held` (a filter that does not fit keeps only its min/max range);
/// a filter whose partial was dropped is disabled.
fn publish(
    publisher: &FilterPublisher,
    filters: &[super::JoinRuntimeFilter],
    partials: Partials,
    held: &mut Reservation,
) {
    for (f, b) in filters.iter().zip(partials.builders) {
        let id = f.filter.id();
        let Some(b) = b else {
            publisher.publish(id, FilterSnapshot::Disabled(DisabledReason::ProducerFailed));
            continue;
        };
        let keys = b.finish();
        let keys = if held.try_grow(keys.heap_bytes()).is_ok() {
            keys
        } else {
            keys.range_only()
        };
        publisher.publish(id, FilterSnapshot::Final(Arc::new(keys)));
    }
    // The partials' memory (hash vectors) is freed with them here.
    drop(partials.memory);
}

/// Empty partials of `shared`'s runtime filters.
fn partials(shared: &Shared) -> Vec<Option<KeyFilterBuilder>> {
    shared
        .runtime
        .iter()
        .map(|f| KeyFilterBuilder::new(f.filter.key_type().clone(), f.options))
        .collect()
}

impl Partials {
    fn heap_bytes(&self) -> usize {
        self.builders
            .iter()
            .flatten()
            .map(KeyFilterBuilder::heap_bytes)
            .sum()
    }

    /// Adds one batch's keys; reserves what the partials grew by, dropping
    /// the Bloom filters' hashes when that fails (an optional filter never
    /// waits for memory or spills).
    fn add(
        &mut self,
        shared: &Shared,
        keys: &[ArrayRef],
        valid: Option<&arrow_buffer::NullBuffer>,
    ) {
        self.add_keys(&shared.runtime, keys, valid);
    }

    fn add_keys(
        &mut self,
        filters: &[super::JoinRuntimeFilter],
        keys: &[ArrayRef],
        valid: Option<&arrow_buffer::NullBuffer>,
    ) {
        for (b, f) in self.builders.iter_mut().zip(filters) {
            let ok = match (b.as_mut(), keys.get(f.key)) {
                (Some(builder), Some(k)) => builder.add(k.as_ref(), valid).is_ok(),
                _ => false,
            };
            if !ok {
                *b = None;
            }
        }
        self.fit();
    }

    /// Reserves what the partials hold beyond their reservations; when that
    /// fails the Bloom hashes go first, the exact keys only if it still
    /// fails without them (an optional filter never waits or fails).
    fn fit(&mut self) {
        let held: usize = self.memory.iter().map(Reservation::size).sum();
        let want = self.heap_bytes();
        let Some(memory) = self.memory.first_mut() else {
            return;
        };
        if want > held && memory.try_grow(want - held).is_err() {
            // The Bloom hashes go first; the exact keys only if the
            // reservation still fails without them.
            for b in self.builders.iter_mut().flatten() {
                b.drop_hashes();
            }
            let want = self.heap_bytes();
            let Some(memory) = self.memory.first_mut() else {
                return;
            };
            if want <= held || memory.try_grow(want - held).is_ok() {
                if want < held {
                    // Only the first reservation shrinks (the others are
                    // merged drivers', released when the partials drop).
                    let extra = (held - want).min(memory.size());
                    memory.shrink(extra);
                }
                return;
            }
            for b in self.builders.iter_mut().flatten() {
                b.drop_bloom();
            }
            let want = self.heap_bytes();
            let Some(memory) = self.memory.first_mut() else {
                return;
            };
            if want < held {
                let extra = (held - want).min(memory.size());
                memory.shrink(extra);
            } else {
                // The few value-set bytes left are not worth failing over.
                let _ = memory.try_grow(want - held);
            }
        }
    }

    fn merge(&mut self, other: Self) {
        for (dst, src) in self.builders.iter_mut().zip(other.builders) {
            *dst = match (dst.take(), src) {
                (Some(mut a), Some(b)) => {
                    a.merge(b);
                    Some(a)
                }
                _ => None,
            };
        }
        self.memory.extend(other.memory);
        // `merge` grew this driver's hashes by the other's (review minor 2).
        self.fit();
    }
}

fn state(global: &GlobalState) -> Result<&BuildGlobal> {
    global
        .downcast_ref::<BuildGlobal>()
        .ok_or_else(|| internal("wrong global state for the hash join build".to_owned()))
}

fn local_mut(local: &mut LocalState) -> Result<&mut BuildLocal> {
    local
        .downcast_mut::<BuildLocal>()
        .ok_or_else(|| internal("wrong local state for the hash join build".to_owned()))
}

/// The row-arena chunk of `rows` (indices into the encoded batch).
/// A build batch's payload: arrow rows, or payload rows of the row layout.
pub(crate) enum Payload {
    Arrow(Rows),
    Rows(crate::rows::EncodedKeys),
}

/// `hashes`: each row's stored hash (its key's, or a spread value for a
/// row with a NULL `=` key that a right join keeps, see [`null_spread`]).
pub(crate) fn arena_chunk(
    shared: &Shared,
    hashing: &Hashing,
    keys: &Encoded,
    hashes: &[u64],
    payload: Option<&Payload>,
    weights: &Weights,
    rows: &[usize],
) -> Result<Batch> {
    let mut columns: Vec<ArrayRef> = vec![Arc::new(UInt64Array::from_iter_values(
        rows.iter().map(|&i| hashes[i]),
    ))];
    let layout = shared.keys.layout().filter(|_| hashing.rows());
    columns.extend(keys.arena_columns(layout.map(AsRef::as_ref), rows)?);
    match (payload, shared.row_payload(hashing.rows())) {
        (Some(Payload::Arrow(p)), _) => {
            columns.push(Arc::new(LargeBinaryArray::from_iter_values(
                rows.iter().map(|&i| p.row(i)),
            )));
        }
        (Some(Payload::Rows(p)), Some(layout)) => {
            columns.extend(super::keys::row_columns(layout, p, rows)?);
        }
        (Some(Payload::Rows(_)), None) => {
            return Err(internal("payload rows without a layout".to_owned()));
        }
        (None, _) => {}
    }
    if shared.join_type.keeps_null_build_keys() {
        // Rows with a NULL `=` key: stored, never indexed.
        columns.push(Arc::new(arrow_array::BooleanArray::from_iter(
            rows.iter().map(|&i| Some(!keys.is_valid(i))),
        )));
    }
    let weights = match weights {
        Weights::Const(w) => Weights::Const(*w),
        Weights::Column(c) => Weights::Column(Int64Array::from_iter_values(
            rows.iter().map(|&i| c.value(i)),
        )),
    };
    let data = record(shared.arena_schema(hashing), columns, rows.len())?;
    Batch::with_weights(data, weights).map_err(|e| internal(e.to_string()))
}

impl BuildSink {
    /// Builds and publishes the runtime filters from the merged partials;
    /// returns the reservation that holds them for the query (a filter that
    /// does not fit keeps only its min/max range).
    fn publish_filters(&self, g: &BuildGlobal, ctx: &TaskContext) -> Reservation {
        let mut held = ctx.consumer().reservation();
        let partials = std::mem::take(&mut *lock(&g.runtime));
        if let Some(publisher) = &self.0.publisher {
            publish(publisher, &self.0.runtime, partials, &mut held);
        }
        held
    }

    async fn spill_parts(
        &self,
        hashing: &Hashing,
        parts: &mut [Part],
        chosen: &[usize],
        ctx: &TaskContext,
        must: bool,
    ) -> Result<(usize, u64)> {
        match &self.0.spill {
            Some(scope) => {
                let schema = self.0.arena_schema(hashing);
                spill_parts(parts, chosen, schema, scope, ctx, must).await
            }
            None => Ok((0, 0)),
        }
    }

    /// Spills partitions of the combined state, as in `spill`, until
    /// `target` bytes are freed (declining an unfundable spill unless it
    /// `must`). Chunks leave the lock while they are
    /// written; the partition is marked spilled first, so nothing indexes it.
    async fn spill_collected(
        &self,
        g: &BuildGlobal,
        target: usize,
        ctx: &TaskContext,
        must: bool,
    ) -> Result<usize> {
        if self.0.spill.is_none() || g.built.get().is_some() {
            return Ok(0);
        }
        let mut taken = {
            let mut c = lock(&g.collect);
            let chosen = choose(&c.parts, g.spilled.load(Ordering::Acquire), target);
            for &p in &chosen {
                g.spilled.fetch_or(1 << p, Ordering::AcqRel);
            }
            let mut taken: Vec<Part> = (0..g.hashing.fanout0()).map(|_| Part::default()).collect();
            for &p in &chosen {
                if let (Some(t), Some(part)) = (taken.get_mut(p), c.parts.get_mut(p)) {
                    t.chunks = std::mem::take(&mut part.chunks);
                    t.memory = std::mem::take(&mut part.memory);
                }
            }
            (taken, chosen)
        };
        let (freed, written) = self
            .spill_parts(&g.hashing, &mut taken.0, &taken.1, ctx, must)
            .await?;
        let mut c = lock(&g.collect);
        c.file_bytes += written;
        for (p, t) in taken.0.into_iter().enumerate() {
            if let Some(part) = c.parts.get_mut(p) {
                // A declined spill gives its chunks back.
                part.chunks.extend(t.chunks);
                part.files.extend(t.files);
                part.memory.extend(t.memory);
            }
        }
        ctx.consumer()
            .signal()
            .set_spillable_bytes(mem_bytes(&c.parts));
        Ok(freed)
    }
}

/// What indexing a partition's in-memory rows reserves, as `(lasting,
/// transient)`: lasting is its [`JoinTable`] index (and match bitmap when
/// `marks`) plus the copied chunk tails that stay beside chunks the arena
/// also slices (held through the probe); transient is the rest of the
/// tails' copy, which replaces chunks dropped once the arena is built
/// ([`RowArena::reblock_bytes`]). `None` if no rows are in memory.
fn build_need(part: &Part, row_layout: bool, marks: bool) -> Option<(usize, usize)> {
    if part.chunks.is_empty() {
        return None;
    }
    // Finalize sizes row-layout indexes from estimates: reserve their growth.
    let rows = part.mem_rows();
    let mut index = JoinTable::index_bytes_estimated(rows, key_bytes(&part.chunks), row_layout);
    if marks {
        index = index.saturating_add(JoinTable::marks_bytes(rows));
    }
    let (copy, kept) = RowArena::reblock_bytes(&part.chunks, 0);
    Some((index.saturating_add(kept), copy.saturating_sub(kept)))
}

/// Returns what a built partition's chunks reserved beyond its arena: the
/// chunks were reserved at their allocated size (builder slack, and tails
/// charged at the chunk they were sliced from), the arena copied the tails
/// into exact blocks and the chunks are gone, so the reservation keeps
/// `bytes`, the arena's allocated bytes ([`RowArena::held_bytes`]), less
/// the `kept` tail bytes the index reservation still holds (so the two
/// together cover the arena; `kept` may be all of it, leaving the chunks'
/// reservation nothing). Never below zero, and only ever shrinks.
fn trim_to_arena(part: &mut Part, kept: usize, bytes: usize) {
    let mut excess = part.bytes().saturating_sub(bytes.saturating_sub(kept));
    for m in part.memory.iter_mut() {
        let cut = excess.min(m.size());
        m.shrink(cut);
        excess -= cut;
    }
}

/// A duration in whole nanoseconds, saturating.
pub(super) fn nanos(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// One partition's finalize work.
struct Job {
    part: usize,
    /// Bytes of its rows in memory.
    bytes: usize,
    chunks: Vec<Batch>,
    /// What indexing it reserves: lasting (index and kept tails) and the
    /// transient copy (see [`build_need`]).
    need: (usize, usize),
    /// Of the lasting part, the arena's copied tails that stay beside chunks
    /// it also slices: reserved through the probe
    /// ([`RowArena::reblock_bytes`]).
    kept: usize,
    /// Its estimated distinct keys (row layout, when sketched).
    keys: Option<usize>,
}

/// An estimate of a sketch's distinct count that is rarely low: the
/// estimate plus three standard errors (3 x 3.3%).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a non-negative estimate, saturating"
)]
fn upper_estimate(sketch: &Hll) -> usize {
    (sketch.estimate() * 1.1).ceil() as usize
}

/// One sketch per level-0 partition, reserved in `memory` (row layout
/// only; `None` without the memory: the indexes are then sized by rows).
fn sketches(
    hashing: &Hashing,
    memory: &crate::context::MemoryHandle,
) -> Option<(Vec<Hll>, Reservation)> {
    if !hashing.rows() {
        return None;
    }
    let sketches: Vec<Hll> = (0..hashing.fanout0())
        .map(|_| Hll::new(SKETCH_BITS))
        .collect();
    let mut held = memory.reservation();
    held.try_grow(sketches.iter().map(Hll::bytes).sum()).ok()?;
    Some((sketches, held))
}

fn mem_bytes(parts: &[Part]) -> usize {
    parts.iter().map(Part::bytes).sum()
}

#[cfg(all(test, loom))]
impl BuildGlobal {
    /// A finalized build whose only work is `grace` (loom models).
    pub(crate) fn with_grace(grace: Grace) -> Arc<Self> {
        let g = Self {
            hashing: Hashing::new(Some(0), false),
            spilled: Arc::new(AtomicU64::new(0)),
            rows: AtomicU64::new(0),
            null_key: AtomicBool::new(false),
            collect: Mutex::new(Collect::default()),
            built: OnceLock::new(),
            idle: Mutex::new(None),
            runtime: Mutex::new(Partials::default()),
            filters: Vec::new(),
            publisher: None,
        };
        let _ = g.built.set(Built {
            tables: Vec::new(),
            spilled: 0,
            grace: Arc::new(Mutex::new(grace)),
            grace_changed: Arc::new(Event::default()),
            _memory: Vec::new(),
        });
        Arc::new(g)
    }
}

impl Sink for BuildSink {
    fn name(&self) -> &'static str {
        "HashJoinBuild"
    }

    fn published_filters(&self) -> Vec<FilterId> {
        if self.0.publisher.is_none() {
            return Vec::new();
        }
        self.0
            .runtime
            .iter()
            .filter(|f| f.options.scan_waits)
            .map(|f| f.filter.id())
            .collect()
    }

    /// Partitions absorb in `combine`, which runs in lane order.
    fn lane_safe(&self) -> bool {
        true
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let hashing = self.0.hashing(ctx.query().config());
        let fanout = hashing.fanout0();
        let rows = hashing.rows();
        Ok(Arc::new(BuildGlobal {
            hashing,
            spilled: Arc::new(AtomicU64::new(0)),
            rows: AtomicU64::new(0),
            null_key: AtomicBool::new(false),
            collect: Mutex::new(Collect {
                parts: (0..fanout).map(|_| Part::default()).collect(),
                sketches: rows.then(|| (0..fanout).map(|_| Hll::new(SKETCH_BITS)).collect()),
                ..Collect::default()
            }),
            built: OnceLock::new(),
            idle: Mutex::new(None),
            runtime: Mutex::new(Partials {
                builders: partials(&self.0),
                memory: Vec::new(),
            }),
            filters: self.0.runtime.iter().map(|f| f.filter.id()).collect(),
            publisher: self.0.publisher.clone(),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        Ok(Box::new(BuildLocal {
            hashing: state(global)?.hashing.clone(),
            parts: (0..state(global)?.hashing.fanout0())
                .map(|_| Part::with_memory(memory.reservation()))
                .collect(),
            spilled: Arc::clone(&state(global)?.spilled),
            signal: Arc::clone(memory.signal()),
            stored: 0,
            file_bytes: 0,
            runtime: Partials {
                builders: partials(&self.0),
                memory: vec![memory.reservation()],
            },
            sketches: sketches(&state(global)?.hashing, memory),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<SinkStatus> {
        let (l, g, s) = (local_mut(local)?, state(global)?, &*self.0);
        let n = input.num_rows();
        if n == 0 {
            return Ok(SinkStatus::NeedMoreInput);
        }
        // Covers the joins that return build rows by their matches too: a
        // build row's net multiplicity is not tracked, so a negative weight
        // would come out as one row.
        if !input.is_unweighted() && s.join_type != JoinType::Inner {
            return Err(unsupported(
                "a weighted (change-stream) build side for a non-inner hash join",
            ));
        }
        let columns = input.data().columns();
        let key_columns: Vec<ArrayRef> = s
            .build_keys
            .iter()
            .map(|&k| Arc::clone(&columns[k]))
            .collect();
        let keys = s.keys.encode(&key_columns, &g.hashing)?;
        if s.join_type == JoinType::NullAwareAnti && any_invalid(keys.valid.as_ref()) {
            g.null_key.store(true, Ordering::Release);
        }
        let payload = match (s.row_payload(g.hashing.rows()), &s.payload) {
            (Some(layout), _) => Some(Payload::Rows(layout.encode(columns)?)),
            (None, Some((conv, _))) => Some(Payload::Arrow(
                conv.convert_columns(columns)
                    .map_err(|e| internal(e.to_string()))?,
            )),
            (None, None) => None,
        };
        let mut by_part = vec![Vec::new(); g.hashing.fanout0()];
        // A NULL `=` key matches nothing: the row is dropped, unless the
        // join returns unmatched build rows; then it is stored flagged in
        // its partition (it spills with it), and no index takes it (in the
        // row layout a NULL `=` key is not told apart from a value). Its
        // stored hash only places it: every NULL key hashes alike, so a
        // spread value keeps such rows from piling into one partition that
        // no repartitioning could split.
        let keep_null = s.join_type.keeps_null_build_keys();
        let hashes: std::borrow::Cow<'_, [u64]> = if keep_null && any_invalid(keys.valid.as_ref()) {
            let key_hashes = keys.hashes();
            std::borrow::Cow::Owned(
                (0..n)
                    .map(|i| {
                        if keys.is_valid(i) {
                            key_hashes[i]
                        } else {
                            null_spread(l.stored + i as u64)
                        }
                    })
                    .collect(),
            )
        } else {
            std::borrow::Cow::Borrowed(keys.hashes())
        };
        for i in (0..n).filter(|&i| keep_null || keys.is_valid(i)) {
            by_part[g.hashing.partition(hashes[i], 0)].push(i);
        }
        let mut chunks = Vec::new();
        let mut stored = 0;
        for (p, rows) in by_part.iter().enumerate().filter(|(_, r)| !r.is_empty()) {
            stored += rows.len() as u64;
            chunks.push((
                p,
                arena_chunk(
                    s,
                    &g.hashing,
                    &keys,
                    &hashes,
                    payload.as_ref(),
                    input.weights(),
                    rows,
                )?,
            ));
        }
        if let Err(bytes) = reserve_chunks(&mut l.parts, &chunks) {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        // Only once the batch is stored: a `NeedMemory` retry sinks the same
        // batch again, which must not count its keys twice.
        if !s.runtime.is_empty() {
            l.runtime.add(s, &key_columns, keys.valid.as_ref());
        }
        for (p, c) in chunks {
            l.parts[p].chunks.push(c);
            let merged = l.parts[p].block(s.arena_schema(&g.hashing))?;
            s.counters.max(Counter::MaxCoalesceCopyBytes, merged as u64);
        }
        if let Some((sketches, _)) = &mut l.sketches {
            for (sketch, rows) in sketches.iter_mut().zip(&by_part) {
                // Distinct keys of the index: NULL-key rows are not in it.
                for &i in rows.iter().filter(|&&i| keys.is_valid(i)) {
                    sketch.add(hashes[i]);
                }
            }
        }
        g.rows.fetch_add(n as u64, Ordering::Relaxed);
        l.stored += stored;
        if s.spill.is_some() {
            l.signal.set_spillable_bytes(mem_bytes(&l.parts));
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = state(global)?;
        let l = local
            .downcast::<BuildLocal>()
            .map_err(|_| internal("wrong local state for the hash join build".to_owned()))?;
        let mut c = lock(&g.collect);
        for (dst, src) in c.parts.iter_mut().zip(l.parts) {
            dst.absorb(src);
        }
        c.stored += l.stored;
        c.file_bytes += l.file_bytes;
        match (c.sketches.as_mut(), l.sketches) {
            (Some(all), Some((mine, memory))) => {
                for (a, m) in all.iter_mut().zip(&mine) {
                    a.merge(m);
                }
                c.sketch_memory.push(memory);
            }
            // A driver without sketches: no estimate for anyone.
            (Some(_), None) => c.sketches = None,
            (None, _) => {}
        }
        lock(&g.runtime).merge(l.runtime);
        // The finalizing driver's signal offers the combined state.
        l.signal.set_spillable_bytes(if self.0.spill.is_some() {
            mem_bytes(&c.parts)
        } else {
            0
        });
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let started = std::time::Instant::now();
            let g = state(global)?;
            let s = &self.0;
            let filter_memory = self.publish_filters(g, ctx);
            // Write what is left of spilled partitions; reserve the index of
            // the rest, spilling the largest while it does not fit. The arenas
            // read the blocks in place; finalize copies only the chunk tails.
            // A spilling build reserves every index and kept tail and the
            // largest transient copy, so one partition can always run and the
            // copies never need the build's size again.
            let mut index = ctx.consumer().reservation();
            let mut held_copy = 0;
            if s.spill.is_some() {
                held_copy = loop {
                    // Once memory ran short, every partition goes to disk: the
                    // probe phase then holds one partition's table per driver,
                    // never tables that fill the budget and cannot spill.
                    if g.spilled.load(Ordering::Acquire) != 0 {
                        g.spilled
                            .store(u64::MAX >> (64 - g.hashing.fanout0()), Ordering::Release);
                    }
                    let spilled = g.spilled.load(Ordering::Acquire);
                    let remnants = {
                        let c = lock(&g.collect);
                        (0..g.hashing.fanout0())
                            .any(|p| spilled & (1 << p) != 0 && !c.parts[p].chunks.is_empty())
                    };
                    if remnants {
                        // These must go to disk.
                        self.spill_collected(g, 0, ctx, true).await?;
                        continue;
                    }
                    let (indexes, copy) = {
                        let c = lock(&g.collect);
                        let rows = g.hashing.rows();
                        let marks = s.join_type.builds_preserved();
                        let needs = c.parts.iter().filter_map(|p| build_need(p, rows, marks));
                        needs.fold((0usize, 0usize), |(i, m), (index, copy)| {
                            (i.saturating_add(index), usize::max(m, copy))
                        })
                    };
                    if index.try_grow(indexes.saturating_add(copy)).is_ok() {
                        break copy;
                    }
                    self.spill_collected(g, 1, ctx, true).await?;
                };
            }
            let mut jobs: Vec<Job> = {
                let mut c = lock(&g.collect);
                let estimates: Vec<Option<usize>> = match &c.sketches {
                    Some(s) => s.iter().map(|h| Some(upper_estimate(h))).collect(),
                    None => vec![None; c.parts.len()],
                };
                c.parts
                    .iter_mut()
                    .zip(estimates)
                    .enumerate()
                    .filter_map(|(i, (p, keys))| {
                        let marks = s.join_type.builds_preserved();
                        Some((i, build_need(p, g.hashing.rows(), marks)?, p, keys))
                    })
                    .map(|(i, need, p, keys)| Job {
                        part: i,
                        bytes: p.bytes(),
                        kept: RowArena::reblock_bytes(&p.chunks, 0).1,
                        chunks: std::mem::take(&mut p.chunks),
                        need,
                        keys,
                    })
                    .collect()
            };
            // Largest first: the last partitions to start are the small ones,
            // so the pool's threads finish close together.
            jobs.sort_by_key(|j| std::cmp::Reverse(j.bytes));
            // Nothing in memory is spillable any more.
            ctx.consumer().signal().set_spillable_bytes(0);
            // One blocking task per partition (the executor's blocking pool
            // bounds how many run at once), each started only once its tails'
            // copy is reserved. A spilling build's reservation already funds
            // the largest transient copy (`spare`), so one partition can always run;
            // without a spill scope a partition's index and copy are asked
            // for through arbitration when nothing else runs.
            let spill = s.spill.is_some();
            let mut spare = held_copy;
            // The concatenation copies reserved at once (for the counter).
            let mut copying = 0usize;
            let mut jobs = std::collections::VecDeque::from(jobs);
            let mut running = futures::stream::FuturesUnordered::new();
            let mut tables = Vec::with_capacity(jobs.len());
            // Per built partition: the kept tails its index reservation
            // holds and its arena's allocated bytes, to true up its chunks'
            // reservation once they are gone.
            let mut arena_held = Vec::with_capacity(jobs.len());
            loop {
                if let Some(&Job {
                    need: (index_bytes, copy),
                    ..
                }) = jobs.front()
                {
                    // A spilling build reserved every index already.
                    let grow = if spill {
                        copy
                    } else {
                        index_bytes.saturating_add(copy)
                    };
                    let funded = if spill && copy <= spare {
                        spare -= copy;
                        true
                    } else if grow == 0 || index.try_grow(grow).is_ok() {
                        true
                    } else if running.is_empty() {
                        index.grow(grow).await.map_err(KrishivError::from)?;
                        true
                    } else {
                        false
                    };
                    if let Some(Job {
                        part: p,
                        chunks,
                        need: (index_bytes, copy),
                        keys,
                        kept,
                        ..
                    }) = jobs.pop_front_if(|_| funded)
                    {
                        copying = copying.saturating_add(copy);
                        self.0
                            .counters
                            .max(Counter::MaxFinalizeCopyBytes, copying as u64);
                        let shared = Arc::clone(&self.0);
                        let hashing = g.hashing.clone();
                        running.push(async move {
                            let table = ctx
                                .run_blocking(move || {
                                    let t0 = std::time::Instant::now();
                                    let schema = shared.arena_schema(&hashing);
                                    let arena = RowArena::over_chunks(schema, &chunks)?;
                                    drop(chunks);
                                    let t1 = std::time::Instant::now();
                                    let layout = shared.keys.layout().filter(|_| hashing.rows());
                                    let table = JoinTable::build_expecting(
                                        arena,
                                        layout,
                                        hashing.g_seed(),
                                        keys,
                                    )
                                    .and_then(|t| {
                                        if shared.join_type.builds_preserved() {
                                            t.with_marks()
                                        } else {
                                            Ok(t)
                                        }
                                    });
                                    let n = &shared.counters;
                                    n.add(Counter::ConcatNs, nanos(t1 - t0));
                                    n.add(Counter::IndexNs, nanos(t1.elapsed()));
                                    n.max(Counter::MaxPartitionNs, nanos(t0.elapsed()));
                                    table
                                })
                                .await?;
                            Ok::<_, KrishivError>((p, table, index_bytes, copy, kept))
                        });
                        continue;
                    }
                }
                let Some(done) = futures::StreamExt::next(&mut running).await else {
                    break;
                };
                let (p, table, lasting, transient, kept) = done?;
                copying = copying.saturating_sub(transient);
                // The index was reserved for one key per row: keep what it
                // holds. The copied tails beside chunks the arena also slices
                // (`kept`, part of `lasting`) stay reserved through the probe.
                index.shrink(
                    lasting
                        .saturating_sub(kept)
                        .saturating_sub(table.index_memory())
                        .min(index.size()),
                );
                // The transient copy replaced chunks that are gone now.
                if spill {
                    // Keep the largest transient copy funded; return the rest.
                    spare = spare.saturating_add(transient);
                    if spare > held_copy {
                        index.shrink((spare - held_copy).min(index.size()));
                        spare = held_copy;
                    }
                } else {
                    index.shrink(transient.min(index.size()));
                }
                let split = table.arena().memory_split();
                let n = &self.0.counters;
                n.add(
                    Counter::ArenaBytes,
                    (split.hash + split.key + split.other) as u64,
                );
                n.add(Counter::ArenaHashBytes, split.hash as u64);
                n.add(Counter::ArenaKeyBytes, split.key as u64);
                n.add(Counter::IndexBytes, table.index_memory() as u64);
                let (keys, buckets, kind) = table.index_size();
                self.0.counters.or(Counter::IndexKinds, kind as u64);
                self.0.counters.add(Counter::HashEntries, keys as u64);
                self.0.counters.add(Counter::HashBuckets, buckets as u64);
                // Of `kept`, what the index reservation still holds once the
                // index is paid: an index past its estimate took the rest.
                let covered = lasting.saturating_sub(table.index_memory()).min(kept);
                arena_held.push((p, covered, table.arena().held_bytes()));
                tables.push((p, table));
            }
            index.shrink(spare.min(index.size()));
            let mut c = lock(&g.collect);
            // Spilled partitions that hold rows: the probe defers only these
            // (an empty partition matches nothing, spilled or not).
            let marked = g.spilled.load(Ordering::Acquire);
            let spilled = (0..g.hashing.fanout0())
                .filter(|&p| marked & (1 << p) != 0 && c.parts[p].rows > 0)
                .fold(0u64, |m, p| m | (1 << p));
            let mut all: Vec<Option<JoinTable>> = (0..g.hashing.fanout0()).map(|_| None).collect();
            for (p, t) in tables {
                all[p] = Some(t);
            }
            // The sketches sized the indexes; their memory goes back.
            c.sketches = None;
            c.sketch_memory.clear();
            // With a spill scope, the tables in memory stay spillable until
            // a probe driver reads them (`IdleBuild`): their memory is kept
            // per partition.
            let idle = s.spill.clone().filter(|_| all.iter().any(Option::is_some));
            let mut tables_memory: Vec<Vec<Reservation>> =
                (0..g.hashing.fanout0()).map(|_| Vec::new()).collect();
            if idle.is_some() {
                for (p, t) in all.iter().enumerate() {
                    if let (Some(t), Some(m)) = (t, tables_memory.get_mut(p)) {
                        m.push(index.split(t.index_memory().min(index.size())));
                    }
                }
            }
            let mut memory = vec![index, filter_memory];
            let mut grace = Grace::new(g.hashing.fanout0());
            grace.keep_unprobed = s.join_type.keeps_null_build_keys();
            for (p, kept, bytes) in arena_held {
                if let Some(part) = c.parts.get_mut(p) {
                    trim_to_arena(part, kept, bytes);
                }
            }
            for (p, part) in c.parts.iter_mut().enumerate() {
                if spilled & (1 << p) == 0 {
                    self.0
                        .counters
                        .add(Counter::SteadyReservedBytes, part.bytes() as u64);
                }
                match tables_memory.get_mut(p) {
                    Some(m) if idle.is_some() && all[p].is_some() => m.append(&mut part.memory),
                    _ => memory.append(&mut part.memory),
                }
                if spilled & (1 << p) != 0 {
                    grace.build[p].files = std::mem::take(&mut part.files);
                    grace.build[p].rows = part.rows;
                }
            }
            let n = &self.0.counters;
            n.add(Counter::BuildRows, c.stored);
            n.add(Counter::SpilledPartitions, u64::from(spilled.count_ones()));
            n.add(Counter::BuildSpillBytes, c.file_bytes);
            n.add(Counter::FinalizeNs, nanos(started.elapsed()));
            let built = Built {
                tables: all,
                spilled,
                grace: Arc::new(Mutex::new(grace)),
                grace_changed: Arc::new(Event::default()),
                _memory: memory,
            };
            drop(c);
            if g.built.get().is_some() || lock(&g.idle).is_some() {
                return Err(internal("hash join build finalized twice".to_owned()));
            }
            let Some(scope) = idle else {
                return g
                    .built
                    .set(built)
                    .map_err(|_| internal("hash join build finalized twice".to_owned()));
            };
            let (hook, consumer) = OutputSpill::register(ctx, "HashJoinBuild tables");
            for r in tables_memory.iter_mut().flatten() {
                r.transfer_to(&consumer).map_err(KrishivError::from)?;
            }
            let bytes = tables_memory.iter().map(|m| held(m)).sum();
            let out = Arc::new(IdleBuild {
                grace: Arc::clone(&built.grace),
                grace_changed: Arc::clone(&built.grace_changed),
                state: Mutex::new(IdleState {
                    built: Some(built),
                    memory: tables_memory,
                }),
                hook: Arc::clone(&hook),
                consumer,
                shared: Arc::clone(&self.0),
                hashing: g.hashing.clone(),
                scope,
                query: Arc::clone(ctx.query()),
                executor: Arc::clone(ctx.executor()),
            });
            let weak: Weak<dyn SpillUnread> = Arc::downgrade(&out) as Weak<IdleBuild>;
            hook.attach(weak);
            hook.set_spillable_bytes(bytes);
            hook.set_idle(true);
            *lock(&g.idle) = Some(out);
            Ok(())
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = local_mut(local)?;
            if self.0.spill.is_none() {
                return Ok(0);
            }
            let chosen = choose(&l.parts, l.spilled.load(Ordering::Acquire), target);
            for &p in &chosen {
                l.spilled.fetch_or(1 << p, Ordering::AcqRel);
            }
            let (freed, written) = self
                .spill_parts(&l.hashing, &mut l.parts, &chosen, ctx, false)
                .await?;
            l.file_bytes += written;
            l.signal.set_spillable_bytes(mem_bytes(&l.parts));
            Ok(freed)
        })
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        self.0.counters.build()
    }

    fn spill_global<'a>(
        &'a self,
        global: &'a GlobalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            self.spill_collected(state(global)?, target, ctx, false)
                .await
        })
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::DataType;
    use krishiv_io::{ColumnPath, FilterId, FilterSnapshot, KeyFilterOptions, RuntimeFilter};
    use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};

    use super::{Partials, Writing, publish};
    use crate::join::grace::Grace;
    use crate::join::{FilterPublisher, JoinRuntimeFilter};
    use crate::sync::{Event, Mutex, lock};
    use crate::values::{LocalValues, Producer};

    /// A write of an idle build's partitions that ends any other way than
    /// with its files (cancelled with its query) leaves the grace phase
    /// failed, never short of rows; one that lands hands over its files.
    #[test]
    fn an_unfinished_write_of_an_idle_build_loses_its_partitions() {
        let grace = Arc::new(Mutex::new(Grace::new(2)));
        let changed = Arc::new(Event::default());
        let writing = |parts| Writing {
            grace: Arc::clone(&grace),
            changed: Arc::clone(&changed),
            parts,
            landed: false,
        };
        lock(&grace).writing = 2;
        writing(1).landed(Vec::new());
        assert_eq!(lock(&grace).writing, 1);
        assert!(lock(&grace).lost.is_none());
        drop(writing(1));
        let g = lock(&grace);
        assert_eq!(g.writing, 0);
        assert!(g.lost.is_some(), "the rows were lost");
    }

    fn reservation(limit: usize) -> Reservation {
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
            limit: Some(limit),
            quantum: 1,
            // The whole limit for the partials: the spill-scratch floor
            // would hide part of it from these exact-size cases.
            spill_scratch_reserve: 0,
            ..MemoryConfig::default()
        });
        let id = pool.register("partials", None);
        Reservation::new(pool, id)
    }

    /// Publishes into `f`'s cell, as its only producer.
    fn publisher(f: &JoinRuntimeFilter) -> FilterPublisher {
        let values = LocalValues::new();
        assert!(values.adopt_filter(Arc::clone(&f.filter)).is_ok());
        FilterPublisher {
            values: Arc::new(values),
            producer: Producer::default(),
        }
    }

    fn filter() -> JoinRuntimeFilter {
        JoinRuntimeFilter {
            key: 0,
            filter: Arc::new(RuntimeFilter::new(
                FilterId(1),
                ColumnPath::new("k"),
                DataType::Int64,
            )),
            options: KeyFilterOptions {
                max_values: 0,
                ..KeyFilterOptions::default()
            },
        }
    }

    /// Memory for the distinct keys but not for their Bloom hashes too: the
    /// hashes go and the exact set stays (`drop_hashes`, review M1), with
    /// the reservation equal to what the partial holds. Were the hashes and
    /// the keys dropped together (`drop_bloom`), only the range would be
    /// published.
    #[test]
    fn without_room_for_the_hashes_the_exact_set_stays() {
        let mut f = filter();
        f.options.max_values = 100;
        let shared = vec![f.clone()];
        // Sparse (not a dense range, which would publish as the range
        // whatever is kept), and more than the `IN` list holds (100), so
        // `drop_bloom` would drop them where `drop_hashes` keeps them.
        let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(
            (0..2000).map(|i: i64| i * 1_000_003),
        ));
        let sizes = |hashes: bool| {
            let mut b = krishiv_io::KeyFilterBuilder::new(DataType::Int64, f.options).unwrap();
            b.add(keys.as_ref(), None).unwrap();
            if !hashes {
                b.drop_hashes();
            }
            b.heap_bytes()
        };
        let (keys_only, with_hashes) = (sizes(false), sizes(true));
        assert!(with_hashes > keys_only, "no hashes to drop");
        let mut p = Partials {
            builders: vec![krishiv_io::KeyFilterBuilder::new(
                DataType::Int64,
                f.options,
            )],
            // Room for the keys and half the hashes: never both.
            memory: vec![reservation(keys_only + (with_hashes - keys_only) / 2)],
        };
        p.add_keys(&shared, &[Arc::clone(&keys)], None);
        assert_eq!(p.memory[0].size(), p.heap_bytes(), "reservation != heap");
        assert_eq!(p.heap_bytes(), keys_only, "the keys were dropped too");
        publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
        let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
            unreachable!("not published")
        };
        assert_eq!(
            k.form(),
            krishiv_io::FilterForm::HashSet(2000),
            "no exact set published"
        );
    }

    /// Without memory for the hashes the Bloom is dropped; without memory
    /// for the published filter only the range is kept; a dropped partial
    /// disables its filter.
    #[test]
    fn memory_pressure_drops_forms_never_fails() {
        let f = filter();
        let shared = vec![f.clone()];
        let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(0..2000));
        let mut p = Partials {
            builders: vec![krishiv_io::KeyFilterBuilder::new(
                DataType::Int64,
                f.options,
            )],
            memory: vec![reservation(1024)],
        };
        let spec_keys = [Arc::clone(&keys)];
        p.add_keys(&shared, &spec_keys, None);
        assert!(p.memory[0].size() <= 1024);
        let mut held = reservation(1 << 20);
        publish(&publisher(&f), &shared, p, &mut held);
        let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
            unreachable!("not published")
        };
        assert!(k.bloom_bytes().is_none() && k.has_range());

        // The hashes fit, the published Bloom does not.
        let f = filter();
        let shared = vec![f.clone()];
        let mut p = Partials {
            builders: vec![krishiv_io::KeyFilterBuilder::new(
                DataType::Int64,
                f.options,
            )],
            memory: vec![reservation(1 << 20)],
        };
        p.add_keys(&shared, &spec_keys, None);
        let mut tiny = reservation(16);
        publish(&publisher(&f), &shared, p, &mut tiny);
        let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
            unreachable!("not published")
        };
        assert!(k.bloom_bytes().is_none() && k.has_range());

        // A key of another type drops the partial; the filter is disabled.
        let f = filter();
        let shared = vec![f.clone()];
        let mut p = Partials {
            builders: vec![krishiv_io::KeyFilterBuilder::new(
                DataType::Int64,
                f.options,
            )],
            memory: Vec::new(),
        };
        let wrong: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1]));
        p.add_keys(&shared, &[wrong], None);
        publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
        assert!(matches!(
            f.filter.published().snapshot,
            FilterSnapshot::Disabled(_)
        ));
    }
}
