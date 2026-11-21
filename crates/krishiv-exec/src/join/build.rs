// SPDX-License-Identifier: Apache-2.0
//! The build side: a [`Sink`] that partitions build rows into row-arena
//! chunks, spills whole partitions on request, and at `finalize` indexes the
//! partitions still in memory.

use std::sync::{Arc, OnceLock, Weak};

use arrow_array::{ArrayRef, Int64Array, LargeBinaryArray, UInt64Array};
use arrow_row::Rows;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
mod spilled;

use self::spilled::SpilledBloom;
use krishiv_io::{DisabledReason, FilterId, FilterSnapshot, KeyFilterBuilder};
use krishiv_memory::{ConsumerHandle, Reservation, SpillFile, SpillScope, SpillSignal};
use krishiv_types::batch::{Batch, Weights};

use super::direct::Direct;
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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

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
    /// Probe drivers that may still read the level-0 tables
    /// ([`Self::probe_joined`], [`Self::probe_left`]).
    probes: AtomicUsize,
    /// A probe driver reached the probe's flush, so every probe driver
    /// exists: the flush starts behind a barrier all of them arrive at.
    probe_flushing: AtomicBool,
    /// Runtime filter partials merged so far, and the memory they hold.
    runtime: Mutex<Partials>,
    /// Shared with every partial of this build ([`Partials::bloom_lost`]).
    bloom_lost: Arc<[AtomicBool]>,
    /// Shared with every partial of this build ([`Partials::spilled`]).
    spilled_blooms: Arc<[SpilledBloom]>,
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
    /// Per filter, shared by every partial of one task's build: set once
    /// any of them dropped that filter's Bloom hashes (a refused
    /// reservation, a spill, or past `bloom_max_items`). The merged filter
    /// can then no longer be a Bloom (`KeyFilterBuilder::merge` of a partial
    /// without hashes has none), so the other drivers stop collecting them
    /// (review N4). Only a hint: each task's build has its own flags.
    bloom_lost: Arc<[AtomicBool]>,
    /// Per filter, shared by every partial of one task's build: the fixed
    /// Bloom of `bloom_spill_bytes` the partials drain their hashes into
    /// once one of them passes `bloom_max_items` or must shed them (a
    /// spilling build), so the Bloom form survives both (TPC-H q09 at
    /// SF1000: a 300M-key build that always spills lost it). See
    /// `build/spilled.rs`.
    spilled: Arc<[SpilledBloom]>,
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
    /// The level-0 tables, `None` once every probe driver is done with
    /// them ([`BuildGlobal::probe_left`]).
    level0: Mutex<Option<Arc<Level0>>>,
    /// Level-0 partitions with build rows on disk (bit per partition).
    pub(crate) spilled: u64,
    /// The spilled partitions' work, shared by the probe drivers' flush
    /// (and by a write of an idle build's partition still in progress).
    pub(crate) grace: Arc<Mutex<Grace>>,
    /// Notified whenever a driver's claim on `grace` ends, or a write of an
    /// idle build's partition does.
    pub(crate) grace_changed: Arc<Event>,
    /// The runtime filters' memory: a scan of another pipeline may still
    /// read a filter after the probe ends, so it stays until the build's
    /// state is dropped.
    _memory: Vec<Reservation>,
}

impl Built {
    /// The level-0 tables, for a probe driver to hold while it probes;
    /// `None` once they were released.
    pub(crate) fn level0(&self) -> Option<Arc<Level0>> {
        lock(&self.level0).clone()
    }
}

/// A finalized build's level-0 tables and the memory they hold (their rows
/// and indexes). Each probe driver holds a reference while it probes, and
/// the build holds one until every probe driver is done, so the memory goes
/// back when the probe ends rather than when its pipeline does: a
/// downstream sink's final phase (an aggregate's merge) gets it.
#[derive(Debug)]
pub(crate) struct Level0 {
    /// Per level-0 partition: its table, or `None` if it spilled or is
    /// empty.
    pub(crate) tables: Vec<Option<JoinTable>>,
    /// The direct index of the tables (perf lever J1) with its decoded
    /// payload (P3): made by the first probe driver that asks, shared by
    /// the others, dropped with the tables (`direct.rs`).
    pub(crate) direct: OnceLock<Option<Direct>>,
    /// Held only to go back with the tables.
    _memory: Vec<Reservation>,
}

impl BuildGlobal {
    /// A probe driver that may read the level-0 tables: its state was made.
    /// Only for a build its probe owns (a prebuilt build may serve more
    /// than one probe, so it is never released).
    pub(crate) fn probe_joined(&self) {
        self.probes.fetch_add(1, Ordering::Relaxed);
    }

    /// A probe driver started the probe's flush.
    pub(crate) fn probe_flushing(&self) {
        self.probe_flushing.store(true, Ordering::Release);
    }

    /// A probe driver that [joined](Self::probe_joined) is done with the
    /// tables: its flush ended, or its state was dropped. It dropped its
    /// own reference first. When the last one leaves after some driver
    /// started the flush, every probe driver has left, and the build's
    /// reference is taken out: the caller drops it, and with it the tables
    /// and their memory. A count that reaches zero before any flush (drivers
    /// whose output was not wanted) releases nothing: later drivers may
    /// still join, and the tables go when the build's state is dropped.
    pub(crate) fn probe_left(&self) -> Option<Arc<Level0>> {
        let last = self.probes.fetch_sub(1, Ordering::AcqRel) == 1;
        if !last || !self.probe_flushing.load(Ordering::Acquire) {
            return None;
        }
        lock(&self.built.get()?.level0).take()
    }
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
        let built = st.built.take()?;
        // The tables' memory goes back with the tables (`Level0`), not with
        // the build: a probe that ends releases it for the sink above.
        // No clone of the tables exists before the seal, so `get_mut` holds.
        let mut level0 = lock(&built.level0);
        let present = level0.is_some();
        let unique = level0.as_mut().and_then(Arc::get_mut);
        debug_assert!(
            unique.is_some() || !present,
            "hash join level-0 tables shared before the build sealed"
        );
        if let Some(l0) = unique {
            for m in &mut st.memory {
                l0._memory.append(m);
            }
        }
        drop(level0);
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
                // Idle: no probe driver holds the tables yet, so the
                // reference is unique.
                let mut l0 = lock(&built.level0);
                let Some(l0) = l0.as_mut().and_then(Arc::get_mut) else {
                    return Ok(0);
                };
                let mut order: Vec<usize> = (0..l0.tables.len())
                    .filter(|&p| l0.tables[p].is_some() && memory.get(p).is_some())
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
                    let Some(table) = l0.tables[p].take() else {
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
/// a filter whose partial was dropped is disabled. Returns the
/// reservations of the fixed Blooms published ([`Partials::spilled`]),
/// which hold them from here.
fn publish(
    publisher: &FilterPublisher,
    filters: &[super::JoinRuntimeFilter],
    partials: Partials,
    held: &mut Reservation,
) -> Vec<Reservation> {
    let mut fixed = Vec::new();
    for (i, (f, b)) in filters.iter().zip(partials.builders).enumerate() {
        let id = f.filter.id();
        let Some(b) = b else {
            if let Some(s) = partials.spilled.get(i) {
                s.close();
            }
            publisher.publish(id, FilterSnapshot::Disabled(DisabledReason::ProducerFailed));
            continue;
        };
        let (bloom, memory) = match partials.spilled.get(i).and_then(SpilledBloom::take) {
            Some((bloom, memory)) => (Some(bloom), Some(memory)),
            None => (None, None),
        };
        let had_fixed = bloom.is_some();
        let keys = b.finish_with(bloom);
        // A published fixed Bloom stays on its own reservation.
        let own = match memory {
            Some(m) if keys.bloom_bytes().is_some() => {
                let size = m.size();
                fixed.push(m);
                size
            }
            _ => 0,
        };
        let keys = if held.try_grow(keys.heap_bytes().saturating_sub(own)).is_ok() {
            keys
        } else {
            if own > 0 {
                fixed.pop();
            }
            keys.range_only()
        };
        if let Some(s) = partials.spilled.get(i).filter(|_| had_fixed) {
            s.note(if own > 0 && keys.bloom_bytes().is_some() {
                spilled::events::SHIPPED
            } else {
                spilled::events::NOT_SHIPPED
            });
        }
        publisher.publish(id, FilterSnapshot::Final(Arc::new(keys)));
    }
    // The partials' memory (hash vectors) is freed with them here.
    drop(partials.memory);
    fixed
}

/// `EXPLAIN ANALYZE`'s `fixed_bloom` text for the build's
/// [`Counter::FixedBloom`] bits.
pub(super) fn fixed_bloom_names(bits: u64) -> String {
    spilled::events::names(bits)
}

/// One "Bloom lost" flag per runtime filter ([`Partials::bloom_lost`]).
fn lost_flags(shared: &Shared) -> Arc<[AtomicBool]> {
    shared
        .runtime
        .iter()
        .map(|_| AtomicBool::new(false))
        .collect()
}

/// One unused fixed Bloom per runtime filter ([`Partials::spilled`]).
fn spilled_blooms(shared: &Shared) -> Arc<[SpilledBloom]> {
    shared
        .runtime
        .iter()
        .map(|_| SpilledBloom::default())
        .collect()
}

/// Empty partials of `shared`'s runtime filters.
fn partials(shared: &Shared) -> Vec<Option<KeyFilterBuilder>> {
    shared
        .runtime
        .iter()
        .map(|f| {
            let mut b = KeyFilterBuilder::new(f.filter.key_type().clone(), f.options)?;
            // The build keeps the fixed filter (`Partials::spilled`).
            b.keep_hashes_past_cap();
            Some(b)
        })
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

    /// Bytes of the Bloom hashes alone: what [`Self::shed_hashes`] frees, so
    /// part of what a build offers the arbitrator (review N3).
    fn hash_bytes(&self) -> usize {
        self.builders
            .iter()
            .flatten()
            .map(KeyFilterBuilder::hash_bytes)
            .sum()
    }

    /// Drops every filter's Bloom hashes and tells the build's other
    /// partials.
    fn drop_hashes(&mut self) {
        for i in 0..self.builders.len() {
            // A drained builder holds only the hashes since its last batch;
            // its Bloom lives in the fixed filter.
            if let Some(Some(b)) = self.builders.get_mut(i)
                && b.has_hashes()
                && !b.is_drained()
            {
                b.drop_hashes();
                self.mark_lost(i);
            }
        }
    }

    /// Drops the hashes of the filters another partial of this build lost
    /// the Bloom of, and frees their fixed filters; returns whether any
    /// hashes went.
    fn follow_lost(&mut self) -> bool {
        let mut dropped = false;
        for (i, (b, lost)) in self
            .builders
            .iter_mut()
            .zip(self.bloom_lost.iter())
            .enumerate()
        {
            if !lost.load(Ordering::Relaxed) {
                continue;
            }
            if let Some(s) = self.spilled.get(i) {
                s.close();
            }
            if let Some(b) = b.as_mut().filter(|b| b.has_hashes()) {
                b.drop_hashes();
                dropped = true;
            }
        }
        dropped
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
        // Another partial dropped a filter's hashes: its merged filter
        // cannot be a Bloom, so these would only hold memory until finalize.
        if self.follow_lost() {
            self.release_excess();
        }
        let mut lost = Vec::new();
        for (i, (b, f)) in self.builders.iter_mut().zip(filters).enumerate() {
            let had = b.as_ref().is_some_and(KeyFilterBuilder::has_hashes);
            let before = b.as_ref().map_or(0, KeyFilterBuilder::rows);
            let ok = match (b.as_mut(), keys.get(f.key)) {
                (Some(builder), Some(k)) => builder.add(k.as_ref(), valid).is_ok(),
                _ => false,
            };
            if !ok {
                *b = None;
                // The filter is disabled: its fixed Bloom is not needed.
                if let Some(s) = self.spilled.get(i) {
                    s.close();
                }
            }
            if let (Some(b), Some(s)) = (b.as_ref(), self.spilled.get(i)) {
                s.saw(b.rows().saturating_sub(before));
            }
            // Past `bloom_max_items` the builder dropped its own hashes.
            if had && !b.as_ref().is_some_and(KeyFilterBuilder::has_hashes) {
                lost.push(i);
            }
        }
        for &i in &lost {
            self.mark_lost(i);
        }
        self.reserve_fixed();
        if self.settle_all(false).0 || !lost.is_empty() {
            self.release_excess();
        }
        self.fit();
    }

    /// Releases what the reservations hold beyond the partials' bytes, after
    /// hashes were dropped (a merge or `bloom_max_items`); only then, so a
    /// few bytes' shrink of a range does not release and re-grow per batch.
    fn release_excess(&mut self) {
        let held: usize = self.memory.iter().map(Reservation::size).sum();
        self.release(held.saturating_sub(self.heap_bytes()));
    }

    /// Reserves what the partials hold beyond their reservations; when that
    /// fails the Bloom hashes go first, the exact keys only if it still
    /// fails without them (an optional filter never waits or fails).
    fn fit(&mut self) {
        let mut held: usize = self.memory.iter().map(Reservation::size).sum();
        let want = self.heap_bytes();
        let Some(memory) = self.memory.first_mut() else {
            return;
        };
        if want > held && memory.try_grow(want - held).is_err() {
            // Hashes with a fixed Bloom move there, then the other Bloom
            // hashes go; the exact keys only if the reservation still
            // fails without them.
            if self.settle_all(true).0 {
                // A new fixed filter may have taken part of the reservations.
                held = self.memory.iter().map(Reservation::size).sum();
                let want = self.heap_bytes();
                let Some(memory) = self.memory.first_mut() else {
                    return;
                };
                if want <= held || memory.try_grow(want - held).is_ok() {
                    self.release(held.saturating_sub(want));
                    return;
                }
            }
            self.drop_hashes();
            let want = self.heap_bytes();
            let Some(memory) = self.memory.first_mut() else {
                return;
            };
            if want <= held || memory.try_grow(want - held).is_ok() {
                self.release(held.saturating_sub(want));
                return;
            }
            for i in 0..self.builders.len() {
                if let Some(Some(b)) = self.builders.get_mut(i) {
                    let had = b.has_hashes();
                    b.drop_bloom();
                    // A drained builder's Bloom (the fixed filter) goes too.
                    if had {
                        self.mark_lost(i);
                    }
                }
            }
            let want = self.heap_bytes();
            if want < held {
                self.release(held - want);
            } else if let Some(memory) = self.memory.first_mut() {
                // The few value-set bytes left are not worth failing over.
                let _ = memory.try_grow(want - held);
            }
        }
    }

    /// Shrinks the reservations, in order, by up to `extra` bytes in all;
    /// returns the bytes released. Merged drivers' reservations shrink too:
    /// they hold the merged hashes as much as the first does.
    fn release(&mut self, mut extra: usize) -> usize {
        let want = extra;
        for m in &mut self.memory {
            let s = extra.min(m.size());
            m.shrink(s);
            extra -= s;
        }
        want - extra
    }

    /// Drops the Bloom filters' hashes and shrinks the reservations to what
    /// is left; returns the bytes freed, net of a fixed filter this created,
    /// and whether any hashes left (a net of 0 may still have moved them
    /// into a new filter, so the spillable bytes changed). A build asked to
    /// spill sheds this optional memory before it writes a table: at 20 M
    /// keys the hashes are up to 160 MB a driver.
    fn shed_hashes(&mut self) -> (usize, bool) {
        let had = self.hash_bytes();
        // Those with a fixed Bloom move there; a filter created now costs
        // what it took from the pool (the rest came out of these
        // reservations), so the build goes on to write tables when the net
        // is short of the request; one that filled gave its bytes back.
        let (_, moved) = self.settle_all(true);
        self.drop_hashes();
        let held: usize = self.memory.iter().map(Reservation::size).sum();
        let net = (self.release(held.saturating_sub(self.heap_bytes())) + moved.freed)
            .saturating_sub(moved.taken);
        (net, self.hash_bytes() < had)
    }

    fn merge(&mut self, other: Self) {
        let mut dropped = Vec::new();
        for (i, (dst, src)) in self.builders.iter_mut().zip(other.builders).enumerate() {
            let had = dst.as_ref().is_some_and(KeyFilterBuilder::has_hashes);
            *dst = match (dst.take(), src) {
                (Some(mut a), Some(b)) => {
                    a.merge(b);
                    Some(a)
                }
                _ => None,
            };
            // Past `bloom_max_items` the merge dropped the hashes: drivers
            // still sinking stop collecting them (review nit of e324818fe).
            if had && !dst.as_ref().is_some_and(KeyFilterBuilder::has_hashes) {
                dropped.push(i);
            }
        }
        for &i in &dropped {
            self.mark_lost(i);
        }
        self.memory.extend(other.memory);
        let mut lost = !dropped.is_empty();
        if self.settle_all(false).0 {
            lost = true;
        }
        if lost {
            // Both partials' reservations still hold the hashes the merge
            // dropped.
            self.release_excess();
        }
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
    /// What a build can free when asked to spill: the Bloom hashes of its
    /// filter partials, which it sheds first, and its tables in memory when
    /// it has a spill scope (review N3: a build with spilled or small
    /// tables but large hashes is asked too).
    fn spillable(&self, parts: &[Part], runtime: &Partials) -> usize {
        let tables = if self.0.spill.is_some() {
            mem_bytes(parts)
        } else {
            0
        };
        runtime.hash_bytes() + tables
    }

    /// Builds and publishes the runtime filters from the merged partials;
    /// returns the reservation that holds them for the query (a filter that
    /// does not fit keeps only its min/max range).
    fn publish_filters(&self, g: &BuildGlobal, ctx: &TaskContext) -> Vec<Reservation> {
        let mut held = ctx.consumer().reservation();
        let partials = std::mem::take(&mut *lock(&g.runtime));
        let mut memory = match &self.0.publisher {
            Some(publisher) => publish(publisher, &self.0.runtime, partials, &mut held),
            None => {
                // Nothing publishes: free any fixed filter now, not when
                // the build's state goes.
                for s in g.spilled_blooms.iter() {
                    s.close();
                }
                Vec::new()
            }
        };
        // What the fixed Blooms did, for `EXPLAIN ANALYZE` (`fixed_bloom`).
        for (s, f) in g.spilled_blooms.iter().zip(self.0.runtime.iter()) {
            if f.options.bloom_spill_bytes.is_some() {
                let (bits, keys) = s.outcome();
                self.0
                    .counters
                    .or(Counter::FixedBloom, bits | spilled::events::PLANNED);
                self.0.counters.max(Counter::FixedBloomKeys, keys);
            }
        }
        memory.push(held);
        memory
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
            .set_spillable_bytes(self.spillable(&c.parts, &lock(&g.runtime)));
        Ok(freed)
    }
}

/// What indexing a partition's in-memory rows reserves, as `(lasting,
/// transient, kept)`: lasting is its [`JoinTable`] index (and match bitmap when
/// `marks`) plus the copied chunk tails that stay beside chunks the arena
/// also slices (held through the probe); transient is the rest of the
/// tails' copy, which replaces chunks dropped once the arena is built
/// ([`RowArena::reblock_bytes`]); kept is the tails' part of lasting.
/// `None` if no rows are in memory.
fn build_need(part: &Part, row_layout: bool, marks: bool) -> Option<(usize, usize, usize)> {
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
    Some((index.saturating_add(kept), copy.saturating_sub(kept), kept))
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
            probes: AtomicUsize::new(0),
            probe_flushing: AtomicBool::new(false),
            runtime: Mutex::new(Partials::default()),
            bloom_lost: Arc::default(),
            spilled_blooms: Arc::default(),
            filters: Vec::new(),
            publisher: None,
        };
        let _ = g.built.set(Built {
            level0: Mutex::new(Some(Arc::new(Level0 {
                tables: Vec::new(),
                direct: OnceLock::new(),
                _memory: Vec::new(),
            }))),
            spilled: 0,
            grace: Arc::new(Mutex::new(grace)),
            grace_changed: Arc::new(Event::default()),
            _memory: Vec::new(),
        });
        Arc::new(g)
    }
}

impl Sink for BuildSink {
    // A build's tables do not depend on its input's batches (J1).
    fn coalesce_input(&self) -> bool {
        true
    }

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
        let bloom_lost = lost_flags(&self.0);
        let spilled_blooms = spilled_blooms(&self.0);
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
            probes: AtomicUsize::new(0),
            probe_flushing: AtomicBool::new(false),
            runtime: Mutex::new(Partials {
                builders: partials(&self.0),
                memory: Vec::new(),
                bloom_lost: Arc::clone(&bloom_lost),
                spilled: Arc::clone(&spilled_blooms),
            }),
            bloom_lost,
            spilled_blooms,
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
                bloom_lost: Arc::clone(&state(global)?.bloom_lost),
                spilled: Arc::clone(&state(global)?.spilled_blooms),
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
        l.signal
            .set_spillable_bytes(self.spillable(&l.parts, &l.runtime));
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
        let mut runtime = lock(&g.runtime);
        runtime.merge(l.runtime);
        // The finalizing driver's signal offers the combined state.
        l.signal
            .set_spillable_bytes(self.spillable(&c.parts, &runtime));
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
            // The partials' hashes went with the publish: offer only the
            // tables from here (review L3 of the N2-N4 pass).
            {
                let c = lock(&g.collect);
                ctx.consumer()
                    .signal()
                    .set_spillable_bytes(self.spillable(&c.parts, &lock(&g.runtime)));
            }
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
                        needs.fold((0usize, 0usize), |(i, m), (index, copy, _)| {
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
                    .map(|(i, (lasting, copy, kept), p, keys)| Job {
                        part: i,
                        bytes: p.bytes(),
                        kept,
                        chunks: std::mem::take(&mut p.chunks),
                        need: (lasting, copy),
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
            let mut memory = vec![index];
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
                level0: Mutex::new(Some(Arc::new(Level0 {
                    tables: all,
                    direct: OnceLock::new(),
                    _memory: memory,
                }))),
                spilled,
                grace: Arc::new(Mutex::new(grace)),
                grace_changed: Arc::new(Event::default()),
                _memory: filter_memory,
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
            // The optional Bloom hashes go before any table is written.
            let (shed, _) = l.runtime.shed_hashes();
            if shed >= target || self.0.spill.is_none() {
                l.signal
                    .set_spillable_bytes(self.spillable(&l.parts, &l.runtime));
                return Ok(shed);
            }
            let chosen = choose(&l.parts, l.spilled.load(Ordering::Acquire), target - shed);
            for &p in &chosen {
                l.spilled.fetch_or(1 << p, Ordering::AcqRel);
            }
            let (freed, written) = self
                .spill_parts(&l.hashing, &mut l.parts, &chosen, ctx, false)
                .await?;
            l.file_bytes += written;
            l.signal
                .set_spillable_bytes(self.spillable(&l.parts, &l.runtime));
            Ok(freed + shed)
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
            let g = state(global)?;
            // The merged partials' Bloom hashes go before any table is
            // written, as in `spill` (review N2).
            let (shed, moved) = lock(&g.runtime).shed_hashes();
            if shed >= target || self.0.spill.is_none() || g.built.get().is_some() {
                // Hashes moved into a new fixed filter free nothing net but
                // are no longer spillable: the signal must say so.
                if shed > 0 || moved {
                    let c = lock(&g.collect);
                    ctx.consumer()
                        .signal()
                        .set_spillable_bytes(self.spillable(&c.parts, &lock(&g.runtime)));
                }
                return Ok(shed);
            }
            Ok(shed + self.spill_collected(g, target - shed, ctx, false).await?)
        })
    }
}

#[cfg(all(test, not(loom)))]
mod tests;
