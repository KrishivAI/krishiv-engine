// SPDX-License-Identifier: Apache-2.0
//! The build side: a [`Sink`] that partitions build rows into row-arena
//! chunks, spills whole partitions on request, and at `finalize` indexes the
//! partitions still in memory.

use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Int64Array, LargeBinaryArray, UInt64Array};
use arrow_row::Rows;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{Reservation, SpillSignal};
use krishiv_types::batch::{Batch, Weights};

use super::grace::{Grace, Part, choose, reserve_chunks, spill_parts};
use super::keys::{Encoded, PARTITIONS, any_invalid, partition, query_seed};
use super::table::{JoinTable, RowArena, key_bytes};
use super::{Counter, JoinType, Shared, record, unsupported};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::hash::HashSeed;
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::sync::{Event, Mutex, lock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The build sink.
#[derive(Debug)]
pub(crate) struct BuildSink(pub(crate) Arc<Shared>);

/// The build's global state; after `finalize`, what the probe reads.
#[derive(Debug)]
pub(crate) struct BuildGlobal {
    /// The query's hash seed.
    pub(crate) seed: HashSeed,
    /// Level-0 partitions spilled by any driver (bit per partition).
    spilled: Arc<AtomicU64>,
    /// Build rows, NULL keys included (null-aware anti: is it empty?).
    pub(crate) rows: AtomicU64,
    /// A build key was NULL (null-aware anti).
    pub(crate) null_key: AtomicBool,
    collect: Mutex<Collect>,
    pub(crate) built: OnceLock<Built>,
}

#[derive(Debug, Default)]
struct Collect {
    parts: Vec<Part>,
    stored: u64,
    file_bytes: u64,
}

/// The finalized build.
#[derive(Debug)]
pub(crate) struct Built {
    /// Per level-0 partition: its table, or `None` if it spilled or is
    /// empty.
    pub(crate) tables: Vec<Option<JoinTable>>,
    /// Level-0 partitions with build rows on disk (bit per partition).
    pub(crate) spilled: u64,
    /// The spilled partitions' work, shared by the probe drivers' flush.
    pub(crate) grace: Mutex<Grace>,
    /// Notified whenever a driver's claim on `grace` ends.
    pub(crate) grace_changed: Event,
    _memory: Vec<Reservation>,
}

struct BuildLocal {
    parts: Vec<Part>,
    spilled: Arc<AtomicU64>,
    signal: Arc<SpillSignal>,
    stored: u64,
    file_bytes: u64,
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
pub(crate) fn arena_chunk(
    shared: &Shared,
    keys: &Encoded,
    payload: Option<&Rows>,
    weights: &Weights,
    rows: &[usize],
) -> Result<Batch> {
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from_iter_values(
            rows.iter().map(|&i| keys.hashes[i]),
        )),
        Arc::new(LargeBinaryArray::from_iter_values(
            rows.iter().map(|&i| keys.rows.row(i)),
        )),
    ];
    if let Some(p) = payload {
        columns.push(Arc::new(LargeBinaryArray::from_iter_values(
            rows.iter().map(|&i| p.row(i)),
        )));
    }
    let weights = match weights {
        Weights::Const(w) => Weights::Const(*w),
        Weights::Column(c) => Weights::Column(Int64Array::from_iter_values(
            rows.iter().map(|&i| c.value(i)),
        )),
    };
    let data = record(&shared.arena_schema, columns, rows.len())?;
    Batch::with_weights(data, weights).map_err(|e| internal(e.to_string()))
}

impl BuildSink {
    async fn spill_parts(
        &self,
        parts: &mut [Part],
        chosen: &[usize],
        ctx: &TaskContext,
        must: bool,
    ) -> Result<(usize, u64)> {
        match &self.0.spill {
            Some(scope) => {
                let schema = &self.0.arena_schema;
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
            let mut taken: Vec<Part> = (0..PARTITIONS).map(|_| Part::default()).collect();
            for &p in &chosen {
                if let (Some(t), Some(part)) = (taken.get_mut(p), c.parts.get_mut(p)) {
                    t.chunks = std::mem::take(&mut part.chunks);
                    t.memory = std::mem::take(&mut part.memory);
                }
            }
            (taken, chosen)
        };
        let (freed, written) = self.spill_parts(&mut taken.0, &taken.1, ctx, must).await?;
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

/// What indexing a partition's in-memory rows reserves: `(index, copy)`,
/// its [`JoinTable`] index and its transient concatenation copy (none for a
/// single chunk); `None` if no rows are in memory.
fn build_need(part: &Part) -> Option<(usize, usize)> {
    if part.chunks.is_empty() {
        return None;
    }
    let index = JoinTable::index_bytes(part.mem_rows(), key_bytes(&part.chunks));
    let copy = if part.chunks.len() > 1 {
        part.bytes()
    } else {
        0
    };
    Some((index, copy))
}

fn mem_bytes(parts: &[Part]) -> usize {
    parts.iter().map(Part::bytes).sum()
}

#[cfg(all(test, loom))]
impl BuildGlobal {
    /// A finalized build whose only work is `grace` (loom models).
    pub(crate) fn with_grace(grace: Grace) -> Arc<Self> {
        let g = Self {
            seed: HashSeed::new(Some(0)),
            spilled: Arc::new(AtomicU64::new(0)),
            rows: AtomicU64::new(0),
            null_key: AtomicBool::new(false),
            collect: Mutex::new(Collect::default()),
            built: OnceLock::new(),
        };
        let _ = g.built.set(Built {
            tables: Vec::new(),
            spilled: 0,
            grace: Mutex::new(grace),
            grace_changed: Event::default(),
            _memory: Vec::new(),
        });
        Arc::new(g)
    }
}

impl Sink for BuildSink {
    fn name(&self) -> &'static str {
        "HashJoinBuild"
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(BuildGlobal {
            seed: query_seed(ctx.query().config().seed),
            spilled: Arc::new(AtomicU64::new(0)),
            rows: AtomicU64::new(0),
            null_key: AtomicBool::new(false),
            collect: Mutex::new(Collect {
                parts: (0..PARTITIONS).map(|_| Part::default()).collect(),
                ..Collect::default()
            }),
            built: OnceLock::new(),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        Ok(Box::new(BuildLocal {
            parts: (0..PARTITIONS)
                .map(|_| Part::with_memory(memory.reservation()))
                .collect(),
            spilled: Arc::clone(&state(global)?.spilled),
            signal: Arc::clone(memory.signal()),
            stored: 0,
            file_bytes: 0,
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
        let keys = s.keys.encode(&key_columns, &g.seed)?;
        if s.join_type == JoinType::NullAwareAnti && any_invalid(keys.valid.as_ref()) {
            g.null_key.store(true, Ordering::Release);
        }
        let payload = match &s.payload {
            Some((conv, _)) => Some(
                conv.convert_columns(columns)
                    .map_err(|e| internal(e.to_string()))?,
            ),
            None => None,
        };
        let mut by_part = vec![Vec::new(); PARTITIONS];
        for i in (0..n).filter(|&i| keys.is_valid(i)) {
            by_part[partition(keys.hashes[i], 0)].push(i);
        }
        let mut chunks = Vec::new();
        let mut stored = 0;
        for (p, rows) in by_part.iter().enumerate().filter(|(_, r)| !r.is_empty()) {
            stored += rows.len() as u64;
            chunks.push((
                p,
                arena_chunk(s, &keys, payload.as_ref(), input.weights(), rows)?,
            ));
        }
        if let Err(bytes) = reserve_chunks(&mut l.parts, &chunks) {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        for (p, c) in chunks {
            l.parts[p].chunks.push(c);
            l.parts[p].coalesce(&s.arena_schema)?;
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
            let g = state(global)?;
            let s = &self.0;
            // Write what is left of spilled partitions; reserve the index of
            // the rest, spilling the largest while it does not fit. A
            // partition's table is built only while its concatenation copy
            // is reserved, so the copies never need the build's size again.
            let mut index = ctx.consumer().reservation();
            let mut held_copy = 0;
            if s.spill.is_some() {
                held_copy = loop {
                    // Once memory ran short, every partition goes to disk: the
                    // probe phase then holds one partition's table per driver,
                    // never tables that fill the budget and cannot spill.
                    if g.spilled.load(Ordering::Acquire) != 0 {
                        g.spilled
                            .store(u64::MAX >> (64 - PARTITIONS), Ordering::Release);
                    }
                    let spilled = g.spilled.load(Ordering::Acquire);
                    let remnants = {
                        let c = lock(&g.collect);
                        (0..PARTITIONS)
                            .any(|p| spilled & (1 << p) != 0 && !c.parts[p].chunks.is_empty())
                    };
                    if remnants {
                        // These must go to disk.
                        self.spill_collected(g, 0, ctx, true).await?;
                        continue;
                    }
                    let (indexes, copy) = {
                        let c = lock(&g.collect);
                        let needs = c.parts.iter().filter_map(build_need);
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
            let jobs: Vec<(usize, Vec<Batch>, (usize, usize))> = {
                let mut c = lock(&g.collect);
                c.parts
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(i, p)| Some((i, build_need(p)?, p)))
                    .map(|(i, need, p)| (i, std::mem::take(&mut p.chunks), need))
                    .collect()
            };
            // Nothing in memory is spillable any more.
            ctx.consumer().signal().set_spillable_bytes(0);
            // One blocking task per partition (the executor's blocking pool
            // bounds how many run at once), each started only once its
            // concatenation copy is reserved. A spilling build's reservation
            // already funds the largest copy (`spare`), so one partition can
            // always run; without a spill scope a partition's index and copy
            // are asked for through arbitration when nothing else runs.
            let spill = s.spill.is_some();
            let mut spare = held_copy;
            let mut jobs = std::collections::VecDeque::from(jobs);
            let mut running = futures::stream::FuturesUnordered::new();
            let mut tables = Vec::with_capacity(jobs.len());
            loop {
                if let Some(&(_, _, (index_bytes, copy))) = jobs.front() {
                    // A spilling build reserved every index already.
                    let grow = if spill {
                        copy
                    } else {
                        index_bytes.saturating_add(copy)
                    };
                    let funded = if spill && copy <= spare {
                        spare -= copy;
                        true
                    } else if index.try_grow(grow).is_ok() {
                        true
                    } else if running.is_empty() {
                        index.grow(grow).await.map_err(KrishivError::from)?;
                        true
                    } else {
                        false
                    };
                    if let Some((p, chunks, (_, copy))) = jobs.pop_front_if(|_| funded) {
                        let shared = Arc::clone(&self.0);
                        running.push(async move {
                            let table = ctx
                                .run_blocking(move || {
                                    let arena = RowArena::concat(&shared.arena_schema, &chunks)?;
                                    drop(chunks);
                                    JoinTable::build(arena)
                                })
                                .await?;
                            Ok::<_, KrishivError>((p, table, copy))
                        });
                        continue;
                    }
                }
                let Some(done) = futures::StreamExt::next(&mut running).await else {
                    break;
                };
                let (p, table, copy) = done?;
                tables.push((p, table));
                if spill {
                    // Keep the largest copy funded; return the rest.
                    spare = spare.saturating_add(copy);
                    if spare > held_copy {
                        index.shrink(spare - held_copy);
                        spare = held_copy;
                    }
                } else {
                    index.shrink(copy);
                }
            }
            index.shrink(spare);
            let mut c = lock(&g.collect);
            // Spilled partitions that hold rows: the probe defers only these
            // (an empty partition matches nothing, spilled or not).
            let marked = g.spilled.load(Ordering::Acquire);
            let spilled = (0..PARTITIONS)
                .filter(|&p| marked & (1 << p) != 0 && c.parts[p].rows > 0)
                .fold(0u64, |m, p| m | (1 << p));
            let mut all: Vec<Option<JoinTable>> = (0..PARTITIONS).map(|_| None).collect();
            for (p, t) in tables {
                all[p] = Some(t);
            }
            let mut memory = vec![index];
            let mut grace = Grace::new();
            for (p, part) in c.parts.iter_mut().enumerate() {
                memory.append(&mut part.memory);
                if spilled & (1 << p) != 0 {
                    grace.build[p].files = std::mem::take(&mut part.files);
                    grace.build[p].rows = part.rows;
                }
            }
            let n = &self.0.counters;
            n.add(Counter::BuildRows, c.stored);
            n.add(Counter::SpilledPartitions, u64::from(spilled.count_ones()));
            n.add(Counter::BuildSpillBytes, c.file_bytes);
            g.built
                .set(Built {
                    tables: all,
                    spilled,
                    grace: Mutex::new(grace),
                    grace_changed: Event::default(),
                    _memory: memory,
                })
                .map_err(|_| internal("hash join build finalized twice".to_owned()))
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
            let (freed, written) = self.spill_parts(&mut l.parts, &chosen, ctx, false).await?;
            l.file_bytes += written;
            l.signal.set_spillable_bytes(mem_bytes(&l.parts));
            Ok(freed)
        })
    }

    fn extra_metrics(&self) -> Vec<(&'static str, u64)> {
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
