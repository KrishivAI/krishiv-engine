// SPDX-License-Identifier: Apache-2.0
//! The probe side: an [`Operator`] joining each input batch against the
//! in-memory partitions (at most `krishiv.exec.batchSize` output rows per
//! call), deferring rows of spilled partitions, and flushing the spilled
//! partitions after the input ends ([`super::grace`]).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, UInt32Array, new_null_array};
use futures::future::BoxFuture;
use krishiv_common::error::{ArithmeticOverflow, KrishivError};
use krishiv_memory::SpillSignal;
use krishiv_types::batch::{Batch, Weights};

use super::build::BuildGlobal;
use super::grace::{self, Flush, Part, WRITE_SLICE, choose, reserve_chunks, spill_parts};
use super::keys::{Hashing, PARTITIONS};
use super::table::{JoinTable, NONE, RowArena};
use super::{Counter, JoinType, Shared, record, take_batch};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{FlushOutput, LocalState, Operator, OperatorOutput};
use crate::sync::lock;

/// The probe operator.
#[derive(Debug)]
pub(crate) struct ProbeOp(pub(crate) Arc<Shared>);

/// The tables a cursor probes.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Tables<'a> {
    /// The level-0 partitions (`None`: no table) and the mask of those
    /// spilled with rows (their probe rows are deferred).
    Level0(&'a [Option<JoinTable>], u64),
    /// One spilled partition, loaded.
    Single(&'a JoinTable),
}

impl<'a> Tables<'a> {
    fn part(self, hashing: &Hashing, hash: u64) -> usize {
        match self {
            Self::Level0(..) => hashing.partition(hash, 0),
            Self::Single(_) => 0,
        }
    }

    fn get(self, part: usize) -> Option<&'a JoinTable> {
        match self {
            Self::Level0(t, _) => t.get(part).and_then(Option::as_ref),
            Self::Single(t) => Some(t),
        }
    }

    fn spilled(self, part: usize) -> bool {
        matches!(self, Self::Level0(_, mask) if mask & (1 << part) != 0)
    }
}

const UNMATCHED: u8 = 0;
const MATCHED: u8 = 1;
/// Deferred to a spilled partition, or a NULL key of a null-aware anti join.
const SKIP: u8 = 2;

/// Iterates the matches of one probe batch, emitting bounded batches; the
/// rows that stay unmatched (or matched) are emitted at the end.
#[derive(Debug)]
pub(crate) struct Cursor {
    batch: Batch,
    heads: Vec<u32>,
    parts: Vec<u8>,
    state: Vec<u8>,
    pos: usize,
    chain: u32,
}

impl Cursor {
    /// Encodes `batch` and finds each row's first match. Also returns the
    /// rows of spilled partitions, per partition.
    pub(crate) fn new(
        shared: &Shared,
        hashing: &Hashing,
        batch: Batch,
        tables: &Tables<'_>,
    ) -> Result<(Self, Vec<Vec<u32>>)> {
        let n = batch.num_rows();
        let cols: Vec<ArrayRef> = shared
            .probe_keys
            .iter()
            .map(|&k| Arc::clone(batch.data().column(k)))
            .collect();
        let keys = shared.keys.encode(&cols, hashing)?;
        let hashes = keys.hashes();
        let mut state = vec![UNMATCHED; n];
        let mut parts = vec![0u8; n];
        let mut lookup = vec![Vec::new(); PARTITIONS];
        let mut deferred = vec![Vec::new(); PARTITIONS];
        for i in 0..n {
            let row = u32::try_from(i).map_err(|_| internal("probe batch too large".to_owned()))?;
            if !keys.is_valid(i) {
                if shared.join_type == JoinType::NullAwareAnti {
                    state[i] = SKIP;
                }
                continue;
            }
            let p = tables.part(hashing, hashes[i]);
            parts[i] = u8::try_from(p).unwrap_or(0);
            if tables.get(p).is_some() {
                lookup[p].push(row);
            } else if tables.spilled(p) {
                state[i] = SKIP;
                deferred[p].push(row);
            }
        }
        let mut heads = vec![NONE; n];
        let mut found = Vec::new();
        for (p, rows) in lookup.iter().enumerate().filter(|(_, r)| !r.is_empty()) {
            if let Some(t) = tables.get(p) {
                found.clear();
                t.find_heads(&keys, rows, &mut found);
                for (&r, &h) in rows.iter().zip(&found) {
                    heads[r as usize] = h;
                }
            }
        }
        let cursor = Self {
            batch,
            heads,
            parts,
            state,
            pos: 0,
            chain: NONE,
        };
        Ok((cursor, deferred))
    }

    /// The next output batch (if any) and whether the input is done.
    pub(crate) fn next(
        &mut self,
        shared: &Shared,
        tables: &Tables<'_>,
        limit: usize,
    ) -> Result<(Option<Batch>, bool)> {
        let n = self.batch.num_rows();
        let quick = !shared.join_type.emits_build() && shared.filter.is_none();
        // Semi / anti candidates produce no rows, so they need no output
        // bound: evaluate the filter on larger slices.
        let limit = if shared.join_type.emits_build() {
            limit
        } else {
            limit.max(4096)
        };
        let mut probe = Vec::new();
        let mut build = Vec::new();
        while probe.len() < limit && self.pos < n {
            let i = self.pos;
            if self.chain == NONE {
                let head = self.heads[i];
                if head == NONE || quick {
                    if head != NONE {
                        self.state[i] = MATCHED;
                    }
                    self.pos += 1;
                    continue;
                }
                self.chain = head;
            }
            let p = usize::from(self.parts[i]);
            probe.push(u32::try_from(i).unwrap_or(u32::MAX));
            build.push((p, self.chain));
            self.chain = tables
                .get(p)
                .map_or(NONE, |t| t.next()[self.chain as usize]);
            if self.chain == NONE {
                self.pos += 1;
            }
        }
        if !probe.is_empty() {
            return Ok((self.pairs(shared, tables, &probe, &build)?, false));
        }
        Ok((self.tail(shared)?, true))
    }

    /// Joins candidate pairs: applies the filter, marks matches and builds
    /// the inner / left outer output.
    fn pairs(
        &mut self,
        shared: &Shared,
        tables: &Tables<'_>,
        probe: &[u32],
        build: &[(usize, u32)],
    ) -> Result<Option<Batch>> {
        let arena = |p: usize| tables.get(p).map(JoinTable::arena);
        let payload = build
            .iter()
            .map(|&(p, r)| arena(p).map_or(&[][..], |a| a.payload(r)));
        let build_cols = shared.decode(payload)?;
        let taken = take_batch(&self.batch, &UInt32Array::from(probe.to_vec()))?;
        let m = probe.len();
        let mut columns: Vec<ArrayRef> = taken.data().columns().to_vec();
        columns.extend(build_cols);
        let mask = match &shared.filter {
            Some(f) => {
                let candidates = record(&shared.filter_schema, columns.clone(), m)?;
                let mask = f.evaluate(&candidates)?;
                if mask.len() != m {
                    return Err(internal(
                        "a join filter returned the wrong row count".to_owned(),
                    ));
                }
                // NULL is not a match.
                Some(if mask.null_count() > 0 {
                    arrow_select::filter::prep_null_mask_filter(&mask)
                } else {
                    mask
                })
            }
            None => None,
        };
        for (j, &i) in probe.iter().enumerate() {
            if mask.as_ref().is_none_or(|k| k.value(j)) {
                self.state[i as usize] = MATCHED;
            }
        }
        if !shared.join_type.emits_build() {
            return Ok(None);
        }
        let unweighted = taken.is_unweighted()
            && build
                .iter()
                .all(|&(p, _)| arena(p).is_none_or(RowArena::is_unweighted));
        let weights = if unweighted {
            Weights::Const(1)
        } else {
            let mut w = Vec::with_capacity(m);
            for (j, &(p, r)) in build.iter().enumerate() {
                let pw = match taken.weights() {
                    Weights::Const(c) => *c,
                    Weights::Column(c) => c.value(j),
                };
                let bw = arena(p).map_or(1, |a| a.weight(r));
                w.push(pw.checked_mul(bw).ok_or_else(overflow)?);
            }
            Weights::Column(Int64Array::from(w))
        };
        let out = Batch::with_weights(record(&shared.schema, columns, m)?, weights)
            .map_err(|e| internal(e.to_string()))?;
        let out = match &mask {
            Some(k) => out.filter(k).map_err(|e| internal(e.to_string()))?,
            None => out,
        };
        Ok((out.num_rows() > 0).then_some(out))
    }

    /// The rows the join type emits once every match is known.
    fn tail(&self, shared: &Shared) -> Result<Option<Batch>> {
        let want = match shared.join_type {
            JoinType::Inner => return Ok(None),
            JoinType::LeftSemi => MATCHED,
            JoinType::LeftOuter | JoinType::LeftAnti | JoinType::NullAwareAnti => UNMATCHED,
        };
        let rows: Vec<u32> = (0..self.state.len())
            .filter(|&i| self.state[i] == want)
            .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
            .collect();
        if rows.is_empty() {
            return Ok(None);
        }
        let taken = if rows.len() == self.batch.num_rows() {
            self.batch.clone()
        } else {
            take_batch(&self.batch, &UInt32Array::from(rows))?
        };
        if shared.join_type != JoinType::LeftOuter {
            return Ok(Some(taken));
        }
        let m = taken.num_rows();
        let (data, weights) = taken.into_parts();
        let mut columns = data.columns().to_vec();
        columns.extend(
            shared
                .build_fields
                .iter()
                .map(|f| new_null_array(f.data_type(), m)),
        );
        let out = Batch::with_weights(record(&shared.schema, columns, m)?, weights);
        out.map(Some).map_err(|e| internal(e.to_string()))
    }
}

fn overflow() -> KrishivError {
    ArithmeticOverflow {
        message: "The product of two join row weights overflows BIGINT".to_owned(),
        alternative: String::new(),
        config: "\"krishiv.ivm\"".to_owned(),
    }
    .into()
}

struct ProbeLocal {
    build: Arc<BuildGlobal>,
    cursor: Option<Cursor>,
    /// Rows of spilled partitions.
    parts: Vec<Part>,
    signal: Arc<SpillSignal>,
    flush: Flush,
    batch_size: usize,
}

fn local_mut(local: &mut LocalState) -> Result<&mut ProbeLocal> {
    local
        .downcast_mut::<ProbeLocal>()
        .ok_or_else(|| internal("wrong local state for the hash join probe".to_owned()))
}

impl ProbeOp {
    fn execute_inner(&self, input: &Batch, l: &mut ProbeLocal) -> Result<OperatorOutput> {
        let s = &*self.0;
        let g = &*l.build;
        let built = g
            .built
            .get()
            .ok_or_else(|| internal("hash join probed before its build".to_owned()))?;
        if s.join_type == JoinType::NullAwareAnti {
            use std::sync::atomic::Ordering;
            if g.null_key.load(Ordering::Acquire) {
                s.counters.add(Counter::ProbeRows, input.num_rows() as u64);
                return Ok(OperatorOutput::NeedMoreInput(None));
            }
            if g.rows.load(Ordering::Acquire) == 0 {
                s.counters.add(Counter::ProbeRows, input.num_rows() as u64);
                return Ok(OperatorOutput::NeedMoreInput(
                    (input.num_rows() > 0).then(|| input.clone()),
                ));
            }
        }
        let tables = Tables::Level0(&built.tables, built.spilled);
        if l.cursor.is_none() {
            if input.num_rows() == 0 {
                return Ok(OperatorOutput::NeedMoreInput(None));
            }
            let (cursor, deferred) = Cursor::new(s, &g.hashing, input.clone(), &tables)?;
            let mut chunks = Vec::new();
            for (p, rows) in deferred
                .into_iter()
                .enumerate()
                .filter(|(_, r)| !r.is_empty())
            {
                let chunk = take_batch(input, &UInt32Array::from(rows))?;
                chunks.push((p, super::compact_views(chunk)?));
            }
            if let Err(bytes) = reserve_chunks(&mut l.parts, &chunks) {
                return Ok(OperatorOutput::NeedMemory { bytes });
            }
            for (p, c) in chunks {
                s.counters
                    .add(Counter::ProbeSpilledRows, c.num_rows() as u64);
                l.parts[p].chunks.push(c);
                l.parts[p].coalesce(&s.probe_schema)?;
            }
            l.signal
                .set_spillable_bytes(l.parts.iter().map(Part::bytes).sum());
            s.counters.add(Counter::ProbeRows, input.num_rows() as u64);
            l.cursor = Some(cursor);
        }
        let limit = l.batch_size;
        let cursor = l
            .cursor
            .as_mut()
            .ok_or_else(|| internal("no probe cursor".to_owned()))?;
        loop {
            let (out, done) = cursor.next(s, &tables, limit)?;
            if done {
                l.cursor = None;
                return Ok(OperatorOutput::NeedMoreInput(out));
            }
            if let Some(b) = out {
                return Ok(OperatorOutput::HaveMoreOutput(b));
            }
        }
    }
}

impl Operator for ProbeOp {
    fn name(&self) -> &'static str {
        "HashJoinProbe"
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::clone(&self.0.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        let dep = ctx
            .dependency()
            .ok_or_else(|| internal("hash join probe without its build".to_owned()))?;
        let build = Arc::clone(dep)
            .downcast::<BuildGlobal>()
            .map_err(|_| internal("wrong build state for the hash join probe".to_owned()))?;
        let memory = ctx.consumer();
        Ok(Box::new(ProbeLocal {
            build,
            cursor: None,
            parts: (0..PARTITIONS)
                .map(|_| Part::with_memory(memory.reservation()))
                .collect(),
            signal: Arc::clone(memory.signal()),
            flush: Flush::default(),
            batch_size: ctx.query().config().batch_size.max(1),
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        self.execute_inner(input, local_mut(local)?)
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = local_mut(local)?;
            let Some(scope) = &self.0.spill else {
                return Ok(0);
            };
            let chosen = choose(&l.parts, 0, target);
            if chosen.is_empty() {
                // Nothing in memory (the grace phase): leave the consumer's
                // credit alone, it holds a partition's read-back headroom.
                return Ok(0);
            }
            let (freed, written) = spill_parts(
                &mut l.parts,
                &chosen,
                &self.0.probe_schema,
                scope,
                ctx,
                false,
            )
            .await?;
            self.0.counters.add(Counter::ProbeSpillBytes, written);
            l.signal
                .set_spillable_bytes(l.parts.iter().map(Part::bytes).sum());
            Ok(freed)
        })
    }

    fn extra_metrics(&self) -> Vec<(&'static str, u64)> {
        self.0.counters.probe()
    }

    fn needs_flush(&self) -> bool {
        true
    }

    fn end_input<'a>(
        &'a self,
        local: &'a mut LocalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let l = local_mut(local)?;
            // Deferred rows go to disk now, so the grace phase starts with
            // its memory free, unless they are too few to pay for a write
            // or for the writers' scratch (they then stay in memory,
            // reserved, until the grace phase joins them).
            let held: usize = l.parts.iter().map(Part::bytes).sum();
            if let Some(scope) = self.0.spill.as_ref().filter(|_| held >= WRITE_SLICE) {
                let all: Vec<usize> = (0..PARTITIONS).collect();
                let schema = &self.0.probe_schema;
                let (_, written) =
                    spill_parts(&mut l.parts, &all, schema, scope, ctx, false).await?;
                self.0.counters.add(Counter::ProbeSpillBytes, written);
            }
            l.signal.set_spillable_bytes(0);
            let built = l
                .build
                .built
                .get()
                .ok_or_else(|| internal("hash join probed before its build".to_owned()))?;
            let mut grace = lock(&built.grace);
            for (dst, src) in grace.probe.iter_mut().zip(std::mem::take(&mut l.parts)) {
                dst.absorb(src);
            }
            Ok(())
        })
    }

    fn flush<'a>(
        &'a self,
        local: &'a mut LocalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<FlushOutput>> {
        Box::pin(async move {
            let l = local_mut(local)?;
            grace::step(&self.0, &l.build.hashing, &l.build, &mut l.flush, ctx).await
        })
    }
}
