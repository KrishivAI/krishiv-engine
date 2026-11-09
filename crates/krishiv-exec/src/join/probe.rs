// SPDX-License-Identifier: Apache-2.0
//! The probe side: an [`Operator`] joining each input batch against the
//! in-memory partitions (at most `krishiv.exec.batchSize` output rows per
//! call), deferring rows of spilled partitions, and flushing the spilled
//! partitions after the input ends ([`super::grace`]).

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, UInt32Array, new_null_array};
use futures::future::BoxFuture;
use krishiv_common::error::{ArithmeticOverflow, KrishivError, ScalarSubqueryTooManyRows};
use krishiv_memory::SpillSignal;
use krishiv_types::batch::{Batch, Weights};

use super::build::{BuildGlobal, Level0, nanos};
use super::grace::{self, Flush, Part, WRITE_SLICE, choose, reserve_chunks, spill_parts};
use super::keys::{Hashing, MAX_RADIX_BITS};
use super::table::{JoinTable, NONE, RowArena};
use super::{Counter, JoinType, Shared, record, take_batch};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::hash::ProbeCount;
use crate::operator::{FlushOutput, LocalState, Operator, OperatorOutput};
use crate::operators::compact_sparse_views;
use crate::operators::select::{RowSelect, Selection, batch_rows};
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

    /// The table of each partition a row can route to ([`Self::part`]), to
    /// read once per row, without [`Self::get`]'s match and bounds check.
    fn by_part(self) -> [Option<&'a JoinTable>; PARTS] {
        let mut out = [None; PARTS];
        match self {
            Self::Level0(..) => {
                for (part, slot) in out.iter_mut().enumerate() {
                    *slot = self.get(part);
                }
            }
            Self::Single(t) => out[0] = Some(t),
        }
        out
    }
}

/// The most partitions: `2^MAX_RADIX_BITS`.
const PARTS: usize = 1 << MAX_RADIX_BITS;

const UNMATCHED: u8 = 0;
const MATCHED: u8 = 1;
/// Deferred to a spilled partition, or a NULL key of a null-aware anti join.
const SKIP: u8 = 2;

/// Iterates the matches of one probe batch, emitting bounded batches; the
/// rows that stay unmatched (or matched) are emitted at the end. With a
/// selection (design P2-E3 U2) the cursor's rows are the selected rows of
/// `batch`, in order: row `i` of the cursor is batch row `sel.row(i)`, and
/// every gather composes its ids through it.
#[derive(Debug)]
pub(crate) struct Cursor {
    batch: Batch,
    sel: Option<Selection>,
    heads: Vec<u32>,
    parts: Vec<u8>,
    state: Vec<u8>,
    pos: usize,
    chain: u32,
    /// How many rows have a match: the pairs to expect, about, to size one
    /// output's buffers without paying for rows that match nothing.
    matches: usize,
    /// Build rows are in the row layout (payload rows).
    rows: bool,
}

impl Cursor {
    /// Encodes `batch` (its rows `sel`, all without one) and finds each
    /// row's first match. Also returns the rows of spilled partitions, per
    /// partition, as rows of `batch`.
    pub(crate) fn new(
        shared: &Shared,
        hashing: &Hashing,
        batch: Batch,
        sel: Option<Selection>,
        tables: &Tables<'_>,
        detailed: bool,
    ) -> Result<(Self, Vec<Vec<u32>>)> {
        let n = sel.as_ref().map_or(batch.num_rows(), Selection::len);
        let picked = sel.as_ref().map(Selection::array);
        // Only the key columns of the selected rows are gathered.
        let cols: Vec<ArrayRef> = shared
            .probe_keys
            .iter()
            .map(|&k| {
                let c = batch.data().column(k);
                match &picked {
                    Some(ids) => arrow_select::take::take(c.as_ref(), ids, None)
                        .map_err(|e| internal(e.to_string())),
                    None => Ok(Arc::clone(c)),
                }
            })
            .collect::<Result<_>>()?;
        let keys = shared.keys.encode(&cols, hashing)?;
        let hashes = keys.hashes();
        let mut state = vec![UNMATCHED; n];
        let mut parts = vec![0u8; n];
        let by_part = tables.by_part();
        let fanout = hashing.fanout0();
        // Hashes spread the rows evenly: a little over a share each.
        let share = n / fanout + n / (4 * fanout) + 1;
        let mut lookup: Vec<Vec<u32>> = by_part[..fanout]
            .iter()
            .map(|t| Vec::with_capacity(if t.is_some() { share } else { 0 }))
            .collect();
        let mut deferred = vec![Vec::new(); fanout];
        let rows = u32::try_from(n).map_err(|_| internal("probe batch too large".to_owned()))?;
        for (i, row) in (0..rows).enumerate() {
            if !keys.is_valid(i) {
                if shared.join_type == JoinType::NullAwareAnti {
                    state[i] = SKIP;
                }
                continue;
            }
            let p = tables.part(hashing, hashes[i]);
            parts[i] = u8::try_from(p).unwrap_or(0);
            if by_part[p % PARTS].is_some() {
                lookup[p].push(row);
            } else if tables.spilled(p) {
                state[i] = SKIP;
                deferred[p].push(sel.as_ref().map_or(Ok(row), |s| s.row(row))?);
            }
        }
        let started = Instant::now();
        let mut heads = vec![NONE; n];
        let mut found = Vec::new();
        let mut count = ProbeCount::default();
        for (p, rows) in lookup.iter().enumerate().filter(|(_, r)| !r.is_empty()) {
            if let Some(t) = tables.get(p) {
                found.clear();
                if detailed {
                    t.find_heads_counted(&keys, rows, &mut found, &mut count);
                } else {
                    t.find_heads(&keys, rows, &mut found);
                }
                for (&r, &h) in rows.iter().zip(&found) {
                    heads[r as usize] = h;
                }
            }
        }
        shared
            .counters
            .add(Counter::ProbeTableNs, nanos(started.elapsed()));
        if count.lookups > 0 {
            shared.counters.add(Counter::Lookups, count.lookups);
            shared.counters.add(Counter::KeyCompares, count.compares);
            shared.counters.max(Counter::MaxProbeLength, count.max);
        }
        let matches = heads.iter().filter(|&&h| h != NONE).count();
        let cursor = Self {
            batch,
            sel,
            matches,
            heads,
            parts,
            state,
            pos: 0,
            chain: NONE,
            rows: hashing.rows(),
        };
        Ok((cursor, deferred))
    }

    /// The next output batch (if any) and whether the input is done.
    pub(crate) fn next(
        &mut self,
        shared: &Shared,
        tables: &Tables<'_>,
        limit: usize,
        partition: u32,
    ) -> Result<(Option<Batch>, bool)> {
        let n = self.state.len();
        let quick = !shared.join_type.emits_build() && shared.filter.is_none();
        // Semi / anti candidates produce no rows, so they need no output
        // bound: evaluate the filter on larger slices.
        let limit = if shared.join_type.emits_build() {
            limit
        } else {
            limit.max(4096)
        };
        // A right semi or anti join only needs to know which build rows
        // match: a build row already marked needs no other pair.
        let marks_only = shared.join_type.builds_preserved() && !shared.join_type.emits_build();
        let by_part = tables.by_part();
        let room = limit.min(self.matches);
        let mut probe = Vec::with_capacity(room);
        let mut build = Vec::with_capacity(room);
        let started = Instant::now();
        while probe.len() < limit && self.pos < n {
            let i = self.pos;
            if self.chain == NONE {
                let head = self.heads[i];
                if head == NONE || quick {
                    if head != NONE {
                        self.state[i] = MATCHED;
                        if marks_only && let Some(t) = by_part[usize::from(self.parts[i]) % PARTS] {
                            t.mark_chain(head);
                        }
                    }
                    self.pos += 1;
                    continue;
                }
                self.chain = head;
            }
            let p = usize::from(self.parts[i]);
            let table = by_part[p % PARTS];
            if !(marks_only && table.is_some_and(|t| t.is_marked(self.chain))) {
                probe.push(u32::try_from(i).unwrap_or(u32::MAX));
                build.push((p, self.chain));
            }
            self.chain = table.map_or(NONE, |t| t.next()[self.chain as usize]);
            if self.chain == NONE {
                self.pos += 1;
            }
        }
        shared
            .counters
            .add(Counter::ProbeTableNs, nanos(started.elapsed()));
        if !probe.is_empty() {
            return Ok((
                self.pairs(shared, tables, &probe, &build, partition)?,
                false,
            ));
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
        partition: u32,
    ) -> Result<Option<Batch>> {
        let arena = |p: usize| tables.get(p).map(JoinTable::arena);
        let build_cols = shared.build_columns(self.rows, build, arena)?;
        let taken = take_batch(&self.batch, &batch_rows(self.sel.as_ref(), probe)?)?;
        let taken = self.compact_selected(taken)?;
        let m = probe.len();
        let mut columns: Vec<ArrayRef> = taken.data().columns().to_vec();
        columns.extend(build_cols);
        let mask = match &shared.filter {
            Some(f) => {
                let candidates = record(&shared.filter_schema, columns.clone(), m)?;
                let mask = f.evaluate_in(&candidates, partition)?;
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
                if shared.join_type.builds_preserved() {
                    let (p, r) = build[j];
                    if let Some(t) = tables.get(p) {
                        t.mark(r);
                    }
                }
                // A single-match join holds at most one match per probe
                // row: the pairs of one row arrive in order, so a row that
                // is already marked matched this row twice. The mark lives
                // in `state`, so it survives across output batches, the
                // in-memory and grace phases alike.
                if shared.single_match && self.state[i as usize] == MATCHED {
                    return Err(too_many_rows());
                }
                self.state[i as usize] = MATCHED;
            }
        }
        if !shared.join_type.emits_build() {
            return Ok(None);
        }
        // Of every table, not only those of these pairs: once per batch.
        let unweighted = taken.is_unweighted()
            && tables
                .by_part()
                .iter()
                .flatten()
                .all(|t| t.arena().is_unweighted());
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

    /// The cursor's rows `rows` of the batch (the batch itself when they
    /// are all of its rows).
    fn gather(&self, rows: &[u32]) -> Result<Batch> {
        if self.sel.is_none() && rows.len() == self.batch.num_rows() {
            return Ok(self.batch.clone());
        }
        let taken = take_batch(&self.batch, &batch_rows(self.sel.as_ref(), rows)?)?;
        self.compact_selected(taken)
    }

    /// `taken` (rows of the batch, its columns) with the string views of a
    /// selective fused filter compacted, as the `Filter` it replaces does:
    /// a `take` keeps every data buffer of the scan's batch, which each
    /// breaker above would charge to the memory pool.
    fn compact_selected(&self, taken: Batch) -> Result<Batch> {
        if self.sel.is_some() {
            compact_sparse_views(&self.batch, taken)
        } else {
            Ok(taken)
        }
    }

    /// The rows the join type emits once every match is known.
    fn tail(&self, shared: &Shared) -> Result<Option<Batch>> {
        let want = match shared.join_type {
            // No probe row waits for the end: inner joins emit pairs as
            // they match, and the joins that build their preserved side
            // return build rows, never unmatched probe rows.
            JoinType::Inner | JoinType::RightSemi | JoinType::RightAnti | JoinType::RightOuter => {
                return Ok(None);
            }
            JoinType::LeftSemi => MATCHED,
            JoinType::LeftOuter
            | JoinType::FullOuter
            | JoinType::LeftAnti
            | JoinType::NullAwareAnti => UNMATCHED,
            // Every probe row once, with whether it matched.
            JoinType::Existence => return self.tail_exists(shared),
        };
        let rows: Vec<u32> = (0..self.state.len())
            .filter(|&i| self.state[i] == want)
            .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
            .collect();
        if rows.is_empty() {
            return Ok(None);
        }
        let taken = self.gather(&rows)?;
        if !shared.join_type.pads_probe() {
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

    /// Every probe row once plus `exists`: whether some build row matched
    /// it. Never NULL: a NULL `=` key matches nothing and reads FALSE.
    /// Rows deferred to a spilled partition are skipped here and joined in
    /// the grace phase.
    fn tail_exists(&self, shared: &Shared) -> Result<Option<Batch>> {
        let mut rows = Vec::new();
        let mut marks: Vec<bool> = Vec::new();
        for (i, &state) in self.state.iter().enumerate() {
            if state == SKIP {
                continue;
            }
            rows.push(u32::try_from(i).map_err(|_| internal("probe batch too large".to_owned()))?);
            marks.push(state == MATCHED);
        }
        if rows.is_empty() {
            return Ok(None);
        }
        let taken = self.gather(&rows)?;
        let (data, weights) = taken.into_parts();
        let mut columns = data.columns().to_vec();
        columns.push(Arc::new(BooleanArray::from(marks)) as ArrayRef);
        let out = Batch::with_weights(record(&shared.schema, columns, data.num_rows())?, weights);
        out.map(Some).map_err(|e| internal(e.to_string()))
    }
}

/// Rows a right join scans per emitted batch at most: a right semi join
/// with few marks returns between scans instead of reading the whole table
/// in one step.
const EMIT_SCAN: usize = 64;

/// The next rows a right join returns from `table` once every probe row
/// is joined, from row `*from` on: a right semi join's marked rows, a right
/// anti, right outer or full outer join's unmatched ones (the outer joins pad
/// them with NULL probe columns). At most `limit` rows per batch; `true` once
/// the table is done.
pub(crate) fn emit_build(
    shared: &Shared,
    rows: bool,
    table: &JoinTable,
    from: &mut u32,
    limit: usize,
) -> Result<(Option<Batch>, bool)> {
    let marked = shared.join_type == JoinType::RightSemi;
    let mut ids = Vec::new();
    let scan = u32::try_from(limit.saturating_mul(EMIT_SCAN)).unwrap_or(u32::MAX);
    table.rows_marked(marked, from, from.saturating_add(scan), limit, &mut ids);
    let done = *from as usize >= table.arena().len();
    Ok((build_rows(shared, rows, table.arena(), &ids)?, done))
}

/// Rows `ids` of `arena` as the right join's output: the build columns,
/// after NULL probe columns for a right or full outer join.
fn build_rows(shared: &Shared, rows: bool, arena: &RowArena, ids: &[u32]) -> Result<Option<Batch>> {
    if ids.is_empty() {
        return Ok(None);
    }
    let picked: Vec<(usize, u32)> = ids.iter().map(|&r| (0, r)).collect();
    let mut columns = Vec::new();
    if matches!(shared.join_type, JoinType::RightOuter | JoinType::FullOuter) {
        columns.extend(
            shared
                .probe_schema
                .fields()
                .iter()
                .map(|f| new_null_array(f.data_type(), ids.len())),
        );
    }
    columns.extend(shared.build_columns(rows, &picked, |_| Some(arena))?);
    Ok(Some(Batch::new(record(
        &shared.schema,
        columns,
        ids.len(),
    )?)))
}

fn overflow() -> KrishivError {
    ArithmeticOverflow {
        message: "The product of two join row weights overflows BIGINT".to_owned(),
        alternative: String::new(),
        config: "\"krishiv.ivm\"".to_owned(),
    }
    .into()
}

/// A probe row matched more than once in a single-match left outer join: a
/// correlated scalar subquery returned more than one row.
fn too_many_rows() -> KrishivError {
    ScalarSubqueryTooManyRows {}.into()
}

struct ProbeLocal {
    build: Arc<BuildGlobal>,
    /// The build's level-0 tables, from this driver's first batch or flush
    /// until it is done with them.
    level0: Option<Arc<Level0>>,
    /// The probe owns its build, so the tables go back once every probe
    /// driver is done ([`BuildGlobal::probe_left`]); `false` for a prebuilt
    /// build, and once this driver left.
    releases: bool,
    /// This driver told the build that the probe's flush started.
    flushing: bool,
    /// Count probe lengths (`ExecConfig::detailed_metrics`).
    detailed: bool,
    cursor: Option<Cursor>,
    /// Rows of spilled partitions.
    parts: Vec<Part>,
    signal: Arc<SpillSignal>,
    flush: Flush,
    batch_size: usize,
    /// The driver's partition: the residual filter evaluates in it.
    partition: u32,
    /// The fused selection (design P2-E3 U2).
    select: Option<RowSelect>,
    /// The selection of the input batch a `NeedMemory` interrupted: the
    /// retry joins the same rows without evaluating (and counting) the row
    /// stage again.
    selected: Option<Option<Selection>>,
}

impl ProbeLocal {
    /// The build's level-0 tables, held from now on by this driver.
    fn level0(&mut self) -> Result<&Level0> {
        if self.level0.is_none() {
            let built = self
                .build
                .built()
                .ok_or_else(|| internal("hash join probed before its build".to_owned()))?;
            self.level0 = built.level0();
        }
        self.level0
            .as_deref()
            .ok_or_else(|| internal("hash join build released before its probe ended".to_owned()))
    }

    /// Done with the build's tables: drops this driver's reference, and
    /// the build's own when this was the last probe driver.
    fn leave(&mut self) {
        if !std::mem::take(&mut self.releases) {
            return;
        }
        self.level0 = None;
        // The last one drops the tables here (freeing large buffers is
        // quick: a few allocations a partition).
        drop(self.build.probe_left());
    }
}

impl Drop for ProbeLocal {
    fn drop(&mut self) {
        self.leave();
    }
}

fn local_mut(local: &mut LocalState) -> Result<&mut ProbeLocal> {
    local
        .downcast_mut::<ProbeLocal>()
        .ok_or_else(|| internal("wrong local state for the hash join probe".to_owned()))
}

impl ProbeOp {
    fn execute_inner(&self, received: &Batch, l: &mut ProbeLocal) -> Result<OperatorOutput> {
        let s = &*self.0;
        let g = &*l.build;
        let built = g
            .built()
            .ok_or_else(|| internal("hash join probed before its build".to_owned()))?;
        if s.join_type.builds_preserved() && !received.is_unweighted() {
            // Whether a build row "matched" depends on the net multiplicity
            // of the probe rows with its key, which marks do not track.
            return Err(super::unsupported(
                "a weighted (change-stream) probe side for a join that returns its build rows",
            ));
        }
        // A new input batch: its fused selection first (design P2-E3 U2),
        // over the batch received, kept across a `NeedMemory` retry; then
        // the columns the probe reads of it.
        let sel = if l.cursor.is_none() && received.num_rows() > 0 {
            match l.selected.take() {
                Some(sel) => sel,
                None => match &mut l.select {
                    Some(select) => select.select(received, l.partition)?,
                    None => None,
                },
            }
        } else {
            None
        };
        let projected;
        let input = match s.select.as_ref().and_then(|x| x.columns.as_deref()) {
            Some(columns) if l.cursor.is_none() => {
                let data = arrow_array::RecordBatch::try_new_with_options(
                    Arc::clone(&s.probe_schema),
                    columns
                        .iter()
                        .map(|&c| Arc::clone(received.data().column(c)))
                        .collect(),
                    &arrow_array::RecordBatchOptions::new()
                        .with_row_count(Some(received.num_rows())),
                )
                .map_err(|e| internal(e.to_string()))?;
                projected = Batch::with_weights(data, received.weights().clone())
                    .map_err(|e| internal(e.to_string()))?;
                &projected
            }
            _ => received,
        };
        if l.cursor.is_none() && sel.as_ref().is_some_and(Selection::is_empty) {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let rows = sel.as_ref().map_or(input.num_rows(), Selection::len);
        if s.join_type == JoinType::NullAwareAnti {
            use std::sync::atomic::Ordering;
            if g.null_key.load(Ordering::Acquire) {
                s.counters.add(Counter::ProbeRows, rows as u64);
                return Ok(OperatorOutput::NeedMoreInput(None));
            }
            if g.rows.load(Ordering::Acquire) == 0 {
                s.counters.add(Counter::ProbeRows, rows as u64);
                let kept = match &sel {
                    _ if rows == 0 => None,
                    Some(sel) => Some(compact_sparse_views(
                        input,
                        take_batch(input, &sel.array())?,
                    )?),
                    None => Some(input.clone()),
                };
                return Ok(OperatorOutput::NeedMoreInput(kept));
            }
        }
        if l.level0.is_none() {
            l.level0 = built.level0();
        }
        let level0 = l.level0.as_deref().ok_or_else(|| {
            internal("hash join build released before its probe ended".to_owned())
        })?;
        let tables = Tables::Level0(&level0.tables, built.spilled);
        if l.cursor.is_none() {
            if input.num_rows() == 0 {
                return Ok(OperatorOutput::NeedMoreInput(None));
            }
            let (cursor, deferred) =
                Cursor::new(s, &g.hashing, input.clone(), sel, &tables, l.detailed)?;
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
                l.selected = Some(cursor.sel);
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
            s.counters.add(Counter::ProbeRows, rows as u64);
            l.cursor = Some(cursor);
        }
        let limit = l.batch_size;
        let partition = l.partition;
        let cursor = l
            .cursor
            .as_mut()
            .ok_or_else(|| internal("no probe cursor".to_owned()))?;
        loop {
            let (out, done) = cursor.next(s, &tables, limit, partition)?;
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
    // A hash probe always needs a flush barrier (`needs_flush`), which already keeps its
    // pipeline at one driver; the grace join's state is not lane-ordered, so it says no.
    fn lane_safe(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "HashJoinProbe"
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::clone(&self.0.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        // The scan below built this driver's row stage before this
        // operator's state (the driver builds its source's first); a stage
        // it cannot find is a broken contract, not a probe without one.
        let select = match self.0.select.as_ref() {
            None => None,
            Some(spec) => {
                let stage = match spec.row_stage.as_ref() {
                    None => None,
                    Some(r) => Some(r.consumer_stage(ctx.partition()).ok_or_else(|| {
                        internal(
                            "a scan handed its row stage to a probe that cannot find it".to_owned(),
                        )
                    })?),
                };
                Some(RowSelect::new(
                    spec.predicate.clone(),
                    spec.narrow.clone(),
                    stage,
                ))
            }
        };
        let dep = ctx
            .dependency()
            .ok_or_else(|| internal("hash join probe without its build".to_owned()))?;
        let build = super::broadcast::build_of(dep)?;
        // A build of this join's own (not a prebuilt one other probes share).
        let releases = dep.downcast_ref::<BuildGlobal>().is_some();
        if releases {
            build.probe_joined();
        }
        let memory = ctx.consumer();
        let fanout = build.hashing.fanout0();
        Ok(Box::new(ProbeLocal {
            build,
            level0: None,
            releases,
            flushing: false,
            detailed: ctx.query().config().detailed_metrics,
            cursor: None,
            parts: (0..fanout)
                .map(|_| Part::with_memory(memory.reservation()))
                .collect(),
            signal: Arc::clone(memory.signal()),
            flush: Flush::default(),
            batch_size: ctx.query().config().batch_size.max(1),
            partition: ctx.partition(),
            select,
            selected: None,
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
            if let Some(freed) = l.flush.spill_merge(target, ctx).await? {
                return Ok(freed);
            }
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

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
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
                let all: Vec<usize> = (0..l.build.hashing.fanout0()).collect();
                let schema = &self.0.probe_schema;
                let (_, written) =
                    spill_parts(&mut l.parts, &all, schema, scope, ctx, false).await?;
                self.0.counters.add(Counter::ProbeSpillBytes, written);
            }
            l.signal.set_spillable_bytes(0);
            let built = l
                .build
                .built()
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
            if l.releases && !std::mem::replace(&mut l.flushing, true) {
                l.build.probe_flushing();
            }
            // The build rows a join that preserves them emits come from the
            // tables: a driver that probed no batch takes them now.
            l.level0()?;
            let tables = l.level0.as_deref().map_or(&[][..], |t| &t.tables[..]);
            let out = grace::step(&self.0, &l.build, tables, &mut l.flush, ctx).await?;
            if matches!(out, FlushOutput::Done) {
                l.leave();
            }
            Ok(out)
        })
    }
}
