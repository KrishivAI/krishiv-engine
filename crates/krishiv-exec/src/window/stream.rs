// SPDX-License-Identifier: Apache-2.0
//! A partition too large for memory, evaluated from disk (see
//! [`spill`](super::spill)).
//!
//! The partition is first copied to a file of its own
//! ([`copy_partition`]); the copy folds the aggregates over the whole
//! partition and counts its rows. Then two readers stream the file: a
//! *lead* reader runs ahead by at most the peer group of the trailing
//! reader's last row and folds what needs a peer group's end (`cume_dist`,
//! aggregates over a `RANGE` frame up to the current row); the *trailing*
//! reader emits the rows, with the ranking functions and the aggregates
//! over a `ROWS` frame up to the current row folded as it goes. Rows fold
//! one by one in partition order, as in memory and as Spark's frames fold
//! them, so the results are the in-memory evaluation's. Memory: a frame per
//! reader, the lead's batch before the one it reads, and one output batch.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_array::builder::{BinaryViewBuilder, StringViewBuilder};
use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, RecordBatchOptions,
    UInt32Array,
};
use arrow_buffer::i256;
use arrow_row::OwnedRow;
use arrow_schema::DataType;
use krishiv_memory::{ConsumerHandle, SpillFile, SpillReader, SpillWriter};
use krishiv_types::batch::Batch;

use super::eval::{
    Slice, arguments, count_f64, decimal_array, decimal_value, decimals, exact_decimal,
    filter_mask, ntile, passes, picks_bytes, replaces, to_view, too_large, view_type,
};
use super::{Env, Shape, WindowFunction};
use crate::aggregate::{Accumulator, AggInput, AggregateExpr, AggregateFunction, Running};
use crate::context::QueryContext;
use crate::error::{Result, internal};
use crate::operators::KeyEncoder;

/// An aggregate's state over a frame that grows row by row.
pub(super) enum Fold {
    /// Through the aggregate's [`Accumulator`]: group 0 holds the state,
    /// group 1 takes rows the filter drops, groups from 2 on take
    /// snapshots.
    Rows(Box<Accumulator>),
    /// A decimal `sum` or `avg`, exact.
    Decimal {
        a: Box<AggregateExpr>,
        sum: i256,
        count: i64,
        ansi: bool,
    },
    /// `min`/`max`/`first`/`last` of strings or binaries: the value picked
    /// (`None` before any row, `Some(false)` for a picked NULL); `buf`
    /// holds the picked bytes when valid, its capacity reused for the next
    /// pick.
    Bytes {
        function: AggregateFunction,
        view: DataType,
        picked: Option<bool>,
        buf: Vec<u8>,
    },
}

impl Fold {
    pub(super) fn new(a: &AggregateExpr, ansi: bool) -> Result<Self> {
        if exact_decimal(a) {
            return Ok(Self::Decimal {
                a: Box::new(a.clone()),
                sum: i256::ZERO,
                count: 0,
                ansi,
            });
        }
        if picks_bytes(a) {
            return Ok(Self::Bytes {
                function: a.function,
                view: view_type(&a.arg_type),
                picked: None,
                buf: Vec::new(),
            });
        }
        // The argument columns are passed as they are, in order.
        let local = AggregateExpr {
            args: (0..a.args.len()).collect(),
            input: AggInput::Raw,
            ..a.clone()
        };
        let mut acc = Accumulator::new(&local, ansi)?;
        acc.resize(2);
        Ok(Self::Rows(Box::new(acc)))
    }

    /// Folds `rows` rows (`args`, the rows `mask` drops folding nothing)
    /// in order; after each row `i` with `snap[i]`, keeps the value. Returns
    /// the kept values, in order.
    pub(super) fn fold(
        &mut self,
        args: &[ArrayRef],
        mask: Option<&BooleanArray>,
        rows: usize,
        snap: &[bool],
    ) -> Result<ArrayRef> {
        if snap.len() != rows {
            return Err(internal(
                "window fold snapshots of the wrong length".to_owned(),
            ));
        }
        match self {
            Self::Rows(acc) => {
                let groups: Vec<u32> = (0..rows).map(|i| u32::from(!passes(mask, i))).collect();
                let parts = vec![0_u32; rows];
                let mut next = 2_u32;
                let mut slots = Vec::with_capacity(rows);
                for &s in snap {
                    slots.push(if s {
                        next = next.checked_add(1).ok_or_else(too_large)?;
                        next - 1
                    } else {
                        Running::NONE
                    });
                }
                acc.resize(next as usize);
                if rows > 0 {
                    acc.update_running(
                        &groups,
                        args,
                        &Running {
                            parts: &parts,
                            snapshots: &slots,
                        },
                    )?;
                }
                let ids: Vec<u32> = (2..next).collect();
                acc.evaluate(&ids)
            }
            Self::Decimal {
                a,
                sum,
                count,
                ansi,
            } => {
                let col = args.first().ok_or_else(|| {
                    internal("decimal window aggregate without argument".to_owned())
                })?;
                let values = decimals(col)?;
                let mut out = Vec::new();
                for (i, v) in values.iter().enumerate().take(rows) {
                    if let Some(v) = v
                        && passes(mask, i)
                    {
                        *sum = sum.wrapping_add(i256::from_i128(*v));
                        *count += 1;
                    }
                    if snap[i] {
                        out.push(decimal_value(a, *sum, *count, *ansi)?);
                    }
                }
                decimal_array(a, out)
            }
            Self::Bytes {
                function,
                view,
                picked,
                buf,
            } => {
                let col =
                    to_view(args.first().ok_or_else(|| {
                        internal("a window aggregate without argument".to_owned())
                    })?)?;
                let value = |i: usize| -> &[u8] {
                    match col.data_type() {
                        DataType::Utf8View => col.as_string_view().value(i).as_bytes(),
                        _ => col.as_binary_view().value(i),
                    }
                };
                let mut out = BytesOut::new(view);
                for (i, &s) in snap.iter().enumerate() {
                    if passes(mask, i) {
                        let valid = col.is_valid(i);
                        let old = (*picked).map(|v| v.then_some(buf.as_slice()));
                        if replaces(*function, old, valid, || value(i))? {
                            if valid {
                                buf.clear();
                                buf.extend_from_slice(value(i));
                            } else {
                                buf.clear();
                            }
                            *picked = Some(valid);
                        }
                    }
                    if s {
                        out.push((*picked).and_then(|v| v.then_some(buf.as_slice())))?;
                    }
                }
                Ok(out.finish())
            }
        }
    }

    /// The value over every row folded so far.
    pub(super) fn value(&mut self) -> Result<ArrayRef> {
        match self {
            Self::Rows(acc) => acc.evaluate(&[0]),
            Self::Decimal {
                a,
                sum,
                count,
                ansi,
            } => decimal_array(a, vec![decimal_value(a, *sum, *count, *ansi)?]),
            Self::Bytes {
                view, picked, buf, ..
            } => {
                let mut out = BytesOut::new(view);
                out.push((*picked).and_then(|v| v.then_some(buf.as_slice())))?;
                Ok(out.finish())
            }
        }
    }
}

/// String or binary view values, copied.
enum BytesOut {
    Text(StringViewBuilder),
    Binary(BinaryViewBuilder),
}

impl BytesOut {
    fn new(view: &DataType) -> Self {
        match view {
            DataType::Utf8View => Self::Text(StringViewBuilder::new()),
            _ => Self::Binary(BinaryViewBuilder::new()),
        }
    }

    fn push(&mut self, v: Option<&[u8]>) -> Result<()> {
        match (self, v) {
            (Self::Text(b), Some(v)) => b.append_value(
                std::str::from_utf8(v)
                    .map_err(|e| internal(format!("window string value: {e}")))?,
            ),
            (Self::Text(b), None) => b.append_null(),
            (Self::Binary(b), v) => b.append_option(v),
        }
        Ok(())
    }

    fn finish(self) -> ArrayRef {
        match self {
            Self::Text(mut b) => Arc::new(b.finish()),
            Self::Binary(mut b) => Arc::new(b.finish()),
        }
    }
}

/// All of `b`'s rows as an evaluation slice.
fn whole_batch(b: &RecordBatch) -> Slice<'_> {
    Slice {
        batch: b,
        offset: 0,
        len: b.num_rows(),
        part_start: &[],
        peer_start: &[],
    }
}

/// Folds `b`'s rows into `fold` (the aggregate of function `j`), keeping
/// the value after each row with `snap`.
fn fold_batch(
    env: &Env,
    j: usize,
    fold: &mut Fold,
    b: &RecordBatch,
    snap: &[bool],
) -> Result<ArrayRef> {
    let f = &env.plan.functions[j];
    let WindowFunction::Aggregate(a) = &f.function else {
        return Err(internal(
            "a window fold of a function that is no aggregate".to_owned(),
        ));
    };
    let s = whole_batch(b);
    let mask = filter_mask(f, &s)?;
    fold.fold(&arguments(a, &s)?, mask.as_ref(), b.num_rows(), snap)
}

/// Per row of `b`: it starts a peer group (`first`: its first row starts
/// one). `prev` is the ordering key of the row before, updated to `b`'s
/// last.
fn peer_flags(
    order: Option<&KeyEncoder>,
    b: &RecordBatch,
    prev: &mut Option<OwnedRow>,
    first: bool,
) -> Result<Vec<bool>> {
    let n = b.num_rows();
    let mut out = match order {
        None => vec![false; n],
        Some(enc) => {
            let rows = enc.encode(b)?;
            let v = (0..n)
                .map(|i| match i {
                    0 => prev.as_ref().is_none_or(|p| p.row() != rows.row(0)),
                    _ => rows.row(i) != rows.row(i - 1),
                })
                .collect();
            if n > 0 {
                *prev = Some(rows.row(n - 1).owned());
            }
            v
        }
    };
    if first && let Some(f) = out.first_mut() {
        *f = true;
    }
    Ok(out)
}

/// A partition on disk.
pub(super) struct OnDisk {
    pub(super) file: SpillFile,
    rows: u64,
    /// The most rows of a batch in the file.
    pub(super) batch_rows: usize,
    /// Per function: its value over the whole partition (aggregates over
    /// the whole partition only).
    whole: Vec<Option<ArrayRef>>,
}

/// The result of [`copy_partition`].
pub(super) struct Copied {
    pub(super) disk: OnDisk,
    /// The rows after the partition in the last batch read.
    pub(super) rest: Option<RecordBatch>,
    /// The partitioning key of the partition's rows.
    pub(super) key: Option<OwnedRow>,
}

/// Writes one partition to `writer`: the rows of `head` (all of it in the
/// partition), then the batches `next` gives up to the first row of
/// another partition. Folds the aggregates over the whole partition on the
/// way. Blocking (file I/O); checks for cancellation between batches.
pub(super) fn copy_partition(
    env: &Env,
    head: Vec<RecordBatch>,
    mut next: impl FnMut() -> Result<Option<RecordBatch>>,
    mut writer: SpillWriter,
    query: &QueryContext,
) -> Result<Copied> {
    let mut whole: Vec<Option<Fold>> = Vec::with_capacity(env.plan.functions.len());
    for f in &env.plan.functions {
        whole.push(match &f.function {
            WindowFunction::Aggregate(a) if f.shape()? == Shape::Whole => {
                Some(Fold::new(a, env.ansi)?)
            }
            _ => None,
        });
    }
    let (mut rows, mut batch_rows) = (0_u64, 0_usize);
    let mut take = |b: &RecordBatch| -> Result<()> {
        let n = b.num_rows();
        if n == 0 {
            return Ok(());
        }
        let none = vec![false; n];
        for (j, fold) in whole.iter_mut().enumerate() {
            if let Some(fold) = fold {
                fold_batch(env, j, fold, b, &none)?;
            }
        }
        crate::join::write_sliced(&mut writer, &Batch::new(b.clone()))?;
        rows += n as u64;
        batch_rows = batch_rows.max(n);
        Ok(())
    };
    let mut key: Option<OwnedRow> = None;
    let last_key = |b: &RecordBatch| -> Result<Option<OwnedRow>> {
        Ok(match &env.partition_keys {
            Some(enc) if b.num_rows() > 0 => Some(enc.encode(b)?.row(b.num_rows() - 1).owned()),
            _ => None,
        })
    };
    // Each batch is dropped once written.
    for b in head {
        take(&b)?;
        if let Some(k) = last_key(&b)? {
            key = Some(k);
        }
    }
    let mut rest = None;
    while let Some(b) = next()? {
        query.check_cancelled()?;
        let n = b.num_rows();
        let split = match &env.partition_keys {
            None => None,
            Some(enc) => {
                let keys = enc.encode(&b)?;
                let at = (0..n).find(|&i| match i {
                    0 => key.as_ref().is_some_and(|k| k.row() != keys.row(0)),
                    _ => keys.row(i) != keys.row(i - 1),
                });
                if n > 0 {
                    let last = at.unwrap_or(n);
                    if last > 0 {
                        key = Some(keys.row(last - 1).owned());
                    }
                }
                at
            }
        };
        if let Some(i) = split {
            take(&b.slice(0, i))?;
            rest = Some(b.slice(i, n - i));
            break;
        }
        take(&b)?;
    }
    let mut values = Vec::with_capacity(whole.len());
    for fold in &mut whole {
        values.push(fold.as_mut().map(Fold::value).transpose()?);
    }
    Ok(Copied {
        disk: OnDisk {
            file: writer.finish()?,
            rows,
            batch_rows,
            whole: values,
        },
        rest,
        key,
    })
}

/// Peer groups the lead reader closed while folding one batch.
struct Chunk {
    /// Each group's end (exclusive row number in the partition).
    ends: Vec<u64>,
    /// Per function: its value at each group's end (lead functions only).
    values: Vec<Option<ArrayRef>>,
}

/// The lead reader (see the [module docs](self)).
struct Lead {
    reader: SpillReader,
    /// The next batch, read but not folded yet, with its peer flags.
    held: Option<(RecordBatch, Vec<bool>)>,
    prev_order: Option<OwnedRow>,
    /// Rows read, and rows folded.
    read: u64,
    folded: u64,
    /// The end of the last group closed.
    closed: u64,
    /// Per function: its fold (aggregates over a `RANGE` frame up to the
    /// current row only).
    folds: Vec<Option<Fold>>,
    queue: VecDeque<Chunk>,
}

impl Lead {
    fn read(&mut self, env: &Env) -> Result<Option<(RecordBatch, Vec<bool>)>> {
        let Some(b) = self.reader.next().transpose()? else {
            return Ok(None);
        };
        let b = b.data().clone();
        let flags = peer_flags(
            env.order_keys.as_deref(),
            &b,
            &mut self.prev_order,
            self.read == 0,
        )?;
        self.read += b.num_rows() as u64;
        Ok(Some((b, flags)))
    }

    /// Folds batches until every group up to row `until` is closed.
    fn advance(&mut self, env: &Env, until: u64, query: &QueryContext) -> Result<()> {
        while self.closed < until {
            // A peer group may span the partition: this can read all of it.
            query.check_cancelled()?;
            let (cur, flags) = match self.held.take() {
                Some(c) => c,
                None => self
                    .read(env)?
                    .ok_or_else(|| internal("the window's lead reader ended early".to_owned()))?,
            };
            let next = self.read(env)?;
            let n = cur.num_rows();
            let next_starts = next
                .as_ref()
                .is_none_or(|(_, f)| f.first().copied().unwrap_or(true));
            let snap: Vec<bool> = (0..n)
                .map(|i| if i + 1 < n { flags[i + 1] } else { next_starts })
                .collect();
            let mut values = Vec::with_capacity(self.folds.len());
            for (j, fold) in self.folds.iter_mut().enumerate() {
                values.push(match fold {
                    Some(fold) => Some(fold_batch(env, j, fold, &cur, &snap)?),
                    None => None,
                });
            }
            let ends: Vec<u64> = snap
                .iter()
                .enumerate()
                .filter(|(_, s)| **s)
                .map(|(i, _)| self.folded + i as u64 + 1)
                .collect();
            self.folded += n as u64;
            if let Some(&e) = ends.last() {
                self.closed = e;
                self.queue.push_back(Chunk { ends, values });
            }
            self.held = next;
        }
        Ok(())
    }
}

/// The emission of one partition from disk (see the [module docs](self)).
pub(super) struct Emit {
    disk: OnDisk,
    trail: SpillReader,
    lead: Option<Lead>,
    /// Rows emitted.
    at: u64,
    /// The current peer group's first row, and the peer groups so far.
    peer_start: u64,
    peers: u64,
    prev_order: Option<OwnedRow>,
    /// Per function: its fold (aggregates over a `ROWS` frame up to the
    /// current row only).
    running: Vec<Option<Fold>>,
    query: Arc<QueryContext>,
}

impl Emit {
    /// Opens the readers of `disk`'s file; they reserve their frames from
    /// `consumer`'s grant credit first.
    pub(super) fn open(
        env: &Env,
        disk: OnDisk,
        consumer: &Arc<ConsumerHandle>,
        query: Arc<QueryContext>,
    ) -> Result<Self> {
        let reader = || -> Result<SpillReader> {
            Ok(disk
                .file
                .reader(consumer.reservation().releasing_to_credit())?)
        };
        let mut running = Vec::with_capacity(env.plan.functions.len());
        let mut folds = Vec::with_capacity(env.plan.functions.len());
        let mut lead = false;
        for f in &env.plan.functions {
            let (r, l) = match &f.function {
                WindowFunction::Aggregate(a) => match f.shape()? {
                    Shape::Whole => (None, None),
                    Shape::RunningRows => (Some(Fold::new(a, env.ansi)?), None),
                    Shape::RunningPeers => (None, Some(Fold::new(a, env.ansi)?)),
                    // Refused before the copy (`spill.rs`).
                    Shape::Frame => {
                        return Err(internal("a frame evaluated from disk".to_owned()));
                    }
                },
                WindowFunction::CumeDist => {
                    lead = true;
                    (None, None)
                }
                _ => (None, None),
            };
            lead |= l.is_some();
            running.push(r);
            folds.push(l);
        }
        let lead = if lead {
            Some(Lead {
                reader: reader()?,
                held: None,
                prev_order: None,
                read: 0,
                folded: 0,
                closed: 0,
                folds,
                queue: VecDeque::new(),
            })
        } else {
            None
        };
        Ok(Self {
            trail: reader()?,
            disk,
            lead,
            at: 0,
            peer_start: 0,
            peers: 0,
            prev_order: None,
            running,
            query,
        })
    }

    /// The next output batch, `None` at the partition's end. Blocking.
    pub(super) fn step(&mut self, env: &Env) -> Result<Option<RecordBatch>> {
        let Some(b) = self.trail.next().transpose()? else {
            return Ok(None);
        };
        let b = b.data().clone();
        let n = b.num_rows();
        let first = self.at == 0;
        let flags = peer_flags(env.order_keys.as_deref(), &b, &mut self.prev_order, first)?;
        let end = self.at + n as u64;
        // Each row's peer group (in the lead's queue), when needed.
        let mut groups: Vec<(usize, usize)> = Vec::new();
        if let Some(lead) = &mut self.lead {
            lead.advance(env, end, &self.query)?;
            groups.reserve_exact(n);
            let (mut k, mut p) = (0, 0);
            for i in 0..n {
                let r = self.at + i as u64;
                loop {
                    let c = lead
                        .queue
                        .get(k)
                        .ok_or_else(|| internal("a window row past its lead".to_owned()))?;
                    if p < c.ends.len() && c.ends[p] > r {
                        break;
                    }
                    p += 1;
                    if p >= c.ends.len() {
                        (k, p) = (k + 1, 0);
                    }
                }
                groups.push((k, p));
            }
        }
        let (mut ranks, mut dense) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for (i, &start) in flags.iter().enumerate() {
            if start {
                self.peer_start = self.at + i as u64;
                self.peers += 1;
            }
            ranks.push(self.peer_start + 1);
            dense.push(self.peers);
        }
        let total = self.disk.rows;
        let int = |v: u64| i32::try_from(v).map_err(|_| too_large());
        let row = |i: usize| self.at + i as u64;
        let mut columns: Vec<ArrayRef> = b.columns().to_vec();
        for (j, f) in env.plan.functions.iter().enumerate() {
            let column: ArrayRef = match &f.function {
                WindowFunction::RowNumber => Arc::new(Int32Array::from(
                    (0..n)
                        .map(|i| int(row(i) + 1))
                        .collect::<Result<Vec<_>>>()?,
                )),
                WindowFunction::Rank => Arc::new(Int32Array::from(
                    ranks.iter().map(|&r| int(r)).collect::<Result<Vec<_>>>()?,
                )),
                WindowFunction::DenseRank => Arc::new(Int32Array::from(
                    dense.iter().map(|&r| int(r)).collect::<Result<Vec<_>>>()?,
                )),
                WindowFunction::PercentRank => {
                    Arc::new(Float64Array::from_iter_values(ranks.iter().map(|&r| {
                        if total <= 1 {
                            0.0
                        } else {
                            count_f64(as_usize(r - 1)) / count_f64(as_usize(total - 1))
                        }
                    })))
                }
                WindowFunction::CumeDist => {
                    let lead = self
                        .lead
                        .as_ref()
                        .ok_or_else(|| internal("cume_dist without a lead".to_owned()))?;
                    Arc::new(Float64Array::from_iter_values(groups.iter().map(
                        |&(k, p)| {
                            count_f64(as_usize(lead.queue[k].ends[p])) / count_f64(as_usize(total))
                        },
                    )))
                }
                WindowFunction::Ntile(buckets) => Arc::new(Int32Array::from(
                    (0..n)
                        .map(|i| {
                            ntile(as_usize(row(i)), as_usize(total), *buckets).ok_or_else(too_large)
                        })
                        .collect::<Result<Vec<_>>>()?,
                )),
                // Refused before the copy (`spill.rs`).
                WindowFunction::Offset { .. } | WindowFunction::NthValue { .. } => {
                    return Err(internal(
                        "an offset function evaluated from disk".to_owned(),
                    ));
                }
                WindowFunction::Aggregate(_) => match f.shape()? {
                    Shape::Whole => {
                        let v = self.disk.whole[j].as_ref().ok_or_else(|| {
                            internal("a whole-partition value missing".to_owned())
                        })?;
                        let zeros = UInt32Array::from(vec![0_u32; n]);
                        arrow_select::take::take(v.as_ref(), &zeros, None)
                            .map_err(|e| internal(e.to_string()))?
                    }
                    Shape::RunningRows => {
                        let fold = self.running[j]
                            .as_mut()
                            .ok_or_else(|| internal("a running fold missing".to_owned()))?;
                        fold_batch(env, j, fold, &b, &vec![true; n])?
                    }
                    Shape::Frame => {
                        return Err(internal("a frame evaluated from disk".to_owned()));
                    }
                    Shape::RunningPeers => {
                        let lead = self
                            .lead
                            .as_ref()
                            .ok_or_else(|| internal("a peer frame without a lead".to_owned()))?;
                        let sources: Vec<&dyn Array> = lead
                            .queue
                            .iter()
                            .map(|c| {
                                c.values[j]
                                    .as_deref()
                                    .ok_or_else(|| internal("a lead value missing".to_owned()))
                            })
                            .collect::<Result<_>>()?;
                        arrow_select::interleave::interleave(&sources, &groups)
                            .map_err(|e| internal(e.to_string()))?
                    }
                },
            };
            columns.push(column);
        }
        // Groups that end before the next row are done with.
        if let Some(lead) = &mut self.lead {
            while lead
                .queue
                .front()
                .and_then(|c| c.ends.last())
                .is_some_and(|&e| e <= end)
            {
                lead.queue.pop_front();
            }
        }
        self.at = end;
        let options = RecordBatchOptions::new().with_row_count(Some(n));
        RecordBatch::try_new_with_options(Arc::clone(&env.schema), columns, &options)
            .map(Some)
            .map_err(|e| internal(format!("window produced an invalid batch: {e}")))
    }
}

/// A row count that fits memory as `usize` (counts of rows read).
fn as_usize(v: u64) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}
