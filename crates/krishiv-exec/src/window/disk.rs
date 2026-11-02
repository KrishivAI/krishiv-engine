// SPDX-License-Identifier: Apache-2.0
//! Frames and offset functions over a partition on disk (P2-T055; the
//! partition's other functions are in [`stream`](super::stream)).
//!
//! Every edge of a frame moves forward through the partition, so it streams:
//! - **Edges.** [`Walk`] is the in-memory frames' own walk, reading the keys
//!   of a `RANGE` frame's value bounds through a reader per edge.
//! - **`lag`, `lead`, `nth_value`.** The rows they read move forward too: a
//!   reader of their own, whatever the offset, keeping a batch's span of
//!   values. With `IGNORE NULLS` it keeps only the non-NULL ones.
//! - **Frames from the partition's start.** One fold, as Spark's
//!   `UnboundedPrecedingWindowFunctionFrame`, its values kept where frames
//!   end. Linear in the partition.
//! - **Other frames.** The batches a batch's frames span are held in memory
//!   and evaluated by [`frames::aggregate`], so values are the in-memory
//!   ones. When they do not fit (a frame to the partition's end, or a sliding
//!   frame wider than memory), each frame folds its rows from the file, in
//!   order: quadratic, as Spark's `UnboundedFollowingWindowFunctionFrame`
//!   and with a frame per batch, but bounded memory.
//!
//! # Memory
//! Each reader's frame is in the headroom the flush phase reserves
//! ([`readers`]). What this module holds beyond that (batches kept past their
//! read, states) is charged to the pool directly, never drawing the credit
//! the readers need.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use krishiv_memory::{ConsumerHandle, Reservation, SpillFile, SpillReader};

use super::eval::{Slice, cost, id};
use super::frames::{self, FrameKeys, RangeKeys, Target, Walk, is_range_offset};
use super::spill::Counters;
use super::stream::{Fold, fold_batch, whole_batch};
use super::{Env, FrameBound, FrameUnits, Shape, WindowFunction, WindowFunctionPlan, WindowPlan};
use crate::aggregate::AggregateExpr;
use crate::context::QueryContext;
use crate::error::{Result, internal};

/// Bytes per row of a function's output, with its gather indices.
const OUTPUT_BYTES: usize = 16 + size_of::<(usize, usize)>();

/// A reservation for `consumer` that takes from the pool, not its credit.
fn direct(consumer: &ConsumerHandle) -> Reservation {
    Reservation::new(Arc::clone(consumer.pool()), consumer.id())
}

/// A forward reader of the partition's rows.
struct Stream {
    reader: SpillReader,
    batch: Option<RecordBatch>,
    /// The rows of `batch`, `[start, end)`.
    start: usize,
    end: usize,
}

impl Stream {
    fn open(file: &SpillFile, consumer: &Arc<ConsumerHandle>) -> Result<Self> {
        Ok(Self {
            reader: file.reader(consumer.reservation().releasing_to_credit())?,
            batch: None,
            start: 0,
            end: 0,
        })
    }

    /// The next batch.
    fn next(&mut self) -> Result<Option<RecordBatch>> {
        let Some(b) = self.reader.next().transpose()? else {
            self.batch = None;
            return Ok(None);
        };
        let b = b.data().clone();
        self.start = self.end;
        self.end += b.num_rows();
        self.batch = Some(b.clone());
        Ok(Some(b))
    }

    /// Moves to the batch of row `j`, at or after the current one: the batch
    /// and the row in it.
    fn at(&mut self, j: usize) -> Result<(RecordBatch, usize)> {
        while j >= self.end {
            if self.next()?.is_none() {
                return Err(internal(format!("window row {j} past its partition")));
            }
        }
        match &self.batch {
            Some(b) if j >= self.start => Ok((b.clone(), j - self.start)),
            _ => Err(internal(format!("window row {j} read backwards"))),
        }
    }
}

/// The ordering keys of one edge of a `RANGE` frame.
struct KeyStream {
    stream: Stream,
    keys: Option<(usize, RangeKeys)>,
    column: usize,
    descending: bool,
    nulls_first: bool,
    ansi: bool,
}

impl KeyStream {
    fn cmp(&mut self, j: usize, target: Option<&Target>) -> Result<std::cmp::Ordering> {
        let (b, row) = self.stream.at(j)?;
        let start = self.stream.start;
        if self.keys.as_ref().is_none_or(|(s, _)| *s != start) {
            let keys = range_keys(
                &b,
                self.column,
                self.descending,
                self.nulls_first,
                self.ansi,
            )?;
            self.keys = Some((start, keys));
        }
        match &self.keys {
            Some((_, keys)) => keys.cmp(row, target),
            None => Err(internal("window keys missing".to_owned())),
        }
    }
}

fn range_keys(
    b: &RecordBatch,
    column: usize,
    descending: bool,
    nulls_first: bool,
    ansi: bool,
) -> Result<RangeKeys> {
    RangeKeys::new(&[whole_batch(b)], column, descending, nulls_first, ansi)?
        .ok_or_else(|| internal("a RANGE frame over a key without value bounds".to_owned()))
}

/// The keys of the current batch, as [`RangeKeys`] are for one edge.
pub(super) fn current_keys(env: &Env, b: &RecordBatch) -> Result<Option<RangeKeys>> {
    if !super::eval::has_range_offsets(&env.plan) {
        return Ok(None);
    }
    let [key] = env.plan.order_by.as_slice() else {
        return Ok(None);
    };
    range_keys(b, key.column, key.descending, key.nulls_first, env.ansi).map(Some)
}

/// What a batch's function needs of the rows being emitted.
pub(super) struct Current<'a> {
    pub(super) batch: &'a RecordBatch,
    /// The batch's first row in the partition.
    pub(super) at: usize,
    /// The partition's rows.
    pub(super) total: usize,
    /// Per row: its peer group, `[start, end)` (`end` only for a frame that
    /// asks for it, [`needs_peer_end`]).
    pub(super) peers: &'a [(usize, usize)],
    pub(super) keys: Option<&'a RangeKeys>,
    pub(super) counters: &'a Counters,
}

/// [`FrameKeys`] of a batch's rows.
struct DiskKeys<'a> {
    current: &'a RangeKeys,
    at: usize,
    lower: &'a mut Option<KeyStream>,
    upper: &'a mut Option<KeyStream>,
}

impl FrameKeys for DiskKeys<'_> {
    fn target(&self, i: usize, bound: FrameBound) -> Result<Option<Target>> {
        self.current.target(i - self.at, bound)
    }

    fn offset(&self, bound: FrameBound) -> FrameBound {
        self.current.offset(bound)
    }

    fn lower(&mut self, j: usize, target: Option<&Target>) -> Result<std::cmp::Ordering> {
        self.lower
            .as_mut()
            .ok_or_else(|| internal("a window frame's lower keys missing".to_owned()))?
            .cmp(j, target)
    }

    fn upper(&mut self, j: usize, target: Option<&Target>) -> Result<std::cmp::Ordering> {
        self.upper
            .as_mut()
            .ok_or_else(|| internal("a window frame's upper keys missing".to_owned()))?
            .cmp(j, target)
    }
}

/// The key readers a frame needs: (lower, upper).
fn key_readers(f: &WindowFunctionPlan) -> (bool, bool) {
    let frame = f.frame;
    let (l, u) = (is_range_offset(frame.lower), is_range_offset(frame.upper));
    if frame.units != FrameUnits::Range || !(l || u) {
        return (false, false);
    }
    let both = [frame.lower, frame.upper]
        .into_iter()
        .all(|b| is_range_offset(b) || b == FrameBound::CurrentRow);
    (both || l, both || u)
}

/// One frame's edges, batch after batch.
struct Edges {
    walk: Walk,
    lower: Option<KeyStream>,
    upper: Option<KeyStream>,
}

impl Edges {
    fn open(
        env: &Env,
        f: &WindowFunctionPlan,
        file: &SpillFile,
        consumer: &Arc<ConsumerHandle>,
    ) -> Result<Self> {
        let keys = |wanted: bool| -> Result<Option<KeyStream>> {
            if !wanted {
                return Ok(None);
            }
            let [key] = env.plan.order_by.as_slice() else {
                return Err(internal("a RANGE frame without its one key".to_owned()));
            };
            Ok(Some(KeyStream {
                stream: Stream::open(file, consumer)?,
                keys: None,
                column: key.column,
                descending: key.descending,
                nulls_first: key.nulls_first,
                ansi: env.ansi,
            }))
        };
        let (lower, upper) = key_readers(f);
        Ok(Self {
            walk: Walk::new(f.frame),
            lower: keys(lower)?,
            upper: keys(upper)?,
        })
    }

    /// The frames of the batch's rows, `[start, end)`.
    fn frames(&mut self, c: &Current<'_>) -> Result<Vec<(usize, usize)>> {
        let n = c.batch.num_rows();
        let mut keys = match c.keys {
            Some(current) if self.lower.is_some() || self.upper.is_some() => Some(DiskKeys {
                current,
                at: c.at,
                lower: &mut self.lower,
                upper: &mut self.upper,
            }),
            _ => None,
        };
        let mut out = Vec::with_capacity(n);
        for q in 0..n {
            let peers = || c.peers.get(q).copied().unwrap_or((0, 0));
            out.push(self.walk.row(c.at + q, (0, c.total), peers, &mut keys)?);
        }
        Ok(out)
    }
}

/// A batch's kept values.
struct Chunk {
    /// The value number of the first.
    first: usize,
    values: ArrayRef,
    /// `IGNORE NULLS`: each value's row.
    rows: Vec<usize>,
}

/// One column's values, read forward (see the [module docs](self)).
struct Values {
    stream: Stream,
    input: usize,
    skip_nulls: bool,
    /// Values read.
    seen: usize,
    done: bool,
    held: VecDeque<Chunk>,
    charge: Reservation,
    /// The bytes charged per chunk: it may hold its whole batch.
    frame: usize,
    data_type: arrow_schema::DataType,
}

impl Values {
    fn open(
        input: usize,
        skip_nulls: bool,
        data_type: &arrow_schema::DataType,
        file: &SpillFile,
        consumer: &Arc<ConsumerHandle>,
    ) -> Result<Self> {
        Ok(Self {
            stream: Stream::open(file, consumer)?,
            input,
            skip_nulls,
            seen: 0,
            done: false,
            held: VecDeque::new(),
            charge: direct(consumer).unguarded(),
            frame: file.max_read_reservation(),
            data_type: data_type.clone(),
        })
    }

    fn drop_before(&mut self, min: usize) {
        while self
            .held
            .front()
            .is_some_and(|c| c.first + c.values.len() <= min)
        {
            self.held.pop_front();
            self.charge.shrink(self.frame);
        }
    }

    /// Reads a batch, keeping values from number `min` on.
    fn load(&mut self, min: usize) -> Result<()> {
        let at = self.stream.end;
        let Some(b) = self.stream.next()? else {
            self.done = true;
            return Ok(());
        };
        let column = b
            .columns()
            .get(self.input)
            .ok_or_else(|| internal(format!("window column {} out of range", self.input)))?;
        let col = frames::viewed(column)?;
        let (values, rows) = if self.skip_nulls {
            let rows: Vec<usize> = (0..col.len())
                .filter(|&i| col.is_valid(i))
                .map(|i| at + i)
                .collect();
            let mask =
                BooleanArray::from((0..col.len()).map(|i| col.is_valid(i)).collect::<Vec<_>>());
            let kept = arrow_select::filter::filter(col.as_ref(), &mask)
                .map_err(|e| internal(e.to_string()))?;
            (kept, rows)
        } else {
            (col, Vec::new())
        };
        let first = self.seen;
        self.seen += values.len();
        self.drop_before(min);
        if !values.is_empty() && first + values.len() > min {
            self.charge
                .try_grow(self.frame)
                .map_err(krishiv_common::error::KrishivError::from)?;
            self.held.push_back(Chunk {
                first,
                values,
                rows,
            });
        }
        Ok(())
    }

    /// The row of value number `k` (`IGNORE NULLS`), keeping values from
    /// number `min` on; `None` past the last.
    fn row_of(&mut self, k: usize, min: usize) -> Result<Option<usize>> {
        while self.seen <= k && !self.done {
            self.load(min)?;
        }
        Ok(self
            .held
            .iter()
            .find(|c| k >= c.first && k < c.first + c.values.len())
            .and_then(|c| c.rows.get(k - c.first).copied()))
    }

    /// Per row: the value number it reads, in increasing order, or row
    /// `fallback`'s value (NULL without one) when there is none.
    fn gather(
        &mut self,
        targets: &[Option<usize>],
        fallback: Option<&ArrayRef>,
    ) -> Result<ArrayRef> {
        let min = targets.iter().flatten().min().copied();
        let max = targets.iter().flatten().max().copied();
        if let (Some(min), Some(max)) = (min, max) {
            while self.seen <= max && !self.done {
                self.load(min)?;
            }
            self.drop_before(min);
        }
        let mut sources: Vec<&dyn Array> = self.held.iter().map(|c| c.values.as_ref()).collect();
        let base = sources.len();
        if let Some(f) = fallback {
            sources.push(f.as_ref());
        }
        let null_source = sources.len();
        let null = arrow_array::new_null_array(&frames::view_of(&self.data_type), 1);
        sources.push(null.as_ref());
        let mut k = 0;
        let mut indices = Vec::with_capacity(targets.len());
        for (i, t) in targets.iter().enumerate() {
            let hit = match t {
                Some(t) if *t < self.seen => {
                    while self
                        .held
                        .get(k)
                        .is_some_and(|c| c.first + c.values.len() <= *t)
                    {
                        k += 1;
                    }
                    let c = self
                        .held
                        .get(k)
                        .filter(|c| c.first <= *t)
                        .ok_or_else(|| internal(format!("window value {t} not held")))?;
                    Some((k, *t - c.first))
                }
                _ => None,
            };
            indices.push(hit.unwrap_or(if fallback.is_some() {
                (base, i)
            } else {
                (null_source, 0)
            }));
        }
        arrow_select::interleave::interleave(&sources, &indices)
            .map_err(|e| internal(e.to_string()))
    }
}

/// Counts a column's non-NULL rows, moving forward.
struct Counter {
    stream: Stream,
    input: usize,
    upto: usize,
    before: usize,
}

impl Counter {
    /// The non-NULL rows before row `s`, at or after the last asked for.
    fn before(&mut self, s: usize) -> Result<usize> {
        while self.upto < s {
            let (b, r) = self.stream.at(self.upto)?;
            let col = b
                .columns()
                .get(self.input)
                .ok_or_else(|| internal(format!("window column {} out of range", self.input)))?;
            let stop = (s - self.stream.start).min(b.num_rows());
            self.before += (r..stop).filter(|&i| col.is_valid(i)).count();
            self.upto = self.stream.start + stop;
        }
        Ok(self.before)
    }
}

/// A frame evaluated from a batch's span of rows, or from the file.
struct Held {
    edges: Edges,
    /// `None` once the frames stopped fitting: they fold from the file.
    stream: Option<Stream>,
    held: VecDeque<(usize, RecordBatch)>,
    charge: Reservation,
    frame: usize,
}

impl Held {
    fn release(&mut self) {
        self.stream = None;
        self.held.clear();
        let size = self.charge.size();
        self.charge.shrink(size);
    }

    /// Holds the rows `[lo, hi)`; `false` if they do not fit.
    fn hold(&mut self, lo: usize, hi: usize, eval: usize) -> Result<bool> {
        let Some(stream) = &mut self.stream else {
            return Ok(false);
        };
        while self
            .held
            .front()
            .is_some_and(|(s, b)| s + b.num_rows() <= lo)
        {
            self.held.pop_front();
            self.charge.shrink(self.frame);
        }
        while stream.end < hi {
            let Some(b) = stream.next()? else {
                return Err(internal("a window frame past its partition".to_owned()));
            };
            if stream.end > lo {
                if self.charge.try_grow(self.frame).is_err() {
                    return Ok(false);
                }
                self.held.push_back((stream.start, b));
            }
        }
        if self.charge.try_grow(eval).is_err() {
            return Ok(false);
        }
        Ok(true)
    }

    fn slices(&self, lo: usize, hi: usize) -> Vec<Slice<'_>> {
        self.held
            .iter()
            .filter_map(|(s, b)| {
                let from = lo.max(*s);
                let to = hi.min(s + b.num_rows());
                (from < to).then(|| Slice {
                    batch: b,
                    offset: from - s,
                    len: to - from,
                    part_start: &[],
                    peer_start: &[],
                })
            })
            .collect()
    }
}

/// The aggregate of `f`.
fn aggregate_of(f: &WindowFunctionPlan) -> Result<&AggregateExpr> {
    match &f.function {
        WindowFunction::Aggregate(a) => Ok(a),
        _ => Err(internal(
            "a window fold of a function that is no aggregate".to_owned(),
        )),
    }
}

/// Each frame's rows folded from the file (see the [module docs](self)).
fn fold_from_file(
    env: &Env,
    j: usize,
    frames: &[(usize, usize)],
    file: &SpillFile,
    consumer: &Arc<ConsumerHandle>,
    query: &QueryContext,
) -> Result<ArrayRef> {
    let a = aggregate_of(&env.plan.functions[j])?;
    let mut charge = direct(consumer).unguarded();
    let state = crate::aggregate::Accumulator::new(a, env.ansi)?
        .group_bytes()
        .max(64)
        .saturating_add(size_of::<Fold>());
    charge
        .try_grow(state.saturating_mul(frames.len()))
        .map_err(krishiv_common::error::KrishivError::from)?;
    let mut folds = Vec::with_capacity(frames.len());
    for _ in frames {
        folds.push(Fold::new(a, env.ansi)?);
    }
    let live = || frames.iter().filter(|(s, e)| s < e);
    if let (Some(lo), Some(hi)) = (live().map(|f| f.0).min(), live().map(|f| f.1).max()) {
        let mut stream = Stream::open(file, consumer)?;
        let mut pos = lo;
        while pos < hi {
            query.check_cancelled()?;
            let (b, r) = stream.at(pos)?;
            let to = hi.min(stream.end);
            let part = b.slice(r, to - pos);
            let none = vec![false; to - pos];
            for (&(s, e), fold) in frames.iter().zip(&mut folds) {
                let (from, until) = (s.max(pos), e.min(to));
                if from < until {
                    let rows = part.slice(from - pos, until - from);
                    fold_batch(env, j, fold, &rows, &none[..until - from])?;
                }
            }
            pos = to;
        }
    }
    let values: Vec<ArrayRef> = folds.iter_mut().map(Fold::value).collect::<Result<_>>()?;
    let refs: Vec<&dyn Array> = values.iter().map(AsRef::as_ref).collect();
    arrow_select::concat::concat(&refs).map_err(|e| internal(e.to_string()))
}

/// How one function reads the partition's file.
pub(super) enum Framed {
    /// `lag`/`lead`.
    Offset { values: Box<OffsetFn> },
    /// `nth_value`.
    Nth(Box<Nth>),
    /// An aggregate over a frame from the partition's start.
    Prefix(Box<Prefix>),
    /// An aggregate over any other frame.
    Held(Box<HeldFrame>),
}

/// See [`Framed::Offset`].
pub(super) struct OffsetFn {
    values: Values,
    offset: i64,
    default: Option<usize>,
    ignore_nulls: bool,
    input: usize,
    /// Non-NULL rows before the next batch.
    before: i64,
}

/// See [`Framed::Nth`].
pub(super) struct Nth {
    edges: Edges,
    values: Values,
    counter: Option<Counter>,
    n: i64,
}

/// See [`Framed::Prefix`].
pub(super) struct Prefix {
    edges: Edges,
    stream: Stream,
    fold: Fold,
    folded: usize,
}

/// See [`Framed::Held`].
pub(super) struct HeldFrame(Held);

/// Whether a frame's end is its row's peer group's, which takes the lead
/// reader.
pub(super) fn needs_peer_end(f: &WindowFunctionPlan) -> bool {
    matches!(f.function, WindowFunction::NthValue { .. })
        && f.frame.units == FrameUnits::Range
        && f.frame.upper == FrameBound::CurrentRow
        && !is_range_offset(f.frame.lower)
}

/// Whether `f` reads the file through [`Framed`].
fn framed(f: &WindowFunctionPlan) -> Result<bool> {
    Ok(match &f.function {
        WindowFunction::Offset { .. } | WindowFunction::NthValue { .. } => true,
        WindowFunction::Aggregate(_) => f.shape()? == Shape::Frame,
        _ => false,
    })
}

/// The bytes evaluating a batch of `rows` rows holds out of the flush phase's
/// headroom: the functions `stream.rs` evaluates, and for each of the others
/// its output and the keys it reads. What they hold beyond that is
/// charged as they go (see the [module docs](self)).
pub(super) fn headroom(plan: &WindowPlan, rows: usize) -> Result<usize> {
    let mut own = plan.clone();
    own.functions.clear();
    let mut others = 0_usize;
    // The current rows' keys, and a batch's per key reader.
    let mut key_copies = 1_usize;
    for f in &plan.functions {
        if framed(f)? {
            others += 1;
            let (lower, upper) = key_readers(f);
            key_copies += usize::from(lower) + usize::from(upper);
        } else {
            own.functions.push(f.clone());
        }
    }
    let cost = cost(
        &own,
        rows,
        1,
        rows,
        rows.saturating_mul(rows.saturating_add(1)) / 2,
    )?;
    Ok(cost
        .saturating_add(rows.saturating_mul(others.saturating_mul(OUTPUT_BYTES)))
        .saturating_add(rows.saturating_mul(key_copies.saturating_mul(frames::RANGE_KEY_BYTES))))
}

/// The readers `f` opens beyond the partition's own.
pub(super) fn readers(f: &WindowFunctionPlan) -> Result<usize> {
    if !framed(f)? {
        return Ok(0);
    }
    let (lower, upper) = key_readers(f);
    let keys = usize::from(lower) + usize::from(upper);
    Ok(match &f.function {
        WindowFunction::NthValue {
            ignore_nulls: true, ..
        } => keys + 2,
        // A row's own reader; an offset function has no frame.
        _ => keys + 1,
    })
}

impl Framed {
    /// The state of function `j`, `None` if it does not read the file.
    pub(super) fn open(
        env: &Env,
        f: &WindowFunctionPlan,
        file: &SpillFile,
        consumer: &Arc<ConsumerHandle>,
    ) -> Result<Option<Self>> {
        if !framed(f)? {
            return Ok(None);
        }
        Ok(Some(match &f.function {
            WindowFunction::Offset {
                input,
                offset,
                default,
                ignore_nulls,
                data_type,
            } => {
                // `IGNORE NULLS` with an offset of 0 is the current row.
                let skip = *ignore_nulls && *offset != 0;
                Self::Offset {
                    values: Box::new(OffsetFn {
                        values: Values::open(*input, skip, data_type, file, consumer)?,
                        offset: *offset,
                        default: *default,
                        ignore_nulls: skip,
                        input: *input,
                        before: 0,
                    }),
                }
            }
            WindowFunction::NthValue {
                input,
                n,
                ignore_nulls,
                data_type,
            } => Self::Nth(Box::new(Nth {
                edges: Edges::open(env, f, file, consumer)?,
                values: Values::open(*input, *ignore_nulls, data_type, file, consumer)?,
                counter: if *ignore_nulls {
                    Some(Counter {
                        stream: Stream::open(file, consumer)?,
                        input: *input,
                        upto: 0,
                        before: 0,
                    })
                } else {
                    None
                },
                n: *n,
            })),
            WindowFunction::Aggregate(a) if f.frame.lower == FrameBound::UnboundedPreceding => {
                Self::Prefix(Box::new(Prefix {
                    edges: Edges::open(env, f, file, consumer)?,
                    stream: Stream::open(file, consumer)?,
                    fold: Fold::new(a, env.ansi)?,
                    folded: 0,
                }))
            }
            _ => Self::Held(Box::new(HeldFrame(Held {
                edges: Edges::open(env, f, file, consumer)?,
                stream: Some(Stream::open(file, consumer)?),
                held: VecDeque::new(),
                charge: direct(consumer),
                frame: file.max_read_reservation(),
            }))),
        }))
    }

    /// The values of function `j` over the batch's rows.
    pub(super) fn column(
        &mut self,
        env: &Env,
        j: usize,
        c: &Current<'_>,
        file: &SpillFile,
        consumer: &Arc<ConsumerHandle>,
        query: &QueryContext,
    ) -> Result<ArrayRef> {
        match self {
            Self::Offset { values } => values.column(c),
            Self::Nth(nth) => nth.column(c),
            Self::Prefix(p) => p.column(env, j, c, query),
            Self::Held(h) => h.0.column(env, j, c, file, consumer, query),
        }
    }
}

impl OffsetFn {
    fn column(&mut self, c: &Current<'_>) -> Result<ArrayRef> {
        let n = c.batch.num_rows();
        let col = c
            .batch
            .columns()
            .get(self.input)
            .ok_or_else(|| internal(format!("window column {} out of range", self.input)))?;
        let mut targets = Vec::with_capacity(n);
        for q in 0..n {
            let valid = i64::from(col.is_valid(q));
            let k = if self.ignore_nulls {
                let k = if self.offset > 0 {
                    (self.before + valid).saturating_add(self.offset - 1)
                } else {
                    self.before.saturating_add(self.offset)
                };
                usize::try_from(k).ok()
            } else {
                let row = i64::try_from(c.at + q).unwrap_or(i64::MAX);
                usize::try_from(row.saturating_add(self.offset))
                    .ok()
                    .filter(|&r| r < c.total)
            };
            targets.push(k);
            self.before += valid;
        }
        let fallback = self
            .default
            .map(|d| {
                let col = c
                    .batch
                    .columns()
                    .get(d)
                    .ok_or_else(|| internal(format!("window column {d} out of range")))?;
                frames::viewed(col)
            })
            .transpose()?;
        self.values.gather(&targets, fallback.as_ref())
    }
}

impl Nth {
    fn column(&mut self, c: &Current<'_>) -> Result<ArrayRef> {
        let frames = self.edges.frames(c)?;
        let skip = usize::try_from(self.n.saturating_sub(1)).unwrap_or(usize::MAX);
        let mut targets = Vec::with_capacity(frames.len());
        let mut min = None;
        for (s, e) in frames {
            if s >= e {
                targets.push(None);
                continue;
            }
            targets.push(match &mut self.counter {
                None => Some(s.saturating_add(skip)).filter(|&j| j < e),
                Some(counter) => {
                    let k = counter.before(s)?.saturating_add(skip);
                    let keep = *min.get_or_insert(k);
                    self.values
                        .row_of(k, keep)?
                        .filter(|&row| row < e)
                        .map(|_| k)
                }
            });
        }
        self.values.gather(&targets, None)
    }
}

impl Prefix {
    fn column(
        &mut self,
        env: &Env,
        j: usize,
        c: &Current<'_>,
        query: &QueryContext,
    ) -> Result<ArrayRef> {
        let ends: Vec<usize> = self.edges.frames(c)?.into_iter().map(|f| f.1).collect();
        let n = ends.len();
        // The state before this batch's rows are folded, then the state at
        // each end.
        let mut parts = vec![self.fold.value()?];
        let mut locs = Vec::with_capacity(n);
        let mut i = 0;
        while i < n && ends[i] <= self.folded {
            locs.push((0, 0));
            i += 1;
        }
        let stop = ends.last().copied().unwrap_or(0);
        while self.folded < stop {
            query.check_cancelled()?;
            let (b, r) = self.stream.at(self.folded)?;
            let to = stop.min(self.stream.end);
            let m = to - self.folded;
            let mut snap = vec![false; m];
            let mut k = 0;
            for (q, s) in snap.iter_mut().enumerate() {
                let end = self.folded + q + 1;
                if i < n && ends[i] == end {
                    *s = true;
                    while i < n && ends[i] == end {
                        locs.push((parts.len(), k));
                        i += 1;
                    }
                    k += 1;
                }
            }
            let kept = fold_batch(env, j, &mut self.fold, &b.slice(r, m), &snap)?;
            // A long frame end reads many batches for one output batch: only
            // those holding a wanted end stay, as many as the output has rows.
            if k > 0 {
                parts.push(kept);
            }
            self.folded = to;
        }
        if locs.len() != n {
            return Err(internal("a window frame's end moved back".to_owned()));
        }
        let refs: Vec<&dyn Array> = parts.iter().map(AsRef::as_ref).collect();
        arrow_select::interleave::interleave(&refs, &locs).map_err(|e| internal(e.to_string()))
    }
}

impl Held {
    fn column(
        &mut self,
        env: &Env,
        j: usize,
        c: &Current<'_>,
        file: &SpillFile,
        consumer: &Arc<ConsumerHandle>,
        query: &QueryContext,
    ) -> Result<ArrayRef> {
        let f = &env.plan.functions[j];
        let a = aggregate_of(f)?;
        let frames = self.edges.frames(c)?;
        let live = || frames.iter().filter(|(s, e)| s < e);
        let (Some(lo), Some(hi)) = (live().map(|f| f.0).min(), live().map(|f| f.1).max()) else {
            return self.no_frames(env, j, c, &frames, a, file, consumer, query);
        };
        // Results per frame, inputs per row held.
        let eval = frames::cost(a, f, (hi - lo).max(frames.len()))?;
        if self.hold(lo, hi, eval)? {
            let rebased: Vec<(u32, u32)> = frames
                .iter()
                .map(|&(s, e)| {
                    if s < e {
                        Ok((id(s - lo)?, id(e - lo)?))
                    } else {
                        Ok((0, 0))
                    }
                })
                .collect::<Result<_>>()?;
            let out = frames::aggregate(a, f, &self.slices(lo, hi), &rebased, env.ansi)?;
            self.charge.shrink(eval);
            return Ok(out);
        }
        self.release();
        c.counters.file_folds.fetch_add(1, Ordering::Relaxed);
        fold_from_file(env, j, &frames, file, consumer, query)
    }

    /// A batch whose frames are all empty: nothing held moves, because later
    /// frames may still read those rows (`ROWS BETWEEN 1000 PRECEDING AND 900
    /// PRECEDING` reads the first rows only after 900 empty frames). A row of
    /// the current batch gives the evaluation its input.
    #[expect(clippy::too_many_arguments, reason = "`column`'s context, passed on")]
    fn no_frames(
        &mut self,
        env: &Env,
        j: usize,
        c: &Current<'_>,
        frames: &[(usize, usize)],
        a: &AggregateExpr,
        file: &SpillFile,
        consumer: &Arc<ConsumerHandle>,
        query: &QueryContext,
    ) -> Result<ArrayRef> {
        let f = &env.plan.functions[j];
        let eval = frames::cost(a, f, frames.len().max(1))?;
        if self.charge.try_grow(eval).is_err() {
            return fold_from_file(env, j, frames, file, consumer, query);
        }
        let row = Slice {
            batch: c.batch,
            offset: 0,
            len: 1,
            part_start: &[],
            peer_start: &[],
        };
        let out = frames::aggregate(a, f, &[row], &vec![(0, 0); frames.len()], env.ansi);
        self.charge.shrink(eval);
        out
    }
}
