// SPDX-License-Identifier: Apache-2.0
//! Evaluation of complete partitions (see the module docs of
//! [`super`]): the boundary maps, ranking functions, and aggregates over
//! the whole partition or a running frame.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal64Type, Decimal128Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, PrimitiveArray, RecordBatch,
    UInt32Array,
};
use arrow_buffer::i256;
use arrow_schema::DataType;
use krishiv_common::error::KrishivUnsupportedFeature;
use krishiv_types::batch::Weights;

use super::{Shape, WindowFunction, WindowFunctionPlan, WindowPlan};
use crate::aggregate::{
    Accumulator, AggInput, AggregateExpr, AggregateFunction, Running, avg_decimal_wide,
    wide_sum_value,
};
use crate::error::{Result, internal};

/// Rows `offset..offset + len` of a buffered batch, with their boundary
/// flags.
pub(super) struct Slice<'a> {
    pub(super) batch: &'a RecordBatch,
    pub(super) offset: usize,
    pub(super) len: usize,
    pub(super) part_start: &'a [bool],
    pub(super) peer_start: &'a [bool],
}

/// Complete partitions, as consecutive slices, and their boundaries.
struct Unit<'a> {
    slices: &'a [Slice<'a>],
    rows: usize,
    /// Each row's partition.
    part_of: Vec<u32>,
    /// Each partition's rows, `[start, end)`.
    parts: Vec<(usize, usize)>,
    /// Each row's peer group.
    peer_of: Vec<u32>,
    /// Each peer group's rows, `[start, end)`.
    peers: Vec<(usize, usize)>,
    /// Each partition's first peer group.
    first_peer: Vec<u32>,
    /// The ordering key, for `RANGE` frames with value bounds.
    order: Option<super::frames::RangeKeys>,
}

/// A vector collected from fallible items with the exact capacity of the
/// iterator's length (a plain `collect` of `Result`s grows by doubling, so
/// it would hold more than the memory reserved for it).
struct Exact<T>(Result<Vec<T>>);

impl<T> FromIterator<Result<T>> for Exact<T> {
    fn from_iter<I: IntoIterator<Item = Result<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut out = Vec::with_capacity(iter.size_hint().0);
        for x in iter {
            match x {
                Ok(v) => out.push(v),
                Err(e) => return Self(Err(e)),
            }
        }
        Self(Ok(out))
    }
}

pub(super) fn id(i: usize) -> Result<u32> {
    u32::try_from(i).map_err(|_| too_large())
}

pub(super) fn too_large() -> krishiv_common::error::KrishivError {
    KrishivUnsupportedFeature {
        feature: "window evaluation over more than 2^32 - 1 rows at once".to_owned(),
    }
    .into()
}

impl<'a> Unit<'a> {
    fn new(slices: &'a [Slice<'a>]) -> Result<Self> {
        let rows: usize = slices.iter().map(|s| s.len).sum();
        // Sized exactly (what `cost` reserves); the first row starts both.
        let starts = |peer: bool| {
            1 + slices
                .iter()
                .flat_map(|s| if peer { s.peer_start } else { s.part_start }.iter())
                .filter(|&&b| b)
                .count()
        };
        let parts = starts(false);
        let peers = starts(true);
        let mut u = Self {
            slices,
            rows,
            part_of: Vec::with_capacity(rows),
            parts: Vec::with_capacity(parts),
            peer_of: Vec::with_capacity(rows),
            peers: Vec::with_capacity(peers),
            first_peer: Vec::with_capacity(parts),
            order: None,
        };
        let mut i = 0;
        for s in slices {
            for (&p, &q) in s.part_start.iter().zip(s.peer_start) {
                if p || i == 0 {
                    if let Some(last) = u.parts.last_mut() {
                        last.1 = i;
                    }
                    u.parts.push((i, i));
                    u.first_peer.push(id(u.peers.len())?);
                }
                if p || q || i == 0 {
                    if let Some(last) = u.peers.last_mut() {
                        last.1 = i;
                    }
                    u.peers.push((i, i));
                }
                u.part_of.push(id(u.parts.len() - 1)?);
                u.peer_of.push(id(u.peers.len() - 1)?);
                i += 1;
            }
        }
        if let Some(last) = u.parts.last_mut() {
            last.1 = rows;
        }
        if let Some(last) = u.peers.last_mut() {
            last.1 = rows;
        }
        Ok(u)
    }

    /// The partition of row `i`: `(start, rows)`.
    fn part(&self, i: usize) -> (usize, usize) {
        let (s, e) = self.parts[self.part_of[i] as usize];
        (s, e - s)
    }
}

/// Evaluates every function of `plan` over the complete partitions
/// `slices`: one column per function, a row per unit row.
pub(super) fn evaluate(
    plan: &WindowPlan,
    slices: &[Slice<'_>],
    ansi: bool,
) -> Result<Vec<ArrayRef>> {
    let mut unit = Unit::new(slices)?;
    if has_range_offsets(plan)
        && let [key] = plan.order_by.as_slice()
    {
        unit.order = super::frames::RangeKeys::new(
            slices,
            key.column,
            key.descending,
            key.nulls_first,
            ansi,
        )?;
    }
    plan.functions
        .iter()
        .map(|f| function(f, &unit, ansi))
        .collect()
}

/// Whether a function of `plan` has a `RANGE` frame with a value bound.
pub(super) fn has_range_offsets(plan: &WindowPlan) -> bool {
    plan.functions.iter().any(|f| {
        f.frame.units == super::FrameUnits::Range
            && (super::frames::is_range_offset(f.frame.lower)
                || super::frames::is_range_offset(f.frame.upper))
    })
}

fn function(f: &WindowFunctionPlan, u: &Unit<'_>, ansi: bool) -> Result<ArrayRef> {
    let n = u.rows;
    let int = |v: i64| i32::try_from(v).map_err(|_| too_large());
    Ok(match &f.function {
        WindowFunction::RowNumber => {
            let v: Vec<i32> = (0..n)
                .map(|i| {
                    i64::try_from(i - u.part(i).0 + 1)
                        .map_err(|_| too_large())
                        .and_then(int)
                })
                .collect::<Exact<_>>()
                .0?;
            Arc::new(Int32Array::from(v))
        }
        WindowFunction::Rank => Arc::new(Int32Array::from(ranks(u, int)?)),
        WindowFunction::DenseRank => {
            let v: Vec<i32> = (0..n)
                .map(|i| {
                    let first = u.first_peer[u.part_of[i] as usize];
                    int(i64::from(u.peer_of[i] - first) + 1)
                })
                .collect::<Exact<_>>()
                .0?;
            Arc::new(Int32Array::from(v))
        }
        WindowFunction::PercentRank => {
            let r = ranks(u, int)?;
            let v: Vec<f64> = (0..n)
                .map(|i| {
                    let rows = u.part(i).1;
                    if rows <= 1 {
                        0.0
                    } else {
                        f64::from(r[i] - 1) / count_f64(rows - 1)
                    }
                })
                .collect();
            Arc::new(Float64Array::from(v))
        }
        WindowFunction::CumeDist => {
            let v: Vec<f64> = (0..n)
                .map(|i| {
                    let (start, rows) = u.part(i);
                    let end = u.peers[u.peer_of[i] as usize].1;
                    count_f64(end - start) / count_f64(rows)
                })
                .collect();
            Arc::new(Float64Array::from(v))
        }
        WindowFunction::Ntile(k) => {
            let v: Vec<i32> = (0..n)
                .map(|i| {
                    let (start, rows) = u.part(i);
                    ntile(i - start, rows, *k).ok_or_else(too_large)
                })
                .collect::<Exact<_>>()
                .0?;
            Arc::new(Int32Array::from(v))
        }
        WindowFunction::Aggregate(a) => aggregate(a, f, u, ansi)?,
        WindowFunction::Offset { .. } | WindowFunction::NthValue { .. } => {
            let part = |i: usize| u.parts[u.part_of[i] as usize];
            let peers = |i: usize| u.peers[u.peer_of[i] as usize];
            let rows = super::frames::Rows {
                part: &part,
                peers: &peers,
                n: u.rows,
                order: u.order.as_ref(),
            };
            if matches!(f.function, WindowFunction::Offset { .. }) {
                super::frames::offset_value(f, u.slices, &rows)?
            } else {
                super::frames::nth_value(f, u.slices, &rows)?
            }
        }
    })
}

/// `count` as a double (Spark divides counts as doubles).
pub(super) fn count_f64(count: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "Spark divides row counts as doubles"
    )]
    let v = count as f64;
    v
}

fn ranks(u: &Unit<'_>, int: impl Fn(i64) -> Result<i32>) -> Result<Vec<i32>> {
    (0..u.rows)
        .map(|i| {
            let start = u.part(i).0;
            let peer = u.peers[u.peer_of[i] as usize].0;
            i64::try_from(peer - start + 1)
                .map_err(|_| too_large())
                .and_then(&int)
        })
        .collect::<Exact<_>>()
        .0
}

/// Spark's `NTile` bucket of the 0-based row `r` of a partition of `rows`
/// rows into `k` buckets: buckets of `rows / k` rows, the first
/// `rows % k` one row larger.
pub(super) fn ntile(r: usize, rows: usize, k: i32) -> Option<i32> {
    let k = usize::try_from(k).ok().filter(|&k| k > 0)?;
    let (size, extra) = (rows / k, rows % k);
    let bucket = if size == 0 {
        r
    } else if r < extra * (size + 1) {
        r / (size + 1)
    } else {
        extra + (r - extra * (size + 1)) / size
    };
    i32::try_from(bucket + 1).ok()
}

/// The argument columns of `a` over slice `s`, and whether each row of the
/// slice passes `filter`.
pub(super) fn arguments(a: &AggregateExpr, s: &Slice<'_>) -> Result<Vec<ArrayRef>> {
    a.args
        .iter()
        .map(|&c| {
            s.batch
                .columns()
                .get(c)
                .map(|col| col.slice(s.offset, s.len))
                .ok_or_else(|| internal(format!("window argument column {c} out of range")))
        })
        .collect()
}

pub(super) fn filter_mask(f: &WindowFunctionPlan, s: &Slice<'_>) -> Result<Option<BooleanArray>> {
    let Some(c) = f.filter else {
        return Ok(None);
    };
    let col = s
        .batch
        .columns()
        .get(c)
        .ok_or_else(|| internal(format!("window filter column {c} out of range")))?;
    Ok(Some(col.slice(s.offset, s.len).as_boolean().clone()))
}

/// Whether row `i` of a filter mask passes (`true`, not NULL).
pub(super) fn passes(mask: Option<&BooleanArray>, i: usize) -> bool {
    mask.is_none_or(|m| m.is_valid(i) && m.value(i))
}

fn aggregate(
    a: &AggregateExpr,
    f: &WindowFunctionPlan,
    u: &Unit<'_>,
    ansi: bool,
) -> Result<ArrayRef> {
    let shape = f.shape()?;
    if shape == Shape::Frame {
        let part = |i: usize| u.parts[u.part_of[i] as usize];
        let peers = |i: usize| u.peers[u.peer_of[i] as usize];
        let rows = super::frames::Rows {
            part: &part,
            peers: &peers,
            n: u.rows,
            order: u.order.as_ref(),
        };
        let frames = super::frames::frames(f.frame, &rows)?;
        return super::frames::aggregate(a, f, u.slices, &frames, ansi);
    }
    if exact_decimal(a) {
        return decimal_aggregate(a, f, shape, u, ansi);
    }
    if picks_bytes(a) {
        return byte_values(a, f, shape, u);
    }
    // The argument columns are passed as they are, in order. The window
    // applies the filter itself (see `filter_mask`); the accumulator
    // never sees it.
    let local = AggregateExpr {
        args: (0..a.args.len()).collect(),
        input: AggInput::Raw,
        mask: None,
        ..a.clone()
    };
    let mut acc = Accumulator::new(&local, ansi)?;
    let parts = u.parts.len();
    // Group `parts` takes rows the filter drops; it is never evaluated.
    let trash = id(parts)?;
    let one = Weights::Const(1);
    match shape {
        Shape::Whole => {
            acc.resize(parts + 1);
            let mut at = 0;
            let mut groups = Vec::new();
            for s in u.slices {
                let mask = filter_mask(f, s)?;
                groups.clear();
                groups.extend((0..s.len).map(|i| {
                    if passes(mask.as_ref(), i) {
                        u.part_of[at + i]
                    } else {
                        trash
                    }
                }));
                acc.update(&groups, &arguments(a, s)?, &one)?;
                at += s.len;
            }
            let ids: Vec<u32> = (0..parts).map(id).collect::<Exact<_>>().0?;
            let values = acc.evaluate(&ids)?;
            take(&values, &u.part_of)
        }
        Shape::Frame => Err(internal("a window frame evaluated above".to_owned())),
        Shape::RunningRows | Shape::RunningPeers => {
            // Group `p` folds partition `p`; group `parts + 1 + k` keeps the
            // state after segment `k` (a row or a peer group).
            let segments = match shape {
                Shape::RunningRows => u.rows,
                _ => u.peers.len(),
            };
            acc.resize(parts + 1 + segments);
            let mut groups: Vec<u32> = Vec::new();
            let mut snapshots: Vec<u32> = Vec::new();
            let mut at = 0;
            let mut segment = 0;
            for s in u.slices {
                let mask = filter_mask(f, s)?;
                let args = arguments(a, s)?;
                let parts_of = &u.part_of[at..at + s.len];
                // Without a filter every row folds into its partition's group.
                if mask.is_some() {
                    groups.clear();
                    groups.extend(
                        parts_of
                            .iter()
                            .enumerate()
                            .map(|(j, &p)| if passes(mask.as_ref(), j) { p } else { trash }),
                    );
                }
                let folded: &[u32] = if mask.is_some() { &groups } else { parts_of };
                // A segment's state is kept after its last row.
                snapshots.clear();
                for row in at..at + s.len {
                    let last = match shape {
                        Shape::RunningRows => true,
                        _ => row + 1 == u.peers[u.peer_of[row] as usize].1,
                    };
                    snapshots.push(if last {
                        let slot = id(parts + 1 + segment)?;
                        if slot == Running::NONE {
                            return Err(too_large());
                        }
                        segment += 1;
                        slot
                    } else {
                        Running::NONE
                    });
                }
                acc.update_running(
                    folded,
                    &args,
                    &Running {
                        parts: parts_of,
                        snapshots: &snapshots,
                    },
                )?;
                at += s.len;
            }
            let ids: Vec<u32> = (0..segments)
                .map(|k| id(parts + 1 + k))
                .collect::<Exact<_>>()
                .0?;
            let values = acc.evaluate(&ids)?;
            match shape {
                // One segment per row, in row order: the values are the
                // output.
                Shape::RunningRows => Ok(values),
                _ => take(&values, &u.peer_of),
            }
        }
    }
}

/// `values[indices[i]]` for each row.
pub(super) fn take(values: &ArrayRef, indices: &[u32]) -> Result<ArrayRef> {
    let idx = UInt32Array::from(indices.to_vec());
    arrow_select::take::take(values.as_ref(), &idx, None).map_err(|e| internal(e.to_string()))
}

/// The unscaled values of a decimal column, in `i128`.
pub(super) fn decimals(col: &ArrayRef) -> Result<Vec<Option<i128>>> {
    Ok(match col.data_type() {
        DataType::Decimal64(..) => col
            .as_primitive::<Decimal64Type>()
            .iter()
            .map(|v| v.map(i128::from))
            .collect(),
        DataType::Decimal128(..) => col.as_primitive::<Decimal128Type>().iter().collect(),
        t => return Err(internal(format!("window decimal aggregate over {t}"))),
    })
}

/// A decimal `sum` or `avg` over a frame, kept exact (see the module docs
/// of [`super`]).
fn decimal_aggregate(
    a: &AggregateExpr,
    f: &WindowFunctionPlan,
    shape: Shape,
    u: &Unit<'_>,
    ansi: bool,
) -> Result<ArrayRef> {
    // The state after each segment: (exact sum, rows summed). Sized exactly
    // (what `cost` reserves): one state per segment.
    let segments = match shape {
        Shape::Whole => u.parts.len(),
        Shape::RunningRows | Shape::Frame => u.rows,
        Shape::RunningPeers => u.peers.len(),
    };
    let mut states: Vec<(i256, i64)> = Vec::with_capacity(segments);
    let mut seg_of: Vec<u32> = Vec::with_capacity(u.rows);
    let (mut sum, mut count) = (i256::ZERO, 0_i64);
    let mut at = 0;
    for sl in u.slices {
        let mask = filter_mask(f, sl)?;
        let col = arguments(a, sl)?
            .into_iter()
            .next()
            .ok_or_else(|| internal("decimal window aggregate without argument".to_owned()))?;
        let values = decimals(&col)?;
        for (i, v) in values.iter().enumerate() {
            let row = at + i;
            if u.parts[u.part_of[row] as usize].0 == row {
                (sum, count) = (i256::ZERO, 0);
            }
            if let Some(v) = v
                && passes(mask.as_ref(), i)
            {
                sum = sum.wrapping_add(i256::from_i128(*v));
                count += 1;
            }
            let end = match shape {
                Shape::Whole => u.parts[u.part_of[row] as usize].1,
                Shape::RunningRows | Shape::Frame => row + 1,
                Shape::RunningPeers => u.peers[u.peer_of[row] as usize].1,
            };
            if row + 1 == end {
                states.push((sum, count));
            }
        }
        at += sl.len;
    }
    // Map rows to the state of their segment.
    for row in 0..u.rows {
        let k = match shape {
            Shape::Whole => u.part_of[row] as usize,
            Shape::RunningRows | Shape::Frame => row,
            Shape::RunningPeers => u.peer_of[row] as usize,
        };
        seg_of.push(id(k)?);
    }
    let mut out: Vec<Option<i128>> = Vec::with_capacity(states.len());
    for &(sum, count) in &states {
        out.push(decimal_value(a, sum, count, ansi)?);
    }
    take(&decimal_array(a, out)?, &seg_of)
}

/// The value of a decimal `sum` or `avg` whose exact state is `sum` over
/// `count` rows: `NULL` without rows or past the result's precision (with
/// ANSI, `NUMERIC_VALUE_OUT_OF_RANGE`).
pub(super) fn decimal_value(
    a: &AggregateExpr,
    sum: i256,
    count: i64,
    ansi: bool,
) -> Result<Option<i128>> {
    let (DataType::Decimal64(p, s) | DataType::Decimal128(p, s)) = a.result_type()? else {
        return Err(internal("decimal window aggregate type".to_owned()));
    };
    let (DataType::Decimal64(_, in_scale) | DataType::Decimal128(_, in_scale)) = a.arg_type else {
        return Err(internal("decimal window aggregate input".to_owned()));
    };
    Ok(if count == 0 {
        None
    } else if a.function == AggregateFunction::Avg {
        match avg_decimal_wide(sum, in_scale, count, p, s) {
            (Some(v), _) => Some(v),
            (None, text) if ansi => {
                return Err(crate::aggregate::out_of_range(text, p, s));
            }
            (None, _) => None,
        }
    } else {
        wide_sum_value(Some(sum), p, s, ansi)?
    })
}

/// Decimal values of `a`'s result type.
pub(super) fn decimal_array(a: &AggregateExpr, out: Vec<Option<i128>>) -> Result<ArrayRef> {
    let result = a.result_type()?;
    Ok(match result {
        DataType::Decimal64(..) => {
            let v: Vec<Option<i64>> = out
                .into_iter()
                .map(|x| x.map(i64::try_from).transpose())
                .collect::<Result<_, _>>()
                .map_err(|_| internal("decimal64 window value out of range".to_owned()))?;
            Arc::new(PrimitiveArray::<Decimal64Type>::from(v).with_data_type(result))
        }
        _ => Arc::new(PrimitiveArray::<Decimal128Type>::from(out).with_data_type(result)),
    })
}

/// Whether `a` is a decimal `sum` or `avg` (kept exact, see the module
/// docs of [`super`]).
pub(super) fn exact_decimal(a: &AggregateExpr) -> bool {
    matches!(
        a.arg_type,
        DataType::Decimal64(..) | DataType::Decimal128(..)
    ) && matches!(a.function, AggregateFunction::Sum | AggregateFunction::Avg)
}

/// Whether `t` is a string or binary type.
fn is_bytes(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Utf8
            | DataType::Utf8View
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::BinaryView
            | DataType::LargeBinary
    )
}

/// Whether `a` picks one row's string or binary value (`min`, `max`,
/// `first`, `last`, `any_value`): evaluated by [`byte_values`], without
/// copying values.
pub(super) fn picks_bytes(a: &AggregateExpr) -> bool {
    use AggregateFunction as F;
    is_bytes(&a.arg_type)
        && matches!(
            a.function,
            F::Min | F::Max | F::First { .. } | F::Last { .. } | F::AnyValue { .. }
        )
}

/// The heap a running frame's snapshots of a growing state hold: each row
/// of a partition copies the whole state so far, which is quadratic in the
/// partition's rows (`collect_list`, `collect_set`, `mode`, `listagg`).
/// `snapshot_bytes` is the bytes all snapshots hold (see [`snapshot_slots`]
/// and [`snapshot_width`]), an upper bound for any partition sizes
/// (distinct keys and peer groups snapshot less).
fn snapshot_heap(a: &AggregateExpr, shape: Shape, snapshot_bytes: usize) -> usize {
    if snapshots_grow(a, shape) {
        snapshot_bytes
    } else {
        0
    }
}

/// The bytes one snapshotted element of the growing aggregates of `plan`
/// takes in `batch`: the widest average value of their argument columns
/// (doubled for `collect_set` and `mode`, which also keep encoded keys), at
/// least 16. A later batch of a partition may hold wider values than this
/// one.
pub(super) fn snapshot_width(plan: &WindowPlan, batch: &RecordBatch) -> usize {
    let rows = batch.num_rows().max(1);
    let mut width = 16;
    for f in &plan.functions {
        let WindowFunction::Aggregate(a) = &f.function else {
            continue;
        };
        if !f.shape().is_ok_and(|s| snapshots_grow(a, s)) {
            continue;
        }
        let keyed = matches!(
            a.function,
            AggregateFunction::CollectSet
                | AggregateFunction::Mode
                | AggregateFunction::ModeMin
                | AggregateFunction::ModeMax
        );
        for &i in &a.args {
            if let Some(col) = batch.columns().get(i) {
                let per = crate::aggregate::kept_bytes(col).div_ceil(rows);
                width = width.max(if keyed { per.saturating_mul(2) } else { per });
            }
        }
    }
    width
}

/// Whether a running frame of `a` snapshots a state that grows with the
/// partition (`collect_list`, `collect_set`, `mode`, `listagg`).
pub(super) fn snapshots_grow(a: &AggregateExpr, shape: Shape) -> bool {
    use AggregateFunction as F;
    shape != Shape::Whole
        && matches!(
            a.function,
            F::CollectList | F::CollectSet | F::Mode | F::ModeMin | F::ModeMax | F::ListAgg { .. }
        )
}

/// The elements the snapshots of a growing state hold for the rows of one
/// batch: the row at position `k` of its partition copies `k` elements.
/// `part_start` flags the rows that start a partition; `open` is the length
/// of the partition the batch continues (rows of earlier batches).
/// Additive across batches, so the cost of a partition is the sum of its
/// batches' costs whatever the skew. Returns the elements and the length of
/// the partition open at the batch's end.
pub(super) fn snapshot_slots(part_start: &[bool], open: usize) -> (usize, usize) {
    let mut held = 0usize;
    let mut len = open;
    for &start in part_start {
        if start {
            len = 0;
        }
        len = len.saturating_add(1);
        held = held.saturating_add(len);
    }
    (held, len)
}

/// The view type a string or binary argument is read as.
pub(super) fn view_type(t: &DataType) -> DataType {
    match t {
        DataType::Binary | DataType::BinaryView | DataType::LargeBinary => DataType::BinaryView,
        _ => DataType::Utf8View,
    }
}

/// A string or binary column as views of its own buffers (arrow-array
/// builds them without copying values when the offsets fit `u32`).
pub(super) fn to_view(col: &ArrayRef) -> Result<ArrayRef> {
    use arrow_array::{BinaryViewArray, StringViewArray};
    Ok(match col.data_type() {
        DataType::Utf8View | DataType::BinaryView => Arc::clone(col),
        DataType::Utf8 => Arc::new(StringViewArray::from(col.as_string::<i32>())),
        DataType::LargeUtf8 => Arc::new(StringViewArray::from(col.as_string::<i64>())),
        DataType::Binary => Arc::new(BinaryViewArray::from(col.as_binary::<i32>())),
        DataType::LargeBinary => Arc::new(BinaryViewArray::from(col.as_binary::<i64>())),
        t => return Err(internal(format!("window value of {t} read as bytes"))),
    })
}

/// The last row of the segment of `row` (exclusive end) and its index.
fn segment(shape: Shape, u: &Unit<'_>, row: usize) -> (usize, usize) {
    match shape {
        Shape::Whole => {
            let p = u.part_of[row] as usize;
            (u.parts[p].1, p)
        }
        Shape::RunningRows | Shape::Frame => (row + 1, row),
        Shape::RunningPeers => {
            let p = u.peer_of[row] as usize;
            (u.peers[p].1, p)
        }
    }
}

/// `min`/`max`/`first`/`last`/`any_value` of string or binary values over a
/// frame: the row each frame picks, then the picked values gathered as views
/// of the input's buffers (no value is copied; Spark compares by bytes).
fn byte_values(
    a: &AggregateExpr,
    f: &WindowFunctionPlan,
    shape: Shape,
    u: &Unit<'_>,
) -> Result<ArrayRef> {
    let view = view_type(&a.arg_type);
    let mut sources: Vec<ArrayRef> = Vec::with_capacity(u.slices.len() + 1);
    for sl in u.slices {
        let col = arguments(a, sl)?
            .into_iter()
            .next()
            .ok_or_else(|| internal("a window aggregate without argument".to_owned()))?;
        sources.push(to_view(&col)?);
    }
    let null_source = sources.len();
    sources.push(arrow_array::new_null_array(&view, 1));
    let value = |k: usize, i: usize| -> &[u8] {
        let c = &sources[k];
        match c.data_type() {
            DataType::Utf8View => c.as_string_view().value(i).as_bytes(),
            _ => c.as_binary_view().value(i),
        }
    };
    let mut chosen: Vec<(usize, usize)> = Vec::new();
    let mut state: Option<(usize, usize)> = None;
    let mut at = 0;
    for (k, sl) in u.slices.iter().enumerate() {
        let mask = filter_mask(f, sl)?;
        for i in 0..sl.len {
            let row = at + i;
            if u.parts[u.part_of[row] as usize].0 == row {
                state = None;
            }
            if passes(mask.as_ref(), i) {
                let valid = sources[k].is_valid(i);
                let picked = state.map(|(sk, si)| sources[sk].is_valid(si).then(|| value(sk, si)));
                if replaces(&a.function, picked, valid, || value(k, i))? {
                    state = Some((k, i));
                }
            }
            if row + 1 == segment(shape, u, row).0 {
                chosen.push(state.unwrap_or((null_source, 0)));
            }
        }
        at += sl.len;
    }
    let indices: Vec<(usize, usize)> = (0..u.rows)
        .map(|row| chosen[segment(shape, u, row).1])
        .collect();
    let refs: Vec<&dyn Array> = sources.iter().map(AsRef::as_ref).collect();
    arrow_select::interleave::interleave(&refs, &indices).map_err(|e| internal(e.to_string()))
}

/// Whether a row (`valid`, its bytes `value`) replaces the value a
/// `min`/`max`/`first`/`last`/`any_value` of strings or binaries picked so
/// far (`picked`: `None` before any row, `Some(None)` for a picked NULL);
/// Spark compares by bytes.
pub(super) fn replaces<'v>(
    function: &AggregateFunction,
    picked: Option<Option<&'v [u8]>>,
    valid: bool,
    value: impl FnOnce() -> &'v [u8],
) -> Result<bool> {
    use AggregateFunction as F;
    Ok(match function {
        F::Min | F::Max if !valid => false,
        F::Min | F::Max => match picked.flatten() {
            Some(old) => {
                let order = value().cmp(old);
                if *function == F::Max {
                    order.is_gt()
                } else {
                    order.is_lt()
                }
            }
            None => true,
        },
        F::First { ignore_nulls } | F::AnyValue { ignore_nulls } => {
            picked.is_none() && (valid || !ignore_nulls)
        }
        F::Last { ignore_nulls } => valid || !ignore_nulls,
        other => return Err(internal(format!("{other:?} of strings by row"))),
    })
}

/// Bytes an array of `rows` values of `t` takes (validity included).
pub(super) fn width(t: &DataType, rows: usize) -> usize {
    let value = t.primitive_width().unwrap_or(16);
    rows * value + rows.div_ceil(8)
}

/// The bytes the evaluation of `plan` over complete partitions holds for
/// `rows` rows in `parts` partitions and `peers` peer groups: the row
/// maps, each function's states and output, and the index vectors that
/// gather outputs. Linear in its arguments, so the cost of a unit is the
/// sum of its batches' costs. `held` is [`snapshot_slots`] of the batch.
pub(super) fn cost(
    plan: &WindowPlan,
    rows: usize,
    parts: usize,
    peers: usize,
    held: usize,
) -> Result<usize> {
    // `part_of`, `peer_of`; `parts` and `first_peer`; `peers`.
    let mut total = rows * 8 + parts * 20 + peers * 16;
    if has_range_offsets(plan) {
        total += rows * super::frames::RANGE_KEY_BYTES;
    }
    for f in &plan.functions {
        let (out, _) = f.output()?;
        total += match &f.function {
            WindowFunction::RowNumber
            | WindowFunction::Rank
            | WindowFunction::DenseRank
            | WindowFunction::Ntile(_)
            | WindowFunction::CumeDist => width(&out, rows),
            // The ranks, then the result.
            WindowFunction::PercentRank => rows * 4 + width(&out, rows),
            WindowFunction::Offset { .. } | WindowFunction::NthValue { .. } => {
                super::frames::offset_cost(f, rows)?
            }
            WindowFunction::Aggregate(a) if f.shape()? == Shape::Frame => {
                super::frames::cost(a, f, rows)?
            }
            WindowFunction::Aggregate(a) => {
                let shape = f.shape()?;
                let segments = match shape {
                    Shape::Whole => parts,
                    Shape::RunningRows | Shape::Frame => rows,
                    Shape::RunningPeers => peers,
                };
                // The gathered output and its indices.
                let gathered = width(&out, rows) + rows * 4;
                if exact_decimal(a) {
                    // States, the segment of each row, the values.
                    segments * size_of::<(i256, i64)>()
                        + rows * 4
                        + segments * size_of::<Option<i128>>()
                        + width(&out, segments)
                        + gathered
                } else if picks_bytes(a) {
                    // Views of the input, picked rows, gather indices.
                    width(&out, rows)
                        + segments * size_of::<(usize, usize)>()
                        + rows * size_of::<(usize, usize)>()
                        + width(&out, rows)
                } else {
                    // The window applies the filter itself (see
                    // `filter_mask`); the accumulator never sees it.
                    let local = AggregateExpr {
                        args: (0..a.args.len()).collect(),
                        input: AggInput::Raw,
                        mask: None,
                        ..a.clone()
                    };
                    let state = Accumulator::new(&local, true)?.group_bytes();
                    let groups = match shape {
                        Shape::Whole => parts,
                        _ => parts + segments,
                    };
                    // Group ids (a whole frame's, or a filtered running
                    // frame's: an unfiltered one folds into the partition
                    // groups as they are), snapshot slots (running frames),
                    // states, evaluated values, and (but for one segment per
                    // row) the gathered output.
                    let ids = if shape == Shape::Whole || f.filter.is_some() {
                        rows * 4
                    } else {
                        0
                    };
                    let slots = if shape == Shape::Whole { 0 } else { rows * 4 };
                    ids + slots
                        + snapshot_heap(a, shape, held)
                        + groups * state
                        + width(&out, segments)
                        + if shape == Shape::RunningRows {
                            0
                        } else {
                            gathered
                        }
                }
            }
        };
    }
    Ok(total)
}
