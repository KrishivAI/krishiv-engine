// SPDX-License-Identifier: Apache-2.0
//! Aggregates over frames that do not start at the partition's first row
//! or end at the current row or peer group: `ROWS` frames with offsets
//! (`n PRECEDING`, `m FOLLOWING`) and frames ending at the partition's
//! last row (`CURRENT ROW AND UNBOUNDED FOLLOWING`, P2-T055 part 2).
//!
//! Each row's frame is a range of the unit's rows, empty when its start is
//! not below its end. The value of a frame is the aggregate of its rows
//! folded in order, as Spark's frames fold them (its sliding frame
//! re-folds the rows from the frame's start whenever the frame moves), so
//! the method depends on the aggregate:
//! - **Segment tree** (Leis et al., VLDB 2015, `research/papers.md` A10,
//!   §4.3.4) for the aggregates whose partial states merge to the same
//!   value in any grouping: `count`, `min`, `max`, `first`, `last`, and an
//!   integer `sum` that wraps (not ANSI). Leaves are rows, each inner node
//!   the merge of its two children (left, then right), and a frame merges
//!   O(log n) nodes left to right. Through the aggregate's own
//!   [`Accumulator`] states, so values, NULLs and orderings are its.
//! - **Removable cumulative** (§4.3.3) for a decimal `sum` or `avg`, exact
//!   in `i256`: a frame's sum and count are differences of prefix sums.
//! - **Strings and binaries** (`min`/`max`/`first`/`last`) pick a row: a
//!   segment tree of row indices compared by bytes, or the next or previous
//!   row that counts; values are gathered as views, never copied.
//! - **In order** for a floating-point `sum` or `avg`, or an integer `sum`
//!   under ANSI: rounding and overflow depend on the order and grouping of
//!   the additions, so each frame folds its rows from its start, as
//!   Spark's does (O(rows × frame) like Spark's).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, UInt32Array};
use arrow_buffer::i256;
use arrow_schema::DataType;
use krishiv_types::batch::Weights;

use super::eval::{
    Slice, arguments, decimal_array, decimal_value, decimals, exact_decimal, filter_mask, id,
    passes, picks_bytes, replaces, to_view, view_type, width,
};
use super::{FrameBound, FrameUnits, WindowFrame, WindowFunction, WindowFunctionPlan};
use crate::aggregate::{Accumulator, AggInput, AggregateExpr, AggregateFunction};
use crate::error::{Result, internal};

/// Merges per call of a frame query, and folded rows per call of an
/// in-order fold: bounds the scratch of one step.
pub(super) const CHUNK: usize = 1 << 14;

/// The rows of one partition unit as the frame computation needs them.
pub(super) struct Rows<'a> {
    /// Each row's partition, `[start, end)`.
    pub(super) part: &'a dyn Fn(usize) -> (usize, usize),
    /// Each row's peer group, `[start, end)`.
    pub(super) peers: &'a dyn Fn(usize) -> (usize, usize),
    pub(super) n: usize,
}

/// Each row's frame, `[start, end)` in unit rows (empty when `start >=
/// end`).
pub(super) fn frames(frame: WindowFrame, rows: &Rows<'_>) -> Result<Vec<(u32, u32)>> {
    let mut out = Vec::with_capacity(rows.n);
    for i in 0..rows.n {
        let (ps, pe) = (rows.part)(i);
        let at = |k: i64| -> usize {
            let i = i64::try_from(i).unwrap_or(i64::MAX);
            let ps = i64::try_from(ps).unwrap_or(0);
            let pe = i64::try_from(pe).unwrap_or(i64::MAX);
            usize::try_from(i.saturating_add(k).clamp(ps, pe)).unwrap_or(0)
        };
        let peers = || (rows.peers)(i);
        let start = match (frame.units, frame.lower) {
            (_, FrameBound::UnboundedPreceding) => ps,
            (FrameUnits::Rows, FrameBound::CurrentRow) => i,
            (FrameUnits::Range, FrameBound::CurrentRow) => peers().0,
            (FrameUnits::Rows, FrameBound::Offset(k)) => at(k),
            (units, lower) => {
                return Err(internal(format!("a {units:?} frame starting at {lower:?}")));
            }
        };
        let end = match (frame.units, frame.upper) {
            (_, FrameBound::UnboundedFollowing) => pe,
            (FrameUnits::Rows, FrameBound::CurrentRow) => i + 1,
            (FrameUnits::Range, FrameBound::CurrentRow) => peers().1,
            (FrameUnits::Rows, FrameBound::Offset(k)) => at(k.saturating_add(1)),
            (units, upper) => {
                return Err(internal(format!("a {units:?} frame ending at {upper:?}")));
            }
        };
        out.push((id(start)?, id(end.max(start))?));
    }
    Ok(out)
}

/// The unit's values of `make` over each slice, as one column.
fn concat(slices: &[Slice<'_>], make: impl Fn(&Slice<'_>) -> Result<ArrayRef>) -> Result<ArrayRef> {
    let parts: Vec<ArrayRef> = slices.iter().map(make).collect::<Result<_>>()?;
    let refs: Vec<&dyn Array> = parts.iter().map(AsRef::as_ref).collect();
    arrow_select::concat::concat(&refs).map_err(|e| internal(e.to_string()))
}

/// The unit's argument columns of `a` and its filter mask.
fn unit_inputs(
    a: &AggregateExpr,
    f: &WindowFunctionPlan,
    slices: &[Slice<'_>],
) -> Result<(Vec<ArrayRef>, Option<BooleanArray>)> {
    let mut args = Vec::with_capacity(a.args.len());
    for k in 0..a.args.len() {
        args.push(concat(slices, |s| {
            let col = arguments(a, s)?
                .into_iter()
                .nth(k)
                .ok_or_else(|| internal("a window argument missing".to_owned()))?;
            // Strings and binaries as views: concatenating them shares
            // their buffers.
            match col.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
                    to_view(&col)
                }
                _ => Ok(col),
            }
        })?);
    }
    let mask = match f.filter {
        Some(_) => Some(
            concat(slices, |s| {
                filter_mask(f, s)?
                    .map(|m| Arc::new(m) as ArrayRef)
                    .ok_or_else(|| internal("a window filter missing".to_owned()))
            })?
            .as_boolean()
            .clone(),
        ),
        None => None,
    };
    Ok((args, mask))
}

/// How the aggregate `a` is evaluated over frames (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Method {
    Tree,
    Decimal,
    Bytes,
    InOrder,
}

pub(super) fn method(a: &AggregateExpr, ansi: bool) -> Result<Method> {
    use AggregateFunction as F;
    Ok(if exact_decimal(a) {
        Method::Decimal
    } else if picks_bytes(a) {
        Method::Bytes
    } else {
        match a.function {
            F::Count | F::Min | F::Max | F::First { .. } | F::Last { .. } => Method::Tree,
            F::Sum if !ansi && a.result_type()? == DataType::Int64 => Method::Tree,
            _ => Method::InOrder,
        }
    })
}

/// The values of `a` (function `f`) over each row's frame.
pub(super) fn aggregate(
    a: &AggregateExpr,
    f: &WindowFunctionPlan,
    slices: &[Slice<'_>],
    frames: &[(u32, u32)],
    ansi: bool,
) -> Result<ArrayRef> {
    let method = method(a, ansi)?;
    if method == Method::Bytes {
        return bytes(&a.function, a, f, slices, frames);
    }
    let (args, mask) = unit_inputs(a, f, slices)?;
    match method {
        Method::Decimal => decimal(a, &args, mask.as_ref(), frames, ansi),
        Method::Bytes => Err(internal("strings over frames picked above".to_owned())),
        Method::Tree => tree(a, &args, mask.as_ref(), frames, ansi),
        Method::InOrder => in_order(a, &args, mask.as_ref(), frames, ansi),
    }
}

/// An accumulator of `a` over its argument columns as they are.
fn accumulator(a: &AggregateExpr, ansi: bool) -> Result<Accumulator> {
    let local = AggregateExpr {
        args: (0..a.args.len()).collect(),
        input: AggInput::Raw,
        ..a.clone()
    };
    Accumulator::new(&local, ansi)
}

/// The segment tree (see the module docs). Node `k` of `1..2n` is a group:
/// leaves `n..2n` are rows, inner node `k` merges `2k` then `2k + 1`; group
/// 0 takes rows the filter drops.
fn tree(
    a: &AggregateExpr,
    args: &[ArrayRef],
    mask: Option<&BooleanArray>,
    frames: &[(u32, u32)],
    ansi: bool,
) -> Result<ArrayRef> {
    let n = frames.len();
    let mut t = accumulator(a, ansi)?;
    t.resize(2 * n);
    let leaves: Vec<u32> = (0..n)
        .map(|i| if passes(mask, i) { id(n + i) } else { Ok(0) })
        .collect::<Result<_>>()?;
    if n > 0 {
        t.update(&leaves, args, &Weights::Const(1))?;
    }
    drop(leaves);
    // Parents in ranges whose children are all built: `[lo, hi)` with
    // `2 * lo >= hi`, merged a chunk of parents at a time so the copied
    // states stay within one chunk's scratch.
    let mut hi = n;
    while hi > 1 {
        let lo = hi.div_ceil(2).max(1);
        for start in (lo..hi).step_by(CHUNK) {
            let end = start.saturating_add(CHUNK).min(hi);
            let parents: Vec<u32> = (start..end).map(id).collect::<Result<_>>()?;
            for side in 0..2 {
                let children: Vec<u32> = (start..end)
                    .map(|k| id(2 * k + side))
                    .collect::<Result<_>>()?;
                let states = t.state(&children)?;
                t.merge(&parents, &states)?;
            }
        }
        hi = lo;
    }
    // Each frame's nodes, left to right, merged into its row's group.
    let mut out = accumulator(a, ansi)?;
    out.resize(n);
    let (mut groups, mut nodes) = (Vec::new(), Vec::new());
    let mut right = Vec::new();
    for (r, &(s, e)) in frames.iter().enumerate() {
        let (mut l, mut h) = (s as usize + n, e as usize + n);
        right.clear();
        while l < h {
            if l & 1 == 1 {
                nodes.push(id(l)?);
                groups.push(id(r)?);
                l += 1;
            }
            if h & 1 == 1 {
                h -= 1;
                right.push(id(h)?);
            }
            l /= 2;
            h /= 2;
        }
        for &k in right.iter().rev() {
            nodes.push(k);
            groups.push(id(r)?);
        }
        if nodes.len() >= CHUNK {
            out.merge(&groups, &t.state(&nodes)?)?;
            groups.clear();
            nodes.clear();
        }
    }
    if !nodes.is_empty() {
        out.merge(&groups, &t.state(&nodes)?)?;
    }
    let ids: Vec<u32> = (0..n).map(id).collect::<Result<_>>()?;
    out.evaluate(&ids)
}

/// Each frame's rows folded in order (see the module docs).
fn in_order(
    a: &AggregateExpr,
    args: &[ArrayRef],
    mask: Option<&BooleanArray>,
    frames: &[(u32, u32)],
    ansi: bool,
) -> Result<ArrayRef> {
    let n = frames.len();
    let mut out = accumulator(a, ansi)?;
    out.resize(n);
    let (mut groups, mut rows) = (Vec::new(), Vec::new());
    let mut flush = |groups: &mut Vec<u32>, rows: &mut Vec<u32>| -> Result<()> {
        if groups.is_empty() {
            return Ok(());
        }
        let idx = UInt32Array::from(std::mem::take(rows));
        let taken: Vec<ArrayRef> = args
            .iter()
            .map(|c| arrow_select::take::take(c.as_ref(), &idx, None))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| internal(e.to_string()))?;
        out.update(groups, &taken, &Weights::Const(1))?;
        groups.clear();
        Ok(())
    };
    for (r, &(s, e)) in frames.iter().enumerate() {
        for j in s..e {
            if passes(mask, j as usize) {
                groups.push(id(r)?);
                rows.push(j);
                if groups.len() >= CHUNK {
                    flush(&mut groups, &mut rows)?;
                }
            }
        }
    }
    flush(&mut groups, &mut rows)?;
    let ids: Vec<u32> = (0..n).map(id).collect::<Result<_>>()?;
    out.evaluate(&ids)
}

/// A decimal `sum` or `avg` by prefix sums (see the module docs).
fn decimal(
    a: &AggregateExpr,
    args: &[ArrayRef],
    mask: Option<&BooleanArray>,
    frames: &[(u32, u32)],
    ansi: bool,
) -> Result<ArrayRef> {
    let col = args
        .first()
        .ok_or_else(|| internal("decimal window aggregate without argument".to_owned()))?;
    let values = decimals(col)?;
    let mut prefix: Vec<(i256, i64)> = Vec::with_capacity(values.len() + 1);
    let (mut sum, mut count) = (i256::ZERO, 0_i64);
    prefix.push((sum, count));
    for (i, v) in values.iter().enumerate() {
        if let Some(v) = v
            && passes(mask, i)
        {
            sum = sum.wrapping_add(i256::from_i128(*v));
            count += 1;
        }
        prefix.push((sum, count));
    }
    let mut out = Vec::with_capacity(frames.len());
    for &(s, e) in frames {
        let (hi, lo) = (prefix[e as usize], prefix[s as usize]);
        out.push(decimal_value(
            a,
            hi.0.wrapping_sub(lo.0),
            hi.1 - lo.1,
            ansi,
        )?);
    }
    decimal_array(a, out)
}

/// `min`/`max`/`first`/`last` of strings or binaries over frames: the row
/// each frame picks, gathered as views of the input's buffers (see the
/// module docs).
fn bytes(
    function: &AggregateFunction,
    a: &AggregateExpr,
    f: &WindowFunctionPlan,
    slices: &[Slice<'_>],
    frames: &[(u32, u32)],
) -> Result<ArrayRef> {
    use AggregateFunction as F;
    let mut sources: Vec<ArrayRef> = Vec::with_capacity(slices.len() + 1);
    let mut masks = Vec::with_capacity(slices.len());
    // Each unit row's slice and row in it.
    let mut loc: Vec<(usize, usize)> = Vec::with_capacity(frames.len());
    for (k, s) in slices.iter().enumerate() {
        let col = arguments(a, s)?
            .into_iter()
            .next()
            .ok_or_else(|| internal("a window aggregate without argument".to_owned()))?;
        sources.push(to_view(&col)?);
        masks.push(filter_mask(f, s)?);
        loc.extend((0..s.len).map(|i| (k, i)));
    }
    let n = loc.len();
    let null_source = sources.len();
    sources.push(arrow_array::new_null_array(&view_type(&a.arg_type), 1));
    let value = |j: usize| -> &[u8] {
        let (k, i) = loc[j];
        let c = &sources[k];
        match c.data_type() {
            DataType::Utf8View => c.as_string_view().value(i).as_bytes(),
            _ => c.as_binary_view().value(i),
        }
    };
    // A row a frame may pick.
    let counts = |j: usize, skip_nulls: bool| {
        let (k, i) = loc[j];
        passes(masks[k].as_ref(), i) && (!skip_nulls || sources[k].is_valid(i))
    };
    let picked: Vec<Option<u32>> = match function {
        F::First { ignore_nulls } => {
            // The first row at or after each row that counts.
            let mut next = vec![u32::MAX; n + 1];
            for j in (0..n).rev() {
                next[j] = if counts(j, *ignore_nulls) {
                    id(j)?
                } else {
                    next[j + 1]
                };
            }
            frames
                .iter()
                .map(|&(s, e)| Some(next[s as usize]).filter(|&j| j < e))
                .collect()
        }
        F::Last { ignore_nulls } => {
            // `prev[j]`: the last row before `j` that counts.
            let mut prev = vec![None; n + 1];
            for j in 0..n {
                prev[j + 1] = if counts(j, *ignore_nulls) {
                    Some(id(j)?)
                } else {
                    prev[j]
                };
            }
            frames
                .iter()
                .map(|&(s, e)| prev[e as usize].filter(|&j| j >= s && s < e))
                .collect()
        }
        F::Min | F::Max => {
            // A segment tree of the row each node picks.
            let better = |x: Option<u32>, y: Option<u32>| -> Result<Option<u32>> {
                Ok(match (x, y) {
                    (Some(x), Some(y)) => {
                        let old = Some(Some(value(x as usize)));
                        if replaces(function, old, true, || value(y as usize))? {
                            Some(y)
                        } else {
                            Some(x)
                        }
                    }
                    (x, None) => x,
                    (None, y) => y,
                })
            };
            let mut t: Vec<Option<u32>> = vec![None; 2 * n];
            for j in 0..n {
                t[n + j] = counts(j, true).then(|| id(j)).transpose()?;
            }
            for k in (1..n).rev() {
                t[k] = better(t[2 * k], t[2 * k + 1])?;
            }
            let mut out = Vec::with_capacity(frames.len());
            for &(s, e) in frames {
                let (mut l, mut h) = (s as usize + n, e as usize + n);
                let (mut left, mut right) = (None, None);
                while l < h {
                    if l & 1 == 1 {
                        left = better(left, t[l])?;
                        l += 1;
                    }
                    if h & 1 == 1 {
                        h -= 1;
                        right = better(t[h], right)?;
                    }
                    l /= 2;
                    h /= 2;
                }
                out.push(better(left, right)?);
            }
            out
        }
        other => return Err(internal(format!("{other:?} of strings by row"))),
    };
    let indices: Vec<(usize, usize)> = picked
        .iter()
        .map(|p| p.map_or((null_source, 0), |j| loc[j as usize]))
        .collect();
    let refs: Vec<&dyn Array> = sources.iter().map(AsRef::as_ref).collect();
    arrow_select::interleave::interleave(&refs, &indices).map_err(|e| internal(e.to_string()))
}

/// The bytes evaluating `a` (function `f`) over the frames of `rows` rows
/// holds at most: the frames, the unit's argument columns (strings as
/// views), and the method's states and scratch (see [`CHUNK`]).
pub(super) fn cost(a: &AggregateExpr, f: &WindowFunctionPlan, rows: usize) -> Result<usize> {
    let (out, _) = f.output()?;
    let arg = |t: &DataType| width(t, rows).max(rows * 16 + rows.div_ceil(8));
    let inputs: usize = a.args.iter().map(|_| arg(&a.arg_type)).sum::<usize>() + rows.div_ceil(8);
    let frames = rows * 8;
    let method = if exact_decimal(a) {
        (rows + 1) * size_of::<(i256, i64)>() + rows * size_of::<Option<i128>>()
    } else if picks_bytes(a) {
        rows * (size_of::<(usize, usize)>() * 2 + 8 + 2 * size_of::<Option<u32>>())
    } else {
        let state = accumulator(a, true)?.group_bytes();
        // The tree's nodes and each row's result (an in-order fold needs
        // the results only), the leaf ids (then the result ids), and one
        // chunk of merges: a chunk of parents while the tree is built, of
        // frame nodes while it is queried, or of rows in order.
        3 * rows * state + rows * 4 + CHUNK * (8 + 2 * state.max(16))
    };
    Ok(frames + inputs + method + width(&out, rows))
}

/// The bytes `lag`/`lead`/`nth_value` (function `f`) hold at most over
/// `rows` rows: the input as views (strings and binaries are copied into
/// views; other types are slices of the input), the default column too for
/// `lag`/`lead`, the counting rows, the picks, the frames (`nth_value`),
/// the gather's locations and indices, and the result.
pub(super) fn offset_cost(f: &WindowFunctionPlan, rows: usize) -> Result<usize> {
    let (out, _) = f.output()?;
    let (data_type, with_default, frames) = match &f.function {
        WindowFunction::Offset {
            data_type, default, ..
        } => (data_type, default.is_some(), 0),
        WindowFunction::NthValue { data_type, .. } => (data_type, false, rows * 8),
        other => return Err(internal(format!("{other:?} is not an offset function"))),
    };
    let views = |t: &DataType| match t {
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => rows * 16 + rows.div_ceil(8),
        _ => 0,
    };
    let sources = views(data_type) * (1 + usize::from(with_default));
    let counting = (rows + 1) * 4 + rows * 4;
    let picks = rows * size_of::<(Option<u32>, Option<u32>)>();
    let gather = rows * size_of::<(usize, usize)>() * 2;
    Ok(sources + counting + picks + frames + gather + width(&out, rows))
}

/// Column `c` of each slice, strings and binaries as views (gathering
/// views shares the input's buffers).
fn sources(slices: &[Slice<'_>], c: usize) -> Result<Vec<ArrayRef>> {
    slices
        .iter()
        .map(|s| {
            let col = s
                .batch
                .columns()
                .get(c)
                .ok_or_else(|| internal(format!("window column {c} out of range")))?
                .slice(s.offset, s.len);
            match col.data_type() {
                DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView => to_view(&col),
                _ => Ok(col),
            }
        })
        .collect()
}

/// The rows of `input` that count (non-NULL with `ignore_nulls`): `before[i]`
/// of them precede unit row `i`, and `at[k]` is the `k`-th.
struct Counting {
    before: Vec<u32>,
    at: Vec<u32>,
}

fn counting(input: &[ArrayRef], n: usize) -> Result<Counting> {
    let mut before = Vec::with_capacity(n + 1);
    let mut at = Vec::new();
    let mut j = 0;
    for col in input {
        for i in 0..col.len() {
            before.push(id(at.len())?);
            if col.is_valid(i) {
                at.push(id(j)?);
            }
            j += 1;
        }
    }
    before.push(id(at.len())?);
    Ok(Counting { before, at })
}

/// Gathers the picked rows: `Some(j)` is unit row `j` of `input`, `None`
/// the fallback (`Some(i)`: unit row `i` of `fallback`; `None`: NULL).
fn gather(
    input: Vec<ArrayRef>,
    fallback: Option<Vec<ArrayRef>>,
    picks: &[(Option<u32>, Option<u32>)],
    data_type: &DataType,
) -> Result<ArrayRef> {
    let mut loc: Vec<(usize, usize)> = Vec::new();
    for (k, c) in input.iter().enumerate() {
        loc.extend((0..c.len()).map(|i| (k, i)));
    }
    let base = input.len();
    let mut all = input;
    if let Some(f) = fallback {
        all.extend(f);
    }
    let null_source = all.len();
    let view = match data_type {
        DataType::Utf8 | DataType::LargeUtf8 => DataType::Utf8View,
        DataType::Binary | DataType::LargeBinary => DataType::BinaryView,
        t => t.clone(),
    };
    all.push(arrow_array::new_null_array(&view, 1));
    let indices: Vec<(usize, usize)> = picks
        .iter()
        .map(|&(pick, fallback)| match (pick, fallback) {
            (Some(j), _) => loc[j as usize],
            (None, Some(i)) => {
                let (k, r) = loc[i as usize];
                (base + k, r)
            }
            (None, None) => (null_source, 0),
        })
        .collect();
    let refs: Vec<&dyn Array> = all.iter().map(AsRef::as_ref).collect();
    arrow_select::interleave::interleave(&refs, &indices).map_err(|e| internal(e.to_string()))
}

/// `lag`/`lead` (see [`WindowFunction::Offset`]).
pub(super) fn offset_value(
    f: &WindowFunctionPlan,
    slices: &[Slice<'_>],
    rows: &Rows<'_>,
) -> Result<ArrayRef> {
    let WindowFunction::Offset {
        input,
        offset,
        default,
        ignore_nulls,
        data_type,
    } = &f.function
    else {
        return Err(internal("lag/lead of another function".to_owned()));
    };
    let values = sources(slices, *input)?;
    let counts = counting(&values, rows.n)?;
    let mut picks = Vec::with_capacity(rows.n);
    for i in 0..rows.n {
        let (ps, pe) = (rows.part)(i);
        let row = i64::try_from(i).unwrap_or(i64::MAX);
        let inside = |j: i64| usize::try_from(j).ok().filter(|&j| j >= ps && j < pe);
        let target = if !*ignore_nulls || *offset == 0 {
            inside(row.saturating_add(*offset))
        } else {
            // The `|offset|`-th counting row after (or before) this one.
            let k = if *offset > 0 {
                i64::from(counts.before[i + 1]).saturating_add(*offset - 1)
            } else {
                i64::from(counts.before[i]).saturating_add(*offset)
            };
            usize::try_from(k)
                .ok()
                .and_then(|k| counts.at.get(k))
                .and_then(|&j| inside(i64::from(j)))
        };
        let fallback = default.map(|_| id(i)).transpose()?;
        picks.push((target.map(id).transpose()?, fallback));
    }
    let fallback = default.map(|d| sources(slices, d)).transpose()?;
    gather(values, fallback, &picks, data_type)
}

/// `nth_value` (see [`WindowFunction::NthValue`]) over each row's frame.
pub(super) fn nth_value(
    f: &WindowFunctionPlan,
    slices: &[Slice<'_>],
    rows: &Rows<'_>,
) -> Result<ArrayRef> {
    let WindowFunction::NthValue {
        input,
        n,
        ignore_nulls,
        data_type,
    } = &f.function
    else {
        return Err(internal("nth_value of another function".to_owned()));
    };
    let values = sources(slices, *input)?;
    let counts = counting(&values, rows.n)?;
    let mut picks = Vec::with_capacity(rows.n);
    for (s, e) in frames(f.frame, rows)? {
        let k = n.saturating_sub(1);
        let j = if *ignore_nulls {
            usize::try_from(i64::from(counts.before[s as usize]).saturating_add(k))
                .ok()
                .and_then(|k| counts.at.get(k).copied())
        } else {
            u32::try_from(i64::from(s).saturating_add(k)).ok()
        };
        picks.push((j.filter(|&j| j < e), None));
    }
    gather(values, None, &picks, data_type)
}
