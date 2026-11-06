// SPDX-License-Identifier: Apache-2.0
//! Session aggregate UDFs (ADR 0045, unit 3b) over window frames, on the
//! UDF's own accumulator ([`accumulator_of`]):
//! - a whole partition folds each partition into a group and evaluates it
//!   once;
//! - a running frame (`UNBOUNDED PRECEDING` to `CURRENT ROW`, by rows or
//!   peers) folds one row or peer group at a time into its partition's
//!   group and evaluates the group after each: `evaluate` does not consume
//!   the state, so no state is copied;
//! - any other frame folds each row's frame into a group of its own, in
//!   chunks of at most [`CHUNK`] frame rows (the cost is the sum of the frame
//!   lengths; `retract` is not used, so a UDF need not support it).
//!
//! The window applies the call's `FILTER` itself; the UDF's null handling
//! is its accumulator's. A partition too large for memory is read from disk
//! as for a built-in ([`stream`](super::stream), [`disk`](super::disk)):
//! the whole-partition and running folds go through [`fold_rows`], and each
//! frame read from disk folds into an accumulator of its own.

use arrow_array::{Array, ArrayRef, UInt32Array};
use arrow_schema::DataType;
use krishiv_types::batch::Weights;

use super::eval::{Unit, id, take, width};
use super::{Shape, WindowFunctionPlan};
use crate::aggregate::{AggInput, AggregateExpr, AggregateFunction, UdafEnv, accumulator_of};
use crate::error::{Result, internal};

/// The most frame rows one chunk of a sliding frame folds at once.
const CHUNK: usize = 1 << 16;

/// Whether `a` is an aggregate UDF call.
pub(super) const fn is_udf(a: &AggregateExpr) -> bool {
    matches!(a.function, AggregateFunction::Udf(_))
}

/// The bytes evaluating `f` over `rows` rows in `parts` partitions holds
/// besides the UDF's own state (which it reports only after folding): the
/// gathered arguments of a chunk, group ids, the values and the output.
pub(super) fn cost(
    f: &WindowFunctionPlan,
    rows: usize,
    parts: usize,
    out: &DataType,
) -> Result<usize> {
    let chunk = match f.shape()? {
        Shape::Frame => CHUNK.min(rows.saturating_mul(rows)) * 8,
        _ => 0,
    };
    Ok(rows * 8 + parts * 4 + chunk + 2 * width(out, rows))
}

/// `a` over the frames of `f` in `u`.
///
/// # Errors
/// As the UDF's accumulator (`FAILED_EXECUTE_UDF`,
/// `KRISHIV_UDF_NOT_AVAILABLE`).
pub(super) fn aggregate(
    a: &AggregateExpr,
    f: &WindowFunctionPlan,
    u: &Unit<'_>,
    ansi: bool,
    udafs: Option<&UdafEnv>,
) -> Result<ArrayRef> {
    // The argument columns are the unit's, in order; the window filters.
    let local = AggregateExpr {
        args: (0..a.args.len()).collect(),
        input: AggInput::Raw,
        mask: None,
        ..a.clone()
    };
    let (args, mask) = super::frames::unit_inputs(a, f, u.slices)?;
    let passes = |i: usize| mask.as_ref().is_none_or(|m| m.is_valid(i) && m.value(i));
    let one = Weights::Const(1);
    let parts = u.parts.len();
    match f.shape()? {
        Shape::Whole => {
            let mut acc = accumulator_of(&local, ansi, udafs)?;
            acc.resize(parts);
            // Rows the FILTER rejects never reach `update`: a UDF's
            // `update` sees only the rows it aggregates.
            if mask.is_some() {
                let kept: Vec<usize> = (0..u.rows).filter(|&i| passes(i)).collect();
                if !kept.is_empty() {
                    let groups: Vec<u32> = kept.iter().map(|&i| u.part_of[i]).collect();
                    acc.update(&groups, &gather(&args, &kept)?, &one)?;
                }
            } else {
                acc.update(&u.part_of, &args, &one)?;
            }
            let ids: Vec<u32> = (0..parts).map(id).collect::<Result<_>>()?;
            take(&acc.evaluate(&ids)?, &u.part_of)
        }
        shape @ (Shape::RunningRows | Shape::RunningPeers) => {
            let mut acc = accumulator_of(&local, ansi, udafs)?;
            acc.resize(parts);
            let mut values: Vec<ArrayRef> = Vec::new();
            // Each segment (a row, or a peer group) folds into its
            // partition's group, which is then evaluated.
            let segments: Vec<(usize, usize)> = match shape {
                Shape::RunningRows => (0..u.rows).map(|i| (i, i + 1)).collect(),
                _ => u.peers.clone(),
            };
            for &(start, end) in &segments {
                let kept: Vec<usize> = (start..end).filter(|&i| passes(i)).collect();
                if !kept.is_empty() {
                    let g = u.part_of[start];
                    let cols = gather(&args, &kept)?;
                    acc.update(&vec![g; kept.len()], &cols, &one)?;
                }
                values.push(acc.evaluate(&[u.part_of[start]])?);
            }
            let values = concat(&values)?;
            match shape {
                Shape::RunningRows => Ok(values),
                _ => take(&values, &u.peer_of),
            }
        }
        Shape::Frame => {
            let part = |i: usize| u.parts[u.part_of[i] as usize];
            let peers = |i: usize| u.peers[u.peer_of[i] as usize];
            let rows = super::frames::Rows {
                part: &part,
                peers: &peers,
                n: u.rows,
                order: u.order.as_ref(),
            };
            let frames = super::frames::frames(f.frame, &rows)?;
            sliding(&local, ansi, udafs, &args, &passes, &frames)
        }
    }
}

/// `a` over `frames` of the rows of `slices` (frames read back from disk):
/// a UDF's sliding frames here, a built-in's through
/// [`frames::aggregate`](super::frames::aggregate).
pub(super) fn frames_aggregate(
    env: &super::Env,
    a: &AggregateExpr,
    f: &WindowFunctionPlan,
    slices: &[super::eval::Slice<'_>],
    frames: &[(u32, u32)],
) -> Result<ArrayRef> {
    if !is_udf(a) {
        return super::frames::aggregate(a, f, slices, frames, env.ansi);
    }
    let local = AggregateExpr {
        args: (0..a.args.len()).collect(),
        input: AggInput::Raw,
        mask: None,
        ..a.clone()
    };
    let (args, mask) = super::frames::unit_inputs(a, f, slices)?;
    let passes = |i: usize| mask.as_ref().is_none_or(|m| m.is_valid(i) && m.value(i));
    sliding(&local, env.ansi, env.udafs.as_ref(), &args, &passes, frames)
}

/// Each row's frame folded into a group of its own, in chunks.
fn sliding(
    local: &AggregateExpr,
    ansi: bool,
    udafs: Option<&UdafEnv>,
    args: &[ArrayRef],
    passes: &dyn Fn(usize) -> bool,
    frames: &[(u32, u32)],
) -> Result<ArrayRef> {
    let one = Weights::Const(1);
    let mut values: Vec<ArrayRef> = Vec::new();
    let mut first = 0;
    while first < frames.len() {
        // Rows of the chunk, and the frame rows each folds.
        let mut members: Vec<usize> = Vec::new();
        let mut groups: Vec<u32> = Vec::new();
        let mut last = first;
        while last < frames.len() {
            let (start, end) = frames[last];
            let len = (end.saturating_sub(start)) as usize;
            if last > first && members.len() + len > CHUNK {
                break;
            }
            let g = id(last - first)?;
            for m in start as usize..end as usize {
                if passes(m) {
                    members.push(m);
                    groups.push(g);
                }
            }
            last += 1;
        }
        let mut acc = accumulator_of(local, ansi, udafs)?;
        acc.resize(last - first);
        if !members.is_empty() {
            acc.update(&groups, &gather(args, &members)?, &one)?;
        }
        let ids: Vec<u32> = (0..last - first).map(id).collect::<Result<_>>()?;
        values.push(acc.evaluate(&ids)?);
        first = last;
    }
    concat(&values)
}

/// Folds `rows` rows of `args` (those `mask` passes) into group 0 of
/// `acc`, in order, and evaluates the group after each row `i` with
/// `snap[i]`: a running frame read from disk.
pub(super) fn fold_rows(
    acc: &mut crate::aggregate::Accumulator,
    args: &[ArrayRef],
    mask: Option<&arrow_array::BooleanArray>,
    rows: usize,
    snap: &[bool],
) -> Result<ArrayRef> {
    let one = Weights::Const(1);
    let mut values: Vec<ArrayRef> = Vec::new();
    let mut pending: Vec<usize> = Vec::new();
    let flush = |acc: &mut crate::aggregate::Accumulator, pending: &mut Vec<usize>| -> Result<()> {
        if !pending.is_empty() {
            acc.update(&vec![0; pending.len()], &gather(args, pending)?, &one)?;
            pending.clear();
        }
        Ok(())
    };
    for (i, &s) in snap.iter().enumerate().take(rows) {
        if mask.is_none_or(|m| m.is_valid(i) && m.value(i)) {
            pending.push(i);
        }
        if s {
            flush(acc, &mut pending)?;
            values.push(acc.evaluate(&[0])?);
        }
    }
    flush(acc, &mut pending)?;
    if values.is_empty() {
        return acc.evaluate(&[]);
    }
    concat(&values)
}

/// `cols` at rows `at`, in order.
fn gather(cols: &[ArrayRef], at: &[usize]) -> Result<Vec<ArrayRef>> {
    let idx: UInt32Array = at
        .iter()
        .map(|&i| id(i))
        .collect::<Result<Vec<u32>>>()?
        .into();
    cols.iter()
        .map(|c| {
            arrow_select::take::take(c.as_ref(), &idx, None).map_err(|e| internal(e.to_string()))
        })
        .collect()
}

fn concat(values: &[ArrayRef]) -> Result<ArrayRef> {
    let refs: Vec<&dyn Array> = values.iter().map(AsRef::as_ref).collect();
    arrow_select::concat::concat(&refs).map_err(|e| internal(e.to_string()))
}
