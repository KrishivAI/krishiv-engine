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
    Accumulator, AggInput, AggregateExpr, AggregateFunction, avg_decimal_wide, wide_sum_value,
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
}

fn id(i: usize) -> Result<u32> {
    u32::try_from(i).map_err(|_| too_large())
}

fn too_large() -> krishiv_common::error::KrishivError {
    KrishivUnsupportedFeature {
        feature: "window evaluation over more than 2^32 - 1 rows at once".to_owned(),
    }
    .into()
}

impl<'a> Unit<'a> {
    fn new(slices: &'a [Slice<'a>]) -> Result<Self> {
        let rows: usize = slices.iter().map(|s| s.len).sum();
        let mut u = Self {
            slices,
            rows,
            part_of: Vec::with_capacity(rows),
            parts: Vec::new(),
            peer_of: Vec::with_capacity(rows),
            peers: Vec::new(),
            first_peer: Vec::new(),
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
    let unit = Unit::new(slices)?;
    plan.functions
        .iter()
        .map(|f| function(f, &unit, ansi))
        .collect()
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
                .collect::<Result<_>>()?;
            Arc::new(Int32Array::from(v))
        }
        WindowFunction::Rank => Arc::new(Int32Array::from(ranks(u, int)?)),
        WindowFunction::DenseRank => {
            let v: Vec<i32> = (0..n)
                .map(|i| {
                    let first = u.first_peer[u.part_of[i] as usize];
                    int(i64::from(u.peer_of[i] - first) + 1)
                })
                .collect::<Result<_>>()?;
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
                .collect::<Result<_>>()?;
            Arc::new(Int32Array::from(v))
        }
        WindowFunction::Aggregate(a) => aggregate(a, f, u, ansi)?,
    })
}

/// `count` as a double (Spark divides counts as doubles).
fn count_f64(count: usize) -> f64 {
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
        .collect()
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
fn arguments(a: &AggregateExpr, s: &Slice<'_>) -> Result<Vec<ArrayRef>> {
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

fn filter_mask(f: &WindowFunctionPlan, s: &Slice<'_>) -> Result<Option<BooleanArray>> {
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
fn passes(mask: Option<&BooleanArray>, i: usize) -> bool {
    mask.is_none_or(|m| m.is_valid(i) && m.value(i))
}

fn aggregate(
    a: &AggregateExpr,
    f: &WindowFunctionPlan,
    u: &Unit<'_>,
    ansi: bool,
) -> Result<ArrayRef> {
    let shape = f.shape()?;
    let decimal = matches!(
        a.arg_type,
        DataType::Decimal64(..) | DataType::Decimal128(..)
    );
    if decimal && matches!(a.function, AggregateFunction::Sum | AggregateFunction::Avg) {
        return decimal_aggregate(a, f, shape, u, ansi);
    }
    // The argument columns are passed as they are, in order.
    let local = AggregateExpr {
        args: (0..a.args.len()).collect(),
        input: AggInput::Raw,
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
            let ids: Vec<u32> = (0..parts).map(id).collect::<Result<_>>()?;
            let values = acc.evaluate(&ids)?;
            take(&values, &u.part_of)
        }
        Shape::RunningRows | Shape::RunningPeers => {
            // Group `p` folds partition `p`; group `parts + 1 + k` keeps the
            // state after segment `k` (a row or a peer group).
            let segments = match shape {
                Shape::RunningRows => u.rows,
                _ => u.peers.len(),
            };
            acc.resize(parts + 1 + segments);
            let mut groups: Vec<u32> = Vec::new();
            let mut at = 0;
            let mut segment = 0;
            for s in u.slices {
                let mask = filter_mask(f, s)?;
                let args = arguments(a, s)?;
                let mut i = 0;
                while i < s.len {
                    let row = at + i;
                    // The rest of the segment within this slice.
                    let end = match shape {
                        Shape::RunningRows => row + 1,
                        _ => u.peers[u.peer_of[row] as usize].1,
                    };
                    let len = (end - row).min(s.len - i);
                    let p = u.part_of[row];
                    groups.clear();
                    groups.extend(
                        (i..i + len).map(|j| if passes(mask.as_ref(), j) { p } else { trash }),
                    );
                    let piece: Vec<ArrayRef> = args.iter().map(|c| c.slice(i, len)).collect();
                    acc.update(&groups, &piece, &one)?;
                    i += len;
                    if at + i == end {
                        acc.copy_group(p as usize, parts + 1 + segment)?;
                        segment += 1;
                    }
                }
                at += s.len;
            }
            let ids: Vec<u32> = (0..segments)
                .map(|k| id(parts + 1 + k))
                .collect::<Result<_>>()?;
            let values = acc.evaluate(&ids)?;
            let of: Vec<u32> = match shape {
                Shape::RunningRows => (0..u.rows).map(id).collect::<Result<_>>()?,
                _ => u.peer_of.clone(),
            };
            take(&values, &of)
        }
    }
}

/// `values[indices[i]]` for each row.
fn take(values: &ArrayRef, indices: &[u32]) -> Result<ArrayRef> {
    let idx = UInt32Array::from(indices.to_vec());
    arrow_select::take::take(values.as_ref(), &idx, None).map_err(|e| internal(e.to_string()))
}

/// The unscaled values of a decimal column, in `i128`.
fn decimals(col: &ArrayRef) -> Result<Vec<Option<i128>>> {
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
    let result = a.result_type()?;
    let (DataType::Decimal64(p, s) | DataType::Decimal128(p, s)) = result else {
        return Err(internal("decimal window aggregate type".to_owned()));
    };
    let (DataType::Decimal64(_, in_scale) | DataType::Decimal128(_, in_scale)) = a.arg_type else {
        return Err(internal("decimal window aggregate input".to_owned()));
    };
    let avg = a.function == AggregateFunction::Avg;
    // The state after each segment: (exact sum, rows summed).
    let mut states: Vec<(i256, i64)> = Vec::new();
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
                Shape::RunningRows => row + 1,
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
            Shape::RunningRows => row,
            Shape::RunningPeers => u.peer_of[row] as usize,
        };
        seg_of.push(id(k)?);
    }
    let mut out: Vec<Option<i128>> = Vec::with_capacity(states.len());
    for &(sum, count) in &states {
        out.push(if count == 0 {
            None
        } else if avg {
            match avg_decimal_wide(sum, in_scale, count, p, s) {
                (Some(v), _) => Some(v),
                (None, text) if ansi => {
                    return Err(crate::aggregate::out_of_range(text, p, s));
                }
                (None, _) => None,
            }
        } else {
            wide_sum_value(Some(sum), p, s, ansi)?
        });
    }
    let values: ArrayRef = match result {
        DataType::Decimal64(..) => {
            let v: Vec<Option<i64>> = out
                .into_iter()
                .map(|x| x.map(i64::try_from).transpose())
                .collect::<Result<_, _>>()
                .map_err(|_| internal("decimal64 window value out of range".to_owned()))?;
            Arc::new(PrimitiveArray::<Decimal64Type>::from(v).with_data_type(result))
        }
        _ => Arc::new(PrimitiveArray::<Decimal128Type>::from(out).with_data_type(result)),
    };
    take(&values, &seg_of)
}
