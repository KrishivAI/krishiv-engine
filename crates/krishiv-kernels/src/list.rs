// SPDX-License-Identifier: Apache-2.0
//! List kernels for the higher-order functions: `array` builds a list per
//! row, and `transform` / `filter` / `exists` / `forall` consume one (see
//! `krishiv-exec`'s `hof`, which evaluates the lambda bodies).
//!
//! Every kernel works on the [`ListArray`] directly, offsets and child
//! values, never one Rust value per row: the caller evaluates a lambda
//! body once over the flattened child values, and these kernels only
//! reshape. A NULL list stays NULL; an empty list stays empty (or is
//! false for `exists`, true for `forall`).
//!
//! Offsets and lengths cross the `i32`/`u32`/`usize` boundary with checked
//! conversions: a batch that does not fit is an engine error, never a
//! wrap.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, ListArray, NullArray};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};

use super::error::KernelError;

/// The offsets of `list`, as `usize` pairs. Offsets are non-negative by
/// construction (`OffsetBuffer`), so they always fit.
fn ranges(list: &ListArray) -> Vec<(usize, usize)> {
    use arrow_buffer::ArrowNativeType as _;
    let offsets = list.value_offsets();
    let mut out = Vec::with_capacity(list.len());
    for pair in offsets.windows(2) {
        out.push((pair[0].as_usize(), pair[1].as_usize()));
    }
    out
}

/// `array(a, b, ...)`: one list per row holding each argument's value
/// (`[a[r], b[r], ...]`), interleaved in one pass in row-major order. An
/// empty argument list is an empty list per row. The arguments share one
/// type (the analyzer casts them); `contains_null` is whether elements may
/// be null (the analyzer's, from the arguments' nullability).
///
/// # Errors
/// [`KernelError`] when the arguments' lengths disagree with `rows`, or a
/// length does not fit a 32-bit offset.
pub fn create(
    args: &[ArrayRef],
    rows: usize,
    contains_null: bool,
) -> Result<ListArray, KernelError> {
    for a in args {
        if a.len() != rows {
            return Err(KernelError::internal(format!(
                "array of {rows} rows over an argument of {} rows",
                a.len()
            )));
        }
    }
    let len = list_len(rows, args.len())?;
    if args.is_empty() {
        let zeros = usize::checked_add(rows, 1).ok_or_else(|| too_big(rows))?;
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0_i32; zeros]));
        let values = NullArray::new(0);
        return Ok(ListArray::new(
            Arc::new(Field::new("element", DataType::Null, contains_null)),
            offsets,
            Arc::new(values),
            None,
        ));
    }
    let refs: Vec<&dyn Array> = args.iter().map(AsRef::as_ref).collect();
    // Row `r` takes element `r` of each argument: `(argument, row)` index
    // pairs interleave the arguments in one pass, in row-major order.
    let mut indices: Vec<(usize, usize)> = Vec::with_capacity(len);
    for r in 0..rows {
        for c in 0..args.len() {
            indices.push((c, r));
        }
    }
    let values = arrow_select::interleave::interleave(&refs, &indices)
        .map_err(|e| KernelError::internal(e.to_string()))?;
    let per_row = i32::try_from(args.len()).map_err(|_| too_big(args.len()))?;
    let mut offsets = Vec::with_capacity(usize::checked_add(rows, 1).ok_or_else(|| too_big(rows))?);
    for r in 0..=i32::try_from(rows).map_err(|_| too_big(rows))? {
        offsets.push(r.checked_mul(per_row).ok_or_else(|| too_big(rows))?);
    }
    let field = Arc::new(Field::new(
        "element",
        values.data_type().clone(),
        contains_null,
    ));
    Ok(ListArray::new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values,
        None,
    ))
}

/// What does not fit a 32-bit offset or index.
fn too_big(n: usize) -> KernelError {
    KernelError::internal(format!("a list of {n} rows"))
}

/// `rows` lists of `per_row` elements: their interleaved length.
///
/// # Errors
/// [`KernelError`] when the length does not fit.
fn list_len(rows: usize, per_row: usize) -> Result<usize, KernelError> {
    rows.checked_mul(per_row.max(1))
        .ok_or_else(|| too_big(rows))
}

/// `transform(list, values)`: the lists of `list` over the new child
/// `values` (the body evaluated over the flattened elements). `values`
/// holds one entry per element of `list`; `contains_null` is the body's
/// nullability.
///
/// # Errors
/// [`KernelError`] when `values` is shorter than the elements.
pub fn transform_list(
    list: &ListArray,
    values: &ArrayRef,
    contains_null: bool,
) -> Result<ListArray, KernelError> {
    let (_, end) = ranges(list).last().copied().unwrap_or((0, 0));
    if values.len() < end {
        return Err(KernelError::internal(format!(
            "transform over {end} elements with {} values",
            values.len()
        )));
    }
    let values = if values.len() == end {
        Arc::clone(values)
    } else {
        values.slice(0, end)
    };
    let field = Arc::new(Field::new(
        "element",
        values.data_type().clone(),
        contains_null,
    ));
    Ok(ListArray::new(
        field,
        list.offsets().clone(),
        values,
        list.nulls().cloned(),
    ))
}

/// `filter(list, keep)`: the elements `keep` marks true (a NULL mark drops
/// its element, as Spark does). `keep` holds one entry per element of
/// `list`.
///
/// # Errors
/// [`KernelError`] when `keep` is shorter than the elements, or the kept
/// length does not fit a 32-bit offset.
pub fn filter_list(list: &ListArray, keep: &BooleanArray) -> Result<ListArray, KernelError> {
    let ranges = ranges(list);
    let (_, end) = ranges.last().copied().unwrap_or((0, 0));
    if keep.len() < end {
        return Err(KernelError::internal(format!(
            "filter over {end} elements with {} marks",
            keep.len()
        )));
    }
    let mut take: Vec<u32> = Vec::with_capacity(end);
    let mut offsets = Vec::with_capacity(
        usize::checked_add(list.len(), 1)
            .ok_or_else(|| KernelError::internal(format!("a filter over {} lists", list.len())))?,
    );
    offsets.push(0_i32);
    for (i, &(start, stop)) in ranges.iter().enumerate() {
        if list.is_null(i) {
            offsets.push(i32::try_from(take.len()).map_err(|_| too_big(take.len()))?);
            continue;
        }
        for e in start..stop {
            if keep.is_valid(e) && keep.value(e) {
                take.push(u32::try_from(e).map_err(|_| too_big(e))?);
            }
        }
        offsets.push(i32::try_from(take.len()).map_err(|_| too_big(take.len()))?);
    }
    let values = arrow_select::take::take(
        list.values().as_ref(),
        &arrow_array::UInt32Array::from(take),
        None,
    )
    .map_err(|e| KernelError::internal(e.to_string()))?;
    // The elements pass through unchanged: the field rides along.
    let field = Arc::clone(list.value_field());
    Ok(ListArray::new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values,
        list.nulls().cloned(),
    ))
}

/// Folds `keep` (one entry per element of `list`) to one boolean per list
/// with Spark's three-valued logic: `any` (`exists`) is true when some
/// mark is true, else NULL when some mark is NULL, else false; `all`
/// (`forall`) is false when some mark is false, else NULL when some mark
/// is NULL, else true. A NULL list stays NULL; an empty one is
/// `any = false`, `all = true`.
fn fold_flags(
    list: &ListArray,
    keep: &BooleanArray,
    any: bool,
) -> Result<BooleanArray, KernelError> {
    let ranges = ranges(list);
    let (_, end) = ranges.last().copied().unwrap_or((0, 0));
    if keep.len() < end {
        return Err(KernelError::internal(format!(
            "fold over {end} elements with {} marks",
            keep.len()
        )));
    }
    let mut out = BooleanArray::builder(list.len());
    for (i, &(start, stop)) in ranges.iter().enumerate() {
        if list.is_null(i) {
            out.append_null();
            continue;
        }
        let mut decided = false;
        let mut seen_null = false;
        for e in start..stop {
            if !keep.is_valid(e) {
                seen_null = true;
                continue;
            }
            if keep.value(e) == any {
                decided = true;
                break;
            }
        }
        if decided {
            out.append_value(any);
        } else if seen_null {
            out.append_null();
        } else {
            out.append_value(!any);
        }
    }
    Ok(out.finish())
}

/// `exists(list, keep)`: whether some element's mark is true.
///
/// # Errors
/// [`KernelError`] when `keep` is shorter than the elements.
pub fn exists_list(list: &ListArray, keep: &BooleanArray) -> Result<BooleanArray, KernelError> {
    fold_flags(list, keep, true)
}

/// `forall(list, keep)`: whether no element's mark is false.
///
/// # Errors
/// [`KernelError`] when `keep` is shorter than the elements.
pub fn forall_list(list: &ListArray, keep: &BooleanArray) -> Result<BooleanArray, KernelError> {
    fold_flags(list, keep, false)
}

#[cfg(test)]
mod tests;
