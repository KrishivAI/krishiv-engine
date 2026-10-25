// SPDX-License-Identifier: Apache-2.0
//! Casts with array, map and struct types, as Spark's `Cast` runs them:
//! element by element (arrays), key and value by key and value (maps) and
//! field by field, by position (structs), each through the ordinary cast of
//! its type, and to `STRING` as `[1, null]`, `{k -> v}` and `{1, null}`.
//!
//! A cast between types that differ only in field names or nullability
//! relabels the data. Values no row reads (under a NULL struct, or in the
//! range of a NULL array) are never cast, so they raise nothing.

use std::fmt::Write as _;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, ListArray, MapArray, OffsetSizeTrait, StringViewArray, StructArray,
    UInt64Array,
};
use arrow_buffer::{ArrowNativeType, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, FieldRef};

use super::cast::{CastParts, apply};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

/// Whether `t` is the carrier of an array, map or struct.
pub(super) fn is_nested(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::List(_) | ArrowType::LargeList(_) | ArrowType::Map(..) | ArrowType::Struct(_)
    )
}

/// `a` through `p`, where the source or the target is nested.
pub(super) fn cast(p: &CastParts, a: &ArrayRef) -> Result<ArrayRef> {
    if a.data_type() == &p.to {
        return Ok(Arc::clone(a));
    }
    let refused = || unsupported(&format!("CAST from {} to {}", a.data_type(), p.to));
    match (a.data_type(), &p.to) {
        (ArrowType::List(_), ArrowType::List(to)) => list::<i32>(p, a, to),
        (ArrowType::LargeList(_), ArrowType::List(to)) => list::<i64>(p, a, to),
        (ArrowType::Map(..), ArrowType::Map(to, sorted)) => map(p, a, to, *sorted),
        (ArrowType::Struct(_), ArrowType::Struct(to)) => {
            let s = a.as_struct();
            if s.num_columns() != to.len() {
                return Err(refused());
            }
            let children = s
                .columns()
                .iter()
                .zip(to.iter())
                .map(|(c, f)| apply(&p.child(f.data_type()), &masked(c, s.nulls())?))
                .collect::<Result<Vec<_>>>()?;
            let out = StructArray::try_new(to.clone(), children, s.nulls().cloned())
                .map_err(|e| internal(&e.to_string()))?;
            Ok(Arc::new(out))
        }
        (from, ArrowType::Utf8View | ArrowType::Utf8) if is_nested(from) => {
            let texts = texts(p, a)?;
            let out: ArrayRef = Arc::new(StringViewArray::from_iter(texts));
            if p.to == ArrowType::Utf8 {
                super::strings::from_view(&out)
            } else {
                Ok(out)
            }
        }
        _ => Err(refused()),
    }
}

/// `a` with its rows also NULL where `parent` is: the values a NULL parent
/// row holds are never read, so they must not be cast.
fn masked(a: &ArrayRef, parent: Option<&NullBuffer>) -> Result<ArrayRef> {
    let Some(parent) = parent.filter(|n| n.null_count() > 0) else {
        return Ok(Arc::clone(a));
    };
    let nulls = NullBuffer::union(a.nulls(), Some(parent));
    let data = a
        .to_data()
        .into_builder()
        .nulls(nulls)
        .build()
        .map_err(|e| internal(&e.to_string()))?;
    Ok(arrow_array::make_array(data))
}

/// The 32-bit offsets and values of a list or map whose rows read only
/// their own values: a NULL row with a non-empty range is dropped from the
/// values (they are taken again without it).
fn compact<O: OffsetSizeTrait>(
    offsets: &[O],
    nulls: Option<&NullBuffer>,
    values: &ArrayRef,
) -> Result<(OffsetBuffer<i32>, ArrayRef)> {
    let at = |i: usize| offsets.get(i).map_or(0, |o| o.as_usize());
    let rows = offsets.len().saturating_sub(1);
    let dirty = nulls.is_some_and(|n| (0..rows).any(|i| n.is_null(i) && at(i + 1) > at(i)));
    let too_big = || unsupported("an array batch with more than 2^31 values");
    if !dirty {
        let first = at(0);
        let narrow = offsets
            .iter()
            .map(|o| i32::try_from(o.as_usize() - first).map_err(|_| too_big()))
            .collect::<Result<Vec<_>>>()?;
        let len = at(rows) - first;
        return Ok((
            OffsetBuffer::new(ScalarBuffer::from(narrow)),
            values.slice(first, len),
        ));
    }
    let mut indices = Vec::new();
    let mut narrow = Vec::with_capacity(offsets.len());
    narrow.push(0_i32);
    for i in 0..rows {
        if nulls.is_none_or(|n| n.is_valid(i)) {
            indices.extend((at(i)..at(i + 1)).map(|v| v as u64));
        }
        narrow.push(i32::try_from(indices.len()).map_err(|_| too_big())?);
    }
    let taken = arrow_select::take::take(values.as_ref(), &UInt64Array::from(indices), None)
        .map_err(|e| internal(&e.to_string()))?;
    Ok((OffsetBuffer::new(ScalarBuffer::from(narrow)), taken))
}

fn list<O: OffsetSizeTrait>(p: &CastParts, a: &ArrayRef, to: &FieldRef) -> Result<ArrayRef> {
    let l = a.as_list::<O>();
    let (offsets, values) = compact(l.value_offsets(), l.nulls(), l.values())?;
    let values = apply(&p.child(to.data_type()), &values)?;
    let out = ListArray::try_new(Arc::clone(to), offsets, values, l.nulls().cloned())
        .map_err(|e| internal(&e.to_string()))?;
    Ok(Arc::new(out))
}

fn map(p: &CastParts, a: &ArrayRef, to: &FieldRef, sorted: bool) -> Result<ArrayRef> {
    let m = a.as_map();
    let entries: ArrayRef = Arc::new(m.entries().clone());
    let (offsets, entries) = compact(m.value_offsets(), m.nulls(), &entries)?;
    let entries = apply(&p.child(to.data_type()), &entries)?;
    let out = MapArray::try_new(
        Arc::clone(to),
        offsets,
        entries.as_struct().clone(),
        m.nulls().cloned(),
        sorted,
    )
    .map_err(|e| internal(&e.to_string()))?;
    Ok(Arc::new(out))
}

/// Each row of `a` as Spark's `CAST(.. AS STRING)` writes it; `None` for a
/// NULL row.
fn texts(p: &CastParts, a: &ArrayRef) -> Result<Vec<Option<String>>> {
    let rows = |offsets: Vec<(usize, usize)>, items: Vec<Option<String>>, open, close| {
        offsets
            .into_iter()
            .enumerate()
            .map(|(i, (from, to))| {
                a.is_valid(i).then(|| {
                    let mut s = String::from(open);
                    for (k, item) in items.get(from..to).unwrap_or_default().iter().enumerate() {
                        if k > 0 {
                            s.push_str(", ");
                        }
                        s.push_str(item.as_deref().unwrap_or("null"));
                    }
                    s.push_str(close);
                    s
                })
            })
            .collect()
    };
    let ranges = |offsets: Vec<usize>| -> Vec<(usize, usize)> {
        offsets.windows(2).map(|w| (w[0], w[1])).collect()
    };
    Ok(match a.data_type() {
        ArrowType::List(_) => {
            let l = a.as_list::<i32>();
            let items = texts(p, l.values())?;
            let offsets = l.value_offsets().iter().map(|o| o.as_usize()).collect();
            rows(ranges(offsets), items, "[", "]")
        }
        ArrowType::LargeList(_) => {
            let l = a.as_list::<i64>();
            let items = texts(p, l.values())?;
            let offsets = l.value_offsets().iter().map(|o| o.as_usize()).collect();
            rows(ranges(offsets), items, "[", "]")
        }
        ArrowType::Map(..) => {
            let m = a.as_map();
            let keys = texts(p, m.keys())?;
            let values = texts(p, m.values())?;
            let pairs = keys
                .into_iter()
                .zip(values)
                .map(|(k, v)| {
                    let mut s = k.unwrap_or_else(|| "null".to_owned());
                    let _ = write!(s, " -> {}", v.as_deref().unwrap_or("null"));
                    Some(s)
                })
                .collect();
            let offsets = m.value_offsets().iter().map(|o| o.as_usize()).collect();
            rows(ranges(offsets), pairs, "{", "}")
        }
        ArrowType::Struct(_) => {
            let s = a.as_struct();
            let columns = s
                .columns()
                .iter()
                .map(|c| texts(p, &masked(c, s.nulls())?))
                .collect::<Result<Vec<_>>>()?;
            let width = columns.len();
            let items = (0..s.len())
                .flat_map(|row| columns.iter().map(move |c| c.get(row).cloned().flatten()))
                .collect();
            let offsets = (0..=s.len()).map(|row| row * width).collect();
            rows(ranges(offsets), items, "{", "}")
        }
        _ => {
            let out = apply(&p.child(&ArrowType::Utf8View), a)?;
            out.as_string_view()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect()
        }
    })
}
