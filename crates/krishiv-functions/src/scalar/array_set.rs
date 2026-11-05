// SPDX-License-Identifier: Apache-2.0
//! Array membership, set and order functions: `array_contains`,
//! `array_distinct`, `array_union`, `array_except`, `array_intersect`,
//! `arrays_overlap`, `sort_array` and `arrays_zip`.
//!
//! Elements compare as Spark's `RowOrdering` does
//! ([`krishiv_kernels::order::cmp_rows`]): NULL equals NULL and sorts first,
//! NaN equals NaN and sorts last, `-0.0` equals `0.0`. `array_distinct` and
//! `array_union` keep the first of each group of equal elements, in order,
//! and write every float `-0.0` as `0.0` (checked on a Spark 4.2.0 run);
//! `sort_array` is stable and keeps the values as they are. A miss in
//! `array_contains` is NULL when the array holds a NULL. `arrays_zip` is as
//! long as its longest array and NULL when any array is.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, ListArray, StructArray, UInt32Array, new_null_array,
};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Fields, TimeUnit};
use arrow_select::concat::concat;
use arrow_select::take::take;
use krishiv_common::error::KrishivError;
use krishiv_kernels::map as kh;
use krishiv_kernels::order::cmp_rows;

use super::args::{Inputs, internal, to_utf8};
use super::array_edit::{MAX_ARRAY_LENGTH, index, list_of, no_memory, rebuild, span, too_long};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "array_contains",
    "array_distinct",
    "array_except",
    "array_intersect",
    "array_union",
    "arrays_overlap",
    "arrays_zip",
    "sort_array",
];

/// Binds the function `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("array_contains", [DataType::List(f), _]) if orders(f.data_type()) => {
            Ok(Binding::new(DataType::Boolean, contains))
        }
        ("array_distinct", [t @ DataType::List(f)]) if orders(f.data_type()) => {
            let field = Arc::clone(f);
            Ok(Binding::new(t.clone(), move |x: &Inputs<'_>| {
                distinct(x, &field)
            }))
        }
        ("array_union", [DataType::List(l), DataType::List(r)])
            if l.data_type() == r.data_type() && orders(l.data_type()) =>
        {
            let field = Arc::new(Field::new(
                "element",
                l.data_type().clone(),
                l.is_nullable() || r.is_nullable(),
            ));
            let ty = DataType::List(Arc::clone(&field));
            Ok(Binding::new(ty, move |x: &Inputs<'_>| union(x, &field)))
        }
        ("array_except" | "array_intersect", [DataType::List(l), DataType::List(r)])
            if l.data_type() == r.data_type() && orders(l.data_type()) =>
        {
            let keep_matches = listed == "array_intersect";
            let nullable = if keep_matches {
                l.is_nullable() && r.is_nullable()
            } else {
                l.is_nullable()
            };
            let field = Arc::new(Field::new("element", l.data_type().clone(), nullable));
            let ty = DataType::List(Arc::clone(&field));
            Ok(Binding::new(ty, move |x: &Inputs<'_>| {
                filtered(x, &field, keep_matches)
            }))
        }
        ("arrays_overlap", [DataType::List(l), DataType::List(r)])
            if l.data_type() == r.data_type() && orders(l.data_type()) =>
        {
            Ok(Binding::new(DataType::Boolean, overlap))
        }
        ("sort_array", [t @ DataType::List(f), DataType::Boolean]) if orders(f.data_type()) => {
            let field = Arc::clone(f);
            Ok(Binding::new(t.clone(), move |x: &Inputs<'_>| {
                sort(x, &field)
            }))
        }
        ("arrays_zip", _) => match zip_binding(args, &types) {
            Some(b) => Ok(b),
            None => Err(bad_types(listed, args)),
        },
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}

/// Whether the kernel orders elements of type `t`.
pub(super) fn orders(t: &DataType) -> bool {
    matches!(t, DataType::Null | DataType::Time64(TimeUnit::Nanosecond)) || kh::supports_key(t)
}

/// Whether `t` holds a float somewhere.
fn has_float(t: &DataType) -> bool {
    match t {
        DataType::Float32 | DataType::Float64 => true,
        DataType::List(f) => has_float(f.data_type()),
        DataType::Struct(fs) => fs.iter().any(|f| has_float(f.data_type())),
        _ => false,
    }
}

/// `a` with every float `-0.0` written `0.0`, in nested lists and structs
/// too.
fn without_negative_zero(a: &ArrayRef) -> Result<ArrayRef, KrishivError> {
    if !has_float(a.data_type()) {
        return Ok(Arc::clone(a));
    }
    Ok(match a.data_type() {
        DataType::Float64 => Arc::new(
            a.as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(|v| if v == 0.0 { 0.0 } else { v }),
        ),
        DataType::Float32 => Arc::new(
            a.as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(|v| if v == 0.0 { 0.0 } else { v }),
        ),
        DataType::List(f) => {
            let list = a.as_list::<i32>();
            Arc::new(
                ListArray::try_new(
                    Arc::clone(f),
                    list.offsets().clone(),
                    without_negative_zero(list.values())?,
                    list.nulls().cloned(),
                )
                .map_err(|e| internal(e.to_string()))?,
            )
        }
        DataType::Struct(fields) => {
            let s = a.as_struct();
            let columns = s
                .columns()
                .iter()
                .map(without_negative_zero)
                .collect::<Result<Vec<_>, _>>()?;
            Arc::new(
                StructArray::try_new(fields.clone(), columns, s.nulls().cloned())
                    .map_err(|e| internal(e.to_string()))?,
            )
        }
        _ => Arc::clone(a),
    })
}

pub(super) fn kernel(e: impl std::fmt::Display) -> KrishivError {
    internal(e.to_string())
}

/// Whether `a[i]` equals `b[j]` (both non-null) as Spark compares them.
pub(super) fn same_value(
    a: &dyn Array,
    i: usize,
    b: &dyn Array,
    j: usize,
) -> Result<bool, KrishivError> {
    if a.data_type() == b.data_type() {
        Ok(cmp_rows(a, i, b, j).map_err(kernel)? == Ordering::Equal)
    } else {
        kh::keys_equal(a, i, b, j).map_err(kernel)
    }
}

/// `array_contains(arr, value)`.
fn contains(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "array_contains")?;
    let value = x.column(1)?;
    let values = list.values();
    let nulls = values.logical_nulls();
    let mut out = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) || value.is_null(row) {
            out.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let mut saw_null = false;
        let mut hit = false;
        for i in start..start + len {
            if nulls.as_ref().is_some_and(|n| n.is_null(i)) {
                saw_null = true;
            } else if same_value(values.as_ref(), i, value.as_ref(), row)? {
                hit = true;
                break;
            }
        }
        out.push(if hit {
            Some(true)
        } else if saw_null {
            None
        } else {
            Some(false)
        });
    }
    Ok(Arc::new(BooleanArray::from(out)))
}

/// The first of each group of equal elements among the `picks` (indexes
/// into `values`), in their order.
fn first_of_each(values: &dyn Array, picks: &[usize]) -> Result<Vec<usize>, KrishivError> {
    if *values.data_type() == DataType::Null {
        return Ok(picks.first().copied().into_iter().collect());
    }
    let mut failure = None;
    let mut order: Vec<usize> = (0..picks.len()).collect();
    order.sort_by(|&p, &q| {
        cmp_rows(values, picks[p], values, picks[q])
            .unwrap_or_else(|e| {
                failure.get_or_insert(e);
                Ordering::Equal
            })
            .then(p.cmp(&q))
    });
    let mut keep = Vec::new();
    let mut last: Option<usize> = None;
    for &p in &order {
        let same = match last {
            Some(l) => {
                cmp_rows(values, picks[l], values, picks[p]).map_err(kernel)? == Ordering::Equal
            }
            None => false,
        };
        if !same {
            keep.push(p);
        }
        last = Some(p);
    }
    if let Some(e) = failure {
        return Err(kernel(e));
    }
    keep.sort_unstable();
    Ok(keep.into_iter().map(|p| picks[p]).collect())
}

fn to_indexes(picks: &[usize]) -> Result<Vec<Option<u32>>, KrishivError> {
    picks.iter().map(|&i| index(i).map(Some)).collect()
}

/// `array_distinct(arr)`.
fn distinct(x: &Inputs<'_>, field: &Arc<Field>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "array_distinct")?;
    let values = list.values();
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) {
            rows.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let picks: Vec<usize> = (start..start + len).collect();
        rows.push(Some(to_indexes(&first_of_each(values.as_ref(), &picks)?)?));
    }
    let out = rebuild(field, values, &rows, "array_distinct", MAX_ARRAY_LENGTH)?;
    rezero(&out)
}

/// `out` (a list) with its floats' `-0.0` written `0.0`.
fn rezero(out: &ArrayRef) -> Result<ArrayRef, KrishivError> {
    without_negative_zero(out)
}

/// `array_union(a, b)`.
fn union(x: &Inputs<'_>, field: &Arc<Field>) -> Result<ArrayRef, KrishivError> {
    let (ca, cb) = (x.column(0)?, x.column(1)?);
    let (a, b) = (list_of(&ca, "array_union")?, list_of(&cb, "array_union")?);
    let values = concat(&[a.values().as_ref(), b.values().as_ref()]).map_err(kernel)?;
    let base = a.values().len();
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if a.is_null(row) || b.is_null(row) {
            rows.push(None);
            continue;
        }
        let (sa, la) = span(a, row);
        let (sb, lb) = span(b, row);
        let picks: Vec<usize> = (sa..sa + la)
            .chain((sb..sb + lb).map(|i| base + i))
            .collect();
        rows.push(Some(to_indexes(&first_of_each(values.as_ref(), &picks)?)?));
    }
    let out = rebuild(field, &values, &rows, "array_union", MAX_ARRAY_LENGTH)?;
    rezero(&out)
}

/// `candidates` (indexes into `values`) sorted as Spark orders elements,
/// so a membership test is a binary search.
fn sorted(values: &dyn Array, mut candidates: Vec<usize>) -> Result<Vec<usize>, KrishivError> {
    if *values.data_type() == DataType::Null {
        return Ok(candidates);
    }
    let mut failure = None;
    candidates.sort_by(|&p, &q| {
        cmp_rows(values, p, values, q).unwrap_or_else(|e| {
            failure.get_or_insert(e);
            Ordering::Equal
        })
    });
    match failure {
        Some(e) => Err(kernel(e)),
        None => Ok(candidates),
    }
}

/// Whether `needle` equals an element of `sorted_candidates` (NULL equals
/// NULL; every NULL is the same value).
fn member(
    values: &dyn Array,
    needle: usize,
    sorted_candidates: &[usize],
) -> Result<bool, KrishivError> {
    if *values.data_type() == DataType::Null {
        return Ok(!sorted_candidates.is_empty());
    }
    let mut failure = None;
    let found = sorted_candidates
        .binary_search_by(|&c| {
            cmp_rows(values, c, values, needle).unwrap_or_else(|e| {
                failure.get_or_insert(e);
                Ordering::Equal
            })
        })
        .is_ok();
    match failure {
        Some(e) => Err(kernel(e)),
        None => Ok(found),
    }
}

/// `array_intersect(a, b)` (`keep_matches`) and `array_except(a, b)`: the
/// distinct elements of `a` that are (or are not) in `b`, in `a`'s order.
fn filtered(
    x: &Inputs<'_>,
    field: &Arc<Field>,
    keep_matches: bool,
) -> Result<ArrayRef, KrishivError> {
    let name = if keep_matches {
        "array_intersect"
    } else {
        "array_except"
    };
    let (ca, cb) = (x.column(0)?, x.column(1)?);
    let (a, b) = (list_of(&ca, name)?, list_of(&cb, name)?);
    let values = concat(&[a.values().as_ref(), b.values().as_ref()]).map_err(kernel)?;
    let base = a.values().len();
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if a.is_null(row) || b.is_null(row) {
            rows.push(None);
            continue;
        }
        let (sa, la) = span(a, row);
        let (sb, lb) = span(b, row);
        let right = sorted(values.as_ref(), (sb..sb + lb).map(|i| base + i).collect())?;
        let mut picks = Vec::new();
        for i in sa..sa + la {
            if member(values.as_ref(), i, &right)? == keep_matches {
                picks.push(i);
            }
        }
        rows.push(Some(to_indexes(&first_of_each(values.as_ref(), &picks)?)?));
    }
    let out = rebuild(field, &values, &rows, name, MAX_ARRAY_LENGTH)?;
    rezero(&out)
}

/// `arrays_overlap(a, b)`: true on a shared non-NULL element; else NULL when
/// both arrays are non-empty and either holds a NULL; else false.
fn overlap(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let (ca, cb) = (x.column(0)?, x.column(1)?);
    let (a, b) = (
        list_of(&ca, "arrays_overlap")?,
        list_of(&cb, "arrays_overlap")?,
    );
    let values = concat(&[a.values().as_ref(), b.values().as_ref()]).map_err(kernel)?;
    let nulls = values.logical_nulls();
    let is_null = |i: usize| nulls.as_ref().is_some_and(|n| n.is_null(i));
    let base = a.values().len();
    let mut out = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if a.is_null(row) || b.is_null(row) {
            out.push(None);
            continue;
        }
        let (sa, la) = span(a, row);
        let (sb, lb) = span(b, row);
        let right: Vec<usize> = (sb..sb + lb)
            .map(|j| base + j)
            .filter(|&j| !is_null(j))
            .collect();
        let saw_null = (sa..sa + la).any(&is_null) || right.len() < lb;
        let right = sorted(values.as_ref(), right)?;
        let mut hit = false;
        for i in (sa..sa + la).filter(|&i| !is_null(i)) {
            if member(values.as_ref(), i, &right)? {
                hit = true;
                break;
            }
        }
        out.push(if hit {
            Some(true)
        } else if saw_null && la > 0 && lb > 0 {
            None
        } else {
            Some(false)
        });
    }
    Ok(Arc::new(BooleanArray::from(out)))
}

/// `sort_array(arr, ascending)`.
fn sort(x: &Inputs<'_>, field: &Arc<Field>) -> Result<ArrayRef, KrishivError> {
    let c = x.column(0)?;
    let list = list_of(&c, "sort_array")?;
    let ascending = x.booleans(1)?;
    let values = list.values();
    let mut rows = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if list.is_null(row) || ascending.is_null(row) {
            rows.push(None);
            continue;
        }
        let (start, len) = span(list, row);
        let mut picks: Vec<usize> = (start..start + len).collect();
        if *values.data_type() != DataType::Null {
            let up = ascending.value(row);
            let mut failure = None;
            picks.sort_by(|&p, &q| {
                let ord = cmp_rows(values.as_ref(), p, values.as_ref(), q).unwrap_or_else(|e| {
                    failure.get_or_insert(e);
                    Ordering::Equal
                });
                if up { ord } else { ord.reverse() }
            });
            if let Some(e) = failure {
                return Err(kernel(e));
            }
        }
        rows.push(Some(to_indexes(&picks)?));
    }
    rebuild(field, values, &rows, "sort_array", MAX_ARRAY_LENGTH)
}

/// `arrays_zip`: the arrays are the first half of `args` when the second
/// half are the field names the analyzer passes as string literals, else
/// all of them (fields named `0`, `1`, ...).
fn zip_binding(args: &[BindArg], types: &[DataType]) -> Option<Binding> {
    let is_array = |t: &DataType| matches!(t, DataType::List(_));
    let all_arrays = |ts: &[DataType]| ts.iter().all(is_array);
    let literal_names = || -> Option<Vec<String>> {
        let half = args.len() / 2;
        if args.is_empty() || !args.len().is_multiple_of(2) || !all_arrays(&types[..half]) {
            return None;
        }
        let mut names = Vec::with_capacity(half);
        for a in &args[half..] {
            let BindArg::Literal(l) = a else { return None };
            let s = to_utf8(l.as_ref()).ok()?;
            if s.is_null(0) {
                return None;
            }
            names.push(s.value(0).to_owned());
        }
        Some(names)
    };
    let names: Vec<String> = match literal_names() {
        Some(names) => names,
        None if all_arrays(types) => (0..types.len()).map(|i| i.to_string()).collect(),
        None => return None,
    };
    let mut fields = Vec::with_capacity(names.len());
    for (name, t) in names.iter().zip(types) {
        let DataType::List(f) = t else { return None };
        fields.push(Field::new(name, f.data_type().clone(), true));
    }
    let fields = Fields::from(fields);
    let element = Arc::new(Field::new(
        "element",
        DataType::Struct(fields.clone()),
        false,
    ));
    let ty = DataType::List(Arc::clone(&element));
    Some(Binding::new(ty, move |x: &Inputs<'_>| {
        zip(x, &fields, &element)
    }))
}

/// `arrays_zip(a, b, ...)` over the first `fields.len()` arguments.
fn zip(x: &Inputs<'_>, fields: &Fields, element: &Arc<Field>) -> Result<ArrayRef, KrishivError> {
    let columns: Vec<ArrayRef> = (0..fields.len())
        .map(|k| x.column(k))
        .collect::<Result<_, _>>()?;
    let lists: Vec<&ListArray> = columns
        .iter()
        .map(|c| list_of(c, "arrays_zip"))
        .collect::<Result<_, _>>()?;
    let mut picks: Vec<Vec<Option<u32>>> = vec![Vec::new(); lists.len()];
    let mut offsets: Vec<i32> = Vec::with_capacity(x.rows() + 1);
    offsets.push(0);
    let mut valid = Vec::with_capacity(x.rows());
    let mut total = 0_usize;
    for row in 0..x.rows() {
        let ok = lists.iter().all(|l| !l.is_null(row));
        valid.push(ok);
        if ok {
            let longest = lists.iter().map(|l| span(l, row).1).max().unwrap_or(0);
            total += longest;
            if total > MAX_ARRAY_LENGTH {
                return Err(too_long("arrays_zip", total));
            }
            for (k, l) in lists.iter().enumerate() {
                let (start, len) = span(l, row);
                picks[k]
                    .try_reserve(longest)
                    .map_err(|_| no_memory(longest))?;
                for i in 0..longest {
                    picks[k].push(if i < len {
                        Some(index(start + i)?)
                    } else {
                        None
                    });
                }
            }
        }
        offsets.push(i32::try_from(total).map_err(|_| internal("an array past 2^31".to_owned()))?);
    }
    let mut children = Vec::with_capacity(lists.len());
    for (l, p) in lists.iter().zip(picks) {
        let values = l.values();
        children.push(if *values.data_type() == DataType::Null {
            new_null_array(&DataType::Null, p.len())
        } else {
            take(values.as_ref(), &UInt32Array::from(p), None).map_err(kernel)?
        });
    }
    let zipped =
        StructArray::try_new_with_length(fields.clone(), children, None, total).map_err(kernel)?;
    let nulls = valid.iter().any(|v| !v).then(|| NullBuffer::from(valid));
    Ok(Arc::new(
        ListArray::try_new(
            Arc::clone(element),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(zipped),
            nulls,
        )
        .map_err(kernel)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(t: DataType) -> BindArg {
        BindArg::from(DataType::List(Arc::new(Field::new("element", t, true))))
    }

    /// A call without the analyzer's name literals names its fields by
    /// position.
    #[test]
    fn arrays_zip_without_names_uses_positions() {
        let args = [list(DataType::Int32), list(DataType::Int32)];
        let (_, binding) = bind("arrays_zip", &args, &BindOptions::new()).unwrap();
        let ty = binding.unwrap().return_type;
        let DataType::List(f) = ty else {
            panic!("{ty}")
        };
        let DataType::Struct(fields) = f.data_type() else {
            panic!("{f:?}")
        };
        let names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, ["0", "1"]);
    }

    /// Time of day orders (as `cmp_rows` does) and a map does not.
    #[test]
    fn time_orders_and_maps_do_not() {
        assert!(orders(&DataType::Time64(TimeUnit::Nanosecond)));
        assert!(!orders(&DataType::Duration(TimeUnit::Second)));
    }
}
