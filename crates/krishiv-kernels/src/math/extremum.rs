// SPDX-License-Identifier: Apache-2.0
//! `greatest` and `least`: NULL arguments are skipped, and the result is
//! NULL only when every argument is. Ties keep the earlier argument, which
//! matters for `-0.0` versus `0.0` (equal in Spark's order) and for floats
//! generally: NaN is the greatest value and equals itself.

use std::cmp::Ordering;

use arrow_array::types::ByteArrayType;
use arrow_array::{Array, ArrowPrimitiveType, BooleanArray, GenericByteArray, PrimitiveArray};
use arrow_buffer::{BooleanBuffer, NullBuffer};

use crate::error::KernelError;
use crate::float::SparkFloat;

/// `greatest` or `least`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Extremum {
    /// `greatest`
    Greatest,
    /// `least`
    Least,
}

impl Extremum {
    /// Whether `candidate` replaces `best`.
    fn replaces(self, candidate: Ordering) -> bool {
        match self {
            Self::Greatest => candidate == Ordering::Greater,
            Self::Least => candidate == Ordering::Less,
        }
    }
}

fn rows_of(lens: impl Iterator<Item = usize>) -> Result<usize, KernelError> {
    let mut lens = lens;
    let first = lens
        .next()
        .ok_or_else(|| KernelError::internal("greatest/least need an argument"))?;
    if lens.any(|l| l != first) {
        return Err(KernelError::internal(
            "greatest/least columns differ in length",
        ));
    }
    Ok(first)
}

/// `greatest` / `least` over primitive columns of one type, compared by `cmp`.
///
/// # Errors
/// `INTERNAL_ERROR` for no column or columns of different lengths.
pub fn extremum_primitive<T: ArrowPrimitiveType>(
    kind: Extremum,
    columns: &[&PrimitiveArray<T>],
    cmp: impl Fn(&T::Native, &T::Native) -> Ordering,
) -> Result<PrimitiveArray<T>, KernelError> {
    let rows = rows_of(columns.iter().map(Array::len))?;
    Ok((0..rows)
        .map(|i| {
            columns
                .iter()
                .filter(|c| c.is_valid(i))
                .map(|c| c.value(i))
                .reduce(|best, v| {
                    if kind.replaces(cmp(&v, &best)) {
                        v
                    } else {
                        best
                    }
                })
        })
        .collect::<PrimitiveArray<T>>()
        .with_data_type(
            columns
                .first()
                .map(|c| c.data_type().clone())
                .unwrap_or(T::DATA_TYPE),
        ))
}

/// Spark's order of floats: NaN is greatest and equals itself, `-0.0 == 0.0`.
fn float_order<F: SparkFloat>(a: &F, b: &F) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    }
}

/// `greatest` / `least` over `float` or `double` columns.
///
/// # Errors
/// As [`extremum_primitive`].
pub fn extremum_float<T>(
    kind: Extremum,
    columns: &[&PrimitiveArray<T>],
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType,
    T::Native: SparkFloat,
{
    extremum_primitive(kind, columns, float_order::<T::Native>)
}

/// `greatest` / `least` over string or binary columns (byte order).
///
/// # Errors
/// As [`extremum_primitive`].
pub fn extremum_bytes<T: ByteArrayType>(
    kind: Extremum,
    columns: &[&GenericByteArray<T>],
) -> Result<GenericByteArray<T>, KernelError>
where
    T::Native: AsRef<[u8]>,
{
    let rows = rows_of(columns.iter().map(Array::len))?;
    Ok((0..rows)
        .map(|i| {
            columns
                .iter()
                .filter(|c| c.is_valid(i))
                .map(|c| c.value(i))
                .reduce(|best, v| {
                    if kind.replaces(v.as_ref().cmp(best.as_ref())) {
                        v
                    } else {
                        best
                    }
                })
        })
        .collect())
}

/// `greatest` / `least` over `boolean` columns (`false < true`).
///
/// # Errors
/// As [`extremum_primitive`].
pub fn extremum_bool(
    kind: Extremum,
    columns: &[&BooleanArray],
) -> Result<BooleanArray, KernelError> {
    let rows = rows_of(columns.iter().map(Array::len))?;
    let picked: Vec<Option<bool>> = (0..rows)
        .map(|i| {
            columns
                .iter()
                .filter(|c| c.is_valid(i))
                .map(|c| c.value(i))
                .reduce(|best, v| if kind.replaces(v.cmp(&best)) { v } else { best })
        })
        .collect();
    let values = BooleanBuffer::from_iter(picked.iter().map(|v| v.unwrap_or(false)));
    let nulls = NullBuffer::from_iter(picked.iter().map(Option::is_some));
    Ok(BooleanArray::new(values, Some(nulls)))
}

/// `greatest` / `least` over struct or list columns of one type, compared
/// by Spark's row order ([`crate::order::cmp_rows`]: fields and elements
/// lexicographically, NULL first, NaN last). NULL arguments are skipped;
/// ties keep the earlier argument.
///
/// # Errors
/// As [`extremum_primitive`], and [`KernelError`] for a type the order
/// refuses (maps).
pub fn extremum_nested(
    kind: Extremum,
    columns: &[&dyn Array],
) -> Result<arrow_array::ArrayRef, KernelError> {
    let rows = rows_of(columns.iter().map(|c| c.len()))?;
    let mut picks: Vec<(usize, usize)> = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut best: Option<usize> = None;
        for (k, c) in columns.iter().enumerate() {
            if c.is_null(i) {
                continue;
            }
            best = match best {
                None => Some(k),
                Some(b) => {
                    let o = crate::order::cmp_rows(columns[k], i, columns[b], i)?;
                    Some(if kind.replaces(o) { k } else { b })
                }
            };
        }
        match best {
            Some(k) => picks.push((k, i)),
            // Every argument is NULL: the first column's NULL row.
            None => picks.push((0, i)),
        }
    }
    arrow_select::interleave::interleave(columns, &picks)
        .map_err(|e| KernelError::internal(e.to_string()))
}

#[cfg(test)]
mod nested_tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, Int32Array, StructArray};
    use arrow_schema::{DataType, Field, Fields};

    use super::{Extremum, extremum_nested};

    fn st(a: &[Option<i32>], valid: &[bool]) -> ArrayRef {
        let f = Fields::from(vec![Field::new("a", DataType::Int32, true)]);
        Arc::new(StructArray::new(
            f,
            vec![Arc::new(Int32Array::from(a.to_vec())) as ArrayRef],
            Some(valid.to_vec().into()),
        ))
    }

    #[test]
    fn picks_by_field_order_skipping_null_structs() {
        let l = st(&[Some(1), None, Some(5)], &[true, false, false]);
        let r = st(&[Some(2), Some(3), None], &[true, true, false]);
        let g = extremum_nested(Extremum::Greatest, &[l.as_ref(), r.as_ref()]).unwrap();
        let g = g.as_any().downcast_ref::<StructArray>().unwrap();
        let a = g.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(a.value(0), 2);
        assert_eq!(a.value(1), 3);
        assert!(g.is_null(2));
        let s = extremum_nested(Extremum::Least, &[l.as_ref(), r.as_ref()]).unwrap();
        let s = s.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(
            s.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            1
        );
    }
}
