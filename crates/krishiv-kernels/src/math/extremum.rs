// SPDX-License-Identifier: Apache-2.0
//! `greatest` and `least`: NULL arguments are skipped, and the result is
//! NULL only when every argument is. Ties keep the earlier argument, which
//! matters for `-0.0` versus `0.0` (equal in Spark's order) and for floats
//! generally: NaN is the greatest value and equals itself.

use std::cmp::Ordering;

use arrow_array::types::ByteArrayType;
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BooleanArray, GenericByteArray, PrimitiveArray,
};
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

/// `greatest` / `least` over struct or array columns of one type, compared in
/// Spark's order (field by field, element by element; a NULL field or element
/// is the smallest value). A NULL argument is skipped, so the result is NULL
/// only where every argument is; a tie keeps the earlier argument.
///
/// # Errors
/// As [`extremum_primitive`], and for a type the order does not cover.
pub fn extremum_nested(kind: Extremum, columns: &[ArrayRef]) -> Result<ArrayRef, KernelError> {
    let rows = rows_of(columns.iter().map(Array::len))?;
    let refs: Vec<&dyn Array> = columns.iter().map(AsRef::as_ref).collect();
    let mut picks = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut best: Option<usize> = None;
        for (c, col) in columns.iter().enumerate() {
            if col.is_null(row) {
                continue;
            }
            best = match best {
                None => Some(c),
                Some(b) => {
                    let ord =
                        crate::order::cmp_rows(columns[c].as_ref(), row, columns[b].as_ref(), row)?;
                    Some(if kind.replaces(ord) { c } else { b })
                }
            };
        }
        picks.push((best.unwrap_or(0), row));
    }
    arrow_select::interleave::interleave(&refs, &picks)
        .map_err(|e| KernelError::internal(e.to_string()))
}

#[cfg(test)]
mod nested_tests {
    use super::*;
    use arrow_array::builder::{Int32Builder, ListBuilder};
    use std::sync::Arc;

    fn lists(rows: &[Option<Vec<Option<i32>>>]) -> ArrayRef {
        let mut b = ListBuilder::new(Int32Builder::new());
        for r in rows {
            match r {
                Some(v) => {
                    for x in v {
                        b.values().append_option(*x);
                    }
                    b.append(true);
                }
                None => b.append(false),
            }
        }
        Arc::new(b.finish())
    }

    #[test]
    fn arrays_compare_element_wise_and_skip_nulls() {
        let a = lists(&[Some(vec![Some(1), Some(2)]), None, None]);
        let b = lists(&[Some(vec![Some(1), None]), Some(vec![Some(5)]), None]);
        let g = extremum_nested(Extremum::Greatest, &[Arc::clone(&a), Arc::clone(&b)]).unwrap();
        assert_eq!(
            g.as_ref(),
            lists(&[Some(vec![Some(1), Some(2)]), Some(vec![Some(5)]), None]).as_ref()
        );
        let l = extremum_nested(Extremum::Least, &[a, b]).unwrap();
        assert_eq!(
            l.as_ref(),
            lists(&[Some(vec![Some(1), None]), Some(vec![Some(5)]), None]).as_ref()
        );
    }
}
