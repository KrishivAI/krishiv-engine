// SPDX-License-Identifier: Apache-2.0
//! Bitwise functions: the `&` `|` `^` `~` operators (Java wrapping, so
//! ANSI and legacy agree), the shifts (`<<` `>>` `>>>` and their function
//! spellings; the shift amount masks as Java's does), `bit_count` (the
//! population count of the sign-extended value, as `Long.bitCount`) and
//! `bit_get`/`getbit` (with `INVALID_PARAMETER_VALUE.BIT_POSITION_RANGE`
//! past the value's width).
//!
//! The analyzer casts the operands to one integral type (or refuses them),
//! so every binding sees one integral width (plus an `int` shift amount).

use std::sync::Arc;

use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType};
use arrow_schema::DataType;
use krishiv_common::error::{InvalidParameterValueBitPositionRange, KrishivError};

use super::args::Inputs;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "&",
    "|",
    "^",
    "~",
    "<<",
    ">>",
    ">>>",
    "shiftleft",
    "shiftright",
    "shiftrightunsigned",
    "bit_count",
    "bit_get",
    "getbit",
];

/// A null-preserving binary integral kernel.
fn binary<T, F>(l: &dyn Array, r: &dyn Array, f: F) -> Result<ArrayRef, KrishivError>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
    F: Fn(T::Native, T::Native) -> T::Native,
{
    use arrow_array::cast::AsArray;
    let (l, r) = (l.as_primitive::<T>(), r.as_primitive::<T>());
    let out: Vec<Option<T::Native>> = (0..l.len())
        .map(|i| (l.is_valid(i) && r.is_valid(i)).then(|| f(l.value(i), r.value(i))))
        .collect();
    Ok(Arc::new(arrow_array::PrimitiveArray::<T>::from_iter(out)))
}

/// A null-preserving unary integral kernel.
fn unary<T, F>(a: &dyn Array, f: F) -> Result<ArrayRef, KrishivError>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
    F: Fn(T::Native) -> T::Native,
{
    use arrow_array::cast::AsArray;
    let a = a.as_primitive::<T>();
    let out: Vec<Option<T::Native>> = (0..a.len())
        .map(|i| a.is_valid(i).then(|| f(a.value(i))))
        .collect();
    Ok(Arc::new(arrow_array::PrimitiveArray::<T>::from_iter(out)))
}

/// `&` `|` `^` of two columns of one integral type.
fn bitwise(t: &DataType, op: char) -> Option<Binding> {
    macro_rules! pair {
        ($t:ty) => {
            Binding::new(t.clone(), move |x: &Inputs<'_>| {
                let (l, r) = (x.column(0)?, x.column(1)?);
                binary::<$t, _>(
                    l.as_ref(),
                    r.as_ref(),
                    match op {
                        '&' => |a, b| a & b,
                        '|' => |a, b| a | b,
                        _ => |a, b| a ^ b,
                    },
                )
            })
        };
    }
    Some(match t {
        DataType::Int8 => pair!(Int8Type),
        DataType::Int16 => pair!(Int16Type),
        DataType::Int32 => pair!(Int32Type),
        DataType::Int64 => pair!(Int64Type),
        _ => return None,
    })
}

/// `~` of one column.
fn bitwise_not(t: &DataType) -> Option<Binding> {
    macro_rules! not {
        ($t:ty) => {
            Binding::new(t.clone(), move |x: &Inputs<'_>| {
                unary::<$t, _>(x.column(0)?.as_ref(), |a| !a)
            })
        };
    }
    Some(match t {
        DataType::Int8 => not!(Int8Type),
        DataType::Int16 => not!(Int16Type),
        DataType::Int32 => not!(Int32Type),
        DataType::Int64 => not!(Int64Type),
        _ => return None,
    })
}

/// A shift of an `int` or `bigint` column by an `int` column. Amounts mask
/// to 5 (int) or 6 (long) bits, as Java's `<<` `>>` `>>>` do.
fn shift(t: &DataType, op: char) -> Option<Binding> {
    Some(match (t, op) {
        (DataType::Int32, '<') => Binding::new(t.clone(), move |x: &Inputs<'_>| {
            shift32(x, |a, b| a.wrapping_shl((b & 31).cast_unsigned()))
        }),
        (DataType::Int64, '<') => Binding::new(t.clone(), move |x: &Inputs<'_>| {
            shift64(x, |a, b| a.wrapping_shl((b & 63).cast_unsigned()))
        }),
        (DataType::Int32, '>') => Binding::new(t.clone(), move |x: &Inputs<'_>| {
            shift32(x, |a, b| a.wrapping_shr((b & 31).cast_unsigned()))
        }),
        (DataType::Int64, '>') => Binding::new(t.clone(), move |x: &Inputs<'_>| {
            shift64(x, |a, b| a.wrapping_shr((b & 63).cast_unsigned()))
        }),
        (DataType::Int32, 'U') => Binding::new(t.clone(), move |x: &Inputs<'_>| {
            shift32(x, |a, b| {
                (a.cast_unsigned().wrapping_shr((b & 31).cast_unsigned())) as i32
            })
        }),
        (DataType::Int64, 'U') => Binding::new(t.clone(), move |x: &Inputs<'_>| {
            shift64(x, |a, b| {
                (a.cast_unsigned().wrapping_shr((b & 63).cast_unsigned())) as i64
            })
        }),
        _ => return None,
    })
}

/// An `int` shift by an `int` amount.
fn shift32(x: &Inputs<'_>, f: impl Fn(i32, i32) -> i32) -> Result<ArrayRef, KrishivError> {
    use arrow_array::cast::AsArray;
    let (l, r) = (
        x.column(0)?.as_primitive::<Int32Type>().clone(),
        x.column(1)?.as_primitive::<Int32Type>().clone(),
    );
    Ok(Arc::new(arrow_array::Int32Array::from(
        (0..l.len())
            .map(|i| (l.is_valid(i) && r.is_valid(i)).then(|| f(l.value(i), r.value(i))))
            .collect::<Vec<_>>(),
    )) as ArrayRef)
}

/// A `bigint` shift by an `int` amount.
fn shift64(x: &Inputs<'_>, f: impl Fn(i64, i32) -> i64) -> Result<ArrayRef, KrishivError> {
    use arrow_array::cast::AsArray;
    let (l, r) = (
        x.column(0)?.as_primitive::<Int64Type>().clone(),
        x.column(1)?.as_primitive::<Int32Type>().clone(),
    );
    Ok(Arc::new(arrow_array::Int64Array::from(
        (0..l.len())
            .map(|i| (l.is_valid(i) && r.is_valid(i)).then(|| f(l.value(i), r.value(i))))
            .collect::<Vec<_>>(),
    )) as ArrayRef)
}

/// `bit_count(x)`: the population count of the sign-extended value (as
/// `Long.bitCount`), or 1/0 for a boolean.
fn bit_count(t: &DataType) -> Option<Binding> {
    use arrow_array::cast::AsArray;
    Some(match t {
        DataType::Int8 => Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
            let a = x.column(0)?.as_primitive::<Int8Type>().clone();
            Ok(Arc::new(arrow_array::Int32Array::from(
                (0..a.len())
                    .map(|i| {
                        a.is_valid(i)
                            .then(|| i64::from(a.value(i)).count_ones() as i32)
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef)
        }),
        DataType::Int16 => Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
            let a = x.column(0)?.as_primitive::<Int16Type>().clone();
            Ok(Arc::new(arrow_array::Int32Array::from(
                (0..a.len())
                    .map(|i| {
                        a.is_valid(i)
                            .then(|| i64::from(a.value(i)).count_ones() as i32)
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef)
        }),
        DataType::Int32 => Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
            let a = x.column(0)?.as_primitive::<Int32Type>().clone();
            Ok(Arc::new(arrow_array::Int32Array::from(
                (0..a.len())
                    .map(|i| {
                        a.is_valid(i)
                            .then(|| i64::from(a.value(i)).count_ones() as i32)
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef)
        }),
        DataType::Int64 => Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
            let a = x.column(0)?.as_primitive::<Int64Type>().clone();
            Ok(Arc::new(arrow_array::Int32Array::from(
                (0..a.len())
                    .map(|i| a.is_valid(i).then(|| a.value(i).count_ones() as i32))
                    .collect::<Vec<_>>(),
            )) as ArrayRef)
        }),
        DataType::Boolean => Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
            let a = x.booleans(0)?;
            Ok(Arc::new(arrow_array::Int32Array::from(
                (0..a.len())
                    .map(|i| a.is_valid(i).then(|| i32::from(a.value(i))))
                    .collect::<Vec<_>>(),
            )) as ArrayRef)
        }),
        _ => return None,
    })
}

/// `bit_get(x, pos)` / `getbit`: bit `pos` of `x` (positions run right to
/// left from zero). Positions outside the value's width raise
/// `INVALID_PARAMETER_VALUE.BIT_POSITION_RANGE`.
fn bit_get(name: &'static str, t: &DataType) -> Option<Binding> {
    use arrow_array::cast::AsArray;
    let width: u32 = match t {
        DataType::Int8 => 8,
        DataType::Int16 => 16,
        DataType::Int32 => 32,
        DataType::Int64 => 64,
        _ => return None,
    };
    let error = move |pos: i32| {
        KrishivError::from(InvalidParameterValueBitPositionRange {
            parameter: "`pos`".to_owned(),
            function_name: format!("`{name}`"),
            upper: width.to_string(),
            invalid_value: pos.to_string(),
        })
    };
    macro_rules! get {
        ($t:ty) => {
            Binding::new(DataType::Int8, move |x: &Inputs<'_>| {
                let (target, pos) = (x.column(0)?, x.column(1)?);
                let (target, pos) = (
                    target.as_primitive::<$t>().clone(),
                    pos.as_primitive::<Int32Type>().clone(),
                );
                let mut out = Vec::with_capacity(target.len());
                for i in 0..target.len() {
                    if !target.is_valid(i) || !pos.is_valid(i) {
                        out.push(None);
                        continue;
                    }
                    let p = pos.value(i);
                    if p < 0 || p.cast_unsigned() >= width {
                        return Err(error(p));
                    }
                    out.push(Some(((i64::from(target.value(i)) >> p) & 1) as i8));
                }
                Ok(Arc::new(arrow_array::Int8Array::from(out)) as ArrayRef)
            })
        };
    }
    Some(match t {
        DataType::Int8 => get!(Int8Type),
        DataType::Int16 => get!(Int16Type),
        DataType::Int32 => get!(Int32Type),
        DataType::Int64 => get!(Int64Type),
        _ => return None,
    })
}

/// Binds the bitwise function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let _ = opts;
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("&", [l, r]) if l == r => ok(bitwise(l, '&'), listed, args),
        ("|", [l, r]) if l == r => ok(bitwise(l, '|'), listed, args),
        ("^", [l, r]) if l == r => ok(bitwise(l, '^'), listed, args),
        ("~", [t]) => ok(bitwise_not(t), listed, args),
        ("<<" | "shiftleft", [t, DataType::Int32]) => ok(shift(t, '<'), listed, args),
        (">>" | "shiftright", [t, DataType::Int32]) => ok(shift(t, '>'), listed, args),
        (">>>" | "shiftrightunsigned", [t, DataType::Int32]) => ok(shift(t, 'U'), listed, args),
        ("bit_count", [t]) => match bit_count(t) {
            Some(b) => Ok(b),
            None => return Some((listed, Err(bad_types(listed, args)))),
        },
        ("bit_get" | "getbit", [t, DataType::Int32]) => match bit_get(listed, t) {
            Some(b) => Ok(b),
            None => return Some((listed, Err(bad_types(listed, args)))),
        },
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, binding))
}

/// A missing binding as the family's unsupported-types error.
fn ok(binding: Option<Binding>, name: &str, args: &[BindArg]) -> Result<Binding, KrishivError> {
    binding.ok_or_else(|| bad_types(name, args))
}
