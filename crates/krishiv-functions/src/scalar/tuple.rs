// SPDX-License-Identifier: Apache-2.0
//! The Tuple sketch scalars (Spark 4.2+): `tuple_sketch_estimate_*`, `tuple_sketch_theta_*`, `tuple_sketch_summary_*`
//! and the set operations `tuple_union_*`, `tuple_intersection_*`, `tuple_difference_*` with their `_theta_` forms
//! (a Theta sketch as the second operand), for `double` and `integer` summaries. Per row Spark parses the mode,
//! reads the first buffer, then the second; a bad buffer is `TUPLE_INVALID_INPUT_SKETCH_BUFFER`.

use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, Float64Array, Int64Array};
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;

use super::args::{Bin, Inputs, Prim};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};
use crate::sketch::theta;
use crate::sketch::tuple::{self as tuple, Either, Mode, Value};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "tuple_sketch_estimate_double",
    "tuple_sketch_estimate_integer",
    "tuple_sketch_theta_double",
    "tuple_sketch_theta_integer",
    "tuple_sketch_summary_double",
    "tuple_sketch_summary_integer",
    "tuple_union_double",
    "tuple_union_integer",
    "tuple_union_theta_double",
    "tuple_union_theta_integer",
    "tuple_intersection_double",
    "tuple_intersection_integer",
    "tuple_intersection_theta_double",
    "tuple_intersection_theta_integer",
    "tuple_difference_double",
    "tuple_difference_integer",
    "tuple_difference_theta_double",
    "tuple_difference_theta_integer",
];

#[derive(Clone, Copy)]
enum Op {
    Estimate,
    Theta,
    Summary,
    Union,
    Intersection,
    Difference,
}

fn row(b: &Bin, i: usize) -> Option<&[u8]> {
    match b {
        Bin::Array(a) => arrow_array::Array::is_valid(a, i).then(|| a.value(i)),
        Bin::Scalar(s) => s.as_deref(),
    }
}

/// One row of a set operation: the first buffer is a tuple sketch, the second a tuple or Theta sketch.
fn set_row<V: Value>(
    op: Op,
    theta_form: bool,
    first: &[u8],
    second: &[u8],
    lg_nom: i32,
    mode: Mode,
    function: &str,
) -> Result<Vec<u8>, KrishivError> {
    let a = tuple::from_bytes::<V>(first, function)?;
    let (tb, thb);
    let b = if theta_form {
        thb = theta::from_bytes(second, function)?;
        Either::Theta(&thb)
    } else {
        tb = tuple::from_bytes::<V>(second, function)?;
        Either::Tuple(&tb)
    };
    let r = match op {
        Op::Union => tuple::union::<V>(&a, b, lg_nom, mode, function)?,
        Op::Intersection => tuple::intersection::<V>(&a, b, mode, function)?,
        _ => tuple::difference::<V>(&a, b, function)?,
    };
    Ok(match op {
        Op::Union => tuple::to_bytes_at(&r, lg_nom),
        _ => tuple::to_bytes(&r),
    })
}

fn mode_at(
    m: &arrow_array::StringArray,
    i: usize,
    function: &str,
) -> Result<Option<Mode>, KrishivError> {
    if arrow_array::Array::is_valid(m, i) {
        Mode::parse(m.value(i), function).map(Some)
    } else {
        Ok(None)
    }
}

fn unary<V: Value + 'static>(op: Op, function: &'static str, integer: bool) -> Binding {
    let out = match (op, integer) {
        (Op::Summary, true) => DataType::Int64,
        _ => DataType::Float64,
    };
    Binding::new(out, move |x: &Inputs<'_>| {
        let b = x.bin(0)?;
        let modes = if matches!(op, Op::Summary) {
            Some(x.strings(1)?)
        } else {
            None
        };
        let mut floats: Vec<Option<f64>> = Vec::new();
        let mut longs: Vec<Option<i64>> = Vec::new();
        for i in 0..x.rows() {
            let mode = match &modes {
                Some(m) => mode_at(m, i, function)?,
                None => Some(Mode::Sum),
            };
            let (Some(bytes), Some(mode)) = (row(&b, i), mode) else {
                floats.push(None);
                longs.push(None);
                continue;
            };
            let s = tuple::from_bytes::<V>(bytes, function)?;
            match op {
                Op::Estimate => floats.push(Some(tuple::estimate(&s))),
                Op::Theta => floats.push(Some(tuple::theta_of(&s))),
                _ => {
                    if integer {
                        longs.push(Some(tuple::summary_integer(&s, mode)));
                    } else {
                        floats.push(Some(tuple::summary_double(&s, mode)));
                    }
                }
            }
        }
        Ok(if matches!(op, Op::Summary) && integer {
            Arc::new(Int64Array::from(longs)) as ArrayRef
        } else {
            Arc::new(Float64Array::from(floats)) as ArrayRef
        })
    })
}

fn binary_op<V: Value + 'static>(
    op: Op,
    theta_form: bool,
    function: &'static str,
    arity: usize,
) -> Binding {
    Binding::new(DataType::Binary, move |x: &Inputs<'_>| {
        let (a, b) = (x.bin(0)?, x.bin(1)?);
        let lg = if matches!(op, Op::Union) {
            Some(x.int(2)?)
        } else {
            None
        };
        let modes = match (op, arity) {
            (Op::Union, 4) => Some(x.strings(3)?),
            (Op::Intersection, 3) => Some(x.strings(2)?),
            _ => None,
        };
        let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(x.rows());
        for i in 0..x.rows() {
            let mode = match &modes {
                Some(m) => mode_at(m, i, function)?,
                None => Some(Mode::Sum),
            };
            let lg_nom = match &lg {
                Some(Prim::Array(v)) => arrow_array::Array::is_valid(v, i).then(|| v.value(i)),
                Some(Prim::Scalar(v)) => *v,
                None => Some(12),
            };
            let (Some(first), Some(second), Some(mode), Some(lg_nom)) =
                (row(&a, i), row(&b, i), mode, lg_nom)
            else {
                out.push(None);
                continue;
            };
            out.push(Some(set_row::<V>(
                op, theta_form, first, second, lg_nom, mode, function,
            )?));
        }
        Ok(Arc::new(BinaryArray::from_iter(out)) as ArrayRef)
    })
}

pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let all_binary_prefix =
        |n: usize| types.len() >= n && types[..n].iter().all(super::args::is_binary);
    let integer = listed.ends_with("_integer");
    let theta_form = listed.contains("_theta_") && !listed.starts_with("tuple_sketch");
    let op = if listed.starts_with("tuple_sketch_estimate") {
        Op::Estimate
    } else if listed.starts_with("tuple_sketch_theta") {
        Op::Theta
    } else if listed.starts_with("tuple_sketch_summary") {
        Op::Summary
    } else if listed.starts_with("tuple_union") {
        Op::Union
    } else if listed.starts_with("tuple_intersection") {
        Op::Intersection
    } else {
        Op::Difference
    };
    let fixed: &'static str = NAMES
        .iter()
        .find(|n| **n == listed)
        .copied()
        .unwrap_or("tuple");
    let n = types.len();
    let shape_ok = match op {
        Op::Estimate | Op::Theta => n == 1 && all_binary_prefix(1),
        Op::Summary => n == 2 && all_binary_prefix(1) && super::args::is_string(&types[1]),
        Op::Union => {
            n == 4
                && all_binary_prefix(2)
                && types[2] == DataType::Int32
                && super::args::is_string(&types[3])
        }
        Op::Intersection => n == 3 && all_binary_prefix(2) && super::args::is_string(&types[2]),
        Op::Difference => n == 2 && all_binary_prefix(2),
    };
    if !shape_ok {
        return Some((listed, Err(bad_types(listed, args))));
    }
    let binding = match (op, integer) {
        (Op::Estimate | Op::Theta | Op::Summary, false) => unary::<f64>(op, fixed, false),
        (Op::Estimate | Op::Theta | Op::Summary, true) => unary::<i32>(op, fixed, true),
        (_, false) => binary_op::<f64>(op, theta_form, fixed, n),
        (_, true) => binary_op::<i32>(op, theta_form, fixed, n),
    };
    Some((listed, Ok(binding)))
}
