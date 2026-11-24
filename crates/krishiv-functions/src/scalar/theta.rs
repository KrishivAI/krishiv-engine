// SPDX-License-Identifier: Apache-2.0
//! `theta_sketch_estimate`, `theta_union`, `theta_intersection` and `theta_difference` (P3-T086), after Spark's
//! `thetasketchesExpressions.scala` (v4.3.0): a buffer that is no compact Theta sketch is
//! `THETA_INVALID_INPUT_SKETCH_BUFFER` (`..._FAMILY` for another sketch family); the union's size is checked per row
//! (`SKETCH_INVALID_LG_NOM_ENTRIES`) before either buffer is read.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, Int64Array};
use arrow_schema::DataType;

use super::args::{Bin, Inputs, Prim};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};
use crate::sketch::theta;

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "theta_sketch_estimate",
    "theta_union",
    "theta_intersection",
    "theta_difference",
];

/// Row `i` of a binary operand: a literal is read at every row without being repeated into a column.
fn row(b: &Bin, i: usize) -> Option<&[u8]> {
    match b {
        Bin::Array(a) => a.is_valid(i).then(|| a.value(i)),
        Bin::Scalar(s) => s.as_deref(),
    }
}

fn is_binary(t: &DataType) -> bool {
    super::args::is_binary(t)
}

pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("theta_sketch_estimate", [t]) if is_binary(t) => {
            Ok(Binding::new(DataType::Int64, |x: &Inputs<'_>| {
                let b = x.bin(0)?;
                let mut out = Vec::with_capacity(x.rows());
                for i in 0..x.rows() {
                    out.push(match row(&b, i) {
                        Some(bytes) => Some(theta::estimate(&theta::from_bytes(
                            bytes,
                            "theta_sketch_estimate",
                        )?)),
                        None => None,
                    });
                }
                Ok(Arc::new(Int64Array::from(out)) as ArrayRef)
            }))
        }
        ("theta_union", [a, b, DataType::Int32]) if is_binary(a) && is_binary(b) => {
            Ok(Binding::new(DataType::Binary, |x: &Inputs<'_>| {
                let (a, b) = (x.bin(0)?, x.bin(1)?);
                let lg = x.int(2)?;
                let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(x.rows());
                for i in 0..x.rows() {
                    let lg_nom = match &lg {
                        Prim::Array(v) => v.is_valid(i).then(|| v.value(i)),
                        Prim::Scalar(v) => *v,
                    };
                    let (Some(lg_nom), Some(first), Some(second)) =
                        (lg_nom, row(&a, i), row(&b, i))
                    else {
                        out.push(None);
                        continue;
                    };
                    // Spark checks the size, then reads the first buffer, then the second.
                    theta::check_lg_nom(lg_nom, "theta_union")?;
                    let first = theta::from_bytes(first, "theta_union")?;
                    let second = theta::from_bytes(second, "theta_union")?;
                    let u = theta::union(&first, &second, lg_nom, "theta_union")?;
                    out.push(Some(theta::to_bytes(&u)));
                }
                Ok(Arc::new(BinaryArray::from_iter(out)) as ArrayRef)
            }))
        }
        (f @ ("theta_intersection" | "theta_difference"), [a, b])
            if is_binary(a) && is_binary(b) =>
        {
            let intersect = f == "theta_intersection";
            Ok(Binding::new(DataType::Binary, move |x: &Inputs<'_>| {
                let (a, b) = (x.bin(0)?, x.bin(1)?);
                let function = if intersect {
                    "theta_intersection"
                } else {
                    "theta_difference"
                };
                let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(x.rows());
                for i in 0..x.rows() {
                    let (Some(first), Some(second)) = (row(&a, i), row(&b, i)) else {
                        out.push(None);
                        continue;
                    };
                    let first = theta::from_bytes(first, function)?;
                    let second = theta::from_bytes(second, function)?;
                    let r = if intersect {
                        theta::intersection(&first, &second, function)?
                    } else {
                        theta::difference(&first, &second, function)?
                    };
                    out.push(Some(theta::to_bytes(&r)));
                }
                Ok(Arc::new(BinaryArray::from_iter(out)) as ArrayRef)
            }))
        }
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}
