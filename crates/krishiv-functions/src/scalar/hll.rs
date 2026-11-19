// SPDX-License-Identifier: Apache-2.0
//! `hll_sketch_estimate` and `hll_union` (P3-T086), after Spark's `HllSketchEstimate` and `HllUnion`
//! (`datasketchesExpressions.scala`, v4.2.0): a buffer that is no HLL sketch is
//! `HLL_INVALID_INPUT_SKETCH_BUFFER`; `hll_union` raises `HLL_UNION_DIFFERENT_LG_K` unless its flag is true.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, Int64Array};
use arrow_schema::DataType;

use super::args::Inputs;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};
use crate::sketch::hll;

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["hll_sketch_estimate", "hll_union"];

fn is_binary(t: &DataType) -> bool {
    super::args::is_binary(t)
}

pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("hll_sketch_estimate", [t]) if is_binary(t) => {
            Ok(Binding::new(DataType::Int64, |x: &Inputs<'_>| {
                let b = x.binaries(0)?;
                let mut out = Vec::with_capacity(b.len());
                for row in b.iter() {
                    out.push(match row {
                        Some(bytes) => Some(hll::estimate(&hll::from_bytes(
                            bytes,
                            "hll_sketch_estimate",
                        )?)),
                        None => None,
                    });
                }
                Ok(Arc::new(Int64Array::from(out)) as ArrayRef)
            }))
        }
        ("hll_union", [a, b, DataType::Boolean]) if is_binary(a) && is_binary(b) => {
            Ok(Binding::new(DataType::Binary, |x: &Inputs<'_>| {
                let (a, b, allow) = (x.binaries(0)?, x.binaries(1)?, x.booleans(2)?);
                let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(a.len());
                for i in 0..a.len() {
                    if a.is_null(i) || b.is_null(i) || allow.is_null(i) {
                        out.push(None);
                        continue;
                    }
                    // Spark reads the first buffer, then the second, then compares.
                    let first = hll::from_bytes(a.value(i), "hll_union")?;
                    let second = hll::from_bytes(b.value(i), "hll_union")?;
                    let u = hll::union(&first, &second, allow.value(i), "hll_union")?;
                    out.push(Some(hll::to_bytes(&u)));
                }
                Ok(Arc::new(BinaryArray::from_iter(out)) as ArrayRef)
            }))
        }
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}
