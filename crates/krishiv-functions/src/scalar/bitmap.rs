// SPDX-License-Identifier: Apache-2.0
//! `bitmap_bucket_number`, `bitmap_bit_position` and `bitmap_count` (P3-T086), after Spark's `BitmapExpressionUtils`: a bucket
//! is 4,096 bytes (32,768 bits); positive values count from 1 and the rest from 0, with Java's truncating division.

use std::sync::Arc;

use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::DataType;

use super::args::Inputs;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "bitmap_bucket_number",
    "bitmap_bit_position",
    "bitmap_count",
];

/// Bits in one bucket (`MAX_BYTES * 8`).
const BUCKET_BITS: i64 = 4096 * 8;

/// `v > 0 ? 1 + (v - 1) / BITS : v / BITS`.
fn bucket_number(v: i64) -> i64 {
    if v > 0 {
        1 + (v - 1) / BUCKET_BITS
    } else {
        v / BUCKET_BITS
    }
}

/// `v > 0 ? (v - 1) % BITS : (-v) % BITS`; `-Long.MIN_VALUE` wraps as it does in Java.
fn bit_position(v: i64) -> i64 {
    if v > 0 {
        (v - 1) % BUCKET_BITS
    } else {
        v.wrapping_neg() % BUCKET_BITS
    }
}

pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    if args.len() != 1 {
        return Some((listed, Err(bad_types(listed, args))));
    }
    let binding = match (listed, args[0].data_type()) {
        ("bitmap_bucket_number" | "bitmap_bit_position", DataType::Int64) => {
            let f: fn(i64) -> i64 = if listed == "bitmap_bucket_number" {
                bucket_number
            } else {
                bit_position
            };
            Ok(Binding::new(DataType::Int64, move |x: &Inputs<'_>| {
                let v = x.array::<Int64Type>(0)?;
                let out: Int64Array = v.iter().map(|o| o.map(f)).collect();
                Ok(Arc::new(out) as ArrayRef)
            }))
        }
        ("bitmap_count", DataType::Binary) => {
            Ok(Binding::new(DataType::Int64, |x: &Inputs<'_>| {
                let b = x.binaries(0)?;
                let out: Int64Array = b
                    .iter()
                    .map(|o| {
                        o.map(|bytes| bytes.iter().map(|byte| i64::from(byte.count_ones())).sum())
                    })
                    .collect();
                Ok(Arc::new(out) as ArrayRef)
            }))
        }
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}
