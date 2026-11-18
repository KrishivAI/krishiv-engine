// SPDX-License-Identifier: Apache-2.0
//! The values of an Arrow column as a Theta sketch sees them (P3-T024, PR-54a): each non-null value
//! in the Iceberg single-value binary serialisation (table spec appendix D, the bytes of
//! [`super::bounds::encode`]), hashed as raw bytes with DataSketches' default seed, as Java's
//! `ThetaSketchAgg` does (`Conversions.toByteBuffer` then `UpdateSketch.update(ByteBuffer)`).
//!
//! - `int`, `date`: 4 bytes little-endian; `long`, `time`, `timestamp`, `timestamptz` (microseconds) and the
//!   nanosecond timestamps: 8 bytes little-endian; `float`, `double`: the raw IEEE bits little-endian (so `-0.0`
//!   and `0.0` are two values and a NaN keeps its payload); `decimal`: the minimal big-endian two's complement
//!   (Java's `BigInteger.toByteArray`); `string`: UTF-8; `binary`, `fixed`: the bytes; `boolean`: one byte, 1 or 0.
//! - A **zero-length** string or binary is skipped, as Java's `update(ByteBuffer)` rejects an empty buffer: a
//!   column of only `""` gives an empty sketch (review B2).
//! - **`uuid` is not sketched** (review S1): Spark reads a `uuid` as a string, so Java's blob is built from the 36
//!   text bytes, not the 16 spec bytes; a union of the two would count every value twice. The column is left out.
//!
//! Hashing goes through [`raw_bytes::from_slice`], never a `&[u8]` (Rust's `Hash` for slices adds a length
//! prefix, which no other implementation hashes).

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "fed by PR-54b's sketches on write; PR-54a tests it"
    )
)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int32Type,
    Int64Type, Time64MicrosecondType, TimestampMicrosecondType, TimestampNanosecondType,
};
use arrow_schema::{DataType, TimeUnit};
use datasketches::hash::value::raw_bytes;
use datasketches::theta::{ThetaSketch, ThetaSketchBuilder};

use super::bounds::min_be_bytes;
use super::types::IcebergType;

/// A sketch as Iceberg's Spark writer builds them: the library's default `k` (4,096) and seed.
///
/// `None` only if the library refused its own defaults (it does not; no `unwrap` in library code).
#[must_use]
pub(crate) fn new_sketch() -> Option<ThetaSketch> {
    ThetaSketchBuilder::default().build().ok()
}

/// Whether a column of Iceberg type `ty` is sketched at all (`uuid` and the nested types are not).
#[must_use]
pub(crate) fn sketched(ty: &IcebergType) -> bool {
    !matches!(ty, IcebergType::Uuid)
}

/// Updates `sketch` with every non-empty, non-null value of `array`, a column of Iceberg type `ty`.
///
/// Returns `false`, having updated nothing, when `ty` is not sketched or `array`'s Arrow type is not one
/// this type is carried in; the caller leaves the column out.
pub(crate) fn update_column(sketch: &mut ThetaSketch, array: &dyn Array, ty: &IcebergType) -> bool {
    for_each_value(array, ty, &mut |bytes| {
        sketch.update(raw_bytes::from_slice(bytes));
    })
}

/// Calls `f` with the serialisation of each value of `array` that a sketch takes (see the module docs).
/// `false`, with `f` not called, when the type pair is not supported.
pub(crate) fn for_each_value(
    array: &dyn Array,
    ty: &IcebergType,
    f: &mut dyn FnMut(&[u8]),
) -> bool {
    use IcebergType as T;
    match (ty, array.data_type()) {
        (T::Boolean, DataType::Boolean) => {
            for v in array.as_boolean().iter().flatten() {
                f(&[u8::from(v)]);
            }
        }
        (T::Int, DataType::Int32) => fixed(array.as_primitive::<Int32Type>(), f, i32::to_le_bytes),
        (T::Date, DataType::Date32) => {
            fixed(array.as_primitive::<Date32Type>(), f, i32::to_le_bytes);
        }
        (T::Long, DataType::Int64) => fixed(array.as_primitive::<Int64Type>(), f, i64::to_le_bytes),
        (T::Time, DataType::Time64(TimeUnit::Microsecond)) => {
            fixed(
                array.as_primitive::<Time64MicrosecondType>(),
                f,
                i64::to_le_bytes,
            );
        }
        (T::Timestamp | T::Timestamptz, DataType::Timestamp(TimeUnit::Microsecond, _)) => {
            fixed(
                array.as_primitive::<TimestampMicrosecondType>(),
                f,
                i64::to_le_bytes,
            );
        }
        (T::TimestampNs | T::TimestamptzNs, DataType::Timestamp(TimeUnit::Nanosecond, _)) => {
            fixed(
                array.as_primitive::<TimestampNanosecondType>(),
                f,
                i64::to_le_bytes,
            );
        }
        (T::Float, DataType::Float32) => {
            fixed(array.as_primitive::<Float32Type>(), f, f32::to_le_bytes);
        }
        (T::Double, DataType::Float64) => {
            fixed(array.as_primitive::<Float64Type>(), f, f64::to_le_bytes);
        }
        (T::Decimal { .. }, DataType::Decimal32(..)) => {
            for v in array.as_primitive::<Decimal32Type>().iter().flatten() {
                f(&min_be_bytes(i128::from(v)));
            }
        }
        (T::Decimal { .. }, DataType::Decimal64(..)) => {
            for v in array.as_primitive::<Decimal64Type>().iter().flatten() {
                f(&min_be_bytes(i128::from(v)));
            }
        }
        (T::Decimal { .. }, DataType::Decimal128(..)) => {
            for v in array.as_primitive::<Decimal128Type>().iter().flatten() {
                f(&min_be_bytes(v));
            }
        }
        (T::String, DataType::Utf8) => {
            non_empty(
                array.as_string::<i32>().iter().flatten().map(str::as_bytes),
                f,
            );
        }
        (T::String, DataType::LargeUtf8) => {
            non_empty(
                array.as_string::<i64>().iter().flatten().map(str::as_bytes),
                f,
            );
        }
        (T::String, DataType::Utf8View) => {
            non_empty(
                array.as_string_view().iter().flatten().map(str::as_bytes),
                f,
            );
        }
        (T::Binary, DataType::Binary) => {
            non_empty(array.as_binary::<i32>().iter().flatten(), f);
        }
        (T::Binary, DataType::LargeBinary) => {
            non_empty(array.as_binary::<i64>().iter().flatten(), f);
        }
        (T::Binary, DataType::BinaryView) => {
            non_empty(array.as_binary_view().iter().flatten(), f);
        }
        (T::Fixed(_) | T::Binary, DataType::FixedSizeBinary(_)) => {
            non_empty(array.as_fixed_size_binary().iter().flatten(), f);
        }
        _ => return false,
    }
    true
}

/// Each non-null value of a primitive array in `to_bytes`'s fixed width.
fn fixed<T: arrow_array::ArrowPrimitiveType, const N: usize>(
    array: &arrow_array::PrimitiveArray<T>,
    f: &mut dyn FnMut(&[u8]),
    to_bytes: fn(T::Native) -> [u8; N],
) {
    for v in array.iter().flatten() {
        f(&to_bytes(v));
    }
}

/// Each non-empty value of a string or binary column (nulls already left out).
fn non_empty<'a>(values: impl Iterator<Item = &'a [u8]>, f: &mut dyn FnMut(&[u8])) {
    for b in values.filter(|b| !b.is_empty()) {
        f(b);
    }
}
