// SPDX-License-Identifier: Apache-2.0
//! Spark's hash functions: `hash` (Murmur3 x86_32, seed 42), `xxhash64`
//! (XXH64, seed 42), `md5`, `sha1`, `sha2` and `crc32`.
//!
//! `hash` and `xxhash64` take any number of columns of any hashable type.
//! The first column is hashed with the seed, each next one with the hash so
//! far; a NULL leaves the hash unchanged. Nested values chain the same way:
//! an array hashes its elements in order, a struct its fields, a map every
//! key then its value. The type decides the bytes hashed:
//!
//! | Type | Hashed as |
//! |---|---|
//! | `boolean` | `int` 1 or 0 |
//! | `tinyint`, `smallint`, `int`, `date`, year-month interval | `int` |
//! | `bigint`, timestamps, day-time interval | `long` |
//! | `float`, `double` | `int`/`long` bits, `-0.0` as `0.0`, one canonical NaN |
//! | `decimal(p <= 18, s)` | the unscaled value as `long` |
//! | `decimal(p > 18, s)` | the bytes of the unscaled value (`BigInteger.toByteArray`) |
//! | `string`, `binary` | the bytes |
//!
//! Spark's day-time intervals and `MAP` hashing are version-independent;
//! `MAP` is rejected by the analyzer unless
//! `spark.sql.legacy.allowHashOnMapType` is set, which is the caller's check.
//!
//! [`routing`] is not a Spark function: it is the engine's seedless routing
//! hash `H_kg` and its key groups (internal exchanges, state keys).

mod digest;
mod murmur3;
pub mod routing;
mod xxh64;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, DurationMicrosecondType, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, IntervalYearMonthType, Time64NanosecondType,
    TimestampMicrosecondType,
};
use arrow_array::{Array, ArrowPrimitiveType, Int32Array, Int64Array};
use arrow_buffer::{ArrowNativeType, NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};

pub use digest::{crc32, md5, sha1, sha2};

use crate::error::KernelError;
use murmur3::Murmur3;
use xxh64::Xxh64;

/// A Spark hash function over the primitive building blocks.
trait SparkHasher {
    /// The running hash.
    type State: Copy;
    /// Spark's default seed.
    const SEED: Self::State;
    fn hash_int(v: i32, seed: Self::State) -> Self::State;
    fn hash_long(v: i64, seed: Self::State) -> Self::State;
    fn hash_bytes(bytes: &[u8], seed: Self::State) -> Self::State;
}

/// `hash(columns...)`: Murmur3 x86_32 with seed 42, over `rows` rows.
///
/// With no columns every row is the seed. NULL columns (`NullArray`)
/// contribute nothing.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a type Spark cannot hash or this kernel
/// does not carry yet; `INTERNAL_ERROR` if a column is not `rows` long.
pub fn hash(columns: &[&dyn Array], rows: usize) -> Result<Int32Array, KernelError> {
    run::<Murmur3>(columns, rows).map(Int32Array::from)
}

/// `xxhash64(columns...)`: XXH64 with seed 42, over `rows` rows.
///
/// # Errors
/// As [`hash()`].
pub fn xxhash64(columns: &[&dyn Array], rows: usize) -> Result<Int64Array, KernelError> {
    run::<Xxh64>(columns, rows).map(Int64Array::from)
}

fn run<H: SparkHasher>(columns: &[&dyn Array], rows: usize) -> Result<Vec<H::State>, KernelError> {
    let plans = columns
        .iter()
        .map(|c| {
            if c.len() == rows {
                Col::plan(*c)
            } else {
                Err(KernelError::internal(format!(
                    "hash column has {} rows, expected {rows}",
                    c.len()
                )))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = crate::error::with_capacity(rows)?;
    for i in 0..rows {
        out.push(plans.iter().fold(H::SEED, |s, c| c.hash::<H>(i, s)));
    }
    Ok(out)
}

/// The values of a primitive column and its validity.
struct Prim<'a, T> {
    values: &'a [T],
    nulls: Option<&'a NullBuffer>,
}

impl<'a, T: ArrowNativeType> Prim<'a, T> {
    fn of<P: ArrowPrimitiveType<Native = T>>(a: &'a dyn Array) -> Self {
        let p = a.as_primitive::<P>();
        Self {
            values: p.values(),
            nulls: p.nulls(),
        }
    }

    /// `f` of the row's value, or `s` for a NULL.
    fn fold<S>(&self, i: usize, s: S, f: impl FnOnce(T, S) -> S) -> S {
        match (self.values.get(i), self.nulls) {
            (Some(v), None) => f(*v, s),
            (Some(v), Some(n)) if n.is_valid(i) => f(*v, s),
            _ => s,
        }
    }
}

/// A column resolved to its type once, so rows only match on the variant.
enum Col<'a> {
    Null,
    Bool(&'a arrow_array::BooleanArray),
    Int8(Prim<'a, i8>),
    Int16(Prim<'a, i16>),
    Int32(Prim<'a, i32>),
    Int64(Prim<'a, i64>),
    Float32(Prim<'a, f32>),
    Float64(Prim<'a, f64>),
    /// `decimal(p <= 18, s)`: the unscaled value as a `long`.
    DecimalLong(Prim<'a, i128>),
    /// `decimal(p > 18, s)`: the bytes of the unscaled value.
    DecimalBytes(Prim<'a, i128>),
    Bytes(Bytes<'a>),
    List(&'a OffsetBuffer<i32>, Option<&'a NullBuffer>, Box<Self>),
    LargeList(&'a OffsetBuffer<i64>, Option<&'a NullBuffer>, Box<Self>),
    Struct(Option<&'a NullBuffer>, Vec<Self>),
    Map(
        &'a OffsetBuffer<i32>,
        Option<&'a NullBuffer>,
        Box<Self>,
        Box<Self>,
    ),
}

/// A string or binary column.
enum Bytes<'a> {
    Utf8(&'a arrow_array::StringArray),
    LargeUtf8(&'a arrow_array::LargeStringArray),
    Binary(&'a arrow_array::BinaryArray),
    LargeBinary(&'a arrow_array::LargeBinaryArray),
}

impl Bytes<'_> {
    fn at(&self, i: usize) -> Option<&[u8]> {
        match self {
            Self::Utf8(a) => a.is_valid(i).then(|| a.value(i).as_bytes()),
            Self::LargeUtf8(a) => a.is_valid(i).then(|| a.value(i).as_bytes()),
            Self::Binary(a) => a.is_valid(i).then(|| a.value(i)),
            Self::LargeBinary(a) => a.is_valid(i).then(|| a.value(i)),
        }
    }
}

fn unsupported(t: &DataType) -> KernelError {
    KernelError::unsupported(format!("hashing a value of type {t}"))
}

impl<'a> Col<'a> {
    fn plan(a: &'a dyn Array) -> Result<Self, KernelError> {
        Ok(match a.data_type() {
            DataType::Null => Self::Null,
            DataType::Boolean => Self::Bool(a.as_boolean()),
            DataType::Int8 => Self::Int8(Prim::of::<Int8Type>(a)),
            DataType::Int16 => Self::Int16(Prim::of::<Int16Type>(a)),
            DataType::Int32 => Self::Int32(Prim::of::<Int32Type>(a)),
            DataType::Date32 => Self::Int32(Prim::of::<Date32Type>(a)),
            DataType::Interval(IntervalUnit::YearMonth) => {
                Self::Int32(Prim::of::<IntervalYearMonthType>(a))
            }
            DataType::Int64 => Self::Int64(Prim::of::<Int64Type>(a)),
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                Self::Int64(Prim::of::<TimestampMicrosecondType>(a))
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                Self::Int64(Prim::of::<DurationMicrosecondType>(a))
            }
            // TIME: Spark hashes the nanoseconds since midnight as a long.
            DataType::Time64(TimeUnit::Nanosecond) => {
                Self::Int64(Prim::of::<Time64NanosecondType>(a))
            }
            DataType::Float32 => Self::Float32(Prim::of::<Float32Type>(a)),
            DataType::Float64 => Self::Float64(Prim::of::<Float64Type>(a)),
            // `decimal(p <= 18, s)` on its 64-bit carrier: the long as is.
            DataType::Decimal64(..) => Self::Int64(Prim::of::<Decimal64Type>(a)),
            DataType::Decimal128(p, _) => {
                let d = Prim::of::<Decimal128Type>(a);
                if *p <= 18 {
                    Self::DecimalLong(d)
                } else {
                    Self::DecimalBytes(d)
                }
            }
            DataType::Utf8 => Self::Bytes(Bytes::Utf8(a.as_string::<i32>())),
            DataType::LargeUtf8 => Self::Bytes(Bytes::LargeUtf8(a.as_string::<i64>())),
            DataType::Binary => Self::Bytes(Bytes::Binary(a.as_binary::<i32>())),
            DataType::LargeBinary => Self::Bytes(Bytes::LargeBinary(a.as_binary::<i64>())),
            DataType::List(_) => {
                let l = a.as_list::<i32>();
                Self::List(
                    l.offsets(),
                    l.nulls(),
                    Box::new(Self::plan(l.values().as_ref())?),
                )
            }
            DataType::LargeList(_) => {
                let l = a.as_list::<i64>();
                Self::LargeList(
                    l.offsets(),
                    l.nulls(),
                    Box::new(Self::plan(l.values().as_ref())?),
                )
            }
            DataType::Struct(_) => {
                let s = a.as_struct();
                let fields = s
                    .columns()
                    .iter()
                    .map(|c| Self::plan(c.as_ref()))
                    .collect::<Result<_, _>>()?;
                Self::Struct(s.nulls(), fields)
            }
            DataType::Map(..) => {
                let m = a.as_map();
                Self::Map(
                    m.offsets(),
                    m.nulls(),
                    Box::new(Self::plan(m.keys().as_ref())?),
                    Box::new(Self::plan(m.values().as_ref())?),
                )
            }
            other => return Err(unsupported(other)),
        })
    }

    /// The hash of row `i` chained onto `s`.
    fn hash<H: SparkHasher>(&self, i: usize, s: H::State) -> H::State {
        match self {
            Self::Null => s,
            Self::Bool(a) => {
                if a.is_valid(i) {
                    H::hash_int(i32::from(a.value(i)), s)
                } else {
                    s
                }
            }
            Self::Int8(p) => p.fold(i, s, |v, s| H::hash_int(v.into(), s)),
            Self::Int16(p) => p.fold(i, s, |v, s| H::hash_int(v.into(), s)),
            Self::Int32(p) => p.fold(i, s, H::hash_int),
            Self::Int64(p) => p.fold(i, s, H::hash_long),
            Self::Float32(p) => p.fold(i, s, |v, s| H::hash_int(float_bits(v), s)),
            Self::Float64(p) => p.fold(i, s, |v, s| H::hash_long(double_bits(v), s)),
            // At most 18 digits fit a long; the fallback is never taken.
            Self::DecimalLong(p) => p.fold(i, s, |v, s| {
                H::hash_long(i64::try_from(v).unwrap_or_default(), s)
            }),
            Self::DecimalBytes(p) => p.fold(i, s, |v, s| {
                let (buf, start) = twos_complement_bytes(v);
                H::hash_bytes(buf.get(start..).unwrap_or_default(), s)
            }),
            Self::Bytes(b) => b.at(i).map_or(s, |bytes| H::hash_bytes(bytes, s)),
            Self::List(offsets, nulls, child) => {
                chain(offsets, *nulls, i, s, |j, s| child.hash::<H>(j, s))
            }
            Self::LargeList(offsets, nulls, child) => {
                chain(offsets, *nulls, i, s, |j, s| child.hash::<H>(j, s))
            }
            Self::Struct(nulls, fields) => {
                if nulls.is_some_and(|n| n.is_null(i)) {
                    s
                } else {
                    fields.iter().fold(s, |s, f| f.hash::<H>(i, s))
                }
            }
            Self::Map(offsets, nulls, keys, values) => chain(offsets, *nulls, i, s, |j, s| {
                values.hash::<H>(j, keys.hash::<H>(j, s))
            }),
        }
    }
}

/// Folds `f` over the child rows of list row `i` (none for a NULL list).
fn chain<O: arrow_array::OffsetSizeTrait, S>(
    offsets: &OffsetBuffer<O>,
    nulls: Option<&NullBuffer>,
    i: usize,
    s: S,
    mut f: impl FnMut(usize, S) -> S,
) -> S {
    if nulls.is_some_and(|n| n.is_null(i)) {
        return s;
    }
    let (Some(start), Some(end)) = (offsets.get(i), offsets.get(i.wrapping_add(1))) else {
        return s;
    };
    (start.as_usize()..end.as_usize()).fold(s, |s, j| f(j, s))
}

/// `Float.floatToIntBits` of the value with `-0.0` as `0.0`.
fn float_bits(v: f32) -> i32 {
    if v == 0.0 {
        0
    } else if v.is_nan() {
        0x7fc0_0000
    } else {
        v.to_bits().cast_signed()
    }
}

/// `Double.doubleToLongBits` of the value with `-0.0` as `0.0`.
fn double_bits(v: f64) -> i64 {
    if v == 0.0 {
        0
    } else if v.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        v.to_bits().cast_signed()
    }
}

/// `BigInteger.toByteArray` of `v`: the shortest big-endian two's
/// complement (at least one byte). Returns the 16-byte buffer and the index
/// of the first byte that belongs to the result.
fn twos_complement_bytes(v: i128) -> ([u8; 16], usize) {
    let magnitude_bits = if v < 0 { !v } else { v }.leading_zeros();
    // Significant bits plus the sign bit, in whole bytes.
    let len = 128_u32
        .saturating_sub(magnitude_bits)
        .saturating_add(1)
        .div_ceil(8);
    let start = 16_usize.saturating_sub(usize::try_from(len).unwrap_or(16));
    (v.to_be_bytes(), start)
}
