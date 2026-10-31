// SPDX-License-Identifier: Apache-2.0
//! Spark `CAST` between the types of TPC-H and the DataFrame slice:
//! integral ↔ integral ↔ float ↔ decimal, string → integral / decimal /
//! float / date / timestamp, date ↔ timestamp (`TIMESTAMP` and `TIMESTAMP_NTZ`), and
//! every one of these → string.
//!
//! [`cast`] takes the Arrow carrier of the Spark type: `Int8..Int64`,
//! `Float32/64`, `Decimal128(p, s)`, `Utf8`/`LargeUtf8`, `Boolean`,
//! `Date32`, `Timestamp(Microsecond, Some(_))` for `TIMESTAMP` (an instant,
//! shown in the session time zone) and `Timestamp(Microsecond, None)` for
//! `TIMESTAMP_NTZ`. The session zone comes in [`CastOptions::time_zone`];
//! the zone name in a `TIMESTAMP` carrier is not read.
//!
//! [`CastMode`] is Spark's: `Ansi` raises (`CAST_INVALID_INPUT`,
//! `CAST_OVERFLOW`, `NUMERIC_VALUE_OUT_OF_RANGE`,
//! `NUMERIC_OUT_OF_SUPPORTED_RANGE`), `Legacy` is `spark.sql.ansi.enabled
//! = false` (integral narrowing wraps, malformed input is NULL) and `Try`
//! is `try_cast` (NULL wherever ANSI would raise). A NULL row never raises.
//! Casts that are not listed fail with `KRISHIV_UNSUPPORTED_FEATURE`
//! (the full cast matrix is a later task).

pub(crate) mod datetime;
pub(crate) mod text;
pub(crate) mod timestamp;

use std::sync::Arc;

use arrow_array::builder::BooleanBufferBuilder;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BinaryArray, BooleanArray, GenericStringArray,
    OffsetSizeTrait, PrimitiveArray, StringArray, cast::AsArray,
};
use arrow_buffer::{NullBuffer, ScalarBuffer, i256};
use arrow_schema::{DataType, TimeUnit};

pub use datetime::{FixedOffset, TimeZoneRules};
pub use timestamp::{TimestampText, timestamp_has_zone, timestamp_text};

use crate::decimal::DecimalNative;
use crate::error::{KernelError, with_capacity};
use crate::integer::SparkIntegral;
use crate::version::IntWidth;

/// How a cast handles a value that has no result in the target type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastMode {
    /// `spark.sql.ansi.enabled = true` (the default): raise.
    Ansi,
    /// `spark.sql.ansi.enabled = false`: wrap integral overflow (Java
    /// narrowing), NULL for anything else.
    Legacy,
    /// `try_cast`: NULL wherever ANSI would raise.
    Try,
}

/// The session state a cast depends on.
#[derive(Debug, Clone, Copy)]
pub struct CastOptions<'a> {
    /// ANSI, legacy or `try_cast`.
    pub mode: CastMode,
    /// `spark.sql.session.timeZone`, for `TIMESTAMP` ↔ `DATE` /
    /// `TIMESTAMP_NTZ` / `STRING`.
    pub time_zone: &'a dyn TimeZoneRules,
}

/// The Spark SQL name of an Arrow carrier type (`INT`, `DECIMAL(10,2)`).
#[must_use]
pub fn spark_type_name(t: &DataType) -> String {
    match t {
        DataType::Boolean => "BOOLEAN".to_owned(),
        DataType::Int8 => "TINYINT".to_owned(),
        DataType::Int16 => "SMALLINT".to_owned(),
        DataType::Int32 => "INT".to_owned(),
        DataType::Int64 => "BIGINT".to_owned(),
        DataType::Float32 => "FLOAT".to_owned(),
        DataType::Float64 => "DOUBLE".to_owned(),
        DataType::Decimal64(p, s) | DataType::Decimal128(p, s) => format!("DECIMAL({p},{s})"),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "STRING".to_owned(),
        DataType::Binary | DataType::LargeBinary => "BINARY".to_owned(),
        DataType::Date32 => "DATE".to_owned(),
        DataType::Timestamp(_, Some(_)) => "TIMESTAMP".to_owned(),
        DataType::Timestamp(_, None) => "TIMESTAMP_NTZ".to_owned(),
        other => other.to_string(),
    }
}

/// Spark's `toSQLString` of a string value: single quotes, with `\` and
/// `'` escaped by a backslash.
pub(crate) fn sql_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len().saturating_add(2));
    out.push('\'');
    for c in s.chars() {
        if c == '\\' || c == '\'' {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('\'');
    out
}

/// One source value, widened.
#[derive(Debug, Clone, Copy)]
enum Num {
    /// An integral value.
    Int(i128),
    /// A decimal: unscaled value and scale.
    Dec(i128, i8),
    /// A float (`f32` widened exactly); `true` when the source is `FLOAT`.
    Float(f64, bool),
}

impl Num {
    /// Spark's `toSQLValue` of the source, for `CAST_OVERFLOW`.
    fn sql_value(self, from: &DataType) -> String {
        match (self, from) {
            (Self::Int(v), DataType::Int16) => format!("{v}S"),
            (Self::Int(v), DataType::Int64) => format!("{v}L"),
            (Self::Int(v), _) => v.to_string(),
            (Self::Dec(v, s), _) => format!("{}BD", decimal_plain(v, s)),
            // NaN and the infinities carry no suffix.
            (Self::Float(v, is_f32), _) if is_f32 || !v.is_finite() => float_string(v, is_f32),
            (Self::Float(v, _), _) => format!("{}D", float_string(v, false)),
        }
    }

    /// The source as Spark's `Decimal` would hold it (the
    /// `NUMERIC_VALUE_OUT_OF_RANGE` value), as `BigDecimal`-style parts.
    fn big_dec(self) -> Option<text::BigDec> {
        match self {
            Self::Int(v) => text::parse_decimal(&v.to_string()),
            Self::Dec(v, s) => text::parse_decimal(&decimal_plain(v, s)),
            // `Decimal(d)` goes through `Double.toString`; NaN and infinity
            // have no decimal value.
            // (A `FLOAT` is widened to `double` first: 1.15F -> 1.1499999761...)
            Self::Float(v, _) => v
                .is_finite()
                .then(|| text::parse_decimal(&float_string(v, false)))
                .flatten(),
        }
    }
}

/// `BigDecimal.toPlainString` of an unscaled value.
fn decimal_plain(v: i128, scale: i8) -> String {
    let mut out = String::new();
    write_decimal_plain(v, scale, &mut out);
    out
}

/// [`decimal_plain`] appended to `out`.
fn write_decimal_plain(v: i128, scale: i8, out: &mut String) {
    use std::fmt::Write as _;
    if v < 0 {
        out.push('-');
    }
    let start = out.len();
    let _ = write!(out, "{}", v.unsigned_abs());
    let s = usize::from(scale.unsigned_abs());
    if s == 0 {
        return;
    }
    let len = out.len().saturating_sub(start);
    if len > s {
        out.insert(out.len().saturating_sub(s), '.');
    } else {
        let pad = "0".repeat(s.saturating_sub(len));
        out.insert_str(start, &format!("0.{pad}"));
    }
}

/// Java's `BigDecimal.toString` for a scale >= 0: plain while the adjusted
/// exponent (`digits - 1 - scale`) is at least -6, else `d.dddE-n`.
fn write_decimal_java(v: i128, scale: i8, out: &mut String) {
    use std::fmt::Write as _;
    let digits = v.unsigned_abs().to_string();
    let adjusted = i64::try_from(digits.len())
        .unwrap_or(0)
        .saturating_sub(1)
        .saturating_sub(i64::from(scale));
    if adjusted >= -6 {
        write_decimal_plain(v, scale, out);
        return;
    }
    if v < 0 {
        out.push('-');
    }
    let (first, rest) = digits.split_at(1);
    let _ = if rest.is_empty() {
        write!(out, "{first}E{adjusted}")
    } else {
        write!(out, "{first}.{rest}E{adjusted}")
    };
}

/// Java's `Double.toString` (or `Float.toString` of the `f32` value).
fn float_string(v: f64, is_f32: bool) -> String {
    let (nan, inf) = (v.is_nan(), v.is_infinite().then_some(v < 0.0));
    if is_f32 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "v is an f32 widened exactly"
        )]
        let f = v as f32;
        text::java_float_string(f, nan, inf)
    } else {
        text::java_float_string(v, nan, inf)
    }
}

/// Runs `f` on every non-NULL row (NULL rows are skipped: they never raise
/// and cost nothing) and collects a primitive array. The output validity
/// is only materialized once a row is NULL.
fn map_rows<P: ArrowPrimitiveType>(
    input: &dyn Array,
    mut f: impl FnMut(usize) -> Result<Option<P::Native>, KernelError>,
) -> Result<PrimitiveArray<P>, KernelError> {
    let len = input.len();
    let mut values = with_capacity::<P::Native>(len)?;
    let mut valid: Option<BooleanBufferBuilder> = None;
    let nulls = input.nulls().filter(|n| n.null_count() > 0);
    for i in 0..len {
        let v = if nulls.is_some_and(|n| n.is_null(i)) {
            None
        } else {
            f(i)?
        };
        match v {
            Some(x) => {
                values.push(x);
                if let Some(b) = valid.as_mut() {
                    b.append(true);
                }
            }
            None => {
                values.push(P::Native::default());
                valid
                    .get_or_insert_with(|| {
                        let mut b = BooleanBufferBuilder::new(len);
                        b.append_n(i, true);
                        b
                    })
                    .append(false);
            }
        }
    }
    let nulls = valid.map(|mut b| NullBuffer::new(b.finish()));
    Ok(PrimitiveArray::new(ScalarBuffer::from(values), nulls))
}

/// The result of a value with no exact result: an error under ANSI, else
/// `legacy` (`None`: NULL).
fn fail<T>(
    mode: CastMode,
    legacy: Option<T>,
    error: impl FnOnce() -> KernelError,
) -> Result<Option<T>, KernelError> {
    match mode {
        CastMode::Ansi => Err(error()),
        CastMode::Legacy => Ok(legacy),
        CastMode::Try => Ok(None),
    }
}

/// A numeric array, resolved once per batch.
#[derive(Clone, Copy)]
enum NumArray<'a> {
    Bool(&'a arrow_array::BooleanArray),
    I8(&'a PrimitiveArray<Int8Type>),
    I16(&'a PrimitiveArray<Int16Type>),
    I32(&'a PrimitiveArray<Int32Type>),
    I64(&'a PrimitiveArray<Int64Type>),
    F32(&'a PrimitiveArray<Float32Type>),
    F64(&'a PrimitiveArray<Float64Type>),
    Dec(&'a PrimitiveArray<Decimal128Type>, i8),
}

impl<'a> NumArray<'a> {
    fn new(a: &'a dyn Array) -> Option<Self> {
        Some(match a.data_type() {
            DataType::Boolean => Self::Bool(a.as_boolean()),
            DataType::Int8 => Self::I8(a.as_primitive()),
            DataType::Int16 => Self::I16(a.as_primitive()),
            DataType::Int32 => Self::I32(a.as_primitive()),
            DataType::Int64 => Self::I64(a.as_primitive()),
            DataType::Float32 => Self::F32(a.as_primitive()),
            DataType::Float64 => Self::F64(a.as_primitive()),
            DataType::Decimal128(_, s) => Self::Dec(a.as_primitive(), *s),
            _ => return None,
        })
    }

    /// Row `i`, widened.
    #[inline]
    fn get(self, i: usize) -> Num {
        match self {
            // `true` is 1, `false` 0 (Spark's boolean -> numeric casts).
            Self::Bool(a) => Num::Int(a.value(i).into()),
            Self::I8(a) => Num::Int(a.value(i).into()),
            Self::I16(a) => Num::Int(a.value(i).into()),
            Self::I32(a) => Num::Int(a.value(i).into()),
            Self::I64(a) => Num::Int(a.value(i).into()),
            Self::F32(a) => Num::Float(a.value(i).into(), true),
            Self::F64(a) => Num::Float(a.value(i), false),
            Self::Dec(a, s) => Num::Dec(a.value(i), s),
        }
    }
}

/// A numeric cast source (`BOOLEAN` included).
fn is_numeric(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(..)
    )
}

/// `value` cast to integral `T` (Spark's `Cast` to byte/short/int/long).
fn to_integral<T: SparkIntegral>(
    v: Num,
    mode: CastMode,
    overflow: impl FnOnce() -> KernelError,
) -> Result<Option<T::Native>, KernelError> {
    let exact = match v {
        Num::Int(x) => x,
        // Truncation towards zero; the range check applies to the result.
        // Divergence (pending an owner decision): for a decimal of
        // precision > 18 to int/smallint/tinyint, Spark's ANSI and try
        // casts go through `double`, so `0.99999999999999999999` becomes 1
        // and `2147483647.99999999999999999999` overflows; Krishiv truncates
        // exactly, as Spark's legacy cast and every bigint cast do.
        Num::Dec(x, s) => x.checked_div(pow10(s.unsigned_abs())).unwrap_or_default(),
        Num::Float(x, _) => return float_to_integral::<T>(x, mode, overflow),
    };
    match T::narrow(exact) {
        (n, false) => Ok(Some(n)),
        // Legacy narrowing keeps the low bits (Java's `(int) long`).
        (n, true) => fail(mode, Some(n), overflow),
    }
}

/// Spark's float → integral: ANSI accepts `floor(x) <= MAX` and
/// `ceil(x) >= MIN` and truncates; legacy saturates like Java's `(int) d`
/// (NaN is 0), then narrows `tinyint`/`smallint` from `int` with a wrap.
fn float_to_integral<T: SparkIntegral>(
    x: f64,
    mode: CastMode,
    overflow: impl FnOnce() -> KernelError,
) -> Result<Option<T::Native>, KernelError> {
    let (lo, hi) = match T::WIDTH {
        IntWidth::Byte => (i128::from(i8::MIN), i128::from(i8::MAX)),
        IntWidth::Short => (i128::from(i16::MIN), i128::from(i16::MAX)),
        IntWidth::Int => (i128::from(i32::MIN), i128::from(i32::MAX)),
        IntWidth::Long => (i128::from(i64::MIN), i128::from(i64::MAX)),
    };
    #[expect(clippy::cast_precision_loss, reason = "Spark compares in double too")]
    let in_range = x.floor() <= hi as f64 && x.ceil() >= lo as f64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Java's saturating double -> long"
    )]
    let java = |x: f64| x as i64;
    if in_range {
        let v = T::narrow(i128::from(java(x))).0;
        return Ok(Some(v));
    }
    let legacy = match T::WIDTH {
        IntWidth::Long => T::narrow(i128::from(java(x))).0,
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Java's saturating double -> int"
        )]
        _ => T::narrow(i128::from(x as i32)).0,
    };
    fail(mode, Some(legacy), overflow)
}

/// `10^k` for `k <= 38` (saturating at `10^38`).
fn pow10(k: u8) -> i128 {
    const POW10: [i128; 39] = {
        let mut t = [1_i128; 39];
        let mut i = 1;
        while i < 39 {
            t[i] = t[i - 1].wrapping_mul(10);
            i += 1;
        }
        t
    };
    POW10[usize::from(k.min(38))]
}

/// `value` as `decimal(p, s)`, rounded `HALF_UP`; the error carries the
/// source as Spark's `Decimal` (for the message).
fn to_decimal(v: Num, p: u8, s: i8) -> Result<Option<i128>, text::BigDec> {
    let scale = s.unsigned_abs();
    let unscaled = match v {
        Num::Int(x) => x.checked_mul(pow10(scale)),
        Num::Dec(x, from) => {
            let from = from.unsigned_abs();
            if from <= scale {
                x.checked_mul(pow10(scale.saturating_sub(from)))
            } else {
                Some(half_up(x, pow10(from.saturating_sub(scale))))
            }
        }
        Num::Float(..) => {
            let Some(d) = v.big_dec() else {
                // NaN / infinity: NULL in every mode.
                return Ok(None);
            };
            d.to_scale(scale).and_then(i256::to_i128)
        }
    };
    let bound = pow10(p);
    match unscaled {
        Some(u) if u < bound && u > bound.wrapping_neg() => Ok(Some(u)),
        _ => Err(v.big_dec().unwrap_or(text::BigDec {
            negative: false,
            digits: Vec::new(),
            exp: 0,
        })),
    }
}

/// `x / d` (`d` a power of ten) rounded half away from zero.
fn half_up(x: i128, d: i128) -> i128 {
    // 64-bit division when both fit: several times cheaper than 128-bit.
    let (q, r) = match (i64::try_from(x), i64::try_from(d)) {
        (Ok(x), Ok(d)) => (
            i128::from(x.checked_div(d).unwrap_or_default()),
            i128::from(x.checked_rem(d).unwrap_or_default()),
        ),
        _ => (
            x.checked_div(d).unwrap_or_default(),
            x.checked_rem(d).unwrap_or_default(),
        ),
    };
    if r != 0 && r.unsigned_abs() >= d.unsigned_abs().wrapping_sub(r.unsigned_abs()) {
        q.wrapping_add(x.signum())
    } else {
        q
    }
}

/// `10^k` for `k <= 22`: exact doubles (`powi` is not guaranteed exact).
const POW10_F64: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];
/// `10^k` for `k <= 10`: exact floats.
const POW10_F32: [f32; 11] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10];

/// `value` as a float; `F32` rounds directly from the source.
fn to_float(v: Num, f32_target: bool) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "Java's long -> double/float rounding"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Java's double -> float rounding"
    )]
    match (v, f32_target) {
        (Num::Int(x), false) => x as f64,
        (Num::Int(x), true) => f64::from(x as f32),
        (Num::Float(x, _), false) => x,
        (Num::Float(x, _), true) => f64::from(x as f32),
        // `BigDecimal.doubleValue` / `floatValue`: correctly rounded. Exact
        // operands make one division correctly rounded too.
        (Num::Dec(x, s), false) if x.unsigned_abs() < 1 << 53 && s.unsigned_abs() <= 22 => {
            (x as f64) / POW10_F64[usize::from(s.unsigned_abs())]
        }
        (Num::Dec(x, s), true) if x.unsigned_abs() < 1 << 24 && s.unsigned_abs() <= 10 => {
            f64::from((x as f32) / POW10_F32[usize::from(s.unsigned_abs())])
        }
        (Num::Dec(x, s), false) => format!("{x}e-{}", s.unsigned_abs())
            .parse()
            .unwrap_or(f64::NAN),
        (Num::Dec(x, s), true) => format!("{x}e-{}", s.unsigned_abs())
            .parse::<f32>()
            .map_or(f64::NAN, f64::from),
    }
}

/// `CAST(input AS to)` with Spark semantics.
///
/// # Errors
/// Under [`CastMode::Ansi`], Spark's error for the first failing non-NULL
/// row (with [`KernelError::row`]); `KRISHIV_UNSUPPORTED_FEATURE` for a
/// cast this kernel does not cover; `UNABLE_TO_ACQUIRE_MEMORY`.
///
/// ```
/// use arrow_array::{Array, Int32Array, StringArray};
/// use arrow_schema::DataType;
/// use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};
///
/// let s = StringArray::from(vec![Some(" 12 "), Some("12.7"), None]);
/// let legacy = CastOptions { mode: CastMode::Legacy, time_zone: &FixedOffset(0) };
/// let out = cast(&s, &DataType::Int32, &legacy)?;
/// assert_eq!(out.as_ref(), &Int32Array::from(vec![Some(12), Some(12), None]) as &dyn Array);
/// let ansi = CastOptions { mode: CastMode::Ansi, ..legacy };
/// let err = cast(&s, &DataType::Int32, &ansi).unwrap_err();
/// assert_eq!(err.row(), Some(1));
/// # Ok::<(), krishiv_kernels::KernelError>(())
/// ```
pub fn cast(
    input: &dyn Array,
    to: &DataType,
    opts: &CastOptions<'_>,
) -> Result<ArrayRef, KernelError> {
    let from = input.data_type();
    if from == to {
        return Ok(arrow_array::make_array(input.to_data()));
    }
    // `Decimal64` is a carrier of `decimal(p <= 18, s)` (design P2-T160):
    // the cast is the `Decimal128` one, with the operand widened and the
    // result narrowed; Spark sees the same type, values and errors.
    if let DataType::Decimal64(p, s) = from {
        let wide = input
            .as_primitive::<Decimal64Type>()
            .unary::<_, Decimal128Type>(i128::from)
            .with_data_type(DataType::Decimal128(*p, *s));
        return cast(&wide, to, opts);
    }
    if let DataType::Decimal64(p, s) = to {
        let out = cast(input, &DataType::Decimal128(*p, *s), opts)?;
        // A valid value is within precision `p <= 18`, so it fits an `i64`;
        // values under NULL slots are arbitrary.
        let narrow = out
            .as_primitive::<Decimal128Type>()
            .unary::<_, Decimal64Type>(|v| v.low64().0)
            .with_precision_and_scale(*p, *s)
            .map_err(|e| KernelError::internal(e.to_string()))?;
        return Ok(Arc::new(narrow));
    }
    let unsupported = || {
        KernelError::unsupported(format!(
            "CAST from {} to {}",
            spark_type_name(from),
            spark_type_name(to)
        ))
    };
    let mode = opts.mode;
    // Spark creates negative-scale decimals only under a legacy flag; every
    // conversion here assumes a scale >= 0.
    if matches!(from, DataType::Decimal128(_, s) if *s < 0)
        || matches!(to, DataType::Decimal128(_, s) if *s < 0)
    {
        return Err(unsupported());
    }
    let (from_name, to_name) = (spark_type_name(from), spark_type_name(to));
    let overflow = |row: usize, v: Num| {
        KernelError::cast_overflow(row, v.sql_value(from), &from_name, &to_name)
    };
    let nums = NumArray::new(input);
    macro_rules! numeric_to_int {
        ($t:ty) => {{
            let nums = nums.ok_or_else(unsupported)?;
            let out = map_rows::<$t>(input, |i| {
                let v = nums.get(i);
                to_integral::<$t>(v, mode, || overflow(i, v))
            })?;
            Ok(Arc::new(out) as ArrayRef)
        }};
    }
    macro_rules! string_to_int {
        ($o:ty, $t:ty) => {{
            let s = input.as_string::<$o>();
            let out = map_rows::<$t>(input, |i| {
                // Out of the target's range is malformed too (Spark's `toInt`);
                // `try_cast` parses as ANSI does (`toLongExact`), so `'1.5'` is NULL.
                let v = text::parse_integral(s.value(i).as_bytes(), mode != CastMode::Legacy)
                    .and_then(|v| match <$t as SparkIntegral>::narrow(i128::from(v)) {
                        (n, false) => Some(n),
                        (_, true) => None,
                    });
                match v {
                    Some(v) => Ok(Some(v)),
                    None => fail(mode, None, || {
                        KernelError::cast_invalid_input(
                            i,
                            sql_string(s.value(i)),
                            &from_name,
                            &to_name,
                        )
                    }),
                }
            })?;
            Ok(Arc::new(out) as ArrayRef)
        }};
    }
    match (from, to) {
        (f, DataType::Int8) if is_numeric(f) => numeric_to_int!(Int8Type),
        (f, DataType::Int16) if is_numeric(f) => numeric_to_int!(Int16Type),
        (f, DataType::Int32) if is_numeric(f) => numeric_to_int!(Int32Type),
        (f, DataType::Int64) if is_numeric(f) => numeric_to_int!(Int64Type),
        // Non-zero is `true` (a NaN is non-zero); never fails.
        (f, DataType::Boolean) if is_numeric(f) && *f != DataType::Boolean => {
            let nums = nums.ok_or_else(unsupported)?;
            let out: BooleanArray = (0..input.len())
                .map(|i| {
                    input.is_valid(i).then(|| match nums.get(i) {
                        Num::Int(v) | Num::Dec(v, _) => v != 0,
                        Num::Float(v, _) => v != 0.0,
                    })
                })
                .collect();
            Ok(Arc::new(out))
        }
        (f, DataType::Float32 | DataType::Float64) if is_numeric(f) => {
            let f32_target = *to == DataType::Float32;
            let nums = nums.ok_or_else(unsupported)?;
            let vals =
                map_rows::<Float64Type>(input, |i| Ok(Some(to_float(nums.get(i), f32_target))))?;
            Ok(float_out(&vals, f32_target))
        }
        (f, DataType::Decimal128(p, s)) if is_numeric(f) => {
            let nums = nums.ok_or_else(unsupported)?;
            let out = map_rows::<Decimal128Type>(input, |i| {
                let v = nums.get(i);
                match to_decimal(v, *p, *s) {
                    Ok(d) => Ok(d),
                    Err(src) => fail(mode, None, || {
                        KernelError::decimal_out_of_range(i, src.plain(), *p, *s)
                    }),
                }
            })?;
            decimal_out(out, *p, *s)
        }
        (DataType::Utf8, DataType::Int8) => string_to_int!(i32, Int8Type),
        (DataType::Utf8, DataType::Int16) => string_to_int!(i32, Int16Type),
        (DataType::Utf8, DataType::Int32) => string_to_int!(i32, Int32Type),
        (DataType::Utf8, DataType::Int64) => string_to_int!(i32, Int64Type),
        (DataType::LargeUtf8, DataType::Int8) => string_to_int!(i64, Int8Type),
        (DataType::LargeUtf8, DataType::Int16) => string_to_int!(i64, Int16Type),
        (DataType::LargeUtf8, DataType::Int32) => string_to_int!(i64, Int32Type),
        (DataType::LargeUtf8, DataType::Int64) => string_to_int!(i64, Int64Type),
        // Both are STRING carriers: the values are kept.
        // Both are STRING carriers: only the offset width changes.
        (DataType::LargeUtf8, DataType::Utf8) => {
            let s = input.as_string::<i64>();
            let offsets = s.value_offsets();
            let base = offsets.first().copied().unwrap_or(0);
            let narrow: Option<Vec<i32>> = offsets
                .iter()
                .map(|&o| i32::try_from(o.saturating_sub(base)).ok())
                .collect();
            let narrow = narrow.ok_or_else(|| {
                KernelError::unsupported("a STRING batch over 2 GiB in a 32-bit-offset carrier")
            })?;
            let len = usize::try_from(narrow.last().copied().unwrap_or(0)).unwrap_or(0);
            let start = usize::try_from(base).unwrap_or(0);
            let values = s.values().slice_with_length(start, len);
            let out = StringArray::try_new(
                arrow_buffer::OffsetBuffer::new(ScalarBuffer::from(narrow)),
                values,
                s.nulls().cloned(),
            )
            .map_err(|e| KernelError::internal(e.to_string()))?;
            Ok(Arc::new(out))
        }
        (DataType::Utf8, DataType::LargeUtf8) => {
            let s = input.as_string::<i32>();
            let wide: Vec<i64> = s.value_offsets().iter().map(|&o| i64::from(o)).collect();
            let out = arrow_array::LargeStringArray::try_new(
                arrow_buffer::OffsetBuffer::new(ScalarBuffer::from(wide)),
                s.values().clone(),
                s.nulls().cloned(),
            )
            .map_err(|e| KernelError::internal(e.to_string()))?;
            Ok(Arc::new(out))
        }
        (DataType::Utf8, DataType::Binary) => {
            // A `STRING` carrier holds bytes: reinterpreting them as
            // `BINARY` never fails (Spark's `CAST(string AS BINARY)` is the
            // UTF-8 bytes). The union planner uses this for binary-safe
            // string branches, whose bytes are not necessarily valid UTF-8.
            let s = input.as_string::<i32>();
            let out =
                BinaryArray::try_new(s.offsets().clone(), s.values().clone(), s.nulls().cloned())
                    .map_err(|e| KernelError::internal(e.to_string()))?;
            Ok(Arc::new(out))
        }
        (DataType::Utf8, _) => from_string(input.as_string::<i32>(), to, opts, &to_name),
        (DataType::LargeUtf8, _) => from_string(input.as_string::<i64>(), to, opts, &to_name),
        (DataType::Binary, DataType::Utf8) => {
            // STRING holds valid UTF-8 only (ADR 0009 update): invalid
            // bytes raise INVALID_UTF8_STRING under ANSI, become U+FFFD in
            // legacy mode (`make_valid_utf8`; maximal-subpart replacement)
            // and NULL under try_cast. Spark keeps the raw bytes (ADR 0023).
            let b = input.as_binary::<i32>();
            let mut out =
                arrow_array::builder::StringBuilder::with_capacity(b.len(), b.values().len());
            for (i, v) in b.iter().enumerate() {
                match v.map(|bytes| (bytes, std::str::from_utf8(bytes))) {
                    None => out.append_null(),
                    Some((_, Ok(s))) => out.append_value(s),
                    Some((bytes, Err(_))) => match mode {
                        CastMode::Ansi => return Err(KernelError::invalid_utf8(i, bytes)),
                        CastMode::Legacy => out.append_value(String::from_utf8_lossy(bytes)),
                        CastMode::Try => out.append_null(),
                    },
                }
            }
            Ok(Arc::new(out.finish()))
        }
        (_, DataType::Utf8) => to_string(input, opts).map(|s| Arc::new(s) as ArrayRef),
        (f, DataType::Timestamp(TimeUnit::Microsecond, zone @ Some(_))) if is_numeric(f) => {
            if *f == DataType::Boolean && mode == CastMode::Ansi {
                return Err(unsupported());
            }
            let nums = nums.ok_or_else(unsupported)?;
            // Legacy `true` is 1 microsecond, not 1 second.
            let boolean = *f == DataType::Boolean;
            let out = map_rows::<TimestampMicrosecondType>(input, |i| match nums.get(i) {
                Num::Int(b) if boolean => Ok(Some(i64::try_from(b).unwrap_or_default())),
                v => to_micros(v, mode, i),
            })?;
            Ok(Arc::new(out.with_timezone_opt(zone.clone())))
        }
        (DataType::Timestamp(TimeUnit::Microsecond, Some(_)), t) if is_numeric(t) => from_micros(
            input.as_primitive::<TimestampMicrosecondType>(),
            t,
            opts,
            &to_name,
        ),
        // Legacy mode casts a date to a number as NULL (ANSI rejects it
        // during analysis).
        (DataType::Date32, t) if is_numeric(t) && mode != CastMode::Ansi => {
            Ok(arrow_array::new_null_array(t, input.len()))
        }
        (DataType::Date32 | DataType::Timestamp(TimeUnit::Microsecond, _), _) => {
            temporal(input, to, opts).ok_or_else(unsupported)?
        }
        _ => Err(unsupported()),
    }
}

/// Numeric seconds since the epoch → `TIMESTAMP` micros: integral values
/// saturate (`TimeUnit.SECONDS.toMicros`), fractions truncate towards
/// zero, NaN and infinity are malformed, a double product beyond `bigint`
/// overflows, a decimal one keeps its low 64 bits (`BigDecimal.longValue`).
fn to_micros(v: Num, mode: CastMode, row: usize) -> Result<Option<i64>, KernelError> {
    const MICROS: i128 = 1_000_000;
    Ok(Some(match v {
        Num::Int(x) => i64::try_from(
            x.saturating_mul(MICROS)
                .clamp(i128::from(i64::MIN), i128::from(i64::MAX)),
        )
        .unwrap_or_default(),
        Num::Dec(x, s) => {
            let s = s.unsigned_abs();
            let m = if s <= 6 {
                x.wrapping_mul(pow10(6_u8.saturating_sub(s)))
            } else {
                x.checked_div(pow10(s.saturating_sub(6)))
                    .unwrap_or_default()
            };
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Spark keeps the low 64 bits"
            )]
            let low = m as i64;
            low
        }
        // Spark widens a float to double first: the message names DOUBLE.
        Num::Float(x, _) if !x.is_finite() => {
            return fail(mode, None, || {
                KernelError::cast_invalid_input(row, float_string(x, false), "DOUBLE", "TIMESTAMP")
            });
        }
        Num::Float(x, _) => {
            let p = x * 1e6;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Java's saturating double -> long"
            )]
            let java = p as i64;
            #[expect(clippy::cast_precision_loss, reason = "Spark compares in double")]
            let fits = p.floor() <= i64::MAX as f64 && p.ceil() >= i64::MIN as f64;
            if !fits {
                return fail(mode, Some(java), || {
                    KernelError::cast_overflow(
                        row,
                        format!("{}D", float_string(p, false)),
                        "DOUBLE",
                        "BIGINT",
                    )
                });
            }
            java
        }
    }))
}

/// `TIMESTAMP` → number: whole seconds (floored) for integral types (out
/// of range raises under ANSI, NULL otherwise), exact seconds for floats
/// and decimals.
fn from_micros(
    t: &PrimitiveArray<TimestampMicrosecondType>,
    to: &DataType,
    opts: &CastOptions<'_>,
    to_name: &str,
) -> Result<ArrayRef, KernelError> {
    use std::fmt::Write as _;
    let mode = opts.mode;
    let overflow = |row: usize| {
        let mut lit = String::from("TIMESTAMP '");
        let v = t.value(row);
        datetime::format_timestamp(
            datetime::utc_to_local(v, opts.time_zone).unwrap_or(v),
            &mut lit,
        );
        let _ = write!(lit, "'");
        KernelError::cast_overflow(row, lit, "TIMESTAMP", to_name)
    };
    macro_rules! int {
        ($t:ty) => {{
            let out = map_rows::<$t>(t, |i| {
                let secs = i128::from(t.value(i).div_euclid(1_000_000));
                match <$t as SparkIntegral>::narrow(secs) {
                    (v, false) => Ok(Some(v)),
                    // Legacy mode gives NULL here, not the wrapped value
                    // (checked on Spark 4.2: TIMESTAMP'2100-01-01' AS INT).
                    (_, true) => fail(mode, None, || overflow(i)),
                }
            })?;
            Ok(Arc::new(out) as ArrayRef)
        }};
    }
    match to {
        DataType::Int8 => int!(Int8Type),
        DataType::Int16 => int!(Int16Type),
        DataType::Int32 => int!(Int32Type),
        DataType::Int64 => int!(Int64Type),
        DataType::Float32 | DataType::Float64 => {
            #[expect(clippy::cast_precision_loss, reason = "Spark divides in double")]
            let vals = map_rows::<Float64Type>(t, |i| Ok(Some(t.value(i) as f64 / 1e6)))?;
            let f32_target = *to == DataType::Float32;
            Ok(float_out(&vals, f32_target))
        }
        DataType::Decimal128(p, s) => {
            let out = map_rows::<Decimal128Type>(t, |i| {
                match to_decimal(Num::Dec(t.value(i).into(), 6), *p, *s) {
                    Ok(d) => Ok(d),
                    Err(src) => fail(mode, None, || {
                        KernelError::decimal_out_of_range(i, src.plain(), *p, *s)
                    }),
                }
            })?;
            decimal_out(out, *p, *s)
        }
        _ => Err(KernelError::unsupported(format!(
            "CAST from TIMESTAMP to {to_name}"
        ))),
    }
}

/// A float result in the target width.
fn float_out(vals: &PrimitiveArray<Float64Type>, f32_target: bool) -> ArrayRef {
    if f32_target {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "values were rounded to f32 already"
        )]
        let f: PrimitiveArray<Float32Type> = vals.unary(|v| v as f32);
        Arc::new(f)
    } else {
        Arc::new(vals.clone())
    }
}

fn decimal_out(a: PrimitiveArray<Decimal128Type>, p: u8, s: i8) -> Result<ArrayRef, KernelError> {
    Ok(Arc::new(
        a.with_precision_and_scale(p, s)
            .map_err(|e| KernelError::internal(e.to_string()))?,
    ))
}

/// String → float, decimal, date or timestamp.
fn from_string<O: OffsetSizeTrait>(
    s: &GenericStringArray<O>,
    to: &DataType,
    opts: &CastOptions<'_>,
    to_name: &str,
) -> Result<ArrayRef, KernelError> {
    let mode = opts.mode;
    let invalid =
        |i: usize| KernelError::cast_invalid_input(i, sql_string(s.value(i)), "STRING", to_name);
    match to {
        DataType::Float32 | DataType::Float64 => {
            let f32_target = *to == DataType::Float32;
            let vals = map_rows::<Float64Type>(s, |i| {
                let b = s.value(i).as_bytes();
                let v = if f32_target {
                    text::parse_float::<f32>(b).map(f64::from)
                } else {
                    text::parse_float::<f64>(b)
                };
                match v {
                    Some(v) => Ok(Some(v)),
                    None => fail(mode, None, || invalid(i)),
                }
            })?;
            Ok(float_out(&vals, f32_target))
        }
        DataType::Decimal128(p, sc) => {
            let (p, sc) = (*p, *sc);
            let bound = i256::from_i128(10)
                .checked_pow(u32::from(p))
                .unwrap_or(i256::MAX);
            let out = map_rows::<Decimal128Type>(s, |i| {
                if let Some(u) = text::parse_decimal_fast(s.value(i).as_bytes(), sc.unsigned_abs())
                    && i256::from_i128(u) < bound
                    && i256::from_i128(u) > bound.wrapping_neg()
                {
                    return Ok(Some(u));
                }
                let Some(d) = text::parse_decimal(s.value(i)) else {
                    return fail(mode, None, || invalid(i));
                };
                if d.integral_digits() > 38 {
                    return fail(mode, None, || {
                        KernelError::numeric_out_of_supported_range(i, s.value(i).to_owned())
                    });
                }
                match d.to_scale(sc.unsigned_abs()).and_then(i256::to_i128) {
                    Some(u)
                        if i256::from_i128(u) < bound
                            && i256::from_i128(u) > bound.wrapping_neg() =>
                    {
                        Ok(Some(u))
                    }
                    _ => fail(mode, None, || {
                        KernelError::decimal_out_of_range(i, d.plain(), p, sc)
                    }),
                }
            })?;
            decimal_out(out, p, sc)
        }
        DataType::Date32 => {
            let out =
                map_rows::<Date32Type>(s, |i| match text::parse_date(s.value(i).as_bytes()) {
                    Some(d) => Ok(Some(d)),
                    None => fail(mode, None, || invalid(i)),
                })?;
            Ok(Arc::new(out))
        }
        DataType::Timestamp(TimeUnit::Microsecond, zone) => {
            let ntz = zone.is_none();
            let out = map_rows::<TimestampMicrosecondType>(s, |i| {
                use timestamp::{Parsed, Zone};
                match timestamp::parse(s.value(i).as_bytes()) {
                    // A region id resolves against the bundled database. An
                    // unknown one is malformed input, like a bad offset
                    // (`CAST_INVALID_INPUT` with ANSI on, NULL without):
                    // Spark's `stringToTimestamp` reads what the database
                    // cannot as no timestamp at all. A zone in the text is
                    // ignored for `TIMESTAMP_NTZ` once it resolves.
                    Parsed::At(local, Zone::Region(id)) => {
                        let Ok(zone) = crate::datetime::resolve_zone(&id) else {
                            return fail(mode, None, || invalid(i));
                        };
                        if ntz {
                            return Ok(Some(local));
                        }
                        match datetime::local_to_utc(local, &zone) {
                            Some(v) => Ok(Some(v)),
                            None => fail(mode, None, || invalid(i)),
                        }
                    }
                    // A zone in the text is ignored for TIMESTAMP_NTZ.
                    Parsed::At(local, _) if ntz => Ok(Some(local)),
                    Parsed::At(local, zone) => {
                        let utc = match zone {
                            Zone::Offset(o) => i64::from(o)
                                .checked_mul(1_000_000)
                                .and_then(|o| local.checked_sub(o)),
                            _ => datetime::local_to_utc(local, opts.time_zone),
                        };
                        match utc {
                            Some(v) => Ok(Some(v)),
                            None => fail(mode, None, || invalid(i)),
                        }
                    }
                    // Spark reads a time of day as today's, for TIMESTAMP only.
                    Parsed::NeedsClock if ntz => fail(mode, None, || invalid(i)),
                    Parsed::NeedsClock => Err(KernelError::unsupported(
                        "a timestamp string that depends on the query clock (a time of day)",
                    )),
                    Parsed::Malformed => fail(mode, None, || invalid(i)),
                }
            })?;
            Ok(Arc::new(out.with_timezone_opt(zone.clone())))
        }
        DataType::Boolean => {
            // Spark's `StringUtils.isTrueString` / `isFalseString` on the
            // trimmed, lower-cased text.
            let flags = map_rows::<Int8Type>(s, |i| {
                // Bytes only, as `UTF8String.trimAll`: a Unicode space stays.
                let t = text::trim_all(s.value(i).as_bytes());
                let is = |word: &str| t.eq_ignore_ascii_case(word.as_bytes());
                if ["t", "true", "y", "yes", "1"].iter().any(|w| is(w)) {
                    Ok(Some(1))
                } else if ["f", "false", "n", "no", "0"].iter().any(|w| is(w)) {
                    Ok(Some(0))
                } else {
                    fail(mode, None, || invalid(i))
                }
            })?;
            let out: BooleanArray = flags.iter().map(|v| v.map(|x| x != 0)).collect();
            Ok(Arc::new(out))
        }
        _ => Err(KernelError::unsupported(format!(
            "CAST from STRING to {to_name}"
        ))),
    }
}

/// `DATE` ↔ `TIMESTAMP` ↔ `TIMESTAMP_NTZ`; `None` for other targets.
fn temporal(
    input: &dyn Array,
    to: &DataType,
    opts: &CastOptions<'_>,
) -> Option<Result<ArrayRef, KernelError>> {
    use datetime::{MICROS_PER_DAY, local_to_utc, utc_to_local};
    let tz = opts.time_zone;
    let overflow = |i| KernelError::datetime_overflow(i, "cast to timestamp");
    let ts_out = |zone: &Option<Arc<str>>| {
        let zone = zone.clone();
        move |a: PrimitiveArray<TimestampMicrosecondType>| -> ArrayRef {
            Arc::new(a.with_timezone_opt(zone))
        }
    };
    Some(match (input.data_type(), to) {
        (DataType::Date32, DataType::Timestamp(TimeUnit::Microsecond, zone)) => {
            let d = input.as_primitive::<Date32Type>();
            map_rows::<TimestampMicrosecondType>(input, |i| {
                let local = i64::from(d.value(i)).checked_mul(MICROS_PER_DAY);
                let micros = if zone.is_some() {
                    local.and_then(|l| local_to_utc(l, tz))
                } else {
                    local
                };
                micros.map(Some).ok_or_else(|| overflow(i))
            })
            .map(ts_out(zone))
        }
        (DataType::Timestamp(TimeUnit::Microsecond, from_zone), DataType::Date32) => {
            let t = input.as_primitive::<TimestampMicrosecondType>();
            map_rows::<Date32Type>(input, |i| {
                let local = if from_zone.is_some() {
                    utc_to_local(t.value(i), tz)
                } else {
                    Some(t.value(i))
                };
                let days = local.ok_or_else(|| overflow(i))?.div_euclid(MICROS_PER_DAY);
                Ok(i32::try_from(days).ok())
            })
            .map(|a| Arc::new(a) as ArrayRef)
        }
        (
            DataType::Timestamp(TimeUnit::Microsecond, from_zone),
            DataType::Timestamp(TimeUnit::Microsecond, to_zone),
        ) => {
            let t = input.as_primitive::<TimestampMicrosecondType>();
            map_rows::<TimestampMicrosecondType>(input, |i| {
                let v = t.value(i);
                let out = match (from_zone.is_some(), to_zone.is_some()) {
                    (true, false) => utc_to_local(v, tz),
                    (false, true) => local_to_utc(v, tz),
                    _ => Some(v),
                };
                out.map(Some).ok_or_else(|| overflow(i))
            })
            .map(ts_out(to_zone))
        }
        _ => return None,
    })
}

/// Any covered type → `STRING`.
fn to_string(input: &dyn Array, opts: &CastOptions<'_>) -> Result<StringArray, KernelError> {
    use std::fmt::Write as _;
    let unsupported = || {
        KernelError::unsupported(format!(
            "CAST from {} to STRING",
            spark_type_name(input.data_type())
        ))
    };
    let mut b = arrow_array::builder::StringBuilder::with_capacity(
        input.len(),
        input.len().saturating_mul(8),
    );
    let mut out = String::new();
    let nums = NumArray::new(input);
    for i in 0..input.len() {
        if input.is_null(i) {
            b.append_null();
            continue;
        }
        out.clear();
        match (input.data_type(), nums) {
            (DataType::Boolean, _) => {
                let _ = write!(out, "{}", input.as_boolean().value(i));
            }
            (_, Some(n)) => match n.get(i) {
                Num::Int(v) => {
                    let _ = write!(out, "{v}");
                }
                // ANSI and try_cast render plainly; legacy mode uses Java's
                // `BigDecimal.toString` (`0E-37`, `1.00E-7`).
                Num::Dec(v, s) if opts.mode == CastMode::Legacy => {
                    write_decimal_java(v, s, &mut out);
                }
                Num::Dec(v, s) => write_decimal_plain(v, s, &mut out),
                Num::Float(v, is_f32) => out.push_str(&float_string(v, is_f32)),
            },
            (DataType::Date32, _) => {
                datetime::format_date(
                    i64::from(input.as_primitive::<Date32Type>().value(i)),
                    &mut out,
                );
            }
            (DataType::Timestamp(TimeUnit::Microsecond, zone), _) => {
                let v = input.as_primitive::<TimestampMicrosecondType>().value(i);
                let local = if zone.is_some() {
                    datetime::utc_to_local(v, opts.time_zone)
                } else {
                    Some(v)
                };
                datetime::format_timestamp(local.unwrap_or(v), &mut out);
            }
            _ => return Err(unsupported()),
        }
        b.append_value(&out);
    }
    Ok(b.finish())
}
