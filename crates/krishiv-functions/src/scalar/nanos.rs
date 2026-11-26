// SPDX-License-Identifier: Apache-2.0
//! Nanosecond timestamps (Spark 4.3 `TIMESTAMP_LTZ(7..=9)` / `TIMESTAMP_NTZ(7..=9)`, Arrow
//! `Timestamp(Nanosecond)`) reaching the datetime bindings.
//!
//! Spark stores such a value as its microseconds and the nanoseconds within the microsecond, and the
//! functions below read only the microseconds: the fields (`hour`, `minute`, the `extract` fields but
//! `SECOND`), `unix_seconds`/`unix_millis`/`unix_micros`, `unix_timestamp` and `date_trunc` (to a
//! microsecond or a coarser unit; `NANOSECOND` is not a unit, so NULL). They bind to the microsecond
//! kernels over the floor of the value; `date_trunc` gives its result back as a nanosecond timestamp of
//! the argument's kind, and `extract(SECOND ..)` adds the nanoseconds to the microsecond seconds as a
//! `decimal(11, 9)`.
//!
//! It is also the family of Spark 4.3's `unix_nanos` (a nanosecond timestamp's nanoseconds since the
//! epoch, a `DECIMAL(21, 0)`) and `timestamp_nanos` (an integral or decimal count, floored, to a
//! `TIMESTAMP_LTZ(9)`). The carrier is Arrow's 64-bit nanoseconds, so a count outside 1677-09-21 ..
//! 2262-04-11 is `DATETIME_OVERFLOW`, where Spark's range reaches year 9999.
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal32Type, Decimal64Type, Decimal128Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType, TimestampNanosecondType,
};
use arrow_array::{Array, ArrayRef, Decimal128Array, TimestampNanosecondArray};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::{DatetimeOverflow, KrishivError};

use super::args::Inputs;
use super::{
    Arg, BindArg, BindOptions, Binding, FamilyResult, ScalarFunction, bind_scalar, internal,
};

/// The family's own functions (Spark 4.3).
pub(super) const NAMES: &[&str] = &["unix_nanos", "timestamp_nanos"];

/// `unix_nanos` over a nanosecond timestamp; `timestamp_nanos` over an integral or decimal count.
pub(super) fn bind_family(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = NAMES.iter().copied().find(|n| *n == name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("unix_nanos", [DataType::Timestamp(TimeUnit::Nanosecond, _)]) => Ok(Binding::new(
            DataType::Decimal128(21, 0),
            |x: &Inputs<'_>| unix_nanos(&x.column(0)?),
        )),
        ("timestamp_nanos", [t]) if count_scale(t).is_some() => {
            let scale = count_scale(t).unwrap_or_default();
            Ok(Binding::new(
                DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("UTC"))),
                move |x: &Inputs<'_>| timestamp_nanos(&x.column(0)?, scale),
            ))
        }
        (_, types) => Err(internal(format!("{listed} over {types:?}"))),
    };
    Some((listed, binding))
}

/// The scale of a count `timestamp_nanos` takes: 0 for an integral type, a decimal's own.
fn count_scale(t: &DataType) -> Option<i8> {
    match t {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => Some(0),
        DataType::Decimal32(_, s) | DataType::Decimal64(_, s) | DataType::Decimal128(_, s) => {
            Some(*s)
        }
        _ => None,
    }
}

/// The nanoseconds since the epoch of each value, as a `DECIMAL(21, 0)`.
fn unix_nanos(a: &ArrayRef) -> Result<ArrayRef, KrishivError> {
    let out: Decimal128Array = a
        .as_primitive::<TimestampNanosecondType>()
        .iter()
        .map(|v| v.map(i128::from))
        .collect();
    Ok(Arc::new(
        out.with_precision_and_scale(21, 0)
            .map_err(|e| internal(e.to_string()))?,
    ))
}

/// The unscaled values of an integral or decimal column, widened.
fn unscaled(a: &ArrayRef) -> Result<Vec<Option<i128>>, KrishivError> {
    Ok(match a.data_type() {
        DataType::Int8 => a
            .as_primitive::<Int8Type>()
            .iter()
            .map(|v| v.map(i128::from))
            .collect(),
        DataType::Int16 => a
            .as_primitive::<Int16Type>()
            .iter()
            .map(|v| v.map(i128::from))
            .collect(),
        DataType::Int32 => a
            .as_primitive::<Int32Type>()
            .iter()
            .map(|v| v.map(i128::from))
            .collect(),
        DataType::Int64 => a
            .as_primitive::<Int64Type>()
            .iter()
            .map(|v| v.map(i128::from))
            .collect(),
        DataType::Decimal32(..) => a
            .as_primitive::<Decimal32Type>()
            .iter()
            .map(|v| v.map(i128::from))
            .collect(),
        DataType::Decimal64(..) => a
            .as_primitive::<Decimal64Type>()
            .iter()
            .map(|v| v.map(i128::from))
            .collect(),
        DataType::Decimal128(..) => a.as_primitive::<Decimal128Type>().iter().collect(),
        t => return Err(internal(format!("timestamp_nanos over {t}"))),
    })
}

/// A count of nanoseconds (at `scale`, floored to a whole nanosecond, as Spark's `RoundingMode.FLOOR`) as a
/// `TIMESTAMP_LTZ(9)`; a count the carrier cannot hold is `DATETIME_OVERFLOW`.
fn timestamp_nanos(a: &ArrayRef, scale: i8) -> Result<ArrayRef, KrishivError> {
    // Spark's decimals have a scale of 0..=38, so the divisor is at most 10^38 (an `i128`).
    let divisor = u32::try_from(scale)
        .ok()
        .and_then(|s| 10_i128.checked_pow(s))
        .ok_or_else(|| internal(format!("timestamp_nanos at scale {scale}")))?;
    let mut out = Vec::with_capacity(a.len());
    for v in unscaled(a)? {
        let Some(v) = v else {
            out.push(None);
            continue;
        };
        let whole = v.div_euclid(divisor);
        let Ok(n) = i64::try_from(whole) else {
            return Err(DatetimeOverflow {
                operation: format!(
                    "create a TIMESTAMP_LTZ(9) from {whole} nanoseconds since the epoch"
                ),
            }
            .into());
        };
        out.push(Some(n));
    }
    Ok(Arc::new(
        TimestampNanosecondArray::from(out).with_timezone("UTC"),
    ))
}

/// Whether `t` is a nanosecond timestamp carrier.
pub(super) const fn is_nanos(t: &DataType) -> bool {
    matches!(t, DataType::Timestamp(TimeUnit::Nanosecond, _))
}

/// The arguments with the first, a nanosecond timestamp, as a column of its microseconds (floor), and the
/// nanoseconds within the microsecond of each row (`date_format` prints them).
pub(super) fn split_first(x: &Inputs<'_>) -> Result<(Vec<Arg>, Vec<u16>), KrishivError> {
    let first = x.column(0)?;
    let DataType::Timestamp(TimeUnit::Nanosecond, zone) = first.data_type() else {
        return Err(internal(format!(
            "a nanosecond timestamp of type {}",
            first.data_type()
        )));
    };
    let rest = first
        .as_primitive::<TimestampNanosecondType>()
        .values()
        .iter()
        .map(|v| u16::try_from(v.rem_euclid(NANOS_PER_MICRO)).unwrap_or_default())
        .collect();
    let mut args = vec![Arg::Array(floor(&first, &zone.clone()))];
    for i in 1..x.arity() {
        let (a, literal) = x.operand(i)?;
        args.push(if literal {
            Arg::Scalar(a)
        } else {
            Arg::Array(a)
        });
    }
    Ok((args, rest))
}

/// The functions that read a nanosecond timestamp's microseconds.
const FLOORED: &[&str] = &[
    "hour",
    "minute",
    "second",
    "unix_seconds",
    "unix_millis",
    "unix_micros",
    "unix_timestamp",
    "to_unix_timestamp",
    "date_trunc",
    "extract",
    "date_part",
    "datepart",
];

const NANOS_PER_MICRO: i64 = 1_000;

/// The zone of `t` when it is a nanosecond timestamp.
fn nanos_zone(t: &DataType) -> Option<Option<Arc<str>>> {
    match t {
        DataType::Timestamp(TimeUnit::Nanosecond, zone) => Some(zone.clone()),
        _ => None,
    }
}

/// The microseconds (floor) of a nanosecond timestamp array.
fn floor(a: &ArrayRef, zone: &Option<Arc<str>>) -> ArrayRef {
    Arc::new(
        a.as_primitive::<TimestampNanosecondType>()
            .unary::<_, TimestampMicrosecondType>(|v| v.div_euclid(NANOS_PER_MICRO))
            .with_timezone_opt(zone.clone()),
    )
}

/// What becomes of the microsecond binding's result.
#[derive(Debug, Clone)]
enum Widen {
    /// It is the answer.
    None,
    /// A microsecond timestamp, given back in nanoseconds (`date_trunc`).
    Timestamp(Option<Arc<str>>),
    /// The seconds at scale 6 of argument `arg`, completed with its nanoseconds at scale 9.
    Seconds(usize),
}

/// A binding over the floor of its nanosecond timestamp arguments.
#[derive(Debug)]
struct Floored {
    inner: Arc<dyn ScalarFunction>,
    /// Per argument, the zone of a nanosecond timestamp argument that is floored.
    floored: Vec<Option<Option<Arc<str>>>>,
    widen: Widen,
    return_type: DataType,
}

/// `name` over `args` when it reads only the microseconds of a nanosecond timestamp argument; `None`
/// when no argument is one or `name` is not such a function.
pub(super) fn bind(
    name: &str,
    args: &[BindArg],
    opts: &BindOptions,
) -> Option<Result<Arc<dyn ScalarFunction>, KrishivError>> {
    let mut floored: Vec<_> = args.iter().map(|a| nanos_zone(&a.data_type())).collect();
    if floored.iter().all(Option::is_none) || !FLOORED.contains(&name) {
        return None;
    }
    // `date_trunc` of a `TIMESTAMP_NTZ(p)` truncates its wall clock and stays zone-less (Spark 4.3): the
    // microsecond kernel reads it as an instant in UTC, and the result is read back the same way.
    let ntz_trunc = name == "date_trunc" && floored.iter().flatten().any(Option::is_none);
    let utc;
    let opts = if ntz_trunc {
        for zone in floored.iter_mut().flatten() {
            *zone = Some(Arc::from("UTC"));
        }
        utc = opts
            .clone()
            .with_time_zone(Arc::new(krishiv_kernels::cast::FixedOffset(0)))
            .with_time_zone_name("UTC");
        &utc
    } else {
        opts
    };
    let micros_args: Vec<BindArg> = args
        .iter()
        .zip(&floored)
        .map(|(a, zone)| match (a, zone) {
            (BindArg::Column { nullable, .. }, Some(zone)) => BindArg::Column {
                ty: DataType::Timestamp(TimeUnit::Microsecond, zone.clone()),
                nullable: *nullable,
            },
            (BindArg::Literal(l), Some(zone)) => BindArg::Literal(floor(l, zone)),
            (a, None) => a.clone(),
        })
        .collect();
    Some(bind_scalar(name, &micros_args, opts).and_then(|inner| {
        let (widen, return_type) = match inner.return_type() {
            DataType::Timestamp(TimeUnit::Microsecond, zone) if name == "date_trunc" => {
                let zone = if ntz_trunc { None } else { zone.clone() };
                (
                    Widen::Timestamp(zone.clone()),
                    DataType::Timestamp(TimeUnit::Nanosecond, zone),
                )
            }
            DataType::Decimal64(_, 6) | DataType::Decimal128(_, 6)
                if matches!(name, "extract" | "date_part" | "datepart") =>
            {
                let arg = floored
                    .iter()
                    .position(Option::is_some)
                    .ok_or_else(|| internal(format!("{name}: no nanosecond argument")))?;
                (Widen::Seconds(arg), DataType::Decimal128(11, 9))
            }
            t => (Widen::None, t.clone()),
        };
        Ok(Arc::new(Floored {
            inner,
            floored,
            widen,
            return_type,
        }) as Arc<dyn ScalarFunction>)
    }))
}

impl Floored {
    fn run(&self, args: &[Arg], rows: usize, partition: u32) -> Result<ArrayRef, KrishivError> {
        let micros: Vec<Arg> = args
            .iter()
            .zip(&self.floored)
            .map(|(a, zone)| match (a, zone) {
                (Arg::Array(x), Some(zone)) => Arg::Array(floor(x, zone)),
                (Arg::Scalar(x), Some(zone)) => Arg::Scalar(floor(x, zone)),
                (a, None) => a.clone(),
            })
            .collect();
        let out = self.inner.invoke_in(&micros, rows, partition)?;
        Ok(match &self.widen {
            Widen::None => out,
            Widen::Timestamp(zone) => Arc::new(
                out.as_primitive::<TimestampMicrosecondType>()
                    .unary::<_, TimestampNanosecondType>(|v| v.saturating_mul(NANOS_PER_MICRO))
                    .with_timezone_opt(zone.clone()),
            ),
            Widen::Seconds(i) => {
                let source = args
                    .get(*i)
                    .ok_or_else(|| internal("extract: a missing argument".to_owned()))?;
                seconds(&out, source)?
            }
        })
    }
}

/// The seconds `micro_seconds` (scale 6) with the nanoseconds within the microsecond of `source`'s
/// values, at scale 9.
fn seconds(micro_seconds: &ArrayRef, source: &Arg) -> Result<ArrayRef, KrishivError> {
    let (values, one) = match source {
        Arg::Array(a) => (a.as_primitive::<TimestampNanosecondType>(), false),
        Arg::Scalar(a) => (a.as_primitive::<TimestampNanosecondType>(), true),
    };
    let rest = |row: usize| {
        let v = values.values().get(if one { 0 } else { row }).copied();
        i128::from(v.unwrap_or_default().rem_euclid(NANOS_PER_MICRO))
    };
    let wide: Vec<Option<i128>> = match micro_seconds.data_type() {
        DataType::Decimal64(..) => micro_seconds
            .as_primitive::<Decimal64Type>()
            .iter()
            .enumerate()
            .map(|(row, v)| v.map(|v| i128::from(v) * 1_000 + rest(row)))
            .collect(),
        DataType::Decimal128(..) => micro_seconds
            .as_primitive::<Decimal128Type>()
            .iter()
            .enumerate()
            .map(|(row, v)| v.map(|v| v * 1_000 + rest(row)))
            .collect(),
        t => return Err(internal(format!("extract: seconds of type {t}"))),
    };
    Ok(Arc::new(
        Decimal128Array::from(wide)
            .with_precision_and_scale(11, 9)
            .map_err(|e| internal(e.to_string()))?,
    ))
}

impl ScalarFunction for Floored {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn return_type(&self) -> &DataType {
        &self.return_type
    }

    fn invoke(&self, args: &[Arg], rows: usize) -> Result<ArrayRef, KrishivError> {
        self.run(args, rows, 0)
    }

    fn invoke_in(
        &self,
        args: &[Arg],
        rows: usize,
        partition: u32,
    ) -> Result<ArrayRef, KrishivError> {
        self.run(args, rows, partition)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int32Array, StringArray, TimestampNanosecondArray};

    use super::*;

    /// `2020-01-01 13:24:35.123456789` and `1960-01-01 13:24:35.000000001`, without a zone.
    fn stamps() -> ArrayRef {
        Arc::new(TimestampNanosecondArray::from(vec![
            Some(1_577_885_075_123_456_789),
            Some(-315_570_925_000_000_000 + 1),
            None,
        ]))
    }

    fn call(name: &str, args: &[Arg], types: &[BindArg]) -> ArrayRef {
        let f = bind_scalar(name, types, &BindOptions::new()).unwrap();
        f.invoke(args, 3).unwrap()
    }

    fn column(a: &ArrayRef) -> BindArg {
        BindArg::Column {
            ty: a.data_type().clone(),
            nullable: true,
        }
    }

    #[test]
    fn unix_nanos_and_timestamp_nanos_round_trip() {
        let ts: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![
            Some(-1_i64),
            None,
            Some(i64::MAX),
        ]));
        let n = unix_nanos(&ts).unwrap();
        let n = n.as_primitive::<Decimal128Type>();
        assert_eq!(n.data_type(), &DataType::Decimal128(21, 0));
        assert_eq!(
            n.iter().collect::<Vec<_>>(),
            vec![Some(-1), None, Some(i128::from(i64::MAX))]
        );
        let small: ArrayRef = Arc::new(Int32Array::from(vec![Some(-7), None]));
        let back = timestamp_nanos(&small, 0).unwrap();
        let back = back.as_primitive::<TimestampNanosecondType>();
        assert_eq!(back.iter().collect::<Vec<_>>(), vec![Some(-7), None]);
        // 1.9 and -1.1 at scale 1 floor to 1 and -2; a Decimal64 carrier reads the same.
        let dec: ArrayRef = Arc::new(
            arrow_array::Decimal64Array::from(vec![Some(19_i64), Some(-11)])
                .with_precision_and_scale(2, 1)
                .unwrap(),
        );
        let floored = timestamp_nanos(&dec, 1).unwrap();
        assert_eq!(
            floored
                .as_primitive::<TimestampNanosecondType>()
                .values()
                .to_vec(),
            vec![1, -2]
        );
        let huge: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(i128::from(i64::MAX) * 10 + 15)])
                .with_precision_and_scale(38, 1)
                .unwrap(),
        );
        let e = timestamp_nanos(&huge, 1).unwrap_err();
        assert!(e.to_string().contains("DATETIME_OVERFLOW"), "{e}");
        assert!(
            e.to_string()
                .contains("from 9223372036854775808 nanoseconds"),
            "{e}"
        );
        for t in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int64,
            DataType::Decimal32(3, 2),
        ] {
            assert!(count_scale(&t).is_some(), "{t}");
        }
        assert_eq!(count_scale(&DataType::Float64), None);
    }

    #[test]
    fn fields_read_the_microseconds() {
        let a = stamps();
        let got = call("hour", &[Arg::Array(Arc::clone(&a))], &[column(&a)]);
        assert_eq!(
            got.as_ref(),
            &Int32Array::from(vec![Some(13), Some(13), None]) as &dyn Array
        );
        let got = call("second", &[Arg::Array(Arc::clone(&a))], &[column(&a)]);
        assert_eq!(
            got.as_ref(),
            &Int32Array::from(vec![Some(35), Some(35), None]) as &dyn Array
        );
    }

    #[test]
    fn extract_second_keeps_nine_digits() {
        let a = stamps();
        let field: ArrayRef = Arc::new(StringArray::from(vec!["SECOND"]));
        let got = call(
            "extract",
            &[Arg::Scalar(Arc::clone(&field)), Arg::Array(Arc::clone(&a))],
            &[BindArg::Literal(field), column(&a)],
        );
        assert_eq!(got.data_type(), &DataType::Decimal128(11, 9));
        let got = got.as_primitive::<Decimal128Type>();
        assert_eq!(got.values()[..2], [35_123_456_789, 35_000_000_001]);
        assert!(got.is_null(2));
    }

    #[test]
    fn date_trunc_gives_a_nanosecond_timestamp_back() {
        let a = stamps();
        let unit: ArrayRef = Arc::new(StringArray::from(vec!["SECOND"]));
        let got = call(
            "date_trunc",
            &[Arg::Scalar(Arc::clone(&unit)), Arg::Array(Arc::clone(&a))],
            &[BindArg::Literal(unit), column(&a)],
        );
        assert_eq!(got.data_type(), a.data_type());
        assert_eq!(
            got.as_primitive::<TimestampNanosecondType>().values()[..2],
            [1_577_885_075_000_000_000, -315_570_925_000_000_000]
        );
    }
}
