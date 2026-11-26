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
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal64Type, Decimal128Type, TimestampMicrosecondType, TimestampNanosecondType,
};
use arrow_array::{Array, ArrayRef, Decimal128Array};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::KrishivError;

use super::{Arg, BindArg, BindOptions, ScalarFunction, bind_scalar, internal};

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
