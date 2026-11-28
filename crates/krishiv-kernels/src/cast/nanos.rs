// SPDX-License-Identifier: Apache-2.0
//! Casts to and from the nanosecond timestamps: Spark 4.3's `TIMESTAMP_LTZ(p)` / `TIMESTAMP_NTZ(p)`,
//! `7 <= p <= 9`, on Arrow `Timestamp(Nanosecond[, zone])`.
//!
//! A value is its microseconds (the floor) and the nanoseconds within that microsecond (`0..1000`),
//! as Spark stores it. Every conversion runs on the microseconds through the microsecond cast and
//! carries the remainder: a zone shift moves whole seconds, a date or a number has none, a string
//! keeps its fraction's digits 7 to 9. Truncation to the target's `p` is the caller's (the
//! precision is not in the Arrow type). The carrier holds 1677-09-21 to 2262-04-11; a value past it
//! is `DATETIME_OVERFLOW` (NULL under `try_cast`), where Spark's range reaches years 1 and 9999.
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Time64NanosecondType, TimestampMicrosecondType, TimestampNanosecondType};
use arrow_array::{Array, ArrayRef, PrimitiveArray};
use arrow_schema::{DataType, TimeUnit};

use super::{CastMode, CastOptions, cast, timestamp};
use crate::KernelError;

const NANOS_PER_MICRO: i64 = 1_000;

/// `input` as microseconds. Under `try_cast` a row whose intermediate conversion overflows (a `DATE`
/// near `i32::MAX` days) is NULL, not an error of the whole batch.
fn to_micros(
    input: &dyn Array,
    micros_type: &DataType,
    opts: &CastOptions<'_>,
) -> Result<ArrayRef, KernelError> {
    match cast(input, micros_type, opts) {
        Err(e) if opts.mode == CastMode::Try && is_overflow(&e) => {
            let mut parts: Vec<ArrayRef> = Vec::with_capacity(input.len());
            for i in 0..input.len() {
                let one = input.slice(i, 1);
                parts.push(match cast(&one, micros_type, opts) {
                    Ok(v) => v,
                    Err(e) if is_overflow(&e) => arrow_array::new_null_array(micros_type, 1),
                    Err(e) => return Err(e),
                });
            }
            let refs: Vec<&dyn Array> = parts.iter().map(AsRef::as_ref).collect();
            arrow_select::concat::concat(&refs).map_err(|e| KernelError::internal(e.to_string()))
        }
        other => other,
    }
}

fn is_overflow(e: &KernelError) -> bool {
    e.error_class() == krishiv_common::error::ErrorClass::DatetimeOverflow
}

/// The cast when `input` or `to` is a nanosecond timestamp, else `None`.
pub(super) fn cast_nanos(
    input: &dyn Array,
    to: &DataType,
    opts: &CastOptions<'_>,
) -> Option<Result<ArrayRef, KernelError>> {
    let from = input.data_type();
    if let DataType::Timestamp(TimeUnit::Nanosecond, from_zone) = from {
        let nanos = input.as_primitive::<TimestampNanosecondType>();
        return Some(match to {
            DataType::Timestamp(TimeUnit::Nanosecond, to_zone) => {
                if from_zone.is_some() == to_zone.is_some() {
                    Ok(Arc::new(nanos.clone().with_timezone_opt(to_zone.clone())) as ArrayRef)
                } else {
                    let rest = sub_micros(nanos);
                    shifted(nanos, from_zone, to_zone, opts).and_then(|micros| {
                        combine(
                            &micros,
                            |i| rest.get(i).copied().unwrap_or(0),
                            to_zone,
                            opts.mode,
                        )
                    })
                }
            }
            DataType::Utf8 => Ok(format(nanos, from_zone.is_some(), opts)),
            DataType::Time64(TimeUnit::Nanosecond) => {
                // The microsecond kernel floors; the within-microsecond digits are added back (a time of
                // day is nanoseconds since midnight, so the sum stays inside the day).
                let rest = sub_micros(nanos);
                cast(&micros_of(nanos, from_zone), to, opts).map(|t| {
                    let t = t.as_primitive::<Time64NanosecondType>();
                    let out: PrimitiveArray<Time64NanosecondType> = (0..t.len())
                        .map(|i| {
                            t.is_valid(i)
                                .then(|| t.value(i).saturating_add(i64::from(rest[i])))
                        })
                        .collect();
                    Arc::new(out) as ArrayRef
                })
            }
            _ => cast(&micros_of(nanos, from_zone), to, opts),
        });
    }
    let DataType::Timestamp(TimeUnit::Nanosecond, to_zone) = to else {
        return None;
    };
    let micros_type = DataType::Timestamp(TimeUnit::Microsecond, to_zone.clone());
    Some(to_micros(input, &micros_type, opts).and_then(|micros| {
        let micros = micros.as_primitive::<TimestampMicrosecondType>();
        match from {
            DataType::Utf8 => {
                let s = input.as_string::<i32>();
                combine(micros, |i| string_rest(s.value(i)), to_zone, opts.mode)
            }
            DataType::LargeUtf8 => {
                let s = input.as_string::<i64>();
                combine(micros, |i| string_rest(s.value(i)), to_zone, opts.mode)
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                let t = input.as_primitive::<Time64NanosecondType>();
                combine(
                    micros,
                    |i| u16::try_from(t.value(i).rem_euclid(NANOS_PER_MICRO)).unwrap_or(0),
                    to_zone,
                    opts.mode,
                )
            }
            _ => combine(micros, |_| 0, to_zone, opts.mode),
        }
    }))
}

/// The nanoseconds within the microsecond a timestamp string names (its fraction's digits 7 to 9).
fn string_rest(s: &str) -> u16 {
    timestamp::parse_nanos(s.as_bytes()).1
}

/// The microseconds (floor) of each value, as a microsecond timestamp in `zone`.
fn micros_of(
    nanos: &PrimitiveArray<TimestampNanosecondType>,
    zone: &Option<Arc<str>>,
) -> PrimitiveArray<TimestampMicrosecondType> {
    nanos
        .unary::<_, TimestampMicrosecondType>(|v| v.div_euclid(NANOS_PER_MICRO))
        .with_timezone_opt(zone.clone())
}

/// The nanoseconds within the microsecond of each value.
fn sub_micros(nanos: &PrimitiveArray<TimestampNanosecondType>) -> Vec<u16> {
    nanos
        .values()
        .iter()
        .map(|v| u16::try_from(v.rem_euclid(NANOS_PER_MICRO)).unwrap_or(0))
        .collect()
}

/// The microseconds of `nanos` moved from `from` to `to` (`TIMESTAMP_LTZ` and `TIMESTAMP_NTZ` differ
/// by the session zone's offset).
fn shifted(
    nanos: &PrimitiveArray<TimestampNanosecondType>,
    from: &Option<Arc<str>>,
    to: &Option<Arc<str>>,
    opts: &CastOptions<'_>,
) -> Result<ArrayRef, KernelError> {
    cast(
        &micros_of(nanos, from),
        &DataType::Timestamp(TimeUnit::Microsecond, to.clone()),
        opts,
    )
}

/// `micros * 1000 + rest(i)` per row, a nanosecond timestamp in `zone`; past the carrier's range
/// `DATETIME_OVERFLOW`, or NULL under `try_cast`.
fn combine(
    micros: &dyn Array,
    rest: impl Fn(usize) -> u16,
    zone: &Option<Arc<str>>,
    mode: CastMode,
) -> Result<ArrayRef, KernelError> {
    let micros = micros.as_primitive::<TimestampMicrosecondType>();
    let mut out = Vec::with_capacity(micros.len());
    for i in 0..micros.len() {
        if micros.is_null(i) {
            out.push(None);
            continue;
        }
        // The product alone can pass the carrier while the sum fits (`i64::MIN` is -9223372036854776 µs
        // and 192 ns), so the sum is built wide and narrowed once.
        let v = i64::try_from(
            i128::from(micros.value(i))
                .saturating_mul(i128::from(NANOS_PER_MICRO))
                .saturating_add(i128::from(rest(i))),
        )
        .ok();
        match (v, mode) {
            (Some(v), _) => out.push(Some(v)),
            (None, CastMode::Try) => out.push(None),
            (None, _) => {
                return Err(KernelError::datetime_overflow(
                    i,
                    "cast to a nanosecond timestamp",
                ));
            }
        }
    }
    Ok(Arc::new(
        PrimitiveArray::<TimestampNanosecondType>::from(out).with_timezone_opt(zone.clone()),
    ))
}

/// Each value as Spark prints it: local date-time (in the session zone for `TIMESTAMP_LTZ`), the
/// fraction up to nine digits without trailing zeros.
fn format(
    nanos: &PrimitiveArray<TimestampNanosecondType>,
    zoned: bool,
    opts: &CastOptions<'_>,
) -> ArrayRef {
    let mut b = arrow_array::builder::StringBuilder::with_capacity(
        nanos.len(),
        nanos.len().saturating_mul(29),
    );
    let mut text = String::new();
    let tz = zoned.then_some(opts.time_zone);
    for v in nanos {
        match v {
            Some(v) => {
                text.clear();
                super::datetime::format_timestamp_nanos_in_zone(v, tz, &mut text);
                b.append_value(&text);
            }
            None => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

#[cfg(test)]
mod tests {
    use arrow_array::{Date32Array, StringArray, TimestampNanosecondArray};

    use super::*;
    use crate::cast::FixedOffset;

    const NTZ: DataType = DataType::Timestamp(TimeUnit::Nanosecond, None);

    fn ltz() -> DataType {
        DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("UTC")))
    }

    fn opts(mode: CastMode, zone: &FixedOffset) -> CastOptions<'_> {
        CastOptions {
            mode,
            time_zone: zone,
        }
    }

    fn strings(a: &ArrayRef) -> Vec<Option<String>> {
        a.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect()
    }

    #[test]
    fn the_carrier_minimum_and_time_digits_survive_casts() {
        let utc = FixedOffset(0);
        let o = opts(CastMode::Ansi, &utc);
        // -9223372036854776 µs and 192 ns: the product alone overflows, the sum is i64::MIN.
        let s = StringArray::from(vec![Some("1677-09-21 00:12:43.145224192")]);
        let v = cast(&s, &NTZ, &o).unwrap();
        assert_eq!(
            v.as_primitive::<TimestampNanosecondType>().value(0),
            i64::MIN
        );
        // A clock-only string keeps digits 7 to 9 (the clock itself comes from the caller).
        assert_eq!(timestamp::parse_nanos(b"10:00:00.123456789").1, 789);
        // TIME <-> nanosecond timestamp keep them both ways.
        let ts =
            TimestampNanosecondArray::from(vec![Some(86_400_000_000_000 + 43_200_123_456_789)]);
        let back = cast(&ts, &DataType::Time64(TimeUnit::Nanosecond), &o).unwrap();
        assert_eq!(
            back.as_primitive::<Time64NanosecondType>().value(0),
            43_200_123_456_789
        );
        // try_cast of a DATE past the range is NULL for that row only.
        let d = Date32Array::from(vec![Some(i32::MAX), Some(0)]);
        let v = cast(&d, &NTZ, &opts(CastMode::Try, &utc)).unwrap();
        assert!(v.is_null(0) && v.is_valid(1));
    }

    #[test]
    fn strings_keep_nine_digits_and_print_them_back() {
        let utc = FixedOffset(0);
        let o = opts(CastMode::Ansi, &utc);
        let s = StringArray::from(vec![
            Some("2020-01-01 00:00:00.123456789"),
            Some("1960-01-01 00:00:00.000000001"),
            Some("2020-01-01 00:00:00.9999990001"),
            Some("2020-01-01"),
            None,
        ]);
        let ns = cast(&s, &NTZ, &o).unwrap();
        let v = ns.as_primitive::<TimestampNanosecondType>();
        assert_eq!(v.value(0), 1_577_836_800_123_456_789);
        assert_eq!(v.value(1), -315_619_200_000_000_000 + 1);
        assert!(v.is_null(4));
        let back = cast(ns.as_ref(), &DataType::Utf8, &o).unwrap();
        assert_eq!(
            strings(&back),
            vec![
                Some("2020-01-01 00:00:00.123456789".to_owned()),
                Some("1960-01-01 00:00:00.000000001".to_owned()),
                Some("2020-01-01 00:00:00.999999".to_owned()),
                Some("2020-01-01 00:00:00".to_owned()),
                None
            ]
        );
    }

    #[test]
    fn dates_and_microseconds_take_the_floor() {
        let utc = FixedOffset(0);
        let o = opts(CastMode::Ansi, &utc);
        // 1960-01-01 00:00:00.000000001 and 1959-12-31 23:59:59.999999999
        let ns = TimestampNanosecondArray::from(vec![
            -315_619_200_000_000_000 + 1,
            -315_619_200_000_000_000 - 1,
        ]);
        let d = cast(&ns, &DataType::Date32, &o).unwrap();
        assert_eq!(
            d.as_primitive::<arrow_array::types::Date32Type>().values(),
            &[-3653, -3654]
        );
        let us = cast(&ns, &DataType::Timestamp(TimeUnit::Microsecond, None), &o).unwrap();
        assert_eq!(
            us.as_primitive::<TimestampMicrosecondType>().values(),
            &[-315_619_200_000_000, -315_619_200_000_001]
        );
        let from_date = cast(&Date32Array::from(vec![18_262]), &NTZ, &o).unwrap();
        assert_eq!(
            from_date.as_primitive::<TimestampNanosecondType>().value(0),
            1_577_836_800_000_000_000
        );
    }

    #[test]
    fn zone_changes_keep_the_nanoseconds() {
        let plus_one = FixedOffset(3600);
        let o = opts(CastMode::Ansi, &plus_one);
        let ntz = TimestampNanosecondArray::from(vec![1_577_836_800_123_456_789]);
        let l = cast(&ntz, &ltz(), &o).unwrap();
        assert_eq!(
            l.as_primitive::<TimestampNanosecondType>().value(0),
            1_577_836_800_123_456_789 - 3_600_000_000_000
        );
        let back = cast(l.as_ref(), &NTZ, &o).unwrap();
        assert_eq!(
            back.as_primitive::<TimestampNanosecondType>().value(0),
            1_577_836_800_123_456_789
        );
        assert_eq!(
            strings(&cast(l.as_ref(), &DataType::Utf8, &o).unwrap()),
            vec![Some("2020-01-01 00:00:00.123456789".to_owned())]
        );
    }

    #[test]
    fn values_past_the_carrier_overflow_or_are_null_under_try() {
        let utc = FixedOffset(0);
        let year_one = Date32Array::from(vec![-719_162]);
        let err = cast(&year_one, &NTZ, &opts(CastMode::Ansi, &utc)).unwrap_err();
        assert_eq!(err.row(), Some(0));
        let t = cast(&year_one, &NTZ, &opts(CastMode::Try, &utc)).unwrap();
        assert!(t.is_null(0));
    }
}
