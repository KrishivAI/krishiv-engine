// SPDX-License-Identifier: Apache-2.0
//! Spark 4.3's casts between `TIME` and the timestamps (`Cast.castToTime`,
//! `castToTimestamp[NTZ]`), with values from the Spark 4.3.0 `cast.sql`
//! goldens and from `SparkDateTimeUtils` at tag v4.3.0.

#![expect(clippy::unwrap_used, reason = "test helpers fail loudly")]

use arrow_array::cast::AsArray as _;
use arrow_array::types::{Time64NanosecondType, TimestampMicrosecondType};
use arrow_array::{Array, Time64NanosecondArray, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};
use krishiv_kernels::cast::{CastMode, CastOptions, Clocked, FixedOffset, TimeZoneRules, cast};

const MICROS_PER_DAY: i64 = 86_400_000_000;
/// 2020-05-17 12:34:56.789012 as microseconds of the wall clock.
const TS: i64 = 1_589_718_896_789_012;

fn ntz() -> DataType {
    DataType::Timestamp(TimeUnit::Microsecond, None)
}

fn ltz() -> DataType {
    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
}

fn time_of(input: &dyn Array, zone: &dyn TimeZoneRules) -> Vec<Option<i64>> {
    let opts = CastOptions {
        mode: CastMode::Ansi,
        time_zone: zone,
    };
    let out = cast(input, &DataType::Time64(TimeUnit::Nanosecond), &opts).unwrap();
    out.as_primitive::<Time64NanosecondType>().iter().collect()
}

fn timestamps(input: &dyn Array, to: &DataType, zone: &dyn TimeZoneRules) -> Vec<Option<i64>> {
    let opts = CastOptions {
        mode: CastMode::Ansi,
        time_zone: zone,
    };
    let out = cast(input, to, &opts).unwrap();
    assert_eq!(out.data_type(), to);
    out.as_primitive::<TimestampMicrosecondType>()
        .iter()
        .collect()
}

/// `12:34:56.789012` as nanoseconds since midnight.
const TIME_NANOS: i64 = (12 * 3_600 + 34 * 60 + 56) * 1_000_000_000 + 789_012_000;

#[test]
fn a_timestamp_ntz_gives_its_wall_clock_time_of_day() {
    // cast.sql 219: 12:34:56.789012; 222: 1969-12-31 23:59:59.123456 is 23:59:59.123456 (floorMod).
    let before_epoch = -876_544; // 1969-12-31 23:59:59.123456
    let a = TimestampMicrosecondArray::from(vec![Some(TS), Some(before_epoch), None]);
    // The zone is not read: a wall-clock value has no instant.
    let got = time_of(&a, &FixedOffset(5 * 3_600));
    assert_eq!(
        got,
        vec![
            Some(TIME_NANOS),
            Some((23 * 3_600 + 59 * 60 + 59) * 1_000_000_000 + 123_456_000),
            None
        ]
    );
}

#[test]
fn a_timestamp_gives_its_time_of_day_in_the_session_zone() {
    // 2020-05-17 10:34:56.789012 UTC is 12:34:56.789012 at +02:00.
    let instant = TS - 2 * 3_600_000_000;
    let a = TimestampMicrosecondArray::from(vec![Some(instant)]).with_timezone("UTC");
    assert_eq!(time_of(&a, &FixedOffset(2 * 3_600)), vec![Some(TIME_NANOS)]);
    // At UTC the same instant is 10:34:56.789012.
    assert_eq!(
        time_of(&a, &FixedOffset(0)),
        vec![Some(TIME_NANOS - 2 * 3_600 * 1_000_000_000)]
    );
}

#[test]
fn a_time_takes_todays_date_in_the_session_zone() {
    // The query starts at 2020-05-17 23:30 UTC, which is already 2020-05-18 at +02:00.
    let start = 18_399 * MICROS_PER_DAY + (23 * 3_600 + 30 * 60) * 1_000_000;
    let plus_two = FixedOffset(2 * 3_600);
    let zone = Clocked {
        inner: &plus_two,
        query_start: start,
    };
    // Nine digits: the ones past the microsecond are dropped (`makeTimestampNTZ`).
    let a = Time64NanosecondArray::from(vec![Some(TIME_NANOS + 345), None]);
    let may_18 = 18_400 * MICROS_PER_DAY;
    assert_eq!(
        timestamps(&a, &ntz(), &zone),
        vec![Some(may_18 + TIME_NANOS / 1_000), None]
    );
    // The local date-time 2020-05-18 12:34:56.789012 at +02:00 as an instant.
    assert_eq!(
        timestamps(&a, &ltz(), &zone),
        vec![Some(may_18 + TIME_NANOS / 1_000 - 2 * 3_600_000_000), None]
    );
}

#[test]
fn a_time_without_the_query_clock_is_refused_by_name() {
    let a = Time64NanosecondArray::from(vec![Some(TIME_NANOS)]);
    let opts = CastOptions {
        mode: CastMode::Ansi,
        time_zone: &FixedOffset(0),
    };
    let err = cast(&a, &ntz(), &opts).unwrap_err();
    assert!(
        err.to_string().contains("KRISHIV_UNSUPPORTED_FEATURE"),
        "{err}"
    );
}
