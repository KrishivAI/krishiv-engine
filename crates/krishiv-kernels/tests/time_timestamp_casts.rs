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

/// A `TIME` to `TIMESTAMP` in a region zone is `LocalDateTime.atZone` (Spark 4.3.0
/// `DateTimeUtils.makeTimestamp(days, nanos, zoneId)`): in a DST gap the local time moves
/// forward by the gap's length, in an overlap it takes the earlier offset. And back: the
/// instant's local time of day in the zone (`timestampToNanosOfDay`).
#[test]
fn a_time_across_a_dst_change_is_read_as_java_reads_a_local_date_time() {
    let la = krishiv_kernels::datetime::resolve_zone("America/Los_Angeles").unwrap();
    let hour = 3_600_000_000_i64;
    let at = |h: i64, m: i64| (h * 3_600 + m * 60) * 1_000_000_000;
    // 2020-03-08 (day 18,329): 02:00 PST jumps to 03:00 PDT; the query starts at 18:00 UTC.
    let gap_day = 18_329 * MICROS_PER_DAY;
    let zone = Clocked {
        inner: &la,
        query_start: gap_day + 18 * hour,
    };
    let a = Time64NanosecondArray::from(vec![Some(at(1, 30)), Some(at(2, 30)), Some(at(3, 30))]);
    assert_eq!(
        timestamps(&a, &ltz(), &zone),
        vec![
            // 01:30 PST = 09:30 UTC; 02:30 does not exist: 03:30 PDT = 10:30 UTC, as 03:30 is.
            Some(gap_day + 9 * hour + hour / 2),
            Some(gap_day + 10 * hour + hour / 2),
            Some(gap_day + 10 * hour + hour / 2),
        ]
    );
    // 2020-11-01 (day 18,567): 02:00 PDT falls back to 01:00 PST; 01:30 happens twice and
    // takes the earlier offset, PDT: 08:30 UTC.
    let overlap_day = 18_567 * MICROS_PER_DAY;
    let zone = Clocked {
        inner: &la,
        query_start: overlap_day + 18 * hour,
    };
    let a = Time64NanosecondArray::from(vec![Some(at(1, 30))]);
    assert_eq!(
        timestamps(&a, &ltz(), &zone),
        vec![Some(overlap_day + 8 * hour + hour / 2)]
    );
    // Both 01:30s come back as 01:30.
    let both = TimestampMicrosecondArray::from(vec![
        Some(overlap_day + 8 * hour + hour / 2),
        Some(overlap_day + 9 * hour + hour / 2),
    ])
    .with_timezone("UTC");
    assert_eq!(time_of(&both, &la), vec![Some(at(1, 30)), Some(at(1, 30))]);
}

/// `h:m:s.micros` as nanoseconds since midnight.
const fn nanos_of(h: i64, m: i64, s: i64, micros: i64) -> i64 {
    ((h * 3_600 + m * 60 + s) * 1_000_000 + micros) * 1_000
}

#[test]
fn an_instant_within_one_offset_of_the_range_still_has_a_time_of_day() {
    // `Long.MIN_VALUE` micros is -290308-12-21 19:59:05.224192 UTC; at -07:52:58 (Los Angeles'
    // mean time, which Spark renders as -290308-12-21 12:06:07.224192) its local time is past the
    // microsecond range, which Spark's `LocalDateTime` still covers.
    let min = TimestampMicrosecondArray::from(vec![Some(i64::MIN)]).with_timezone("UTC");
    let la = FixedOffset(-(7 * 3_600 + 52 * 60 + 58));
    assert_eq!(time_of(&min, &la), vec![Some(nanos_of(12, 6, 7, 224_192))]);
    // `Long.MAX_VALUE` micros is +294247-01-10 04:00:54.775807 UTC; at +02:00, 06:00:54.775807.
    let max = TimestampMicrosecondArray::from(vec![Some(i64::MAX)]).with_timezone("UTC");
    assert_eq!(
        time_of(&max, &FixedOffset(2 * 3_600)),
        vec![Some(nanos_of(6, 0, 54, 775_807))]
    );
}
