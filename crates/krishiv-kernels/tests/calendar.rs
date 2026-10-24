// SPDX-License-Identifier: Apache-2.0
//! The calendar kernels (`monthname` ... `convert_timezone`) against values
//! recorded from Spark 4.2 (the function matrix and the SQL goldens) and
//! against independent oracles, in UTC and in a zone with a DST gap and
//! overlap.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use arrow_array::types::{Date32Type, Int32Type, TimestampMicrosecondType};
use arrow_array::{
    Array, BooleanArray, Date32Array, Int32Array, StringArray, TimestampMicrosecondArray,
    cast::AsArray,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, TimeZoneRules, cast};
use krishiv_kernels::datetime::{
    DatePart, TruncUnit, Zones, convert_timezone, day_name, extract_date, extract_timestamp,
    from_utc_timestamp, last_day, make_date, month_name, months_between, next_day,
    to_utc_timestamp, trunc_date, trunc_timestamp, unix_date,
};
use krishiv_kernels::{KernelCtx, SparkVersion};
use proptest::prelude::*;

const UTC: FixedOffset = FixedOffset(0);
const HOUR: i64 = 3_600_000_000;

fn options() -> CastOptions<'static> {
    CastOptions {
        mode: CastMode::Ansi,
        time_zone: &UTC,
    }
}

fn date(s: &str) -> i32 {
    let out = cast(&StringArray::from(vec![s]), &DataType::Date32, &options()).unwrap();
    out.as_primitive::<Date32Type>().value(0)
}

/// A wall clock (or UTC instant) as microseconds.
fn ts(s: &str) -> i64 {
    let to = DataType::Timestamp(TimeUnit::Microsecond, None);
    let out = cast(&StringArray::from(vec![s]), &to, &options()).unwrap();
    out.as_primitive::<TimestampMicrosecondType>().value(0)
}

fn dates(v: &[Option<i32>]) -> Date32Array {
    Date32Array::from(v.to_vec())
}

fn strs(v: &[Option<&str>]) -> StringArray {
    StringArray::from(v.to_vec())
}

fn stamps(v: &[Option<i64>]) -> TimestampMicrosecondArray {
    TimestampMicrosecondArray::from(v.to_vec())
}

/// A zone like Los Angeles in 2024: -08:00, and -07:00 from the UTC instant
/// 2024-03-10 10:00 (the wall clock skips 02:00-03:00) to 2024-11-03 09:00
/// (01:00-02:00 happens twice).
#[derive(Debug)]
struct Pacific;

fn dst_start() -> i64 {
    ts("2024-03-10 10:00:00") / 1_000_000
}

fn dst_end() -> i64 {
    ts("2024-11-03 09:00:00") / 1_000_000
}

impl TimeZoneRules for Pacific {
    fn offset_at_utc(&self, utc_seconds: i64) -> i32 {
        if (dst_start()..dst_end()).contains(&utc_seconds) {
            -7 * 3600
        } else {
            -8 * 3600
        }
    }

    fn offset_for_local(&self, local_seconds: i64) -> i32 {
        let valid = |offset: i32| self.offset_at_utc(local_seconds - i64::from(offset)) == offset;
        // An overlap takes the earlier (summer) offset; a gap, like any other
        // wall clock, the winter one (the offset before the transition).
        if valid(-7 * 3600) {
            -7 * 3600
        } else {
            -8 * 3600
        }
    }
}

#[test]
fn names_and_unix_dates() {
    // 1582-10-04, 10-05 and 10-15 are all ordinary days: no Julian cutover.
    let days = [
        date("1582-10-04"),
        date("1582-10-05"),
        date("1582-10-15"),
        date("0001-01-01"),
        date("9999-12-31"),
        date("2024-02-29"),
    ];
    let a = dates(&days.map(Some));
    let months: Vec<_> = month_name(&a)
        .iter()
        .map(|s| s.unwrap().to_owned())
        .collect();
    assert_eq!(months, ["Oct", "Oct", "Oct", "Jan", "Dec", "Feb"]);
    let weekdays: Vec<_> = day_name(&a).iter().map(|s| s.unwrap().to_owned()).collect();
    // Spark: dayname(DATE'1582-10-10') is Sun, 0001-01-01 is Mon, 9999-12-31 Fri.
    assert_eq!(weekdays, ["Mon", "Tue", "Fri", "Mon", "Fri", "Thu"]);
    assert_eq!(date("1582-10-05") - date("1582-10-04"), 1);
    assert_eq!(
        unix_date(&dates(&[
            Some(date("1970-01-02")),
            None,
            Some(date("0001-01-01"))
        ]))
        .iter()
        .collect::<Vec<_>>(),
        [Some(1), None, Some(-719_162)]
    );
    let nulls = dates(&[None]);
    assert!(month_name(&nulls).is_null(0) && day_name(&nulls).is_null(0));
    // The names of the whole calendar cycle, every month and weekday.
    let every: Vec<Option<i32>> = (0..366).map(|d| Some(date("2024-01-01") + d)).collect();
    let all = month_name(&dates(&every));
    assert_eq!(
        all.iter()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        12
    );
    assert_eq!(
        day_name(&dates(&every))
            .iter()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        7
    );
}

#[test]
fn last_day_of_every_kind_of_month() {
    let last = |d: &str| last_day(&dates(&[Some(date(d))])).value(0);
    for (from, to) in [
        ("2024-02-10", "2024-02-29"),
        ("2023-02-10", "2023-02-28"),
        ("1900-02-01", "1900-02-28"),
        ("2000-02-29", "2000-02-29"),
        ("2009-01-12", "2009-01-31"),
        ("2024-04-30", "2024-04-30"),
        ("2024-12-01", "2024-12-31"),
        ("0001-01-01", "0001-01-31"),
        ("9999-12-31", "9999-12-31"),
        ("1582-10-04", "1582-10-31"),
        ("-0001-02-10", "-0001-02-28"),
    ] {
        assert_eq!(last(from), date(to), "{from}");
    }
    assert!(last_day(&dates(&[None])).is_null(0));
    // Spark adds in `int`: the end of the range wraps like Java's.
    let max = last_day(&dates(&[Some(i32::MAX)])).value(0);
    assert_eq!(max, i32::MAX.wrapping_sub(11).wrapping_add(31));
}

#[test]
fn next_day_names_and_edges() {
    let thu = date("2015-07-23");
    let run =
        |d: i32, name: &str, ansi: bool| next_day(&dates(&[Some(d)]), &strs(&[Some(name)]), ansi);
    // Spark's example: the Monday after Thursday 2015-07-23.
    assert_eq!(run(thu, "Mon", true).unwrap().value(0), date("2015-07-27"));
    for (names, offset) in [
        (["MO", "mon", "Monday"], 4),
        (["TU", "TUE", "tuesday"], 5),
        (["we", "WED", "WEDNESDAY"], 6),
        (["TH", "thu", "Thursday"], 7),
        (["Fr", "FRI", "friday"], 1),
        (["SA", "sat", "SATURDAY"], 2),
        (["su", "SUN", "Sunday"], 3),
    ] {
        for name in names {
            assert_eq!(
                run(thu, name, true).unwrap().value(0),
                thu + offset,
                "{name}"
            );
        }
    }
    // The same weekday is the next week, never the day itself.
    assert_eq!(
        run(date("2024-02-29"), "TU", true).unwrap().value(0),
        date("2024-03-05")
    );
    assert_eq!(
        run(date("2024-02-29"), "TH", true).unwrap().value(0),
        date("2024-03-07")
    );
    // An unknown name: ILLEGAL_DAY_OF_WEEK with ANSI, NULL without.
    let e = run(thu, "xx", true).unwrap_err();
    assert_eq!(
        (e.error_class(), e.row()),
        (ErrorClass::IllegalDayOfWeek, Some(0))
    );
    assert!(e.to_string().contains("Illegal input for day of week: xx"));
    assert!(run(thu, "xx", false).unwrap().is_null(0));
    for bad in ["", " MON", "MON ", "MONDA", "1"] {
        assert!(run(thu, bad, false).unwrap().is_null(0), "{bad:?}");
    }
    // NULLs, and a bad name on a NULL row never raises.
    let out = next_day(
        &dates(&[None, Some(thu), Some(thu)]),
        &strs(&[Some("xx"), None, Some("FR")]),
        true,
    )
    .unwrap();
    assert_eq!(
        out.iter().collect::<Vec<_>>(),
        [None, None, Some(date("2015-07-24"))]
    );
    // The error names the failing row.
    let e = next_day(
        &dates(&[Some(thu), Some(thu)]),
        &strs(&[Some("MO"), Some("zz")]),
        true,
    )
    .unwrap_err();
    assert_eq!(e.row(), Some(1));
    // `int` arithmetic wraps at the top of the range, as Spark's does.
    assert!(run(i32::MAX, "SU", true).unwrap().value(0) < 0);
    let e = next_day(&dates(&[Some(1)]), &strs(&[Some("MO"), Some("TU")]), true).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn trunc_date_levels() {
    let trunc = |d: &str, f: &str| {
        let out = trunc_date(&dates(&[Some(date(d))]), &strs(&[Some(f)])).unwrap();
        (!out.is_null(0)).then(|| out.value(0))
    };
    // 2019-08-04 is a Sunday: its week starts on Monday 07-29.
    assert_eq!(trunc("2019-08-04", "week"), Some(date("2019-07-29")));
    assert_eq!(trunc("2019-08-05", "WEEK"), Some(date("2019-08-05")));
    assert_eq!(trunc("2024-02-29", "week"), Some(date("2024-02-26")));
    assert_eq!(trunc("2009-02-12", "MM"), Some(date("2009-02-01")));
    assert_eq!(trunc("2009-02-12", "mon"), Some(date("2009-02-01")));
    assert_eq!(trunc("2009-02-12", "MONTH"), Some(date("2009-02-01")));
    assert_eq!(trunc("2015-10-27", "YEAR"), Some(date("2015-01-01")));
    assert_eq!(trunc("2015-10-27", "yyyy"), Some(date("2015-01-01")));
    assert_eq!(trunc("2015-10-27", "YY"), Some(date("2015-01-01")));
    for (d, q) in [
        ("2024-01-31", "2024-01-01"),
        ("2024-03-31", "2024-01-01"),
        ("2024-04-01", "2024-04-01"),
        ("2024-08-15", "2024-07-01"),
        ("2024-12-31", "2024-10-01"),
    ] {
        assert_eq!(trunc(d, "QUARTER"), Some(date(q)), "{d}");
    }
    // The ends of the calendar and the 1582 switch.
    assert_eq!(trunc("0001-01-01", "week"), Some(date("0001-01-01")));
    assert_eq!(trunc("0001-01-05", "year"), Some(date("0001-01-01")));
    assert_eq!(trunc("9999-12-31", "year"), Some(date("9999-01-01")));
    assert_eq!(trunc("1582-10-15", "week"), Some(date("1582-10-11")));
    assert_eq!(trunc("1582-10-09", "month"), Some(date("1582-10-01")));
    // A date has no day level or finer, and an unknown name is NULL.
    for f in [
        "DAY",
        "DD",
        "HOUR",
        "MINUTE",
        "SECOND",
        "MILLISECOND",
        "MICROSECOND",
        "",
        "1",
        "century",
    ] {
        assert_eq!(trunc("2024-02-29", f), None, "{f:?}");
    }
    // The first year of the smallest dates starts before the smallest date.
    let min = dates(&[Some(i32::MIN)]);
    let e = trunc_date(&min, &strs(&[Some("year")])).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DatetimeOverflow);
    assert!(trunc_date(&min, &strs(&[Some("day")])).unwrap().is_null(0));
    // NULLs on either side, and a bad length.
    let out = trunc_date(&dates(&[None, Some(1)]), &strs(&[Some("year"), None])).unwrap();
    assert!(out.is_null(0) && out.is_null(1));
    let e = trunc_date(&dates(&[Some(1)]), &strs(&[])).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn trunc_units_parse_with_spark_aliases() {
    for (names, unit) in [
        (&["year", "YYYY", "yy"][..], TruncUnit::Year),
        (&["Quarter"], TruncUnit::Quarter),
        (&["month", "MON", "mm"], TruncUnit::Month),
        (&["week"], TruncUnit::Week),
        (&["day", "DD"], TruncUnit::Day),
        (&["HOUR"], TruncUnit::Hour),
        (&["minute"], TruncUnit::Minute),
        (&["second"], TruncUnit::Second),
        (&["millisecond"], TruncUnit::Millisecond),
        (&["MICROSECOND"], TruncUnit::Microsecond),
    ] {
        for n in names {
            assert_eq!(TruncUnit::parse(n), Some(unit), "{n}");
        }
    }
    assert!(TruncUnit::Microsecond < TruncUnit::Day && TruncUnit::Week < TruncUnit::Year);
    assert_eq!(TruncUnit::parse("decade"), None);
    assert_eq!(TruncUnit::parse(" year"), None);
}

fn truncated(f: &str, t: i64, zone: &dyn TimeZoneRules) -> Option<i64> {
    let out = trunc_timestamp(&strs(&[Some(f)]), &stamps(&[Some(t)]), zone).unwrap();
    (!out.is_null(0)).then(|| out.value(0))
}

#[test]
fn trunc_timestamp_in_utc() {
    let t = ts("2015-03-05 09:32:05.123456");
    let want = |s: &str| Some(ts(s));
    assert_eq!(truncated("YEAR", t, &UTC), want("2015-01-01 00:00:00"));
    assert_eq!(truncated("MM", t, &UTC), want("2015-03-01 00:00:00"));
    assert_eq!(truncated("QUARTER", t, &UTC), want("2015-01-01 00:00:00"));
    assert_eq!(truncated("WEEK", t, &UTC), want("2015-03-02 00:00:00"));
    assert_eq!(truncated("DD", t, &UTC), want("2015-03-05 00:00:00"));
    assert_eq!(truncated("HOUR", t, &UTC), want("2015-03-05 09:00:00"));
    assert_eq!(truncated("MINUTE", t, &UTC), want("2015-03-05 09:32:00"));
    assert_eq!(truncated("SECOND", t, &UTC), want("2015-03-05 09:32:05"));
    assert_eq!(
        truncated("MILLISECOND", t, &UTC),
        want("2015-03-05 09:32:05.123")
    );
    assert_eq!(truncated("MICROSECOND", t, &UTC), Some(t));
    assert_eq!(truncated("century", t, &UTC), None);
    // Before the epoch the cut still goes down.
    let before = ts("1969-12-31 23:59:59.999999");
    assert_eq!(
        truncated("SECOND", before, &UTC),
        want("1969-12-31 23:59:59")
    );
    assert_eq!(
        truncated("MILLISECOND", before, &UTC),
        want("1969-12-31 23:59:59.999")
    );
    assert_eq!(truncated("YEAR", before, &UTC), want("1969-01-01 00:00:00"));
    assert_eq!(
        truncated("YEAR", ts("0001-06-01 12:00:00"), &UTC),
        want("0001-01-01 00:00:00")
    );
    // A NULL format or timestamp is NULL.
    let out = trunc_timestamp(&strs(&[None, Some("DD")]), &stamps(&[Some(0), None]), &UTC).unwrap();
    assert!(out.is_null(0) && out.is_null(1));
    // Lengths must agree.
    let e = trunc_timestamp(&strs(&[Some("DD")]), &stamps(&[]), &UTC).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn trunc_timestamp_in_a_zone_with_dst() {
    // The UTC instant of a Pacific wall clock.
    let at = |wall: &str, offset: i64| ts(wall) - offset * HOUR;
    let day = |s: &str| ts(s);
    // Whole days: midnight of the local day, whichever offset it had.
    let noon_after = at("2024-03-10 12:00:00", -7);
    assert_eq!(
        truncated("DD", noon_after, &Pacific),
        Some(at("2024-03-10 00:00:00", -8))
    );
    // 2024-03-10 starts 23 hours before the day after.
    assert_eq!(
        at("2024-03-11 00:00:00", -7) - at("2024-03-10 00:00:00", -8),
        23 * HOUR
    );
    assert_eq!(
        truncated("HOUR", noon_after, &Pacific),
        Some(at("2024-03-10 12:00:00", -7))
    );
    // The date levels start the local day.
    assert_eq!(
        truncated("WEEK", noon_after, &Pacific),
        Some(at("2024-03-04 00:00:00", -8))
    );
    assert_eq!(
        truncated("MONTH", noon_after, &Pacific),
        Some(at("2024-03-01 00:00:00", -8))
    );
    assert_eq!(
        truncated("YEAR", noon_after, &Pacific),
        Some(at("2024-01-01 00:00:00", -8))
    );
    assert_eq!(
        truncated("QUARTER", at("2024-08-20 10:00:00", -7), &Pacific),
        Some(at("2024-07-01 00:00:00", -7))
    );
    // The overlap 01:00-02:00 on 2024-11-03 happens twice: an hour or a day
    // cut keeps the side the instant is on.
    let first = at("2024-11-03 01:30:00", -7);
    let second = at("2024-11-03 01:30:00", -8);
    assert_eq!(second - first, HOUR);
    assert_eq!(
        truncated("HOUR", first, &Pacific),
        Some(at("2024-11-03 01:00:00", -7))
    );
    assert_eq!(
        truncated("HOUR", second, &Pacific),
        Some(at("2024-11-03 01:00:00", -8))
    );
    assert_eq!(
        truncated("MINUTE", second, &Pacific),
        Some(at("2024-11-03 01:30:00", -8))
    );
    assert_eq!(
        truncated("DD", second, &Pacific),
        Some(at("2024-11-03 00:00:00", -7))
    );
    // Sub-second levels ignore the zone.
    assert_eq!(truncated("SECOND", first + 5, &Pacific), Some(first));
    // A zone ahead of UTC: 23:30 UTC on the 9th is already the 10th.
    let ahead = FixedOffset(3600);
    assert_eq!(
        truncated("DD", day("2024-03-09 23:30:00"), &ahead),
        Some(day("2024-03-09 23:00:00"))
    );
    // Out of range: the cut would pass the smallest timestamp.
    let e =
        trunc_timestamp(&strs(&[Some("YEAR")]), &stamps(&[Some(i64::MIN + 1)]), &UTC).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DatetimeOverflow);
    let e =
        trunc_timestamp(&strs(&[Some("SECOND")]), &stamps(&[Some(i64::MIN)]), &UTC).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DatetimeOverflow);
}

fn between(end: &str, start: &str, round: bool, zone: &dyn TimeZoneRules) -> f64 {
    let out = months_between(
        &stamps(&[Some(ts(end))]),
        &stamps(&[Some(ts(start))]),
        &BooleanArray::from(vec![round]),
        zone,
    )
    .unwrap();
    out.value(0)
}

#[test]
fn months_between_follows_spark() {
    // Values from Spark 4.2.
    assert_eq!(
        between("1997-02-28 10:30:00", "1996-10-30 00:00:00", true, &UTC),
        3.94959677
    );
    assert_eq!(
        between("1997-02-28 10:30:00", "1996-10-30 00:00:00", false, &UTC),
        3.9495967741935485
    );
    assert_eq!(
        between("2024-02-29 00:00:00", "1996-10-30 00:00:00", true, &UTC),
        327.96774194
    );
    assert_eq!(
        between("2024-02-29 00:00:00", "1996-10-30 00:00:00", false, &UTC),
        327.96774193548384
    );
    // The microseconds are dropped from the time of day.
    assert_eq!(
        between(
            "2024-02-29 12:34:56.123456",
            "1996-10-30 00:00:00",
            false,
            &UTC
        ),
        327.9846535244922
    );
    assert_eq!(
        between(
            "2024-02-29 12:34:56.123456",
            "1996-10-30 00:00:00",
            true,
            &UTC
        ),
        327.98465352
    );
    assert_eq!(
        between("1997-02-28 10:30:00", "2024-02-29 00:00:00", true, &UTC),
        -324.0
    );
    // Equal days, or two last days of a month: whole months, whatever the time.
    assert_eq!(
        between("2020-03-15 23:00:00", "2020-01-15 01:00:00", true, &UTC),
        2.0
    );
    assert_eq!(
        between("2020-02-29 05:00:00", "2020-01-31 20:00:00", true, &UTC),
        1.0
    );
    assert_eq!(
        between("2021-02-28 00:00:00", "2020-02-29 00:00:00", true, &UTC),
        12.0
    );
    assert_eq!(
        between("2020-01-01 00:00:00", "2020-01-01 00:00:00", false, &UTC),
        0.0
    );
    // One last day only: 31-day months.
    let one = between("2020-02-29 00:00:00", "2020-01-30 00:00:00", false, &UTC);
    assert_eq!(one, 1.0 + (29.0 - 30.0) / 31.0);
    // The calendar ends and 1582.
    assert_eq!(
        between("9999-12-31 00:00:00", "0001-12-31 00:00:00", true, &UTC),
        9998.0 * 12.0
    );
    assert_eq!(
        between("1582-10-15 00:00:00", "1582-10-04 00:00:00", false, &UTC),
        11.0 / 31.0
    );
}

/// Spark's formula for whole months `months` and the day and seconds of the
/// day of each side.
fn formula(months: f64, (d1, s1): (i64, i64), (d2, s2): (i64, i64)) -> f64 {
    months + ((d1 - d2) * 86_400 + s1 - s2) as f64 / 2_678_400.0
}

#[test]
fn months_between_reads_wall_clocks_in_the_zone() {
    let months = |end: i64, start: i64, zone: &dyn TimeZoneRules| {
        months_between(
            &stamps(&[Some(end)]),
            &stamps(&[Some(start)]),
            &BooleanArray::from(vec![false]),
            zone,
        )
        .unwrap()
        .value(0)
    };
    let (end, start) = (ts("2024-04-01 03:00:00"), ts("2024-02-10 12:00:00"));
    let h = 3_600;
    // In UTC: April 1st 03:00 and February 10th 12:00.
    assert_eq!(
        months(end, start, &UTC),
        formula(2.0, (1, 3 * h), (10, 12 * h))
    );
    // At -08:00 the same instants are March 31st 19:00 and February 10th
    // 04:00: another month and other days, so another answer.
    let la = FixedOffset(-8 * 3600);
    assert_eq!(
        months(end, start, &la),
        formula(1.0, (31, 19 * h), (10, 4 * h))
    );
    // On the DST day the time of day counts from the instant the local day
    // started: noon PDT is 11 hours after midnight PST, not 12.
    let noon = ts("2024-03-10 12:00:00") + 7 * HOUR;
    let before = ts("2024-02-09 12:00:00") + 8 * HOUR;
    assert_eq!(
        months(noon, before, &Pacific),
        formula(1.0, (10, 11 * h), (9, 12 * h))
    );
}

#[test]
fn months_between_nulls_and_edges() {
    let end = stamps(&[None, Some(0), Some(0), Some(0)]);
    let start = stamps(&[Some(0), None, Some(0), Some(i64::MAX)]);
    let round = BooleanArray::from(vec![Some(true), Some(true), None, Some(true)]);
    let e = months_between(&end, &start, &round, &FixedOffset(3600)).unwrap_err();
    // The first three rows are NULL; the last overflows the local time.
    assert_eq!(
        (e.row(), e.error_class()),
        (Some(3), ErrorClass::DatetimeOverflow)
    );
    let out = months_between(
        &end.slice(0, 3),
        &start.slice(0, 3),
        &round.slice(0, 3),
        &UTC,
    )
    .unwrap();
    assert_eq!(out.null_count(), 3);
    let e = months_between(&end, &start.slice(0, 2), &round, &UTC).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let none = BooleanArray::from(Vec::<bool>::new());
    assert_eq!(
        months_between(&stamps(&[]), &stamps(&[]), &none, &UTC)
            .unwrap()
            .len(),
        0
    );
}

fn make(y: i32, m: i32, d: i32, ansi: bool) -> Result<Option<i32>, (ErrorClass, String)> {
    let one = |v: i32| Int32Array::from(vec![v]);
    make_date(&one(y), &one(m), &one(d), ansi, &KernelCtx::new())
        .map(|a| (!a.is_null(0)).then(|| a.value(0)))
        .map_err(|e| (e.error_class(), e.to_string()))
}

#[test]
fn make_date_valid_and_invalid() {
    assert_eq!(make(2019, 1, 1, true), Ok(Some(date("2019-01-01"))));
    assert_eq!(make(12, 12, 12, true), Ok(Some(date("0012-12-12"))));
    assert_eq!(make(2024, 2, 29, true), Ok(Some(date("2024-02-29"))));
    assert_eq!(make(2000, 2, 29, true), Ok(Some(date("2000-02-29"))));
    assert_eq!(make(0, 1, 1, true), Ok(Some(date("0000-01-01"))));
    assert_eq!(make(-1, 1, 28, true), Ok(Some(date("-0001-01-28"))));
    assert_eq!(make(999_999, 3, 18, true), Ok(Some(date("+999999-03-18"))));
    assert_eq!(make(9999, 12, 31, true), Ok(Some(date("9999-12-31"))));
    // Spark's messages, the `java.time` texts, with ANSI on...
    let field = ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion;
    let msg = |r: Result<Option<i32>, (ErrorClass, String)>| {
        let (class, text) = r.unwrap_err();
        assert_eq!(class, field);
        text
    };
    assert!(
        msg(make(2000, 13, 1, true))
            .contains("Invalid value for MonthOfYear (valid values 1 - 12): 13.")
    );
    assert!(
        msg(make(2000, 1, 33, true))
            .contains("Invalid value for DayOfMonth (valid values 1 - 28/31): 33.")
    );
    assert!(msg(make(2000, 0, 1, true)).contains("MonthOfYear (valid values 1 - 12): 0"));
    assert!(msg(make(2000, 1, 0, true)).contains("DayOfMonth (valid values 1 - 28/31): 0"));
    assert!(
        msg(make(2023, 2, 29, true))
            .contains("Invalid date 'February 29' as '2023' is not a leap year.")
    );
    assert!(msg(make(1900, 2, 29, true)).contains("'1900' is not a leap year"));
    assert!(msg(make(2024, 2, 30, true)).contains("Invalid date 'FEBRUARY 30'."));
    assert!(msg(make(2024, 4, 31, true)).contains("Invalid date 'APRIL 31'."));
    assert!(
        msg(make(1_000_000_000, 1, 1, true))
            .contains("Invalid value for Year (valid values -999999999 - 999999999): 1000000000")
    );
    assert!(msg(make(i32::MIN, 1, 1, true)).contains("Year (valid values -999999999"));
    assert!(msg(make(2000, 1, 33, true)).contains("If necessary set \"spark.sql.ansi.enabled\""));
    // ...and NULL with it off.
    for (y, m, d) in [
        (2000, 13, 1),
        (2000, 1, 33),
        (2023, 2, 29),
        (2024, 4, 31),
        (i32::MAX, 1, 1),
    ] {
        assert_eq!(make(y, m, d, false), Ok(None), "{y}-{m}-{d}");
    }
}

#[test]
fn make_date_range_nulls_and_versions() {
    // A year whose dates leave the `DATE` range is ARITHMETIC_OVERFLOW in both
    // modes, with the Spark-version text.
    for ansi in [true, false] {
        let (class, text) = make(6_000_000, 1, 1, ansi).unwrap_err();
        assert_eq!(class, ErrorClass::ArithmeticOverflow);
        assert!(text.contains("overflow."), "{text}");
    }
    let v40 = KernelCtx::new().with_spark_version(SparkVersion::V4_0);
    let one = |v: i32| Int32Array::from(vec![v]);
    let e = make_date(&one(6_000_000), &one(1), &one(1), true, &v40).unwrap_err();
    assert!(e.to_string().contains("integer overflow."), "{e}");
    // The last year whose dates fit is 5881580 (July 11 is day 2^31 - 1).
    assert_eq!(make(5_881_580, 7, 11, true), Ok(Some(i32::MAX)));
    assert_eq!(
        make(5_881_580, 7, 12, true).unwrap_err().0,
        ErrorClass::ArithmeticOverflow
    );
    // NULL rows never raise, whatever the other fields hold.
    let out = make_date(
        &Int32Array::from(vec![None, Some(2000), Some(2000)]),
        &Int32Array::from(vec![Some(13), None, Some(2)]),
        &Int32Array::from(vec![Some(33), Some(33), Some(30)]),
        false,
        &KernelCtx::new(),
    )
    .unwrap();
    assert_eq!(out.null_count(), 3);
    let out = make_date(
        &Int32Array::from(vec![None, Some(2024)]),
        &Int32Array::from(vec![Some(13), Some(2)]),
        &Int32Array::from(vec![Some(33), Some(29)]),
        true,
        &KernelCtx::new(),
    )
    .unwrap();
    assert_eq!((out.is_null(0), out.value(1)), (true, date("2024-02-29")));
    let e = make_date(
        &Int32Array::from(vec![2024]),
        &Int32Array::from(vec![1, 2]),
        &Int32Array::from(vec![1]),
        true,
        &KernelCtx::new(),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

fn convert(
    f: impl Fn(
        &TimestampMicrosecondArray,
        &Zones<'_>,
    ) -> Result<TimestampMicrosecondArray, krishiv_kernels::KernelError>,
    t: i64,
    zone: &dyn TimeZoneRules,
) -> i64 {
    f(&stamps(&[Some(t)]), &Zones::One(zone)).unwrap().value(0)
}

#[test]
fn utc_to_zone_and_back() {
    let seoul = FixedOffset(9 * 3600);
    // Spark: from_utc_timestamp('2016-08-31', 'Asia/Seoul') is 09:00.
    assert_eq!(
        convert(from_utc_timestamp, ts("2016-08-31 00:00:00"), &seoul),
        ts("2016-08-31 09:00:00")
    );
    assert_eq!(
        convert(to_utc_timestamp, ts("2016-08-31 09:00:00"), &seoul),
        ts("2016-08-31 00:00:00")
    );
    // The offset at the instant: the day after the spring change is -07:00.
    assert_eq!(
        convert(from_utc_timestamp, ts("2024-03-10 12:00:00"), &Pacific),
        ts("2024-03-10 05:00:00")
    );
    assert_eq!(
        convert(from_utc_timestamp, ts("2024-03-10 09:59:59"), &Pacific),
        ts("2024-03-10 01:59:59")
    );
    // A wall clock in the skipped hour moves forward, one in the repeated
    // hour takes the earlier offset.
    assert_eq!(
        convert(to_utc_timestamp, ts("2024-03-10 02:30:00"), &Pacific),
        ts("2024-03-10 10:30:00")
    );
    assert_eq!(
        convert(to_utc_timestamp, ts("2024-11-03 01:30:00"), &Pacific),
        ts("2024-11-03 08:30:00")
    );
    // NULL values and NULL zones are NULL; the zone of each row is its own.
    let zones = [Some(seoul), None, Some(FixedOffset(-3600))];
    let t = stamps(&[Some(0), Some(0), None]);
    let out = from_utc_timestamp(&t, &Zones::PerRow(&zones)).unwrap();
    assert_eq!(out.iter().collect::<Vec<_>>(), [Some(9 * HOUR), None, None]);
    // The result must stay in range.
    let e = from_utc_timestamp(&stamps(&[Some(i64::MAX)]), &Zones::One(&seoul)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DatetimeOverflow);
    let e = to_utc_timestamp(&stamps(&[Some(i64::MIN)]), &Zones::One(&seoul)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DatetimeOverflow);
}

#[test]
fn convert_timezone_between_zones() {
    let moscow = FixedOffset(3 * 3600);
    let la = FixedOffset(-8 * 3600);
    let wall = |s: &str| stamps(&[Some(ts(s))]);
    // Spark: convert_timezone('Europe/Moscow', 'America/Los_Angeles',
    // timestamp_ntz'2022-01-01 00:00:00') is 2021-12-31 13:00:00.
    let out = convert_timezone(
        &Zones::One(&moscow),
        &Zones::One(&la),
        &wall("2022-01-01 00:00:00"),
    )
    .unwrap();
    assert_eq!(out.value(0), ts("2021-12-31 13:00:00"));
    // Through a DST zone: 02:30 in the skipped hour is 03:30 PDT, so 10:30 UTC.
    let out = convert_timezone(
        &Zones::One(&Pacific),
        &Zones::One(&UTC),
        &wall("2024-03-10 02:30:00"),
    )
    .unwrap();
    assert_eq!(out.value(0), ts("2024-03-10 10:30:00"));
    let out = convert_timezone(
        &Zones::One(&UTC),
        &Zones::One(&Pacific),
        &wall("2024-11-03 08:30:00"),
    )
    .unwrap();
    assert_eq!(out.value(0), ts("2024-11-03 01:30:00"));
    // Same zone in and out leaves the wall clock, except in a gap.
    let out = convert_timezone(
        &Zones::One(&Pacific),
        &Zones::One(&Pacific),
        &wall("2024-07-04 12:00:00"),
    )
    .unwrap();
    assert_eq!(out.value(0), ts("2024-07-04 12:00:00"));
    // NULL value, source or target.
    let pairs = [Some(moscow), None, Some(moscow)];
    let targets = [Some(la), Some(la), None];
    let t = stamps(&[Some(0), Some(0), Some(0)]);
    let out = convert_timezone(&Zones::PerRow(&pairs), &Zones::PerRow(&targets), &t).unwrap();
    assert_eq!(out.null_count(), 2);
    let e = convert_timezone(
        &Zones::One(&la),
        &Zones::One(&moscow),
        &stamps(&[Some(i64::MAX)]),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DatetimeOverflow);
    // A zone column must cover the rows.
    let short = [Some(moscow)];
    let two = stamps(&[Some(0), Some(0)]);
    let e = from_utc_timestamp(&two, &Zones::PerRow(&short)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let e = convert_timezone(&Zones::One(&moscow), &Zones::PerRow(&short), &two).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let e = convert_timezone(&Zones::PerRow(&short), &Zones::One(&moscow), &two).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn the_zone_functions_agree_with_the_extract_kernels() {
    // `from_utc_timestamp` then a UTC read of the fields equals reading the
    // fields in the zone.
    let seoul = FixedOffset(9 * 3600);
    let t = ts("2024-02-29 20:15:00");
    let shifted = from_utc_timestamp(&stamps(&[Some(t)]), &Zones::One(&seoul)).unwrap();
    let local = extract_timestamp(DatePart::Hour, &stamps(&[Some(t)]), Some(&seoul)).unwrap();
    let wall = extract_timestamp(DatePart::Hour, &shifted, None).unwrap();
    assert_eq!(local.as_primitive::<Int32Type>().value(0), 5);
    assert_eq!(wall.as_primitive::<Int32Type>().value(0), 5);
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(miri) { 2 } else { 512 }, failure_persistence: None, ..ProptestConfig::default() })]

    /// The last day of a month ends that month and the next day starts the
    /// following one; truncations never move forward and land on a boundary.
    #[test]
    fn month_ends_and_truncations_are_consistent(d in -700_000_i32..3_000_000) {
        let field = |part, x: i32| {
            extract_date(part, &dates(&[Some(x)])).unwrap().as_primitive::<Int32Type>().value(0)
        };
        let last = last_day(&dates(&[Some(d)])).value(0);
        prop_assert!(last >= d);
        prop_assert_eq!(field(DatePart::Month, last), field(DatePart::Month, d));
        prop_assert_ne!(field(DatePart::Month, last + 1), field(DatePart::Month, d));
        let trunc = |f: &str| trunc_date(&dates(&[Some(d)]), &strs(&[Some(f)])).unwrap().value(0);
        let (week, month, quarter, year) = (trunc("week"), trunc("month"), trunc("quarter"), trunc("year"));
        prop_assert!(week <= d && d < week + 7);
        prop_assert_eq!(field(DatePart::DayOfWeekIso, week), 1);
        prop_assert_eq!(field(DatePart::Day, month), 1);
        prop_assert!(year <= quarter && quarter <= month && month <= d);
        prop_assert_eq!(field(DatePart::DayOfYear, year), 1);
        prop_assert_eq!(field(DatePart::Month, quarter) % 3, 1);
        // `next_day` is the first such weekday after the date.
        for name in ["MO", "TU", "WE", "TH", "FR", "SA", "SU"] {
            let n = next_day(&dates(&[Some(d)]), &strs(&[Some(name)]), true).unwrap().value(0);
            prop_assert!(n > d && n <= d + 7);
        }
    }

    /// Truncating an instant is idempotent and never later; every level is
    /// a boundary of the next coarser one.
    #[test]
    fn instants_truncate_down(t in -60_000_000_000_000_000_i64..60_000_000_000_000_000, level in 0_usize..10) {
        let units = ["MICROSECOND", "MILLISECOND", "SECOND", "MINUTE", "HOUR", "DD", "WEEK", "MM", "QUARTER", "YEAR"];
        for zone in [&UTC as &dyn TimeZoneRules, &Pacific] {
            let once = truncated(units[level], t, zone).unwrap();
            prop_assert!(once <= t);
            prop_assert_eq!(truncated(units[level], once, zone), Some(once));
            // The levels from a microsecond to a day, and from a day to a
            // year, each cut further down as they get coarser.
            let chains = [&units[..=5], &[units[5], units[7], units[8], units[9]][..]];
            for chain in chains {
                let cuts: Vec<i64> = chain.iter().map(|u| truncated(u, t, zone).unwrap()).collect();
                prop_assert!(cuts.windows(2).all(|w| w[1] <= w[0]), "{chain:?} {cuts:?}");
            }
        }
    }

    /// `months_between` of a date with itself is 0, antisymmetric, and a
    /// whole number of months when the days of the month agree.
    #[test]
    fn months_between_is_antisymmetric(a in -500_000_i64..2_900_000, b in -500_000_i64..2_900_000) {
        let day = 86_400_000_000_i64;
        let (x, y) = (a * day, b * day);
        let run = |e: i64, s: i64| months_between(&stamps(&[Some(e)]), &stamps(&[Some(s)]), &BooleanArray::from(vec![false]), &UTC).unwrap().value(0);
        prop_assert_eq!(run(x, x), 0.0);
        prop_assert!((run(x, y) + run(y, x)).abs() < 1e-9);
        let field = |part, d: i64| {
            extract_date(part, &dates(&[Some(i32::try_from(d).unwrap())])).unwrap().as_primitive::<Int32Type>().value(0)
        };
        if field(DatePart::Day, a) == field(DatePart::Day, b) {
            prop_assert_eq!(run(x, y).fract(), 0.0);
        }
    }
}
