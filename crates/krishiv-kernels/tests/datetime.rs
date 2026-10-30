// SPDX-License-Identifier: Apache-2.0
//! Datetime kernels against values recorded from Spark 4.2 (4.0 and 4.1
//! gave the same values; the overflow message differs by version), and
//! property tests against independent oracles.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use arrow_array::types::{Date32Type, Decimal128Type, Int32Type, TimestampMicrosecondType};
use arrow_array::{
    Array, Date32Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray,
    cast::AsArray,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};
use krishiv_kernels::datetime::{
    DatePart, current_date, date_add, date_add_months, date_diff, date_sub, extract_date,
    extract_timestamp, timestamp_add_day_time, timestamp_add_months,
};
use krishiv_kernels::policy::OverflowMode;
use krishiv_kernels::{KernelCtx, Operand, SparkVersion};
use proptest::prelude::*;

const UTC: FixedOffset = FixedOffset(0);

fn date(s: &str) -> i32 {
    let a = StringArray::from(vec![s]);
    let out = cast(
        &a,
        &DataType::Date32,
        &CastOptions {
            mode: CastMode::Ansi,
            time_zone: &UTC,
        },
    )
    .unwrap();
    out.as_primitive::<Date32Type>().value(0)
}

fn ts(s: &str) -> i64 {
    let a = StringArray::from(vec![s]);
    let to = DataType::Timestamp(TimeUnit::Microsecond, None);
    let out = cast(
        &a,
        &to,
        &CastOptions {
            mode: CastMode::Ansi,
            time_zone: &UTC,
        },
    )
    .unwrap();
    out.as_primitive::<TimestampMicrosecondType>().value(0)
}

fn int_field(part: DatePart, d: i32) -> i32 {
    let out = extract_date(part, &Date32Array::from(vec![d])).unwrap();
    out.as_primitive::<Int32Type>().value(0)
}

fn dates(v: &[i32]) -> Date32Array {
    Date32Array::from(v.to_vec())
}

#[test]
fn date_fields_match_spark() {
    use DatePart::{DayOfWeek, DayOfWeekIso, DayOfYear, Month, Quarter, Week, Year, YearOfWeek};
    // (date, year, yearofweek, quarter, month, week, day, dayofweek, dow_iso, doy)
    let rows: &[(&str, [i32; 9])] = &[
        ("2024-03-10", [2024, 2024, 1, 3, 10, 10, 1, 7, 70]),
        ("2021-01-01", [2021, 2020, 1, 1, 53, 1, 6, 5, 1]),
        ("2020-12-31", [2020, 2020, 4, 12, 53, 31, 5, 4, 366]),
        ("2024-12-31", [2024, 2025, 4, 12, 1, 31, 3, 2, 366]),
        ("2018-12-31", [2018, 2019, 4, 12, 1, 31, 2, 1, 365]),
        ("2016-01-03", [2016, 2015, 1, 1, 53, 3, 1, 7, 3]),
        ("2016-01-04", [2016, 2016, 1, 1, 1, 4, 2, 1, 4]),
        ("0001-01-01", [1, 1, 1, 1, 1, 1, 2, 1, 1]),
        ("-0001-12-31", [-1, -1, 4, 12, 52, 31, 6, 5, 365]),
        ("9999-12-31", [9999, 9999, 4, 12, 52, 31, 6, 5, 365]),
        ("+10000-01-01", [10000, 9999, 1, 1, 52, 1, 7, 6, 1]),
        ("1582-10-10", [1582, 1582, 4, 10, 40, 10, 1, 7, 283]),
        ("1970-01-01", [1970, 1970, 1, 1, 1, 1, 5, 4, 1]),
        ("1969-12-31", [1969, 1970, 4, 12, 1, 31, 4, 3, 365]),
        ("2000-02-29", [2000, 2000, 1, 2, 9, 29, 3, 2, 60]),
        ("1900-03-01", [1900, 1900, 1, 3, 9, 1, 5, 4, 60]),
    ];
    let order = [
        Year,
        YearOfWeek,
        Quarter,
        Month,
        Week,
        DatePart::Day,
        DayOfWeek,
        DayOfWeekIso,
        DayOfYear,
    ];
    for (text, want) in rows {
        let d = date(text);
        let got: Vec<i32> = order.iter().map(|p| int_field(*p, d)).collect();
        assert_eq!(got, want, "{text}");
        // The time fields of a DATE are zero.
        assert_eq!(int_field(DatePart::Hour, d), 0);
        assert_eq!(int_field(DatePart::Minute, d), 0);
    }
    let a = Date32Array::from(vec![Some(date("2024-03-10")), None]);
    let out = extract_date(Month, &a).unwrap();
    assert_eq!(
        (out.as_primitive::<Int32Type>().value(0), out.is_null(1)),
        (3, true)
    );
    let secs = extract_date(DatePart::Second, &a).unwrap();
    assert_eq!(secs.data_type(), &DataType::Decimal128(8, 6));
    assert_eq!(secs.as_primitive::<Decimal128Type>().value(0), 0);
}

#[test]
fn timestamp_fields_match_spark() {
    let zone: Option<&dyn krishiv_kernels::cast::TimeZoneRules> = None;
    let field = |part, t: &str| {
        let a = TimestampMicrosecondArray::from(vec![ts(t)]);
        extract_timestamp(part, &a, zone).unwrap()
    };
    let int = |part, t: &str| field(part, t).as_primitive::<Int32Type>().value(0);
    let sec = |t: &str| {
        field(DatePart::Second, t)
            .as_primitive::<Decimal128Type>()
            .value(0)
    };
    assert_eq!(int(DatePart::Hour, "2024-03-10 13:45:56.789012"), 13);
    assert_eq!(int(DatePart::Minute, "2024-03-10 13:45:56.789012"), 45);
    assert_eq!(sec("2024-03-10 13:45:56.789012"), 56_789_012);
    assert_eq!(sec("1969-12-31 23:59:59.999999"), 59_999_999);
    assert_eq!(int(DatePart::Hour, "1969-12-31 23:59:59.999999"), 23);
    assert_eq!(int(DatePart::Day, "1969-12-31 23:59:59.999999"), 31);
    assert_eq!(int(DatePart::Year, "0001-01-01 00:00:00"), 1);
    assert_eq!(sec("0001-01-01 00:00:00"), 0);
    assert_eq!(int(DatePart::Day, "2024-11-03 01:30:00"), 3);
    // An instant in a zone: Los Angeles (-08:00) is still on the 9th.
    let a = TimestampMicrosecondArray::from(vec![Some(ts("2024-03-10 01:30:00")), None]);
    let la = FixedOffset(-8 * 3600);
    let out = extract_timestamp(DatePart::Day, &a, Some(&la)).unwrap();
    assert_eq!(out.as_primitive::<Int32Type>().value(0), 9);
    assert!(out.is_null(1));
    let out = extract_timestamp(DatePart::Hour, &a, Some(&la)).unwrap();
    assert_eq!(out.as_primitive::<Int32Type>().value(0), 17);
    let edge = TimestampMicrosecondArray::from(vec![i64::MAX]);
    let e = extract_timestamp(DatePart::Year, &edge, Some(&FixedOffset(3600))).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DatetimeOverflow);
}

#[test]
fn day_arithmetic_wraps_with_ansi_off_and_raises_with_ansi_on() {
    let ctx = KernelCtx::new();
    let start = date("2016-07-30");
    let d = |days| Operand::<Int32Type>::Scalar(Some(days));
    let run = |sub: bool, days: i32, mode| {
        let s = Operand::Array(&dates(&[start]));
        if sub {
            date_sub(s, d(days), mode, &ctx)
        } else {
            date_add(s, d(days), mode, &ctx)
        }
    };
    for mode in [OverflowMode::JavaWrap, OverflowMode::AnsiError] {
        assert_eq!(run(false, 1, mode).unwrap().value(0), 17_013);
        assert_eq!(run(true, 1, mode).unwrap().value(0), 17_011);
    }
    // ANSI off: Java `int` wrap, Spark's value.
    let wrap = OverflowMode::JavaWrap;
    assert_eq!(run(false, i32::MAX, wrap).unwrap().value(0), -2_147_466_637);
    assert_eq!(run(false, i32::MIN, wrap).unwrap().value(0), -2_147_466_636);
    assert_eq!(run(true, i32::MIN, wrap).unwrap().value(0), -2_147_466_636);
    // ANSI on: ARITHMETIC_OVERFLOW (Spark wraps silently, ADR 0023).
    for (sub, days) in [(false, i32::MAX), (true, i32::MIN)] {
        let e = run(sub, days, OverflowMode::AnsiError).unwrap_err();
        assert_eq!(
            (e.row(), e.error_class()),
            (Some(0), ErrorClass::ArithmeticOverflow)
        );
        assert!(
            e.to_string()
                .starts_with("[ARITHMETIC_OVERFLOW] overflow. If necessary")
        );
        // try_add / try_subtract: NULL.
        assert!(
            run(sub, days, OverflowMode::NullOnOverflow)
                .unwrap()
                .is_null(0)
        );
    }
    // The 4.0 and 4.1 text.
    let old = KernelCtx::new().with_spark_version(SparkVersion::V4_0);
    let s = Operand::Array(&dates(&[start]));
    let e = date_add(s, d(i32::MAX), OverflowMode::AnsiError, &old).unwrap_err();
    assert!(
        e.to_string()
            .starts_with("[ARITHMETIC_OVERFLOW] integer overflow.")
    );
    // NULL rows never raise.
    let a = Date32Array::from(vec![Some(i32::MAX), None]);
    let n = Int32Array::from(vec![Some(0), Some(i32::MAX)]);
    let out = date_add(
        Operand::Array(&a),
        Operand::Array(&n),
        OverflowMode::AnsiError,
        &ctx,
    )
    .unwrap();
    assert_eq!((out.value(0), out.is_null(1)), (i32::MAX, true));
    let diff = |e: i32, s: i32, mode| {
        date_diff(
            Operand::Array(&dates(&[e])),
            Operand::Scalar(Some(s)),
            mode,
            &ctx,
        )
    };
    let ansi = OverflowMode::AnsiError;
    assert_eq!(
        diff(date("2024-02-29"), date("2009-07-30"), ansi)
            .unwrap()
            .value(0),
        5327
    );
    assert_eq!(
        diff(date("2009-07-31"), date("2024-02-29"), wrap)
            .unwrap()
            .value(0),
        -5326
    );
    assert_eq!(diff(i32::MAX, -1, wrap).unwrap().value(0), i32::MIN);
    assert_eq!(
        diff(i32::MAX, -1, ansi).unwrap_err().error_class(),
        ErrorClass::ArithmeticOverflow
    );
    assert!(
        diff(i32::MAX, -1, OverflowMode::NullOnOverflow)
            .unwrap()
            .is_null(0)
    );
    let e = date_diff(
        Operand::Scalar(Some(1)),
        Operand::Scalar(Some(1)),
        ansi,
        &ctx,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn year_month_intervals_clamp_the_day() {
    let ctx = KernelCtx::new();
    let add = |d: &str, months: i32, negate: bool| {
        let out = date_add_months(
            Operand::Array(&dates(&[date(d)])),
            Operand::Scalar(Some(months)),
            negate,
            false,
            &ctx,
        )
        .unwrap();
        out.value(0)
    };
    assert_eq!(add("2020-01-31", 1, false), date("2020-02-29"));
    assert_eq!(add("2020-03-31", 1, true), date("2020-02-29"));
    assert_eq!(add("2020-02-29", 12, false), date("2021-02-28"));
    assert_eq!(add("1998-12-01", 0, false), date("1998-12-01"));
    assert_eq!(add("1993-07-01", 3, false), date("1993-10-01"));
    assert_eq!(add("9999-12-31", 1, false), date("+10000-01-31"));
    assert_eq!(add("0001-01-01", 1, true), date("0000-12-01"));
    // Overflow: Spark raises ARITHMETIC_OVERFLOW with its per-version text.
    for (v, text) in [
        (SparkVersion::V4_0, "integer overflow"),
        (SparkVersion::V4_1, "integer overflow"),
        (SparkVersion::V4_2, "overflow"),
    ] {
        let ctx = KernelCtx::new().with_spark_version(v);
        let e = date_add_months(
            Operand::Array(&dates(&[date("2020-01-01")])),
            Operand::Scalar(Some(i32::MAX)),
            false,
            false,
            &ctx,
        )
        .unwrap_err();
        assert_eq!(
            (e.row(), e.error_class()),
            (Some(0), ErrorClass::ArithmeticOverflow)
        );
        assert!(
            e.to_string()
                .starts_with(&format!("[ARITHMETIC_OVERFLOW] {text}. If necessary"))
        );
    }
    // try_add: NULL instead.
    let a = dates(&[date("2020-01-01"), date("2020-01-01")]);
    let m = Int32Array::from(vec![Some(i32::MAX), Some(1)]);
    let out = date_add_months(Operand::Array(&a), Operand::Array(&m), false, true, &ctx).unwrap();
    assert_eq!((out.is_null(0), out.value(1)), (true, date("2020-02-01")));
}

#[test]
fn timestamp_intervals() {
    type Zone<'a> = Option<&'a dyn krishiv_kernels::cast::TimeZoneRules>;
    let months = |s: &str, m: i32, negate: bool, zone: Zone<'_>| {
        let t = TimestampMicrosecondArray::from(vec![ts(s)]);
        let (m, z) = (Operand::Scalar(Some(m)), zone);
        timestamp_add_months(Operand::Array(&t), m, negate, z, false).map(|a| a.value(0))
    };
    let micros = |s: &str, m: i64, negate: bool, zone: Zone<'_>| {
        let t = TimestampMicrosecondArray::from(vec![ts(s)]);
        let m = Operand::Scalar(Some(m));
        timestamp_add_day_time(Operand::Array(&t), m, negate, zone, false).map(|a| a.value(0))
    };
    let at = |s| ts(s);
    assert_eq!(
        months("2020-01-31 10:00:00", 1, false, None).unwrap(),
        at("2020-02-29 10:00:00")
    );
    assert_eq!(
        months("2020-01-31 10:00:00", 1, true, None).unwrap(),
        at("2019-12-31 10:00:00")
    );
    assert_eq!(
        months("2020-01-01 10:00:00", 14, false, None).unwrap(),
        at("2021-03-01 10:00:00")
    );
    // In a zone the calendar arithmetic is on the local time.
    let la = FixedOffset(-8 * 3600);
    let local = months("2020-01-31 18:00:00", 1, false, Some(&la)).unwrap();
    assert_eq!(local, at("2020-02-29 18:00:00"));
    let (day, hour) = (86_400_000_000_i64, 3_600_000_000_i64);
    assert_eq!(
        micros("2020-01-01 10:00:00", 90 * day, true, None).unwrap(),
        at("2019-10-03 10:00:00")
    );
    assert_eq!(
        micros("2020-01-01 10:00:00", 36 * hour, false, None).unwrap(),
        at("2020-01-02 22:00:00")
    );
    // Overflow: DATETIME_OVERFLOW, or NULL for try_add.
    let max = TimestampMicrosecondArray::from(vec![i64::MAX]);
    let add = |m: i64, negate: bool, null| {
        let m = Operand::Scalar(Some(m));
        timestamp_add_day_time(Operand::Array(&max), m, negate, None, null)
    };
    assert_eq!(
        add(1, false, false).unwrap_err().error_class(),
        ErrorClass::DatetimeOverflow
    );
    assert!(add(1, false, true).unwrap().is_null(0));
    assert_eq!(
        add(i64::MIN, true, false).unwrap_err().error_class(),
        ErrorClass::DatetimeOverflow
    );
    let e = timestamp_add_months(
        Operand::Array(&max),
        Operand::Scalar(Some(1)),
        false,
        None,
        false,
    );
    assert_eq!(e.unwrap_err().error_class(), ErrorClass::DatetimeOverflow);
    // NULLs, mismatched lengths and two literals.
    let a = TimestampMicrosecondArray::from(vec![Some(0), None]);
    let ms = Int64Array::from(vec![Some(1), Some(1)]);
    let out = timestamp_add_day_time(Operand::Array(&a), Operand::Array(&ms), false, None, false)
        .unwrap();
    assert_eq!((out.value(0), out.is_null(1)), (1, true));
    let short = Int64Array::from(vec![1]);
    let uneven = timestamp_add_day_time(
        Operand::Array(&a),
        Operand::Array(&short),
        false,
        None,
        false,
    );
    let (one, two) = (Operand::Scalar(Some(1)), Operand::Scalar(Some(1)));
    let literals = timestamp_add_day_time(one, two, false, None, false);
    for e in [uneven.unwrap_err(), literals.unwrap_err()] {
        assert_eq!(e.error_class(), ErrorClass::InternalError);
    }
}

#[test]
fn timestamp_strings_that_need_the_query_clock() {
    let ltz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    let ntz = DataType::Timestamp(TimeUnit::Microsecond, None);
    let run = |s: &str, to: &DataType, mode| {
        let opts = CastOptions {
            mode,
            time_zone: &UTC,
        };
        cast(&StringArray::from(vec![s]), to, &opts)
    };
    // A region id resolves against the bundled database: midnight in Los
    // Angeles (UTC-08:00 in January) is 08:00 UTC. A time of day needs the
    // query clock.
    let out = run(
        "2020-01-01 00:00:00 America/Los_Angeles",
        &ltz,
        CastMode::Ansi,
    )
    .unwrap();
    assert_eq!(
        out.as_primitive::<TimestampMicrosecondType>().value(0),
        18_262 * 86_400_000_000 + 8 * 3_600_000_000
    );
    for s in ["10:00", "T10:00:00"] {
        let e = run(s, &ltz, CastMode::Ansi).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{s}"
        );
    }
    // Spark reads a bare time of day as today's for TIMESTAMP only.
    let e = run("10:00", &ntz, CastMode::Ansi).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::CastInvalidInput);
    assert!(run("10:00", &ntz, CastMode::Try).unwrap().is_null(0));
    // The offset moves the instant out of the microsecond range.
    let edge = "+294247-01-10 04:00:54.775807 -01:00";
    assert!(run(edge, &ltz, CastMode::Try).unwrap().is_null(0));
    assert!(run(edge, &ltz, CastMode::Ansi).is_err());
}

#[test]
fn the_current_date_is_the_session_zones() {
    // 2024-03-10 01:30:00 UTC.
    let now = ts("2024-03-10 01:30:00");
    assert_eq!(current_date(now, &UTC).unwrap(), date("2024-03-10"));
    assert_eq!(
        current_date(now, &FixedOffset(-8 * 3600)).unwrap(),
        date("2024-03-09")
    );
    assert_eq!(
        current_date(now, &FixedOffset(14 * 3600)).unwrap(),
        date("2024-03-10")
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(miri) { 2 } else { 512 }, failure_persistence: None, ..ProptestConfig::default() })]

    /// `date - date` and `date + days` are inverse, and `extract` agrees with
    /// a civil-calendar round trip through the string cast.
    #[test]
    fn dates_round_trip(d in -700_000_i32..3_000_000, days in -100_000_i32..100_000) {
        let ctx = KernelCtx::new();
        let start = dates(&[d]);
        let moved = date_add(Operand::Array(&start), Operand::Scalar(Some(days)), OverflowMode::AnsiError, &ctx).unwrap();
        let back = date_diff(Operand::Array(&moved), Operand::Array(&start), OverflowMode::AnsiError, &ctx).unwrap();
        prop_assert_eq!(back.value(0), days);
        // Year, month and day rebuild the date.
        let y = int_field(DatePart::Year, d);
        let m = int_field(DatePart::Month, d);
        let dd = int_field(DatePart::Day, d);
        let text = if y >= 0 { format!("{y:04}-{m:02}-{dd:02}") } else { format!("-{:04}-{m:02}-{dd:02}", -y) };
        let text = if y >= 10_000 { format!("+{text}") } else { text };
        prop_assert_eq!(date(&text), d);
        // ISO weekday cycles with the date; day of week is the same day shifted.
        let iso = int_field(DatePart::DayOfWeekIso, d);
        let dow = int_field(DatePart::DayOfWeek, d);
        prop_assert!((1..=7).contains(&iso));
        prop_assert_eq!(dow, iso % 7 + 1);
        let week = int_field(DatePart::Week, d);
        prop_assert!((1..=53).contains(&week));
        let doy = int_field(DatePart::DayOfYear, d);
        prop_assert!((1..=366).contains(&doy));
    }

    /// Adding `n` months then subtracting them lands in the same month, and
    /// the day only ever clamps down.
    #[test]
    fn months_keep_the_calendar(d in -600_000_i32..2_900_000, n in -2_000_i32..2_000) {
        let ctx = KernelCtx::new();
        let start = dates(&[d]);
        let up = date_add_months(Operand::Array(&start), Operand::Scalar(Some(n)), false, false, &ctx).unwrap();
        let month_of = |d: i32| {
            i64::from(int_field(DatePart::Year, d)) * 12 + i64::from(int_field(DatePart::Month, d)) - 1
        };
        prop_assert_eq!(month_of(up.value(0)), month_of(d) + i64::from(n));
        prop_assert!(int_field(DatePart::Day, up.value(0)) <= int_field(DatePart::Day, d));
        let back = date_add_months(Operand::Array(&up), Operand::Scalar(Some(n)), true, false, &ctx).unwrap();
        prop_assert_eq!(month_of(back.value(0)), month_of(d));
    }
}
