// SPDX-License-Identifier: Apache-2.0
//! The calendar bindings: field accessors, `add_months`, `last_day`,
//! `next_day`, `trunc`, `date_trunc`, `months_between`, `make_date`, the
//! zone conversions and the string carriers they read.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Float64Type, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Int8Array, LargeStringArray, StringArray, StringViewArray,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};
use krishiv_kernels::{KernelCtx, SparkVersion};

use super::{
    BindArg, BindOptions, bind, bind_scalar, bind_with, class, col, dates, int_values, ints, lit,
    ltz, ntz, run, strs, text,
};

fn day(s: &str) -> i32 {
    let zone = FixedOffset(0);
    let options = CastOptions {
        mode: CastMode::Ansi,
        time_zone: &zone,
    };
    let out = cast(&StringArray::from(vec![s]), &DataType::Date32, &options).unwrap();
    out.as_primitive::<Date32Type>().value(0)
}

/// A wall clock (or a UTC instant) as microseconds.
fn micros(s: &str) -> i64 {
    let zone = FixedOffset(0);
    let options = CastOptions {
        mode: CastMode::Ansi,
        time_zone: &zone,
    };
    let to = DataType::Timestamp(TimeUnit::Microsecond, None);
    let out = cast(&StringArray::from(vec![s]), &to, &options).unwrap();
    out.as_primitive::<TimestampMicrosecondType>().value(0)
}

fn date_values(a: &ArrayRef) -> Vec<Option<i32>> {
    a.as_primitive::<Date32Type>().iter().collect()
}

fn stamp_values(a: &ArrayRef) -> Vec<Option<i64>> {
    a.as_primitive::<TimestampMicrosecondType>()
        .iter()
        .collect()
}

fn plus_one() -> BindOptions {
    BindOptions::new().with_time_zone(Arc::new(FixedOffset(3600)))
}

fn legacy() -> BindOptions {
    BindOptions::new().with_ansi(false)
}

#[test]
fn field_accessors_of_dates_and_timestamps() {
    // 2024-02-29 is a Thursday, the 60th day, in ISO week 9.
    let d = dates(&[Some(day("2024-02-29")), None, Some(day("1582-10-15"))]);
    assert_eq!(
        int_values(&run("dayofweek", &[&d])),
        [Some(5), None, Some(6)]
    );
    assert_eq!(int_values(&run("weekday", &[&d])), [Some(3), None, Some(4)]);
    assert_eq!(
        int_values(&run("dayofyear", &[&d])),
        [Some(60), None, Some(288)]
    );
    assert_eq!(
        int_values(&run("weekofyear", &[&d])),
        [Some(9), None, Some(41)]
    );
    assert_eq!(int_values(&run("quarter", &[&d])), [Some(1), None, Some(4)]);
    // Spark: weekday('2007-02-03') is 5 (a Saturday), dayofweek is 7.
    let sat = dates(&[Some(day("2007-02-03"))]);
    assert_eq!(int_values(&run("weekday", &[&sat])), [Some(5)]);
    assert_eq!(int_values(&run("dayofweek", &[&sat])), [Some(7)]);
    // A timestamp reads its fields in the session zone, a timestamp_ntz as is.
    let t = micros("2024-02-29 23:34:56.123456");
    let instant = ltz(&[Some(t), None]);
    let f = bind_with("dayofweek", &[&instant], &plus_one());
    assert_eq!(
        int_values(&f.invoke(&[col(&instant)], 2).unwrap()),
        [Some(6), None]
    );
    let wall = ntz(&[Some(t)]);
    let f = bind_with("dayofweek", &[&wall], &plus_one());
    assert_eq!(int_values(&f.invoke(&[col(&wall)], 1).unwrap()), [Some(5)]);
    for (name, utc, local) in [("hour", 23, 0), ("minute", 34, 34), ("second", 56, 56)] {
        assert_eq!(
            int_values(&run(name, &[&instant])),
            [Some(utc), None],
            "{name}"
        );
        let f = bind_with(name, &[&instant], &plus_one());
        assert_eq!(
            int_values(&f.invoke(&[col(&instant)], 2).unwrap()),
            [Some(local), None],
            "{name} in +01:00"
        );
        assert_eq!(int_values(&run(name, &[&wall])), [Some(utc)], "{name} ntz");
    }
    // `second` drops the microseconds; the end of a minute stays 59.
    let edge = ltz(&[
        Some(micros("1969-12-31 23:59:59.999999")),
        Some(micros("0001-01-01 00:00:00")),
    ]);
    assert_eq!(int_values(&run("second", &[&edge])), [Some(59), Some(0)]);
    assert_eq!(int_values(&run("minute", &[&edge])), [Some(59), Some(0)]);
    assert_eq!(f_type("second", &edge), DataType::Int32);
}

fn f_type(name: &str, arg: &ArrayRef) -> DataType {
    bind(name, &[arg]).return_type().clone()
}

#[test]
fn names_and_unix_dates() {
    let d = dates(&[
        Some(day("2024-02-29")),
        None,
        Some(day("1582-10-10")),
        Some(day("0001-01-01")),
    ]);
    assert_eq!(
        text(&run("monthname", &[&d])),
        [
            Some("Feb".to_owned()),
            None,
            Some("Oct".to_owned()),
            Some("Jan".to_owned())
        ]
    );
    // Spark: dayname(DATE'1582-10-10') is Sun.
    assert_eq!(
        text(&run("dayname", &[&d])),
        [
            Some("Thu".to_owned()),
            None,
            Some("Sun".to_owned()),
            Some("Mon".to_owned())
        ]
    );
    assert_eq!(f_type("dayname", &d), DataType::Utf8);
    assert_eq!(
        int_values(&run("unix_date", &[&d])),
        [Some(19_782), None, Some(-141_432), Some(-719_162)]
    );
}

#[test]
fn add_months_clamps_and_raises_in_every_mode() {
    let d = dates(&[Some(day("2016-08-31")), None, Some(day("2020-01-31"))]);
    let n = ints(&[Some(1), Some(1), Some(1)]);
    assert_eq!(
        date_values(&run("add_months", &[&d, &n])),
        [Some(day("2016-09-30")), None, Some(day("2020-02-29"))]
    );
    // Months of any narrow width, negative, and from a literal.
    let small: ArrayRef = Arc::new(Int8Array::from(vec![Some(-1), None, Some(12)]));
    assert_eq!(
        date_values(&run("add_months", &[&d, &small])),
        [Some(day("2016-07-31")), None, Some(day("2021-01-31"))]
    );
    let one = ints(&[Some(2)]);
    let f = bind("add_months", &[&d, &one]);
    assert_eq!(f.return_type(), &DataType::Date32);
    assert_eq!(
        date_values(&f.invoke(&[col(&d), lit(&one)], 3).unwrap()),
        [Some(day("2016-10-31")), None, Some(day("2020-03-31"))]
    );
    // An overflowing result is ARITHMETIC_OVERFLOW whatever ANSI says, with
    // the text of the Spark version.
    let far = ints(&[Some(i32::MAX)]);
    for opts in [BindOptions::new(), legacy()] {
        let f = bind_with("add_months", &[&d, &far], &opts);
        let e = f.invoke(&[col(&d), lit(&far)], 3).unwrap_err();
        assert_eq!(class(&e), ErrorClass::ArithmeticOverflow);
        assert!(e.to_string().contains("overflow. If necessary"), "{e}");
    }
    let v40 =
        BindOptions::new().with_kernel(KernelCtx::new().with_spark_version(SparkVersion::V4_0));
    let f = bind_with("add_months", &[&d, &far], &v40);
    let e = f.invoke(&[col(&d), lit(&far)], 3).unwrap_err();
    assert!(e.to_string().contains("integer overflow."), "{e}");
    // A NULL row never raises.
    let lone = dates(&[None]);
    let f = bind_with("add_months", &[&lone, &far], &legacy());
    assert_eq!(
        f.invoke(&[col(&lone), lit(&far)], 1).unwrap().null_count(),
        1
    );
}

#[test]
fn last_day_of_the_month() {
    let d = dates(&[
        Some(day("2009-01-12")),
        Some(day("2024-02-10")),
        Some(day("2023-02-10")),
        None,
    ]);
    assert_eq!(
        date_values(&run("last_day", &[&d])),
        [
            Some(day("2009-01-31")),
            Some(day("2024-02-29")),
            Some(day("2023-02-28")),
            None
        ]
    );
}

/// Calls `name(first, second)` with the second as a column of every string
/// carrier and returns the three results, which must agree.
fn over_string_carriers(name: &str, first: &ArrayRef, second: &[Option<&str>]) -> Vec<ArrayRef> {
    let carriers: [ArrayRef; 3] = [
        Arc::new(StringArray::from(second.to_vec())),
        Arc::new(LargeStringArray::from(second.to_vec())),
        Arc::new(StringViewArray::from(second.to_vec())),
    ];
    carriers
        .iter()
        .map(|c| {
            let f = bind_with(name, &[first, c], &BindOptions::new().with_ansi(false));
            f.invoke(&[col(first), col(c)], second.len()).unwrap()
        })
        .collect()
}

#[test]
fn next_day_reads_every_string_carrier_and_follows_ansi() {
    let start = day("2015-07-23");
    let d = dates(&[Some(start), Some(start), Some(start), None, Some(start)]);
    let names = [Some("Mon"), Some("xx"), None, Some("TU"), Some("sunday")];
    let outs = over_string_carriers("next_day", &d, &names);
    let want = [
        Some(day("2015-07-27")),
        None,
        None,
        None,
        Some(day("2015-07-26")),
    ];
    for out in &outs {
        assert_eq!(date_values(out), want);
    }
    // ANSI on raises ILLEGAL_DAY_OF_WEEK for the bad name, on that row only.
    let names = strs(&names);
    let f = bind("next_day", &[&d, &names]);
    assert_eq!(f.return_type(), &DataType::Date32);
    let e = f.invoke(&[col(&d), col(&names)], 5).unwrap_err();
    assert_eq!(class(&e), ErrorClass::IllegalDayOfWeek);
    // A NULL day name, or NULL date, hides the bad one beside it.
    let (d, n) = (dates(&[None, Some(start)]), strs(&[Some("xx"), None]));
    let f = bind("next_day", &[&d, &n]);
    assert_eq!(f.invoke(&[col(&d), col(&n)], 2).unwrap().null_count(), 2);
    // A literal weekday name.
    let lit_name = strs(&[Some("fri")]);
    let d = dates(&[Some(start)]);
    let f = bind("next_day", &[&d, &lit_name]);
    let out = f.invoke(&[col(&d), lit(&lit_name)], 1).unwrap();
    assert_eq!(date_values(&out), [Some(start + 1)]);
}

#[test]
fn trunc_reads_literal_and_column_formats() {
    let d = dates(&[
        Some(day("2019-08-04")),
        Some(day("2019-08-04")),
        Some(day("2019-08-04")),
        None,
    ]);
    let formats = [Some("week"), Some("DAY"), None, Some("year")];
    for out in over_string_carriers("trunc", &d, &formats) {
        assert_eq!(
            date_values(&out),
            [Some(day("2019-07-29")), None, None, None]
        );
    }
    let lit_format = strs(&[Some("MM")]);
    let f = bind("trunc", &[&d, &lit_format]);
    assert_eq!(f.return_type(), &DataType::Date32);
    let out = f.invoke(&[col(&d), lit(&lit_format)], 4).unwrap();
    assert_eq!(
        date_values(&out),
        [
            Some(day("2019-08-01")),
            Some(day("2019-08-01")),
            Some(day("2019-08-01")),
            None
        ]
    );
    let null_format = strs(&[None]);
    let f = bind("trunc", &[&d, &null_format]);
    assert_eq!(
        f.invoke(&[col(&d), lit(&null_format)], 4)
            .unwrap()
            .null_count(),
        4
    );
}

#[test]
fn date_trunc_cuts_instants_in_the_session_zone() {
    // 2015-03-05 09:32:05.123456 UTC; at +01:00 it is 10:32.
    let t = micros("2015-03-05 09:32:05.123456");
    let instants = ltz(&[Some(t), Some(t), None, Some(t)]);
    let formats = [Some("hour"), Some("YEAR"), Some("DAY"), None];
    let carriers: [ArrayRef; 3] = [
        Arc::new(StringArray::from(formats.to_vec())),
        Arc::new(LargeStringArray::from(formats.to_vec())),
        Arc::new(StringViewArray::from(formats.to_vec())),
    ];
    let utc_type = DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC")));
    for (opts, offset) in [(BindOptions::new(), 0), (plus_one(), 3_600_000_000)] {
        for format in &carriers {
            let types = [
                format.data_type().clone().into(),
                instants.data_type().clone().into(),
            ];
            let f = bind_scalar("date_trunc", &types, &opts).unwrap();
            assert_eq!(f.return_type(), &utc_type);
            let out = f.invoke(&[col(format), col(&instants)], 4).unwrap();
            assert_eq!(
                stamp_values(&out),
                [
                    Some(micros("2015-03-05 09:00:00")),
                    Some(micros("2015-01-01 00:00:00") - offset),
                    None,
                    None
                ]
            );
        }
    }
    // A timestamp_ntz is read in the session zone first: 09:32 at +01:00 is
    // 08:32 UTC on the same day, and the day starts at midnight +01:00.
    let wall = ntz(&[Some(t), Some(t), None, Some(t)]);
    let name = strs(&[Some("DD")]);
    let types = [
        BindArg::Literal(Arc::clone(&name)),
        wall.data_type().clone().into(),
    ];
    let f = bind_scalar("date_trunc", &types, &plus_one()).unwrap();
    let out = f.invoke(&[lit(&name), col(&wall)], 4).unwrap();
    let midnight = micros("2015-03-05 00:00:00") - 3_600_000_000;
    assert_eq!(
        stamp_values(&out),
        [Some(midnight), Some(midnight), None, Some(midnight)]
    );
}

#[test]
fn months_between_defaults_to_rounding() {
    let end = ltz(&[
        Some(micros("1997-02-28 10:30:00")),
        None,
        Some(micros("2024-02-29 12:34:56.123456")),
    ]);
    let start = ltz(&[
        Some(micros("1996-10-30 00:00:00")),
        Some(0),
        Some(micros("1996-10-30 00:00:00")),
    ]);
    let out = run("months_between", &[&end, &start]);
    let values: Vec<Option<f64>> = out.as_primitive::<Float64Type>().iter().collect();
    assert_eq!(values, [Some(3.94959677), None, Some(327.98465352)]);
    // roundOff as a column or a literal.
    let round = Arc::new(BooleanArray::from(vec![
        Some(false),
        Some(true),
        Some(false),
    ])) as ArrayRef;
    let f = bind("months_between", &[&end, &start, &round]);
    assert_eq!(f.return_type(), &DataType::Float64);
    let out = f.invoke(&[col(&end), col(&start), col(&round)], 3).unwrap();
    let values: Vec<Option<f64>> = out.as_primitive::<Float64Type>().iter().collect();
    assert_eq!(
        values,
        [Some(3.9495967741935485), None, Some(327.9846535244922)]
    );
    let no = Arc::new(BooleanArray::from(vec![false])) as ArrayRef;
    let f = bind("months_between", &[&end, &start, &no]);
    let out = f.invoke(&[col(&end), col(&start), lit(&no)], 3).unwrap();
    assert_eq!(
        out.as_primitive::<Float64Type>().value(0),
        3.9495967741935485
    );
    // A timestamp_ntz is read in the session zone: at +01:00 the wall clock
    // 10:30 is 09:30 UTC, the same day, so the answer does not move.
    let wall = ntz(&[Some(micros("1997-02-28 10:30:00"))]);
    let wall_start = ntz(&[Some(micros("1996-10-30 00:00:00"))]);
    let f = bind_with("months_between", &[&wall, &wall_start], &plus_one());
    let out = f.invoke(&[col(&wall), col(&wall_start)], 1).unwrap();
    assert_eq!(out.as_primitive::<Float64Type>().value(0), 3.94959677);
    // A NULL roundOff gives NULL.
    let null_round = Arc::new(BooleanArray::from(vec![None])) as ArrayRef;
    let f = bind("months_between", &[&end, &start, &null_round]);
    let out = f
        .invoke(&[col(&end), col(&start), lit(&null_round)], 3)
        .unwrap();
    assert_eq!(out.null_count(), 3);
}

#[test]
fn make_date_follows_ansi_and_the_version() {
    let (y, m, d) = (
        ints(&[Some(2019), Some(2000), Some(2023), None]),
        ints(&[Some(1), Some(13), Some(2), Some(1)]),
        ints(&[Some(1), Some(1), Some(29), Some(1)]),
    );
    let f = bind_with("make_date", &[&y, &m, &d], &legacy());
    assert_eq!(f.return_type(), &DataType::Date32);
    let out = f.invoke(&[col(&y), col(&m), col(&d)], 4).unwrap();
    assert_eq!(
        date_values(&out),
        [Some(day("2019-01-01")), None, None, None]
    );
    let f = bind("make_date", &[&y, &m, &d]);
    let e = f.invoke(&[col(&y), col(&m), col(&d)], 4).unwrap_err();
    assert_eq!(
        class(&e),
        ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion
    );
    assert!(
        e.to_string()
            .contains("MonthOfYear (valid values 1 - 12): 13"),
        "{e}"
    );
    // Literal fields.
    let (y, m, d) = (ints(&[Some(12)]), ints(&[Some(12)]), ints(&[Some(12)]));
    let f = bind("make_date", &[&y, &m, &d]);
    let out = f.invoke(&[lit(&y), lit(&m), lit(&d)], 2).unwrap();
    assert_eq!(date_values(&out), [Some(day("0012-12-12")); 2]);
    // The overflow text follows the Spark version.
    let huge = ints(&[Some(6_000_000)]);
    let v41 =
        BindOptions::new().with_kernel(KernelCtx::new().with_spark_version(SparkVersion::V4_1));
    let f = bind_with("make_date", &[&huge, &m, &d], &v41);
    let e = f.invoke(&[lit(&huge), lit(&m), lit(&d)], 1).unwrap_err();
    assert_eq!(class(&e), ErrorClass::ArithmeticOverflow);
    assert!(e.to_string().contains("overflow."), "{e}");
}

#[test]
fn zone_conversions_with_fixed_offset_zones() {
    let instants = ltz(&[Some(micros("2016-08-31 00:00:00")), None]);
    let zone = strs(&[Some("+09:00")]);
    let args = |t: &ArrayRef| {
        [
            t.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&zone)),
        ]
    };
    let f = bind_scalar("from_utc_timestamp", &args(&instants), &BindOptions::new()).unwrap();
    assert_eq!(
        f.return_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC")))
    );
    let out = f.invoke(&[col(&instants), lit(&zone)], 2).unwrap();
    assert_eq!(
        stamp_values(&out),
        [Some(micros("2016-08-31 09:00:00")), None]
    );
    let f = bind_scalar("to_utc_timestamp", &args(&instants), &BindOptions::new()).unwrap();
    let out = f.invoke(&[col(&instants), lit(&zone)], 2).unwrap();
    assert_eq!(
        stamp_values(&out),
        [Some(micros("2016-08-30 15:00:00")), None]
    );
    // A timestamp_ntz is read in the session zone first.
    let wall = ntz(&[Some(micros("2016-08-31 01:00:00"))]);
    let f = bind_scalar("from_utc_timestamp", &args(&wall), &plus_one()).unwrap();
    let out = f.invoke(&[col(&wall), lit(&zone)], 1).unwrap();
    assert_eq!(stamp_values(&out), [Some(micros("2016-08-31 09:00:00"))]);
    // A NULL literal zone gives NULL for every row; the zone of a column is
    // read per row.
    let null = strs(&[None]);
    let f = bind_scalar(
        "from_utc_timestamp",
        &[
            instants.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&null)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    assert_eq!(
        f.invoke(&[col(&instants), lit(&null)], 2)
            .unwrap()
            .null_count(),
        2
    );
    let zones = strs(&[Some("UTC"), Some("-03:30")]);
    let both = ltz(&[Some(0), Some(0)]);
    let f = bind("from_utc_timestamp", &[&both, &zones]);
    let out = f.invoke(&[col(&both), col(&zones)], 2).unwrap();
    assert_eq!(stamp_values(&out), [Some(0), Some(-12_600_000_000)]);
    // A zone column with a region id fails when its row is read.
    let regions = strs(&[None, Some("Asia/Seoul")]);
    let f = bind("from_utc_timestamp", &[&both, &regions]);
    let e = f.invoke(&[col(&both), col(&regions)], 2).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    // ... and a repeated zone is resolved once (the answer is the same).
    let repeated = strs(&[Some("+01:00"), Some("+01:00"), Some("+02:00")]);
    let three = ltz(&[Some(0), Some(0), Some(0)]);
    let out = run("to_utc_timestamp", &[&three, &repeated]);
    assert_eq!(
        stamp_values(&out),
        [
            Some(-3_600_000_000),
            Some(-3_600_000_000),
            Some(-7_200_000_000)
        ]
    );
}

#[test]
fn convert_timezone_takes_two_or_three_arguments() {
    let wall = ntz(&[Some(micros("2022-01-01 00:00:00")), None]);
    let (from, to) = (strs(&[Some("+03:00")]), strs(&[Some("-08:00")]));
    let three = [
        BindArg::Literal(Arc::clone(&from)),
        BindArg::Literal(Arc::clone(&to)),
        wall.data_type().clone().into(),
    ];
    let f = bind_scalar("convert_timezone", &three, &BindOptions::new()).unwrap();
    assert_eq!(
        f.return_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    let out = f.invoke(&[lit(&from), lit(&to), col(&wall)], 2).unwrap();
    assert_eq!(
        stamp_values(&out),
        [Some(micros("2021-12-31 13:00:00")), None]
    );
    // Two arguments: the source is the session zone (+01:00 here).
    let two = [
        BindArg::Literal(Arc::clone(&to)),
        wall.data_type().clone().into(),
    ];
    let f = bind_scalar("convert_timezone", &two, &plus_one()).unwrap();
    let out = f.invoke(&[lit(&to), col(&wall)], 2).unwrap();
    assert_eq!(
        stamp_values(&out),
        [Some(micros("2021-12-31 15:00:00")), None]
    );
    // A timestamp (an instant) is read as the wall clock of the session zone.
    let instant = ltz(&[Some(micros("2022-01-01 00:00:00"))]);
    let two = [
        BindArg::Literal(Arc::clone(&to)),
        instant.data_type().clone().into(),
    ];
    let f = bind_scalar("convert_timezone", &two, &plus_one()).unwrap();
    assert_eq!(
        f.return_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    let out = f.invoke(&[lit(&to), col(&instant)], 1).unwrap();
    assert_eq!(
        stamp_values(&out),
        [Some(micros("2021-12-31 15:00:00") + 3_600_000_000)]
    );
    // Zone columns, with NULLs.
    let (sources, targets) = (
        strs(&[Some("UTC"), None, Some("UTC")]),
        strs(&[Some("+01:00"), Some("UTC"), None]),
    );
    let w3 = ntz(&[Some(0), Some(0), Some(0)]);
    let out = run("convert_timezone", &[&sources, &targets, &w3]);
    assert_eq!(stamp_values(&out), [Some(3_600_000_000), None, None]);
}

#[test]
fn convert_timezone_refuses_unresolvable_literal_zones_when_bound() {
    let wall = DataType::Timestamp(TimeUnit::Microsecond, None);
    let zone = |z: &str| BindArg::Literal(strs(&[Some(z)]));
    let refused = |args: Vec<BindArg>| {
        bind_scalar("convert_timezone", &args, &BindOptions::new()).unwrap_err()
    };
    // The target of the two-argument form, and either zone of the three.
    let e = refused(vec![zone("Europe/Brussels"), wall.clone().into()]);
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    let e = refused(vec![
        zone("UTC"),
        zone("America/Los_Angeles"),
        wall.clone().into(),
    ]);
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    let e = refused(vec![
        zone("Europe/Moscow"),
        zone("UTC"),
        wall.clone().into(),
    ]);
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    let e = refused(vec![zone("UTC"), zone(""), wall.into()]);
    assert_eq!(class(&e), ErrorClass::InvalidTimezone);
}

#[test]
fn months_between_overflow_is_raised_by_the_binding() {
    // The local time of the largest instant at +01:00 does not exist.
    let end = ltz(&[Some(i64::MAX)]);
    let start = ltz(&[Some(0)]);
    let f = bind_with("months_between", &[&end, &start], &plus_one());
    let e = f.invoke(&[col(&end), col(&start)], 1).unwrap_err();
    assert_eq!(class(&e), ErrorClass::DatetimeOverflow);
}

#[test]
fn zone_text_is_resolved_or_refused() {
    let t = ltz(&[Some(0)]);
    let bind_zone = |zone: &str| {
        let z = strs(&[Some(zone)]);
        bind_scalar(
            "from_utc_timestamp",
            &[t.data_type().clone().into(), BindArg::Literal(z)],
            &BindOptions::new(),
        )
    };
    for ok in [
        "UTC",
        "Z",
        "GMT",
        "+08:00",
        "-03:30",
        "UTC+05:30",
        "GMT-3",
        "+5",
    ] {
        assert!(bind_zone(ok).is_ok(), "{ok}");
    }
    // Region ids, short ids and offsets spelled another way need the tz
    // database (or a fuller offset parser): unsupported, never a wrong answer.
    for region in [
        "Asia/Seoul",
        "America/Los_Angeles",
        "CET",
        "PST",
        "abc",
        "true",
        "+0530",
        "UTC+05:30:15",
        "+19:00",
        "Etc/GMT+5",
    ] {
        let e = bind_zone(region).unwrap_err();
        assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature, "{region}");
    }
    // Text no zone can have: INVALID_TIMEZONE with the text as given.
    for bad in [
        "",
        "1",
        " UTC",
        "UTC ",
        "2024-02-29 12:34:56.123456",
        "1.5",
        "Ä€日本語😀é",
        " \t a\u{3000}\n ",
        "a b",
        "/Seoul",
    ] {
        let e = bind_zone(bad).unwrap_err();
        assert_eq!(class(&e), ErrorClass::InvalidTimezone, "{bad:?}");
        assert!(
            e.to_string().contains(bad.trim_matches(['\n', ' ', '\t'])) || bad.trim().is_empty(),
            "{e}"
        );
    }
    // A column zone raises the same errors per row.
    let zones = strs(&[Some("")]);
    let f = bind("from_utc_timestamp", &[&t, &zones]);
    let e = f.invoke(&[col(&t), col(&zones)], 1).unwrap_err();
    assert_eq!(class(&e), ErrorClass::InvalidTimezone);
}

#[test]
fn argument_types_the_calendar_functions_do_not_take() {
    let cases: [(&str, Vec<DataType>); 12] = [
        ("hour", vec![DataType::Date32]),
        ("weekday", vec![DataType::Int32]),
        ("monthname", vec![DataType::Utf8]),
        ("unix_date", vec![DataType::Int32]),
        ("add_months", vec![DataType::Date32, DataType::Int64]),
        ("next_day", vec![DataType::Date32, DataType::Int32]),
        ("trunc", vec![DataType::Utf8, DataType::Utf8]),
        ("date_trunc", vec![DataType::Utf8, DataType::Date32]),
        ("months_between", vec![DataType::Date32, DataType::Date32]),
        (
            "make_date",
            vec![DataType::Int64, DataType::Int32, DataType::Int32],
        ),
        ("from_utc_timestamp", vec![DataType::Date32, DataType::Utf8]),
        ("convert_timezone", vec![DataType::Utf8, DataType::Date32]),
    ];
    for (name, types) in cases {
        let args: Vec<BindArg> = types.into_iter().map(BindArg::from).collect();
        let e = bind_scalar(name, &args, &BindOptions::new()).unwrap_err();
        assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature, "{name}");
        assert!(e.to_string().contains(name), "{e}");
    }
    // The wrong number of arguments, and a zone literal of other than one row.
    let e = bind_scalar("last_day", &[], &BindOptions::new()).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    let e = bind_scalar(
        "months_between",
        &vec![BindArg::from(DataType::Timestamp(TimeUnit::Microsecond, None)); 4],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
}
