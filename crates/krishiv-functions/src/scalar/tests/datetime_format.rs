// SPDX-License-Identifier: Apache-2.0
//! The pattern-driven datetime bindings: `date_format`, `to_date`,
//! `to_timestamp` and its `_ltz`/`_ntz`/`try_` forms, `unix_timestamp`,
//! `to_unix_timestamp`, `from_unixtime` and `current_timezone`.
//!
//! Expected values pinned against Spark 4.2 runs (the function matrix and
//! the SQL goldens) are asserted as such; everything else follows Spark's
//! documented semantics and is marked unverified (no Spark oracle is
//! available here).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{Array, ArrayRef, Date32Array, Int64Array, TimestampMicrosecondArray};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::cast::FixedOffset;

use super::{
    BindArg, BindOptions, bind, bind_scalar, bind_with, class, col, lit, ltz, ntz, strs, text,
};

fn utc() -> BindOptions {
    BindOptions::new()
}

fn legacy() -> BindOptions {
    BindOptions::new().with_ansi(false)
}

fn plus_thirty() -> BindOptions {
    BindOptions::new()
        .with_time_zone(Arc::new(FixedOffset(19_800)))
        .with_time_zone_name(Arc::from("+05:30"))
}

fn stamps(a: &ArrayRef) -> Vec<Option<i64>> {
    a.as_primitive::<TimestampMicrosecondType>()
        .iter()
        .collect()
}

fn dates(a: &ArrayRef) -> Vec<Option<i32>> {
    a.as_primitive::<Date32Type>().iter().collect()
}

fn longs(a: &ArrayRef) -> Vec<Option<i64>> {
    a.as_primitive::<Int64Type>().iter().collect()
}

/// `name` over string column `values` with literal pattern `pattern`.
fn run_pat(name: &str, values: &[Option<&str>], pattern: &str, opts: &BindOptions) -> ArrayRef {
    let v = strs(values);
    let p = strs(&[Some(pattern)]);
    let f = bind_scalar(
        name,
        &[
            v.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        opts,
    )
    .unwrap();
    f.invoke(&[col(&v), lit(&p)], v.len()).unwrap()
}

#[test]
fn date_format_round_trips_common_patterns() {
    // Unverified against a Spark run (values follow the documented Java
    // pattern language and the golden formatter): UTC session.
    let out = run_pat(
        "date_format",
        &[Some("2020-01-27 20:06:11.847"), None],
        "yyyy-MM-dd HH:mm:ss",
        &utc(),
    );
    assert_eq!(text(&out), [Some("2020-01-27 20:06:11".to_owned()), None]);
    let out = run_pat(
        "date_format",
        &[Some("2020-07-04 00:00:00")],
        "MMM d, yyyy",
        &utc(),
    );
    assert_eq!(text(&out), [Some("Jul 4, 2020".to_owned())]);
    let out = run_pat(
        "date_format",
        &[Some("2020-07-04 00:00:00")],
        "EEEE",
        &utc(),
    );
    assert_eq!(text(&out), [Some("Saturday".to_owned())]);
    // Fractional seconds keep their digits; `a` is AM/PM.
    let out = run_pat(
        "date_format",
        &[Some("2020-07-04 01:02:03.123456")],
        "S SS SSS SSSSSS",
        &utc(),
    );
    assert_eq!(text(&out), [Some("1 12 123 123456".to_owned())]);
    let out = run_pat(
        "date_format",
        &[Some("2020-07-04 01:02:03"), Some("2020-07-04 13:02:03")],
        "h a",
        &utc(),
    );
    assert_eq!(
        text(&out),
        [Some("1 AM".to_owned()), Some("1 PM".to_owned())]
    );
    // Fixed-offset zones render from the session zone.
    let out = run_pat(
        "date_format",
        &[Some("2020-07-04 01:02:03")],
        "z XXX VV",
        &utc(),
    );
    assert_eq!(text(&out), [Some("UTC Z UTC".to_owned())]);
}

#[test]
fn date_format_reads_dates_timestamps_and_ntz() {
    // Unverified against a Spark run: a date formats its midnight, a
    // timestamp_ntz its wall clock.
    let d: ArrayRef = Arc::new(Date32Array::from(vec![Some(18_262)]));
    let p = strs(&[Some("yyyy-MM-dd HH:mm")]);
    let f = bind_scalar(
        "date_format",
        &[
            d.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &utc(),
    )
    .unwrap();
    assert_eq!(
        text(&f.invoke(&[col(&d), lit(&p)], 1).unwrap()),
        [Some("2020-01-01 00:00".to_owned())]
    );
    // The session zone shifts the wall clock of an instant: the epoch is
    // 05:30 at +05:30.
    let epoch = ltz(&[Some(0)]);
    let f = bind_scalar(
        "date_format",
        &[
            epoch.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &plus_thirty(),
    )
    .unwrap();
    assert_eq!(
        text(&f.invoke(&[col(&epoch), lit(&p)], 1).unwrap()),
        [Some("1970-01-01 05:30".to_owned())]
    );
    let w = ntz(&[Some(18_262 * 86_400_000_000 + 3_600_000_000)]);
    let f = bind_scalar(
        "date_format",
        &[
            w.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &utc(),
    )
    .unwrap();
    assert_eq!(
        text(&f.invoke(&[col(&w), lit(&p)], 1).unwrap()),
        [Some("2020-01-01 01:00".to_owned())]
    );
}

#[test]
fn to_timestamp_parses_with_and_without_a_pattern() {
    // Pinned against Spark 4.2 runs (the function matrix): UTC session.
    let out = run_pat(
        "to_timestamp",
        &[Some("2016-12-31 00:12:00"), None],
        "yyyy-MM-dd HH:mm:ss",
        &utc(),
    );
    assert_eq!(stamps(&out), [Some(1_483_143_120_000_000), None]);
    // One argument is the default grammar (Spark's `CAST`).
    let v = strs(&[Some("2016-12-31 00:12:00"), None]);
    let f = bind("to_timestamp", &[&v]);
    assert_eq!(
        stamps(&f.invoke(&[col(&v)], 2).unwrap()),
        [Some(1_483_143_120_000_000), None]
    );
    // `to_timestamp` over a date is its midnight.
    let d: ArrayRef = Arc::new(Date32Array::from(vec![Some(17_590)]));
    let f = bind("to_timestamp", &[&d]);
    assert_eq!(
        stamps(&f.invoke(&[col(&d)], 1).unwrap()),
        [Some(17_590 * 86_400_000_000)]
    );
}

#[test]
fn to_timestamp_reads_integers_as_seconds() {
    // Pinned against Spark 4.2 (`to_timestamp(1)` is one second past the
    // epoch, shown in the suite zone; here UTC).
    let v: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
    let f = bind("to_timestamp", &[&v]);
    assert_eq!(
        stamps(&f.invoke(&[col(&v)], 2).unwrap()),
        [Some(1_000_000), None]
    );
    // Overflow raises even without ANSI (Java's `ArithmeticException`
    // escapes `try_to_timestamp` too); NaN is NULL. Unverified against a
    // Spark run.
    let big: ArrayRef = Arc::new(Int64Array::from(vec![Some(i64::MAX)]));
    let f = bind("to_timestamp", &[&big]);
    let e = f.invoke(&[col(&big)], 1).unwrap_err();
    assert_eq!(class(&e), ErrorClass::DatetimeOverflow);
    let f = bind_with("to_timestamp", &[&big], &legacy());
    let e = f.invoke(&[col(&big)], 1).unwrap_err();
    assert_eq!(class(&e), ErrorClass::DatetimeOverflow);
    let nan: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![Some(f64::NAN)]));
    let f = bind("to_timestamp", &[&nan]);
    assert!(f.invoke(&[col(&nan)], 1).unwrap().is_null(0));
}

#[test]
fn to_date_follows_the_same_grammar_through_a_date() {
    // Pinned against Spark 4.2 runs: `to_date('2016-12-31',
    // 'yyyy-MM-dd')` is day 17136.
    let out = run_pat("to_date", &[Some("2016-12-31"), None], "yyyy-MM-dd", &utc());
    assert_eq!(dates(&out), [Some(17_166), None]);
    // Missing fields default (Spark 2.4 compatibility): `"16"` with `dd`
    // is the 16th of January 1970.
    let out = run_pat("to_date", &[Some("16")], "dd", &utc());
    assert_eq!(dates(&out), [Some(15)]);
    // One argument casts.
    let v = strs(&[Some("2016-12-31")]);
    let f = bind("to_date", &[&v]);
    assert_eq!(dates(&f.invoke(&[col(&v)], 1).unwrap()), [Some(17_166)]);
}

#[test]
fn try_forms_return_null_for_bad_input_in_both_modes() {
    for opts in [utc(), legacy()] {
        let out = run_pat("try_to_timestamp", &[Some("abc")], "yyyy-MM-dd", &opts);
        assert!(out.is_null(0));
        let out = run_pat("try_to_date", &[Some("abc")], "yyyy-MM-dd", &opts);
        assert!(out.is_null(0));
        // A NULL pattern row is NULL, not an error.
        let v = strs(&[Some("2016-12-31")]);
        let p = strs(&[None]);
        let f = bind_scalar(
            "try_to_timestamp",
            &[
                v.data_type().clone().into(),
                BindArg::Literal(Arc::clone(&p)),
            ],
            &opts,
        )
        .unwrap();
        assert!(f.invoke(&[col(&v), lit(&p)], 1).unwrap().is_null(0));
    }
}

#[test]
fn ansi_parse_failures_carry_java_text() {
    // Pinned against Spark 4.2 runs (datetime-parsing-invalid.sql.out).
    for (s, fmt, message) in [
        ("1", "yy", "Text '1' could not be parsed at index 0"),
        (
            "123",
            "yy",
            "Text '123' could not be parsed, unparsed text found at index 2",
        ),
        (
            "2020-01-27T20:06:11.847",
            "yyyy-MM-dd HH:mm:ss.SSS",
            "Text '2020-01-27T20:06:11.847' could not be parsed at index 10",
        ),
        (
            "Unparseable",
            "yyyy-MM-dd HH:mm:ss.SSS",
            "Text 'Unparseable' could not be parsed at index 0",
        ),
        (
            "366",
            "D",
            "Invalid date 'DayOfYear 366' as '1970' is not a leap year",
        ),
        (
            "2019-366",
            "yyyy-DDD",
            "Text '2019-366' could not be parsed: Invalid date 'DayOfYear 366' as '2019' is not a leap year",
        ),
        (
            "30-365",
            "dd-DDD",
            "Conflict found: Field DayOfMonth 30 differs from DayOfMonth 31 derived from 1970-12-31.",
        ),
        (
            "2020-01-365",
            "yyyy-dd-DDD",
            "Text '2020-01-365' could not be parsed: Conflict found: Field DayOfMonth 30 differs from DayOfMonth 1 derived from 2020-12-30",
        ),
        (
            "2020-11-31-366",
            "yyyy-MM-dd-DDD",
            "Text '2020-11-31-366' could not be parsed: Invalid date 'NOVEMBER 31'",
        ),
    ] {
        let v = strs(&[Some(s)]);
        let p = strs(&[Some(fmt)]);
        let f = bind_scalar(
            "to_timestamp",
            &[
                v.data_type().clone().into(),
                BindArg::Literal(Arc::clone(&p)),
            ],
            &utc(),
        )
        .unwrap();
        let e = f.invoke(&[col(&v), lit(&p)], 1).unwrap_err();
        assert_eq!(class(&e), ErrorClass::CannotParseTimestamp, "{s} {fmt}");
        assert!(e.to_string().contains(message), "{e}");
        assert!(e.to_string().contains("`try_to_timestamp`"), "{e}");
        // `to_date` reports the same failure with its own function.
        let f = bind_scalar(
            "to_date",
            &[
                v.data_type().clone().into(),
                BindArg::Literal(Arc::clone(&p)),
            ],
            &utc(),
        )
        .unwrap();
        let e = f.invoke(&[col(&v), lit(&p)], 1).unwrap_err();
        assert_eq!(class(&e), ErrorClass::CannotParseTimestamp, "{s} {fmt}");
        assert!(e.to_string().contains("`try_to_date`"), "{e}");
        // With ANSI off both are NULL.
        for name in ["to_timestamp", "to_date"] {
            let f = bind_scalar(
                name,
                &[
                    v.data_type().clone().into(),
                    BindArg::Literal(Arc::clone(&p)),
                ],
                &legacy(),
            )
            .unwrap();
            assert!(
                f.invoke(&[col(&v), lit(&p)], 1).unwrap().is_null(0),
                "{name} {s}"
            );
        }
    }
    // A year past the microsecond range overflows, even without ANSI.
    let v = strs(&[Some("294248")]);
    let p = strs(&[Some("y")]);
    for opts in [utc(), legacy()] {
        let f = bind_scalar(
            "to_timestamp",
            &[
                v.data_type().clone().into(),
                BindArg::Literal(Arc::clone(&p)),
            ],
            &opts,
        )
        .unwrap();
        let e = f.invoke(&[col(&v), lit(&p)], 1).unwrap_err();
        assert_eq!(class(&e), ErrorClass::DatetimeOverflow);
    }
}

#[test]
fn column_patterns_compile_per_value_behind_a_bound() {
    // Unverified against a Spark run: a column pattern compiles once per
    // distinct value. Nine distinct patterns exceed the eight-entry cache;
    // eviction changes nothing observable.
    let values = [
        "2020-01-05",
        "05/01/2020",
        "01-05-2020",
        "20200105",
        "05012020",
        "01/05/2020",
        "2020/01/05",
        "05-01-2020",
        "01052020",
    ];
    let patterns = [
        "yyyy-MM-dd",
        "dd/MM/yyyy",
        "MM-dd-yyyy",
        "yyyyMMdd",
        "ddMMyyyy",
        "MM/dd/yyyy",
        "yyyy/MM/dd",
        "dd-MM-yyyy",
        "MMddyyyy",
    ];
    let v: Vec<Option<&str>> = values.iter().map(|s| Some(*s)).collect();
    let p: Vec<Option<&str>> = patterns.iter().map(|s| Some(*s)).collect();
    let v = strs(&v);
    let p = strs(&p);
    let g = bind_scalar(
        "to_date",
        &[v.data_type().clone().into(), p.data_type().clone().into()],
        &utc(),
    )
    .unwrap();
    assert_eq!(
        dates(&g.invoke(&[col(&v), col(&p)], 9).unwrap()),
        [Some(18_266); 9]
    );
    // A bad pattern in a column row raises even without ANSI (Spark
    // validates the pattern outside the parse `try`) ...
    let bad = strs(&[Some("2020-01-05"), Some("2020-01-05")]);
    let pats = strs(&[Some("yyyy-MM-dd"), Some("QQQQQ")]);
    let g = bind_scalar(
        "to_date",
        &[
            bad.data_type().clone().into(),
            pats.data_type().clone().into(),
        ],
        &legacy(),
    )
    .unwrap();
    let e = g.invoke(&[col(&bad), col(&pats)], 2).unwrap_err();
    // `Q` cannot parse (Spark rejects it for parsing, unlike formatting,
    // where five `Q`s are a LENGTH error).
    assert_eq!(
        class(&e),
        ErrorClass::InvalidDatetimePatternIllegalCharacter
    );
    // ... while a bad value in another row is NULL under `try_to_date`.
    let vals = strs(&[Some("nope"), Some("2020-01-05")]);
    let g = bind_scalar(
        "try_to_date",
        &[
            vals.data_type().clone().into(),
            pats.data_type().clone().into(),
        ],
        &legacy(),
    )
    .unwrap();
    // The second row's pattern is still bad, so this raises too: patterns
    // always raise, values may be NULL.
    let e = g.invoke(&[col(&vals), col(&pats)], 2).unwrap_err();
    assert_eq!(
        class(&e),
        ErrorClass::InvalidDatetimePatternIllegalCharacter
    );
}

#[test]
fn literal_patterns_fail_when_bound() {
    // Pinned against Spark 4.2 runs (datetime-formatting-invalid.sql.out
    // exercises the same classes through `date_format`).
    for (pattern, want) in [
        (
            "GGGGG",
            ErrorClass::InconsistentBehaviorCrossVersionDatetimePatternRecognition,
        ),
        (
            "yyyyyyy",
            ErrorClass::InconsistentBehaviorCrossVersionDatetimePatternRecognition,
        ),
        ("QQQQQ", ErrorClass::InvalidDatetimePatternLength),
        (
            "SSSSSSSSSS",
            ErrorClass::InconsistentBehaviorCrossVersionDatetimePatternRecognition,
        ),
        (
            "aa",
            ErrorClass::InconsistentBehaviorCrossVersionDatetimePatternRecognition,
        ),
        (
            "Y",
            ErrorClass::InconsistentBehaviorCrossVersionDatetimePatternRecognition,
        ),
        (
            "e",
            ErrorClass::InconsistentBehaviorCrossVersionDatetimeWeekBasedPattern,
        ),
        ("A", ErrorClass::InvalidDatetimePatternIllegalCharacter),
    ] {
        let v = strs(&[Some("2018-11-17 13:33:33.333")]);
        let p = strs(&[Some(pattern)]);
        let e = bind_scalar(
            "date_format",
            &[
                v.data_type().clone().into(),
                BindArg::Literal(Arc::clone(&p)),
            ],
            &utc(),
        )
        .unwrap_err();
        assert_eq!(class(&e), want, "{pattern}");
    }
}

#[test]
fn unix_timestamp_counts_seconds_with_and_without_a_pattern() {
    // Pinned against Spark 4.2 (`to_unix_timestamp('2016-04-08',
    // 'yyyy-MM-dd')` is 1460098800 in the suite zone; here UTC it is
    // 1460073600): UTC session.
    let out = run_pat(
        "to_unix_timestamp",
        &[Some("2016-04-08"), None],
        "yyyy-MM-dd",
        &utc(),
    );
    assert_eq!(longs(&out), [Some(1_460_073_600), None]);
    // Without a pattern the default grammar applies.
    let v = strs(&[Some("2016-04-08 00:00:00")]);
    let f = bind("to_unix_timestamp", &[&v]);
    assert_eq!(
        longs(&f.invoke(&[col(&v)], 1).unwrap()),
        [Some(1_460_073_600)]
    );
    // Dates and timestamps ignore the pattern.
    let d: ArrayRef = Arc::new(Date32Array::from(vec![Some(17_590)]));
    let p = strs(&[Some("HH")]);
    let f = bind_scalar(
        "unix_timestamp",
        &[
            d.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &utc(),
    )
    .unwrap();
    assert_eq!(
        longs(&f.invoke(&[col(&d), lit(&p)], 1).unwrap()),
        [Some(17_590 * 86_400)]
    );
    // No arguments is now.
    let f = bind_scalar(
        "unix_timestamp",
        &[],
        &BindOptions::new().with_query_start(1_500_000),
    )
    .unwrap();
    assert_eq!(
        longs(&f.invoke(&[], 3).unwrap()),
        [Some(1), Some(1), Some(1)]
    );
    // Garbage is CANNOT_PARSE_TIMESTAMP with ANSI on, NULL without.
    let v = strs(&[Some("abc")]);
    let p = strs(&[Some("yyyy-MM-dd")]);
    let f = bind_scalar(
        "unix_timestamp",
        &[
            v.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &utc(),
    )
    .unwrap();
    let e = f.invoke(&[col(&v), lit(&p)], 1).unwrap_err();
    assert_eq!(class(&e), ErrorClass::CannotParseTimestamp);
    assert!(e.to_string().contains("`try_to_timestamp`"), "{e}");
}

#[test]
fn from_unixtime_renders_seconds_in_the_session_zone() {
    // Unverified against a Spark run: 0 renders the epoch, in the session
    // zone.
    let v: ArrayRef = Arc::new(Int64Array::from(vec![Some(0), None]));
    let f = bind("from_unixtime", &[&v]);
    let out = text(&f.invoke(&[col(&v)], 2).unwrap());
    assert_eq!(out, [Some("1970-01-01 00:00:00".to_owned()), None]);
    let f = strs(&[Some("yyyy")]);
    let g = bind_scalar(
        "from_unixtime",
        &[
            v.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&f)),
        ],
        &utc(),
    )
    .unwrap();
    assert_eq!(
        text(&g.invoke(&[col(&v), lit(&f)], 2).unwrap()),
        [Some("1970".to_owned()), None]
    );
}

#[test]
fn current_timezone_names_the_session_zone() {
    // Unverified against a Spark run (the goldens run in a region zone,
    // which cannot cross the plan boundary yet).
    let f = bind_scalar("current_timezone", &[], &utc()).unwrap();
    assert_eq!(
        text(&f.invoke(&[], 2).unwrap()),
        [Some("UTC".to_owned()), Some("UTC".to_owned())]
    );
    let f = bind_scalar("current_timezone", &[], &plus_thirty()).unwrap();
    assert_eq!(
        text(&f.invoke(&[], 1).unwrap()),
        [Some("+05:30".to_owned())]
    );
}

#[test]
fn ltz_and_ntz_parse_with_their_zone_rules() {
    // Unverified against a Spark run: `to_timestamp_ltz` keeps the
    // instant, `to_timestamp_ntz` the wall clock.
    let v = strs(&[Some("2019-10-06 10:11:12.123456PST")]);
    let p = strs(&[Some("yyyy-MM-dd HH:mm:ss.SSSSSS[zzz]")]);
    for name in ["to_timestamp_ltz", "to_timestamp"] {
        let f = bind_scalar(
            name,
            &[
                v.data_type().clone().into(),
                BindArg::Literal(Arc::clone(&p)),
            ],
            &utc(),
        )
        .unwrap();
        // 10:11:12.123456 at -07:00 (PDT in October) is 17:11:12 UTC.
        assert_eq!(
            stamps(&f.invoke(&[col(&v), lit(&p)], 1).unwrap()),
            [Some(1_570_381_872_123_456)],
            "{name}"
        );
    }
    let f = bind_scalar(
        "to_timestamp_ntz",
        &[
            v.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &utc(),
    )
    .unwrap();
    assert_eq!(
        stamps(&f.invoke(&[col(&v), lit(&p)], 1).unwrap()),
        [Some(1_570_356_672_123_456)]
    );
    // A zone in the text is ignored for `TIMESTAMP_NTZ`.
    let w = strs(&[Some("1970-01-01 01:00 +01:00")]);
    let q = strs(&[Some("yyyy-MM-dd HH:mm XXX")]);
    let f = bind_scalar(
        "to_timestamp_ntz",
        &[
            w.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&q)),
        ],
        &utc(),
    )
    .unwrap();
    assert_eq!(
        stamps(&f.invoke(&[col(&w), lit(&q)], 1).unwrap()),
        [Some(3_600_000_000)]
    );
}

#[test]
fn region_zones_convert_across_dst() {
    // Unverified against a Spark run: conversions in America/Los_Angeles
    // follow `java.time` (gap forward, overlap earlier offset).
    let zone = strs(&[Some("America/Los_Angeles")]);
    let utc_midnight: ArrayRef = Arc::new(
        TimestampMicrosecondArray::from(vec![Some(1_577_865_600_000_000)]).with_timezone("UTC"),
    );
    let f = bind_scalar(
        "from_utc_timestamp",
        &[
            utc_midnight.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&zone)),
        ],
        &utc(),
    )
    .unwrap();
    // 2020-01-01 08:00 UTC has wall midnight in Los Angeles (UTC-08:00),
    // so the result is midnight UTC.
    assert_eq!(
        stamps(&f.invoke(&[col(&utc_midnight), lit(&zone)], 1).unwrap()),
        [Some(1_577_836_800_000_000)]
    );
    // An unknown zone is INVALID_TIMEZONE, at bind and per row.
    let bad = strs(&[Some("America/Nope")]);
    let e = bind_scalar(
        "from_utc_timestamp",
        &[
            utc_midnight.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&bad)),
        ],
        &utc(),
    )
    .unwrap_err();
    assert_eq!(class(&e), ErrorClass::InvalidTimezone);
    let cols = strs(&[Some("America/Nope")]);
    let f = bind("from_utc_timestamp", &[&utc_midnight, &cols]);
    let e = f.invoke(&[col(&utc_midnight), col(&cols)], 1).unwrap_err();
    assert_eq!(class(&e), ErrorClass::InvalidTimezone);
}

#[test]
fn unknown_zones_in_strings_fail_like_bad_input() {
    // Unverified against a Spark run (Java's `Unknown time-zone ID`):
    // CANNOT_PARSE_TIMESTAMP with ANSI on, NULL without.
    let v = strs(&[Some("2020 Foo/Bar")]);
    let p = strs(&[Some("yyyy VV")]);
    let f = bind_scalar(
        "to_timestamp",
        &[
            v.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &utc(),
    )
    .unwrap();
    let e = f.invoke(&[col(&v), lit(&p)], 1).unwrap_err();
    assert_eq!(class(&e), ErrorClass::CannotParseTimestamp);
    assert!(
        e.to_string().contains("Unknown time-zone ID: Foo/Bar"),
        "{e}"
    );
    let f = bind_scalar(
        "try_to_timestamp",
        &[
            v.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &utc(),
    )
    .unwrap();
    assert!(f.invoke(&[col(&v), lit(&p)], 1).unwrap().is_null(0));
}

#[test]
fn numerics_with_a_pattern_are_unresolved() {
    // Spark analyses `to_timestamp(1, fmt)` but its `gettimestamp`
    // replacement never resolves: an internal error for every evaluated
    // row (the message is approximate; no golden or matrix case compares
    // it). Unverified against a Spark run.
    let v: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
    let p = strs(&[Some("yyyy-MM-dd")]);
    let f = bind_scalar(
        "to_timestamp",
        &[
            v.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &utc(),
    )
    .unwrap();
    let e = f.invoke(&[col(&v), lit(&p)], 2).unwrap_err();
    assert_eq!(class(&e), ErrorClass::InternalError);
    assert!(e.to_string().contains("gettimestamp"), "{e}");
    // All-NULL rows never evaluate: NULL, not an error.
    let n: ArrayRef = Arc::new(Int64Array::from(vec![None, None]));
    let f = bind_scalar(
        "to_timestamp",
        &[
            n.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&p)),
        ],
        &utc(),
    )
    .unwrap();
    assert_eq!(f.invoke(&[col(&n), lit(&p)], 2).unwrap().null_count(), 2);
}

#[test]
fn wrong_types_and_counts_do_not_bind() {
    // The analyzer rejects these first; the binding refuses them as
    // unsupported.
    let v = strs(&[Some("x")]);
    for (name, args) in [
        ("date_format", vec![v.data_type().clone().into()]),
        (
            "to_date",
            vec![
                Int64Array::from(vec![Some(1)]).data_type().clone().into(),
                v.data_type().clone().into(),
                v.data_type().clone().into(),
            ],
        ),
        ("from_unixtime", vec![v.data_type().clone().into()]),
        (
            "unix_timestamp",
            vec![Int64Array::from(vec![Some(1)]).data_type().clone().into()],
        ),
    ] {
        let args: Vec<BindArg> = args;
        assert!(
            bind_scalar(name, &args, &utc()).is_err(),
            "{name} bound {args:?}"
        );
    }
}
