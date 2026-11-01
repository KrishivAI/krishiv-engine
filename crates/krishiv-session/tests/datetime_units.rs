// SPDX-License-Identifier: Apache-2.0
//! `timestampadd`, `timestampdiff` (and the unit forms of `dateadd` and
//! `datediff`) and the `make_timestamp` family end to end, under the
//! `America/Los_Angeles` session zone. Expected values come from a Spark
//! 4.2.0 run in that zone.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = America/Los_Angeles")).unwrap();
    s
}

/// The single column of `SELECT CAST(<expr> AS STRING) <from>` as text.
fn column(s: &Session, expr: &str, from: &str) -> Vec<Option<String>> {
    let sql = format!("SELECT CAST({expr} AS STRING) {from}");
    let df = block_on(s.sql(&sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        let c = b.column(0);
        for i in 0..b.num_rows() {
            out.push((!c.is_null(i)).then(|| match c.data_type() {
                arrow_schema::DataType::Utf8View => c.as_string_view().value(i).to_owned(),
                _ => c.as_string::<i32>().value(i).to_owned(),
            }));
        }
    }
    out
}

fn one(s: &Session, expr: &str) -> Option<String> {
    column(s, expr, "").remove(0)
}

fn error_of(s: &Session, expr: &str) -> krishiv_common::error::KrishivError {
    let sql = format!("SELECT {expr}");
    block_on(async { s.sql(&sql).await?.collect().await })
        .err()
        .unwrap_or_else(|| panic!("{sql} ran"))
}

#[test]
fn timestampadd_moves_calendar_units_on_the_wall_clock() {
    let s = session();
    for (expr, want) in [
        (
            "timestampadd(MONTH, -1, TIMESTAMP '2022-02-14 01:02:03')",
            "2022-01-14 01:02:03",
        ),
        // Across the 2022-03-13 DST change a day keeps the wall clock, 24
        // hours do not.
        (
            "timestampadd(DAY, 1, TIMESTAMP '2022-03-12 12:00:00')",
            "2022-03-13 12:00:00",
        ),
        (
            "timestampadd(HOUR, 24, TIMESTAMP '2022-03-12 12:00:00')",
            "2022-03-13 13:00:00",
        ),
        (
            "timestampadd(WEEK, 1, TIMESTAMP_NTZ '2022-02-14 01:02:03')",
            "2022-02-21 01:02:03",
        ),
        (
            "timestampadd(YEAR, 1, DATE '2022-02-15')",
            "2023-02-15 00:00:00",
        ),
        (
            "timestampadd(QUARTER, 1, TIMESTAMP '2022-11-30 10:00:00')",
            "2023-02-28 10:00:00",
        ),
        (
            "timestampadd(MILLISECOND, 1500, TIMESTAMP '2022-01-01 00:00:00')",
            "2022-01-01 00:00:01.5",
        ),
        // `dateadd` and `date_add` with a unit are `timestampadd`.
        (
            "dateadd(HOUR, 24, TIMESTAMP '2022-03-12 12:00:00')",
            "2022-03-13 13:00:00",
        ),
        ("date_add(DAY, 1, DATE '2022-02-14')", "2022-02-15 00:00:00"),
    ] {
        assert_eq!(one(&s, expr).as_deref(), Some(want), "{expr}");
    }
    assert_eq!(
        one(
            &s,
            "timestampadd(DAY, NULL, TIMESTAMP '2022-01-01 00:00:00')"
        ),
        None
    );
    // Rows with NULLs and several values in one batch.
    assert_eq!(
        column(
            &s,
            "timestampadd(DAY, n, TIMESTAMP '2022-01-01 00:00:00')",
            "FROM VALUES (1), (NULL), (-1) AS t(n)"
        ),
        vec![
            Some("2022-01-02 00:00:00".to_owned()),
            None,
            Some("2021-12-31 00:00:00".to_owned())
        ]
    );
}

#[test]
fn timestampadd_overflow_names_the_operation() {
    let s = session();
    for expr in [
        "timestampadd(YEAR, 9223372036854775807, TIMESTAMP '2022-02-14 01:02:03')",
        "timestampadd(MICROSECOND, 9223372036854775807, TIMESTAMP '2022-02-14 01:02:03')",
        "timestampadd(YEAR, 100000000, TIMESTAMP '2022-02-14 01:02:03')",
    ] {
        let err = error_of(&s, expr);
        assert_eq!(err.error_class(), ErrorClass::DatetimeOverflow, "{expr}");
    }
    let err = error_of(
        &s,
        "timestampadd(YEAR, 100000000, TIMESTAMP '2022-02-14 01:02:03')",
    );
    assert!(
        err.to_string().contains(
            "Datetime operation overflow: add 100000000L YEAR to TIMESTAMP '2022-02-14 01:02:03'"
        ),
        "{err}"
    );
}

#[test]
fn timediff_is_timestampdiff() {
    let s = session();
    for (expr, want) in [
        (
            "timediff(HOUR, TIMESTAMP '2022-02-14 01:02:03', TIMESTAMP '2022-02-14 12:00:03')",
            "10",
        ),
        (
            "timediff(QUARTER, TIMESTAMP '2023-08-10 01:02:03', TIMESTAMP '2022-01-14 01:02:03')",
            "-6",
        ),
        ("timediff(DAY, DATE '2022-02-15', DATE '2023-02-15')", "365"),
    ] {
        assert_eq!(one(&s, expr).as_deref(), Some(want), "{expr}");
    }
}

#[test]
fn timestampdiff_counts_whole_units_on_the_wall_clock() {
    let s = session();
    for (expr, want) in [
        (
            "timestampdiff(DAY, TIMESTAMP '2022-03-12 12:00:00', TIMESTAMP '2022-03-13 12:00:00')",
            "1",
        ),
        (
            "timestampdiff(HOUR, TIMESTAMP '2022-03-12 12:00:00', TIMESTAMP '2022-03-13 12:00:00')",
            "24",
        ),
        (
            "timestampdiff(MONTH, TIMESTAMP '2022-01-31 10:00:00', TIMESTAMP '2022-02-28 09:00:00')",
            "0",
        ),
        (
            "timestampdiff(MONTH, TIMESTAMP '2022-02-28 09:00:00', TIMESTAMP '2022-01-31 10:00:00')",
            "0",
        ),
        (
            "timestampdiff(QUARTER, DATE '2022-01-31', DATE '2023-02-28')",
            "4",
        ),
        (
            "timestampdiff(WEEK, TIMESTAMP '2022-02-14', TIMESTAMP '2022-02-27 23:59:59')",
            "1",
        ),
        (
            "timestampdiff(YEAR, TIMESTAMP '2022-02-14', TIMESTAMP '2020-02-14 00:00:01')",
            "-1",
        ),
        ("date_diff(DAY, DATE '2022-02-14', DATE '2022-03-14')", "28"),
        (
            "datediff(MICROSECOND, TIMESTAMP '1900-01-01', TIMESTAMP '2100-01-01')",
            "6311433600000000",
        ),
    ] {
        assert_eq!(one(&s, expr).as_deref(), Some(want), "{expr}");
    }
    assert_eq!(
        one(&s, "timestampdiff(WEEK, TIMESTAMP '2022-02-14', NULL)"),
        None
    );
}

#[test]
fn make_timestamp_reads_its_fields_in_the_zone() {
    let s = session();
    for (expr, want) in [
        (
            "make_timestamp(2021, 7, 11, 6, 30, 45.678)",
            "2021-07-11 06:30:45.678",
        ),
        (
            "make_timestamp(2021, 7, 11, 6, 30, 45.678, 'CET')",
            "2021-07-10 21:30:45.678",
        ),
        // 60 seconds rolls over to the next minute.
        ("make_timestamp(1, 1, 1, 1, 1, 60)", "0001-01-01 01:02:00"),
        // A local time in the spring-forward gap moves forward.
        (
            "make_timestamp(2021, 3, 14, 2, 30, 0)",
            "2021-03-14 03:30:00",
        ),
        (
            "make_timestamp(2021, 11, 7, 1, 30, 0)",
            "2021-11-07 01:30:00",
        ),
        (
            "make_timestamp_ntz(2021, 7, 11, 6, 30, 45.678)",
            "2021-07-11 06:30:45.678",
        ),
        (
            "make_timestamp_ltz(2021, 7, 11, 6, 30, 45.678, '+01:00')",
            "2021-07-10 22:30:45.678",
        ),
    ] {
        assert_eq!(one(&s, expr).as_deref(), Some(want), "{expr}");
    }
    assert_eq!(
        column(
            &s,
            "make_timestamp(a, 7, 11, 6, 30, 45)",
            "FROM VALUES (2021), (NULL), (2022) AS t(a)"
        ),
        vec![
            Some("2021-07-11 06:30:45".to_owned()),
            None,
            Some("2022-07-11 06:30:45".to_owned())
        ]
    );
    assert_eq!(
        one(&s, "make_timestamp(2021, 7, 11, 6, 30, 45, NULL)"),
        None
    );
    assert_eq!(one(&s, "make_timestamp(1, 1, 1, 1, 1, NULL)"), None);
}

#[test]
fn make_timestamp_errors_are_sparks_and_try_gives_null() {
    let s = session();
    for (expr, class, text) in [
        (
            "make_timestamp(2021, 13, 11, 6, 30, 45.678)",
            ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion,
            "Invalid value for MonthOfYear (valid values 1 - 12): 13",
        ),
        (
            "make_timestamp(2021, 2, 30, 6, 30, 45.678)",
            ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion,
            "Invalid date 'FEBRUARY 30'",
        ),
        (
            "make_timestamp(2021, 7, 11, 25, 30, 45)",
            ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion,
            "Invalid value for HourOfDay (valid values 0 - 23): 25",
        ),
        (
            "make_timestamp(2021, 7, 11, 6, 61, 45)",
            ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion,
            "Invalid value for MinuteOfHour (valid values 0 - 59): 61",
        ),
        (
            "make_timestamp(1, 1, 1, 1, 1, 61)",
            ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion,
            "Invalid value for SecondOfMinute (valid values 0 - 59): 61",
        ),
        (
            "make_timestamp(1, 1, 1, 1, 1, 99.999999)",
            ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion,
            "Invalid value for SecondOfMinute (valid values 0 - 59): 99",
        ),
        (
            "make_timestamp(2021, 7, 11, 6, 30, 60.007)",
            ErrorClass::InvalidFractionOfSecond,
            "but the provided value is 60.007",
        ),
        (
            "make_timestamp(2021, 7, 11, 6, 30, 45, 'Mars/Base')",
            ErrorClass::InvalidTimezone,
            "Mars/Base",
        ),
    ] {
        let err = error_of(&s, expr);
        assert_eq!(err.error_class(), class, "{expr}: {err}");
        assert!(err.to_string().contains(text), "{expr}: {err}");
    }
    // `try_` forms and ANSI off return NULL for a bad field.
    assert_eq!(one(&s, "try_make_timestamp(2021, 13, 11, 6, 30, 45)"), None);
    assert_eq!(
        one(&s, "try_make_timestamp(2021, 7, 11, 6, 30, 60.007)"),
        None
    );
    assert_eq!(
        one(&s, "try_make_timestamp(2021, 7, 11, 6, 30, 60)").as_deref(),
        Some("2021-07-11 06:31:00")
    );
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    assert_eq!(one(&s, "make_timestamp(2021, 13, 11, 6, 30, 45)"), None);
}

fn session_in(zone: &str) -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql(&format!("SET spark.sql.session.timeZone = {zone}"))).unwrap();
    s
}

/// The overlap takes the earlier offset and a gap moves forward, with the
/// 30-minute step of `Australia/Lord_Howe` and the half-hour offset of
/// `Asia/Kolkata` (microseconds from a Spark 4.2.0 run).
#[test]
fn gaps_and_overlaps_resolve_as_java_time_does() {
    let la = session();
    assert_eq!(
        one(
            &la,
            "CAST(CAST(make_timestamp(2021, 11, 7, 1, 30, 0) AS DOUBLE) * 1000000 AS BIGINT)"
        )
        .as_deref(),
        Some("1636273800000000")
    );
    assert_eq!(
        one(
            &la,
            "CAST(CAST(timestampadd(DAY, 1, TIMESTAMP '2021-11-06 01:30:00') AS DOUBLE) * 1000000 AS BIGINT)"
        )
        .as_deref(),
        Some("1636273800000000")
    );
    assert_eq!(
        one(
            &la,
            "CAST(CAST(timestampadd(HOUR, 1, TIMESTAMP '2021-11-07 01:30:00') AS DOUBLE) * 1000000 AS BIGINT)"
        )
        .as_deref(),
        Some("1636277400000000")
    );
    let lord_howe = session_in("Australia/Lord_Howe");
    for (expr, want) in [
        (
            "make_timestamp(2021, 10, 3, 2, 15, 0)",
            "2021-10-03 02:45:00",
        ),
        (
            "CAST(CAST(make_timestamp(2021, 10, 3, 2, 15, 0) AS DOUBLE) * 1000000 AS BIGINT)",
            "1633189500000000",
        ),
        (
            "CAST(CAST(make_timestamp(2021, 4, 4, 1, 45, 0) AS DOUBLE) * 1000000 AS BIGINT)",
            "1617461100000000",
        ),
        (
            "timestampadd(DAY, 1, make_timestamp(2021, 10, 2, 2, 15, 0))",
            "2021-10-03 02:45:00",
        ),
    ] {
        assert_eq!(one(&lord_howe, expr).as_deref(), Some(want), "{expr}");
    }
    let kolkata = session_in("Asia/Kolkata");
    assert_eq!(
        one(
            &kolkata,
            "CAST(CAST(make_timestamp(2021, 7, 11, 6, 30, 45) AS DOUBLE) * 1000000 AS BIGINT)"
        )
        .as_deref(),
        Some("1625965245000000")
    );
    assert_eq!(
        one(
            &kolkata,
            "timestampdiff(HOUR, TIMESTAMP '2021-07-11 00:00:00', TIMESTAMP '2021-07-11 23:59:59')"
        )
        .as_deref(),
        Some("23")
    );
}

/// A zone-less operand of `timestampdiff` is read in the session zone first,
/// so a time inside a spring-forward gap moves forward.
#[test]
fn timestampdiff_reads_zone_less_values_in_the_session_zone() {
    let s = session();
    assert_eq!(
        one(
            &s,
            "timestampdiff(HOUR, TIMESTAMP_NTZ '2022-03-13 02:30:00', TIMESTAMP_NTZ '2022-03-13 04:30:00')"
        )
        .as_deref(),
        Some("1")
    );
    assert_eq!(
        one(
            &s,
            "timestampdiff(HOUR, TIMESTAMP '2022-03-13 01:30:00', TIMESTAMP_NTZ '2022-03-13 02:30:00')"
        )
        .as_deref(),
        Some("2")
    );
}

#[test]
fn make_timestamp_seconds_follow_sparks_int_truncation_and_check_order() {
    let s = session();
    // The unscaled seconds are truncated to an int before they are split.
    let err = error_of(&s, "make_timestamp(2021, 1, 1, 1, 1, 3000)");
    assert!(
        err.to_string()
            .contains("SecondOfMinute (valid values 0 - 59): -1295"),
        "{err}"
    );
    assert_eq!(
        one(&s, "make_timestamp(2021, 1, 1, 1, 1, 4324.967296)").as_deref(),
        Some("2021-01-01 01:01:30")
    );
    // A fraction of 60 seconds is refused before the month is checked.
    let err = error_of(&s, "make_timestamp(2021, 13, 1, 1, 1, 60.5)");
    assert_eq!(
        err.error_class(),
        ErrorClass::InvalidFractionOfSecond,
        "{err}"
    );
}

#[test]
fn an_unknown_zone_is_invalid_timezone_even_for_the_try_forms() {
    let s = session();
    for expr in [
        "make_timestamp(2021, 7, 11, 6, 30, 45, 'Mars/Base')",
        "try_make_timestamp(2021, 7, 11, 6, 30, 45, 'Mars/Base')",
    ] {
        assert_eq!(
            error_of(&s, expr).error_class(),
            ErrorClass::InvalidTimezone,
            "{expr}"
        );
    }
    // A session zone that names no region is refused when it is set (see
    // `session_time_zone.rs`), so no call ever reads one.
    let e = block_on(s.sql("SET spark.sql.session.timeZone = Europe/Pariss")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidConfValueTimeZone);
}

#[test]
fn make_timestamp_ntz_without_ansi_gives_null_for_a_bad_field() {
    let s = session();
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    assert_eq!(one(&s, "make_timestamp_ntz(2021, 13, 11, 6, 30, 45)"), None);
    assert_eq!(
        one(&s, "try_make_timestamp_ntz(2021, 2, 30, 6, 30, 45)"),
        None
    );
    assert_eq!(
        one(&s, "try_make_timestamp_ntz(2021, 2, 28, 6, 30, 45)").as_deref(),
        Some("2021-02-28 06:30:45")
    );
}

/// A join condition holds no nondeterministic expression
/// (`INVALID_NON_DETERMINISTIC_EXPRESSIONS`), even through a SQL function.
#[test]
fn a_nondeterministic_join_condition_is_refused() {
    let s = session();
    block_on(s.sql("CREATE FUNCTION nd_pick() RETURNS INT RETURN FLOOR(RAND() * 6) + 1")).unwrap();
    for sql in [
        "SELECT * FROM VALUES(1) AS a(x) JOIN VALUES(1) AS b(y) ON rand() > 0.5",
        "SELECT * FROM VALUES(1) AS a(x) JOIN VALUES(1) AS b(y) ON nd_pick() = 1",
    ] {
        let r = block_on(async { s.sql(sql).await?.collect().await });
        let e = r.expect_err(sql);
        assert!(
            format!("{e:?}").contains("InvalidNonDeterministicExpressions"),
            "{sql}: {e:?}"
        );
    }
}

/// With `spark.sql.legacy.identifierClause` on, `IDENTIFIER('c1')` is a
/// column reference, qualified and backquoted forms included.
#[test]
fn legacy_identifier_function_names_a_column() {
    let s = session();
    block_on(s.sql("SET spark.sql.legacy.identifierClause = true")).unwrap();
    for (expr, want) in [
        ("IDENTIFIER('c1')", "1"),
        ("IDENTIFIER('t.c1')", "1"),
        ("IDENTIFIER('`c1`')", "1"),
        ("IDENTIFIER('c1') + 1", "2"),
    ] {
        let got = column(&s, expr, "FROM VALUES(1) AS T(c1)");
        assert_eq!(got, vec![Some(want.to_owned())], "{expr}");
    }
}

/// A modifier on the legacy `IDENTIFIER('c1')` call is not dropped: the call
/// stays a function call, which does not resolve.
#[test]
fn legacy_identifier_call_with_a_modifier_is_not_a_column() {
    let s = session();
    block_on(s.sql("SET spark.sql.legacy.identifierClause = true")).unwrap();
    let r = block_on(async {
        s.sql("SELECT IDENTIFIER('c1') FILTER (WHERE missing) FROM VALUES(1) AS T(c1)")
            .await?
            .collect()
            .await
    });
    assert!(r.is_err(), "{r:?}");
}
