// SPDX-License-Identifier: Apache-2.0
//! Row 6 (T025 rest): the column names Spark gives `RuntimeReplaceable` datetime calls, parses that
//! return `TIMESTAMP_NTZ`, the order of a lazily built formatter's errors, `time_trunc` on a string and
//! the arity of `make_timestamp_ntz`. Expected values: Spark 4.2.0 goldens (`try_arithmetic`,
//! `extract`, `timestamp-ltz`, `timestamp-ntz`, `time`, `timestampNTZ/timestamp`,
//! `typeCoercion/native/*`).
#![expect(clippy::unwrap_used, reason = "test code")]

use futures::executor::block_on;
use krishiv_session::Session;

fn session(conf: &[&str]) -> Session {
    let s = Session::new().unwrap();
    for c in conf {
        block_on(async { s.sql(&format!("set {c}")).await?.collect().await }).unwrap();
    }
    s
}

/// The column names and Arrow type names of `sql`'s result.
fn schema(s: &Session, sql: &str) -> Result<Vec<(String, String)>, String> {
    let batches =
        block_on(async { s.sql(sql).await?.collect().await }).map_err(|e| e.to_string())?;
    Ok(batches[0]
        .schema()
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().to_string()))
        .collect())
}

fn names(s: &Session, sql: &str) -> Vec<String> {
    schema(s, sql)
        .unwrap()
        .into_iter()
        .map(|(n, _)| n)
        .collect()
}

fn class(s: &Session, sql: &str) -> String {
    block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_err()
        .error_class()
        .name()
        .to_owned()
}

#[test]
fn replaceable_calls_show_the_name_spark_gives_them() {
    let s = session(&["spark.sql.timeType.enabled=true"]);
    assert_eq!(
        names(
            &s,
            "SELECT try_add(DATE '2021-01-01', 1), try_add(1, DATE '2021-01-01')"
        ),
        [
            "try_add(DATE '2021-01-01', 1)",
            "try_add(1, DATE '2021-01-01')"
        ]
    );
    assert_eq!(
        names(&s, "select datepart('year', DATE '2020-01-01')"),
        ["datepart(year FROM DATE '2020-01-01')"]
    );
    // The date and time forms of the ltz constructor show as `make_timestamp`; the numeric ones as written.
    assert_eq!(
        names(
            &s,
            "SELECT make_timestamp_ltz(make_date(2021, 7, 11), make_time(6, 30, 45.678))"
        ),
        ["make_timestamp(make_date(2021, 7, 11), make_time(6, 30, 45.678))"]
    );
    assert_eq!(
        names(&s, "SELECT make_timestamp_ltz(2021, 7, 11, 6, 30, 45.678)"),
        ["make_timestamp_ltz(2021, 7, 11, 6, 30, 45.678)"]
    );
    assert_eq!(
        names(&s, "SELECT try_to_time('00:12:00')"),
        ["try_to_time(to_time(00:12:00))"]
    );
    assert_eq!(
        names(
            &s,
            "SELECT convert_timezone('Europe/Brussels', TIMESTAMP_NTZ '2022-03-23 00:00:00')"
        ),
        [
            "convert_timezone(current_timezone(), Europe/Brussels, TIMESTAMP_NTZ '2022-03-23 00:00:00')"
        ]
    );
}

#[test]
fn make_timestamp_ntz_takes_at_most_six_arguments_and_the_ltz_form_seven() {
    let s = session(&[]);
    assert_eq!(
        class(
            &s,
            "SELECT make_timestamp_ntz(2021, 07, 11, 6, 30, 45.678, 'CET')"
        ),
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"
    );
    schema(
        &s,
        "SELECT make_timestamp_ltz(2021, 07, 11, 6, 30, 45.678, 'CET')",
    )
    .unwrap();
}

#[test]
fn parses_to_timestamp_ntz_report_patterns_without_the_legacy_fallback() {
    let s = session(&[
        "spark.sql.timestampType=TIMESTAMP_NTZ",
        "spark.sql.ansi.enabled=false",
    ]);
    for pattern in ["yyyy-MM-dd GGGGG", "dd MM yyyy EEEEE", "dd MM yyyy EEEEEE"] {
        assert_eq!(
            class(&s, &format!("select to_timestamp('x', '{pattern}')")),
            "INVALID_DATETIME_PATTERN.WITH_SUGGESTION",
            "{pattern}"
        );
    }
    // A number is read as a string, and an untyped NULL stays NULL, both as TIMESTAMP_NTZ.
    for sql in ["select to_timestamp(1)", "select to_timestamp(null)"] {
        let types = schema(&s, sql).unwrap();
        assert_eq!(types[0].1, "Timestamp(µs)", "{sql}");
    }
    // Without the NTZ setting the same pattern still falls back to the legacy check.
    let plain = session(&[]);
    assert_eq!(
        class(&plain, "select to_timestamp('x', 'yyyy-MM-dd GGGGG')"),
        "INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_PATTERN_RECOGNITION"
    );
}

#[test]
fn from_unixtime_reads_its_arguments_before_it_reports_the_pattern() {
    let s = session(&["spark.sql.ansi.enabled=true"]);
    block_on(async {
        s.sql("create temporary view t as select 'aa' as a")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(
        class(&s, "select from_unixtime('2018-01-01', a) from t"),
        "CAST_INVALID_INPUT"
    );
    // A bad pattern with a good seconds argument is still the pattern's error.
    assert_eq!(
        class(&s, "select from_unixtime(1, a) from t"),
        "INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_PATTERN_RECOGNITION"
    );
}

#[test]
fn time_trunc_reads_a_string_as_time() {
    let s = session(&["spark.sql.timeType.enabled=true"]);
    let types = schema(&s, "SELECT time_trunc('HOUR', '12:34:56')").unwrap();
    assert_eq!(types[0].0, "time_trunc(HOUR, 12:34:56)");
    // Other time functions still refuse a string for their time parameter.
    assert_eq!(
        class(&s, "SELECT make_timestamp_ntz(DATE'2025-06-20', '0:0:0')"),
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
}

/// The first value of `sql`'s result as text.
fn text(s: &Session, sql: &str) -> Option<String> {
    use arrow_array::cast::AsArray;
    let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let a = b[0].column(0);
    if a.is_null(0) {
        return None;
    }
    Some(a.as_string_opt::<i32>().map_or_else(
        || a.as_string_view().value(0).to_owned(),
        |x| x.value(0).to_owned(),
    ))
}

#[test]
fn date_difference_under_legacy_intervals_is_whole_months_and_days() {
    // `spark.sql.legacy.interval.enabled`: `date - date` is a calendar interval of
    // `Period.between(start, end)` (Spark 4.2.0 local run, 2026-11-06; `postgreSQL/date` #46-53).
    let s = session(&["spark.sql.legacy.interval.enabled=true"]);
    for (end, start, expected) in [
        ("2000-03-30", "2000-01-31", "1 months 30 days"),
        ("2000-02-29", "2000-01-31", "29 days"),
        ("2000-01-31", "2000-03-30", "-1 months -30 days"),
        ("2000-01-15", "2000-03-20", "-2 months -5 days"),
        ("2000-01-31", "2000-02-29", "-29 days"),
        ("1957-04-09", "2000-01-01", "-42 years -8 months -22 days"),
        ("2039-04-10", "2000-01-01", "39 years 3 months 9 days"),
        ("2000-01-01", "2000-01-01", "0 seconds"),
        ("2001-02-28", "2000-02-29", "11 months 30 days"),
        ("1999-12-31", "2000-03-01", "-2 months -1 days"),
    ] {
        let sql = format!("select cast(date '{end}' - date '{start}' as string)");
        assert_eq!(text(&s, &sql).as_deref(), Some(expected), "{sql}");
    }
    assert_eq!(
        text(&s, "select typeof(date '2000-01-02' - date '2000-01-01')").as_deref(),
        Some("interval")
    );
    assert_eq!(
        text(
            &s,
            "select cast(date '2000-01-02' - cast(null as date) as string)"
        ),
        None
    );
    assert_eq!(
        text(
            &s,
            "select cast(date '2000-01-02' - '2000-01-01' as string)"
        )
        .as_deref(),
        Some("1 days")
    );
    assert_eq!(
        names(&s, "select date '2000-01-02' - date '2000-01-01'"),
        ["(DATE '2000-01-02' - DATE '2000-01-01')"]
    );
}
