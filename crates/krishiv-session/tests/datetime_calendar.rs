// SPDX-License-Identifier: Apache-2.0
//! The calendar functions end to end: the function-matrix cells recorded
//! from Spark 4.2 (`tests/fixtures/spark-4.2.0/function-matrix`, both ANSI
//! modes) replayed through SQL, and the literal statements of the
//! `date.sql` and `datetime-special.sql` goldens that use only these
//! functions.
//!
//! A matrix cell must give Spark's rows (as `CAST(x AS STRING)` renders
//! them) or Spark's error class and parameters. A cell the engine cannot
//! run yet (intervals, `TIME`, region time zones, nested values) is
//! counted with the reason, and the exact skip set is pinned by
//! `expected_not_run` (with the two parameter normalizations counted too),
//! so a regression that turns a result into `unsupported` fails instead of
//! passing silently.
#![cfg(test)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Date32Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;
use serde_json::Value;

/// The functions whose cells are replayed.
const FUNCTIONS: &[&str] = &[
    "hour",
    "minute",
    "second",
    "dayofweek",
    "dayofyear",
    "weekofyear",
    "weekday",
    "quarter",
    "monthname",
    "dayname",
    "unix_date",
    "add_months",
    "last_day",
    "next_day",
    "trunc",
    "date_trunc",
    "months_between",
    "make_date",
    "from_utc_timestamp",
    "to_utc_timestamp",
    "convert_timezone",
];

fn session(ansi: bool) -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql(&format!("SET spark.sql.ansi.enabled = {ansi}"))).unwrap();
    // The matrix was recorded in UTC.
    block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
    s
}

/// The rows of `batches` as text (`None` for NULL).
fn rows_of(batches: &[RecordBatch]) -> Vec<Vec<Option<String>>> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let row = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(i) {
                        return None;
                    }
                    Some(match c.data_type() {
                        ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                        ArrowType::LargeUtf8 => c.as_string::<i64>().value(i).to_owned(),
                        _ => c.as_string::<i32>().value(i).to_owned(),
                    })
                })
                .collect();
            out.push(row);
        }
    }
    out
}

/// Runs `sql` (a `SELECT` of one expression) and renders the value as a
/// string, as the matrix recorded it.
fn render(s: &Session, sql: &str) -> Result<Vec<Vec<Option<String>>>, KrishivError> {
    let expr = sql.strip_prefix("SELECT ").unwrap_or(sql);
    let wrapped = format!("SELECT CAST(c AS STRING) FROM (SELECT {expr} AS c)");
    let df = block_on(s.sql(&wrapped))?;
    Ok(rows_of(&block_on(df.collect())?))
}

/// What Spark recorded for a case in a mode.
enum Expected {
    Rows(Vec<Vec<Option<String>>>),
    Error(String, BTreeMap<String, String>),
}

fn expected(case: &Value, mode: &str) -> Option<Expected> {
    let mut cell = &case[mode];
    if let Some(other) = cell["same_as"].as_str() {
        cell = &case[other];
    }
    if let Some(e) = cell.get("error") {
        let params = e["message_parameters"]
            .as_object()?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_owned()))
            .collect();
        return Some(Expected::Error(
            e["error_class"].as_str()?.to_owned(),
            params,
        ));
    }
    let rows = cell["rows"]
        .as_array()?
        .iter()
        .map(|r| {
            r.as_array()
                .map(|r| r.iter().map(|v| v.as_str().map(str::to_owned)).collect())
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Expected::Rows(rows))
}

#[derive(Default)]
struct Tally {
    passed: usize,
    failures: String,
    not_run: BTreeMap<String, usize>,
    /// Cells compared after dropping the registry's missing `ansiConfig`
    /// parameter from Spark's `CAST_INVALID_INPUT`.
    normalized_cast_invalid_input: usize,
    /// Cells compared after filling Spark's default `roundOff` into the
    /// `months_between` call shown in its `sqlExpr`.
    normalized_months_between: usize,
}

fn check(tally: &mut Tally, s: &Session, case: &Value, mode: &str) {
    let sql = case["sql"].as_str().unwrap_or_default();
    let id = case["id"].as_str().unwrap_or_default();
    let Some(want) = expected(case, mode) else {
        *tally
            .not_run
            .entry("no recorded rows".to_owned())
            .or_default() += 1;
        return;
    };
    let got = render(s, sql);
    let outcome: Result<(), String> = match (&want, &got) {
        // The cast of a binary that is not UTF-8 to a string is the cast
        // kernel's own business (`INVALID_UTF8_STRING`), not a calendar one.
        (_, Err(e)) if e.error_class() == ErrorClass::InvalidUtf8String => {
            *tally
                .not_run
                .entry("cast of invalid UTF-8 bytes to a string".to_owned())
                .or_default() += 1;
            return;
        }
        (_, Err(e)) if e.error_class() == ErrorClass::KrishivUnsupportedFeature => {
            let feature = e
                .message_parameters()
                .iter()
                .find(|(k, _)| *k == "feature")
                .map_or_else(String::new, |(_, v)| v.clone());
            // The planner names the whole call: keep the function.
            let feature = match feature.strip_prefix("the expression ") {
                Some(call) => format!(
                    "the function {} (it has no binding for these arguments: a region time zone)",
                    call.split('(').next().unwrap_or_default()
                ),
                None => feature,
            };
            *tally
                .not_run
                .entry(format!("unsupported: {feature}"))
                .or_default() += 1;
            return;
        }
        (Expected::Rows(rows), Ok(g)) => (rows == g)
            .then_some(())
            .ok_or_else(|| format!("rows {g:?} where Spark has {rows:?}")),
        (Expected::Rows(rows), Err(e)) => Err(format!("{e} where Spark has {rows:?}")),
        (Expected::Error(class, _), Ok(g)) => Err(format!("rows {g:?} where Spark has {class}")),
        // `convert_timezone(zone, ts)` is rewritten to the three-argument
        // call with `current_timezone()` before Spark analyzes it, so its
        // type errors name that call and the third parameter.
        (Expected::Error(_, params), Err(_))
            if params
                .get("sqlExpr")
                .is_some_and(|e| e.contains("current_timezone()")) =>
        {
            *tally
                .not_run
                .entry("analysis errors of the rewritten convert_timezone".to_owned())
                .or_default() += 1;
            return;
        }
        (Expected::Error(class, params), Err(e)) => {
            let ours: BTreeMap<String, String> = e
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect();
            let mut params = params.clone();
            // The registry's `CAST_INVALID_INPUT` has no `ansiConfig`.
            if class == "CAST_INVALID_INPUT" {
                params.remove("ansiConfig");
                tally.normalized_cast_invalid_input += 1;
            }
            // Spark shows `months_between`'s default `roundOff` in the call.
            if let Some(shown) = params.get_mut("sqlExpr")
                && let Some(plain) = ours.get("sqlExpr")
                && shown
                    .strip_suffix(", true)\"")
                    .map(|s| format!("{s})\""))
                    .as_ref()
                    == Some(plain)
            {
                *shown = plain.clone();
                tally.normalized_months_between += 1;
            }
            if e.error_class().name() != class {
                Err(format!(
                    "{} where Spark has {class}",
                    e.error_class().name()
                ))
            } else if ours != params {
                Err(format!("{class} with {ours:?} where Spark has {params:?}"))
            } else {
                Ok(())
            }
        }
    };
    match outcome {
        Ok(()) => tally.passed += 1,
        Err(why) => {
            let _ = writeln!(tally.failures, "{id} [{mode}] {sql}: {why}");
        }
    }
}

#[test]
fn matrix_cells_match_spark() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/function-matrix/cases/datetime.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let cases: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .filter(|c: &Value| {
            c["function"]
                .as_str()
                .is_some_and(|f| FUNCTIONS.contains(&f))
        })
        .collect();
    assert!(cases.len() > 400, "{}", cases.len());
    let mut tally = Tally::default();
    for (mode, ansi) in [("ansi_on", true), ("ansi_off", false)] {
        let s = session(ansi);
        for case in &cases {
            if case["conf"].as_object().is_some_and(|c| !c.is_empty()) {
                *tally
                    .not_run
                    .entry("session conf override".to_owned())
                    .or_default() += 1;
                continue;
            }
            check(&mut tally, &s, case, mode);
        }
    }
    let mut report = format!("{} cells passed\n", tally.passed);
    for (why, n) in &tally.not_run {
        let _ = writeln!(report, "{n:5} not run: {why}");
    }
    let _ = writeln!(
        report,
        "normalized: {} CAST_INVALID_INPUT, {} months_between",
        tally.normalized_cast_invalid_input, tally.normalized_months_between
    );
    println!("{report}");
    let shown: Vec<&str> = tally.failures.lines().take(25).collect();
    assert!(shown.is_empty(), "{}\n{report}", shown.join("\n"));
    assert!(tally.passed > 1_000, "{report}");
    // The skip set is pinned: a regression that turns a result into
    // `unsupported` (or any new skip) changes these counts and fails here
    // instead of passing silently.
    assert_eq!(tally.not_run, expected_not_run(), "{report}");
    assert_eq!(tally.normalized_cast_invalid_input, 96, "{report}");
    assert_eq!(tally.normalized_months_between, 44, "{report}");
}

/// The exact cells the engine does not run: region time zones (no tz
/// database), `TIME` and `INTERVAL` literals, nested values, invalid UTF-8
/// bytes, the rewritten `convert_timezone` analysis errors, session-conf
/// overrides and cases with no recorded rows.
fn expected_not_run() -> BTreeMap<String, usize> {
    BTreeMap::from([
        (
            "analysis errors of the rewritten convert_timezone".to_owned(),
            24,
        ),
        ("cast of invalid UTF-8 bytes to a string".to_owned(), 23),
        ("no recorded rows".to_owned(), 62),
        ("session conf override".to_owned(), 4),
        ("unsupported: CAST from STRING to BOOLEAN".to_owned(), 4),
        ("unsupported: INTERVAL literals".to_owned(), 144),
        ("unsupported: TIME literals".to_owned(), 6),
        ("unsupported: function DATE".to_owned(), 4),
        ("unsupported: function map".to_owned(), 74),
        ("unsupported: function named_struct".to_owned(), 74),
        (
            "unsupported: the function convert_timezone (it has no binding for these arguments: a region time zone)"
                .to_owned(),
            68,
        ),
        (
            "unsupported: the function from_utc_timestamp (it has no binding for these arguments: a region time zone)"
                .to_owned(),
            24,
        ),
        (
            "unsupported: the function to_utc_timestamp (it has no binding for these arguments: a region time zone)"
                .to_owned(),
            24,
        ),
        ("unsupported: the time data type (\"TIME\")".to_owned(), 6),
    ])
}

// ---------------------------------------------------------------------------
// The literal statements of the SQL goldens (`tests/spark-golden/v4.2.0`).
//
// Claimed: the statements of `date.sql` and `datetime-special.sql` that use
// only the functions of this change (and `year`/`month`, which sit in the
// same statements) with constructs the engine has. Not claimed:
// `unix_date(DATE(...))` and `date_from_unix_date` (the `DATE()` function and
// the latter are not here), `to_date` and the unit forms of `date_add`,
// `date_diff`, `timestampadd` and `timestampdiff` (their own changes),
// `make_timestamp` and `try_make_timestamp`, whose type follows
// `spark.sql.timestampType`, the `TIME`-typed forms, the `convert_timezone`
// and `make_timestamp_*` statements with region zones (`Europe/Brussels`,
// `CET`), and everything that reads intervals.
// ---------------------------------------------------------------------------

/// The row of `SELECT <exprs>` as Spark prints it: columns joined by tabs.
fn row(s: &Session, exprs: &[&str]) -> String {
    let columns: Vec<String> = (0..exprs.len()).map(|i| format!("c{i}")).collect();
    let items: Vec<String> = exprs
        .iter()
        .zip(&columns)
        .map(|(e, c)| format!("{e} AS {c}"))
        .collect();
    let casts: Vec<String> = columns
        .iter()
        .map(|c| format!("CAST({c} AS STRING)"))
        .collect();
    let sql = format!(
        "SELECT {} FROM (SELECT {})",
        casts.join(", "),
        items.join(", ")
    );
    let df = block_on(s.sql(&sql)).unwrap();
    let rows = rows_of(&block_on(df.collect()).unwrap());
    assert_eq!(rows.len(), 1, "{exprs:?}");
    rows[0]
        .iter()
        .map(|v| v.clone().unwrap_or_else(|| "NULL".to_owned()))
        .collect::<Vec<_>>()
        .join("\t")
}

/// The output names of `SELECT <exprs>`.
fn names(s: &Session, exprs: &[&str]) -> Vec<String> {
    let df = block_on(s.sql(&format!("SELECT {}", exprs.join(", ")))).unwrap();
    block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// The error of `SELECT <expr>`, from analysis or execution.
fn failure(s: &Session, expr: &str) -> KrishivError {
    match block_on(s.sql(&format!("SELECT {expr}"))).and_then(|df| block_on(df.collect())) {
        Ok(b) => panic!("{expr} ran: {b:?}"),
        Err(e) => e,
    }
}

fn params(e: &KrishivError) -> BTreeMap<String, String> {
    e.message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect()
}

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

/// Registers a one-column table `name`.
fn register(s: &Session, name: &str, field: Field, values: ArrayRef) {
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values]).unwrap();
    s.register_batches(name, schema, vec![batch]).unwrap();
}

const ANSI_CONF: &str = "\"spark.sql.ansi.enabled\"";

#[test]
fn make_date_goldens() {
    let s = session(true);
    // date.sql
    let both = ["make_date(2019, 1, 1)", "make_date(12, 12, 12)"];
    assert_eq!(row(&s, &both), "2019-01-01\t0012-12-12");
    assert_eq!(
        names(&s, &both),
        ["make_date(2019, 1, 1)", "make_date(12, 12, 12)"]
    );
    for (expr, range) in [
        (
            "make_date(2000, 13, 1)",
            "Invalid value for MonthOfYear (valid values 1 - 12): 13",
        ),
        (
            "make_date(2000, 1, 33)",
            "Invalid value for DayOfMonth (valid values 1 - 28/31): 33",
        ),
    ] {
        let e = failure(&s, expr);
        assert_eq!(
            e.error_class(),
            ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion
        );
        assert_eq!(
            params(&e),
            map(&[("ansiConfig", ANSI_CONF), ("rangeMessage", range)])
        );
    }
    // datetime-special.sql
    let special = ["make_date(999999, 3, 18)", "make_date(-1, 1, 28)"];
    assert_eq!(row(&s, &special), "+999999-03-18\t-0001-01-28");
    assert_eq!(
        names(&s, &special),
        ["make_date(999999, 3, 18)", "make_date(-1, 1, 28)"]
    );
    // With ANSI off the invalid dates are NULL.
    let legacy = session(false);
    assert_eq!(
        row(
            &legacy,
            &["make_date(2000, 13, 1)", "make_date(2000, 1, 33)"]
        ),
        "NULL\tNULL"
    );
}

#[test]
fn field_goldens() {
    let s = session(true);
    // date.sql: `dayofweek`, `weekday`, `year`, `month` and `dayOfYear` accept
    // dates, strings and both timestamp types.
    let probes = |f: &str| {
        [
            format!("{f}('2007-02-03')"),
            format!("{f}('2009-07-30')"),
            format!("{f}('2017-05-27')"),
            format!("{f}(null)"),
            format!("{f}('1582-10-15 13:10:15')"),
            format!("{f}(timestamp_ltz'1582-10-15 13:10:15')"),
            format!("{f}(timestamp_ntz'1582-10-15 13:10:15')"),
        ]
    };
    for (f, want) in [
        ("dayofweek", "7\t5\t7\tNULL\t6\t6\t6"),
        ("weekday", "5\t3\t5\tNULL\t4\t4\t4"),
    ] {
        let exprs = probes(f);
        let refs: Vec<&str> = exprs.iter().map(String::as_str).collect();
        assert_eq!(row(&s, &refs), want, "{f}");
    }
    let shown = names(
        &s,
        &[
            "dayofweek('2007-02-03')",
            "dayofweek(null)",
            "dayofweek(timestamp_ntz'1582-10-15 13:10:15')",
        ],
    );
    assert_eq!(
        shown,
        [
            "dayofweek(2007-02-03)",
            "dayofweek(NULL)",
            "dayofweek(TIMESTAMP_NTZ '1582-10-15 13:10:15')"
        ]
    );
    let others = [
        ("year", ["1500", "1582", "1582", "1582"]),
        ("month", ["1", "10", "10", "10"]),
        ("dayOfYear", ["1", "288", "288", "288"]),
    ];
    for (f, want) in others {
        let exprs = [
            format!("{f}('1500-01-01')"),
            format!("{f}('1582-10-15 13:10:15')"),
            format!("{f}(timestamp_ltz'1582-10-15 13:10:15')"),
            format!("{f}(timestamp_ntz'1582-10-15 13:10:15')"),
        ];
        let refs: Vec<&str> = exprs.iter().map(String::as_str).collect();
        assert_eq!(row(&s, &refs), want.join("\t"), "{f}");
    }
    assert_eq!(
        names(&s, &["dayOfYear('1500-01-01')"]),
        ["dayofyear(1500-01-01)"]
    );
}

#[test]
fn next_day_goldens() {
    let s = session(true);
    for (expr, want) in [
        ("next_day(\"2015-07-23\", \"Mon\")", "2015-07-27"),
        ("next_day(\"2015-07-23 12:12:12\", \"Mon\")", "2015-07-27"),
        (
            "next_day(timestamp_ltz\"2015-07-23 12:12:12\", \"Mon\")",
            "2015-07-27",
        ),
        (
            "next_day(timestamp_ntz\"2015-07-23 12:12:12\", \"Mon\")",
            "2015-07-27",
        ),
        ("next_day(null, \"Mon\")", "NULL"),
        ("next_day(null, \"xx\")", "NULL"),
    ] {
        assert_eq!(row(&s, &[expr]), want, "{expr}");
    }
    assert_eq!(
        names(&s, &["next_day(\"2015-07-23\", \"Mon\")"]),
        ["next_day(2015-07-23, Mon)"]
    );
    assert_eq!(
        names(
            &s,
            &["next_day(timestamp_ntz\"2015-07-23 12:12:12\", \"Mon\")"]
        ),
        ["next_day(TIMESTAMP_NTZ '2015-07-23 12:12:12', Mon)"]
    );
    // `ILLEGAL_DAY_OF_WEEK` for an unknown name...
    let e = failure(&s, "next_day(\"2015-07-23\", \"xx\")");
    assert_eq!(e.error_class(), ErrorClass::IllegalDayOfWeek);
    assert_eq!(params(&e), map(&[("string", "xx")]));
    // ... and `CAST_INVALID_INPUT` for a date that is not one (the registry's
    // `CAST_INVALID_INPUT` carries no `ansiConfig`: a cast concern).
    let e = failure(&s, "next_day(\"xx\", \"Mon\")");
    assert_eq!(e.error_class(), ErrorClass::CastInvalidInput);
    assert_eq!(
        params(&e),
        map(&[
            ("expression", "'xx'"),
            ("sourceType", "\"STRING\""),
            ("targetType", "\"DATE\"")
        ])
    );
    // With ANSI off the unknown name is NULL.
    let legacy = session(false);
    assert_eq!(
        row(
            &legacy,
            &[
                "next_day(\"2015-07-23\", \"xx\")",
                "next_day(\"xx\", \"Mon\")"
            ]
        ),
        "NULL\tNULL"
    );
}

#[test]
fn calendar_functions_in_queries_over_columns() {
    let s = session(true);
    let exprs = [
        "d",
        "last_day(d)",
        "next_day(d, 'FR')",
        "trunc(d, 'MM')",
        "add_months(d, 1)",
        "monthname(d)",
        "dayname(d)",
        "quarter(d)",
        "weekofyear(d)",
    ];
    let columns: Vec<String> = exprs
        .iter()
        .map(|e| format!("CAST({e} AS STRING)"))
        .collect();
    register(
        &s,
        "dates",
        Field::new("d", ArrowType::Date32, true),
        Arc::new(Date32Array::from(vec![Some(19_782), None, Some(19_722)])),
    );
    let sql = format!("SELECT {} FROM dates ORDER BY d", columns.join(", "));
    let df = block_on(s.sql(&sql)).unwrap();
    let got: Vec<Vec<String>> = rows_of(&block_on(df.collect()).unwrap())
        .into_iter()
        .map(|r| {
            r.into_iter()
                .map(|v| v.unwrap_or_else(|| "NULL".to_owned()))
                .collect()
        })
        .collect();
    assert_eq!(got[0], ["NULL"; 9]);
    assert_eq!(
        got[1],
        [
            "2023-12-31",
            "2023-12-31",
            "2024-01-05",
            "2023-12-01",
            "2024-01-31",
            "Dec",
            "Sun",
            "4",
            "52"
        ]
    );
    assert_eq!(
        got[2],
        [
            "2024-02-29",
            "2024-02-29",
            "2024-03-01",
            "2024-02-01",
            "2024-03-29",
            "Feb",
            "Thu",
            "1",
            "9"
        ]
    );
    // CASE does not run the branch of an unselected row: `make_date(2000, 13, 1)`
    // raises, but not for a row the WHEN skips.
    register(
        &s,
        "months",
        Field::new("m", ArrowType::Int32, false),
        Arc::new(Int32Array::from(vec![1, 13])),
    );
    let guarded = "SELECT CASE WHEN m <= 12 THEN CAST(make_date(2000, m, 1) AS STRING) \
                   ELSE 'none' END FROM months";
    let df = block_on(s.sql(guarded)).unwrap();
    let out = rows_of(&block_on(df.collect()).unwrap());
    assert_eq!(
        out,
        [[Some("2000-01-01".to_owned())], [Some("none".to_owned())]]
    );
    // Without the guard the same row raises.
    let e = block_on(s.sql("SELECT CAST(make_date(2000, m, 1) AS STRING) FROM months"))
        .and_then(|df| block_on(df.collect()))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DatetimeFieldOutOfBoundsWithSuggestion,
        "{e}"
    );
}
