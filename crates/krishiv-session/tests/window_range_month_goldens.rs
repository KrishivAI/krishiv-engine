// SPDX-License-Identifier: Apache-2.0
//! `RANGE` window frames with a year-month interval bound against Spark 4.2.0 (the fixture also holds the 3.5.7, 4.0.0
//! and 4.1.0 results, which this test does not compare): each window case of
//! `tests/fixtures/spark-4.2.0/window-range-month-goldens.json` (from real Spark runs, see
//! `krishiv-internal/research/spark-notes/range-month-bounds.md`) runs through a session and must give Spark's
//! rows (compared as sets) or error class. Cases that are not `RANGE` windows (plain timestamp arithmetic) and
//! cases whose result columns carry Spark-generated names are listed as skipped, never silently dropped.
#![cfg(test)]

use std::fmt::Write as _;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, TimeUnit};
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::{Map, Value};

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/window-range-month-goldens.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Days since the epoch of a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// Epoch microseconds of Spark's JSON timestamp text (`2024-04-30T23:59:59.999Z`, `...-08:00`).
fn micros_of(text: &str) -> Option<i64> {
    let (date, rest) = text.split_once('T')?;
    let mut p = date.split('-');
    let (y, m, d): (i64, i64, i64) = (
        p.next()?.parse().ok()?,
        p.next()?.parse().ok()?,
        p.next()?.parse().ok()?,
    );
    let (clock, offset) = if let Some(c) = rest.strip_suffix('Z') {
        (c, 0)
    } else if !rest.contains(['+', '-']) {
        (rest, 0)
    } else {
        let at = rest.rfind(['+', '-'])?;
        let (c, o) = rest.split_at(at);
        let sign = if o.starts_with('-') { -1 } else { 1 };
        // `hh:mm`, or `hh:mm:ss` (year 0001 in Los Angeles is -07:52:58).
        let mut parts = o[1..].split(':');
        let mut next = || parts.next().map_or(Some(0), |p| p.parse::<i64>().ok());
        let (oh, om, os) = (next()?, next()?, next()?);
        (c, sign * (oh * 3600 + om * 60 + os))
    };
    let (hms, frac) = clock.split_once('.').unwrap_or((clock, "0"));
    let mut t = hms.split(':');
    let secs = t.next()?.parse::<i64>().ok()? * 3600
        + t.next()?.parse::<i64>().ok()? * 60
        + t.next()?.parse::<i64>().ok()?;
    let micros: i64 = format!("{frac:0<6}")[..6].parse().ok()?;
    Some(((days_from_civil(y, m, d) * 86_400 + secs - offset) * 1_000_000) + micros)
}

/// A result value in the fixture's JSON form; timestamps as epoch milliseconds (Spark's JSON writes millis).
fn value(a: &ArrayRef, i: usize) -> Value {
    if a.is_null(i) {
        return Value::Null;
    }
    match a.data_type() {
        DataType::Int32 => Value::from(a.as_primitive::<Int32Type>().value(i)),
        DataType::Int64 => Value::from(a.as_primitive::<Int64Type>().value(i)),
        // Spark's JSON writes a decimal as a number.
        DataType::Decimal128(_, scale) => {
            let unscaled = a
                .as_primitive::<arrow_array::types::Decimal128Type>()
                .value(i);
            Value::from(unscaled as f64 / 10_f64.powi(i32::from(*scale)))
        }
        DataType::Date32 => {
            let (y, m, d) = civil_from_days(i64::from(a.as_primitive::<Date32Type>().value(i)));
            Value::from(format!("{y:04}-{m:02}-{d:02}"))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => Value::from(
            a.as_primitive::<TimestampMicrosecondType>()
                .value(i)
                .div_euclid(1000),
        ),
        DataType::List(_) | DataType::LargeList(_) | DataType::ListView(_) => {
            let inner = if let Some(l) = a.as_list_opt::<i32>() {
                l.value(i)
            } else {
                a.as_list::<i64>().value(i)
            };
            let mut v: Vec<Value> = (0..inner.len()).map(|j| value(&inner, j)).collect();
            v.sort_by_key(ToString::to_string);
            Value::Array(v)
        }
        t => panic!("result type {t}"),
    }
}

/// Spark's value normalised like ours: timestamps to epoch microseconds, lists sorted.
fn want_value(v: &Value, ours: &Value) -> Value {
    match (v, ours) {
        (Value::String(s), Value::Number(_)) => {
            micros_of(s).map_or_else(|| v.clone(), |m| Value::from(m.div_euclid(1000)))
        }
        (Value::Array(a), _) => {
            let mut a = a.clone();
            a.sort_by_key(ToString::to_string);
            Value::Array(a)
        }
        _ => v.clone(),
    }
}

enum Outcome {
    Checked,
    Skipped(String),
}

fn check(s: &Session, case: &Value, report: &mut String) -> Outcome {
    let id = case["id"].as_str().unwrap();
    let sql = case["sql"].as_str().unwrap();
    if !sql.contains("RANGE BETWEEN") || !sql.trim_start().starts_with("SELECT") {
        return Outcome::Skipped(format!("{id}: not a RANGE window query"));
    }
    let want = &case["results"]["4.2.0"];
    let conf = s.conf();
    let settings = case["conf"].as_object().unwrap();
    for (k, v) in settings {
        conf.set(k, v.as_str().unwrap()).unwrap();
    }
    let result = block_on(async {
        let df = s.sql(sql).await?;
        let schema = df.schema().await?;
        let batches = df.collect().await?;
        Ok::<_, krishiv_common::error::KrishivError>((schema, batches))
    });
    for k in settings.keys() {
        conf.unset(k).unwrap();
    }
    if let Some(err) = want.get("error") {
        let class = err["class"].as_str();
        return match (result, class) {
            (Err(e), Some(class)) => {
                // The message text of the mixed-bound-type error: Spark's, without its position suffix.
                let spark = err["message"].as_str().unwrap_or_default();
                let head = |m: &str| m.split(" SQLSTATE").next().unwrap_or(m).to_owned();
                if class.ends_with("SPECIFIED_WINDOW_FRAME_DIFF_TYPES")
                    && head(&e.to_string()) != head(spark)
                {
                    let _ = writeln!(
                        report,
                        "{id}: message\n  got  {}\n  want {}",
                        head(&e.to_string()),
                        head(spark)
                    );
                }
                if e.error_class().name() != class {
                    let _ = writeln!(
                        report,
                        "{id}: error {} vs Spark {class}",
                        e.error_class().name()
                    );
                }
                Outcome::Checked
            }
            (Ok(_), Some(class)) => {
                let _ = writeln!(report, "{id}: ran; Spark raised {class}");
                Outcome::Checked
            }
            // Spark raises a bare `ArithmeticException("long overflow")`, in both ANSI modes: the engine raises the
            // registry class that describes it, as its datetime kernels do.
            (Err(e), None) => {
                if e.error_class().name() != "DATETIME_OVERFLOW" {
                    let _ = writeln!(
                        report,
                        "{id}: Spark's unclassified error; got {}",
                        e.error_class().name()
                    );
                }
                Outcome::Checked
            }
            (Ok(_), None) => {
                let _ = writeln!(report, "{id}: ran; Spark raised an unclassified error");
                Outcome::Checked
            }
        };
    }
    let (schema, batches) = match result {
        Ok(r) => r,
        Err(e) => {
            let _ = writeln!(report, "{id}: failed: {e}");
            return Outcome::Checked;
        }
    };
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let rows = want["rows"].as_array().unwrap();
    if rows.iter().any(|r| {
        r.as_object()
            .unwrap()
            .keys()
            .any(|k| !names.contains(&k.as_str()))
    }) {
        return Outcome::Skipped(format!("{id}: result column names differ from Spark's"));
    }
    let mut got: Vec<String> = Vec::new();
    let mut expected: Vec<String> = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let mut m = Map::new();
            for (n, c) in names.iter().zip(b.columns()) {
                let v = value(c, i);
                if !v.is_null() {
                    m.insert((*n).to_owned(), v);
                }
            }
            got.push(Value::Object(m).to_string());
        }
    }
    let probe: Vec<ArrayRef> = batches
        .first()
        .map(|b| b.columns().to_vec())
        .unwrap_or_default();
    for r in rows {
        let mut m = Map::new();
        for (k, v) in r.as_object().unwrap() {
            let ty_is_time = names
                .iter()
                .position(|n| n == k)
                .and_then(|p| probe.get(p))
                .is_some_and(|c| matches!(c.data_type(), DataType::Timestamp(..)));
            let ours = if ty_is_time {
                Value::from(0)
            } else {
                Value::Null
            };
            m.insert(k.clone(), want_value(v, &ours));
        }
        expected.push(Value::Object(m).to_string());
    }
    got.sort();
    expected.sort();
    if got != expected {
        let _ = writeln!(report, "{id}: rows\n  got  {got:?}\n  want {expected:?}");
    }
    Outcome::Checked
}

/// Cases the test does not run: plain timestamp arithmetic that is not a `RANGE` window (the kernels' own tests
/// cover it).
const EXPECTED_SKIPPED: usize = 7;

/// Cases Krishiv does not match yet, with the reason. Each must still mismatch: a fix removes its line.
const KNOWN_GAPS: &[(&str, &str)] = &[
    (
        "d_calendar_interval_bound_legacy_off",
        "RANGE_FRAME_INVALID_TYPE for a calendar-interval bound (ours: unsupported)",
    ),
    (
        "inv_ymi_key_month_bound",
        "a year-month interval ordering key",
    ),
    (
        "ymi_key_ymi_bound_same_type",
        "a year-month interval ordering key",
    ),
    (
        "ymi_year_key_month_bound",
        "a year-month interval ordering key",
    ),
    (
        "d_null_bound",
        "a NULL or CAST(.. AS INTERVAL) bound is not a literal to the analyzer",
    ),
    (
        "d_cast_bound",
        "a NULL or CAST(.. AS INTERVAL) bound is not a literal to the analyzer",
    ),
];

#[test]
fn range_month_windows_match_spark() {
    let g = fixture();
    let s = Session::new().unwrap();
    let mut report = String::new();
    let (mut checked, mut skipped, mut gaps) = (0, Vec::new(), 0);
    for case in g["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let mut one = String::new();
        match check(&s, case, &mut one) {
            Outcome::Skipped(why) => {
                // A known gap must stay checkable: one that is skipped could never be seen fixed.
                if KNOWN_GAPS.iter().any(|(g, _)| *g == id) {
                    let _ = writeln!(
                        report,
                        "{id}: a known gap that can no longer be checked ({why})"
                    );
                }
                skipped.push(why);
            }
            Outcome::Checked => {
                checked += 1;
                match (KNOWN_GAPS.iter().find(|(g, _)| *g == id), one.is_empty()) {
                    (Some(_), false) => gaps += 1,
                    (Some((_, why)), true) => {
                        let _ = writeln!(
                            report,
                            "{id}: now matches Spark; remove it from KNOWN_GAPS ({why})"
                        );
                    }
                    (None, _) => report.push_str(&one),
                }
            }
        }
    }
    // Every skipped case is named in the failure text, and their number is pinned: a new skip is a decision.
    assert_eq!(
        skipped.len(),
        EXPECTED_SKIPPED,
        "skipped cases:\n{}",
        skipped.join("\n")
    );
    assert!(checked >= 100, "only {checked} cases checked");
    assert_eq!(
        gaps,
        KNOWN_GAPS.len(),
        "a known gap is missing from the fixture"
    );
    assert!(
        report.is_empty(),
        "mismatches against Spark 4.2.0:\n{report}\n({checked} checked, {gaps} known gaps, {} skipped:\n{})",
        skipped.len(),
        skipped.join("\n")
    );
}

/// A `DAY` interval bound over a `DATE` key is Spark 4.0 and later; 3.5.7 raises `RANGE_FRAME_INVALID_TYPE` (the
/// fixture case `d_dt_day_bound` holds both results; the golden test above compares 4.2.0 only).
///
/// Known differences of 3.5.7 not modelled (the engine has one message per class): its most-negative-interval error is
/// the base class `INTERVAL_ARITHMETIC_OVERFLOW` (4.x: `.WITHOUT_SUGGESTION`), and `RANGE_FRAME_INVALID_TYPE` says
/// "does not match" where 4.x says "does not support".
#[test]
fn a_day_bound_on_a_date_key_needs_spark_4_0() {
    use krishiv_plan::analyzer::SparkProfile;
    let sql = "SELECT count(*) OVER (ORDER BY d RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW) \
               FROM (VALUES (DATE'2024-01-31'), (DATE'2024-02-29')) t(d)";
    let run = |profile| {
        let s = Session::builder().profile(profile).build().unwrap();
        block_on(async { s.sql(sql).await?.collect().await })
    };
    let e = run(SparkProfile::V3_5).unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "DATATYPE_MISMATCH.RANGE_FRAME_INVALID_TYPE"
    );
    for profile in [SparkProfile::V4_0, SparkProfile::V4_1, SparkProfile::V4_2] {
        assert!(run(profile).is_ok(), "{profile:?}");
    }
}

/// Bounds of two types over a date or timestamp key raise `SPECIFIED_WINDOW_FRAME_DIFF_TYPES`, before either type is
/// checked against the key (the fixture's `d_diff_types_*` and `ts_diff_types_*` cases compare the message text).
#[test]
fn bounds_of_two_types_are_refused_before_the_key_is_checked() {
    let s = Session::new().unwrap();
    for (key, rows) in [
        ("d", "(DATE'2024-01-31'), (DATE'2024-02-29')"),
        (
            "ts",
            "(TIMESTAMP'2024-01-31 10:00:00'), (TIMESTAMP'2024-02-29 10:00:00')",
        ),
    ] {
        for frame in [
            "1 PRECEDING AND INTERVAL '1' MONTH FOLLOWING",
            "INTERVAL '1' MONTH PRECEDING AND INTERVAL '1' DAY FOLLOWING",
            "INTERVAL '1' YEAR PRECEDING AND INTERVAL '1' MONTH FOLLOWING",
        ] {
            let sql = format!(
                "SELECT count(*) OVER (ORDER BY {key} RANGE BETWEEN {frame}) FROM (VALUES {rows}) t({key})"
            );
            let e = block_on(async { s.sql(&sql).await?.collect().await }).unwrap_err();
            assert_eq!(
                e.error_class().name(),
                "DATATYPE_MISMATCH.SPECIFIED_WINDOW_FRAME_DIFF_TYPES",
                "{sql}"
            );
        }
    }
}

/// A DataFrame registered as a temporary view is analyzed again when the view is read: a day-time bound over a
/// timestamp key (kept as its interval, not as a bare integer) and a month bound must analyze a second time.
#[test]
fn a_window_with_interval_bounds_survives_a_dataframe_temp_view() {
    let s = Session::new().unwrap();
    for (name, bound) in [
        ("v_day_time", "INTERVAL '1' DAY PRECEDING AND CURRENT ROW"),
        ("v_months", "INTERVAL '1' MONTH PRECEDING AND CURRENT ROW"),
    ] {
        for (key, rows) in [
            (
                "ts",
                "(TIMESTAMP'2024-01-31 10:00:00'), (TIMESTAMP'2024-02-01 09:00:00'), (TIMESTAMP'2024-02-29 10:00:00')",
            ),
            (
                "d",
                "(DATE'2024-01-31'), (DATE'2024-02-01'), (DATE'2024-02-29')",
            ),
        ] {
            if key == "d" && name == "v_day_time" {
                continue; // a DATE key takes its day-time bound as whole days in an integer, which re-analyzes as before
            }
            let view = format!("{name}_{key}");
            let q = format!(
                "SELECT {key}, count(*) OVER (ORDER BY {key} RANGE BETWEEN {bound}) AS c FROM (VALUES {rows}) t({key})"
            );
            let direct = block_on(async { s.sql(&q).await?.collect().await }).unwrap();
            block_on(async { s.sql(&q).await?.create_or_replace_temp_view(&view).await }).unwrap();
            let again = block_on(async {
                s.sql(&format!("SELECT * FROM {view}"))
                    .await?
                    .collect()
                    .await
            });
            let again = again.unwrap_or_else(|e| panic!("{view}: {e}"));
            assert_eq!(
                direct
                    .iter()
                    .map(arrow_array::RecordBatch::num_rows)
                    .sum::<usize>(),
                again
                    .iter()
                    .map(arrow_array::RecordBatch::num_rows)
                    .sum::<usize>(),
                "{view}"
            );
            assert_eq!(format!("{direct:?}"), format!("{again:?}"), "{view}");
        }
    }
}
