// SPDX-License-Identifier: Apache-2.0
//! Metric views (P3 4.2-MV) against the Spark runs in `tests/fixtures/spark-{4.2.1,4.3.0}/metric-views/`
//! (`tests/scripts/gen_metric_view_goldens.py`; the README there gives the format and the `match`
//! modes). Each case runs in a fresh session and database, as the generator ran it. A case
//! matches when every `sql` and `conf` step does: an error by class and SQLSTATE, plus its
//! parameters unless the step compares by `class_sqlstate_prefix`; a result by its column names
//! and rows. `api` steps (DataFrame calls) are checked by the Connect test.
//!
//! `KNOWN_GAPS` lists the cases not matched yet, by the step that adds them (design note
//! `P3-metric-views.md` §7): each must still fail, so a fix removes it from the list.
#![cfg(test)]

use std::collections::BTreeSet;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, DecimalType, Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

#[path = "metric_views/gaps.rs"]
mod gaps;

fn fixture(release: &str) -> Value {
    let path = format!(
        "{}/../../tests/fixtures/spark-{release}/metric-views/cases.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn cell(c: &ArrayRef, i: usize) -> Value {
    if c.is_null(i) {
        return Value::Null;
    }
    match c.data_type() {
        DataType::Boolean => Value::from(c.as_boolean().value(i)),
        DataType::Int32 => Value::from(c.as_primitive::<Int32Type>().value(i)),
        DataType::Int64 => Value::from(c.as_primitive::<Int64Type>().value(i)),
        DataType::Float64 => {
            let v = c.as_primitive::<Float64Type>().value(i);
            // Spark's JSON writes these as strings.
            if v.is_nan() {
                Value::from("NaN")
            } else if v.is_infinite() {
                Value::from(if v > 0.0 { "Infinity" } else { "-Infinity" })
            } else {
                Value::from(v)
            }
        }
        DataType::Decimal128(_, scale) => {
            let v = c.as_primitive::<Decimal128Type>().value(i);
            Value::from(Decimal128Type::format_decimal(v, 38, *scale))
        }
        DataType::Utf8 => Value::from(c.as_string::<i32>().value(i)),
        DataType::Utf8View => Value::from(c.as_string_view().value(i)),
        DataType::LargeUtf8 => Value::from(c.as_string::<i64>().value(i)),
        other => Value::from(format!("<{other}>")),
    }
}

fn rows(batches: &[RecordBatch]) -> Vec<Value> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            out.push(Value::Array(
                b.columns().iter().map(|c| cell(c, i)).collect(),
            ));
        }
    }
    out
}

/// A row as text, with `<normalized>` cells kept as the fixture wrote them.
/// A text cell without what differs between runs and patch releases: the creating release
/// (`Created By`, the engine says 4.2.0, the goldens ran on 4.2.1 and 4.3.0) and, inside
/// `SHOW TABLE EXTENDED`'s information, the times the goldens print as `<normalized>`.
fn text(cell: &str) -> String {
    let release = |s: &str| {
        s.strip_prefix("Spark ")
            .is_some_and(|v| v.split('.').all(|p| p.parse::<u32>().is_ok()))
    };
    if release(cell) {
        return "Spark <release>".to_owned();
    }
    if !cell.contains('\n') {
        return cell.to_owned();
    }
    cell.split('\n')
        .map(|line| match line.split_once(": ") {
            Some(("Created Time" | "Last Access", _)) => {
                format!(
                    "{}: <normalized>",
                    line.split_once(": ").map_or("", |p| p.0)
                )
            }
            Some(("Created By", v)) if release(v) => "Created By: Spark <release>".to_owned(),
            _ => line.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalized(row: &Value, expected: &Value) -> String {
    match (row, expected) {
        (Value::Array(r), Value::Array(e)) => {
            let cells: Vec<String> = r
                .iter()
                .zip(e)
                .map(|(r, e)| {
                    if e == "<normalized>" {
                        e.to_string()
                    } else if let (Some(a), Some(b)) = (r.as_f64(), e.as_f64()) {
                        if a == b { e.to_string() } else { a.to_string() }
                    } else if let Some(r) = r.as_str() {
                        Value::String(text(r)).to_string()
                    } else {
                        r.to_string()
                    }
                })
                .collect();
            format!("[{}]", cells.join(","))
        }
        _ => row.to_string(),
    }
}

fn error_matches(e: &KrishivError, want: &Value, by_class: bool) -> Result<(), String> {
    let condition = want["condition"].as_str().unwrap_or("INTERNAL_ERROR");
    let class = e.error_class().name();
    if class != condition {
        return Err(format!("class {class} vs {condition}: {e}"));
    }
    if let Some(state) = want["sqlstate"].as_str()
        && e.sql_state() != Some(state)
    {
        return Err(format!("SQLSTATE {:?} vs {state}: {e}", e.sql_state()));
    }
    if by_class {
        return Ok(());
    }
    // The complete maps: a parameter Spark gives that the engine leaves out fails too.
    let ours: std::collections::BTreeMap<&str, &str> = e
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.as_str()))
        .collect();
    let theirs: std::collections::BTreeMap<&str, &str> = want["parameters"]
        .as_object()
        .map(|m| {
            m.iter()
                // Spark 4.2's cast errors still pass `ansiConfig`, which their message no longer
                // names (`CAST_INVALID_INPUT` in `spark-error-conditions-4.2.1.json`): not shown,
                // so not compared.
                .filter(|(k, _)| !(condition == "CAST_INVALID_INPUT" && k.as_str() == "ansiConfig"))
                .filter_map(|(k, v)| Some((k.as_str(), v.as_str()?)))
                .collect()
        })
        .unwrap_or_default();
    if ours != theirs {
        return Err(format!("parameters {ours:?} vs {theirs:?}"));
    }
    Ok(())
}

fn run_step(s: &Session, step: &Value) -> Result<(), String> {
    let outcome = &step["outcome"];
    if step["kind"] == "conf" {
        let (k, v) = (
            step["key"].as_str().unwrap(),
            step["value"].as_str().unwrap(),
        );
        return s.conf().set(k, v).map_err(|e| e.to_string());
    }
    let text = step["sql"].as_str().unwrap();
    let result = block_on(async {
        let df = s.sql(text).await?;
        let schema = df.schema().await?;
        Ok::<_, KrishivError>((schema, df.collect().await?))
    });
    let by_class = step["match"] == "class_sqlstate_prefix";
    match (result, outcome.get("error")) {
        (Err(e), Some(want)) => error_matches(&e, want, by_class),
        (Err(e), None) => Err(format!("failed: {e}")),
        (Ok(_), Some(want)) => Err(format!("succeeded, Spark failed: {}", want["message"])),
        (Ok((schema, batches)), None) => {
            // The columns first, in order: an extra or renamed column fails even when the rows
            // compare equal cell by cell.
            if let Some(columns) = outcome["schema"].as_array() {
                let theirs: Vec<&str> = columns.iter().filter_map(|c| c["name"].as_str()).collect();
                let ours: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
                if ours != theirs {
                    return Err(format!("columns {ours:?} vs {theirs:?}"));
                }
            }
            let mut want = outcome["rows"].as_array().cloned().unwrap_or_default();
            let mut got = rows(&batches);
            if outcome["row_order"] == "sorted" {
                got.sort_by_key(Value::to_string);
                want.sort_by_key(Value::to_string);
            }
            if got.len() != want.len() {
                return Err(format!("{} rows vs {}: {got:?}", got.len(), want.len()));
            }
            let mut got: Vec<String> = got
                .iter()
                .zip(&want)
                .map(|(g, w)| normalized(g, w))
                .collect();
            let mut want: Vec<String> = want.iter().map(|w| normalized(w, w)).collect();
            if outcome["row_order"] == "sorted" {
                got.sort();
                want.sort();
            }
            if got == want {
                Ok(())
            } else {
                Err(format!("rows {got:?} vs {want:?}"))
            }
        }
    }
}

/// The first step of `case` that does not match, if any.
fn run_case(case: &Value) -> Option<String> {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", &dir.path().display().to_string())
        .unwrap();
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    let db = case["database"].as_str().unwrap_or_default();
    for setup in [format!("CREATE DATABASE {db}"), format!("USE {db}")] {
        block_on(async { s.sql(&setup).await?.collect().await }).unwrap();
    }
    for (i, step) in case["steps"].as_array().into_iter().flatten().enumerate() {
        if step["kind"] == "api" {
            continue;
        }
        if let Err(why) = run_step(&s, step) {
            return Some(format!("step {i}: {why}"));
        }
    }
    None
}

fn check(release: &str, gaps: &[&str]) {
    let fixture = fixture(release);
    let mut failing = BTreeSet::new();
    let mut report = String::new();
    let mut all = String::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if let Some(why) = run_case(case) {
            all.push_str(&format!("{name}: {why}\n"));
            failing.insert(name.to_owned());
            if !gaps.contains(&name) {
                report.push_str(&format!("{name}: {why}\n"));
            }
        }
    }
    // `METRIC_VIEWS_FAILURES=<dir>` writes every failing case, gaps included, and why.
    if let Ok(dir) = std::env::var("METRIC_VIEWS_FAILURES") {
        std::fs::write(format!("{dir}/failures-{release}.txt"), &all).unwrap();
    }
    let gaps: BTreeSet<String> = gaps.iter().map(|g| (*g).to_owned()).collect();
    let fixed: Vec<_> = gaps.difference(&failing).collect();
    assert!(
        report.is_empty() && fixed.is_empty(),
        "{release}: not matched (and not a known gap):\n{report}\nknown gaps that now match (remove them): {fixed:?}"
    );
}

#[test]
fn spark_4_2_1() {
    check("4.2.1", gaps::GAPS);
}

#[test]
fn spark_4_3_0() {
    let gaps: Vec<&str> = gaps::GAPS.iter().chain(gaps::GAPS_4_3).copied().collect();
    check("4.3.0", &gaps);
}

/// The classes only these goldens raise (`cargo xtask registry-coverage` reads test sources, not
/// fixtures). Each must be the expected error of a step in a 4.2.1 case that is no known gap, so
/// `spark_4_2_1` matches it.
const GOLDEN_CLASSES: &[&str] = &[
    "MISSING_ATTRIBUTES.RESOLVED_ATTRIBUTE_APPEAR_IN_OPERATION",
    "UNSUPPORTED_FEATURE.ANALYZE_VIEW",
    "UNSUPPORTED_INSERT.RDD_BASED",
    "UNSUPPORTED_SHOW_CREATE_TABLE.ON_METRIC_VIEW",
];

#[test]
fn the_goldens_raise_these_classes() {
    let fixture = fixture("4.2.1");
    for class in GOLDEN_CLASSES {
        let matched = fixture["cases"].as_array().unwrap().iter().any(|case| {
            !gaps::GAPS.contains(&case["name"].as_str().unwrap_or_default())
                && case["steps"].as_array().into_iter().flatten().any(|step| {
                    step["kind"] != "api" && step["outcome"]["error"]["condition"] == *class
                })
        });
        assert!(matched, "no matched 4.2.1 case expects {class}");
    }
}
