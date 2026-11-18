// SPDX-License-Identifier: Apache-2.0
//! A JSON or CSV read of only the corrupt-record column (`UNSUPPORTED_FEATURE.QUERY_ONLY_CORRUPT_RECORD_COLUMN`,
//! SQLSTATE 0A000; PY-076) against Spark 4.2.0: the probes of
//! `tests/fixtures/spark-4.2.0/csv-json-corrupt-record/goldens.json` (`json.*`, `csv.*`, `pq_as_json.*`; real Spark
//! runs, generator `tests/scripts/gen_csv_json_corrupt_goldens.scala`) run through a session over the fixture's
//! input files. The error is raised at the first action, not when the frame is built; a count, `SELECT *`, a mix
//! with another column and a cached read all work.
#![cfg(test)]

use std::path::Path;

use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::{DataFrame, Session};
use serde_json::Value;
use tempfile::TempDir;

const QUERY_ONLY: &str = "UNSUPPORTED_FEATURE.QUERY_ONLY_CORRUPT_RECORD_COLUMN";

fn goldens() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/csv-json-corrupt-record/goldens.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// The fixture's input files in a fresh directory.
fn inputs(g: &Value) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (name, hex) in g["inputs_hex"].as_object().unwrap() {
        std::fs::write(dir.path().join(name), unhex(hex.as_str().unwrap())).unwrap();
    }
    dir
}

/// Spark's `Row.toString` of every row, joined by ` | ` (the generator's `rows`).
fn render(batches: &[RecordBatch]) -> String {
    let mut rows = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(i) {
                        return "null".to_owned();
                    }
                    match c.data_type() {
                        DataType::Int32 => c.as_primitive::<Int32Type>().value(i).to_string(),
                        DataType::Int64 => c.as_primitive::<Int64Type>().value(i).to_string(),
                        DataType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                        DataType::LargeUtf8 => c.as_string::<i64>().value(i).to_owned(),
                        DataType::Utf8View => c.as_string_view().value(i).to_owned(),
                        other => panic!("unexpected {other}"),
                    }
                })
                .collect();
            rows.push(format!("[{}]", cells.join(",")));
        }
    }
    rows.join(" | ")
}

/// What an action gives: the rows, or the error's class and SQLSTATE.
type Outcome = Result<String, (String, Option<String>)>;

fn fail(e: &KrishivError) -> (String, Option<String>) {
    (
        e.error_class().name().to_owned(),
        e.error_class().sql_state().map(str::to_owned),
    )
}

fn rows(df: Result<DataFrame, KrishivError>) -> Outcome {
    let df = df.map_err(|e| fail(&e))?;
    block_on(df.collect())
        .map(|b| render(&b))
        .map_err(|e| fail(&e))
}

/// `count()` of `SELECT * FROM <from>`: an aggregate, so the scan reads no column of its own.
fn count(s: &Session, from: &str) -> Outcome {
    rows(sql(s, &format!("SELECT count(*) FROM {from}")))
        .map(|r| r.trim_matches(['[', ']']).to_owned())
}

fn sql(s: &Session, q: &str) -> Result<DataFrame, KrishivError> {
    block_on(s.sql(q))
}

/// `CREATE OR REPLACE TEMPORARY VIEW v (schema) USING fmt OPTIONS (path, extra...)`.
fn view(s: &Session, fmt: &str, path: &str, schema: &str, extra: &str) {
    sql(
        s,
        &format!(
            "CREATE OR REPLACE TEMPORARY VIEW v ({schema}) USING {fmt} OPTIONS (path '{path}'{extra})"
        ),
    )
    .unwrap();
}

fn check(g: &Value, failures: &mut Vec<String>, label: &str, got: &Outcome) {
    let want = &g["probes"][label];
    let ok = match (want["status"].as_str().unwrap(), got) {
        ("ok", Ok(v)) => {
            // Strings in the file are `\u`-escaped for non-ASCII only; these values are ASCII.
            want["value"].as_str().unwrap() == v
        }
        ("error", Err((class, state))) => {
            want["condition"].as_str() == Some(class.as_str())
                && (want["condition"].as_str() != Some(QUERY_ONLY)
                    || (state.as_deref() == want["sqlstate"].as_str()))
        }
        _ => false,
    };
    if !ok {
        failures.push(format!(
            "{label}: Spark {}, got {got:?}",
            if want["status"] == "ok" {
                want["value"].to_string()
            } else {
                want["condition"].to_string()
            }
        ));
    }
}

/// Probe labels this engine does not match yet, with why; each must still differ, so the list only shrinks.
const KNOWN: &[(&str, &str)] = &[];

fn run(fmt: &str, file: &str) {
    let g = goldens();
    let dir = inputs(&g);
    let path = dir.path().join(file).display().to_string();
    let s = Session::new().unwrap();
    let sch = "a INT, _corrupt_record STRING";
    let sch2 = "a INT, _bad STRING";
    let bad = ", columnNameOfCorruptRecord '_bad'";
    let mut out: Vec<(String, Outcome)> = Vec::new();
    let mut put = |l: &str, o: Outcome| out.push((format!("{fmt}.{l}"), o));

    view(&s, fmt, &path, sch, "");
    let v = || sql(&s, "SELECT * FROM v");
    put(
        "select_corrupt",
        rows(sql(&s, "SELECT _corrupt_record FROM v")),
    );
    put("count", count(&s, "v"));
    put("select_star", rows(v()));
    put(
        "where_corrupt_notnull",
        rows(sql(&s, "SELECT * FROM v WHERE _corrupt_record IS NOT NULL")),
    );
    put(
        "where_corrupt_notnull_count",
        count(&s, "v WHERE _corrupt_record IS NOT NULL"),
    );
    put(
        "where_corrupt_notnull_select_a",
        rows(sql(&s, "SELECT a FROM v WHERE _corrupt_record IS NOT NULL")),
    );
    put(
        "select_corrupt_a",
        rows(sql(&s, "SELECT _corrupt_record, a FROM v")),
    );
    put("select_a", rows(sql(&s, "SELECT a FROM v")));
    // The schema of the select is known before any action.
    let schema = block_on(sql(&s, "SELECT _corrupt_record FROM v").unwrap().schema()).unwrap();
    assert_eq!(
        schema.fields().len(),
        1,
        "{fmt}: the schema needs no action"
    );

    let cached = || v().map(|d| d.cache());
    let d = cached().unwrap();
    block_on(d.count()).unwrap();
    put(
        "cached_select_corrupt",
        rows(d.select_expr(["_corrupt_record"])),
    );
    put(
        "cached_where_corrupt",
        rows(d.filter_expr("_corrupt_record IS NOT NULL")),
    );
    put(
        "cache_then_select_corrupt_no_prior_action",
        rows(cached().and_then(|d| d.select_expr(["_corrupt_record"]))),
    );

    // The renamed corrupt column first in the schema: a count and a filtered count (the filter reads it, so Spark
    // raises) behave as for the default name.
    view(&s, fmt, &path, "_bad STRING, a INT", bad);
    put("renamed.first.count", count(&s, "v"));
    assert!(
        count(&s, "v WHERE _bad IS NOT NULL").is_err_and(|e| e.0 == QUERY_ONLY),
        "{fmt}: a filter on the renamed column"
    );
    // The DataFrame `count()` (Python `df.count()`) is a count over the frame, whatever it selects.
    view(&s, fmt, &path, sch, "");
    let want = g["probes"][format!("{fmt}.count").as_str()]["value"]
        .as_str()
        .unwrap()
        .to_owned();
    for q in [
        "SELECT * FROM v",
        "SELECT _corrupt_record FROM v",
        "SELECT * FROM v WHERE a IS NULL OR a IS NOT NULL",
    ] {
        let n = block_on(sql(&s, q).unwrap().count()).map(|n| n.to_string());
        assert_eq!(
            n.as_deref().ok(),
            Some(want.as_str()),
            "{fmt}: df.count() of `{q}`"
        );
    }
    // `explain` shows the plan; only running it raises (Spark's check is in the reader).
    for mode in [
        krishiv_session::ExplainMode::Simple,
        krishiv_session::ExplainMode::Extended,
    ] {
        let df = sql(&s, "SELECT _corrupt_record FROM v").unwrap();
        let text = block_on(df.explain(mode));
        assert!(text.is_ok(), "{fmt}: explain: {text:?}");
    }
    // KNOWN DIFFERENCES (they never refuse a valid query): Spark raises for a UNION ALL branch or a CTE that
    // reads only the corrupt column; here the branch / CTE is not seen as such.
    let union = rows(sql(
        &s,
        "SELECT CAST(a AS STRING) FROM v UNION ALL SELECT _corrupt_record FROM v",
    ));
    let cte = rows(sql(
        &s,
        "WITH c AS (SELECT _corrupt_record FROM v) SELECT * FROM c",
    ));
    // The check sees the columns the plan references by attribute; a union branch and a CTE body rename them
    // (positional union inputs; the CTE reference has attributes of its own), so neither is recognised.
    assert!(union.is_ok(), "{fmt}: known difference, union: {union:?}");
    assert!(cte.is_ok(), "{fmt}: known difference, cte: {cte:?}");
    view(&s, fmt, &path, sch2, bad);
    put("renamed.select_bad", rows(sql(&s, "SELECT _bad FROM v")));
    put(
        "renamed.select_bad_a",
        rows(sql(&s, "SELECT _bad, a FROM v")),
    );
    put(
        "renamed.select_default_name",
        rows(sql(&s, "SELECT _corrupt_record FROM v")),
    );
    view(&s, fmt, &path, sch, bad);
    put(
        "renamed.default_name_in_schema.select_corrupt",
        rows(sql(&s, "SELECT _corrupt_record FROM v")),
    );
    view(
        &s,
        fmt,
        &path,
        sch2,
        &format!("{bad}, mode 'DROPMALFORMED'"),
    );
    put(
        "renamed.select_bad_dropmalformed",
        rows(sql(&s, "SELECT _bad FROM v")),
    );
    view(&s, fmt, &path, sch, ", mode 'FAILFAST'");
    put(
        "failfast.select_corrupt",
        rows(sql(&s, "SELECT _corrupt_record FROM v")),
    );
    s.conf()
        .set("spark.sql.columnNameOfCorruptRecord", "_c2")
        .unwrap();
    view(&s, fmt, &path, "a INT, _c2 STRING", "");
    put("sql_session_conf_name", rows(sql(&s, "SELECT _c2 FROM v")));
    s.conf()
        .unset("spark.sql.columnNameOfCorruptRecord")
        .unwrap();
    put(
        "sql_table_function",
        rows(sql(
            &s,
            &format!("SELECT _corrupt_record FROM {fmt}.`{path}`"),
        )),
    );
    put(
        "sql_table_function_star",
        rows(sql(&s, &format!("SELECT * FROM {fmt}.`{path}`"))),
    );
    if fmt == "json" {
        let ds = || s.read().format("json").load([path.as_str()]);
        put(
            "inferred.select_corrupt",
            rows(ds().select_expr(["_corrupt_record"])),
        );
        put("inferred.select_star", rows(Ok(ds())));
        put("inferred.count", count(&s, &format!("json.`{path}`")));
        put(
            "inferred.where_corrupt",
            rows(ds().filter_expr("_corrupt_record IS NOT NULL")),
        );
        let d = ds().cache();
        block_on(d.count()).unwrap();
        put(
            "inferred.cached_select_corrupt",
            rows(d.select_expr(["_corrupt_record"])),
        );
        view(&s, fmt, &path, sch, ", multiLine 'true'");
        put(
            "multiline.select_corrupt",
            rows(sql(&s, "SELECT _corrupt_record FROM v")),
        );
    }
    drop(put);
    let mut failures = Vec::new();
    for (label, got) in &out {
        if label.ends_with("renamed.first.count") {
            let want = g["probes"][format!("{fmt}.count").as_str()]["value"]
                .as_str()
                .unwrap();
            assert_eq!(got.as_deref(), Ok(want), "{label}");
            continue;
        }
        let known = KNOWN.iter().find(|(l, _)| l == label);
        let mut f = Vec::new();
        check(&g, &mut f, label, got);
        match (known, f.is_empty()) {
            (None, false) => failures.extend(f),
            (Some((l, _)), true) => failures.push(format!("{l} is in KNOWN but matches")),
            _ => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn json_corrupt_only_reads_match_spark() {
    run("json", "bad.json");
}

#[test]
fn csv_corrupt_only_reads_match_spark() {
    run("csv", "bad.csv");
}

/// A Parquet directory read as JSON infers `struct<_corrupt_record:string>`, so every read of its rows is a read of
/// only that column.
#[test]
fn a_parquet_directory_read_as_json_is_refused_whenever_its_rows_are_read() {
    let g = goldens();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pq").display().to_string();
    let s = Session::new().unwrap();
    let df = sql(&s, "SELECT id AS a, CAST(id AS STRING) AS b FROM range(3)").unwrap();
    block_on(df.write().parquet(&path)).unwrap();
    let ds = || s.read().format("json").load([path.as_str()]);
    let counted = block_on(ds().count());
    assert!(
        counted.is_ok(),
        "df.count() of the only-corrupt-column read: {counted:?}"
    );
    let schema = block_on(ds().schema()).unwrap();
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "_corrupt_record");
    let mut failures = Vec::new();
    for (label, got) in [
        ("pq_as_json.collect_count", rows(Ok(ds()))),
        ("pq_as_json.select_star", rows(ds().select_expr(["*"]))),
        (
            "pq_as_json.select_corrupt",
            rows(ds().select_expr(["_corrupt_record"])),
        ),
        ("pq_as_json.cached_collect", {
            let d = ds().cache();
            let first = block_on(d.count()).map_err(|e| fail(&e));
            first.and_then(|_| rows(Ok(d)))
        }),
    ] {
        check(&g, &mut failures, label, &got);
    }
    // `count()` reads no column, so it is not refused (Spark: the number of newline-separated "lines" of the one
    // binary part file, 2). Krishiv's count is whatever its line splitting gives; the difference is recorded, not
    // asserted, as the brief allows.
    let n = count(&s, &format!("json.`{path}`"))
        .expect("a count of the only-corrupt-column read works");
    let spark = g["probes"]["pq_as_json.count"]["value"].as_str().unwrap();
    if n != spark {
        eprintln!(
            "KNOWN DIFFERENCE pq_as_json.count: Spark {spark} (lines of the binary file), here {n} (the lines of Krishiv's own Parquet file: a different file, so not comparable)"
        );
    }
    assert!(n.parse::<u64>().is_ok(), "{n}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
