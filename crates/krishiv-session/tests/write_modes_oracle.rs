// SPDX-License-Identifier: Apache-2.0
//! `DataFrameWriter.saveAsTable` against the Spark 4.3.0 run of `tests/fixtures/spark-4.3.0/write-semantics/
//! save_as_table_modes.json` (P3-T023, PR-30): each scripted case builds its existing table, runs the write and
//! compares the table's rows (unordered) or the error class. Cases the run skipped are not replayed. A case that
//! does not match yet is named in `KNOWN_GAPS` with its reason, and must still fail (a fixed one leaves the list).
#![cfg(test)]

use arrow_array::Array;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

/// Cases whose Krishiv result differs from Spark's, with the reason; each must still differ.
const KNOWN_GAPS: &[(&str, &str)] = &[];

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.3.0/write-semantics/save_as_table_modes.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| a.iter().map(|x| x.as_str().unwrap().to_owned()).collect())
        .unwrap_or_default()
}

fn literal(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_owned(),
        Value::String(s) => format!("'{}'", s.replace('\'', "''")),
        other => other.to_string(),
    }
}

/// The table `t`'s rows as sorted text, every column cast to a string.
fn rows_of(s: &Session) -> Result<Vec<String>, String> {
    let df = s.table("t");
    let schema = block_on(df.schema()).map_err(|e| e.to_string())?;
    let cols: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| format!("coalesce(cast(`{}` AS string), 'NULL')", f.name()))
        .collect();
    let line = format!("concat_ws('|', {}) AS line", cols.join(", "));
    let out = block_on(df.select_expr([line]).map_err(|e| e.to_string())?.collect())
        .map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for b in out {
        let a = b.column(0);
        for i in 0..a.len() {
            rows.push(arrow_cast::display::array_value_to_string(a, i).map_err(|e| e.to_string())?);
        }
    }
    rows.sort();
    Ok(rows)
}

fn expected_rows(rows: &Value) -> Vec<String> {
    let mut v: Vec<String> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| match x {
                    Value::Null => "NULL".to_owned(),
                    Value::String(s) => s.clone(),
                    o => o.to_string(),
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    v.sort();
    v
}

/// Runs one case; `Ok` is the comparison result (`Err` text says what differs).
fn run(case: &Value) -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let s = Session::builder()
        .config("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .build()
        .map_err(|e| e.to_string())?;
    if let Some(conf) = case.get("conf").and_then(Value::as_object) {
        for (k, v) in conf {
            s.conf()
                .set(k, v.as_str().unwrap())
                .map_err(|e| e.to_string())?;
        }
    }
    let sql = |text: &str| -> Result<(), String> {
        block_on(async { s.sql(text).await?.collect().await })
            .map(drop)
            .map_err(|e| format!("setup `{text}`: {e}"))
    };
    if let Some(existing) = case["existing"].as_object() {
        if let Some(view) = existing.get("view") {
            sql(view.as_str().unwrap())?;
        } else {
            let parts = strings(&existing["partitionBy"]);
            let by = if parts.is_empty() {
                String::new()
            } else {
                format!(" PARTITIONED BY ({})", parts.join(", "))
            };
            sql(&format!(
                "CREATE TABLE t ({}) USING {}{by}",
                existing["schema"].as_str().unwrap(),
                existing["format"].as_str().unwrap()
            ))?;
            for row in existing["rows"].as_array().unwrap() {
                let vals: Vec<String> = row.as_array().unwrap().iter().map(literal).collect();
                sql(&format!("INSERT INTO t VALUES ({})", vals.join(", ")))?;
            }
        }
    }
    let w = &case["write"];
    let df = block_on(s.sql(w["query"].as_str().unwrap())).map_err(|e| e.to_string())?;
    let mut writer = df
        .write()
        .format(w["format"].as_str().unwrap())
        .mode(w["mode"].as_str().unwrap());
    let parts = strings(&w["partitionBy"]);
    if !parts.is_empty() {
        writer = writer.partition_by(parts);
    }
    if let Some(opts) = w["options"].as_object() {
        for (k, v) in opts {
            writer = writer.option(k, v.as_str().unwrap());
        }
    }
    let result = block_on(writer.save_as_table("t"));
    let observed = &case["observed"];
    match observed["kind"].as_str().unwrap() {
        "error" => match result {
            Ok(()) => Err(format!(
                "expected {}, the write succeeded",
                observed["class"]
            )),
            Err(e) => {
                let got = e.error_class().name().to_owned();
                if got == observed["class"].as_str().unwrap() {
                    compare_params(observed, &e)
                } else {
                    Err(format!("expected {}, got {got}: {e}", observed["class"]))
                }
            }
        },
        "rows" => {
            result.map_err(|e| format!("the write failed: {e}"))?;
            let got = rows_of(&s)?;
            let want = expected_rows(&observed["rows"]);
            if got == want {
                Ok(())
            } else {
                Err(format!("rows {got:?}, expected {want:?}"))
            }
        }
        other => Err(format!("unreplayable kind {other}")),
    }
}

/// The observed parameters against the error's, with the run's scratch schema (`ws_...`) read as `default`.
fn compare_params(observed: &Value, e: &krishiv_common::error::KrishivError) -> Result<(), String> {
    let Some(params) = observed["params"].as_object() else {
        return Ok(());
    };
    let norm = |v: &str| {
        let mut out = String::new();
        let mut rest = v;
        while let Some(i) = rest.find("ws_") {
            out.push_str(&rest[..i]);
            let tail = &rest[i..];
            let end = tail
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(tail.len());
            out.push_str("default");
            rest = &tail[end..];
        }
        out.push_str(rest);
        out
    };
    for (k, want) in params {
        let got = e
            .message_parameters()
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.clone());
        let want = norm(want.as_str().unwrap());
        // Spark prints a table as `spark_catalog.db.t`; Krishiv's catalog name is read the same way.
        if got.as_deref().map(norm).as_deref() != Some(want.as_str()) {
            return Err(format!("parameter {k}: expected {want}, got {got:?}: {e}"));
        }
    }
    Ok(())
}

#[test]
fn save_as_table_modes_match_the_spark_run() {
    let doc = fixture();
    let mut failures = Vec::new();
    let mut replayed = 0;
    for case in doc["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        if case["observed"]["kind"] == "skipped" {
            continue;
        }
        replayed += 1;
        let gap = KNOWN_GAPS.iter().find(|(g, _)| *g == id);
        match (run(case), gap) {
            (Ok(()), Some((_, why))) => failures.push(format!(
                "{id}: now matches; remove it from KNOWN_GAPS ({why})"
            )),
            (Err(e), None) => failures.push(format!("{id}: {e}")),
            _ => {}
        }
    }
    assert!(replayed >= 40, "only {replayed} cases replayed");
    assert!(
        failures.is_empty(),
        "{} of {replayed} differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
