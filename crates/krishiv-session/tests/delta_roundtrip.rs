// SPDX-License-Identifier: Apache-2.0
//! Krishiv's SQL in the Delta round trip (P3-T028b, PR-53b rest part 2; feature `delta`): on the
//! `sql_plain`, `sql_part` (partitioned by `x`) and `sql_dv` (deletion vectors) tables that Spark
//! 4.2.0 with delta-spark 4.4.1 makes (`tests/scripts/gen_delta_roundtrip.py create`), Krishiv runs
//! an `INSERT` (data files and statistics from its Delta writer, through a catalog table over the
//! location: ``INSERT INTO delta.`<path>` `` is not supported yet), a copy-on-write `DELETE` and
//! `UPDATE` that rewrite Spark's files and its own (on `sql_dv` a Spark file with a deletion
//! vector), on `sql_part` a whole-file `DELETE` of a partition, and a second `INSERT`
//! ([`krishiv_sql_on_spark_tables`], `KRISHIV-SQL.json`); Spark reads every version back, with
//! each commit's history entry and metrics, and runs `DELETE`, `UPDATE`, `MERGE` and an append on
//! top (`readback`); [`krishiv_reads_spark_sql_on_top`] compares the two engines' records.
//!
//! Both tests are `#[ignore]`d steps of `tests/scripts/delta_roundtrip.sh` (Docker), run with
//! `KRISHIV_DELTA_RT_OUT` naming the shared directory. No expected value is written by hand
//! except the known `isBlindAppend` difference (the connector's CLAUDE.md).
#![cfg(test)]

use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::{Value, json};

const OUT_VAR: &str = "KRISHIV_DELTA_RT_OUT";

/// The tables Krishiv's SQL runs on, and whether each is partitioned by `x`.
const TABLES: [(&str, bool); 3] = [("sql_plain", false), ("sql_part", true), ("sql_dv", false)];

fn out_dir() -> PathBuf {
    PathBuf::from(
        std::env::var(OUT_VAR).unwrap_or_else(|_| panic!("{OUT_VAR} names the output directory")),
    )
}

fn read_json(name: &str) -> Value {
    let text = std::fs::read_to_string(out_dir().join(name))
        .unwrap_or_else(|e| panic!("{name}: {e} (run the earlier steps first)"));
    serde_json::from_str(&text).unwrap()
}

fn table_dir(name: &str) -> PathBuf {
    out_dir().join("delta").join(name)
}

fn table(name: &str) -> String {
    format!("delta.`{}`", table_dir(name).display())
}

fn cell(column: &dyn Array, row: usize) -> Value {
    if column.is_null(row) {
        return Value::Null;
    }
    match column.data_type() {
        ArrowType::Int64 => Value::from(column.as_primitive::<Int64Type>().value(row)),
        ArrowType::Int32 => Value::from(column.as_primitive::<Int32Type>().value(row)),
        ArrowType::Utf8 => Value::from(column.as_string::<i32>().value(row)),
        ArrowType::Utf8View => Value::from(column.as_string_view().value(row)),
        other => panic!("no record form for {other}"),
    }
}

fn rows_of(batches: &[RecordBatch]) -> Vec<Value> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            out.push(Value::Array(
                b.columns().iter().map(|c| cell(c.as_ref(), r)).collect(),
            ));
        }
    }
    out
}

fn sql(s: &Session, text: &str) -> Vec<Value> {
    block_on(async {
        let df = s.sql(text).await?;
        df.collect().await
    })
    .map(|b| rows_of(&b))
    .unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// Krishiv's read of the table's latest version, as rows of `[id, x, name]`.
fn krishiv_read(s: &Session, name: &str) -> Vec<Value> {
    sql(s, &format!("SELECT id, x, name FROM {}", table(name)))
}

/// The latest version in the table's log.
fn latest_version(name: &str) -> u64 {
    std::fs::read_dir(table_dir(name).join("_delta_log"))
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".json")?.parse::<u64>().ok()
        })
        .max()
        .unwrap()
}

/// The JSON actions of one commit.
fn commit_actions(name: &str, version: u64) -> Vec<Value> {
    std::fs::read_to_string(
        table_dir(name)
            .join("_delta_log")
            .join(format!("{version:020}.json")),
    )
    .unwrap()
    .lines()
    .map(|l| serde_json::from_str(l).unwrap())
    .collect()
}

/// Spark's rows (`{"id", "x", "name"}` objects) as Krishiv's `[id, x, name]` form.
fn spark_rows(v: &Value) -> Vec<Value> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| json!([r["id"], r["x"], r["name"]]))
        .collect()
}

fn sorted(mut v: Vec<Value>) -> Vec<Value> {
    v.sort_by_key(std::string::ToString::to_string);
    v
}

/// `VALUES` rows for `ids` with `x = id % 3` (0 for a NULL id) and `name = n<id>`, as Krishiv's
/// connector step writes them.
fn values(ids: &[Option<i64>]) -> String {
    ids.iter()
        .map(|i| match i {
            Some(i) => format!("({i}, {}, 'n{i}')", i % 3),
            None => "(NULL, 0, 'n-1')".to_owned(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Step 2b: Krishiv's SQL on Spark's `sql_*` tables.
#[test]
#[ignore = "step 2b of tests/scripts/delta_roundtrip.sh (needs SPARK-CREATE.json from Spark)"]
fn krishiv_sql_on_spark_tables() {
    let spark = read_json("SPARK-CREATE.json");
    assert_eq!(spark["_oracle"]["delta"], "4.4.1");
    let s = Session::new().unwrap();
    let mut record = serde_json::Map::new();
    for (name, partitioned) in TABLES {
        let created = &spark[name];
        let t = table(name);
        // Known gap: `INSERT INTO delta.`<path>`` is refused (`TABLE_OR_VIEW_NOT_FOUND`;
        // delta-spark writes it), so the inserts go through a catalog table over the same
        // location (no commit); `DELETE` and `UPDATE` run by path.
        let catalog = format!("rt_{name}");
        sql(
            &s,
            &format!(
                "CREATE TABLE {catalog} USING delta LOCATION '{}'",
                table_dir(name).display()
            ),
        );
        let read = krishiv_read(&s, name);
        assert_eq!(
            sorted(read.clone()),
            sorted(spark_rows(&created["rows"])),
            "{name}: Krishiv reads Spark's table"
        );
        // The ids are the connector step's, so Spark's DML on top (ids 0, 104, 105, 110) finds
        // the same rows. A column list: the table's column order is the log's.
        let first = (100..106).map(Some).chain([None]).collect::<Vec<_>>();
        let mut statements = vec![
            (
                "insert",
                format!(
                    "INSERT INTO {catalog} (id, x, name) VALUES {}",
                    values(&first)
                ),
            ),
            // A row of Spark's first file (on `sql_dv` the file with Spark's deletion vector) and
            // one of Krishiv's.
            ("delete", format!("DELETE FROM {t} WHERE id IN (2, 101)")),
            (
                "update",
                format!("UPDATE {t} SET name = 'k' WHERE id IN (3, 102)"),
            ),
        ];
        if partitioned {
            // A partition-only condition: whole files of Spark's and Krishiv's go.
            statements.push(("delete", format!("DELETE FROM {t} WHERE x = 2")));
        }
        statements.push((
            "insert",
            format!(
                "INSERT INTO {catalog} (id, x, name) VALUES {}",
                values(&[Some(110), Some(111)])
            ),
        ));
        let mut steps = Vec::new();
        for (what, text) in statements {
            let before = latest_version(name);
            let output = sql(&s, &text);
            let version = latest_version(name);
            assert_eq!(version, before + 1, "{name}: {text} commits once");
            steps.push(json!({
                "what": what,
                "sql": text,
                "output": output,
                "version": version,
                "rows": krishiv_read(&s, name),
            }));
        }
        record.insert(name.to_owned(), json!({ "read": read, "steps": steps }));
    }
    std::fs::write(
        out_dir().join("KRISHIV-SQL.json"),
        serde_json::to_vec_pretty(&Value::Object(record)).unwrap(),
    )
    .unwrap();
}

/// A commit metric of Spark's history entry (delta-spark records them as strings).
fn metric(h: &Value, key: &str) -> Option<i64> {
    h["operationMetrics"][key].as_str()?.parse().ok()
}

/// Step 4b: Spark read what Krishiv's SQL committed, and Krishiv reads Spark's DML on top.
#[test]
#[ignore = "step 4b of tests/scripts/delta_roundtrip.sh (needs SPARK-READBACK.json from Spark)"]
fn krishiv_reads_spark_sql_on_top() {
    let created = read_json("SPARK-CREATE.json");
    let krishiv = read_json("KRISHIV-SQL.json");
    let spark = read_json("SPARK-READBACK.json");
    let s = Session::new().unwrap();
    for (name, partitioned) in TABLES {
        let k = &krishiv[name];
        let sp = &spark[name];
        let steps = k["steps"].as_array().unwrap();
        let last = steps.last().unwrap();
        let first = created[name]["version"].as_u64().unwrap();
        let versions: Vec<u64> = steps
            .iter()
            .map(|st| st["version"].as_u64().unwrap())
            .collect();
        assert_eq!(
            versions,
            (first + 1..=first + steps.len() as u64).collect::<Vec<_>>(),
            "{name}: Krishiv's versions follow Spark's"
        );
        assert_eq!(
            steps.len(),
            if partitioned { 5 } else { 4 },
            "{name}: steps"
        );
        assert_eq!(
            sp["before"]["version"], last["version"],
            "{name}: Spark sees Krishiv's last version"
        );
        assert_eq!(
            sp["before"]["protocol"], created[name]["protocol"],
            "{name}: Krishiv's commits leave the protocol alone"
        );
        assert_eq!(
            sorted(spark_rows(&sp["before"]["rows"])),
            sorted(last["rows"].as_array().unwrap().clone()),
            "{name}: Spark reads Krishiv's result"
        );
        let mut previous = k["read"].as_array().unwrap().clone();
        for st in steps {
            let v = st["version"].as_u64().unwrap();
            let key = v.to_string();
            let rows = st["rows"].as_array().unwrap().clone();
            assert_eq!(
                sorted(spark_rows(&sp["at"][&key])),
                sorted(rows.clone()),
                "{name}: Spark at Krishiv's version {v} ({})",
                st["sql"]
            );
            let h = &sp["history"][&key];
            assert_eq!(
                h["readVersion"].as_u64(),
                Some(v - 1),
                "{name}: readVersion of {v}"
            );
            assert_eq!(
                h["isolationLevel"], "Serializable",
                "{name}: isolation of {v}"
            );
            // The rows the statement changed, from Krishiv's own reads before and after it.
            let gone = previous.iter().filter(|r| !rows.contains(r)).count() as i64;
            let new = rows.iter().filter(|r| !previous.contains(r)).count() as i64;
            let affected = st["output"][0][0].as_i64();
            match st["what"].as_str().unwrap() {
                "insert" => {
                    assert_eq!(h["operation"], "WRITE", "{name}: operation of {v}");
                    assert_eq!(h["isBlindAppend"], true, "{name}: isBlindAppend of {v}");
                    assert_eq!(
                        h["operationParameters"]["mode"], "Append",
                        "{name}: mode of {v}"
                    );
                    assert_eq!(
                        metric(h, "numOutputRows"),
                        Some(new),
                        "{name}: rows written at {v}"
                    );
                    // Krishiv's writer gives every added file its statistics.
                    for add in commit_actions(name, v).iter().filter_map(|a| a.get("add")) {
                        let stats: Value =
                            serde_json::from_str(add["stats"].as_str().unwrap()).unwrap();
                        for field in ["minValues", "maxValues", "nullCount"] {
                            assert!(
                                stats[field].get("id").is_some(),
                                "{name}: {field}.id in Krishiv's stats at {v}: {stats}"
                            );
                        }
                    }
                }
                // Known difference: no `isBlindAppend` on a commit that is not a blind append
                // (delta_kernel writes the field only when true); Spark's history shows NULL.
                "delete" => {
                    assert_eq!(h["operation"], "DELETE", "{name}: operation of {v}");
                    assert_eq!(
                        h["isBlindAppend"],
                        Value::Null,
                        "{name}: isBlindAppend of {v}"
                    );
                    assert_eq!(new, 0, "{name}: a DELETE adds no row");
                    assert_eq!(
                        metric(h, "numDeletedRows"),
                        Some(gone),
                        "{name}: deleted at {v}"
                    );
                    if let Some(n) = affected.filter(|n| *n >= 0) {
                        assert_eq!(n, gone, "{name}: Krishiv's DELETE output at {v}");
                    }
                }
                "update" => {
                    assert_eq!(h["operation"], "UPDATE", "{name}: operation of {v}");
                    assert_eq!(
                        h["isBlindAppend"],
                        Value::Null,
                        "{name}: isBlindAppend of {v}"
                    );
                    assert_eq!(
                        metric(h, "numUpdatedRows"),
                        Some(new),
                        "{name}: updated at {v}"
                    );
                    assert_eq!(
                        affected,
                        Some(new),
                        "{name}: Krishiv's UPDATE output at {v}"
                    );
                }
                other => panic!("{other}"),
            }
            assert!(
                gone + new > 0,
                "{name}: {} changed rows",
                st["sql"].as_str().unwrap()
            );
            previous = rows;
        }
        // Spark's DML did work on Krishiv's files.
        let dml = sp["dml"].as_object().unwrap();
        let mut expected = vec![
            ("DELETE", "numDeletedRows"),
            ("UPDATE", "numUpdatedRows"),
            ("MERGE", "numTargetRowsDeleted"),
            ("MERGE", "numTargetRowsInserted"),
        ];
        if name == "sql_dv" {
            expected.push(("DELETE", "numDeletionVectorsAdded"));
        }
        for (op, key) in expected {
            let found: Vec<i64> = dml
                .values()
                .filter(|h| h["operation"] == op)
                .map(|h| metric(h, key).unwrap_or(0))
                .collect();
            assert_eq!(found.len(), 1, "{name}: one {op} commit of Spark's");
            assert!(found[0] > 0, "{name}: Spark's {op} {key}");
        }
        // Krishiv reads Spark's DML on top of its commits.
        assert_eq!(
            sorted(krishiv_read(&s, name)),
            sorted(spark_rows(&sp["after"]["rows"])),
            "{name}: Krishiv reads Spark's result"
        );
    }
}
