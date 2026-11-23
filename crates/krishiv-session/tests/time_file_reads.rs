// SPDX-License-Identifier: Apache-2.0
//! `TIME` columns read and written through the CSV and JSON data sources, end
//! to end (Spark 4.3.0 run: `tests/fixtures/spark-4.3.0/time-file-reads.json`;
//! the readers' own cases are in `krishiv-io` `csv::time_tests` and
//! `json::time_tests`): `spark.sql.timeType.enabled` off refuses the read, on
//! reads each text as `CAST(text AS TIME)` whatever the declared precision.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn session(gate: bool) -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql(&format!("SET spark.sql.timeType.enabled = {gate}"))).unwrap();
    s
}

/// Every row's columns as text, in row order.
fn texts(batches: &[RecordBatch]) -> Vec<Vec<Option<String>>> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| (!c.is_null(i)).then(|| c.as_string_view().value(i).to_owned()))
                    .collect(),
            );
        }
    }
    out
}

fn row(id: &str, t: Option<&str>) -> Vec<Option<String>> {
    vec![Some(id.to_owned()), t.map(str::to_owned)]
}

fn read(s: &Session, format: &str, path: &str, ddl: &str) -> Vec<Vec<Option<String>>> {
    let df = s
        .read()
        .format(format)
        .option("header", "true")
        .schema_ddl(ddl)
        .unwrap()
        .load([path]);
    block_on(df.create_or_replace_temp_view("v")).unwrap();
    let df =
        block_on(s.sql("SELECT CAST(id AS STRING), CAST(t AS STRING) FROM v ORDER BY id")).unwrap();
    texts(&block_on(df.collect()).unwrap())
}

#[test]
fn csv_and_json_read_time_columns_as_a_cast() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("a.csv");
    std::fs::write(
        &csv,
        "id,t\n1,12:34:56\n2,1:2:3\n3,12:34:56.5 PM\n4,24:00:00\n5,\n6,T01:02:03.123\n",
    )
    .unwrap();
    let json = dir.path().join("a.json");
    std::fs::write(
        &json,
        "{\"id\":1,\"t\":\"12:34:56\"}\n{\"id\":2,\"t\":\"1:2:3\"}\n{\"id\":3,\"t\":45296}\n\
         {\"id\":4,\"t\":\"24:00:00\"}\n{\"id\":5,\"t\":null}\n{\"id\":6,\"t\":\"T01:02:03.123\"}\n",
    )
    .unwrap();
    let s = session(true);
    // TIME(0) keeps the fraction it read, as Spark does.
    assert_eq!(
        read(&s, "csv", csv.to_str().unwrap(), "id INT, t TIME(0)"),
        vec![
            row("1", Some("12:34:56")),
            row("2", Some("01:02:03")),
            row("3", Some("12:34:56.5")),
            row("4", None),
            row("5", None),
            row("6", Some("01:02:03.123")),
        ]
    );
    assert_eq!(
        read(&s, "json", json.to_str().unwrap(), "id INT, t TIME(3)"),
        vec![
            row("1", Some("12:34:56")),
            row("2", Some("01:02:03")),
            row("3", None),
            row("4", None),
            row("5", None),
            row("6", Some("01:02:03.123")),
        ]
    );
}

#[test]
fn the_gate_off_refuses_a_time_column() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("a.csv");
    std::fs::write(&csv, "id,t\n1,12:34:56\n").unwrap();
    let s = session(false);
    let df = s
        .read()
        .format("csv")
        .option("header", "true")
        .schema_ddl("id INT, t TIME")
        .unwrap()
        .load([csv.to_str().unwrap()]);
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnsupportedTimeType, "{e}");
}

#[test]
fn a_written_time_column_reads_back() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let s = session(true);
    let df = block_on(s.sql(
        "SELECT 1 AS id, CAST(TIME'12:34:56.120' AS TIME(3)) AS t UNION ALL SELECT 2, CAST(NULL AS TIME(3))",
    ))
    .unwrap();
    block_on(
        df.write()
            .option("header", "true")
            .csv(out.to_str().unwrap()),
    )
    .unwrap();
    let mut written = String::new();
    for f in std::fs::read_dir(&out).unwrap() {
        let p = f.unwrap().path();
        if p.extension().is_some_and(|e| e == "csv") {
            written.push_str(&std::fs::read_to_string(p).unwrap());
        }
    }
    // The value's own digits, as Spark writes them; a NULL is an empty field.
    assert!(written.contains("1,12:34:56.12\n"), "{written}");
    assert!(written.contains("2,\n"), "{written}");
    assert_eq!(
        read(&s, "csv", out.to_str().unwrap(), "id INT, t TIME(3)"),
        vec![row("1", Some("12:34:56.12")), row("2", None)]
    );
}
