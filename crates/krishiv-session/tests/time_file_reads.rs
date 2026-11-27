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

fn typeof_t(s: &Session, df: &krishiv_session::DataFrame) -> String {
    block_on(df.create_or_replace_temp_view("i")).unwrap();
    let out = block_on(s.sql("SELECT typeof(t) FROM i LIMIT 1")).unwrap();
    texts(&block_on(out.collect()).unwrap())[0][0]
        .clone()
        .unwrap()
}

/// Schema inference gives `TIME(6)` only with the gate on (Spark 4.3.0: CSV
/// under `preferDate`, JSON under `inferTimestamp`); with it off the same
/// texts are timestamps, as Spark infers them.
#[test]
fn inference_gives_time_only_with_the_gate_on() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("a.csv");
    std::fs::write(&csv, "id,t\n0,12:34:56\n1,01:02:03.5\n").unwrap();
    let json = dir.path().join("a.json");
    std::fs::write(
        &json,
        "{\"id\":0,\"t\":\"12:34:56\"}\n{\"id\":1,\"t\":\"01:02:03.5\"}\n",
    )
    .unwrap();
    for (gate, csv_type, json_type) in [
        (true, "time(6)", "time(6)"),
        (false, "timestamp", "timestamp"),
    ] {
        let s = session(gate);
        let df = s
            .read()
            .format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load([csv.to_str().unwrap()]);
        assert_eq!(typeof_t(&s, &df), csv_type, "csv, gate {gate}");
        let df = s
            .read()
            .format("json")
            .option("inferTimestamp", "true")
            .load([json.to_str().unwrap()]);
        assert_eq!(typeof_t(&s, &df), json_type, "json, gate {gate}");
    }
}

/// Inference and a `TIME(6)` read under each profile, against a real run of
/// its Spark (`tests/scripts/probe_time_text_profiles.py`): Spark 4.2.1 never
/// infers `TIME`, gate on or off (`krishiv.io.text.inferTime`), and cuts the
/// default grammar to microseconds while a `timeFormat` keeps every digit
/// (`krishiv.io.text.timeNanos`); 4.3.0 infers `time(6)` with the gate on and
/// keeps nine digits.
#[test]
fn text_time_follows_the_profile_runs() {
    use krishiv_plan::analyzer::SparkProfile;
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("a.csv");
    std::fs::write(&csv, "id,t\n0,12:34:56\n1,01:02:03.5\n").unwrap();
    let json = dir.path().join("a.json");
    std::fs::write(&json, "{\"id\":0,\"t\":\"12:34:56\"}\n").unwrap();
    let nine = dir.path().join("nine.csv");
    std::fs::write(&nine, "t\n12:34:56.123456789\n").unwrap();
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/time-text-profiles.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(run["spark"], version);
        let s = Session::builder().profile(profile).build().unwrap();
        for gate in ["false", "true"] {
            let want = &run["infer"][gate];
            block_on(s.sql(&format!("SET spark.sql.timeType.enabled = {gate}"))).unwrap();
            let df = s
                .read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load([csv.to_str().unwrap()]);
            assert_eq!(typeof_t(&s, &df), want["csv"], "{version} csv, gate {gate}");
            let df = s
                .read()
                .format("json")
                .option("inferTimestamp", "true")
                .load([json.to_str().unwrap()]);
            assert_eq!(
                typeof_t(&s, &df),
                want["json"],
                "{version} json, gate {gate}"
            );
            assert_eq!(
                scalar(&s, "SELECT schema_of_csv('14:30:45')").as_deref(),
                want["schema_of_csv"].as_str(),
                "{version} schema_of_csv, gate {gate}"
            );
        }
        for (name, pattern) in [
            ("default", None),
            ("HH:mm:ss.SSSSSSSSS", Some("HH:mm:ss.SSSSSSSSS")),
        ] {
            let mut reader = s.read().format("csv").option("header", "true");
            if let Some(p) = pattern {
                reader = reader.option("timeFormat", p);
            }
            let df = reader
                .schema_ddl("t TIME(6)")
                .unwrap()
                .load([nine.to_str().unwrap()]);
            block_on(df.create_or_replace_temp_view("n")).unwrap();
            let out = block_on(s.sql("SELECT CAST(t AS STRING) FROM n")).unwrap();
            let rows = texts(&block_on(out.collect()).unwrap());
            assert_eq!(
                rows[0][0].as_deref(),
                run["read"][name].as_str(),
                "{version} read, {name}"
            );
        }
    }
}

/// With the gate off a JSON read and a CSV write of a `TIME` are refused too.
#[test]
fn the_gate_off_refuses_json_reads_and_writes() {
    let dir = tempfile::tempdir().unwrap();
    let json = dir.path().join("a.json");
    std::fs::write(&json, "{\"id\":1,\"t\":\"12:34:56\"}\n").unwrap();
    let s = session(false);
    let df = s
        .read()
        .format("json")
        .schema_ddl("id INT, t TIME(3)")
        .unwrap()
        .load([json.to_str().unwrap()]);
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnsupportedTimeType, "{e}");
    let df = block_on(s.sql("SELECT TIME'01:02:03' AS t")).unwrap();
    let out = dir.path().join("out");
    let e = block_on(df.write().csv(out.to_str().unwrap())).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnsupportedTimeType, "{e}");
}

/// A `TIME(3)` column keeps the digits it read past its precision.
#[test]
fn a_read_value_keeps_digits_past_its_precision() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("a.csv");
    std::fs::write(&csv, "id,t\n1,12:34:56.123456\n").unwrap();
    let s = session(true);
    let df = s
        .read()
        .format("csv")
        .option("header", "true")
        .schema_ddl("id INT, t TIME(3)")
        .unwrap()
        .load([csv.to_str().unwrap()]);
    block_on(df.create_or_replace_temp_view("p")).unwrap();
    let out = block_on(s.sql(
        "SELECT CAST(t = TIME'12:34:56.123' AS STRING), CAST(t = TIME'12:34:56.123456' AS STRING), \
         CAST(hour(t) AS STRING), typeof(t) FROM p",
    ))
    .unwrap();
    assert_eq!(
        texts(&block_on(out.collect()).unwrap()),
        vec![vec![
            Some("false".to_owned()),
            Some("true".to_owned()),
            Some("12".to_owned()),
            Some("time(3)".to_owned()),
        ]]
    );
}

fn scalar(s: &Session, sql: &str) -> Option<String> {
    let out = block_on(s.sql(sql)).unwrap();
    texts(&block_on(out.collect()).unwrap())[0][0].clone()
}

/// A read or write option named `spark.sql.timeType.enabled` does not open
/// the gate: Spark 4.3.0 reads only `SQLConf` (`CSVInferSchema`,
/// `JsonInferSchema`, the data source check), so with the session gate off
/// the text is still inferred as a timestamp.
#[test]
fn a_per_read_gate_option_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("a.csv");
    std::fs::write(&csv, "id,t\n0,12:34:56\n").unwrap();
    let json = dir.path().join("a.json");
    std::fs::write(&json, "{\"id\":0,\"t\":\"12:34:56\"}\n").unwrap();
    let s = session(false);
    let df = s
        .read()
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("spark.sql.timeType.enabled", "true")
        .load([csv.to_str().unwrap()]);
    assert_eq!(typeof_t(&s, &df), "timestamp", "csv");
    let df = s
        .read()
        .format("json")
        .option("inferTimestamp", "true")
        .option("SPARK.SQL.TIMETYPE.ENABLED", "true")
        .load([json.to_str().unwrap()]);
    assert_eq!(typeof_t(&s, &df), "timestamp", "json");
    // Nor does it close the gate the session opened.
    let s = session(true);
    let df = s
        .read()
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("spark.sql.timeType.enabled", "false")
        .load([csv.to_str().unwrap()]);
    assert_eq!(typeof_t(&s, &df), "time(6)", "csv, session gate on");
}

/// `schema_of_csv` infers `TIME(6)` under the session gate (Spark 4.3.0:
/// `STRUCT<_c0: TIME(6)>`) and a timestamp without it.
#[test]
fn schema_of_csv_follows_the_session_gate() {
    let on = session(true);
    assert_eq!(
        scalar(&on, "SELECT schema_of_csv('12:34:56')").as_deref(),
        Some("STRUCT<_c0: TIME(6)>")
    );
    assert_eq!(
        scalar(&on, "SELECT schema_of_csv('12:34:56', map('sep', ','))").as_deref(),
        Some("STRUCT<_c0: TIME(6)>")
    );
    let off = session(false);
    assert_eq!(
        scalar(
            &off,
            "SELECT schema_of_csv('12:34:56', map('spark.sql.timeType.enabled', 'true'))"
        )
        .as_deref(),
        Some("STRUCT<_c0: TIMESTAMP>")
    );
}

/// A `timeFormat` with an offset or a zone reads the clock fields and ignores
/// the zone (Spark 4.3.0's `toLocalTime`); an unknown region does not parse.
/// Writing, `timeFormat=HH:mm:ss` keeps the fraction (Spark's fraction
/// formatter for the default pattern). Spark 4.3.0 run values.
#[test]
fn time_format_zones_and_the_default_write_pattern() {
    let s = session(true);
    for (sql, want) in [
        (
            "SELECT CAST(try_to_time('12:34:56+02:00', 'HH:mm:ssXXX') AS STRING)",
            Some("12:34:56"),
        ),
        (
            "SELECT CAST(try_to_time('12:34:56 Europe/Paris', 'HH:mm:ss VV') AS STRING)",
            Some("12:34:56"),
        ),
        (
            "SELECT CAST(try_to_time('12:34:56 Foo/Bar', 'HH:mm:ss VV') AS STRING)",
            None,
        ),
        (
            "SELECT CAST(from_csv('12:34:56+02:00', 't TIME(6)', map('timeFormat', 'HH:mm:ssXXX')) AS STRING)",
            Some("{12:34:56}"),
        ),
        (
            "SELECT CAST(from_json('{\"t\":\"12:34:56+02:00\"}', 't TIME(6)', map('timeFormat', 'HH:mm:ssXXX')) AS STRING)",
            Some("{12:34:56}"),
        ),
        (
            "SELECT to_csv(named_struct('t', TIME'12:34:56.123456'), map('timeFormat', 'HH:mm:ss'))",
            Some("12:34:56.123456"),
        ),
        (
            "SELECT to_json(named_struct('t', TIME'12:34:56.123456'), map('timeFormat', 'HH:mm:ss'))",
            Some("{\"t\":\"12:34:56.123456\"}"),
        ),
    ] {
        assert_eq!(scalar(&s, sql).as_deref(), want, "{sql}");
    }
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("z.csv");
    std::fs::write(&csv, "id,t\n1,12:34:56+02:00\n").unwrap();
    let df = s
        .read()
        .format("csv")
        .option("header", "true")
        .option("timeFormat", "HH:mm:ssXXX")
        .schema_ddl("id INT, t TIME(6)")
        .unwrap()
        .load([csv.to_str().unwrap()]);
    block_on(df.create_or_replace_temp_view("z")).unwrap();
    assert_eq!(
        scalar(&s, "SELECT CAST(t AS STRING) FROM z").as_deref(),
        Some("12:34:56")
    );
    let out = dir.path().join("out");
    let df = block_on(s.sql("SELECT TIME'12:34:56.123456' AS t")).unwrap();
    block_on(
        df.write()
            .option("timeFormat", "HH:mm:ss")
            .json(out.to_str().unwrap()),
    )
    .unwrap();
    let mut written = String::new();
    for f in std::fs::read_dir(&out).unwrap() {
        let p = f.unwrap().path();
        if p.extension().is_some_and(|e| e == "json") {
            written.push_str(&std::fs::read_to_string(p).unwrap());
        }
    }
    assert_eq!(written, "{\"t\":\"12:34:56.123456\"}\n");
}
