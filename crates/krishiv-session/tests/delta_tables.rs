// SPDX-License-Identifier: Apache-2.0
//! Delta tables through the session (feature `delta`, P3-T021 PR-08b part 2), against Spark
//! 4.2.0 + delta-spark 4.4.1 (`tests/fixtures/delta-4.4.1/`): reads by path and by catalog
//! table, `DESCRIBE` of a `USING delta LOCATION` table without columns, the schema and
//! partitioning checks, the write and DDL refusals, `tableNameOrPath` in both forms, and the
//! snapshot cache across queries.
#![cfg(test)]

use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/delta-4.4.1")
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// An oracle table copied to a temporary directory.
struct Table {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
}

impl Table {
    fn copy(name: &str) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        copy_dir(&fixtures().join(name), &dir);
        Self { _tmp: tmp, dir }
    }

    fn path(&self) -> String {
        self.dir.to_string_lossy().into_owned()
    }

    /// Moves the last commit of `session-appends` aside (into the parent directory).
    fn hold_last_commit(&self) {
        for ext in ["json", "crc"] {
            std::fs::rename(self.last(ext), self.dir.with_extension(ext)).unwrap();
        }
    }

    /// Puts it back.
    fn land_last_commit(&self) {
        for ext in ["crc", "json"] {
            std::fs::rename(self.dir.with_extension(ext), self.last(ext)).unwrap();
        }
    }

    fn last(&self, ext: &str) -> PathBuf {
        self.dir
            .join(format!("_delta_log/00000000000000000002.{ext}"))
    }
}

fn session_record(name: &str) -> Value {
    let all: Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("SESSION.json")).unwrap()).unwrap();
    all[name].clone()
}

/// A cell as the oracle's typed JSON writes it (strings, integers, NULL, decimals and UTC
/// timestamps).
fn cell(column: &dyn Array, row: usize) -> Value {
    if column.is_null(row) {
        return Value::Null;
    }
    match column.data_type() {
        ArrowType::Int64 => Value::from(column.as_primitive::<Int64Type>().value(row)),
        ArrowType::Utf8 => Value::from(column.as_string::<i32>().value(row)),
        ArrowType::Utf8View => Value::from(column.as_string_view().value(row)),
        ArrowType::Decimal128(_, _) => {
            let a = column.as_primitive::<Decimal128Type>();
            serde_json::json!({ "decimal": a.value_as_string(row) })
        }
        ArrowType::Timestamp(_, _) => {
            let micros = column.as_primitive::<TimestampMicrosecondType>().value(row);
            let t = chrono_like(micros);
            serde_json::json!({ "timestamp": t })
        }
        other => panic!("no oracle form for {other}"),
    }
}

/// `2024-01-01T00:00:00` for whole seconds (the oracle's naive ISO form in UTC).
fn chrono_like(micros: i64) -> String {
    let secs = micros.div_euclid(1_000_000);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn rows_of(batches: &[RecordBatch]) -> Vec<Vec<Value>> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            out.push(b.columns().iter().map(|c| cell(c.as_ref(), r)).collect());
        }
    }
    out
}

fn query(s: &Session, sql: &str) -> Vec<Vec<Value>> {
    rows_of(&block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap())
}

fn error(s: &Session, sql: &str) -> KrishivError {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql} succeeded"),
        Err(e) => e,
    }
}

fn param(e: &KrishivError, key: &str) -> String {
    e.message_parameters()
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| panic!("{e} has no {key}"))
}

fn ids(rows: &[Vec<Value>]) -> Vec<i64> {
    rows.iter().map(|r| r[0].as_i64().unwrap()).collect()
}

#[test]
fn a_path_reads_as_spark_reads_it() {
    let t = Table::copy("session-table");
    let s = Session::new().unwrap();
    let want = session_record("select")["rows"].clone();
    let by_sql = query(
        &s,
        &format!("SELECT * FROM delta.`{}` ORDER BY id", t.path()),
    );
    assert_eq!(Value::from(by_sql), want);
    let df = s.read().format("delta").load([t.path()]).sort(["id"]);
    let by_reader = rows_of(&block_on(df.collect()).unwrap());
    assert_eq!(Value::from(by_reader), want);
    // Never listed as files: a filter on the partition column reads its rows only.
    let p = query(
        &s,
        &format!("SELECT id FROM delta.`{}` WHERE p = 'y'", t.path()),
    );
    assert_eq!(ids(&p), [2]);
}

#[test]
fn one_query_reads_one_snapshot_and_the_next_query_the_new_one() {
    let t = Table::copy("session-appends");
    t.hold_last_commit();
    let s = Session::new().unwrap();
    let count = |s: &Session| {
        query(
            s,
            &format!(
                "SELECT count(*) FROM delta.`{0}` a JOIN delta.`{0}/` b ON a.id = b.id",
                t.path()
            ),
        )
    };
    assert_eq!(count(&s), [[Value::from(2)]]);
    t.land_last_commit();
    // A new query sees the commit (Delta's `per_query` freshness).
    assert_eq!(count(&s), [[Value::from(3)]]);
}

/// The oracle's `DESCRIBE` rows with a NULL comment as the empty string: Krishiv's `DESCRIBE`
/// prints a missing comment as `""` for every table, Spark as NULL (a known difference of
/// `DESCRIBE`, not of Delta tables).
fn comments_as_empty(rows: &Value) -> Value {
    Value::from(
        rows.as_array()
            .unwrap()
            .iter()
            .map(|r| {
                Value::from(
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|c| {
                            if c.is_null() {
                                Value::from("")
                            } else {
                                c.clone()
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
    )
}

#[test]
fn a_catalog_table_without_columns_takes_the_logs() {
    let t = Table::copy("session-table");
    let s = Session::new().unwrap();
    block_on(s.sql(&format!(
        "CREATE TABLE d USING delta LOCATION '{}'",
        t.path()
    )))
    .unwrap();
    let describe = session_record("describe");
    let got = rows_of(&block_on(block_on(s.sql("DESCRIBE d")).unwrap().collect()).unwrap());
    assert_eq!(Value::from(got), comments_as_empty(&describe["rows"]));
    assert_eq!(
        Value::from(query(&s, "SELECT * FROM d ORDER BY id")),
        session_record("select")["rows"]
    );
    assert_eq!(
        Value::from(query(&s, "SELECT * FROM default.d ORDER BY id")),
        session_record("select")["rows"]
    );
    // Matching columns and partitioning are accepted, as in Spark.
    assert!(session_record("create_matching_columns_partitioned")["error"].is_null());
    block_on(s.sql(&format!(
        "CREATE TABLE same2 (id BIGINT, name STRING, amount DECIMAL(10, 2), ts TIMESTAMP, \
         p STRING) USING delta PARTITIONED BY (p) LOCATION '{}'",
        t.path()
    )))
    .unwrap();
    let got = rows_of(&block_on(block_on(s.sql("DESCRIBE same2")).unwrap().collect()).unwrap());
    assert_eq!(
        Value::from(got),
        comments_as_empty(&session_record("describe_matching_columns")["rows"])
    );
}

/// The class, SQLSTATE and parameters of `e` against the oracle's record, the table path
/// elided as the record elides it.
fn assert_oracle_error(e: &KrishivError, want: &Value, path: &str, skip: &[&str]) {
    assert_eq!(
        e.error_class().name(),
        want["condition"].as_str().unwrap(),
        "{e}"
    );
    assert_eq!(e.sql_state(), want["sqlstate"].as_str(), "{e}");
    for (k, v) in want["parameters"].as_object().unwrap() {
        if skip.contains(&k.as_str()) {
            continue;
        }
        assert_eq!(
            param(e, k).replace(path, "<path>"),
            v.as_str().unwrap(),
            "{k} of {e}"
        );
    }
}

#[test]
fn a_column_list_must_match_the_log() {
    let t = Table::copy("session-table");
    let s = Session::new().unwrap();
    let path = t.path();
    let e = error(
        &s,
        &format!("CREATE TABLE m (id BIGINT, other STRING) USING delta LOCATION '{path}'"),
    );
    let want = &session_record("schema_mismatch")["error"];
    assert_oracle_error(&e, want, &path, &["schemaDifferences"]);
    // delta-spark lists the missing fields in its set's order: the same lines, the same names.
    let lines = |text: &str| -> Vec<(String, Vec<String>)> {
        let mut out: Vec<(String, Vec<String>)> = text
            .lines()
            .map(|l| {
                let (head, names) = l.split_once(": ").unwrap();
                let mut names: Vec<String> = names.split(", ").map(str::to_owned).collect();
                names.sort();
                (head.to_owned(), names)
            })
            .collect();
        out.sort();
        out
    };
    assert_eq!(
        lines(&param(&e, "schemaDifferences")),
        lines(want["parameters"]["schemaDifferences"].as_str().unwrap())
    );
    let e = error(
        &s,
        &format!(
            "CREATE TABLE m2 (id INT, name STRING, amount DECIMAL(10, 2), ts TIMESTAMP, p STRING) \
             USING delta LOCATION '{path}'"
        ),
    );
    assert_oracle_error(
        &e,
        &session_record("schema_mismatch_type")["error"],
        &path,
        &[],
    );
    // The schema matches but the partitioning is not given.
    let e = error(
        &s,
        &format!(
            "CREATE TABLE same (id BIGINT, name STRING, amount DECIMAL(10, 2), ts TIMESTAMP, \
             p STRING) USING delta LOCATION '{path}'"
        ),
    );
    assert_oracle_error(
        &e,
        &session_record("create_matching_columns")["error"],
        &path,
        &[],
    );
    // Nothing was registered.
    assert_eq!(
        error(&s, "SELECT * FROM m").error_class(),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn writes_and_ddl_are_refused_by_name() {
    let t = Table::copy("session-table");
    let s = Session::new().unwrap();
    let path = t.path();
    block_on(s.sql(&format!("CREATE TABLE d USING delta LOCATION '{path}'"))).unwrap();
    let missing = tempfile::tempdir().unwrap();
    let gone = missing.path().join("gone").to_string_lossy().into_owned();
    let empty = missing.path().to_string_lossy().into_owned();
    for (probe, sql) in [
        (
            "insert",
            "INSERT INTO d VALUES (3, 'c', 3.00, NULL, 'x')".to_owned(),
        ),
        (
            "alter_add_columns",
            "ALTER TABLE d ADD COLUMNS (extra INT)".to_owned(),
        ),
        (
            "alter_set_properties",
            "ALTER TABLE d SET TBLPROPERTIES ('k' = 'v')".to_owned(),
        ),
        (
            "ctas",
            "CREATE TABLE c USING delta AS SELECT 1 AS id".to_owned(),
        ),
        (
            "create_without_location",
            "CREATE TABLE managed (id BIGINT) USING delta".to_owned(),
        ),
        (
            "create_missing_location",
            format!("CREATE TABLE gone USING delta LOCATION '{gone}'"),
        ),
        (
            "create_missing_location_with_columns",
            format!("CREATE TABLE gone2 (id BIGINT) USING delta LOCATION '{gone}'"),
        ),
        (
            "create_empty_location",
            format!("CREATE TABLE hollow USING delta LOCATION '{empty}'"),
        ),
    ] {
        // Spark + delta-spark runs each of them: a write Krishiv does not make yet.
        assert!(session_record(probe)["error"].is_null(), "{probe}");
        let e = error(&s, &sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation,
            "{probe}: {e}"
        );
        assert!(param(&e, "operation").contains("Delta"), "{probe}: {e}");
    }
    // The table still reads, unchanged; nothing else was created.
    assert_eq!(ids(&query(&s, "SELECT id FROM d ORDER BY id")), [1, 2]);
    for name in ["c", "managed", "gone", "gone2", "hollow"] {
        assert_eq!(
            error(&s, &format!("SELECT * FROM {name}")).error_class(),
            ErrorClass::TableOrViewNotFound,
            "{name}"
        );
    }
    assert!(!Path::new(&gone).exists());
}

/// delta-spark compares `TBLPROPERTIES` (and options) given to `CREATE TABLE .. USING delta
/// LOCATION` with the log's configuration; until a probe pins that rule, Krishiv refuses them by
/// name rather than storing them unchecked. A `path` option is the location, not a property.
#[test]
fn table_properties_on_a_delta_create_are_refused_by_name() {
    let t = Table::copy("session-table");
    let s = Session::new().unwrap();
    let path = t.path();
    for sql in [
        format!("CREATE TABLE p USING delta LOCATION '{path}' TBLPROPERTIES ('k' = 'v')"),
        format!("CREATE TABLE p USING delta OPTIONS (mergeSchema 'true') LOCATION '{path}'"),
    ] {
        let e = error(&s, &sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation,
            "{sql}: {e}"
        );
        assert!(param(&e, "operation").contains("Delta"), "{sql}: {e}");
    }
    assert_eq!(
        error(&s, "SELECT * FROM p").error_class(),
        ErrorClass::TableOrViewNotFound
    );
    block_on(s.sql(&format!(
        "CREATE TABLE p USING delta OPTIONS (path '{path}')"
    )))
    .unwrap();
    assert_eq!(ids(&query(&s, "SELECT id FROM p ORDER BY id")), [1, 2]);
}

#[test]
fn a_refused_table_is_named_as_delta_spark_names_it() {
    let t = Table::copy("session-appends");
    t.hold_last_commit();
    let s = Session::new().unwrap();
    let path = t.path();
    block_on(s.sql(&format!(
        "CREATE TABLE refused USING delta LOCATION '{path}'"
    )))
    .unwrap();
    // A new commit adds the unknown feature, as the probe's did.
    std::fs::write(
        t.last("json"),
        r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["krishivProbeFeature"],"writerFeatures":["krishivProbeFeature"]}}
"#,
    )
    .unwrap();
    // Krishiv names its own version where delta-spark names its.
    let skip = ["deltaVersion"];
    for (probe, sql) in [
        ("refused_catalog_table", "SELECT * FROM refused".to_owned()),
        (
            "refused_qualified_catalog_table",
            "SELECT * FROM default.refused".to_owned(),
        ),
        ("refused_path_sql", format!("SELECT * FROM delta.`{path}`")),
        (
            "refused_path_trailing_slash",
            format!("SELECT * FROM delta.`{path}/`"),
        ),
    ] {
        let e = error(&s, &sql);
        assert_oracle_error(&e, &session_record(probe)["error"], &path, &skip);
    }
    let e = block_on(s.read().format("delta").load([&path]).collect()).unwrap_err();
    assert_oracle_error(
        &e,
        &session_record("refused_path_reader")["error"],
        &path,
        &skip,
    );
}

#[test]
fn a_temporary_view_over_a_delta_table() {
    let t = Table::copy("session-appends");
    t.hold_last_commit();
    let s = Session::new().unwrap();
    block_on(s.sql(&format!(
        "CREATE TEMPORARY VIEW v USING delta OPTIONS (path '{}')",
        t.path()
    )))
    .unwrap();
    assert_eq!(ids(&query(&s, "SELECT id FROM v ORDER BY id")), [1, 2]);
    t.land_last_commit();
    assert_eq!(ids(&query(&s, "SELECT id FROM v ORDER BY id")), [1, 2, 3]);
    let e = error(
        &s,
        &format!(
            "CREATE TEMPORARY VIEW w (id BIGINT) USING delta OPTIONS (path '{}')",
            t.path()
        ),
    );
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn a_dataframe_temp_view_and_the_metrics() {
    let t = Table::copy("session-appends");
    let s = Session::new().unwrap();
    let df = s.read().format("delta").load([t.path()]);
    block_on(df.create_or_replace_temp_view("dv")).unwrap();
    assert_eq!(ids(&query(&s, "SELECT id FROM dv ORDER BY id")), [1, 2, 3]);
    let names: Vec<String> = s
        .metrics()
        .gauges
        .iter()
        .map(|m| m.name.to_owned())
        .collect();
    for name in [
        "krishiv.delta.log_pool.wait_ms",
        "krishiv.delta.snapshot_cache.hits",
        "krishiv.delta.snapshot_cache.misses",
    ] {
        assert!(names.iter().any(|n| n == name), "{name} in {names:?}");
    }
}
