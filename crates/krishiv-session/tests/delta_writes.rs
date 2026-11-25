// SPDX-License-Identifier: Apache-2.0
//! Delta writes through the session (feature `delta`, P3-T021 PR-37a part 2) against Spark
//! 4.2.0 + delta-spark 4.4.1 (`tests/fixtures/delta-4.4.1/WRITES.json`, `gen_delta_fixtures.py
//! /out writes`): every write mode as SQL and as DataFrame writes, replayed on a Krishiv-created
//! table and compared with the oracle's rows and commits (`write_modes_match_oracle`), v1
//! `saveAsTable` overwrite (`delta_v1_save_as_table_overwrite`), the replace refusals
//! (`delta_replace_is_refused`), the change-data-feed overwrite refusal, and a round trip.
#![cfg(test)]

use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
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

fn oracle() -> Value {
    serde_json::from_slice(&std::fs::read(fixtures().join("WRITES.json")).unwrap()).unwrap()
}

fn cell(column: &dyn Array, row: usize) -> Value {
    if column.is_null(row) {
        return Value::Null;
    }
    match column.data_type() {
        ArrowType::Int64 => Value::from(column.as_primitive::<Int64Type>().value(row)),
        ArrowType::Int32 => Value::from(
            column
                .as_primitive::<arrow_array::types::Int32Type>()
                .value(row),
        ),
        ArrowType::Utf8 => Value::from(column.as_string::<i32>().value(row)),
        ArrowType::Utf8View => Value::from(column.as_string_view().value(row)),
        other => panic!("no oracle form for {other}"),
    }
}

fn rows_of(batches: &[RecordBatch]) -> Value {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            out.push(Value::Array(
                b.columns().iter().map(|c| cell(c.as_ref(), r)).collect(),
            ));
        }
    }
    Value::Array(out)
}

fn query(s: &Session, sql: &str) -> Value {
    rows_of(&block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap())
}

fn run(s: &Session, sql: &str) -> Result<(), KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await }).map(drop)
}

/// The JSON actions of the latest commit under `dir`.
fn last_commit(dir: &Path) -> Vec<Value> {
    let log = dir.join("_delta_log");
    let last = std::fs::read_dir(&log)
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".json")?.parse::<u64>().ok()
        })
        .max()
        .unwrap();
    std::fs::read_to_string(log.join(format!("{last:020}.json")))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn versions(dir: &Path) -> usize {
    std::fs::read_dir(dir.join("_delta_log"))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".json")
        })
        .count()
}

fn commit_info(actions: &[Value]) -> Value {
    actions
        .iter()
        .find_map(|a| a.get("commitInfo").cloned())
        .unwrap()
}

fn count(actions: &[Value], kind: &str) -> usize {
    actions.iter().filter(|a| a.get(kind).is_some()).count()
}

/// The commit's operation, parameters and counts against the oracle's (bytes and the
/// kernel-owned fields differ: recorded differences in the connector's `write` module).
fn compare(step: &Value, dir: &Path) {
    let name = step["step"].as_str().unwrap();
    let ours = last_commit(dir);
    let theirs = &step["commits"][0];
    let (oi, ti) = (commit_info(&ours), &theirs["commitInfo"]);
    assert_eq!(oi["operation"], ti["operation"], "{name}");
    let mut want = ti["operationParameters"].clone();
    if let Some(d) = want.get_mut("isDynamicPartitionOverwrite") {
        *d = Value::from("true");
    }
    assert_eq!(oi["operationParameters"], want, "{name}");
    for k in ["numFiles", "numOutputRows", "numRemovedFiles"] {
        assert_eq!(
            oi["operationMetrics"].get(k),
            ti["operationMetrics"].get(k),
            "{name} {k}"
        );
    }
    assert_eq!(
        oi["isBlindAppend"].as_bool().unwrap_or(false),
        ti["isBlindAppend"].as_bool().unwrap_or(false),
        "{name}"
    );
    assert_eq!(oi["readVersion"], ti["readVersion"], "{name}");
    assert_eq!(
        count(&ours, "add"),
        theirs["adds"].as_array().unwrap().len(),
        "{name}"
    );
    assert_eq!(
        count(&ours, "remove"),
        theirs["removes"].as_array().unwrap().len(),
        "{name}"
    );
}

#[test]
fn write_modes_match_oracle() {
    let oracle = oracle();
    let steps = oracle["steps"].as_array().unwrap();
    let step = |n: &str| steps.iter().find(|s| s["step"] == n).unwrap().clone();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let rows = |name: &str| {
        assert_eq!(
            query(&s, "SELECT id, v, p FROM wm ORDER BY id"),
            step(name)["rows"],
            "{name}"
        );
    };
    let df = |sql: &str| block_on(s.sql(sql)).unwrap();

    run(
        &s,
        &format!(
            "CREATE TABLE wm USING delta PARTITIONED BY (p) LOCATION '{path}' AS \
             SELECT CAST(id AS BIGINT) AS id, v, p FROM VALUES (1, 'a', 'x'), (2, 'b', 'y') AS s(id, v, p)"
        ),
    )
    .unwrap();
    rows("ctas");
    assert_eq!(versions(&dir), 1);

    run(&s, "INSERT INTO wm VALUES (3, 'c', 'x'), (4, 'd', NULL)").unwrap();
    rows("append");
    compare(&step("append"), &dir);

    run(
        &s,
        "INSERT OVERWRITE wm VALUES (5, 'e', 'x'), (6, 'f', 'y')",
    )
    .unwrap();
    rows("overwrite");
    compare(&step("overwrite"), &dir);

    run(&s, "INSERT INTO wm VALUES (7, 'g', 'y'), (8, 'h', 'z')").unwrap();
    rows("append_again");
    compare(&step("append_again"), &dir);

    run(&s, "SET spark.sql.sources.partitionOverwriteMode = dynamic").unwrap();
    run(
        &s,
        "INSERT OVERWRITE wm VALUES (20, 'dyn', 'x'), (21, 'dyn', 'z')",
    )
    .unwrap();
    run(&s, "RESET spark.sql.sources.partitionOverwriteMode").unwrap();
    rows("dynamic_overwrite");
    compare(&step("dynamic_overwrite"), &dir);

    block_on(
        df("SELECT CAST(30 AS BIGINT) AS id, 'rw' AS v, 'y' AS p")
            .write()
            .format("delta")
            .mode("overwrite")
            .option("replaceWhere", "p = 'y'")
            .save(&path),
    )
    .unwrap();
    rows("replace_where");
    compare(&step("replace_where"), &dir);

    let before = versions(&dir);
    let e = block_on(
        df("SELECT CAST(31 AS BIGINT) AS id, 'rw' AS v, 'x' AS p")
            .write()
            .format("delta")
            .mode("overwrite")
            .option("replaceWhere", "p = 'y'")
            .save(&path),
    )
    .unwrap_err();
    let theirs = step("replace_where_mismatch")["error"].clone();
    assert_eq!(
        e.error_class().name(),
        theirs["condition"].as_str().unwrap()
    );
    assert_eq!(versions(&dir), before);
    rows("replace_where_mismatch");

    run(
        &s,
        "INSERT OVERWRITE wm PARTITION (p = 'z') VALUES (40, 'st')",
    )
    .unwrap();
    rows("static_partition_overwrite");
    compare(&step("static_partition_overwrite"), &dir);

    block_on(
        df("SELECT CAST(50 AS BIGINT) AS id, 'df' AS v, 'x' AS p")
            .write()
            .format("delta")
            .mode("overwrite")
            .save(&path),
    )
    .unwrap();
    rows("insert_overwrite_dataframe_mode");
    compare(&step("insert_overwrite_dataframe_mode"), &dir);

    block_on(
        df("SELECT CAST(51 AS BIGINT) AS id, 'df' AS v, 'w' AS p")
            .write()
            .format("delta")
            .mode("append")
            .save(&path),
    )
    .unwrap();
    rows("append_dataframe");
    compare(&step("append_dataframe"), &dir);

    let e = block_on(
        df("SELECT CAST(52 AS BIGINT) AS id, 'df' AS v, 'w' AS p")
            .write()
            .format("delta")
            .save(&path),
    )
    .unwrap_err();
    let theirs = step("save_errorifexists")["error"].clone();
    assert_eq!(
        e.error_class().name(),
        theirs["condition"].as_str().unwrap()
    );
    assert_eq!(
        e.to_string(),
        format!(
            "[DELTA_PATH_EXISTS] {} SQLSTATE: 42K04",
            theirs["message"]
                .as_str()
                .unwrap()
                .trim_start_matches("[DELTA_PATH_EXISTS] ")
                .replace("<path>", &path)
        )
    );
    rows("save_errorifexists");

    let before = versions(&dir);
    block_on(
        df("SELECT CAST(53 AS BIGINT) AS id, 'df' AS v, 'w' AS p")
            .write()
            .format("delta")
            .mode("ignore")
            .save(&path),
    )
    .unwrap();
    assert_eq!(versions(&dir), before, "ignore writes nothing");
    rows("save_ignore");

    run(&s, "INSERT INTO wm SELECT id + 100, v, p FROM wm").unwrap();
    rows("insert_reading_itself");
    compare(&step("insert_reading_itself"), &dir);
}

#[test]
fn delta_save_as_table_append_infers_partitioning_and_overwrite_refuses_bucketing() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("t").to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let df = |sql: &str| block_on(s.sql(sql)).unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE inf USING delta PARTITIONED BY (p) LOCATION '{path}' AS \
             SELECT CAST(1 AS BIGINT) AS id, 'a' AS v, 'x' AS p"
        ),
    )
    .unwrap();
    // An append that leaves partitioning unspecified takes the table's own.
    block_on(
        df("SELECT CAST(2 AS BIGINT) AS id, 'b' AS v, 'y' AS p")
            .write()
            .format("delta")
            .mode("append")
            .save_as_table("inf"),
    )
    .unwrap();
    assert_eq!(block_on(s.table("inf").count()).unwrap(), 2);
    // An overwrite that asks for bucketing is refused and leaves the rows.
    let r = block_on(
        df("SELECT CAST(3 AS BIGINT) AS id, 'c' AS v, 'z' AS p")
            .write()
            .format("delta")
            .bucket_by(4, ["id"])
            .mode("overwrite")
            .save_as_table("inf"),
    );
    assert!(r.is_err());
    assert_eq!(block_on(s.table("inf").count()).unwrap(), 2);
}

#[test]
fn delta_v1_save_as_table_overwrite() {
    let oracle = oracle();
    let steps = oracle["save_as_table"].as_array().unwrap();
    let step = |n: &str| steps.iter().find(|s| s["step"] == n).unwrap().clone();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let df = |sql: &str| block_on(s.sql(sql)).unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE sat USING delta PARTITIONED BY (p) LOCATION '{path}' AS \
             SELECT CAST(1 AS BIGINT) AS id, 'a' AS v, 'x' AS p"
        ),
    )
    .unwrap();
    block_on(
        df("SELECT CAST(2 AS BIGINT) AS id, 'b' AS v, 'y' AS p")
            .write()
            .format("delta")
            .mode("overwrite")
            .partition_by(["p"])
            .save_as_table("sat"),
    )
    .unwrap();
    let same = step("save_as_table_overwrite_same_schema");
    assert_eq!(
        query(&s, "SELECT id, v, p FROM sat ORDER BY id"),
        same["rows"]
    );
    let info = commit_info(&last_commit(&dir));
    // A recorded difference: delta-spark names it `CREATE OR REPLACE TABLE AS SELECT`.
    assert_eq!(info["operation"], "WRITE");
    assert_eq!(count(&last_commit(&dir), "remove"), 1);
    assert_eq!(
        count(&last_commit(&dir), "metaData"),
        0,
        "no metadata change"
    );

    let e = block_on(
        df("SELECT CAST(3 AS BIGINT) AS id, 3 AS other, 'y' AS p")
            .write()
            .format("delta")
            .mode("overwrite")
            .partition_by(["p"])
            .save_as_table("sat"),
    )
    .unwrap_err();
    let theirs = &step("save_as_table_overwrite_other_schema")["error"];
    assert_eq!(
        e.error_class().name(),
        theirs["condition"].as_str().unwrap(),
        "{e}"
    );
    assert_eq!(e.sql_state(), theirs["sqlstate"].as_str());
    assert_eq!(
        e.to_string(),
        format!(
            "{} SQLSTATE: {}",
            theirs["message"].as_str().unwrap(),
            theirs["sqlstate"].as_str().unwrap()
        ),
        "the oracle's text"
    );

    // No partitionBy: the table keeps its partitioning, as delta-spark keeps it.
    block_on(
        df("SELECT CAST(4 AS BIGINT) AS id, 'd' AS v, 'y' AS p")
            .write()
            .format("delta")
            .mode("overwrite")
            .save_as_table("sat"),
    )
    .unwrap();
    let other = step("save_as_table_overwrite_other_partitioning");
    assert_eq!(
        query(&s, "SELECT id, v, p FROM sat ORDER BY id"),
        other["rows"]
    );

    block_on(
        df("SELECT CAST(5 AS BIGINT) AS id, 'e' AS v, 'x' AS p")
            .write()
            .format("delta")
            .mode("append")
            .save_as_table("sat"),
    )
    .unwrap();
    let append = step("save_as_table_append");
    assert_eq!(
        query(&s, "SELECT id, v, p FROM sat ORDER BY id"),
        append["rows"]
    );
    compare(&append, &dir);
}

#[test]
fn delta_replace_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!("CREATE TABLE r USING delta LOCATION '{path}' AS SELECT CAST(1 AS BIGINT) AS id"),
    )
    .unwrap();
    for sql in [
        "REPLACE TABLE r USING delta AS SELECT CAST(2 AS BIGINT) AS id".to_owned(),
        "CREATE OR REPLACE TABLE r USING delta AS SELECT CAST(2 AS BIGINT) AS id".to_owned(),
        "REPLACE TABLE r (id BIGINT) USING delta".to_owned(),
    ] {
        let e = run(&s, &sql).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation,
            "{sql}: {e}"
        );
    }
    let e = block_on(
        block_on(s.sql("SELECT CAST(2 AS BIGINT) AS id"))
            .unwrap()
            .write()
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(&path),
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    assert_eq!(query(&s, "SELECT id FROM r"), serde_json::json!([[1]]));
    assert_eq!(versions(&dir), 1);
}

#[test]
fn an_overwrite_of_a_change_data_feed_table_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join("write-features/change-data-feed"), &dir);
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!("CREATE TABLE cdf USING delta LOCATION '{path}'"),
    )
    .unwrap();
    let e = run(&s, "INSERT OVERWRITE cdf VALUES (2)").unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    assert!(e.to_string().contains("change data feed"), "{e}");
    // An append is allowed (design section 6.2).
    run(&s, "INSERT INTO cdf VALUES (2)").unwrap();
    assert_eq!(
        query(&s, "SELECT id FROM cdf ORDER BY id"),
        serde_json::json!([[1], [2]])
    );
}

#[test]
fn a_ctas_into_an_occupied_location_fails_as_delta_spark() {
    let oracle = oracle();
    let theirs = &oracle["ctas_existing_location"][0]["error"];
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!("CREATE TABLE created (id BIGINT, p STRING) USING delta PARTITIONED BY (p) LOCATION '{path}'"),
    )
    .unwrap();
    let create = &oracle["create_new_location"][0]["commits"][0];
    let ours = last_commit(&dir);
    assert_eq!(
        commit_info(&ours)["operation"],
        create["commitInfo"]["operation"]
    );
    assert_eq!(count(&ours, "add"), 0);
    let e = run(
        &s,
        &format!(
            "CREATE TABLE ctas_x USING delta LOCATION '{path}' AS SELECT CAST(1 AS BIGINT) AS id, 'x' AS p"
        ),
    )
    .unwrap_err();
    assert_eq!(
        e.error_class().name(),
        theirs["condition"].as_str().unwrap(),
        "{e}"
    );
    assert_eq!(e.sql_state(), theirs["sqlstate"].as_str());
    assert!(e.to_string().contains("`default`.`ctas_x`"), "{e}");
}

#[test]
fn a_new_table_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!("CREATE TABLE n (id BIGINT, s STRING, p INT) USING delta PARTITIONED BY (p) LOCATION '{path}'"),
    )
    .unwrap();
    assert_eq!(versions(&dir), 1);
    run(&s, "INSERT INTO n VALUES (1, 'a', 1), (2, NULL, NULL)").unwrap();
    assert_eq!(
        query(&s, "SELECT id, s, p FROM n ORDER BY id"),
        serde_json::json!([[1, "a", 1], [2, null, null]])
    );
    // The same table by path.
    assert_eq!(
        query(&s, &format!("SELECT count(*) FROM delta.`{path}`")),
        serde_json::json!([[2]])
    );
}

/// An `INSERT` whose rows read a file table (a frame analyzed afresh by each action) writes into
/// the Delta table it opened: the target is opened in the analysis that is planned. An append of
/// no rows commits nothing.
#[test]
fn an_insert_from_a_file_table_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("d").to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(&s, "CREATE TABLE src (a INT) USING parquet").unwrap();
    run(&s, "INSERT INTO src VALUES (5), (6)").unwrap();
    run(
        &s,
        &format!("CREATE TABLE d (a INT) USING delta LOCATION '{path}'"),
    )
    .unwrap();
    run(&s, "INSERT INTO d SELECT a FROM src").unwrap();
    run(&s, "CREATE TABLE m USING delta AS SELECT 1 AS a").unwrap();
    run(&s, "INSERT INTO m SELECT a FROM src").unwrap();
    run(&s, "INSERT OVERWRITE TABLE d SELECT a + 1 FROM src").unwrap();
    // An append of no rows writes no version, as delta-spark 4.4.1 (an empty overwrite commits).
    let dir = tmp.path().join("d");
    let before = versions(&dir);
    run(&s, "INSERT INTO d SELECT a FROM src WHERE a > 100").unwrap();
    assert_eq!(versions(&dir), before);
    assert_eq!(
        query(&s, "SELECT a FROM d ORDER BY a"),
        serde_json::json!([[6], [7]])
    );
    assert_eq!(
        query(&s, "SELECT a FROM m ORDER BY a"),
        serde_json::json!([[1], [5], [6]])
    );
}

/// Writes of no rows, as delta-spark 4.4.1 records them (probe 2026-11-18, krishiv-internal
/// `ops/reports/2026-11-18-o2-empty-append.md`): an append (SQL, reading its own target, or a
/// DataFrame `save`) and a dynamic partition overwrite write no version; a static overwrite and
/// a `replaceWhere` commit one version that removes the files.
#[test]
fn empty_writes_commit_as_delta_spark() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("e");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE e (a INT, p STRING) USING delta PARTITIONED BY (p) LOCATION '{path}'"
        ),
    )
    .unwrap();
    run(&s, "INSERT INTO e VALUES (1, 'x')").unwrap();
    let base = versions(&dir);
    run(&s, "INSERT INTO e SELECT a, p FROM e WHERE a > 100").unwrap();
    assert_eq!(versions(&dir), base, "self-reading append");
    let empty = block_on(s.sql("SELECT 1 AS a, 'x' AS p WHERE false")).unwrap();
    block_on(empty.write().format("delta").mode("append").save(&path)).unwrap();
    assert_eq!(versions(&dir), base, "DataFrame append");
    run(&s, "SET spark.sql.sources.partitionOverwriteMode = dynamic").unwrap();
    run(
        &s,
        "INSERT OVERWRITE TABLE e SELECT a, p FROM e WHERE a > 100",
    )
    .unwrap();
    run(&s, "RESET spark.sql.sources.partitionOverwriteMode").unwrap();
    assert_eq!(versions(&dir), base, "dynamic overwrite");
    assert_eq!(query(&s, "SELECT a FROM e"), serde_json::json!([[1]]));
    block_on(
        empty
            .write()
            .format("delta")
            .mode("overwrite")
            .option("replaceWhere", "p = 'x'")
            .save(&path),
    )
    .unwrap();
    assert_eq!(versions(&dir), base + 1, "replaceWhere");
    assert_eq!(count(&last_commit(&dir), "remove"), 1);
    run(&s, "INSERT INTO e VALUES (2, 'y')").unwrap();
    run(
        &s,
        "INSERT OVERWRITE TABLE e SELECT a, p FROM e WHERE a > 100",
    )
    .unwrap();
    assert_eq!(versions(&dir), base + 3, "static overwrite");
    assert_eq!(count(&last_commit(&dir), "remove"), 1);
    assert_eq!(count(&last_commit(&dir), "add"), 0);
    assert_eq!(
        query(&s, "SELECT count(*) FROM e"),
        serde_json::json!([[0]])
    );
}

/// A `replaceWhere` is read under the session's parser options, as delta-spark parses it with
/// the session's parser: under `spark.sql.ansi.doubleQuotedIdentifiers`, `"p" = 'y'` names the
/// partition column (by default it would compare the string `p`).
#[test]
fn a_replace_where_reads_a_double_quoted_name_as_the_column_under_ansi_identifiers() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("q");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE q (a INT, p STRING) USING delta PARTITIONED BY (p) LOCATION '{path}'"
        ),
    )
    .unwrap();
    run(&s, "INSERT INTO q VALUES (1, 'x'), (2, 'y')").unwrap();
    run(&s, "SET spark.sql.ansi.doubleQuotedIdentifiers = true").unwrap();
    let new = block_on(s.sql("SELECT 3 AS a, 'y' AS p")).unwrap();
    block_on(
        new.write()
            .format("delta")
            .mode("overwrite")
            .option("replaceWhere", "\"p\" = 'y'")
            .save(&path),
    )
    .unwrap();
    assert_eq!(
        query(&s, "SELECT a, p FROM q ORDER BY a"),
        serde_json::json!([[1, "x"], [3, "y"]])
    );
}

/// A NULL into a `NOT NULL` column of a Delta table, through SQL and the DataFrame writer, at
/// the top level and in a nested struct field, fails as delta-spark does (`WRITE-EDGES.json`
/// `not_null`) and commits nothing.
#[test]
fn a_null_in_a_not_null_column_fails_as_delta_spark() {
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("WRITE-EDGES.json")).unwrap())
            .unwrap();
    let steps = oracle["not_null"].as_array().unwrap().clone();
    let theirs = |name: &str| {
        steps
            .iter()
            .find(|s| s["step"] == name)
            .unwrap_or_else(|| panic!("{name}"))["error"]
            .clone()
    };
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE nn (id BIGINT, a INT NOT NULL, s STRUCT<x: INT NOT NULL, y: INT>) \
             USING delta LOCATION '{path}'"
        ),
    )
    .unwrap();
    run(
        &s,
        "INSERT INTO nn VALUES (1, 1, named_struct('x', 1, 'y', 1))",
    )
    .unwrap();
    let check = |name: &str, e: KrishivError| {
        let t = theirs(name);
        assert_eq!(
            e.error_class().name(),
            t["condition"].as_str().unwrap(),
            "{name}: {e}"
        );
        assert_eq!(e.sql_state(), t["sqlstate"].as_str(), "{name}");
        assert!(
            e.to_string().contains(t["message"].as_str().unwrap()),
            "{name}: {e}"
        );
        assert_eq!(versions(&dir), 2, "{name}: nothing committed");
    };
    let df = |sql: &str| block_on(s.sql(sql)).unwrap();
    for (name, select) in [
        (
            "dataframe_null_top_level",
            "SELECT CAST(2 AS BIGINT) AS id, CAST(NULL AS INT) AS a, named_struct('x', 2, 'y', 2) AS s",
        ),
        (
            "dataframe_null_nested",
            "SELECT CAST(3 AS BIGINT) AS id, 3 AS a, named_struct('x', CAST(NULL AS INT), 'y', 3) AS s",
        ),
        (
            "dataframe_null_struct",
            "SELECT CAST(4 AS BIGINT) AS id, 4 AS a, CAST(NULL AS STRUCT<x: INT, y: INT>) AS s",
        ),
    ] {
        let e = block_on(
            df(select)
                .write()
                .format("delta")
                .mode("append")
                .save(&path),
        )
        .unwrap_err();
        check(name, e);
    }
    let e = run(
        &s,
        "INSERT INTO nn VALUES (5, NULL, named_struct('x', 5, 'y', 5))",
    )
    .unwrap_err();
    check("sql_null_top_level", e);
    assert_eq!(query(&s, "SELECT id FROM nn"), serde_json::json!([[1]]));
}

/// `maxRecordsPerFile` reaches the Delta writer from `df.write.format("delta")`: the files roll
/// over at it, on a new table and an existing one, `0` is no limit (`WRITE-EDGES.json`
/// `max_records`), and a value that is not a number is refused (delta-spark raises the JVM's
/// unclassed `NumberFormatException`; Krishiv names the option).
#[test]
fn max_records_per_file_reaches_the_writer() {
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("WRITE-EDGES.json")).unwrap())
            .unwrap();
    let steps = oracle["max_records"].as_array().unwrap().clone();
    let adds = |name: &str| {
        steps
            .iter()
            .find(|s| s["step"] == name)
            .unwrap_or_else(|| panic!("{name}"))["commits"][0]["adds"]
            .as_array()
            .unwrap()
            .len()
    };
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let save = |value: &str, mode: &str| {
        block_on(
            block_on(s.sql(
                "SELECT CAST(id AS BIGINT) AS id FROM VALUES (0), (1), (2), (3), (4) AS v(id)",
            ))
            .unwrap()
            .write()
            .format("delta")
            .mode(mode)
            .option("maxRecordsPerFile", value)
            .save(&path),
        )
    };
    save("2", "errorifexists").unwrap();
    assert_eq!(count(&last_commit(&dir), "add"), adds("create_two"));
    save("1", "append").unwrap();
    assert_eq!(count(&last_commit(&dir), "add"), adds("append_one"));
    let before = versions(&dir);
    let e = save("abc", "append").unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivInvalidFormatOptionValue,
        "{e}"
    );
    assert_eq!(versions(&dir), before);
    save("0", "append").unwrap();
    assert_eq!(count(&last_commit(&dir), "add"), adds("append_zero"));
    // A value below 0 is no limit too: delta-spark reads a long and limits only above 0.
    save("-1", "append").unwrap();
    assert_eq!(count(&last_commit(&dir), "add"), adds("append_negative"));
}

/// A write to an existing Delta catalog table takes `maxRecordsPerFile` from the writer, as
/// the path write does (`WRITE-EDGES.json` `max_records_table`): `insertInto` (append and
/// overwrite), `saveAsTable` append and `writeTo(..).append()`, and a value that is not a
/// number is refused there too.
#[test]
fn max_records_per_file_reaches_a_catalog_append() {
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("WRITE-EDGES.json")).unwrap())
            .unwrap();
    let steps = oracle["max_records_table"].as_array().unwrap().clone();
    let step = |name: &str| {
        steps
            .iter()
            .find(|s| s["step"] == name)
            .unwrap_or_else(|| panic!("{name}"))
            .clone()
    };
    let adds = |name: &str| step(name)["commits"][0]["adds"].as_array().unwrap().len();
    let rows_after = |name: &str| step(name)["rows"].as_array().unwrap().len();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!("CREATE TABLE mr (id BIGINT) USING delta LOCATION '{path}'"),
    )
    .unwrap();
    let writer = |value: &str, mode: &str| {
        block_on(
            s.sql("SELECT CAST(id AS BIGINT) AS id FROM VALUES (0), (1), (2), (3), (4) AS v(id)"),
        )
        .unwrap()
        .write()
        .format("delta")
        .mode(mode)
        .option("maxRecordsPerFile", value)
    };
    let rows =
        || usize::try_from(query(&s, "SELECT count(*) FROM mr")[0][0].as_u64().unwrap()).unwrap();
    block_on(writer("1", "append").insert_into("mr")).unwrap();
    assert_eq!(count(&last_commit(&dir), "add"), adds("insert_into_one"));
    assert_eq!(rows(), rows_after("insert_into_one"));
    block_on(writer("1", "overwrite").insert_into("mr")).unwrap();
    assert_eq!(
        count(&last_commit(&dir), "add"),
        adds("insert_into_overwrite_one")
    );
    assert_eq!(rows(), rows_after("insert_into_overwrite_one"));
    let before = versions(&dir);
    let e = block_on(writer("abc", "append").insert_into("mr")).unwrap_err();
    assert!(step("insert_into_not_a_number")["error"].is_object());
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivInvalidFormatOptionValue,
        "{e}"
    );
    assert_eq!(versions(&dir), before);
    block_on(writer("1", "append").save_as_table("mr")).unwrap();
    assert_eq!(
        count(&last_commit(&dir), "add"),
        adds("save_as_table_append_one")
    );
    assert_eq!(rows(), rows_after("save_as_table_append_one"));
    block_on(
        block_on(
            s.sql("SELECT CAST(id AS BIGINT) AS id FROM VALUES (0), (1), (2), (3), (4) AS v(id)"),
        )
        .unwrap()
        .write_to_table("mr")
        .option("maxRecordsPerFile", "1")
        .append(),
    )
    .unwrap();
    assert_eq!(
        count(&last_commit(&dir), "add"),
        adds("save_as_table_append_one")
    );
    let e = block_on(writer("abc", "append").save_as_table("mr")).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivInvalidFormatOptionValue,
        "{e}"
    );
    assert_eq!(rows(), 15);
}

/// E34: a retried `INSERT` under the same `krishiv.write.idempotencyKey` appends once (a `txn`
/// action in the first commit); the next key appends again.
#[test]
fn a_retried_insert_with_the_same_idempotency_key_appends_once() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("k");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!("CREATE TABLE k (a INT) USING delta LOCATION '{path}'"),
    )
    .unwrap();
    s.conf()
        .set("krishiv.write.idempotencyKey", "job-1")
        .unwrap();
    run(&s, "INSERT INTO k VALUES (1), (2)").unwrap();
    let after_first = versions(&dir);
    run(&s, "INSERT INTO k VALUES (1), (2)").unwrap();
    assert_eq!(versions(&dir), after_first, "the retry committed nothing");
    assert_eq!(
        query(&s, "SELECT a FROM k ORDER BY a"),
        serde_json::json!([[1], [2]])
    );
    assert!(
        last_commit(&dir)
            .iter()
            .any(|a| a["txn"]["appId"] == "krishiv.idempotency:job-1")
    );
    s.conf()
        .set("krishiv.write.idempotencyKey", "job-2")
        .unwrap();
    run(&s, "INSERT INTO k VALUES (3)").unwrap();
    assert_eq!(
        query(&s, "SELECT a FROM k ORDER BY a"),
        serde_json::json!([[1], [2], [3]])
    );
}

/// E34: a retried `UPDATE` or `DELETE` under the same `krishiv.write.idempotencyKey` changes the
/// table once, even when the statement is not idempotent (`a = a + 1`); a new key applies again.
#[test]
fn a_retried_update_and_delete_with_the_same_idempotency_key_apply_once() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("k");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!("CREATE TABLE k (a INT) USING delta LOCATION '{path}'"),
    )
    .unwrap();
    run(&s, "INSERT INTO k VALUES (1), (2), (3)").unwrap();
    s.conf()
        .set("krishiv.write.idempotencyKey", "upd-1")
        .unwrap();
    run(&s, "UPDATE k SET a = a + 10").unwrap();
    let after_first = versions(&dir);
    run(&s, "UPDATE k SET a = a + 10").unwrap();
    assert_eq!(versions(&dir), after_first, "the retry committed nothing");
    assert_eq!(
        query(&s, "SELECT a FROM k ORDER BY a"),
        serde_json::json!([[11], [12], [13]])
    );
    assert!(
        last_commit(&dir)
            .iter()
            .any(|a| a["txn"]["appId"] == "krishiv.idempotency:upd-1")
    );
    s.conf()
        .set("krishiv.write.idempotencyKey", "del-1")
        .unwrap();
    assert_eq!(
        query(&s, "DELETE FROM k WHERE a = 11"),
        serde_json::json!([[1]])
    );
    let after_delete = versions(&dir);
    // The retry changed nothing: its result row claims no rows (not the scan's count of 1).
    assert_eq!(
        query(&s, "DELETE FROM k WHERE a = 12"),
        serde_json::json!([[0]])
    );
    assert_eq!(versions(&dir), after_delete, "the retried key is a no-op");
    assert_eq!(
        query(&s, "SELECT a FROM k ORDER BY a"),
        serde_json::json!([[12], [13]])
    );
}

/// E34: a keyed `save` whose key the table records succeeds under `errorifexists` and `overwrite`
/// and writes nothing (as a directory write does); without the key the path still exists.
#[test]
fn a_keyed_save_retry_succeeds_under_errorifexists_and_overwrite() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("k");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let mut after_first = 0;
    for mode in ["errorifexists", "errorifexists", "overwrite"] {
        block_on(
            block_on(s.sql("SELECT CAST(1 AS INT) AS a UNION ALL SELECT 2"))
                .unwrap()
                .write()
                .format("delta")
                .mode(mode)
                .option("krishiv.write.idempotencyKey", "job-1")
                .save(&path),
        )
        .unwrap();
        if after_first == 0 {
            after_first = versions(&dir);
        }
        assert_eq!(versions(&dir), after_first, "{mode} committed again");
    }
    let e = block_on(
        block_on(s.sql("SELECT 1 AS a"))
            .unwrap()
            .write()
            .format("delta")
            .mode("errorifexists")
            .save(&path),
    )
    .unwrap_err();
    assert!(
        e.message().contains(&path) || e.message().contains("exists"),
        "{e}"
    );
}

/// E34: `saveAsTable` that creates a Delta table records its explicit key in version 0, so the same
/// append retried writes nothing; the key is not kept in the table's options.
#[test]
fn a_keyed_save_as_table_creation_is_retried_once() {
    let tmp = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    for _ in 0..2 {
        block_on(
            block_on(s.sql("SELECT CAST(1 AS INT) AS a"))
                .unwrap()
                .write()
                .format("delta")
                .mode("append")
                .option("krishiv.write.idempotencyKey", "job-1")
                .option("path", tmp.path().join("t").to_string_lossy().as_ref())
                .save_as_table("keyed_t"),
        )
        .unwrap();
    }
    assert_eq!(query(&s, "SELECT a FROM keyed_t"), serde_json::json!([[1]]));
    // The key is not kept in the table's options: a later append without a key writes.
    block_on(
        block_on(s.sql("SELECT CAST(2 AS INT) AS a"))
            .unwrap()
            .write()
            .format("delta")
            .mode("append")
            .save_as_table("keyed_t"),
    )
    .unwrap();
    assert_eq!(
        query(&s, "SELECT a FROM keyed_t ORDER BY a"),
        serde_json::json!([[1], [2]])
    );
}

#[test]
fn a_bucketed_save_as_table_of_delta_is_refused_before_the_table_exists() {
    let s = Session::new().unwrap();
    let e = block_on(
        block_on(s.sql("SELECT CAST(1 AS BIGINT) AS id"))
            .unwrap()
            .write()
            .format("delta")
            .bucket_by(4, ["id"])
            .save_as_table("bt"),
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    assert!(
        block_on(s.sql("SELECT * FROM bt")).is_err(),
        "no table was registered"
    );
}

/// `insertInto` overwrite of a partitioned Delta catalog table ignores a
/// `partitionOverwriteMode` writer option: the session configuration alone makes it dynamic,
/// as in delta-spark (`WRITE-EDGES.json` `insert_into_overwrite_mode`).
#[test]
fn insert_into_overwrite_mode_follows_the_session_config_only() {
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("WRITE-EDGES.json")).unwrap())
            .unwrap();
    let steps = oracle["insert_into_overwrite_mode"]
        .as_array()
        .unwrap()
        .clone();
    let rows_after = |name: &str| {
        let step = steps
            .iter()
            .find(|s| s["step"] == name)
            .unwrap_or_else(|| panic!("{name}"));
        assert!(step["error"].is_null(), "{name}");
        step["rows"].clone()
    };
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("t").to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE iom (id BIGINT, p STRING) USING delta PARTITIONED BY (p) \
             LOCATION '{path}'"
        ),
    )
    .unwrap();
    run(&s, "INSERT INTO iom VALUES (1, 'a'), (2, 'b')").unwrap();
    let overwrite = |id: i64, option: &str| {
        let df = block_on(s.sql(&format!("SELECT CAST({id} AS BIGINT) AS id, 'a' AS p"))).unwrap();
        block_on(
            df.write()
                .mode("overwrite")
                .option("partitionOverwriteMode", option)
                .insert_into("iom"),
        )
        .unwrap();
    };
    let rows = || query(&s, "SELECT id, p FROM iom ORDER BY id");
    overwrite(3, "dynamic");
    assert_eq!(rows(), rows_after("option_dynamic_conf_unset"));
    run(&s, "INSERT INTO iom VALUES (4, 'b')").unwrap();
    assert_eq!(rows(), rows_after("insert_b"));
    run(&s, "SET spark.sql.sources.partitionOverwriteMode = dynamic").unwrap();
    overwrite(5, "static");
    run(&s, "RESET spark.sql.sources.partitionOverwriteMode").unwrap();
    assert_eq!(rows(), rows_after("option_static_conf_dynamic"));
}

/// A table root reached through a symlinked directory (macOS `/var` -> `/private/var`) writes and
/// reads: every location is compared in one form.
#[cfg(unix)]
#[test]
fn a_new_table_under_a_symlinked_directory_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real");
    std::fs::create_dir(&real).unwrap();
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let path = link.join("delta-table").to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!("CREATE TABLE n (id BIGINT) USING delta LOCATION '{path}'"),
    )
    .unwrap();
    run(&s, "INSERT INTO n VALUES (1), (2)").unwrap();
    assert_eq!(
        query(&s, &format!("SELECT count(*) FROM delta.`{path}`")),
        serde_json::json!([[2]])
    );
    run(&s, "INSERT INTO n VALUES (3)").unwrap();
    assert_eq!(
        query(&s, "SELECT count(*) FROM n"),
        serde_json::json!([[3]])
    );
}

/// A Parquet directory written and read through a symlinked directory.
#[cfg(unix)]
#[test]
fn a_parquet_write_under_a_symlinked_directory_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real");
    std::fs::create_dir(&real).unwrap();
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let path = link.join("out").to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    run(
        &s,
        &format!(
            "INSERT OVERWRITE DIRECTORY '{path}' USING parquet SELECT 1 AS id UNION ALL SELECT 2"
        ),
    )
    .unwrap();
    assert_eq!(
        query(&s, &format!("SELECT count(*) FROM parquet.`{path}`")),
        serde_json::json!([[2]])
    );
}
