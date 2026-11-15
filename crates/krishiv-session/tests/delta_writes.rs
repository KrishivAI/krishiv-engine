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
