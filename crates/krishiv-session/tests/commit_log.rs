// SPDX-License-Identifier: Apache-2.0
//! Writes through the commit log (`krishiv.sql.files.commitProtocol`, P2-T082b
//! unit 3): Krishiv readers see only committed files.
#![expect(clippy::unwrap_used, reason = "test code")]

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

fn session_with(rows: &[i32]) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(rows.to_vec()))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

fn ids(s: &Session, path: &str) -> Vec<i64> {
    let df = block_on(s.sql(&format!("SELECT CAST(id AS BIGINT) FROM parquet.`{path}`"))).unwrap();
    let mut out = Vec::new();
    for b in block_on(df.collect()).unwrap() {
        let c = b
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        out.extend(c.iter().map(Option::unwrap));
    }
    out.sort_unstable();
    out
}

fn log_versions(root: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(root.join("_krishiv_log"))
        .map(|d| {
            d.map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

#[test]
fn appends_and_overwrites_go_through_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = session_with(&[1, 2]);
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    assert_eq!(
        log_versions(&root),
        ["00000000000000000000.json", "00000000000000000001.json"]
    );
    assert_eq!(ids(&s, &path), [1, 1, 2, 2]);
    let s2 = session_with(&[7]);
    block_on(s2.table("t").write().mode("overwrite").parquet(&path)).unwrap();
    assert_eq!(log_versions(&root).len(), 3);
    assert_eq!(ids(&s2, &path), [7]);
}

#[test]
fn a_file_nobody_committed_is_invisible_to_readers_of_a_logged_directory() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = session_with(&[1, 2]);
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    // Another engine's file (a copy of a committed one) in the directory.
    let data = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::copy(&data, root.join("part-99999-foreign-c000.snappy.parquet")).unwrap();
    assert_eq!(ids(&s, &path), [1, 2]);
}

#[test]
fn the_first_logged_append_adopts_the_files_already_there() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = session_with(&[1, 2]);
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "NONE")
        .unwrap();
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    assert!(log_versions(&root).is_empty());
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    assert_eq!(ids(&s, &path), [1, 1, 2, 2]);
}

#[test]
fn escaped_partition_values_survive_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'a/b%c' AS p")).unwrap();
    block_on(df.write().partition_by(["p"]).mode("append").parquet(&path)).unwrap();
    assert_eq!(ids(&s, &path), [1]);
}
