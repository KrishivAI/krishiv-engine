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
    // Version 0 initializes the log (before any file is written), then one commit per append.
    assert_eq!(
        log_versions(&root),
        [
            "00000000000000000000.json",
            "00000000000000000001.json",
            "00000000000000000002.json"
        ]
    );
    assert_eq!(ids(&s, &path), [1, 1, 2, 2]);
    let s2 = session_with(&[7]);
    block_on(s2.table("t").write().mode("overwrite").parquet(&path)).unwrap();
    assert_eq!(log_versions(&root).len(), 4);
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

#[test]
fn a_dynamic_overwrite_keeps_the_other_partitions_in_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'a' AS p UNION ALL SELECT 2, 'b'")).unwrap();
    block_on(df.write().partition_by(["p"]).mode("append").parquet(&path)).unwrap();
    s.conf()
        .set("spark.sql.sources.partitionOverwriteMode", "DYNAMIC")
        .unwrap();
    let df = block_on(s.sql("SELECT 3 AS id, 'a' AS p")).unwrap();
    block_on(
        df.write()
            .partition_by(["p"])
            .mode("overwrite")
            .parquet(&path),
    )
    .unwrap();
    // `p=a` was replaced; `p=b` is still visible through the log.
    assert_eq!(ids(&s, &path), [2, 3]);
}

#[test]
fn an_existing_log_is_kept_by_a_session_that_does_not_ask_for_one() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let logged = session_with(&[1]);
    block_on(logged.table("t").write().mode("append").parquet(&path)).unwrap();
    // A fresh session with the default protocol appends: the rows stay visible.
    let plain = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![2]))],
    )
    .unwrap();
    plain.register_batches("t", schema, vec![batch]).unwrap();
    block_on(plain.table("t").write().mode("append").parquet(&path)).unwrap();
    assert_eq!(ids(&plain, &path), [1, 2]);
}

#[test]
fn a_first_write_that_died_leaves_its_files_outside_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.to_str().unwrap().to_owned();
    // A first logged append initializes the log (adopting nothing), then dies before its commit:
    // simulated by a stray data file written after the log exists.
    let s = session_with(&[1]);
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    let data = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::copy(&data, root.join("part-00000-dead-c000.snappy.parquet")).unwrap();
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    // The dead write's file is not adopted by the next append.
    assert_eq!(ids(&s, &path), [1, 1]);
}

#[test]
fn a_wildcard_read_of_a_logged_root_skips_the_log_directory() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = session_with(&[4, 5]);
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    assert_eq!(ids(&s, &format!("{path}/*")), [4, 5]);
}

#[test]
fn a_wildcard_read_of_a_logged_root_skips_uncommitted_files_too() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = session_with(&[4, 5]);
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    let data = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::copy(&data, root.join("part-99999-dead-c000.snappy.parquet")).unwrap();
    assert_eq!(ids(&s, &path), [4, 5]);
    assert_eq!(ids(&s, &format!("{path}/*")), [4, 5]);
}

#[test]
fn a_wildcard_over_partition_directories_of_a_logged_root_skips_uncommitted_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'a' AS p")).unwrap();
    block_on(df.write().partition_by(["p"]).mode("append").parquet(&path)).unwrap();
    let part = root.join("p=a");
    let data = std::fs::read_dir(&part)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::copy(&data, part.join("part-99999-dead.c000.snappy.parquet")).unwrap();
    assert_eq!(ids(&s, &format!("{path}/p=*")), [1]);
}

#[test]
fn a_wildcard_read_notices_a_committed_file_that_is_gone() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = session_with(&[1]);
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    block_on(s.table("t").write().mode("append").parquet(&path)).unwrap();
    // One committed file disappears (as a concurrent overwrite's cleanup would do it).
    let mut files: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
        .collect();
    files.sort();
    std::fs::remove_file(&files[0]).unwrap();
    let df = block_on(s.sql(&format!("SELECT id FROM parquet.`{path}/*`")));
    let err = match df {
        Ok(df) => block_on(df.collect()).unwrap_err(),
        Err(e) => e,
    };
    assert_eq!(err.error_class().name(), "KRISHIV_CONCURRENT_COMMIT");
}

#[test]
fn a_partition_subdirectory_read_with_base_path_shows_only_committed_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'a' AS p")).unwrap();
    block_on(df.write().partition_by(["p"]).mode("append").parquet(&path)).unwrap();
    let part = root.join("p=a");
    let data = std::fs::read_dir(&part)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::copy(&data, part.join("part-99999-dead.c000.snappy.parquet")).unwrap();
    let df = s
        .read()
        .option("basePath", path.as_str())
        .parquet([format!("{path}/p=a").as_str()]);
    let n = block_on(df.count()).unwrap();
    assert_eq!(n, 1, "the abandoned file is outside the root's log");
}

#[test]
fn a_partition_read_with_base_path_notices_a_committed_file_that_is_gone() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    for _ in 0..2 {
        let df = block_on(s.sql("SELECT 1 AS id, 'a' AS p")).unwrap();
        block_on(df.write().partition_by(["p"]).mode("append").parquet(&path)).unwrap();
    }
    let part = root.join("p=a");
    let mut files: Vec<_> = std::fs::read_dir(&part)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
        .collect();
    files.sort();
    assert_eq!(files.len(), 2);
    let read = |s: &Session| {
        s.read()
            .option("basePath", path.as_str())
            .parquet([format!("{path}/p=a").as_str()])
    };
    assert_eq!(block_on(read(&s).count()).unwrap(), 2);
    // One of the directory's committed files disappears (a concurrent overwrite's cleanup):
    // the read refuses instead of returning the rows that are left.
    std::fs::remove_file(&files[0]).unwrap();
    let err = block_on(read(&s).count()).unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_CONCURRENT_COMMIT");
}

#[test]
fn a_glob_below_a_base_path_root_shows_only_committed_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'a' AS p")).unwrap();
    block_on(df.write().partition_by(["p"]).mode("append").parquet(&path)).unwrap();
    let part = root.join("p=a");
    let data = std::fs::read_dir(&part)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::copy(&data, part.join("part-99999-dead.c000.snappy.parquet")).unwrap();
    let df = s
        .read()
        .option("basePath", path.as_str())
        .parquet([format!("{path}/p=a/*.parquet").as_str()]);
    assert_eq!(
        block_on(df.count()).unwrap(),
        1,
        "the abandoned file is outside the root's log"
    );
}

#[test]
fn a_partition_directory_read_by_its_own_path_shows_only_the_roots_committed_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'a' AS p")).unwrap();
    block_on(df.write().partition_by(["p"]).mode("append").parquet(&path)).unwrap();
    let part = root.join("p=a");
    let data = std::fs::read_dir(&part)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::copy(&data, part.join("part-99999-dead.c000.snappy.parquet")).unwrap();
    // No basePath: the log of the root above the `p=a` directory still applies.
    let df = s.read().parquet([format!("{path}/p=a").as_str()]);
    assert_eq!(
        block_on(df.count()).unwrap(),
        1,
        "the abandoned file is not read"
    );
    // And a glob below the root.
    let df = s.read().parquet([format!("{path}/p=a/*.parquet").as_str()]);
    assert_eq!(block_on(df.count()).unwrap(), 1);
}

#[test]
fn a_write_into_a_partition_directory_of_a_logged_root_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'a' AS p")).unwrap();
    block_on(df.write().partition_by(["p"]).mode("append").parquet(&path)).unwrap();
    let df = block_on(s.sql("SELECT 2 AS id")).unwrap();
    let err = block_on(
        df.write()
            .mode("append")
            .parquet(format!("{path}/p=a").as_str()),
    )
    .unwrap_err();
    assert_eq!(
        err.error_class().name(),
        "KRISHIV_UNSUPPORTED_FEATURE",
        "{err}"
    );
    // Nothing was added, and the root still reads its one committed row.
    assert!(!root.join("p=a/_krishiv_log").exists());
    let n = block_on(s.read().parquet([path.as_str()]).count()).unwrap();
    assert_eq!(n, 1);
}

/// A nested partition path is written when no ancestor has a log, even if its parent exists: the
/// ancestors are probed through the store as the path names it, not the one anchored at the target.
#[test]
fn a_nested_partition_path_without_an_ancestor_log_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "NONE")
        .unwrap();
    std::fs::create_dir_all(root.join("p=a")).unwrap();
    let df = block_on(s.sql("SELECT 1 AS id")).unwrap();
    let nested = root.join("p=a/q=b");
    block_on(df.write().mode("append").parquet(nested.to_str().unwrap())).unwrap();
    let n = block_on(s.read().parquet([nested.to_str().unwrap()]).count()).unwrap();
    assert_eq!(n, 1);
}

/// With `basePath` on a partition directory whose root holds the log, the root's log applies.
#[test]
fn a_base_path_on_a_partition_directory_uses_the_roots_log() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("out");
    let path = root.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'a' AS p, 'x' AS q")).unwrap();
    block_on(
        df.write()
            .partition_by(["p", "q"])
            .mode("append")
            .parquet(&path),
    )
    .unwrap();
    let leaf = root.join("p=a/q=x");
    let data = std::fs::read_dir(&leaf)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::copy(&data, leaf.join("part-99999-dead.c000.snappy.parquet")).unwrap();
    let base = root.join("p=a");
    let df = s
        .read()
        .option("basePath", base.to_str().unwrap())
        .parquet([leaf.to_str().unwrap()]);
    assert_eq!(
        block_on(df.count()).unwrap(),
        1,
        "the abandoned file is not read"
    );
}

/// A glob whose prefix is a partition directory with a log of its own, below an unlogged `basePath`.
#[test]
fn a_glob_under_a_base_path_uses_a_log_between_the_base_and_the_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let logged = out.join("p=a");
    let path = logged.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'x' AS q")).unwrap();
    block_on(df.write().partition_by(["q"]).mode("append").parquet(&path)).unwrap();
    let leaf = logged.join("q=x");
    let data = std::fs::read_dir(&leaf)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::copy(&data, leaf.join("part-99999-dead.c000.snappy.parquet")).unwrap();
    let df = s
        .read()
        .option("basePath", out.to_str().unwrap())
        .parquet([format!("{}/*.parquet", leaf.to_str().unwrap()).as_str()]);
    assert_eq!(
        block_on(df.count()).unwrap(),
        1,
        "the abandoned file is not read"
    );
}

/// A log made at an intermediate partition directory between `basePath` and the read path while the
/// read lists is noticed by the recheck (the same resolution order as the first lookup).
#[test]
fn an_intermediate_log_below_a_base_path_is_found_by_the_recheck_order() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let mid = out.join("p=a");
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "LOG")
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS id, 'x' AS q")).unwrap();
    block_on(
        df.write()
            .partition_by(["q"])
            .mode("append")
            .parquet(mid.to_str().unwrap()),
    )
    .unwrap();
    // The directory read by its own path under an unlogged basePath: the log between them applies.
    let leaf = mid.join("q=x");
    let df = s
        .read()
        .option("basePath", out.to_str().unwrap())
        .parquet([leaf.to_str().unwrap()]);
    assert_eq!(block_on(df.count()).unwrap(), 1);
}
