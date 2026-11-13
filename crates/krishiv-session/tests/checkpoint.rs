// SPDX-License-Identifier: Apache-2.0
//! `checkpoint` and `localCheckpoint` (design P3-T083 §3.3): the lineage is cut, a missing
//! directory is Spark's error, the snapshot's files go with it, `system.caches` shows it.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

/// A session whose warehouse is `dir`, with the managed table `src(id)` holding 1, 2, 3.
fn session(dir: &std::path::Path, pairs: &[(&str, &str)]) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.join("wh").to_str().unwrap())
        .unwrap();
    for (k, v) in pairs {
        s.conf().set(k, v).unwrap();
    }
    run(&s, "CREATE TABLE src (id INT) USING parquet");
    run(&s, "INSERT INTO src VALUES (1), (2), (3)");
    s
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap();
}

fn ids(df: &krishiv_session::DataFrame) -> Vec<i32> {
    let mut out: Vec<i32> = block_on(df.collect())
        .unwrap()
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    out.sort_unstable();
    out
}

fn files(dir: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| {
            let p = e.path();
            if p.is_dir() {
                files(&p)
            } else {
                usize::from(p.extension().is_some_and(|x| x == "spill"))
            }
        })
        .sum()
}

#[test]
fn a_reliable_checkpoint_needs_a_directory_eager_or_not() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    for eager in [true, false] {
        let e = block_on(s.table("src").checkpoint(eager)).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp3016, "{eager}");
        assert!(
            e.to_string()
                .contains("Checkpoint directory has not been set in the SparkContext"),
            "{e}"
        );
    }
}

#[test]
fn a_local_checkpoint_does_not_see_later_writes_a_cache_does() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let cp = block_on(s.table("src").local_checkpoint(true, None)).unwrap();
    let cached = s.table("src").cache();
    assert_eq!(ids(&cached), vec![1, 2, 3]);
    run(&s, "INSERT INTO src VALUES (4)");
    assert_eq!(ids(&cp), vec![1, 2, 3], "the lineage is cut");
    assert_eq!(ids(&cached), vec![1, 2, 3, 4], "the cache is invalidated");
    assert!(!block_on(cp.is_cached()).unwrap());
    run(&s, "CLEAR CACHE");
    assert_eq!(ids(&cp), vec![1, 2, 3], "CLEAR CACHE does not reach it");
    // A frame derived from the checkpoint reads the same rows.
    let derived =
        cp.filter(krishiv_session::functions::col("id").gt(krishiv_session::functions::lit(1)));
    assert_eq!(ids(&derived), vec![2, 3]);
}

#[test]
fn a_lazy_checkpoint_takes_its_rows_at_the_first_action() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    run(&s, "INSERT INTO src VALUES (4)");
    let derived =
        cp.filter(krishiv_session::functions::col("id").gt(krishiv_session::functions::lit(1)));
    assert_eq!(ids(&derived), vec![2, 3, 4], "the first action's rows");
    run(&s, "INSERT INTO src VALUES (5)");
    assert_eq!(ids(&cp), vec![1, 2, 3, 4], "fixed by the first action");
}

#[test]
fn the_krishiv_key_sets_the_checkpoint_directory() {
    let dir = tempfile::tempdir().unwrap();
    let cp_dir = dir.path().join("checkpoints");
    let s = session(
        dir.path(),
        &[("krishiv.checkpoint.dir", cp_dir.to_str().unwrap())],
    );
    let cp = block_on(s.table("src").checkpoint(true)).unwrap();
    assert_eq!(files(&cp_dir), 1);
    assert_eq!(ids(&cp), vec![1, 2, 3]);
}

#[test]
fn a_reliable_checkpoint_writes_under_the_directory_and_deletes_with_the_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let cp_dir = dir.path().join("checkpoints");
    let s = session(
        dir.path(),
        &[("spark.checkpoint.dir", cp_dir.to_str().unwrap())],
    );
    let cp = block_on(s.table("src").checkpoint(true)).unwrap();
    assert_eq!(files(&cp_dir), 1);
    run(&s, "INSERT INTO src VALUES (4)");
    assert_eq!(ids(&cp), vec![1, 2, 3]);
    let derived = cp.limit(10);
    drop(cp);
    assert_eq!(files(&cp_dir), 1, "a derived frame still owns the snapshot");
    assert_eq!(ids(&derived), vec![1, 2, 3]);
    drop(derived);
    assert_eq!(files(&cp_dir), 0, "the files go with the snapshot");
}

#[test]
fn system_caches_has_a_checkpoint_row() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let row = |s: &Session| -> String {
        let b = block_on(async {
            s.sql("SELECT concat_ws('|', CAST(entries AS STRING), CAST(memory_bytes > 0 AS STRING)) FROM system.caches WHERE cache = 'checkpoint'")
                .await?
                .collect()
                .await
        })
        .unwrap();
        let c = b[0].column(0);
        c.as_string_view_opt()
            .map(|v| v.value(0).to_owned())
            .unwrap_or_else(|| c.as_string::<i32>().value(0).to_owned())
    };
    assert_eq!(row(&s), "0|false");
    let cp = block_on(s.table("src").local_checkpoint(true, None)).unwrap();
    assert_eq!(row(&s), "1|true");
    drop(cp);
    assert_eq!(row(&s), "0|false");
}

/// `f` run on a thread of its own, failing the test when it does not end in 60 seconds.
fn within_a_minute<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .expect("the call hung")
}

/// A `cache()` of a frame standing on a lazy checkpoint registers without a hang: registering
/// analyzes the frame, which takes the checkpoint's rows by running a query, which reads the
/// registry again (review B1).
#[test]
fn caching_a_lazy_checkpoint_does_not_hang() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    let cached = cp.cache();
    let (n, is_cached) = within_a_minute(move || {
        let n = block_on(cached.count()).unwrap();
        (n, block_on(cached.is_cached()).unwrap())
    });
    assert_eq!(n, 3);
    assert!(is_cached);
    // A frame derived from the checkpoint, cached, then read.
    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    let derived =
        cp.filter(krishiv_session::functions::col("id").gt(krishiv_session::functions::lit(1)));
    let derived = derived.cache();
    let rows = within_a_minute(move || ids(&derived));
    assert_eq!(rows, vec![2, 3]);
}

#[test]
fn a_frame_with_no_columns_checkpoints_with_its_rows() {
    let dir = tempfile::tempdir().unwrap();
    let cp_dir = dir.path().join("checkpoints");
    let s = session(
        dir.path(),
        &[("spark.checkpoint.dir", cp_dir.to_str().unwrap())],
    );
    for reliable in [false, true] {
        let df = block_on(s.sql("SELECT * EXCEPT (id) FROM src")).unwrap();
        let cp = if reliable {
            block_on(df.checkpoint(true))
        } else {
            block_on(df.local_checkpoint(true, None))
        }
        .unwrap();
        assert_eq!(block_on(cp.count()).unwrap(), 3);
    }
}

/// A cached lazy checkpoint answers `schema()` first, and two cached frames sharing one lazy
/// checkpoint both read: taking the rows must not wait on a registration that waits on them.
#[test]
fn registrations_sharing_a_lazy_checkpoint_do_not_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    let cached = cp.cache();
    let schema_fields = within_a_minute(move || {
        let n = block_on(cached.schema()).unwrap().fields().len();
        (n, block_on(cached.count()).unwrap())
    });
    assert_eq!(schema_fields, (1, 3));

    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    let a = cp.limit(10).cache();
    let b = cp.limit(20).cache();
    let counts =
        within_a_minute(move || (block_on(a.count()).unwrap(), block_on(b.count()).unwrap()));
    assert_eq!(counts, (3, 3));
}

#[test]
fn a_temp_view_of_a_lazy_file_checkpoint_keeps_the_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let path = dir.path().join("rows.parquet");
    let path = path.to_str().unwrap().to_owned();
    block_on(s.table("src").write().mode("overwrite").parquet(&path)).unwrap();
    let df = s.read().parquet([path.as_str()]);
    let cp = block_on(df.local_checkpoint(false, None)).unwrap();
    block_on(cp.create_or_replace_temp_view("snap")).unwrap();
    block_on(
        s.table("src")
            .filter(krishiv_session::functions::col("id").gt(krishiv_session::functions::lit(0)))
            .write()
            .mode("append")
            .parquet(&path),
    )
    .unwrap();
    assert_eq!(
        ids(&block_on(s.sql("SELECT * FROM snap")).unwrap()),
        vec![1, 2, 3]
    );
}
