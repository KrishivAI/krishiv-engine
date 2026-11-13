// SPDX-License-Identifier: Apache-2.0
//! The relation cache: `df.cache` / `persist` / `unpersist`, the byte cap,
//! invalidation, and the SQL forms that touch it.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::functions::{col, lit};
use krishiv_session::{Session, StorageLevel};

/// A session with the memory table `t(id)` holding `values`.
fn int_session(values: Vec<i32>) -> Session {
    let session = Session::new().unwrap();
    register(&session, "t", values);
    session
}

fn register(session: &Session, name: &str, values: Vec<i32>) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(values))],
    )
    .unwrap();
    session.register_batches(name, schema, vec![batch]).unwrap();
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int32, false)]))
}

/// The `id` column of `batches`, unordered.
fn ids(batches: &[RecordBatch]) -> Vec<i32> {
    let mut out: Vec<i32> = batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    out.sort_unstable();
    out
}

/// The schema and the sorted rows of `batches`, each row rendered from its
/// columns' values: the oracle for "the cache returned what the query returns".
fn rendered(batches: &[RecordBatch]) -> (Vec<String>, Vec<String>) {
    let fields = batches
        .first()
        .map(|b| {
            b.schema()
                .fields()
                .iter()
                .map(|f| format!("{}:{:?}:{}", f.name(), f.data_type(), f.is_nullable()))
                .collect()
        })
        .unwrap_or_default();
    let mut rows: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(move |i| {
                b.columns()
                    .iter()
                    .map(|c| format!("{:?}", c.slice(i, 1)))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect();
    rows.sort();
    (fields, rows)
}

/// A query-clock column is the proof that later actions read the
/// snapshot: recomputing from the source would read a new query clock.
#[test]
fn later_actions_return_the_values_of_the_first_run() {
    let s = int_session(vec![1, 2, 3]);
    block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
    let df = block_on(s.sql("SELECT id, current_timestamp() AS ts FROM t"))
        .unwrap()
        .cache();
    let first = rendered(&block_on(df.collect()).unwrap());
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(rendered(&block_on(df.collect()).unwrap()), first);
    // Without the entry the query runs again and reads a new clock.
    block_on(df.unpersist()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert_ne!(rendered(&block_on(df.collect()).unwrap()), first);
}

#[test]
fn cached_collect_serves_the_snapshot_not_the_source() {
    let s = int_session(vec![1, 2]);
    let df = s.table("t").cache();
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![1, 2]);
    assert!(block_on(df.is_cached()).unwrap());
    // Replacing the table drops the registration and what was cached from
    // the old one, as replacing a temporary view uncaches it in Spark: both
    // frames read the new rows, and nothing is cached any more.
    register(&s, "t", vec![99, 100, 101]);
    assert_eq!(block_on(s.table("t").count()).unwrap(), 3);
    assert!(!block_on(df.is_cached()).unwrap());
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![99, 100, 101]);
    assert!(!block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
}

#[test]
fn uncached_frames_do_not_fill_and_derived_frames_fill_entries() {
    let s = int_session(vec![1, 2]);
    // No `cache()`: no entry, however often the frame runs.
    let df = s.table("t");
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![1, 2]);
    assert!(!block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
    // A frame derived from a cached frame is not itself cached, but its
    // first action materializes the registered plan (Spark's "first use
    // fills"), before the cached frame ever ran.
    let cached = s.table("t").cache();
    let filtered = cached.filter(col("id").gt(lit(0)));
    assert!(!block_on(filtered.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
    assert_eq!(ids(&block_on(filtered.collect()).unwrap()), vec![1, 2]);
    assert!(
        s.cache_reserved() > 0,
        "the derived frame's action filled t"
    );
    // An identical frame built elsewhere sees the registration too.
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![1, 2]);
    // Replacing the table drops the registration: both read the new table.
    register(&s, "t", vec![99]);
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![99]);
    assert_eq!(ids(&block_on(filtered.collect()).unwrap()), vec![99]);
    assert!(!block_on(cached.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
}

#[test]
fn principals_never_share_an_entry() {
    use krishiv_plan::analyzer::{Principal, PrincipalKind};
    let data = || {
        let schema = schema();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )
        .unwrap();
        (schema, batch)
    };
    let (schema, batch) = data();
    let a = Session::builder()
        .principal(Principal::new("a", PrincipalKind::User))
        .build()
        .unwrap();
    a.register_batches("t", schema, vec![batch]).unwrap();
    let (schema, batch) = data();
    let b = Session::builder()
        .principal(Principal::new("b", PrincipalKind::User))
        .build()
        .unwrap();
    b.register_batches("t", schema, vec![batch]).unwrap();
    assert_eq!(block_on(a.table("t").cache().collect()).unwrap().len(), 1);
    assert!(block_on(a.table("t").is_cached()).unwrap());
    assert!(a.cache_reserved() > 0);
    // B runs the same query but holds no entry: per-session caches never
    // cross principals (the pool counters confirm it).
    assert!(!block_on(b.table("t").is_cached()).unwrap());
    assert_eq!(b.cache_reserved(), 0);
    assert_eq!(block_on(b.table("t").count()).unwrap(), 2);
}

#[test]
fn unpersist_returns_the_pool_to_baseline() {
    let s = int_session(vec![1, 2, 3]);
    let baseline = s.cache_reserved();
    let df = s.table("t").cache();
    assert_eq!(block_on(df.collect()).unwrap().len(), 1);
    assert!(s.cache_reserved() > baseline);
    let df = block_on(df.unpersist()).unwrap();
    assert_eq!(s.cache_reserved(), baseline);
    assert!(!block_on(df.is_cached()).unwrap());
    // Unpersisting again is a no-op.
    let df = block_on(df.unpersist()).unwrap();
    assert_eq!(s.cache_reserved(), baseline);
    assert!(!block_on(df.is_cached()).unwrap());
}

#[test]
fn persist_accepts_every_storage_level() {
    let s = int_session(vec![1]);
    for level in [
        StorageLevel::MEMORY_ONLY,
        StorageLevel::MEMORY_ONLY_2,
        StorageLevel::MEMORY_AND_DISK,
        StorageLevel::MEMORY_AND_DISK_2,
        StorageLevel::DISK_ONLY,
        StorageLevel::DISK_ONLY_2,
        StorageLevel::OFF_HEAP,
    ] {
        let df = s.table("t").persist(level);
        assert_eq!(block_on(df.storage_level()).unwrap(), level);
        assert_eq!(block_on(df.count()).unwrap(), 1);
        assert!(block_on(df.is_cached()).unwrap());
        block_on(df.unpersist()).unwrap();
        assert_eq!(block_on(df.storage_level()).unwrap(), StorageLevel::NONE);
    }
    assert_eq!(s.cache_reserved(), 0);
}

#[test]
fn the_cap_runs_uncached_not_failed() {
    let s = Session::builder()
        .config("krishiv.cache.maxBytes", "1b")
        .build()
        .unwrap();
    register(&s, "t", vec![1, 2]);
    let baseline = s.cache_reserved();
    let df = s.table("t").cache();
    // One byte cannot hold the snapshot: the query still succeeds, and the
    // registration stays (pending).
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![1, 2]);
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), baseline);
}

/// A running fill holds its batches in the `relation_cache` pool, so they
/// count against the memory limit while the query streams; a cancelled fill
/// gives them back and caches nothing.
#[test]
fn a_running_fill_is_charged_to_the_pool_and_released_on_cancel() {
    use futures::StreamExt as _;
    let s = int_session((0..4096).collect());
    let baseline = s.cache_reserved();
    let df = s.table("t").cache();
    let mut stream = block_on(df.execute_stream()).unwrap();
    let first = block_on(stream.next()).unwrap().unwrap();
    assert!(first.num_rows() > 0);
    assert!(
        s.cache_reserved() > baseline,
        "the buffered batch is not charged: {} <= {baseline}",
        s.cache_reserved()
    );
    // Dropped before the end: nothing stored, the bytes given back.
    drop(stream);
    assert_eq!(s.cache_reserved(), baseline);
    // Run to the end, the same bytes become the stored snapshot.
    assert_eq!(block_on(df.count()).unwrap(), 4096);
    assert!(block_on(df.is_cached()).unwrap());
    assert!(s.cache_reserved() > baseline);
    block_on(df.unpersist()).unwrap();
    assert_eq!(s.cache_reserved(), baseline);
}

/// A pool that refuses the fill's bytes (`krishiv.memory.limit`) leaves the
/// query uncached, never failed.
#[test]
fn a_refused_reservation_runs_uncached_not_failed() {
    let s = Session::builder()
        .config("krishiv.memory.limit", "1b")
        .build()
        .unwrap();
    register(&s, "t", (0..4096).collect());
    let df = s.table("t").cache();
    assert_eq!(block_on(df.count()).unwrap(), 4096);
    assert!(block_on(df.is_cached()).unwrap(), "registered, not filled");
    assert_eq!(s.cache_reserved(), 0);
}

#[test]
fn refresh_table_drops_only_that_tables_entries() {
    let s = int_session(vec![1]);
    register(&s, "u", vec![7]);
    let t = s.table("t").cache();
    let u = s.table("u").cache();
    assert_eq!(block_on(t.count()).unwrap(), 1);
    assert_eq!(block_on(u.count()).unwrap(), 1);
    // Replacing `t` drops its registration and keeps `u`'s.
    register(&s, "t", vec![2, 3]);
    assert!(!block_on(t.is_cached()).unwrap());
    assert!(block_on(u.is_cached()).unwrap());
    assert_eq!(ids(&block_on(t.collect()).unwrap()), vec![2, 3]);
    // REFRESH TABLE drops only that table's data and keeps its registration.
    let t = s.table("t").cache();
    assert_eq!(block_on(t.count()).unwrap(), 2);
    assert_eq!(relation_entries(&s), 2);
    block_on(s.sql("REFRESH TABLE t")).unwrap();
    assert!(block_on(t.is_cached()).unwrap());
    assert!(block_on(u.is_cached()).unwrap());
    assert_eq!(relation_entries(&s), 1, "only u's data stays");
    assert_eq!(ids(&block_on(u.collect()).unwrap()), vec![7]);
    assert_eq!(ids(&block_on(t.collect()).unwrap()), vec![2, 3]);
    assert_eq!(relation_entries(&s), 2, "the next action refilled t");
    // A missing table fails as in Spark, quoting the whole name.
    for sql in [
        "REFRESH TABLE nosuch",
        "REFRESH TABLE nosuchdb.t",
        "UNCACHE TABLE nosuchdb.t",
    ] {
        let e = block_on(s.sql(sql)).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound, "{sql}");
    }
    block_on(s.sql("UNCACHE TABLE IF EXISTS nosuchdb.t")).unwrap();
}

#[test]
fn clear_cache_and_uncache_table() {
    let s = int_session(vec![1]);
    let df = s.table("t").cache();
    assert_eq!(block_on(df.count()).unwrap(), 1);
    assert!(block_on(df.is_cached()).unwrap());
    block_on(s.sql("CLEAR CACHE")).unwrap();
    assert!(!block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
    // CLEAR CACHE dropped the registration: running the frame caches nothing.
    assert_eq!(block_on(df.count()).unwrap(), 1);
    assert!(!block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
    let df = df.cache();
    assert!(block_on(df.is_cached()).unwrap());
    block_on(s.sql("UNCACHE TABLE t")).unwrap();
    assert!(!block_on(df.is_cached()).unwrap());
    // `IF EXISTS` tolerates a missing table; without it, the table must exist.
    block_on(s.sql("UNCACHE TABLE IF EXISTS missing")).unwrap();
    let e = block_on(s.sql("UNCACHE TABLE missing")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
}

/// Every plan and expression form the cache key encodes: each query is
/// cached, run twice (the second action reads the entry) and must return
/// what the uncached query returns.
#[test]
fn every_plan_shape_caches_and_reads_back_the_same_rows() {
    let s = int_session(vec![1, 2, 3, 4]);
    register(&s, "u", vec![3, 4, 5]);
    for sql in [
        "SELECT id, id + 1 AS next, CAST(id AS BIGINT) AS wide FROM t",
        "SELECT * FROM t WHERE id > 1 AND id NOT IN (SELECT id FROM u)",
        "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id)",
        "SELECT t.id, (SELECT max(id) FROM u) AS top FROM t",
        "SELECT t.id AS a, u.id AS b FROM t JOIN u ON t.id = u.id",
        "SELECT t.id FROM t LEFT JOIN u USING (id)",
        "SELECT t.id FROM t CROSS JOIN u",
        "SELECT id % 2 AS parity, count(*) AS n, count(DISTINCT id) AS d FROM t GROUP BY id % 2 HAVING count(*) > 1",
        "SELECT id, count(*) FILTER (WHERE id > 2) FROM t GROUP BY ROLLUP (id)",
        "SELECT id, count(*) FROM t GROUP BY CUBE (id)",
        "SELECT id, count(*) FROM t GROUP BY GROUPING SETS ((id), ())",
        "SELECT id, sum(id) OVER (PARTITION BY id % 2 ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        "SELECT id, rank() OVER w FROM t WINDOW w AS (ORDER BY id DESC NULLS LAST)",
        "SELECT id FROM t ORDER BY id DESC LIMIT 2",
        "SELECT id FROM t UNION ALL SELECT id FROM u",
        "SELECT id FROM t INTERSECT SELECT id FROM u",
        "SELECT id FROM t EXCEPT SELECT id FROM u",
        "WITH c AS (SELECT id FROM t) SELECT c.id FROM c JOIN c AS d ON c.id = d.id",
        "SELECT * FROM VALUES (1, 'a'), (2, 'b') AS v(n, s)",
        "SELECT CASE WHEN id > 2 THEN 'big' ELSE 'small' END AS size FROM t",
        "SELECT transform(array(id, id + 1), x -> x * 2) AS doubled FROM t",
        "SELECT named_struct('a', id, 'b', id + 1).a AS a FROM t",
        "SELECT * FROM t PIVOT (count(id) FOR id IN (1, 2))",
        "SELECT * FROM (SELECT 1 AS a, 2 AS b) UNPIVOT (v FOR k IN (a, b))",
        "SELECT * FROM range(3)",
        "SELECT * EXCEPT (id) , 1 AS one FROM t",
        "SELECT DATE '2024-01-02' AS d, TIMESTAMP '2024-01-02 03:04:05' AS ts, t.id FROM t",
        "SELECT builtin.abs(id) FROM t",
    ] {
        let uncached = block_on(
            block_on(s.sql(sql))
                .unwrap_or_else(|e| panic!("{sql}: {e:?}"))
                .collect(),
        )
        .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
        let df = block_on(s.sql(sql)).unwrap().cache();
        for _ in 0..2 {
            let got = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
            assert_eq!(rendered(&got), rendered(&uncached), "{sql}");
        }
        assert!(block_on(df.is_cached()).unwrap(), "{sql}");
        block_on(df.unpersist()).unwrap();
    }
    assert_eq!(s.cache_reserved(), 0);
}

/// A result with two columns of one name is not cached (a scan finds its
/// columns by name), and runs uncached with the right values.
#[test]
fn results_with_duplicate_column_names_run_uncached() {
    let s = int_session(vec![1, 2, 3]);
    register(&s, "u", vec![2, 3, 4]);
    let df = block_on(s.sql("SELECT t.id, u.id FROM t JOIN u ON t.id = u.id"))
        .unwrap()
        .cache();
    for _ in 0..2 {
        let rows = block_on(df.collect()).unwrap();
        let (left, right): (Vec<_>, Vec<_>) = rows
            .iter()
            .map(|b| {
                (
                    b.column(0).as_primitive::<Int32Type>().values().to_vec(),
                    b.column(1).as_primitive::<Int32Type>().values().to_vec(),
                )
            })
            .unzip();
        assert_eq!(left.concat(), right.concat());
        assert_eq!(left.concat().len(), 2);
    }
    // Registered, never filled.
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
}

#[test]
fn cache_table_sql_reads_the_relation_into_the_cache() {
    let s = int_session(vec![1, 2]);
    assert_eq!(s.cache_reserved(), 0);
    block_on(s.sql("CACHE TABLE t")).unwrap();
    assert!(s.cache_reserved() > 0);
    block_on(s.sql("UNCACHE TABLE t")).unwrap();
    assert_eq!(s.cache_reserved(), 0);
    // `LAZY` only registers; the first read fills it.
    block_on(s.sql("CACHE LAZY TABLE t")).unwrap();
    assert!(block_on(s.table("t").is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
    assert_eq!(block_on(s.table("t").count()).unwrap(), 2);
    let lazy = s.cache_reserved();
    assert!(lazy > 0);
    // `AS query` makes the temporary view first, then caches it eagerly.
    assert_eq!(relation_entries(&s), 1);
    block_on(s.sql("CACHE TABLE doubled AS SELECT id * 2 AS d FROM t")).unwrap();
    assert!(block_on(s.table("doubled").is_cached()).unwrap());
    assert_eq!(relation_entries(&s), 2, "the eager count filled doubled");
    block_on(s.sql("CLEAR CACHE")).unwrap();
    assert_eq!(s.cache_reserved(), 0);
    // Refreshing a function is a no-op; refreshing a path drops file entries.
    block_on(s.sql("REFRESH FUNCTION f")).unwrap();
    block_on(s.sql("REFRESH '/tmp/path'")).unwrap();
    // `IDENTIFIER(...)` names are evaluated first, so they match entries like any other spelling.
    block_on(s.sql("CACHE TABLE IDENTIFIER('t')")).unwrap();
    block_on(s.sql("REFRESH TABLE IDENTIFIER('t')")).unwrap();
    block_on(s.sql("UNCACHE TABLE IDENTIFIER('t')")).unwrap();
    assert_eq!(s.cache_reserved(), 0);
}

#[test]
fn hidden_cache_names_are_reserved() {
    let s = Session::new().unwrap();
    let e = s
        .register_batches("__krishiv_cache_0", schema(), vec![])
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
    // Reads of one behave exactly as missing tables.
    let missing = block_on(s.sql("SELECT * FROM nope")).unwrap_err();
    let e = block_on(s.sql("SELECT * FROM `__krishiv_cache_0`")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert_eq!(e.sql_state(), missing.sql_state());
}

fn part_file(dir: &tempfile::TempDir, sub: &str) -> std::path::PathBuf {
    let root = dir.path().join(sub);
    let mut files: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
        .collect();
    files.sort();
    assert_eq!(files.len(), 1, "one part file per write: {files:?}");
    files.pop().unwrap()
}

#[test]
fn engine_writes_invalidate_but_external_ones_stay_invisible() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let data = data.to_str().unwrap().to_owned();
    let other = dir.path().join("other");
    let other = other.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    register(&s, "t", vec![1, 2]);
    block_on(
        s.table("t")
            .write()
            .mode("overwrite")
            .format("parquet")
            .save(&data),
    )
    .unwrap();
    let df = s.read().parquet([data.clone()]).cache();
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![1, 2]);
    assert!(block_on(df.is_cached()).unwrap());
    // The engine's own overwrite of the cached path invalidates: the next
    // action lists the files again.
    register(&s, "t", vec![3]);
    block_on(
        s.table("t")
            .write()
            .mode("overwrite")
            .format("parquet")
            .save(&data),
    )
    .unwrap();
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![3]);
    assert!(block_on(df.is_cached()).unwrap());
    // An external overwrite of the cached path stays invisible, as in
    // Spark: no listing, no version, no invalidation. The bytes are
    // replaced behind the engine's back (a copy of another valid part
    // file), so only the cache decides what the next action reads.
    register(&s, "t", vec![4, 5, 6]);
    block_on(
        s.table("t")
            .write()
            .mode("overwrite")
            .format("parquet")
            .save(&other),
    )
    .unwrap();
    // The write above dropped the entry, so the fresh rows cache again first.
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![3]);
    let src = part_file(&dir, "other");
    let dst = part_file(&dir, "data");
    std::fs::copy(src, dst).unwrap();
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![3]);
}

#[test]
fn refresh_path_drops_file_entries() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let data = data.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    register(&s, "t", vec![1]);
    block_on(
        s.table("t")
            .write()
            .mode("overwrite")
            .format("parquet")
            .save(&data),
    )
    .unwrap();
    let df = s.read().parquet([data.clone()]).cache();
    assert_eq!(block_on(df.count()).unwrap(), 1);
    assert!(block_on(df.is_cached()).unwrap());
    assert!(s.cache_reserved() > 0);
    block_on(s.sql(&format!("REFRESH '{data}'"))).unwrap();
    // The data goes, the registration stays.
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
}

#[test]
fn views_and_dropped_tables_drop_their_entries() {
    let s = int_session(vec![1, 2]);
    block_on(s.sql("CREATE TEMPORARY VIEW v AS SELECT id FROM t")).unwrap();
    let df = s.table("v").cache();
    assert_eq!(block_on(df.count()).unwrap(), 2);
    assert!(block_on(df.is_cached()).unwrap());
    block_on(s.sql("CREATE OR REPLACE TEMPORARY VIEW v AS SELECT id FROM t WHERE id > 1")).unwrap();
    assert!(!block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
    assert_eq!(block_on(s.table("v").count()).unwrap(), 1);
    let again = s.table("v").cache();
    assert_eq!(block_on(again.count()).unwrap(), 1);
    block_on(s.sql("DROP VIEW v")).unwrap();
    assert_eq!(s.cache_reserved(), 0);
    let again = s.table("t").cache();
    assert_eq!(block_on(again.count()).unwrap(), 2);
    assert!(s.drop_temp_view("t"));
    assert_eq!(s.cache_reserved(), 0);
}

#[test]
fn a_zero_cap_disables_caching_even_for_an_empty_result() {
    let s = int_session(vec![1, 2]);
    block_on(s.sql("SET krishiv.cache.maxBytes = 0")).unwrap();
    let df = s.table("t").cache();
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![1, 2]);
    let empty = s.table("t").filter(col("id").gt(lit(100))).cache();
    assert_eq!(block_on(empty.count()).unwrap(), 0);
    assert_eq!(s.cache_reserved(), 0);
    assert!(
        block_on(df.is_cached()).unwrap(),
        "registered, never filled"
    );
    assert_eq!(ids(&block_on(df.collect()).unwrap()), vec![1, 2]);
    assert_eq!(s.cache_reserved(), 0, "nothing filled under a zero cap");
}

#[test]
fn a_fill_that_started_before_an_invalidation_is_not_stored() {
    let s = int_session(vec![1, 2]);
    let df = s.table("t").cache();
    let mut stream = block_on(df.execute_stream()).unwrap();
    // The fill is under way when the table is refreshed.
    block_on(s.sql("REFRESH TABLE t")).unwrap();
    while block_on(futures::StreamExt::next(&mut stream)).is_some() {}
    drop(stream);
    assert_eq!(s.cache_reserved(), 0);
    // The registration is pending again: the next action fills it.
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(block_on(df.count()).unwrap(), 2);
    assert!(s.cache_reserved() > 0);
}

#[test]
fn a_cached_frame_can_be_written_from_its_snapshot() {
    let s = int_session(vec![1, 2]);
    let df = s.table("t").cache();
    assert_eq!(block_on(df.count()).unwrap(), 2);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out");
    block_on(df.write().format("csv").save(path.to_str().unwrap())).unwrap();
    assert!(path.exists());
}

/// A snapshot computed under one resolution configuration (ANSI mode) is
/// never served under another: the registration stays (Spark's `isCached`
/// answers for the plan), and the next action refills it. A query-clock
/// column shows which rows were served.
#[test]
fn a_configuration_change_does_not_serve_the_old_snapshot() {
    let s = int_session(vec![1, 2]);
    block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    let df = block_on(s.sql("SELECT id, current_timestamp() AS ts FROM t"))
        .unwrap()
        .cache();
    let first = rendered(&block_on(df.collect()).unwrap());
    assert_eq!(rendered(&block_on(df.collect()).unwrap()), first);
    block_on(s.sql("SET spark.sql.ansi.enabled = true")).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    let second = rendered(&block_on(df.collect()).unwrap());
    assert_ne!(
        second, first,
        "the snapshot of the other configuration was served"
    );
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(rendered(&block_on(df.collect()).unwrap()), second);
}

/// `persist` registers at once (Spark's `cacheQuery`): `is_cached` and
/// `storage_level` answer from the call on, before any action, and nothing
/// is materialized until one runs.
#[test]
fn registration_is_eager_and_materialization_lazy() {
    let s = int_session(vec![1, 2]);
    let uncached = s.table("t");
    assert_eq!(
        block_on(uncached.storage_level()).unwrap(),
        StorageLevel::NONE
    );
    let df = s.table("t").cache();
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(
        block_on(df.storage_level()).unwrap(),
        StorageLevel::MEMORY_AND_DISK
    );
    // The registry answers for the plan, not the frame.
    assert!(block_on(uncached.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0, "nothing ran yet");
    assert_eq!(block_on(df.count()).unwrap(), 2);
    assert!(s.cache_reserved() > 0);
}

/// A second `persist` of a cached plan keeps the first level (Spark logs
/// "Asked to cache already cached data."); `NONE` is a no-op.
#[test]
fn a_second_persist_keeps_the_first_level() {
    let s = int_session(vec![1]);
    let df = s.table("t").persist(StorageLevel::DISK_ONLY);
    let again = s.table("t").persist(StorageLevel::MEMORY_ONLY);
    assert_eq!(
        block_on(again.storage_level()).unwrap(),
        StorageLevel::DISK_ONLY
    );
    assert_eq!(
        block_on(df.storage_level()).unwrap(),
        StorageLevel::DISK_ONLY
    );
    let none = s
        .table("t")
        .filter(col("id").gt(lit(0)))
        .persist(StorageLevel::NONE);
    assert!(!block_on(none.is_cached()).unwrap());
    assert_eq!(block_on(none.storage_level()).unwrap(), StorageLevel::NONE);
}

#[test]
fn unpersisting_an_uncached_frame_is_a_no_op() {
    let s = int_session(vec![1]);
    let df = s.table("t");
    let df = block_on(df.unpersist()).unwrap();
    assert!(!block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
    // Unpersist drops the registration, filled or not.
    let cached = s.table("t").cache();
    block_on(cached.unpersist()).unwrap();
    assert!(!block_on(cached.is_cached()).unwrap());
    assert_eq!(block_on(cached.count()).unwrap(), 1);
    assert_eq!(s.cache_reserved(), 0);
}

/// `cache()`, and `CACHE TABLE` without a level, use
/// `spark.sql.defaultCacheStorageLevel`; `CACHE TABLE ... OPTIONS` names a
/// level in any letter case.
#[test]
fn the_default_level_and_the_cache_table_option() {
    let s = int_session(vec![1]);
    register(&s, "u", vec![2]);
    register(&s, "w", vec![3]);
    block_on(s.sql("SET spark.sql.defaultCacheStorageLevel = DISK_ONLY")).unwrap();
    let df = s.table("t").cache();
    assert_eq!(
        block_on(df.storage_level()).unwrap(),
        StorageLevel::DISK_ONLY
    );
    block_on(s.sql("CACHE LAZY TABLE u")).unwrap();
    assert_eq!(
        block_on(s.table("u").storage_level()).unwrap(),
        StorageLevel::DISK_ONLY
    );
    block_on(s.sql("CACHE LAZY TABLE w OPTIONS ('storageLevel' = 'memory_only_2')")).unwrap();
    assert_eq!(
        block_on(s.table("w").storage_level()).unwrap(),
        StorageLevel::MEMORY_ONLY_2
    );
    assert_eq!(s.cache_reserved(), 0, "LAZY reads nothing");
}

/// A failed eager `CACHE TABLE v AS query` unregisters and drops the view it
/// made, then raises the query's error (Spark's `CacheTableAsSelectExec`).
#[test]
fn a_failed_eager_cache_table_drops_its_view() {
    let s = int_session(vec![1]);
    block_on(s.sql("SET spark.sql.ansi.enabled = true")).unwrap();
    let e = block_on(s.sql("CACHE TABLE v AS SELECT id / 0 AS q FROM t")).unwrap_err();
    assert_eq!(e.error_class().name(), "DIVIDE_BY_ZERO", "{e}");
    let e = block_on(s.table("v").count()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert_eq!(s.cache_reserved(), 0);
}

/// A frame of a command is not cached (Spark logs "Asked to cache a plan
/// that is inapplicable for caching").
#[test]
fn command_frames_are_not_registered() {
    let s = int_session(vec![1]);
    let df = block_on(s.sql("SET krishiv.cache.maxBytes = 1g"))
        .unwrap()
        .cache();
    assert!(!block_on(df.is_cached()).unwrap());
}

/// Only one action fills a key: a second action while the first fills runs
/// the plan itself and stores nothing.
#[test]
fn one_fill_per_key_while_another_action_fills() {
    use futures::StreamExt as _;
    let s = int_session((0..4096).collect());
    let df = s.table("t").cache();
    let mut first = block_on(df.execute_stream()).unwrap();
    assert!(block_on(first.next()).unwrap().unwrap().num_rows() > 0);
    let holding = s.cache_reserved();
    assert!(holding > 0);
    // The second action does not wait and does not fill.
    assert_eq!(block_on(df.count()).unwrap(), 4096);
    assert_eq!(s.cache_reserved(), holding, "the second action filled too");
    // Dropping the first fill leaves the registration pending.
    drop(first);
    assert_eq!(s.cache_reserved(), 0);
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(block_on(df.count()).unwrap(), 4096);
    assert_eq!(s.cache_reserved(), holding.max(s.cache_reserved()));
    assert!(s.cache_reserved() > 0);
}

/// Many concurrent actions over one registered plan store one snapshot.
#[test]
fn concurrent_actions_store_one_snapshot() {
    let one = int_session((0..2048).collect());
    let df = one.table("t").cache();
    assert_eq!(block_on(df.count()).unwrap(), 2048);
    let single = one.cache_reserved();
    let s = int_session((0..2048).collect());
    let df = s.table("t").filter(col("id").gt_eq(lit(0))).cache();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let df = df.clone();
            std::thread::spawn(move || block_on(df.count()).unwrap())
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), 2048);
    }
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(block_on(df.count()).unwrap(), 2048);
    assert!(s.cache_reserved() > 0);
    assert!(
        s.cache_reserved() <= single * 2,
        "more than one snapshot: {} vs {single}",
        s.cache_reserved()
    );
}

/// The most rows one source produced (a scan, or the output side of a
/// breaker): a read of the 4096-row source shows 4096.
fn source_rows(metrics: &krishiv_exec::QueryMetrics) -> u64 {
    metrics
        .operators
        .values()
        .filter(|o| format!("{:?}", o.role) == "Source")
        .map(|o| o.metrics.rows_out)
        .max()
        .unwrap_or(0)
}

/// Design P3-T083 3.6: re-use avoids recomputation, shown by operator
/// counters. Once filled, an action over the cached plan and an action over
/// a derived plan read only the snapshot's row, never the source's 4096.
#[test]
fn reuse_reads_the_snapshot_not_the_source() {
    let s = int_session((0..4096).collect());
    let cached = s
        .table("t")
        .select([lit(1).alias("one")])
        .distinct()
        .cache();
    let derived = cached.select([(col("one") + lit(1)).alias("two")]);
    // The derived frame's first action fills the cached plan (its own query
    // reads the source), then reads the snapshot.
    let (rows, metrics) = block_on(derived.collect_with_metrics()).unwrap();
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(source_rows(&metrics), 1);
    let (_, metrics) = block_on(cached.collect_with_metrics()).unwrap();
    assert_eq!(
        source_rows(&metrics),
        1,
        "the cached plan reread the source"
    );
    let (_, metrics) = block_on(derived.collect_with_metrics()).unwrap();
    assert_eq!(
        source_rows(&metrics),
        1,
        "the derived plan reread the source"
    );
    // Unpersisted, the derived plan reads the source again.
    block_on(cached.unpersist()).unwrap();
    let (_, metrics) = block_on(derived.collect_with_metrics()).unwrap();
    assert_eq!(source_rows(&metrics), 4096);
}

/// An engine write to a cached permanent table drops the data and keeps the
/// registration (Spark's recache): `is_cached` stays true and the next
/// action refills with the new rows.
#[test]
fn invalidation_keeps_the_registration() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .config("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .build()
        .unwrap();
    block_on(s.sql("CREATE TABLE p (id INT) USING parquet")).unwrap();
    block_on(s.sql("INSERT INTO p VALUES (1)")).unwrap();
    let df = s.table("p").cache();
    assert_eq!(block_on(df.count()).unwrap(), 1);
    let filled = s.cache_reserved();
    assert!(filled > 0);
    block_on(s.sql("INSERT INTO p VALUES (2)")).unwrap();
    assert!(block_on(df.is_cached()).unwrap());
    assert_eq!(s.cache_reserved(), 0);
    assert_eq!(block_on(df.count()).unwrap(), 2);
    assert!(s.cache_reserved() > 0, "the next action refilled it");
    assert_eq!(relation_entries(&s), 1);
}

/// The relation cache's filled entries, from `system.caches`.
fn relation_entries(s: &Session) -> i64 {
    let rows = block_on(
        block_on(s.sql("SELECT entries FROM system.caches WHERE cache = 'relation'"))
            .unwrap()
            .collect(),
    )
    .unwrap();
    rows[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0)
}
