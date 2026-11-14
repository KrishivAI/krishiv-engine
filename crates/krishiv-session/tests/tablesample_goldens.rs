// SPDX-License-Identifier: Apache-2.0
//! `TABLESAMPLE` rows against `tests/fixtures/spark-4.2.0/tablesample.json`
//! (a real Spark 4.2.0 run at `local[1]`, generator
//! `tests/scripts/gen_tablesample.py`).
//!
//! Spark draws one `XORShiftRandom(seed + i)` value per row of input
//! partition `i`, so its rows depend on how the input is split. A session
//! with one compute thread splits `range(0, 1000)`, `VALUES` and a table of
//! one batch into one partition, as `local[1]` does; `range(0, 1000, 1, 4)`
//! is four slices of 250 rows in both engines, whatever the thread count;
//! `UNION ALL` numbers its inputs' partitions one after another, as Spark's
//! `UnionRDD`. The fixture's Parquet tables are registered as in-memory
//! tables of the same rows in the same order (one partition, as Spark's one
//! small file), and again through a Parquet file Krishiv wrote; `pq3`
//! (`repartition(3)`'s round-robin rows in three files) is not reproduced.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_exec::{ExecutorConfig, TokioExecutor};
use krishiv_session::Session;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/tablesample.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn session(threads: usize) -> Session {
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: threads,
        io_threads: 1,
        blocking_threads: threads,
        io_blocking_threads: 2,
    })
    .unwrap();
    let s = Session::builder().executor(executor).build().unwrap();
    for sql in [
        "CREATE TEMPORARY VIEW t AS SELECT id FROM range(0, 1000)",
        "CREATE TEMPORARY VIEW t4 AS SELECT id FROM range(0, 1000, 1, 4)",
        "CREATE TEMPORARY VIEW big AS SELECT id FROM range(0, 2000000)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    // `pq1`: `id`, `id * 2 AS b`, `cast(id AS string) AS s`, in id order.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..1000)),
            Arc::new(Int64Array::from_iter_values((0..1000).map(|i| i * 2))),
            Arc::new(StringArray::from_iter_values(
                (0..1000).map(|i| i.to_string()),
            )),
        ],
    )
    .unwrap();
    s.register_batches("pq1", schema, vec![batch]).unwrap();
    s
}

/// The first column of `sql`'s rows, sorted (as the generator sorts them).
fn ids(s: &Session, sql: &str) -> Vec<i64> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0);
            match c.data_type() {
                DataType::Int32 => c
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .iter()
                    .map(|v| i64::from(*v))
                    .collect::<Vec<_>>(),
                _ => c.as_primitive::<Int64Type>().values().to_vec(),
            }
        })
        .collect();
    out.sort_unstable();
    out
}

fn want(case: &Value) -> Vec<i64> {
    case["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

/// Every case but `pq3`'s, at one compute thread.
#[test]
fn one_thread_rows_match_spark() {
    let s = session(1);
    let doc = fixture();
    let mut checked = 0;
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let sql = case["sql"].as_str().unwrap();
        if name.starts_with("parquet3files") {
            continue;
        }
        let got = ids(&s, sql);
        assert_eq!(got.len() as u64, case["count"].as_u64().unwrap(), "{sql}");
        if name == "parquet_big_1pct_seed7" {
            let first: Vec<i64> = case["ids_first50"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            assert_eq!(got[..50], first[..], "{sql}");
            let joined = got.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
            let digest = Sha256::digest(joined.as_bytes());
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(
                hex,
                case["ids_sha256_comma_joined"].as_str().unwrap(),
                "{sql}"
            );
        } else {
            assert_eq!(got, want(case), "{sql}");
        }
        checked += 1;
    }
    assert_eq!(checked, 37);
}

/// `range(0, 1000, 1, 4)` is split as Spark splits it at any thread count.
#[test]
fn four_slice_range_matches_spark_at_four_threads() {
    let s = session(4);
    let doc = fixture();
    for case in doc["cases"].as_array().unwrap() {
        if case["partitions"].as_u64() == Some(4) {
            let sql = case["sql"].as_str().unwrap();
            assert_eq!(ids(&s, sql), want(case), "{sql}");
        }
    }
}

/// An unseeded sample draws its seed once per physical plan, so one run
/// keeps one stream; of one row it keeps the row or not.
#[test]
fn unseeded_samples_run() {
    let s = session(1);
    for _ in 0..5 {
        let n = ids(&s, "SELECT * FROM (SELECT 1 a) TABLESAMPLE (50 PERCENT)").len();
        assert!(n <= 1);
        let n = ids(&s, "SELECT id FROM t TABLESAMPLE (10 PERCENT)").len();
        assert!((40..200).contains(&n), "{n}");
    }
    // `REPEATABLE (abc)` is a table alias with a column alias, unseeded.
    let df = block_on(s.sql("SELECT * FROM t TABLESAMPLE (10 PERCENT) REPEATABLE (abc)")).unwrap();
    assert_eq!(block_on(df.schema()).unwrap().field(0).name(), "abc");
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

/// The `pq1` cases through a real Parquet file scan: one small file is one
/// partition, as in Spark.
#[test]
fn parquet_file_rows_match_spark() {
    let s = session(1);
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("pq1");
    let path = path.to_str().unwrap();
    let df = block_on(s.sql("SELECT id, id * 2 AS b, cast(id AS string) AS s FROM range(0, 1000)"))
        .unwrap();
    block_on(df.write().parquet(path)).unwrap();
    run(
        &s,
        &format!(
            "CREATE TEMPORARY VIEW pqf(id BIGINT, b BIGINT, s STRING) USING parquet OPTIONS (path '{path}')"
        ),
    );
    let doc = fixture();
    let mut checked = 0;
    for case in doc["cases"].as_array().unwrap() {
        if !case["name"].as_str().unwrap().starts_with("parquet1") {
            continue;
        }
        let sql = case["sql"].as_str().unwrap().replace("pq1", "pqf");
        assert_eq!(ids(&s, &sql), want(case), "{sql}");
        checked += 1;
    }
    assert_eq!(checked, 4);
}

/// A table of six files cut into several partitions: a file scan under a
/// sample reads each partition whole, so `REPEATABLE` keeps its rows at any
/// thread count and on every run.
#[test]
fn repeatable_rows_hold_across_threads_and_runs_over_files() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("multi");
    let path = path.to_str().unwrap().to_owned();
    let writer = session(1);
    for i in 0..6 {
        let df = block_on(writer.sql(&format!(
            "SELECT id, cast(id AS string) AS s FROM range({}, {})",
            i * 20_000,
            (i + 1) * 20_000
        )))
        .unwrap();
        block_on(df.write().mode("append").parquet(&path)).unwrap();
    }
    let sql = "SELECT id FROM multi TABLESAMPLE (10 PERCENT) REPEATABLE (7)";
    let view = format!(
        "CREATE TEMPORARY VIEW multi(id BIGINT, s STRING) USING parquet OPTIONS (path '{path}')"
    );
    // One partition (the six small files packed together) ...
    let one = session(1);
    run(&one, &view);
    let packed = ids(&one, sql);
    // ... and one per file or less: each partition draws its own stream.
    let split = |threads| {
        let s = session(threads);
        run(&s, "SET spark.sql.files.maxPartitionBytes=65536");
        run(&s, "SET spark.sql.files.openCostInBytes=0");
        run(&s, &view);
        s
    };
    let serial = ids(&split(1), sql);
    assert_ne!(serial, packed, "the files were not cut into partitions");
    let four = split(4);
    for _ in 0..3 {
        assert_eq!(ids(&four, sql), serial);
    }
}

/// `range(0, 1000)` at four threads is four slices, each read by one driver:
/// the same rows on every run.
#[test]
fn repeatable_rows_hold_across_runs_at_four_threads() {
    let s = session(4);
    let sql = "SELECT id FROM t TABLESAMPLE (10 PERCENT) REPEATABLE (7)";
    let first = ids(&s, sql);
    for _ in 0..3 {
        assert_eq!(ids(&s, sql), first);
    }
}

/// One DataFrame of an unseeded sample keeps its rows across actions, as a
/// Spark Dataset keeps its executed plan; a new statement draws again.
#[test]
fn an_unseeded_dataframe_keeps_its_rows() {
    let s = session(1);
    let df = block_on(s.sql("SELECT id FROM t TABLESAMPLE (10 PERCENT)")).unwrap();
    let rows = |df: &krishiv_session::DataFrame| {
        let mut out: Vec<i64> = block_on(df.collect())
            .unwrap()
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect();
        out.sort_unstable();
        out
    };
    let first = rows(&df);
    assert_eq!(rows(&df), first);
    // Twenty statements: some draw another seed (1 in 1000 each would not).
    let other = (0..20).any(|_| {
        let df = block_on(s.sql("SELECT id FROM t TABLESAMPLE (10 PERCENT)")).unwrap();
        rows(&df) != first
    });
    assert!(other);
}

/// A sampled plan survives the plan codec: with every plan round-tripped
/// through it, the rows are Spark's.
#[test]
fn a_sampled_plan_round_trips_through_the_codec() {
    let s = session(1);
    run(&s, "SET krishiv.exec.debug.planRoundTrip=true");
    let doc = fixture();
    for case in doc["cases"].as_array().unwrap() {
        if case["name"] == "view_10pct_seed7" || case["partitions"].as_u64() == Some(4) {
            let sql = case["sql"].as_str().unwrap();
            assert_eq!(ids(&s, sql), want(case), "{sql}");
        }
    }
}
