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
//! small file); `pq3` (three files whose packing order Spark decides) is
//! not reproduced.
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
