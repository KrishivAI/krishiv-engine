// SPDX-License-Identifier: Apache-2.0
//! Apache's Rust Spark Connect client (`apache-spark-connect`) against a running Krishiv Connect
//! server (P3-T089, PR-40). The tests are `#[ignore]`d so a plain `cargo test` needs no server;
//! CI starts `krishiv server` and runs them with `--ignored`. `KRISHIV_CONNECT_URL` overrides the
//! default `sc://127.0.0.1:15002`. Expected values are Spark 4.x's.
use spark_connect::functions::{col, count, sum};
use spark_connect::lit;
use spark_connect::session::SparkSession;

fn session() -> SparkSession {
    let url =
        std::env::var("KRISHIV_CONNECT_URL").unwrap_or_else(|_| "sc://127.0.0.1:15002".into());
    SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("connect to the Krishiv server")
}

#[test]
#[ignore = "needs a running krishiv server"]
fn session_reports_a_spark_version() {
    let version = session().version().expect("version");
    // The server's default session profile is Spark 4.3 (`krishiv-connect-server/tests/rpcs.rs`).
    assert_eq!(version, "4.3.0");
}

#[test]
#[ignore = "needs a running krishiv server"]
fn range_select_filter_count() {
    let spark = session();
    let df = spark
        .range(100)
        .unwrap()
        .filter(col("id").gt(lit(89)))
        .select([col("id")]);
    assert_eq!(df.count().unwrap(), 10);
}

#[test]
#[ignore = "needs a running krishiv server"]
fn sql_group_by_collect() {
    let spark = session();
    let rows = spark
        .sql("SELECT id % 3 AS k, count(*) AS n FROM range(10) GROUP BY id % 3 ORDER BY k")
        .unwrap()
        .collect()
        .unwrap();
    let got: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| {
            (
                r.get(0).unwrap().as_i64().unwrap(),
                r.get(1).unwrap().as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(got, vec![(0, 4), (1, 3), (2, 3)]);
}

#[test]
#[ignore = "needs a running krishiv server"]
fn dataframe_aggregate() {
    let spark = session();
    let rows = spark
        .range(10)
        .unwrap()
        .agg([sum(col("id")), count(col("id"))])
        .collect()
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get(0).unwrap().as_i64(), Some(45));
    assert_eq!(rows[0].get(1).unwrap().as_i64(), Some(10));
}

#[test]
#[ignore = "needs a running krishiv server"]
fn record_batches_carry_every_row() {
    let spark = session();
    let batches = spark
        .range(25_000)
        .unwrap()
        .collect_record_batches()
        .unwrap();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 25_000);
    // `spark.sql.execution.arrow.maxRecordsPerBatch` defaults to 10,000.
    assert!(
        batches.iter().all(|b| b.num_rows() <= 10_000),
        "a batch exceeds the cap"
    );
}

#[test]
#[ignore = "needs a running krishiv server"]
fn to_arrow_returns_an_ipc_stream() {
    let spark = session();
    let bytes = spark.range(5).unwrap().to_arrow().unwrap();
    assert!(!bytes.is_empty());
    // The client has no IPC reader here, so the contents are checked on the same plan's rows.
    let ids: Vec<Option<i64>> = spark
        .range(5)
        .unwrap()
        .collect()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap().as_i64())
        .collect();
    assert_eq!(ids, [Some(0), Some(1), Some(2), Some(3), Some(4)]);
}

#[test]
#[ignore = "needs a running krishiv server"]
fn create_table_as_select_and_read() {
    let spark = session();
    spark
        .sql("DROP TABLE IF EXISTS rust_client_ctas")
        .unwrap()
        .collect()
        .unwrap();
    spark
        .sql("CREATE TABLE rust_client_ctas AS SELECT id FROM range(7)")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        spark
            .sql("SELECT * FROM rust_client_ctas")
            .unwrap()
            .count()
            .unwrap(),
        7
    );
    spark
        .sql("DROP TABLE rust_client_ctas")
        .unwrap()
        .collect()
        .unwrap();
}

/// `DataFrameWriter.saveAsTable` sends a `WriteOperation` command, which the coverage table lists
/// as Missing (owner P3-T023). The `blocked_` prefix keeps it out of CI's run until that lands;
/// remove the prefix and the skip in `nightly.yml` then.
#[test]
#[ignore = "needs a running krishiv server"]
fn blocked_t023_save_as_table() {
    let spark = session();
    spark
        .sql("DROP TABLE IF EXISTS rust_client_t")
        .unwrap()
        .collect()
        .unwrap();
    spark
        .range(7)
        .unwrap()
        .write()
        .mode("overwrite")
        .save_as_table("rust_client_t")
        .unwrap();
    assert_eq!(
        spark
            .sql("SELECT * FROM rust_client_t")
            .unwrap()
            .count()
            .unwrap(),
        7
    );
    spark
        .sql("DROP TABLE rust_client_t")
        .unwrap()
        .collect()
        .unwrap();
}

/// Spark analyses `spark.sql(..)` eagerly, so the error may surface at `sql` or at `collect`.
#[test]
#[ignore = "needs a running krishiv server"]
fn a_bad_query_is_an_error_not_a_hang() {
    let spark = session();
    let outcome = spark
        .sql("SELECT * FROM no_such_table_rc")
        .and_then(|df| df.collect());
    let err = outcome.expect_err("an unknown table must fail");
    assert_eq!(err.error_class, "TABLE_OR_VIEW_NOT_FOUND");
}
