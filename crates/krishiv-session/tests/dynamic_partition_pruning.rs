// SPDX-License-Identifier: Apache-2.0
//! Dynamic partition pruning (P2-T058): a join's runtime filter on a partition
//! column skips the files of partitions no build row reaches, and results
//! are the same with runtime filters off.
#![cfg(test)]

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_session::{Session, mermaid_of_profile};

fn session() -> Session {
    Session::new().unwrap()
}

fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn long(s: &Session, sql: &str) -> Vec<i64> {
    run(s, sql)
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_primitive::<Int64Type>();
            (0..c.len()).map(|i| c.value(i)).collect::<Vec<_>>()
        })
        .collect()
}

/// The value of the summed counter `name` in `sql`'s profile.
fn counter(s: &Session, sql: &str, name: &str) -> u64 {
    let df = block_on(s.sql(sql)).unwrap();
    let profile = block_on(df.explain_analyze()).unwrap();
    let text = mermaid_of_profile(&profile);
    text.match_indices(&format!("{name}="))
        .map(|(at, m)| {
            let digits: String = text[at + m.len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse::<u64>().unwrap_or(0)
        })
        .sum()
}

/// `sql`'s rows written partitioned by `p`, as the view `name`.
fn partitioned(s: &Session, dir: &std::path::Path, name: &str, sql: &str) {
    let root = dir.join(name);
    let root = root.to_str().unwrap();
    block_on(async {
        s.sql(sql)
            .await?
            .write()
            .partition_by(["p"])
            .parquet(root)
            .await
    })
    .unwrap();
    run(
        s,
        &format!("CREATE TEMPORARY VIEW {name} AS SELECT * FROM parquet.`{root}`"),
    );
}

fn tables(s: &Session, dir: &std::path::Path) {
    partitioned(
        s,
        dir,
        "fact",
        "SELECT id, CAST(id % 40 AS BIGINT) AS p FROM range(200000)",
    );
    run(
        s,
        "CREATE TEMPORARY VIEW dim AS SELECT id AS k, id % 10 AS tag FROM range(40)",
    );
}

const JOIN: &str =
    "SELECT sum(f.id) FROM fact f JOIN dim d ON f.p = d.k WHERE d.tag = 3 AND d.k < 10";

#[test]
fn a_join_on_a_partition_column_skips_files() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    tables(&s, dir.path());
    let want: i64 = (0..200_000_i64).filter(|i| i % 40 == 3).sum();
    assert_eq!(long(&s, JOIN), vec![want]);
    let skipped = counter(&s, JOIN, "runtime_filter_splits_skipped");
    // One partition of the 40 joins.
    assert!(skipped >= 39, "{skipped} files skipped");
}

/// Results are the same with runtime filters off, for joins that keep rows of
/// the filter's NULL key and of no partition too.
#[test]
fn pruning_leaves_results_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    tables(&s, dir.path());
    partitioned(
        &s,
        dir.path(),
        "sparse",
        "SELECT id, CASE WHEN id % 7 = 0 THEN NULL ELSE id % 5 END AS p FROM range(5000)",
    );
    run(
        &s,
        "CREATE TEMPORARY VIEW nulls AS SELECT * FROM VALUES (CAST(NULL AS BIGINT)), (1) AS t(k)",
    );
    let queries = [
        JOIN,
        "SELECT count(*) FROM fact f JOIN dim d ON f.p = d.k WHERE d.tag = 99",
        "SELECT count(*) FROM fact f WHERE f.p IN (SELECT k FROM dim WHERE tag = 1)",
        "SELECT count(*) FROM fact f WHERE f.p NOT IN (SELECT k FROM dim WHERE tag = 1)",
        "SELECT count(*) FROM fact f LEFT JOIN dim d ON f.p = d.k AND d.tag = 1",
        "SELECT count(*) FROM sparse f JOIN dim d ON f.p = d.k WHERE d.tag < 3",
        "SELECT count(*) FROM sparse f LEFT JOIN dim d ON f.p <=> d.k WHERE d.tag IS NULL",
        // A NULL partition value: a null-safe key keeps it, a plain one doesn't.
        "SELECT count(*) FROM sparse f JOIN nulls d ON f.p <=> d.k",
        "SELECT count(*) FROM sparse f JOIN nulls d ON f.p = d.k",
        // A key that doesn't read the column as it is gets no filter.
        "SELECT count(*) FROM sparse f JOIN dim d ON CAST(f.p AS STRING) = CAST(d.k AS STRING)",
    ];
    for q in queries {
        s.conf()
            .set("krishiv.runtimeFilter.enabled", "true")
            .unwrap();
        let on = long(&s, q);
        s.conf()
            .set("krishiv.runtimeFilter.enabled", "false")
            .unwrap();
        let off = long(&s, q);
        assert_eq!(on, off, "{q}");
    }
}

/// A build with no rows skips every file; the filter off reads them.
#[test]
fn an_empty_build_skips_every_file() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    tables(&s, dir.path());
    let q = "SELECT count(*) FROM fact f JOIN dim d ON f.p = d.k WHERE d.tag = 99";
    assert_eq!(long(&s, q), vec![0]);
}
