// SPDX-License-Identifier: Apache-2.0
//! The relation cache over file sources (db-benchmark finding: `persist()` of a
//! CSV or Parquet read re-scanned the files on every action).
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::{Session, StorageLevel};

/// The relation row of `system.caches`: entries, memory bytes, disk bytes,
/// pending registrations.
fn relation_row(s: &Session) -> [i64; 4] {
    let batches = block_on(async {
        s.sql(
            "SELECT entries, memory_bytes, disk_bytes, entries_pending \
             FROM system.caches WHERE cache = 'relation'",
        )
        .await?
        .collect()
        .await
    })
    .unwrap();
    let b = &batches[0];
    let v = |i: usize| {
        b.column(i)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0)
    };
    [v(0), v(1), v(2), v(3)]
}

/// Writes `n` rows `(id1 string, id4 int, v1 int)` as `format` under `dir`.
fn write_source(s: &Session, dir: &std::path::Path, format: &str, n: i64) -> String {
    let path = dir.join(format);
    let path = path.to_str().unwrap().to_owned();
    block_on(async {
        s.sql(&format!(
            "SELECT concat('id', CAST(id % 7 AS STRING)) AS id1, CAST(id % 11 AS INT) AS id4, \
             CAST(id AS INT) AS v1 FROM range({n})"
        ))
        .await?
        .write()
        .format(format)
        .option("header", "true")
        .save(&path)
        .await
    })
    .unwrap();
    path
}

/// Reads `path` as `format`, as the db-benchmark scripts do (`header`,
/// `inferSchema` for CSV).
fn read(s: &Session, format: &str, path: &str) -> krishiv_session::DataFrame {
    let r = s.read().format(format);
    let r = if format == "csv" {
        r.option("header", "true").option("inferSchema", "true")
    } else {
        r
    };
    r.load([path])
}

/// The relation cache's hit counter (`system.caches`).
fn hits(s: &Session) -> i64 {
    let b = block_on(async {
        s.sql("SELECT hits FROM system.caches WHERE cache = 'relation'")
            .await?
            .collect()
            .await
    })
    .unwrap();
    b[0].column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0)
}

/// The benchmark's question over the view `x`: its row count.
fn question(s: &Session) -> u64 {
    block_on(async {
        s.sql("SELECT id1, sum(v1) AS v1 FROM x GROUP BY id1")
            .await?
            .count()
            .await
    })
    .unwrap()
}

/// `x = read(...).persist(); x.count(); x.createOrReplaceTempView("x")`, then
/// SQL over `x`: the count fills the entry, and both a later count and the SQL
/// over the view read it. The view used to be a file view of its own, listed
/// and scanned on every query, so every benchmark question re-read the files.
fn a_view_over_a_persisted_read_reads_the_cache(format: &str) {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let path = write_source(&s, dir.path(), format, 5000);
    let x = read(&s, format, &path).persist(StorageLevel::MEMORY_ONLY);
    assert_eq!(block_on(x.count()).unwrap(), 5000);
    let [entries, memory, _, pending] = relation_row(&s);
    assert_eq!(
        (entries, pending),
        (1, 0),
        "{format}: the count filled the entry"
    );
    assert!(memory > 0, "{format}: the entry holds its rows");
    let filled = hits(&s);
    assert_eq!(block_on(x.count()).unwrap(), 5000);
    assert_eq!(
        hits(&s),
        filled + 1,
        "{format}: a second count reads the cache"
    );
    block_on(x.create_or_replace_temp_view("x")).unwrap();
    assert_eq!(question(&s), 7);
    assert_eq!(
        hits(&s),
        filled + 2,
        "{format}: SQL over the view reads the cache"
    );
    // Unpersisted, the view reads the files again.
    let x = block_on(x.unpersist()).unwrap();
    assert_eq!(question(&s), 7);
    assert_eq!(hits(&s), filled + 2);
    drop(x);
}

#[test]
fn a_view_over_a_persisted_parquet_read_reads_the_cache() {
    a_view_over_a_persisted_read_reads_the_cache("parquet");
}

#[test]
fn a_view_over_a_persisted_csv_read_with_inferred_schema_reads_the_cache() {
    a_view_over_a_persisted_read_reads_the_cache("csv");
}
