// SPDX-License-Identifier: Apache-2.0
//! TPC-H read from S3 (P2-T084 acceptance): the tables of a local TPC-H
//! Parquet directory are uploaded to an S3-compatible emulator and the same
//! queries run over the local files and over `s3a://`; the answers must be
//! equal.
//!
//! Runs only with an emulator and data (otherwise it returns at once):
//! `KRISHIV_TEST_S3_ENDPOINT=http://127.0.0.1:9000` (MinIO, credentials
//! `minioadmin`/`minioadmin`, bucket `krishiv-test`) and `KRISHIV_TPCH_DIR`
//! (a directory of `<table>.parquet` files or `<table>/` directories of part
//! files, as `krishiv-datasets tpch` writes). `tests/scripts/object_store_emulators.sh`
//! starts the emulators and sets both.
#![expect(clippy::unwrap_used, reason = "test code")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytes::Bytes;
use futures::executor::block_on;
use krishiv_common::config::Config;
use krishiv_io::{IoConfig, IoScheduler, StoreRegistry};
use krishiv_session::Session;

const QUERIES: [&str; 3] = [
    "SELECT count(*) FROM lineitem",
    "SELECT l_returnflag, l_linestatus, sum(l_quantity), sum(l_extendedprice), \
     sum(l_extendedprice * (1 - l_discount)), count(*) \
     FROM lineitem WHERE l_shipdate <= DATE '1998-09-02' \
     GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus",
    "SELECT sum(l_extendedprice * l_discount) FROM lineitem \
     WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' \
     AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24",
];

/// The Parquet files of `table` under `dir`: `<table>.parquet`, or the part
/// files of `<table>/`.
fn table_files(dir: &Path, table: &str) -> Vec<PathBuf> {
    let single = dir.join(format!("{table}.parquet"));
    if single.is_file() {
        return vec![single];
    }
    let mut parts: Vec<PathBuf> = std::fs::read_dir(dir.join(table))
        .map(|d| {
            d.map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
                .collect()
        })
        .unwrap_or_default();
    parts.sort();
    parts
}

/// The result's batches as text (their `Debug` form is deterministic).
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.collect())
        .unwrap()
        .iter()
        .map(|b| format!("{b:?}"))
        .collect()
}

#[test]
fn tpch_over_s3_equals_tpch_over_local_files() {
    let (Some(endpoint), Some(dir)) = (
        std::env::var("KRISHIV_TEST_S3_ENDPOINT")
            .ok()
            .filter(|v| !v.is_empty()),
        std::env::var("KRISHIV_TPCH_DIR")
            .ok()
            .filter(|v| !v.is_empty()),
    ) else {
        return;
    };
    let dir = PathBuf::from(dir);
    let pairs = [
        ("fs.s3a.endpoint", endpoint.as_str()),
        ("fs.s3a.access.key", "minioadmin"),
        ("fs.s3a.secret.key", "minioadmin"),
        ("fs.s3a.path.style.access", "true"),
        ("fs.s3a.endpoint.region", "us-east-1"),
    ];
    // Upload every table under a fresh prefix.
    let (config, _) = Config::from_pairs(pairs).unwrap();
    let config = Arc::new(config);
    let sched = IoScheduler::new(IoConfig::from_config(&config));
    let registry = StoreRegistry::new(config);
    let prefix = format!("tpch-{}", std::process::id());
    let io = sched.context();
    let mut urls: Vec<(String, Vec<String>)> = Vec::new();
    for table in ["lineitem"] {
        let files = table_files(&dir, table);
        assert!(
            !files.is_empty(),
            "no {table} Parquet under {}",
            dir.display()
        );
        let mut remote = Vec::new();
        for (i, f) in files.iter().enumerate() {
            let url = format!("s3a://krishiv-test/{prefix}/{table}/part-{i:05}.parquet");
            let (store, path, _) = registry.ensure(&sched, &url).unwrap();
            let bytes = Bytes::from(std::fs::read(f).unwrap());
            block_on(io.put_object(&store, &path, bytes)).unwrap();
            remote.push(url);
        }
        urls.push((table.to_owned(), remote));
    }
    let s = Session::new().unwrap();
    for (k, v) in pairs {
        s.conf().set(k, v).unwrap();
    }
    // `lineitem` over S3; `lineitem_local` over the files.
    for (table, remote) in &urls {
        let r: Vec<&str> = remote.iter().map(String::as_str).collect();
        block_on(s.read().parquet(r).create_or_replace_temp_view(table)).unwrap();
        let local = table_files(&dir, table);
        let l: Vec<&str> = local.iter().map(|p| p.to_str().unwrap()).collect();
        block_on(
            s.read()
                .parquet(l)
                .create_or_replace_temp_view(&format!("{table}_local")),
        )
        .unwrap();
    }
    for sql in QUERIES {
        let over_s3 = rows(&s, sql);
        let over_files = rows(&s, &sql.replace("lineitem", "lineitem_local"));
        assert!(!over_s3.is_empty(), "{sql}");
        assert_eq!(over_s3, over_files, "{sql}");
    }
}
