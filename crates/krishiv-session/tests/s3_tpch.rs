// SPDX-License-Identifier: Apache-2.0
//! TPC-H read from S3 (P2-T084 acceptance): the eight tables of a local
//! TPC-H Parquet directory are uploaded to an S3-compatible emulator and the
//! 22 queries (`bench/queries/tpch`) run over the local files and over
//! `s3a://`; the answers must be equal.
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

use arrow_array::RecordBatch;
use arrow_select::concat::concat_batches;
use bytes::Bytes;
use futures::executor::block_on;
use krishiv_common::config::Config;
use krishiv_io::{IoConfig, IoScheduler, StoreRegistry, register};
use krishiv_session::Session;

const TABLES: [&str; 8] = [
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

/// The text of TPC-H query `q`.
fn query(q: usize) -> String {
    let path = format!(
        "{}/../../bench/queries/tpch/q{q:02}.sql",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(path).unwrap()
}

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

/// The result as one batch (so batch boundaries do not matter); `None` for
/// no batch.
fn rows(s: &Session, sql: &str) -> Option<RecordBatch> {
    let df = block_on(s.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let first = batches.first()?;
    Some(concat_batches(&first.schema(), &batches).unwrap())
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
    // The store layer times its requests with Tokio, which `futures::executor` does not provide.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut urls: Vec<(String, Vec<String>)> = Vec::new();
    for table in TABLES {
        let files = table_files(&dir, table);
        assert!(
            !files.is_empty(),
            "no {table} Parquet under {}",
            dir.display()
        );
        let mut remote = Vec::new();
        for (i, f) in files.iter().enumerate() {
            let url = format!("s3a://krishiv-test/{prefix}/{table}/part-{i:05}.parquet");
            let (store, path, _) = register(&sched, &registry, &url).unwrap();
            let bytes = Bytes::from(std::fs::read(f).unwrap());
            rt.block_on(io.put_object(&store, &path, bytes)).unwrap();
            remote.push(url);
        }
        urls.push((table.to_owned(), remote));
    }
    // The same names over S3 in one session and over the files in another.
    let (remote, local) = (Session::new().unwrap(), Session::new().unwrap());
    for (k, v) in pairs {
        remote.conf().set(k, v).unwrap();
    }
    for (table, urls) in &urls {
        let r: Vec<&str> = urls.iter().map(String::as_str).collect();
        block_on(remote.read().parquet(r).create_or_replace_temp_view(table)).unwrap();
        let files = table_files(&dir, table);
        let l: Vec<&str> = files.iter().map(|p| p.to_str().unwrap()).collect();
        block_on(local.read().parquet(l).create_or_replace_temp_view(table)).unwrap();
    }
    for q in 1..=22 {
        let sql = query(q);
        let over_s3 = rows(&remote, &sql).unwrap_or_else(|| panic!("q{q:02}: no batch"));
        assert!(over_s3.num_rows() > 0, "q{q:02} returned no rows");
        // Compared as values: a batch's `Debug` text shows only the first
        // and last rows of a long column.
        let over_files = rows(&local, &sql).unwrap_or_else(|| panic!("q{q:02}: no local batch"));
        assert!(
            over_s3 == over_files,
            "q{q:02}: {} rows over S3, {} over the files differ",
            over_s3.num_rows(),
            over_files.num_rows()
        );
    }
}
