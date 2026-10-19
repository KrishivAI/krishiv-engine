// SPDX-License-Identifier: Apache-2.0
//! A table of files reads the files it was opened with. Spark 4.2 keeps a
//! file-backed temporary view's listing from the view's creation, so a
//! query after the files were overwritten fails with
//! `FAILED_READ_FILE.FILE_NOT_EXIST`; Krishiv lists a view's paths again
//! for each query (ADR 0023), so it reads the files as they are. This test
//! pins both halves at the planner: a plan over a stale listing fails as
//! Spark does, a plan over a fresh listing reads the new files.

use std::path::Path;

use krishiv_common::error::ErrorClass;

use super::tpch::{parquet_files, rows};

#[test]
fn a_fresh_listing_reads_overwritten_files() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/planner/t.parquet");
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("v")).unwrap();
    std::fs::copy(&fixture, dir.path().join("v/part-0.parquet")).unwrap();
    let list = || -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir.path().join("v"))
            .unwrap()
            .map(|e| format!("v/{}", e.unwrap().file_name().to_string_lossy()))
            .collect();
        v.sort();
        v
    };
    let sql = "SELECT count(*) FROM v";
    let stale = parquet_files(dir.path(), &[("v", list())], true, &rt);
    let plan = stale.plan_sql(sql).unwrap();
    assert_eq!(rows(&super::run(&plan).unwrap()), ["8"]);

    // Overwrite: the old file goes, two new ones come.
    std::fs::remove_file(dir.path().join("v/part-0.parquet")).unwrap();
    for f in ["v/part-1.parquet", "v/part-2.parquet"] {
        std::fs::copy(&fixture, dir.path().join(f)).unwrap();
    }
    let e = super::run(&stale.plan_sql(sql).unwrap()).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::FailedReadFileFileNotExist,
        "{e}"
    );
    let fresh = parquet_files(dir.path(), &[("v", list())], true, &rt);
    assert_eq!(
        rows(&super::run(&fresh.plan_sql(sql).unwrap()).unwrap()),
        ["16"]
    );
}
