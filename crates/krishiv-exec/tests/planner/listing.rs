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

/// A table of many files opens on a multi-threaded runtime's task (the
/// future is `Send`), reading the footers in bounded chunks at a small
/// request limit; the statistics cover every file.
#[test]
fn a_table_of_many_files_opens_in_chunks() {
    use krishiv_exec::planner::{FileTable, TableProvider as _};
    use krishiv_io::parquet::ParquetFormat;
    use krishiv_io::{FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, StoreId};
    use krishiv_plan::Literal;
    use object_store::ObjectStore;
    use std::sync::Arc;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/planner");
    let store = Arc::new(object_store::local::LocalFileSystem::new_with_prefix(&dir).unwrap());
    let sched = IoScheduler::new(IoConfig {
        max_concurrent_requests: 3,
        ..IoConfig::default()
    });
    let id = StoreId::new("file://");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let meta = rt
        .block_on(object_store::ObjectStoreExt::head(
            store.as_ref(),
            &object_store::path::Path::from("t.parquet"),
        ))
        .unwrap();
    // 301 files: 25 full chunks of 12 pending reads and one of 1.
    let files: Vec<FileRef> = (0..301)
        .map(|_| FileRef::from_meta(id.clone(), &meta))
        .collect();
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        arrow_schema::DataType::Int32,
        true,
    )]));
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let options = FormatOptions::defaults(ParquetFormat.options());
    let io = sched.context();
    let table = rt
        .block_on(rt.spawn(async move {
            let config = krishiv_common::config::Config::new();
            FileTable::open(format, io, schema, files, options, &config, 4).await
        }))
        .unwrap()
        .unwrap();
    let stats = table.statistics().unwrap();
    assert_eq!(stats.rows, 301 * 8);
    let id = &stats.columns["id"];
    assert_eq!(id.min, Some(Literal::Integer(1)));
    assert_eq!(id.max, Some(Literal::Integer(8)));
    assert_eq!(id.nulls, Some(0));
}
