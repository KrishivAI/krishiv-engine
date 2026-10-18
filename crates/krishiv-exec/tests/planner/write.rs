// SPDX-License-Identifier: Apache-2.0
//! `df.write.parquet(path)`: a query's plan written through the directory
//! commit (`DirectoryWrite`) and read back.

use std::path::Path;
use std::sync::Arc;

use krishiv_common::error::ErrorClass;
use krishiv_exec::FileWriteSink;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    DirectoryWrite, FileFormat, FormatOptions, IoConfig, IoScheduler, SaveMode, StoreId,
};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;

use super::tpch::{parquet_dir, rows};

#[test]
fn query_results_are_written_committed_and_read_back() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/planner");
    let db = parquet_dir(&fixtures, &["t"], true, &rt);
    let sql = "SELECT id, s, d * 2 AS d2 FROM t WHERE id > 2 ORDER BY id";
    let want = rows(&super::run(&db.plan_sql(sql).unwrap()).unwrap());

    let out = tempfile::tempdir().unwrap();
    let store = Arc::new(LocalFileSystem::new_with_prefix(out.path()).unwrap());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("file://out");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let options = FormatOptions::defaults(ParquetFormat.options());
    let begin = |mode| {
        rt.block_on(DirectoryWrite::begin(
            &io,
            id.clone(),
            StorePath::from("result"),
            mode,
            "job-1",
        ))
    };

    let w = begin(SaveMode::ErrorIfExists).unwrap().unwrap();
    let plan = db.plan_sql(sql).unwrap();
    let sink = Arc::new(
        FileWriteSink::try_new(
            Arc::clone(&format),
            io.clone(),
            Arc::clone(plan.schema()),
            options.clone(),
            w.write_options(0),
        )
        .unwrap(),
    );
    let schema = Arc::clone(plan.schema());
    let write = plan.into_write(Arc::clone(&sink)).unwrap();
    assert!(
        write
            .to_string()
            .starts_with("WriteFiles parquet to file://out/result"),
        "{write}"
    );
    let written = rows(&super::run(&write).unwrap());
    let counted: usize = written
        .iter()
        .map(|r| r.rsplit('|').next().unwrap().parse::<usize>().unwrap())
        .sum();
    assert_eq!(counted, want.len());
    let files = rt
        .block_on(w.commit(&io, format.as_ref(), schema, &options, sink.files()))
        .unwrap();
    assert_eq!(files.len(), 1);

    // Read the file back as a table.
    let back = tempfile::tempdir().unwrap();
    let file = out.path().join(files[0].path.as_ref());
    std::fs::copy(file, back.path().join("r.parquet")).unwrap();
    let db = parquet_dir(back.path(), &["r"], true, &rt);
    let got = rows(&super::run(&db.plan_sql("SELECT * FROM r ORDER BY id").unwrap()).unwrap());
    assert_eq!(got, want);

    // `error` mode on the committed path.
    let e = begin(SaveMode::ErrorIfExists).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::PathAlreadyExists);
    // A sink for another schema is refused.
    let other = Arc::new(
        FileWriteSink::try_new(
            format,
            io.clone(),
            Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "x",
                arrow_schema::DataType::Int32,
                true,
            )])),
            options,
            begin(SaveMode::Overwrite)
                .unwrap()
                .unwrap()
                .write_options(0),
        )
        .unwrap(),
    );
    let e = db
        .plan_sql("SELECT * FROM r")
        .unwrap()
        .into_write(other)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}
