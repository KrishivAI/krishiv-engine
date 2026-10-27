// SPDX-License-Identifier: Apache-2.0
//! `df.write.parquet(path)`: a query's plan written through the directory
//! commit (`DirectoryWrite`) and read back.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_common::error::ErrorClass;
use krishiv_exec::physical::{PhysNode, TaskSpec, WorkerContext, instantiate};
use krishiv_exec::planner::MemoryTables;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    DirectoryWrite, FileFormat, FileStatistics, FormatOptions, FormatRegistry, IoConfig,
    IoScheduler, Precision, SaveMode, StoreId, WrittenFileInfo,
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
    let schema = Arc::clone(plan.schema());
    let write = plan
        .into_write(krishiv_exec::physical::WritePlan {
            format: "parquet".to_owned(),
            options: BTreeMap::new(),
            store: "file://out".to_owned(),
            dir: "result".to_owned(),
            job_uuid: "job-1".to_owned(),
            max_records_per_file: 0,
            target_file_bytes: 0,
        })
        .with_io(io.clone())
        .with_formats(Arc::new(FormatRegistry::with_defaults()))
        .with_config(Arc::new(Config::new()));
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
    // The commit runs from the plan's result rows (path, size, rows).
    let files: Vec<WrittenFileInfo> = written
        .iter()
        .map(|r| {
            let mut parts = r.split('|');
            let path = parts.next().unwrap().to_owned();
            let size: u64 = parts.next().unwrap().parse().unwrap();
            let rows: u64 = parts.next().unwrap().parse().unwrap();
            WrittenFileInfo {
                path: StorePath::from(path.as_str()),
                size,
                rows,
                statistics: FileStatistics::with_rows(Precision::Exact(rows)),
            }
        })
        .collect();
    let files = rt
        .block_on(w.commit(&io, format.as_ref(), schema, &options, files))
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
    // A worker without an I/O context cannot instantiate a write.
    let plan = krishiv_exec::physical::PhysicalPlan::new(
        PhysNode::WriteFiles {
            input: Box::new(PhysNode::OneRow),
            write: Some(krishiv_exec::physical::WritePlan {
                format: "parquet".to_owned(),
                options: BTreeMap::new(),
                store: "file://out".to_owned(),
                dir: "result".to_owned(),
                job_uuid: "job-1".to_owned(),
                max_records_per_file: 0,
                target_file_bytes: 0,
            }),
        },
        krishiv_exec::physical::WritePlan::result_schema(),
        krishiv_exec::planner::ExplainNode::leaf("WriteFiles", ""),
        krishiv_exec::physical::ExprEnv {
            ansi: true,
            legacy_statistical_aggregate: false,
            zone: None,
            query_start_micros: 0,
            user: Arc::from("u"),
        },
    );
    let worker = WorkerContext::new(Arc::new(MemoryTables::new()));
    let e = instantiate(&plan, &worker).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("I/O context"), "{e}");
    // So is a worker without the format (an empty registry): Spark's
    // built-in source without an implementation is unsupported.
    let worker = WorkerContext::new(Arc::new(MemoryTables::new())).with_io(io.clone());
    let e = instantiate(&plan, &worker).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("parquet"), "{e}");
    // An unknown format option fails when the write instantiates.
    let bad = krishiv_exec::physical::WritePlan {
        format: "parquet".to_owned(),
        options: BTreeMap::from([(
            "compression".to_owned(),
            krishiv_common::secret::OptionValue::Plain("bogus".to_owned()),
        )]),
        store: "file://out".to_owned(),
        dir: "result".to_owned(),
        job_uuid: "job-1".to_owned(),
        max_records_per_file: 0,
        target_file_bytes: 0,
    };
    let bad_plan = krishiv_exec::physical::PhysicalPlan::new(
        PhysNode::WriteFiles {
            input: Box::new(PhysNode::OneRow),
            write: Some(bad),
        },
        krishiv_exec::physical::WritePlan::result_schema(),
        krishiv_exec::planner::ExplainNode::leaf("WriteFiles", ""),
        krishiv_exec::physical::ExprEnv {
            ansi: true,
            legacy_statistical_aggregate: false,
            zone: None,
            query_start_micros: 0,
            user: Arc::from("u"),
        },
    );
    let worker = WorkerContext::new(Arc::new(MemoryTables::new()))
        .with_io(io.clone())
        .with_formats(Arc::new(FormatRegistry::with_defaults()))
        .with_config(Arc::new(Config::new()));
    let e = krishiv_exec::physical::instantiate_task(&bad_plan, &worker, &TaskSpec::whole())
        .unwrap_err();
    assert!(
        e.to_string().contains("bogus"),
        "invalid option surfaces: {e}"
    );
    // Another task's files name its partition.
    let worker = WorkerContext::new(Arc::new(MemoryTables::new()))
        .with_io(io.clone())
        .with_formats(Arc::new(FormatRegistry::with_defaults()))
        .with_config(Arc::new(Config::new()));
    let task = TaskSpec {
        partition: 3,
        partitions: 4,
        ..TaskSpec::whole()
    };
    let _ = krishiv_exec::physical::instantiate_task(&plan, &worker, &task).expect("task 3 runs");
}

/// Two tasks of one stage write disjoint files named by their partition;
/// one commit lists both.
#[test]
fn two_tasks_write_disjoint_files_named_by_partition() {
    use krishiv_exec::physical::WritePlan;
    use krishiv_memory::{MemoryConfig, TreePool};

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let out = tempfile::tempdir().unwrap();
    let store = Arc::new(LocalFileSystem::new_with_prefix(out.path()).unwrap());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("file://out");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let options = FormatOptions::defaults(ParquetFormat.options());
    let w = rt
        .block_on(DirectoryWrite::begin(
            &io,
            id,
            StorePath::from("two"),
            SaveMode::ErrorIfExists,
            "job-two",
        ))
        .unwrap()
        .unwrap();
    // One row, so every task writes exactly one file.
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = arrow_array::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Int32Array::from(vec![1]))],
    )
    .unwrap();
    let mem: Arc<dyn krishiv_memory::MemoryPool> = TreePool::new(&MemoryConfig::default());
    let table = Arc::new(krishiv_exec::planner::MemoryTable::new(
        Arc::clone(&schema),
        vec![krishiv_types::batch::Batch::new(batch)],
    ));
    let tables =
        Arc::new(MemoryTables::new().with_table(&["spark_catalog", "default", "t"], table));
    let config = Arc::new(Config::new());
    let mut all = Vec::new();
    for partition in [0_u32, 1_u32] {
        let scan = krishiv_plan::Scan {
            source: krishiv_plan::ReadSource::Table {
                name: vec![
                    "spark_catalog".to_owned(),
                    "default".to_owned(),
                    "t".to_owned(),
                ],
                options: BTreeMap::new(),
            },
            output: vec![krishiv_plan::BoundColumn {
                id: krishiv_plan::AttrId(1),
                name: "id".to_owned(),
                data_type: krishiv_types::DataType::Integer,
                nullable: false,
            }],
            version: None,
        };
        let node = PhysNode::Scan(krishiv_exec::physical::ScanNode {
            id: krishiv_exec::physical::ScanId(1),
            scan,
            projection: vec![krishiv_io::ColumnPath::new("id")],
            filter: None,
            runtime_filters: Vec::new(),
            row_stage: krishiv_exec::RowStageSettings::default(),
        });
        let write = WritePlan {
            format: "parquet".to_owned(),
            options: BTreeMap::new(),
            store: "file://out".to_owned(),
            dir: "two".to_owned(),
            job_uuid: "job-two".to_owned(),
            max_records_per_file: 0,
            target_file_bytes: 0,
        };
        let plan = krishiv_exec::physical::PhysicalPlan::new(
            PhysNode::WriteFiles {
                input: Box::new(node),
                write: Some(write),
            },
            WritePlan::result_schema(),
            krishiv_exec::planner::ExplainNode::leaf("WriteFiles", ""),
            krishiv_exec::physical::ExprEnv {
                ansi: true,
                legacy_statistical_aggregate: false,
                zone: None,
                query_start_micros: 0,
                user: Arc::from("u"),
            },
        );
        let worker = WorkerContext::new(Arc::clone(&tables) as _)
            .with_io(io.clone())
            .with_formats(Arc::new(FormatRegistry::with_defaults()))
            .with_config(Arc::clone(&config));
        let task = TaskSpec {
            partition,
            partitions: 2,
            ..TaskSpec::whole()
        };
        let node =
            krishiv_exec::physical::instantiate_task(&plan, &worker, &task).expect("task runs");
        let query = krishiv_exec::QueryContext::new(
            krishiv_exec::QueryId(u64::from(partition)),
            krishiv_exec::WorkloadClass::Batch,
            krishiv_exec::ExecConfig::default(),
            mem.child(&format!("q{partition}"), None),
        );
        let runner = krishiv_exec::QueryRunner::new(
            krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig::default()).unwrap(),
        );
        let batches = rt.block_on(async {
            let mut stream = runner.execute(node, Arc::clone(&query)).await.unwrap();
            let mut out = Vec::new();
            use futures::StreamExt as _;
            while let Some(b) = stream.next().await {
                out.push(b.unwrap().into_parts().0);
            }
            query.finished().await;
            out
        });
        assert_eq!(batches.len(), 1, "partition {partition}");
        let path = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringViewArray>()
            .unwrap()
            .value(0)
            .to_owned();
        let want_prefix = format!("two/part-{partition:05}-job-two-c000");
        assert!(
            path.starts_with(&want_prefix),
            "partition {partition}: {path}"
        );
        let size = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap()
            .value(0);
        let rows = batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap()
            .value(0);
        all.push(WrittenFileInfo {
            path: StorePath::from(path.as_str()),
            size: u64::try_from(size).unwrap(),
            rows: u64::try_from(rows).unwrap(),
            statistics: FileStatistics::with_rows(Precision::Exact(u64::try_from(rows).unwrap())),
        });
    }
    assert_ne!(all[0].path, all[1].path);
    let files = rt
        .block_on(w.commit(&io, format.as_ref(), schema, &options, all))
        .unwrap();
    assert_eq!(files.len(), 2);
    let mut names: Vec<String> = files.iter().map(|f| f.path.to_string()).collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "two/part-00000-job-two-c000.snappy.parquet".to_owned(),
            "two/part-00001-job-two-c000.snappy.parquet".to_owned(),
        ]
    );
}

/// A retried task never overwrites the committed attempt: attempt 0 names
/// files exactly as before, attempt 1 appends `_1` to the job id, and the
/// commit deletes the uncommitted attempt's files.
#[test]
fn a_retried_attempt_leaves_no_orphan_after_commit() {
    use krishiv_exec::physical::WritePlan;
    use krishiv_memory::{MemoryConfig, TreePool};

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let out = tempfile::tempdir().unwrap();
    let store = Arc::new(LocalFileSystem::new_with_prefix(out.path()).unwrap());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("file://out");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let options = FormatOptions::defaults(ParquetFormat.options());
    let w = rt
        .block_on(DirectoryWrite::begin(
            &io,
            id.clone(),
            StorePath::from("retry"),
            SaveMode::ErrorIfExists,
            "job-retry",
        ))
        .unwrap()
        .unwrap();
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = arrow_array::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Int32Array::from(vec![7]))],
    )
    .unwrap();
    let mem: Arc<dyn krishiv_memory::MemoryPool> = TreePool::new(&MemoryConfig::default());
    let table = Arc::new(krishiv_exec::planner::MemoryTable::new(
        Arc::clone(&schema),
        vec![krishiv_types::batch::Batch::new(batch)],
    ));
    let tables =
        Arc::new(MemoryTables::new().with_table(&["spark_catalog", "default", "t"], table));
    let config = Arc::new(Config::new());
    let run_attempt = |attempt: u32| {
        let scan = krishiv_plan::Scan {
            source: krishiv_plan::ReadSource::Table {
                name: vec![
                    "spark_catalog".to_owned(),
                    "default".to_owned(),
                    "t".to_owned(),
                ],
                options: BTreeMap::new(),
            },
            output: vec![krishiv_plan::BoundColumn {
                id: krishiv_plan::AttrId(1),
                name: "id".to_owned(),
                data_type: krishiv_types::DataType::Integer,
                nullable: false,
            }],
            version: None,
        };
        let node = PhysNode::Scan(krishiv_exec::physical::ScanNode {
            id: krishiv_exec::physical::ScanId(1),
            scan,
            projection: vec![krishiv_io::ColumnPath::new("id")],
            filter: None,
            runtime_filters: Vec::new(),
            row_stage: krishiv_exec::RowStageSettings::default(),
        });
        let write = WritePlan {
            format: "parquet".to_owned(),
            options: BTreeMap::new(),
            store: "file://out".to_owned(),
            dir: "retry".to_owned(),
            job_uuid: "job-retry".to_owned(),
            max_records_per_file: 0,
            target_file_bytes: 0,
        };
        let plan = krishiv_exec::physical::PhysicalPlan::new(
            PhysNode::WriteFiles {
                input: Box::new(node),
                write: Some(write),
            },
            WritePlan::result_schema(),
            krishiv_exec::planner::ExplainNode::leaf("WriteFiles", ""),
            krishiv_exec::physical::ExprEnv {
                ansi: true,
                legacy_statistical_aggregate: false,
                zone: None,
                query_start_micros: 0,
                user: Arc::from("u"),
            },
        );
        let worker = WorkerContext::new(Arc::clone(&tables) as _)
            .with_io(io.clone())
            .with_formats(Arc::new(FormatRegistry::with_defaults()))
            .with_config(Arc::clone(&config));
        let task = TaskSpec {
            attempt,
            ..TaskSpec::whole()
        };
        let node =
            krishiv_exec::physical::instantiate_task(&plan, &worker, &task).expect("attempt runs");
        let query = krishiv_exec::QueryContext::new(
            krishiv_exec::QueryId(u64::from(attempt)),
            krishiv_exec::WorkloadClass::Batch,
            krishiv_exec::ExecConfig::default(),
            mem.child(&format!("q{attempt}"), None),
        );
        let runner = krishiv_exec::QueryRunner::new(
            krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig::default()).unwrap(),
        );
        rt.block_on(async {
            let mut stream = runner.execute(node, Arc::clone(&query)).await.unwrap();
            let mut out = Vec::new();
            use futures::StreamExt as _;
            while let Some(b) = stream.next().await {
                out.push(b.unwrap().into_parts().0);
            }
            query.finished().await;
            let batch = &out[0];
            let path = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::StringViewArray>()
                .unwrap()
                .value(0)
                .to_owned();
            if attempt == 0 {
                assert_eq!(
                    path, "retry/part-00000-job-retry-c000.snappy.parquet",
                    "{path}"
                );
            } else {
                assert_eq!(
                    path, "retry/part-00000-job-retry_1-c000.snappy.parquet",
                    "{path}"
                );
            }
            let size = u64::try_from(
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .unwrap()
                    .value(0),
            )
            .unwrap();
            let rows = u64::try_from(
                batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .unwrap()
                    .value(0),
            )
            .unwrap();
            WrittenFileInfo {
                path: StorePath::from(path.as_str()),
                size,
                rows,
                statistics: FileStatistics::with_rows(Precision::Exact(rows)),
            }
        })
    };
    let committed = run_attempt(0);
    let uncommitted = run_attempt(1);
    // Both attempts wrote; only attempt 0 commits.
    let listed: Vec<String> = rt
        .block_on(io.list(&id, &StorePath::from("retry")))
        .unwrap()
        .into_iter()
        .map(|m| m.location.to_string())
        .collect();
    assert!(listed.contains(&committed.path.to_string()), "{listed:?}");
    assert!(listed.contains(&uncommitted.path.to_string()), "{listed:?}");
    let files = rt
        .block_on(w.commit(
            &io,
            format.as_ref(),
            schema,
            &options,
            vec![committed.clone()],
        ))
        .unwrap();
    assert_eq!(files, vec![committed]);
    let listed: Vec<String> = rt
        .block_on(io.list(&id, &StorePath::from("retry")))
        .unwrap()
        .into_iter()
        .map(|m| m.location.to_string())
        .collect();
    assert!(
        listed.contains(&"retry/part-00000-job-retry-c000.snappy.parquet".to_owned()),
        "{listed:?}"
    );
    assert!(
        !listed.iter().any(|p| p.contains("job-retry_1")),
        "late attempt survived: {listed:?}"
    );
}

/// An empty partition other than 0 writes no file; partition 0 of an empty
/// result still commits the schema-bearing file, named as before.
#[test]
fn empty_partitions_write_no_file_except_partition_zero_commits_one() {
    use krishiv_exec::physical::WritePlan;
    use krishiv_memory::{MemoryConfig, TreePool};

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let out = tempfile::tempdir().unwrap();
    let store = Arc::new(LocalFileSystem::new_with_prefix(out.path()).unwrap());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("file://out");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let options = FormatOptions::defaults(ParquetFormat.options());
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let mem: Arc<dyn krishiv_memory::MemoryPool> = TreePool::new(&MemoryConfig::default());
    // No rows at all.
    let table = Arc::new(krishiv_exec::planner::MemoryTable::new(
        Arc::clone(&schema),
        Vec::new(),
    ));
    let tables =
        Arc::new(MemoryTables::new().with_table(&["spark_catalog", "default", "t"], table));
    let config = Arc::new(Config::new());
    let run_partition = |partition: u32| {
        let scan = krishiv_plan::Scan {
            source: krishiv_plan::ReadSource::Table {
                name: vec![
                    "spark_catalog".to_owned(),
                    "default".to_owned(),
                    "t".to_owned(),
                ],
                options: BTreeMap::new(),
            },
            output: vec![krishiv_plan::BoundColumn {
                id: krishiv_plan::AttrId(1),
                name: "id".to_owned(),
                data_type: krishiv_types::DataType::Integer,
                nullable: false,
            }],
            version: None,
        };
        let node = PhysNode::Scan(krishiv_exec::physical::ScanNode {
            id: krishiv_exec::physical::ScanId(1),
            scan,
            projection: vec![krishiv_io::ColumnPath::new("id")],
            filter: None,
            runtime_filters: Vec::new(),
            row_stage: krishiv_exec::RowStageSettings::default(),
        });
        let write = WritePlan {
            format: "parquet".to_owned(),
            options: BTreeMap::new(),
            store: "file://out".to_owned(),
            dir: "empty".to_owned(),
            job_uuid: "job-empty".to_owned(),
            max_records_per_file: 0,
            target_file_bytes: 0,
        };
        let plan = krishiv_exec::physical::PhysicalPlan::new(
            PhysNode::WriteFiles {
                input: Box::new(node),
                write: Some(write),
            },
            WritePlan::result_schema(),
            krishiv_exec::planner::ExplainNode::leaf("WriteFiles", ""),
            krishiv_exec::physical::ExprEnv {
                ansi: true,
                legacy_statistical_aggregate: false,
                zone: None,
                query_start_micros: 0,
                user: Arc::from("u"),
            },
        );
        let worker = WorkerContext::new(Arc::clone(&tables) as _)
            .with_io(io.clone())
            .with_formats(Arc::new(FormatRegistry::with_defaults()))
            .with_config(Arc::clone(&config));
        let task = TaskSpec {
            partition,
            partitions: 2,
            ..TaskSpec::whole()
        };
        let node =
            krishiv_exec::physical::instantiate_task(&plan, &worker, &task).expect("empty runs");
        let query = krishiv_exec::QueryContext::new(
            krishiv_exec::QueryId(u64::from(partition)),
            krishiv_exec::WorkloadClass::Batch,
            krishiv_exec::ExecConfig::default(),
            mem.child(&format!("q{partition}"), None),
        );
        let runner = krishiv_exec::QueryRunner::new(
            krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig::default()).unwrap(),
        );
        rt.block_on(async {
            let mut stream = runner.execute(node, Arc::clone(&query)).await.unwrap();
            let mut rows = 0;
            use futures::StreamExt as _;
            while let Some(b) = stream.next().await {
                let b = b.unwrap().into_parts().0;
                // The result batch lists files; an empty partition lists none.
                // (An empty batch still has the path/size/rows columns.)
                rows += b.num_rows();
            }
            query.finished().await;
            rows
        })
    };
    assert_eq!(run_partition(1), 0, "partition 1 writes no file");
    assert_eq!(run_partition(0), 0, "partition 0 writes no file itself");
    let w = rt
        .block_on(DirectoryWrite::begin(
            &io,
            id,
            StorePath::from("empty"),
            SaveMode::ErrorIfExists,
            "job-empty",
        ))
        .unwrap()
        .unwrap();
    let files = rt
        .block_on(w.commit(&io, format.as_ref(), schema, &options, Vec::new()))
        .unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(
        files[0].path.to_string(),
        "empty/part-00000-job-empty-c000.snappy.parquet"
    );
    assert_eq!(files[0].rows, 0);
}

/// A write without a write plan (a plan written before the `write_plan`
/// feature) cannot instantiate: there is no sink to build.
#[test]
fn a_write_without_a_write_plan_is_refused() {
    let plan = krishiv_exec::physical::PhysicalPlan::new(
        PhysNode::WriteFiles {
            input: Box::new(PhysNode::OneRow),
            write: None,
        },
        krishiv_exec::physical::WritePlan::result_schema(),
        krishiv_exec::planner::ExplainNode::leaf("WriteFiles", ""),
        krishiv_exec::physical::ExprEnv {
            ansi: true,
            legacy_statistical_aggregate: false,
            zone: None,
            query_start_micros: 0,
            user: Arc::from("u"),
        },
    );
    let worker = WorkerContext::new(Arc::new(MemoryTables::new()));
    let e =
        krishiv_exec::physical::instantiate_task(&plan, &worker, &TaskSpec::whole()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(
        e.to_string().contains(
            "a write without a write plan (written by a writer before the write_plan feature)"
        ),
        "{e}"
    );
}

/// A write without a configuration cannot instantiate: format options
/// parse against the session's fallbacks, and silently using defaults
/// would parse them differently than the session's.
#[test]
fn a_write_without_a_configuration_is_refused() {
    let out = tempfile::tempdir().unwrap();
    let store = Arc::new(LocalFileSystem::new_with_prefix(out.path()).unwrap());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("file://out");
    sched.register_store(id, Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let plan = krishiv_exec::physical::PhysicalPlan::new(
        PhysNode::WriteFiles {
            input: Box::new(PhysNode::OneRow),
            write: Some(krishiv_exec::physical::WritePlan {
                format: "parquet".to_owned(),
                options: BTreeMap::new(),
                store: "file://out".to_owned(),
                dir: "result".to_owned(),
                job_uuid: "job-1".to_owned(),
                max_records_per_file: 0,
                target_file_bytes: 0,
            }),
        },
        krishiv_exec::physical::WritePlan::result_schema(),
        krishiv_exec::planner::ExplainNode::leaf("WriteFiles", ""),
        krishiv_exec::physical::ExprEnv {
            ansi: true,
            legacy_statistical_aggregate: false,
            zone: None,
            query_start_micros: 0,
            user: Arc::from("u"),
        },
    );
    let worker = WorkerContext::new(Arc::new(MemoryTables::new()))
        .with_io(io)
        .with_formats(Arc::new(FormatRegistry::with_defaults()));
    let e = instantiate(&plan, &worker).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(
        e.to_string().contains("a write without a configuration"),
        "{e}"
    );
}
