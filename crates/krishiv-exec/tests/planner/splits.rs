// SPDX-License-Identifier: Apache-2.0
//! Scan splits as task data: a scan's results are the same however its
//! files (or relation batches) are dealt across tasks, and version-pinned
//! reads fail when a file changes.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::physical::{
    ExprEnv, PhysNode, PhysicalPlan, ScanId, ScanNode, ScanSplits, TaskSpec, WorkerContext,
    assign_batch_splits, assign_file_splits, instantiate_task, plan_file_partitions,
};
use krishiv_exec::planner::{
    ExplainNode, FileTable, MemoryTable, TableProvider, TableProviderResolver,
};
use krishiv_exec::{
    ExecConfig, ExecutorConfig, QueryContext, QueryId, QueryRunner, TokioExecutor, WorkloadClass,
};
use krishiv_io::csv::CsvFormat;
use krishiv_io::split::SplitConfig;
use krishiv_io::{ColumnPath, FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, StoreId};
use krishiv_memory::{MemoryConfig, RelationStore, TreePool};
use krishiv_plan::{AttrId, BoundColumn, ReadSource, Scan};
use krishiv_types::DataType;
use krishiv_types::batch::Batch;
use object_store::memory::InMemory;
use object_store::path::Path as StorePath;
use object_store::{ObjectStore, ObjectStoreExt};

use super::cell;

/// A table provider that always serves `table`.
#[derive(Debug)]
struct Stub {
    table: Arc<dyn TableProvider>,
}

impl TableProviderResolver for Stub {
    fn provider(&self, _scan: &Scan) -> Result<Arc<dyn TableProvider>, KrishivError> {
        Ok(Arc::clone(&self.table))
    }
}

/// A single-column (`id`, non-null `INT`) scan plan over `table`.
fn scan_plan(table: Arc<dyn TableProvider>, nullable: bool) -> (PhysicalPlan, Arc<Stub>) {
    let scan = Scan {
        source: ReadSource::Table {
            name: vec![
                "spark_catalog".to_owned(),
                "default".to_owned(),
                "t".to_owned(),
            ],
            options: BTreeMap::new(),
        },
        output: vec![BoundColumn {
            id: AttrId(1),
            name: "id".to_owned(),
            data_type: DataType::Integer,
            nullable,
        }],
        version: None,
    };
    let node = PhysNode::Scan(ScanNode {
        id: ScanId(1),
        scan,
        projection: vec![ColumnPath::new("id")],
        filter: None,
        runtime_filters: Vec::new(),
        row_stage: krishiv_exec::RowStageSettings::default(),
    });
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        ArrowType::Int32,
        nullable,
    )]));
    let plan = PhysicalPlan::new(
        node,
        schema,
        ExplainNode::leaf("Scan", ""),
        ExprEnv {
            ansi: true,
            legacy_statistical_aggregate: false,
            zone: None,
            query_start_micros: 0,
            user: Arc::from("u"),
        },
    );
    let tables = Arc::new(Stub { table });
    (plan, tables)
}

/// The plan's only scan id.
fn scan_id(plan: &PhysicalPlan) -> ScanId {
    let mut ids = Vec::new();
    let mut stack = vec![plan.root()];
    while let Some(node) = stack.pop() {
        if let PhysNode::Scan(scan) = node {
            ids.push(scan.id);
        }
        stack.extend(node.children());
    }
    assert_eq!(ids.len(), 1, "one scan: {ids:?}");
    ids[0]
}

/// Runs `plan` as `task` on `worker` to completion.
fn run_as(
    rt: &tokio::runtime::Runtime,
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    task: &TaskSpec,
) -> Result<Vec<RecordBatch>, KrishivError> {
    let node = instantiate_task(plan, worker, task)?;
    let pool: Arc<dyn krishiv_memory::MemoryPool> = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(u64::from(task.partition)),
        WorkloadClass::Batch,
        ExecConfig::default(),
        pool.child(&format!("q{}", task.partition), None),
    );
    let runner = QueryRunner::new(TokioExecutor::new(&ExecutorConfig::default())?);
    rt.block_on(async {
        let mut stream = runner.execute(node, Arc::clone(&query)).await?;
        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            out.push(batch?.into_parts().0);
        }
        query.finished().await;
        Ok(out)
    })
}

/// The `id` column of `batches` as text, sorted.
fn rows_of(batches: &[RecordBatch]) -> Vec<String> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            rows.push(cell(batch.column(0).as_ref(), row));
        }
    }
    rows.sort();
    rows
}

fn int_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int32, false)]))
}

/// The schema of the file tests: file scans report nullable columns.
fn file_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int32, true)]))
}

/// `values` as `Int32` batches of three rows.
fn int_batches(schema: &SchemaRef, values: &[i32]) -> Vec<Batch> {
    values
        .chunks(3)
        .map(|chunk| {
            let data = RecordBatch::try_new(
                Arc::clone(schema),
                vec![Arc::new(Int32Array::from(chunk.to_vec()))],
            )
            .unwrap();
            Batch::new(data)
        })
        .collect()
}

/// Runs the plan whole and as 1..=4 tasks with `splits_for`, and checks the
/// union of each task count equals the single run with no duplicates.
fn check_union(
    rt: &tokio::runtime::Runtime,
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    id: ScanId,
    splits_for: impl Fn(usize) -> Vec<ScanSplits>,
) {
    let whole = rows_of(
        &run_as(rt, plan, worker, &TaskSpec::whole()).unwrap_or_else(|e| panic!("whole: {e}")),
    );
    assert!(!whole.is_empty(), "the test data has rows");
    for tasks in 1..=4_usize {
        let splits = splits_for(tasks);
        assert_eq!(splits.len(), tasks);
        let mut union = Vec::new();
        for (partition, splits) in splits.into_iter().enumerate() {
            let task = TaskSpec {
                partition: u32::try_from(partition).unwrap(),
                partitions: u32::try_from(tasks).unwrap(),
                scans: BTreeMap::from([(id, splits)]),
                ..TaskSpec::whole()
            };
            let batches =
                run_as(rt, plan, worker, &task).unwrap_or_else(|e| panic!("task {partition}: {e}"));
            union.extend(rows_of(&batches));
        }
        union.sort();
        assert_eq!(union, whole, "{tasks} tasks");
    }
}

#[test]
fn batch_ranges_union_to_the_whole_table() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let store = RelationStore::new();
    let schema = int_schema();
    let batches = int_batches(&schema, &(0..14).collect::<Vec<_>>());
    assert_eq!(batches.len(), 5);
    let lease = store.insert(&schema, &batches).unwrap();
    let table = Arc::new(MemoryTable::new(schema, batches).with_relation(lease.clone()));
    let (plan, tables) = scan_plan(table as Arc<dyn TableProvider>, false);
    let id = scan_id(&plan);
    let worker = WorkerContext::new(tables as Arc<dyn TableProviderResolver>)
        .with_relations(Arc::clone(&store));
    // Five batches over up to four tasks: some tasks read several, some
    // (four tasks over five batches) read one, and the union is the table.
    check_union(&rt, &plan, &worker, id, |tasks| {
        assign_batch_splits(lease.reference(), lease.batches(), tasks)
    });
}

#[test]
fn a_scan_without_an_entry_reads_the_whole_table() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let store = RelationStore::new();
    let schema = int_schema();
    let batches = int_batches(&schema, &[1, 2, 3, 4]);
    let lease = store.insert(&schema, &batches).unwrap();
    let table = Arc::new(MemoryTable::new(schema, batches).with_relation(lease));
    let (plan, tables) = scan_plan(table as Arc<dyn TableProvider>, false);
    let worker = WorkerContext::new(tables as Arc<dyn TableProviderResolver>).with_relations(store);
    // No `scans` entry: the scan plans its own splits, exactly as before.
    let whole = run_as(&rt, &plan, &worker, &TaskSpec::whole()).unwrap();
    assert_eq!(rows_of(&whole), ["1", "2", "3", "4"]);
}

#[test]
fn a_batch_split_without_a_relation_store_is_an_internal_error() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let schema = int_schema();
    let table = Arc::new(MemoryTable::new(
        Arc::clone(&schema),
        int_batches(&schema, &[1]),
    ));
    let (plan, tables) = scan_plan(table as Arc<dyn TableProvider>, false);
    let id = scan_id(&plan);
    let worker = WorkerContext::new(tables as Arc<dyn TableProviderResolver>);
    let task = TaskSpec {
        scans: BTreeMap::from([(
            id,
            ScanSplits::Batches {
                relation: krishiv_memory::RelationRef {
                    hash: krishiv_memory::RelationHash::from_bytes([7; 32]),
                },
                range: 0..1,
            },
        )]),
        ..TaskSpec::whole()
    };
    let e = run_as(&rt, &plan, &worker, &task).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert!(e.to_string().contains("relation store"), "{e}");
}

#[test]
fn a_batch_range_past_the_batches_is_an_internal_error() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let store = RelationStore::new();
    let schema = int_schema();
    let batches = int_batches(&schema, &[1, 2]);
    let lease = store.insert(&schema, &batches).unwrap();
    let table = Arc::new(MemoryTable::new(schema, batches).with_relation(lease.clone()));
    let (plan, tables) = scan_plan(table as Arc<dyn TableProvider>, false);
    let id = scan_id(&plan);
    let worker = WorkerContext::new(tables as Arc<dyn TableProviderResolver>).with_relations(store);
    // Only 0..1 is valid over one batch; an empty range reads nothing, and
    // anything past the batches is a coordinator bug.
    let valid = TaskSpec {
        scans: BTreeMap::from([(
            id,
            ScanSplits::Batches {
                relation: lease.reference(),
                range: 0..1,
            },
        )]),
        ..TaskSpec::whole()
    };
    assert_eq!(
        rows_of(&run_as(&rt, &plan, &worker, &valid).unwrap()),
        ["1", "2"]
    );
    let empty = TaskSpec {
        scans: BTreeMap::from([(
            id,
            ScanSplits::Batches {
                relation: lease.reference(),
                range: 1..1,
            },
        )]),
        ..TaskSpec::whole()
    };
    assert!(rows_of(&run_as(&rt, &plan, &worker, &empty).unwrap()).is_empty());
    for range in [0..2_usize, 1..2_usize, 2..3_usize] {
        let task = TaskSpec {
            scans: BTreeMap::from([(
                id,
                ScanSplits::Batches {
                    relation: lease.reference(),
                    range: range.clone(),
                },
            )]),
            ..TaskSpec::whole()
        };
        let e = run_as(&rt, &plan, &worker, &task).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{range:?}: {e}");
    }
}

/// Files in an in-memory store: `put` returns a new ETag every time, so
/// overwriting a file changes its version.
struct Files {
    rt: tokio::runtime::Runtime,
    store: Arc<InMemory>,
    id: StoreId,
    io: krishiv_io::IoContext,
    format: Arc<dyn FileFormat>,
    options: FormatOptions,
    config: Arc<Config>,
}

impl Files {
    fn new(options: FormatOptions, config: Config) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let store = Arc::new(InMemory::new());
        let sched = IoScheduler::new(IoConfig::default());
        let id = StoreId::new("mem://bucket");
        sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
        let io = sched.context();
        Self {
            rt,
            store,
            id,
            io,
            format: Arc::new(CsvFormat),
            options,
            config: Arc::new(config),
        }
    }

    /// Writes `files` (name to bytes) and returns their listed refs.
    fn put(&self, files: &[(&str, &str)]) -> Vec<FileRef> {
        let mut refs = Vec::new();
        for (name, body) in files {
            let path = StorePath::from(*name);
            self.rt
                .block_on(
                    self.store
                        .put(&path, object_store::PutPayload::from(body.to_string())),
                )
                .unwrap();
            refs.push(self.head(name));
        }
        refs
    }

    /// Overwrites `name` with `body`: the version changes.
    fn overwrite(&self, name: &str, body: &str) {
        let before = self.head(name).version;
        self.rt
            .block_on(self.store.put(
                &StorePath::from(name),
                object_store::PutPayload::from(body.to_string()),
            ))
            .unwrap();
        assert_ne!(self.head(name).version, before, "a new version");
    }

    fn head(&self, name: &str) -> FileRef {
        let meta = self
            .rt
            .block_on(ObjectStoreExt::head(
                self.store.as_ref(),
                &StorePath::from(name),
            ))
            .unwrap();
        FileRef::from_meta(self.id.clone(), &meta)
    }

    /// A file table over `refs` (in listing order).
    fn table(&self, refs: Vec<FileRef>) -> Arc<FileTable> {
        let schema = file_schema();
        Arc::new(
            self.rt
                .block_on(FileTable::open(
                    Arc::clone(&self.format),
                    self.io.clone(),
                    schema,
                    refs,
                    self.options.clone(),
                    &self.config,
                    2,
                ))
                .unwrap(),
        )
    }

    /// The splits of `table` with `total_slots` worker slots.
    fn splits(
        &self,
        table: &FileTable,
        total_slots: u64,
    ) -> (
        Vec<krishiv_exec::physical::SplitFileRef>,
        Vec<krishiv_io::split::FilePartition>,
    ) {
        use krishiv_exec::physical::SplitFileRef;
        let refs: Vec<SplitFileRef> = table
            .files()
            .iter()
            .map(SplitFileRef::from_file_ref)
            .collect();
        let capabilities = self.format.capabilities(&self.options);
        let cfg = SplitConfig::from_config(&self.config, total_slots);
        let partitions =
            plan_file_partitions(&cfg, &refs, |f| capabilities.is_splittable(f.path.as_ref()));
        (refs, partitions)
    }
}

#[test]
fn file_ranges_union_to_the_single_run() {
    let mut config = Config::new();
    // Small splits: the big file is cut into several byte-range splits.
    config
        .set("spark.sql.files.maxPartitionBytes", "32")
        .unwrap();
    let files = Files::new(FormatOptions::defaults(CsvFormat.options()), config);
    let big: String = (1..=100).map(|i| format!("{i}\n")).collect();
    let refs = files.put(&[
        ("a.csv", big.as_str()),
        ("b.csv", "101\n102\n"),
        ("c.csv", "103\n"),
    ]);
    let table = files.table(refs);
    let (plan, tables) = scan_plan(Arc::clone(&table) as Arc<dyn TableProvider>, true);
    let id = scan_id(&plan);
    let worker = WorkerContext::new(tables as Arc<dyn TableProviderResolver>);
    let (listed, partitions) = files.splits(&table, 2);
    assert!(partitions.len() > 2, "several splits: {partitions:?}");
    check_union(&files.rt, &plan, &worker, id, |tasks| {
        assign_file_splits(&listed, &partitions, tasks)
    });
}

#[test]
fn unsplittable_csv_stays_whole_and_unions() {
    let mut config = Config::new();
    config
        .set("spark.sql.files.maxPartitionBytes", "8")
        .unwrap();
    let mut values = BTreeMap::new();
    values.insert(
        "multiLine".to_owned(),
        krishiv_common::secret::OptionValue::from_user("multiLine", "true"),
    );
    let (options, _) = CsvFormat.options().parse(&values, &config).unwrap();
    let files = Files::new(options, config);
    // `multiLine` files are never cut, so each file reads whole even when
    // it is bigger than the split limit.
    let refs = files.put(&[("a.csv", "1\n2\n3\n4\n"), ("b.csv", "5\n")]);
    let table = files.table(refs);
    let (listed, partitions) = files.splits(&table, 2);
    assert_eq!(
        partitions.len(),
        2,
        "one whole split per file: {partitions:?}"
    );
    let (plan, tables) = scan_plan(table as Arc<dyn TableProvider>, true);
    let id = scan_id(&plan);
    let worker = WorkerContext::new(tables as Arc<dyn TableProviderResolver>);
    check_union(&files.rt, &plan, &worker, id, |tasks| {
        assign_file_splits(&listed, &partitions, tasks)
    });
}

#[test]
fn a_changed_file_fails_with_file_changed_during_read() {
    let files = Files::new(FormatOptions::defaults(CsvFormat.options()), Config::new());
    let refs = files.put(&[("a.csv", "1\n2\n"), ("b.csv", "3\n")]);
    let table = files.table(refs);
    let (listed, partitions) = files.splits(&table, 1);
    // New bytes under an old name: the version changes.
    files.overwrite("a.csv", "10\n20\n30\n40\n");
    let (plan, tables) = scan_plan(table as Arc<dyn TableProvider>, true);
    let id = scan_id(&plan);
    let worker = WorkerContext::new(tables as Arc<dyn TableProviderResolver>);
    let task = TaskSpec {
        partition: 0,
        partitions: 1,
        scans: BTreeMap::from([(
            id,
            assign_file_splits(&listed, &partitions, 1).pop().unwrap(),
        )]),
        ..TaskSpec::whole()
    };
    let e = run_as(&files.rt, &plan, &worker, &task).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivFileChangedDuringRead,
        "{e}"
    );
}

#[test]
fn more_tasks_than_splits_reads_nothing_extra() {
    let files = Files::new(FormatOptions::defaults(CsvFormat.options()), Config::new());
    let refs = files.put(&[("only.csv", "1\n")]);
    let table = files.table(refs);
    let (listed, partitions) = files.splits(&table, 1);
    assert_eq!(partitions.len(), 1);
    let (plan, tables) = scan_plan(table as Arc<dyn TableProvider>, true);
    let id = scan_id(&plan);
    let worker = WorkerContext::new(tables as Arc<dyn TableProviderResolver>);
    // Four tasks, one split: three tasks read nothing, the union is the row.
    check_union(&files.rt, &plan, &worker, id, |tasks| {
        assign_file_splits(&listed, &partitions, tasks)
    });
}

#[test]
fn zero_files_read_nothing() {
    let files = Files::new(FormatOptions::defaults(CsvFormat.options()), Config::new());
    let table = files.table(Vec::new());
    let (listed, partitions) = files.splits(&table, 2);
    assert!(partitions.is_empty());
    let (plan, tables) = scan_plan(table as Arc<dyn TableProvider>, true);
    let id = scan_id(&plan);
    let worker = WorkerContext::new(tables as Arc<dyn TableProviderResolver>);
    for tasks in [1_usize, 3] {
        for (partition, splits) in assign_file_splits(&listed, &partitions, tasks)
            .into_iter()
            .enumerate()
        {
            let task = TaskSpec {
                partition: u32::try_from(partition).unwrap(),
                partitions: u32::try_from(tasks).unwrap(),
                scans: BTreeMap::from([(id, splits)]),
                ..TaskSpec::whole()
            };
            let got = run_as(&files.rt, &plan, &worker, &task).unwrap();
            assert!(rows_of(&got).is_empty());
        }
    }
}

#[test]
fn splits_of_the_wrong_kind_are_rejected() {
    // A memory scan reads batches, never files; a file scan reads files,
    // never batches.
    let schema = int_schema();
    let memory = MemoryTable::new(Arc::clone(&schema), int_batches(&schema, &[1]));
    let factory = memory.scan().unwrap_or_else(|e| panic!("scan builds: {e}"));
    let Err(e) = factory.with_file_splits(Vec::new(), Vec::new()) else {
        panic!("a memory scan has no files")
    };
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert!(e.to_string().contains("does not read files"), "{e}");

    let files = Files::new(FormatOptions::defaults(CsvFormat.options()), Config::new());
    let table = files.table(files.put(&[("a.csv", "1\n")]));
    let factory = table.scan().unwrap_or_else(|e| panic!("scan builds: {e}"));
    let Err(e) = factory.with_memory_batches(Vec::new()) else {
        panic!("a file scan has no batches")
    };
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert!(
        e.to_string().contains("does not read in-memory batches"),
        "{e}"
    );
}

#[test]
fn two_scans_of_the_same_memory_table_share_the_buffers() {
    use arrow_array::Array as _;
    let store = RelationStore::new();
    let schema = int_schema();
    let batches = int_batches(&schema, &[1, 2, 3, 4, 5, 6]);
    let lease = store.insert(&schema, &batches).unwrap();
    // The table holds only the lease: both scans read the stored batches.
    let table = MemoryTable::from_lease(
        Arc::clone(&schema),
        batches.iter().map(|b| b.num_rows() as u64).sum(),
        lease.clone(),
    );
    let _first_scan = table
        .scan()
        .unwrap_or_else(|e| panic!("first scan builds: {e}"));
    let _second_scan = table
        .scan()
        .unwrap_or_else(|e| panic!("second scan builds: {e}"));
    let first = store.resolve(&lease.hash(), 0..lease.batches()).unwrap();
    let second = store.resolve(&lease.hash(), 0..lease.batches()).unwrap();
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        let a_ptr = a.data().column(0).to_data().buffers()[0]
            .as_slice()
            .as_ptr();
        let b_ptr = b.data().column(0).to_data().buffers()[0]
            .as_slice()
            .as_ptr();
        assert_eq!(
            a_ptr, b_ptr,
            "two scans share the stored arrays instead of decoding per scan"
        );
    }
}
