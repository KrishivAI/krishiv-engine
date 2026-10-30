// SPDX-License-Identifier: Apache-2.0
//! The executor's file scan and write over the mock format, end to end:
//! split planning, `_metadata` columns, pushdowns, footer pipelining, single
//! flight footer reads, prefix pacing, and prefetch hiding store latency
//! (measured on Tokio's paused clock, never the wall clock).
#![cfg(test)]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use futures::future::BoxFuture;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::testing::Break;
use krishiv_exec::{
    ExecConfig, ExecNode, FileScan, FileSelections, FileWrite, FileWriteSink, FilterPushdown,
    FnPredicate, GlobalState, LocalState, MemoryScan, PipelineBuilder, PushedFilter, QueryContext,
    QueryId, QueryRunner, ScanPushdown, ScanPushdowns, Sink, SinkStatus, Source, TaskContext,
    TokioExecutor, WorkloadClass,
};
use krishiv_io::io::FaultStore;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::split::{FilePartition, FileSplit, SplitConfig};
use krishiv_io::{
    ByteRange, CmpOp, ColumnPath, DirectoryWrite, FileFormat, FileRef, IoConfig, IoCounters,
    IoScheduler, MetadataCols, MetadataNeeds, PushdownExpr, RowSelection, SaveMode, ScalarValue,
    ScanOptions, StoreId, StorePolicy, WriteOptions,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_testkit::file_format::mock::MockFormat;
use krishiv_testkit::file_format::{Env, FormatFixture, MockFixture, data};
use krishiv_types::batch::Batch;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("s", DataType::Utf8View, true),
    ]))
}

/// `rows` rows with `i` = `first..first + rows`.
fn sorted(rows: usize, first: i32) -> RecordBatch {
    let b = data::batch(&schema(), rows, 3).unwrap();
    let i: arrow_array::Int32Array = (first..first + i32::try_from(rows).unwrap())
        .map(Some)
        .collect();
    RecordBatch::try_new(schema(), vec![Arc::new(i), Arc::clone(b.column(1))]).unwrap()
}

fn keep_all() -> Arc<dyn krishiv_exec::Predicate> {
    Arc::new(FnPredicate(|b: &RecordBatch| {
        Ok(BooleanArray::from(vec![true; b.num_rows()]))
    }))
}

fn splits(max_partition_bytes: u64) -> SplitConfig {
    SplitConfig {
        max_partition_bytes,
        open_cost_in_bytes: 0,
        min_partition_num: 1,
        max_partition_num: None,
    }
}

/// As [`splits`], with an open cost that keeps every split one morsel
/// (the floor of a morsel's size): for tests of what happens inside one.
fn whole_splits(max_partition_bytes: u64) -> SplitConfig {
    SplitConfig {
        open_cost_in_bytes: 1 << 40,
        ..splits(max_partition_bytes)
    }
}

struct Leaf(Arc<dyn Source>);

impl ExecNode for Leaf {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> krishiv_exec::Result<()> {
        b.set_source(Arc::clone(&self.0)).map(drop)
    }
}

async fn run(
    pool: &Arc<dyn MemoryPool>,
    plan: Arc<dyn ExecNode>,
    slots: usize,
) -> Result<Vec<RecordBatch>, KrishivError> {
    run_with(pool, plan, slots, ExecConfig::default()).await
}

async fn run_with(
    pool: &Arc<dyn MemoryPool>,
    plan: Arc<dyn ExecNode>,
    slots: usize,
    config: ExecConfig,
) -> Result<Vec<RecordBatch>, KrishivError> {
    let runner = QueryRunner::new(TokioExecutor::deterministic(slots)?);
    let query = QueryContext::new(QueryId(7), WorkloadClass::Batch, config, Arc::clone(pool));
    let mut stream = runner.execute(plan, Arc::clone(&query)).await?;
    let mut out = Vec::new();
    let mut error = None;
    while let Some(b) = stream.next().await {
        match b {
            Ok(b) => out.push(b.into_parts().0),
            Err(e) => {
                error.get_or_insert(e);
            }
        }
    }
    query.finished().await;
    error.map_or(Ok(out), Err)
}

/// One task's write of the suite store under `dir`.
fn write_options(dir: &str, task: u32, max_records_per_file: u64) -> WriteOptions {
    WriteOptions {
        store: StoreId::new("memory://suite"),
        dir: Path::from(dir),
        task,
        job_uuid: "u".into(),
        max_records_per_file,
        target_file_bytes: 0,
    }
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// Three files written with the mock format (100-row units).
fn three_files(env: &Env) -> Vec<FileRef> {
    let fx = MockFixture::default();
    (0..3)
        .map(|k| {
            env.write(
                &fx,
                &format!("t/f{k}"),
                &schema(),
                &[sorted(1000, k * 1000)],
                &fx.unit_rows(100),
            )
            .unwrap()
        })
        .collect()
}

fn scan_of(env: &Env, files: Vec<FileRef>, cfg: SplitConfig, scan: ScanOptions) -> FileScan {
    let fx = MockFixture::default();
    FileScan::new(
        fx.format(),
        env.io.clone(),
        schema(),
        files,
        env.options(&fx, &[]).unwrap(),
        scan,
        cfg,
    )
}

#[test]
fn file_scan_reads_planned_partitions_with_metadata_columns() {
    let env = Env::new(None).unwrap();
    let files = three_files(&env);
    let scan = scan_of(
        &env,
        files.clone(),
        splits(files[0].size / 2),
        ScanOptions::default(),
    );
    let metadata = MetadataCols {
        file_path: true,
        file_name: true,
        file_size: true,
        file_block_start: true,
        file_block_length: true,
        file_modification_time: true,
        row_index: true,
    };
    let source = scan
        .build(ScanPushdowns {
            projection: Some(vec![ColumnPath::new("s"), ColumnPath::new("i")]),
            metadata,
            ..ScanPushdowns::default()
        })
        .unwrap();
    let out_schema = source.schema();
    assert_eq!(
        out_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["s", "i", "_metadata"]
    );
    let DataType::Struct(meta_fields) = out_schema.field(2).data_type() else {
        panic!("_metadata is not a struct");
    };
    assert_eq!(
        meta_fields
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        [
            "file_path",
            "file_name",
            "file_size",
            "file_block_start",
            "file_block_length",
            "file_modification_time",
            "row_index"
        ]
    );
    let out = env
        .block_on(run(&env.query, Arc::new(Leaf(source)), 3))
        .unwrap();
    assert_eq!(total_rows(&out), 3000);
    // Every (file, row_index) once; i = 1000 * file + row_index.
    let mut seen = BTreeSet::new();
    let mut blocks = BTreeSet::new();
    for b in &out {
        let i = b.column(1).as_primitive::<Int32Type>();
        let m = b.column(2).as_struct();
        let path = m.column(0).as_string_view();
        let name = m.column(1).as_string_view();
        let size = m.column(2).as_primitive::<Int64Type>();
        let start = m.column(3).as_primitive::<Int64Type>();
        let len = m.column(4).as_primitive::<Int64Type>();
        let mtime = m.column(5).as_primitive::<TimestampMicrosecondType>();
        let row = m.column(6).as_primitive::<Int64Type>();
        for r in 0..b.num_rows() {
            let k = files
                .iter()
                .position(|f| f.path.filename() == Some(name.value(r)))
                .unwrap();
            let f = &files[k];
            assert_eq!(path.value(r), format!("memory://suite/{}", f.path));
            assert_eq!(size.value(r), i64::try_from(f.size).unwrap());
            assert_eq!(Some(mtime.value(r)), f.modified_micros);
            assert_eq!(
                i64::from(i.value(r)),
                1000 * i64::try_from(k).unwrap() + row.value(r)
            );
            assert!(seen.insert((k, row.value(r))));
            blocks.insert((k, start.value(r), len.value(r)));
        }
    }
    assert_eq!(seen.len(), 3000);
    // Each file was read as two splits.
    assert_eq!(blocks.len(), 6, "{blocks:?}");
    assert_eq!(env.root.reserved(), 0);
}

#[test]
fn pushdowns_become_one_immutable_request() {
    let env = Env::new(None).unwrap();
    let files = three_files(&env);
    let scan = scan_of(&env, files, splits(1 << 30), ScanOptions::default());
    let typed = PushedFilter::typed(
        keep_all(),
        PushdownExpr::cmp(ColumnPath::new("i"), CmpOp::Lt, ScalarValue::Int32(150)),
    );
    let opaque = PushedFilter::opaque(keep_all());
    // The Filter above a file scan is always kept.
    assert_eq!(scan.filter_pushdown(&typed), FilterPushdown::Inexact);
    assert_eq!(scan.filter_pushdown(&opaque), FilterPushdown::Unsupported);
    assert!(scan.supports_selection());
    assert_eq!(scan.table_schema(), schema());
    assert!(format!("{scan:?}").contains("mock"));
    #[derive(Debug)]
    struct FirstTen;
    impl FileSelections for FirstTen {
        fn selection(&self, _file: &FileRef) -> Option<RowSelection> {
            Some(RowSelection::excluding(10, []))
        }
    }
    let source = scan
        .source(ScanPushdowns {
            filters: vec![typed.clone(), opaque],
            // Ignored with filters (the Filter above may drop rows).
            limit: Some(1),
            ..ScanPushdowns::default()
        })
        .unwrap();
    assert!(format!("{source:?}").contains("FileScanSource"));
    let req = source.request();
    assert!(matches!(req.predicate, Some(PushdownExpr::Cmp { .. })));
    assert_eq!(req.limit, None);
    assert_eq!(source.partitions().len(), 1);
    let source: Arc<dyn Source> = Arc::new(source);
    let out = env
        .block_on(run(&env.query, Arc::new(Leaf(Arc::clone(&source))), 2))
        .unwrap();
    // Statistics pruning kept a superset of i < 150: two units of the first file.
    assert!(total_rows(&out) >= 150);
    assert!(total_rows(&out) <= 200, "{} rows", total_rows(&out));
    // Several typed filters are one conjunction; a selection per file.
    let both = scan
        .build(ScanPushdowns {
            filters: vec![typed.clone(), typed],
            selection: Some(Arc::new(FirstTen)),
            ..ScanPushdowns::default()
        })
        .unwrap();
    let out = env
        .block_on(run(&env.query, Arc::new(Leaf(both)), 2))
        .unwrap();
    assert_eq!(total_rows(&out), 10);
    // A limit without filters stops the scan early (whole batches).
    let limited = scan
        .build(ScanPushdowns {
            limit: Some(150),
            ..ScanPushdowns::default()
        })
        .unwrap();
    let out = env
        .block_on(run(&env.query, Arc::new(Leaf(limited)), 1))
        .unwrap();
    assert!(
        (150..=300).contains(&total_rows(&out)),
        "{}",
        total_rows(&out)
    );
    // Unknown columns are refused when the scan is built.
    let bad = scan.build(ScanPushdowns {
        projection: Some(vec![ColumnPath::new("nope")]),
        ..ScanPushdowns::default()
    });
    assert!(bad.is_err());
    assert_eq!(env.root.reserved(), 0);
}

#[test]
fn file_write_writes_and_returns_files_through_the_executor() {
    let env = Env::new(None).unwrap();
    let fx = MockFixture::default();
    let input = MemoryScan::try_new(
        schema(),
        (0..4).map(|k| Batch::new(sorted(500, k * 500))).collect(),
    )
    .unwrap();
    let sink = Arc::new(
        FileWriteSink::try_new(
            fx.format(),
            env.io.clone(),
            schema(),
            env.options(&fx, &fx.unit_rows(100)).unwrap(),
            write_options("out", 3, 700),
        )
        .unwrap(),
    );
    assert!(format!("{sink:?}").contains("FileWriteSink"));
    let plan = Arc::new(FileWrite {
        input: Arc::new(Leaf(Arc::new(input))),
        sink: Arc::clone(&sink),
    });
    assert!(format!("{plan:?}").contains("FileWrite"));
    let out = env.block_on(run(&env.query, plan, 2)).unwrap();
    let files = sink.files();
    assert_eq!(total_rows(&out), files.len());
    assert_eq!(files.iter().map(|f| f.rows).sum::<u64>(), 2000);
    assert!(files.iter().all(|f| f.rows <= 700));
    let listed: Vec<String> = out[0]
        .column(0)
        .as_string_view()
        .iter()
        .map(|p| p.unwrap().to_owned())
        .collect();
    assert_eq!(
        listed,
        files.iter().map(|f| f.path.to_string()).collect::<Vec<_>>()
    );
    // Read back through the file scan.
    let back: Vec<FileRef> = files
        .iter()
        .map(|f| env.block_on(env.file(&f.path)).unwrap())
        .collect();
    let scan = scan_of(&env, back, splits(1 << 30), ScanOptions::default())
        .build(ScanPushdowns::default())
        .unwrap();
    let rows = env
        .block_on(run(&env.query, Arc::new(Leaf(scan)), 2))
        .unwrap();
    let mut values: Vec<i32> = rows
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    values.sort_unstable();
    assert_eq!(values, (0..2000).collect::<Vec<_>>());
    assert_eq!(env.root.reserved(), 0);
}

#[test]
fn a_failed_write_reports_the_writer_error_and_leaves_nothing() {
    let env = Env::new(None).unwrap();
    let fx = MockFixture::default();
    let input = MemoryScan::try_new(
        schema(),
        (0..20).map(|k| Batch::new(sorted(100, k * 100))).collect(),
    )
    .unwrap();
    let sink = Arc::new(
        FileWriteSink::try_new(
            fx.format(),
            env.io.clone(),
            schema(),
            env.options(&fx, &[]).unwrap(),
            write_options("bad", 0, 100),
        )
        .unwrap(),
    );
    env.store.fail_next_puts(1000);
    let plan = Arc::new(FileWrite {
        input: Arc::new(Leaf(Arc::new(input))),
        sink,
    });
    let e = env.block_on(run(&env.query, plan, 1)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TaskWriteFailed);
    let left = env.block_on(env.inner.list(Some(&Path::from("bad"))).collect::<Vec<_>>());
    assert!(left.is_empty());
    assert_eq!(env.root.reserved(), 0);
    // Weighted (change-stream) rows are refused.
    let (data, _) = Batch::new(sorted(2, 0)).into_parts();
    let weighted = Batch::with_weights(
        data,
        krishiv_types::batch::Weights::Column(arrow_array::Int64Array::from(vec![1, -1])),
    )
    .unwrap();
    let input = MemoryScan::try_new(schema(), vec![weighted]).unwrap();
    let sink = Arc::new(
        FileWriteSink::try_new(
            fx.format(),
            env.io.clone(),
            schema(),
            env.options(&fx, &[]).unwrap(),
            write_options("w", 0, 0),
        )
        .unwrap(),
    );
    let plan = Arc::new(FileWrite {
        input: Arc::new(Leaf(Arc::new(input))),
        sink,
    });
    let e = env.block_on(run(&env.query, plan, 1)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

/// A harness whose scheduler runs requests on its own (paused) runtime, as
/// the engine's I/O runtime, behind a store with latency.
struct Timed {
    rt: tokio::runtime::Runtime,
    store: Arc<FaultStore>,
    inner: Arc<InMemory>,
    sched: Arc<IoScheduler>,
    pool: Arc<dyn MemoryPool>,
}

impl Timed {
    fn new(policy: StorePolicy) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let inner = Arc::new(InMemory::new());
        let store = FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
        let sched = IoScheduler::with_runtime(IoConfig::default(), rt.handle().clone());
        sched.register_store_with(
            StoreId::new("memory://suite"),
            Arc::clone(&store) as Arc<dyn ObjectStore>,
            policy,
        );
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        Self {
            rt,
            store,
            inner,
            sched,
            pool,
        }
    }

    /// Writes `rows` sorted rows as `name` (100-row units).
    fn file(&self, name: &str, rows: usize) -> FileRef {
        let fx = MockFixture::default();
        let io = self.sched.context();
        let mut user = std::collections::BTreeMap::new();
        for (k, v) in fx.unit_rows(100) {
            user.insert(
                k.clone(),
                krishiv_common::secret::OptionValue::from_user(&k, v),
            );
        }
        let o = fx
            .format()
            .options()
            .parse(&user, &krishiv_common::config::Config::new())
            .unwrap()
            .0;
        self.rt.block_on(async {
            let mut enc = fx.format().writer(&io, schema(), &o).await.unwrap();
            enc.write(&sorted(rows, 0)).unwrap();
            let mut up = io
                .upload(&StoreId::new("memory://suite"), Path::from(name))
                .unwrap();
            while let Some(b) = enc.take_bytes() {
                up.write(b).await.unwrap();
            }
            up.write(enc.finish().unwrap().tail).await.unwrap();
            up.finish().await.unwrap();
            let meta = object_store::ObjectStoreExt::head(self.inner.as_ref(), &Path::from(name))
                .await
                .unwrap();
            FileRef::from_meta(StoreId::new("memory://suite"), &meta)
        })
    }
}

/// Downstream work after every batch: the driver is busy for `0` before
/// it pulls the next batch (a sink that blocks on a timer).
struct Busy(Duration);

impl Sink for Busy {
    fn global_state(&self, _ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        Ok(Arc::new(()))
    }
    fn local_state(
        &self,
        _ctx: &TaskContext,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        Ok(Box::new(()))
    }
    fn sink(
        &self,
        _input: Batch,
        _local: &mut LocalState,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<SinkStatus> {
        Ok(SinkStatus::Blocked(Box::pin(tokio::time::sleep(self.0))))
    }
    fn combine(&self, _local: LocalState, _global: &GlobalState) -> krishiv_exec::Result<()> {
        Ok(())
    }
    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, krishiv_exec::Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

impl Source for Busy {
    fn schema(&self) -> SchemaRef {
        schema()
    }
    fn global_state(&self, _ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        Ok(Arc::new(()))
    }
    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }
    fn local_state(
        &self,
        _ctx: &TaskContext,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        Ok(Box::new(()))
    }
    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        _global: &'a GlobalState,
    ) -> BoxFuture<'a, krishiv_exec::Result<Option<Batch>>> {
        Box::pin(async { Ok(None) })
    }
}

/// Acceptance (P2-T117b): with prefetch, the store's latency overlaps the
/// driver's downstream work instead of adding to it.
#[test]
fn prefetch_hides_store_latency() {
    const UNITS: u32 = 20;
    let latency = Duration::from_millis(20);
    let work = Duration::from_millis(30);
    let elapsed = |prefetch_bytes: u64| {
        let t = Timed::new(StorePolicy::UNPACED);
        let file = t.file("lat/f", 100 * UNITS as usize);
        t.store.set_latency(latency);
        let fx = MockFixture::default();
        let scan = FileScan::new(
            fx.format(),
            t.sched.context(),
            schema(),
            vec![file],
            krishiv_io::FormatOptions::defaults(fx.format().options()),
            ScanOptions {
                prefetch_bytes,
                ..ScanOptions::default()
            },
            whole_splits(1 << 30),
        )
        .build(ScanPushdowns::default())
        .unwrap();
        let plan = Arc::new(Break(
            Arc::new(Leaf(scan)) as Arc<dyn ExecNode>,
            Arc::new(Busy(work)),
            false,
        ));
        t.rt.block_on(async {
            let start = tokio::time::Instant::now();
            let out = run(&t.pool, plan, 1).await.unwrap();
            assert!(out.is_empty());
            start.elapsed()
        })
    };
    let serial = elapsed(0);
    let overlapped = elapsed(64 << 20);
    // Without prefetch every unit waits for its fetch, then for the work.
    assert!(serial >= (latency + work) * UNITS, "{serial:?}");
    // With prefetch the next unit is fetched during the work: only the
    // footer and the first unit wait for the store.
    assert!(overlapped <= work * UNITS + latency * 3, "{overlapped:?}");
}

/// The footer of the next file in a partition is read ahead (on the I/O
/// runtime) while the current file decodes: every footer is read once,
/// and every open after the first finds its footer read.
#[test]
fn footers_are_read_ahead_across_a_partition() {
    let t = Timed::new(StorePolicy::UNPACED);
    let files: Vec<FileRef> = (0..4).map(|k| t.file(&format!("p/f{k}"), 300)).collect();
    t.store.set_latency(Duration::from_millis(10));
    let fx = MockFixture::default();
    let o = krishiv_io::FormatOptions::defaults(fx.format().options());
    let partition = FilePartition {
        index: 0,
        splits: files
            .iter()
            .enumerate()
            .map(|(file, f)| FileSplit {
                file,
                range: ByteRange::whole(f.size),
            })
            .collect(),
    };
    let ctx = t.sched.context();
    let scan = FileScan::new(
        fx.format(),
        ctx.clone(),
        schema(),
        files,
        o,
        ScanOptions::default(),
        whole_splits(1 << 30),
    )
    .with_partitions(vec![partition])
    .source(ScanPushdowns::default())
    .unwrap();
    let scan = Arc::new(scan);
    let out =
        t.rt.block_on(run(
            &t.pool,
            Arc::new(Leaf(Arc::clone(&scan) as Arc<dyn Source>)),
            1,
        ))
        .unwrap();
    assert_eq!(total_rows(&out), 1200);
    assert_eq!(t.store.stats().tail_gets, 4, "one footer read per file");
    assert_eq!(IoCounters::get(&ctx.counters().cache_misses), 4);
    assert!(IoCounters::get(&ctx.counters().cache_hits) >= 3);
    let m = scan.metrics();
    assert_eq!((m.splits, m.rows), (4, 1200));
    assert!(m.units > 0 && m.bytes > 0);
}

/// One split of a file whose units are many is cut into about four
/// morsels per driver; the drivers read every unit once and the rows come
/// back in file order, with the planned split as `_metadata`'s block.
#[test]
fn a_split_is_cut_into_morsels_for_every_driver() {
    let t = Timed::new(StorePolicy::UNPACED);
    let file = t.file("m/f", 4000);
    let fx = MockFixture::default();
    let o = krishiv_io::FormatOptions::defaults(fx.format().options());
    let size = file.size;
    let scan = FileScan::new(
        fx.format(),
        t.sched.context(),
        schema(),
        vec![file],
        o,
        ScanOptions::default(),
        splits(1 << 30),
    )
    .source(ScanPushdowns {
        metadata: MetadataCols {
            file_block_start: true,
            file_block_length: true,
            ..MetadataCols::default()
        },
        ..ScanPushdowns::default()
    })
    .unwrap();
    assert_eq!(scan.partitions().len(), 1);
    let scan = Arc::new(scan);
    let out =
        t.rt.block_on(run(
            &t.pool,
            Arc::new(Leaf(Arc::clone(&scan) as Arc<dyn Source>)),
            4,
        ))
        .unwrap();
    let ids: Vec<i32> = out
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    assert_eq!(ids, (0..4000).collect::<Vec<i32>>());
    for b in &out {
        let m = b.column(b.num_columns() - 1).as_struct();
        let block = |name: &str| m.column_by_name(name).unwrap().as_primitive::<Int64Type>();
        let (start, len) = (block("file_block_start"), block("file_block_length"));
        assert!(start.values().iter().all(|v| *v == 0));
        assert!(
            len.values()
                .iter()
                .all(|v| *v == i64::try_from(size).unwrap())
        );
    }
    // 4 drivers, 4 morsels each: 16 pieces of the one split.
    let m = scan.metrics();
    assert_eq!((m.splits, m.rows), (16, 4000));
    assert!(size > 16);
}

/// The mock format without selections, row indexes or writing.
#[derive(Debug)]
struct Limited;

impl krishiv_io::FileFormat for Limited {
    fn name(&self) -> &'static str {
        "limited"
    }
    fn options(&self) -> &'static krishiv_io::OptionSchema {
        MockFormat.options()
    }
    fn capabilities(&self, o: &krishiv_io::FormatOptions) -> krishiv_io::Capabilities {
        let mut c = MockFormat.capabilities(o);
        c.selection = false;
        c.row_index = false;
        c.write = false;
        c
    }
    fn file_suffix(&self, o: &krishiv_io::FormatOptions) -> String {
        MockFormat.file_suffix(o)
    }
    fn read_metadata<'a>(
        &'a self,
        io: &'a krishiv_io::IoContext,
        f: &'a FileRef,
        n: MetadataNeeds,
        o: &'a krishiv_io::FormatOptions,
    ) -> BoxFuture<'a, krishiv_io::FormatResult<Arc<dyn krishiv_io::FileMetadata>>> {
        MockFormat.read_metadata(io, f, n, o)
    }
    fn plan_units(
        &self,
        m: &dyn krishiv_io::FileMetadata,
        s: &ByteRange,
        r: &krishiv_io::ScanRequest,
    ) -> krishiv_io::FormatResult<krishiv_io::UnitPlan> {
        MockFormat.plan_units(m, s, r)
    }
    fn open<'a>(
        &'a self,
        io: &'a krishiv_io::IoContext,
        m: Arc<dyn krishiv_io::FileMetadata>,
        u: krishiv_io::UnitPlan,
        r: &'a krishiv_io::ScanRequest,
        o: &'a krishiv_io::FormatOptions,
    ) -> BoxFuture<'a, krishiv_io::FormatResult<Box<dyn krishiv_io::Decoder>>> {
        MockFormat.open(io, m, u, r, o)
    }
    fn writer<'a>(
        &'a self,
        io: &'a krishiv_io::IoContext,
        s: SchemaRef,
        o: &'a krishiv_io::FormatOptions,
    ) -> BoxFuture<'a, krishiv_io::FormatResult<Box<dyn krishiv_io::Encoder>>> {
        MockFormat.writer(io, s, o)
    }
}

/// Pushdowns a format cannot honour are refused when the scan is built;
/// a format that cannot write has no file sink.
#[test]
fn unsupported_pushdowns_and_writes_are_refused() {
    let env = Env::new(None).unwrap();
    let files = three_files(&env);
    let o = env.options(&MockFixture::default(), &[]).unwrap();
    let scan = FileScan::new(
        Arc::new(Limited),
        env.io.clone(),
        schema(),
        files,
        o.clone(),
        ScanOptions::default(),
        splits(1 << 30),
    );
    assert!(!scan.supports_selection());
    #[derive(Debug)]
    struct All;
    impl FileSelections for All {
        fn selection(&self, _file: &FileRef) -> Option<RowSelection> {
            None
        }
    }
    // `source` and `output_schema` refuse the same pushdowns, alike.
    let refused = |p: ScanPushdowns| {
        let a = ScanPushdown::output_schema(&scan, &p).unwrap_err();
        let b = scan.source(p).unwrap_err();
        assert_eq!(a.to_string(), b.to_string());
        assert_eq!(b.error_class(), ErrorClass::KrishivUnsupportedFeature);
    };
    refused(ScanPushdowns {
        selection: Some(Arc::new(All)),
        ..ScanPushdowns::default()
    });
    refused(ScanPushdowns {
        metadata: MetadataCols {
            row_index: true,
            ..MetadataCols::default()
        },
        ..ScanPushdowns::default()
    });
    let e = FileWriteSink::try_new(
        Arc::new(Limited),
        env.io,
        schema(),
        o,
        write_options("x", 0, 0),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

/// `_metadata.file_modification_time` of a file listed without one is an
/// error, never a made-up time.
#[test]
fn a_modification_time_is_never_invented() {
    let env = Env::new(None).unwrap();
    let mut files = three_files(&env);
    files[0].modified_micros = None;
    let metadata = MetadataCols {
        file_modification_time: true,
        ..MetadataCols::default()
    };
    let source = scan_of(&env, files, splits(1 << 30), ScanOptions::default())
        .build(ScanPushdowns {
            metadata,
            ..ScanPushdowns::default()
        })
        .unwrap();
    let e = env
        .block_on(run(&env.query, Arc::new(Leaf(source)), 1))
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert_eq!(env.root.reserved(), 0);
}

/// A write whose last file fails to upload when it is finished fails the
/// query and leaves nothing; the result source has the files' schema.
#[test]
fn a_write_failing_at_finish_leaves_nothing() {
    let env = Env::new(None).unwrap();
    let fx = MockFixture::default();
    let input = MemoryScan::try_new(schema(), vec![Batch::new(sorted(50, 0))]).unwrap();
    let sink = Arc::new(
        FileWriteSink::try_new(
            fx.format(),
            env.io.clone(),
            schema(),
            env.options(&fx, &[]).unwrap(),
            write_options("fin", 0, 0),
        )
        .unwrap(),
    );
    assert_eq!(Source::schema(sink.as_ref()).fields().len(), 3);
    env.store.fail_next_puts(1000);
    let plan = Arc::new(FileWrite {
        input: Arc::new(Leaf(Arc::new(input))),
        sink: Arc::clone(&sink),
    });
    let e = env.block_on(run(&env.query, plan, 1)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TaskWriteFailed);
    assert!(sink.files().is_empty());
    let left = env.block_on(env.inner.list(Some(&Path::from("fin"))).collect::<Vec<_>>());
    assert!(left.is_empty());
    assert_eq!(env.root.reserved(), 0);
}

/// Several drivers of one task, and tasks with adjacent numbers writing to
/// one directory, never write two files under one name.
#[test]
fn concurrent_writers_never_share_a_file_name() {
    let env = Env::new(None).unwrap();
    let fx = MockFixture::default();
    let config = ExecConfig {
        morsel_rows: 100,
        ..ExecConfig::default()
    };
    let mut written = 0;
    for task in [3, 4] {
        let input = MemoryScan::try_new(
            schema(),
            (0..6).map(|k| Batch::new(sorted(100, k * 100))).collect(),
        )
        .unwrap();
        let sink = Arc::new(
            FileWriteSink::try_new(
                fx.format(),
                env.io.clone(),
                schema(),
                env.options(&fx, &[]).unwrap(),
                write_options("same", task, 40),
            )
            .unwrap(),
        );
        let plan = Arc::new(FileWrite {
            input: Arc::new(Leaf(Arc::new(input))),
            sink: Arc::clone(&sink),
        });
        env.block_on(run_with(&env.query, plan, 3, config.clone()))
            .unwrap();
        let files = sink.files();
        assert_eq!(files.iter().map(|f| f.rows).sum::<u64>(), 600);
        written += files.len();
    }
    let listed = env.block_on(
        env.inner
            .list(Some(&Path::from("same")))
            .collect::<Vec<_>>(),
    );
    assert_eq!(listed.len(), written, "a file overwrote another");
}

fn parquet_options(kv: &[(&str, &str)]) -> krishiv_io::FormatOptions {
    let user = kv
        .iter()
        .map(|(k, v)| {
            (
                (*k).to_owned(),
                krishiv_common::secret::OptionValue::from_user(k, (*v).to_owned()),
            )
        })
        .collect();
    ParquetFormat
        .options()
        .parse(&user, &krishiv_common::config::Config::new())
        .unwrap()
        .0
}

/// `df.write.mode(m).parquet(path)` end to end: the directory write begins,
/// the executor writes the files, the commit publishes them.
fn write_directory(
    env: &Env,
    dir: &str,
    mode: SaveMode,
    uuid: &str,
    batches: Vec<RecordBatch>,
) -> Result<Vec<String>, KrishivError> {
    let o = parquet_options(&[("compression", "zstd")]);
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let Some(w) = env.block_on(DirectoryWrite::begin(
        &env.io,
        StoreId::new("memory://suite"),
        Path::from(dir),
        mode,
        uuid,
    ))?
    else {
        return Ok(Vec::new());
    };
    let input = MemoryScan::try_new(schema(), batches.into_iter().map(Batch::new).collect())?;
    let mut options = w.write_options(0);
    options.max_records_per_file = 600;
    let sink = Arc::new(FileWriteSink::try_new(
        Arc::clone(&format),
        env.io.clone(),
        schema(),
        o.clone(),
        options,
    )?);
    let plan = Arc::new(FileWrite {
        input: Arc::new(Leaf(Arc::new(input))),
        sink: Arc::clone(&sink),
    });
    env.block_on(run(&env.query, plan, 2))?;
    let files = env.block_on(w.commit(&env.io, format.as_ref(), schema(), &o, sink.files()))?;
    Ok(files.into_iter().map(|f| f.path.to_string()).collect())
}

fn listed(env: &Env, dir: &str) -> Vec<String> {
    let mut v: Vec<String> = env
        .block_on(
            env.io
                .list(&StoreId::new("memory://suite"), &Path::from(dir)),
        )
        .unwrap()
        .into_iter()
        .map(|m| m.location.to_string())
        .collect();
    v.sort();
    v
}

#[test]
fn parquet_directory_writes_follow_spark_save_modes() {
    let env = Env::new(None).unwrap();
    let files = write_directory(
        &env,
        "pq",
        SaveMode::ErrorIfExists,
        "a",
        vec![sorted(1000, 0), sorted(500, 1000)],
    )
    .unwrap();
    assert!(files.len() >= 3, "{files:?}");
    assert!(
        files.iter().all(|f| f.ends_with(".zstd.parquet")),
        "{files:?}"
    );
    let mut want = files.clone();
    want.push("pq/_SUCCESS".to_owned());
    want.sort();
    assert_eq!(listed(&env, "pq"), want);
    // Every file reads back with the rows written.
    let mut rows = 0;
    for f in &files {
        let file = env.block_on(env.file(&Path::from(f.as_str()))).unwrap();
        let md = env
            .block_on(env.io.read_metadata(
                &ParquetFormat,
                &file,
                MetadataNeeds::default(),
                &parquet_options(&[]),
            ))
            .unwrap();
        rows += match md.statistics().rows {
            krishiv_io::Precision::Exact(n) => n,
            other => panic!("{other:?}"),
        };
    }
    assert_eq!(rows, 1500);
    let e =
        write_directory(&env, "pq", SaveMode::ErrorIfExists, "b", vec![sorted(1, 0)]).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::PathAlreadyExists);
    assert!(
        write_directory(&env, "pq", SaveMode::Ignore, "b", vec![sorted(1, 0)])
            .unwrap()
            .is_empty()
    );
    let appended = write_directory(&env, "pq", SaveMode::Append, "c", vec![sorted(10, 0)]).unwrap();
    assert_eq!(listed(&env, "pq").len(), want.len() + appended.len());
    // Overwrite with no rows: one empty file holding the schema remains.
    let replaced = write_directory(&env, "pq", SaveMode::Overwrite, "d", Vec::new()).unwrap();
    assert_eq!(replaced, ["pq/part-00000-d-c000.zstd.parquet"]);
    assert_eq!(
        listed(&env, "pq"),
        ["pq/_SUCCESS", "pq/part-00000-d-c000.zstd.parquet"]
    );
    assert_eq!(env.root.reserved(), 0);
}

#[test]
fn a_bad_write_setting_fails_when_the_write_is_planned() {
    let env = Env::new(None).unwrap();
    let e = FileWriteSink::try_new(
        Arc::new(ParquetFormat),
        env.io.clone(),
        schema(),
        parquet_options(&[("compression", "bogus")]),
        write_options("x", 0, 0),
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::CodecNotAvailableWithAvailableCodecsSuggestion
    );
    let e = krishiv_io::FileWriter::new(
        &env.io,
        Arc::new(ParquetFormat),
        schema(),
        parquet_options(&[("compression", "lzo")]),
        write_options("x", 0, 0),
        env.reservation("w"),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(listed(&env, "x").is_empty());
}
