// SPDX-License-Identifier: Apache-2.0
//! Yield-discipline enforcement (P2-T061): no compute worker blocks, so a
//! scan awaiting slow I/O never delays a concurrent short query, no poll
//! exceeds the stall threshold across the TPC-H suite, and one very large
//! batch is processed in chunks. Every test runs with the stall detector in
//! `fail_on_stall` mode: any non-yielding poll fails the query, so the
//! assertions below rest on the stall counters, not on wall-clock timing
//! (the wall-clock bounds are generous multiples of the injected delay).
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{
    Array, Date32Array, Decimal128Array, Int32Array, Int64Array, RecordBatch, StringViewArray,
};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, InternalError, KrishivError};
use krishiv_exec::physical::WorkerContext;
use krishiv_exec::planner::{
    DefaultPlanner, MemoryTable, MemoryTables, PhysicalPlanner, PlanningContext, ProviderStatistics,
};
use krishiv_exec::testing::{Break, Count, RangeSource, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, Executor, ExecutorConfig, LocalState, MorselCursor, Operator,
    OperatorOutput, QueryContext, QueryId, QueryRunner, Result, Scheduler as _, Source,
    TaskContext, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};
use krishiv_types::batch::Batch;
use object_store::memory::InMemory;
use object_store::path::Path;

/// A compute-worker poll may never run this long (E01: operators do about a
/// millisecond of work per call). `KRISHIV_YIELD_STALL_MS` overrides it for
/// experiments only.
const STALL_THRESHOLD_MS: u64 = 10;

/// Instrumented builds (`cargo llvm-cov` sets `LLVM_PROFILE_FILE` for the test
/// binary) do the same work more slowly: a slice of 24 ms with no time waiting
/// for a CPU has been seen on a coverage runner. The bound is a property of the
/// uninstrumented code, which the plain test job holds to it; here it scales.
const INSTRUMENTED_STALL_THRESHOLD_MS: u64 = 100;

/// The threshold given the override and the profile-file variable.
fn stall_threshold_for(stall_ms: Option<&str>, profile_file: Option<&std::ffi::OsStr>) -> Duration {
    let default = if profile_file.is_some() {
        INSTRUMENTED_STALL_THRESHOLD_MS
    } else {
        STALL_THRESHOLD_MS
    };
    Duration::from_millis(stall_ms.and_then(|v| v.parse().ok()).unwrap_or(default))
}

fn stall_threshold() -> Duration {
    stall_threshold_for(
        std::env::var("KRISHIV_YIELD_STALL_MS").ok().as_deref(),
        std::env::var_os("LLVM_PROFILE_FILE").as_deref(),
    )
}

#[test]
fn only_instrumented_builds_get_a_longer_stall_threshold() {
    use std::ffi::OsStr;
    let profile = Some(OsStr::new("x.profraw"));
    assert_eq!(stall_threshold_for(None, None), Duration::from_millis(10));
    assert_eq!(
        stall_threshold_for(None, profile),
        Duration::from_millis(100)
    );
    assert_eq!(
        stall_threshold_for(Some("7"), profile),
        Duration::from_millis(7)
    );
}

fn root(limit: usize) -> Arc<TreePool> {
    TreePool::new(&MemoryConfig {
        limit: Some(limit),
        ..MemoryConfig::default()
    })
}

fn query(pool: &Arc<TreePool>, id: u64) -> Arc<QueryContext> {
    QueryContext::new(
        QueryId(id),
        WorkloadClass::Interactive,
        ExecConfig {
            fail_on_stall: true,
            stall_threshold: stall_threshold(),
            ..ExecConfig::default()
        },
        pool.child(&format!("q{id}"), None),
    )
}

fn executor(threads: usize) -> Arc<TokioExecutor> {
    TokioExecutor::new(&ExecutorConfig {
        compute_threads: threads,
        io_threads: 1,
        blocking_threads: threads,
        io_blocking_threads: 2,
    })
    .unwrap()
}

/// Runs `plan` to the end: every batch, or the query's error. Returns once
/// every task has ended.
async fn run(
    runner: &QueryRunner,
    plan: Arc<dyn ExecNode>,
    q: &Arc<QueryContext>,
) -> Result<Vec<Batch>> {
    let mut stream = runner.execute(plan, Arc::clone(q)).await?;
    let mut out = Vec::new();
    let mut error = None;
    while let Some(b) = stream.next().await {
        match b {
            Ok(b) => out.push(b),
            Err(e) => error = Some(e),
        }
    }
    drop(stream);
    q.finished().await;
    error.map_or(Ok(out), Err)
}

fn rows(batches: &[Batch]) -> usize {
    batches.iter().map(Batch::num_rows).sum()
}

fn assert_released(pool: &TreePool) {
    assert_eq!(
        pool.reserved(),
        0,
        "the query's memory is released at its end"
    );
}

fn assert_no_stalls(q: &QueryContext, what: &str) {
    let drivers = q.metrics().drivers;
    assert_eq!(
        drivers.stalls, 0,
        "{what} counted {} stalls (worst poll {:?})",
        drivers.stalls, drivers.max_slice
    );
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

/// An object-store wrapper that adds `delay` to every read: slow I/O that
/// still yields, because the delay is an async timer on the I/O path and
/// never a blocked worker thread.
#[derive(Debug)]
struct ThrottledStore {
    inner: Arc<InMemory>,
    delay: Duration,
}

impl ThrottledStore {
    fn new(delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(InMemory::new()),
            delay,
        })
    }

    async fn put(&self, path: &Path, bytes: Vec<u8>) -> Result<()> {
        object_store::ObjectStoreExt::put(self.inner.as_ref(), path, bytes.into())
            .await
            .map(|_| ())
            .map_err(|e| internal(&e.to_string()))
    }

    async fn get(&self, path: &Path) -> Result<Vec<u8>> {
        tokio::time::sleep(self.delay).await;
        object_store::ObjectStoreExt::get(self.inner.as_ref(), path)
            .await
            .map_err(|e| internal(&e.to_string()))?
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| internal(&e.to_string()))
    }
}

fn encode(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut writer =
        FileWriter::try_new(&mut buf, batch.schema_ref()).map_err(|e| internal(&e.to_string()))?;
    writer.write(batch).map_err(|e| internal(&e.to_string()))?;
    writer.finish().map_err(|e| internal(&e.to_string()))?;
    drop(writer);
    Ok(buf)
}

fn decode(bytes: &[u8]) -> Result<RecordBatch> {
    let mut reader =
        FileReader::try_new(Cursor::new(bytes), None).map_err(|e| internal(&e.to_string()))?;
    reader
        .next()
        .ok_or_else(|| internal("the throttled store held no batch"))?
        .map_err(|e| internal(&e.to_string()))
}

/// A scan of one batch per store key, each read paying the store's delay:
/// slow I/O that awaits instead of blocking.
struct SlowScan {
    store: Arc<ThrottledStore>,
    schema: SchemaRef,
    keys: Vec<Path>,
}

impl SlowScan {
    fn new(store: Arc<ThrottledStore>, schema: SchemaRef, keys: Vec<Path>) -> Arc<Self> {
        Arc::new(Self {
            store,
            schema,
            keys,
        })
    }
}

impl Source for SlowScan {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<krishiv_exec::GlobalState> {
        Ok(Arc::new(MorselCursor::new(self.keys.len(), None)))
    }

    fn max_parallelism(&self, _global: &krishiv_exec::GlobalState) -> usize {
        1
    }

    fn local_state(
        &self,
        _ctx: &TaskContext,
        _global: &krishiv_exec::GlobalState,
    ) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        global: &'a krishiv_exec::GlobalState,
    ) -> futures::future::BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let Some(m) = global
                .downcast_ref::<MorselCursor>()
                .ok_or_else(|| internal("wrong global state for SlowScan"))?
                .claim()
            else {
                return Ok(None);
            };
            let bytes = self.store.get(&self.keys[m]).await?;
            decode(&bytes).map(|b| Some(Batch::new(b)))
        })
    }
}

fn int_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]))
}

/// A scan awaiting slow I/O (+`DELAY` per request) does not stall a
/// concurrent short query on the same one-thread runtime: the short query
/// finishes before the scan (far below the scan's total of `REQUESTS`
/// delays), and neither query counts a stall. The comparison is relative
/// (`short_elapsed < slow_elapsed`), never an absolute wall-clock bound:
/// a loaded runner may take arbitrarily long in real time without any
/// stall.
#[tokio::test]
async fn a_slow_scan_does_not_delay_a_short_query() {
    const REQUESTS: usize = 30;
    const DELAY: Duration = Duration::from_millis(20);

    let exec = executor(1);
    let runner = QueryRunner::new(Arc::clone(&exec) as Arc<dyn Executor>);
    let pool = root(1 << 30);
    let store = ThrottledStore::new(DELAY);
    let schema = int_schema();
    let keys: Vec<Path> = (0..REQUESTS)
        .map(|i| Path::from(format!("part-{i:02}")))
        .collect();
    for key in &keys {
        let values = Int64Array::from_iter_values(0..100);
        let data = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(values) as Arc<dyn Array>],
        )
        .unwrap();
        store.put(key, encode(&data).unwrap()).await.unwrap();
    }
    let slow = Arc::new(Break(
        Arc::new(Scan(SlowScan::new(
            Arc::clone(&store),
            Arc::clone(&schema),
            keys,
        ))),
        Count::new(),
        false,
    ));
    let slow_q = query(&pool, 1);
    let slow_start = Instant::now();
    let mut slow_stream = runner.execute(slow, Arc::clone(&slow_q)).await.unwrap();
    let short_q = query(&pool, 2);
    let short_plan = Arc::new(Break(
        Arc::new(Scan(RangeSource::new(
            10 * krishiv_exec::DEFAULT_BATCH_SIZE as u64,
        ))),
        Count::new(),
        false,
    ));
    let short_start = Instant::now();
    let short = run(&runner, short_plan, &short_q).await.unwrap();
    let short_elapsed = short_start.elapsed();
    assert_eq!(rows(&short), 2, "count and sum of a 10-batch range");
    let mut slow_rows = 0;
    while let Some(b) = slow_stream.next().await {
        slow_rows += b.unwrap().num_rows();
    }
    drop(slow_stream);
    slow_q.finished().await;
    let slow_elapsed = slow_start.elapsed();
    assert_eq!(slow_rows, 2, "count and sum of the throttled scan");
    // The throttle engaged: at least half the injected delays elapsed.
    assert!(
        slow_elapsed >= DELAY * u32::try_from(REQUESTS).unwrap() / 2,
        "the scan took only {slow_elapsed:?}"
    );
    // The short query finished far below the scan's total time: it was
    // not held behind the scan.
    assert!(
        short_elapsed < slow_elapsed,
        "the short query outlasted the scan: {short_elapsed:?} vs {slow_elapsed:?}"
    );
    assert_no_stalls(&short_q, "the short query");
    assert_no_stalls(&slow_q, "the throttled scan");
    assert_released(&pool);
}

/// One TPC-H table as an Arrow schema: the benchmark shape, generated
/// in-process at a tiny scale (correctness is the existing suites' job;
/// here every query must only complete without stalling).
struct TpchTable {
    name: &'static str,
    fields: Vec<(&'static str, DataType)>,
}

fn tpch_tables() -> Vec<TpchTable> {
    let decimal = || DataType::Decimal128(15, 2);
    let string = || DataType::Utf8View;
    vec![
        TpchTable {
            name: "region",
            fields: vec![
                ("r_regionkey", DataType::Int32),
                ("r_name", string()),
                ("r_comment", string()),
            ],
        },
        TpchTable {
            name: "nation",
            fields: vec![
                ("n_nationkey", DataType::Int32),
                ("n_name", string()),
                ("n_regionkey", DataType::Int32),
                ("n_comment", string()),
            ],
        },
        TpchTable {
            name: "supplier",
            fields: vec![
                ("s_suppkey", DataType::Int32),
                ("s_name", string()),
                ("s_address", string()),
                ("s_nationkey", DataType::Int32),
                ("s_phone", string()),
                ("s_acctbal", decimal()),
                ("s_comment", string()),
            ],
        },
        TpchTable {
            name: "part",
            fields: vec![
                ("p_partkey", DataType::Int32),
                ("p_name", string()),
                ("p_mfgr", string()),
                ("p_brand", string()),
                ("p_type", string()),
                ("p_size", DataType::Int32),
                ("p_container", string()),
                ("p_retailprice", decimal()),
                ("p_comment", string()),
            ],
        },
        TpchTable {
            name: "partsupp",
            fields: vec![
                ("ps_partkey", DataType::Int32),
                ("ps_suppkey", DataType::Int32),
                ("ps_availqty", DataType::Int32),
                ("ps_supplycost", decimal()),
                ("ps_comment", string()),
            ],
        },
        TpchTable {
            name: "customer",
            fields: vec![
                ("c_custkey", DataType::Int32),
                ("c_name", string()),
                ("c_address", string()),
                ("c_nationkey", DataType::Int32),
                ("c_phone", string()),
                ("c_acctbal", decimal()),
                ("c_mktsegment", string()),
                ("c_comment", string()),
            ],
        },
        TpchTable {
            name: "orders",
            fields: vec![
                ("o_orderkey", DataType::Int32),
                ("o_custkey", DataType::Int32),
                ("o_orderstatus", string()),
                ("o_totalprice", decimal()),
                ("o_orderdate", DataType::Date32),
                ("o_orderpriority", string()),
                ("o_clerk", string()),
                ("o_shippriority", DataType::Int32),
                ("o_comment", string()),
            ],
        },
        TpchTable {
            name: "lineitem",
            fields: vec![
                ("l_orderkey", DataType::Int32),
                ("l_partkey", DataType::Int32),
                ("l_suppkey", DataType::Int32),
                ("l_linenumber", DataType::Int32),
                ("l_quantity", decimal()),
                ("l_extendedprice", decimal()),
                ("l_discount", decimal()),
                ("l_tax", decimal()),
                ("l_returnflag", string()),
                ("l_linestatus", string()),
                ("l_shipdate", DataType::Date32),
                ("l_commitdate", DataType::Date32),
                ("l_receiptdate", DataType::Date32),
                ("l_shipinstruct", string()),
                ("l_shipmode", string()),
                ("l_comment", string()),
            ],
        },
    ]
}

/// Strings appearing in the 22 queries' literals, so filters and joins match
/// some rows and the operators run over non-empty intermediates.
const STRINGS: [&str; 16] = [
    "ASIA",
    "AMERICA",
    "AFRICA",
    "BRAZIL",
    "FRANCE",
    "GERMANY",
    "ECONOMY ANODIZED STEEL",
    "BUILDING",
    "regular deposits",
    "deliver in person",
    "green",
    "red",
    "SM CASE",
    "AIR",
    "RAIL",
    "1-URGENT",
];

fn tpch_column(field: &Field, rows: usize) -> Result<Arc<dyn Array>> {
    // Key columns hold unique values (`i + 1`), so joins match the way
    // TPC-H's do; anything else would fan out combinatorially on a tiny
    // table and measure skew, not yielding.
    let is_key = field.name().ends_with("key");
    let array: Arc<dyn Array> = match field.data_type() {
        DataType::Int32 if is_key => Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|i| i32::try_from(i + 1).unwrap()),
        )),
        DataType::Int32 => Arc::new(Int32Array::from_iter_values(
            (0..rows).map(|i| i32::try_from(i % 7 + 1).unwrap()),
        )),
        DataType::Utf8View => Arc::new(StringViewArray::from_iter_values(
            (0..rows).map(|i| STRINGS[i % STRINGS.len()]),
        )),
        DataType::Date32 => Arc::new(Date32Array::from_iter_values(
            (0..rows).map(|i| 8_766 + i32::try_from(i % 2_500).unwrap()),
        )),
        DataType::Decimal128(_, _) => Arc::new(
            Decimal128Array::from_iter_values(
                (0..rows).map(|i| i128::try_from(i % 7 + 1).unwrap() * 100),
            )
            .with_precision_and_scale(15, 2)
            .map_err(|e| internal(&e.to_string()))?,
        ),
        other => return Err(internal(&format!("no generator for {other}"))),
    };
    Ok(array)
}

/// Rows per table of the in-process TPC-H data set; `KRISHIV_YIELD_TPCH_ROWS`
/// scales it up for nightly runs.
fn tpch_rows() -> usize {
    std::env::var("KRISHIV_YIELD_TPCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100)
}

fn tpch_session(rows: usize) -> Result<(MemoryCatalog, Arc<MemoryTables>)> {
    let mut catalog = MemoryCatalog::new("spark_catalog");
    let mut tables = MemoryTables::new();
    for table in tpch_tables() {
        let schema: SchemaRef = Arc::new(Schema::new(
            table
                .fields
                .iter()
                .map(|(name, data_type)| Field::new(*name, data_type.clone(), false))
                .collect::<Vec<_>>(),
        ));
        let columns: Vec<Arc<dyn Array>> = schema
            .fields()
            .iter()
            .map(|f| tpch_column(f, rows))
            .collect::<Result<_>>()?;
        let data = RecordBatch::try_new(Arc::clone(&schema), columns)
            .map_err(|e| internal(&e.to_string()))?;
        let table_columns = schema
            .fields()
            .iter()
            .map(|f| {
                Ok(TableColumn {
                    name: f.name().clone(),
                    data_type: krishiv_types::DataType::from_arrow_field(f)
                        .map_err(|e| internal(&e.to_string()))?,
                    nullable: f.is_nullable(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        catalog = catalog.with_table(&["default"], table.name, table_columns, None);
        tables = tables.with_table(
            &["spark_catalog", "default", table.name],
            Arc::new(MemoryTable::new(schema, vec![Batch::new(data)])),
        );
    }
    Ok((catalog, Arc::new(tables)))
}

const TPCH_QUERIES: [&str; 22] = [
    include_str!("../../../bench/queries/tpch/q01.sql"),
    include_str!("../../../bench/queries/tpch/q02.sql"),
    include_str!("../../../bench/queries/tpch/q03.sql"),
    include_str!("../../../bench/queries/tpch/q04.sql"),
    include_str!("../../../bench/queries/tpch/q05.sql"),
    include_str!("../../../bench/queries/tpch/q06.sql"),
    include_str!("../../../bench/queries/tpch/q07.sql"),
    include_str!("../../../bench/queries/tpch/q08.sql"),
    include_str!("../../../bench/queries/tpch/q09.sql"),
    include_str!("../../../bench/queries/tpch/q10.sql"),
    include_str!("../../../bench/queries/tpch/q11.sql"),
    include_str!("../../../bench/queries/tpch/q12.sql"),
    include_str!("../../../bench/queries/tpch/q13.sql"),
    include_str!("../../../bench/queries/tpch/q14.sql"),
    include_str!("../../../bench/queries/tpch/q15.sql"),
    include_str!("../../../bench/queries/tpch/q16.sql"),
    include_str!("../../../bench/queries/tpch/q17.sql"),
    include_str!("../../../bench/queries/tpch/q18.sql"),
    include_str!("../../../bench/queries/tpch/q19.sql"),
    include_str!("../../../bench/queries/tpch/q20.sql"),
    include_str!("../../../bench/queries/tpch/q21.sql"),
    include_str!("../../../bench/queries/tpch/q22.sql"),
];

/// All 22 TPC-H queries over the in-process generator at a tiny scale run
/// without a single stalled poll: with `fail_on_stall` any poll past the
/// threshold is already a query error, and the counters must stay at zero.
#[test]
fn tpch_suite_counts_no_stalls() {
    let rows = tpch_rows();
    let (catalog, tables) = tpch_session(rows).unwrap();
    let ctx = context(
        catalog,
        ResolutionConf {
            ansi: true,
            ..ResolutionConf::default()
        },
    );
    let config = Arc::new(Config::new());
    let exec = executor(4);
    let runner = QueryRunner::new(exec);
    let pool = root(1 << 30);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for (i, sql) in TPCH_QUERIES.iter().enumerate() {
        let q = i + 1;
        let plan =
            krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .unwrap_or_else(|e| panic!("q{q:02}: {e}"));
        let plan: Plan = block_on(DefaultAnalyzer.analyze(&plan, &ctx))
            .unwrap_or_else(|e| panic!("q{q:02}: {e}"))
            .plan;
        let stats = ProviderStatistics(Arc::clone(&tables) as _);
        let optimized = krishiv_optimizer::optimize_with(&plan, &stats)
            .unwrap_or_else(|e| panic!("q{q:02}: {e}"));
        let physical = DefaultPlanner
            .plan(
                &optimized,
                &PlanningContext::new(Arc::clone(&tables) as _, Arc::clone(&config)),
            )
            .unwrap_or_else(|e| panic!("q{q:02}: {e}"));
        let worker = WorkerContext::new(Arc::clone(&tables) as _);
        let query = query(&pool, u64::try_from(q).unwrap());
        let start = Instant::now();
        let out = rt
            .block_on(async {
                let mut stream = runner.submit(physical, worker, Arc::clone(&query)).await?;
                let mut out = Vec::new();
                while let Some(b) = stream.next().await {
                    out.push(b?.into_parts().0);
                }
                query.finished().await;
                Ok::<_, KrishivError>(out)
            })
            .unwrap_or_else(|e| panic!("q{q:02}: {e}"));
        let elapsed = start.elapsed();
        let got: usize = out.iter().map(RecordBatch::num_rows).sum();
        eprintln!("q{q:02}: {got} rows in {elapsed:.2?}");
        assert_no_stalls(&query, &format!("q{q:02}"));
    }
    assert_released(&pool);
}

/// Splits one input batch into `chunk`-row outputs, so one very large batch
/// is processed in bounded slices; the driver yields every 8 outputs.
struct Chunked {
    chunk: usize,
}

impl Operator for Chunked {
    fn schema(&self) -> SchemaRef {
        int_schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(0usize))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        let offset = local
            .downcast_mut::<usize>()
            .ok_or_else(|| internal("wrong local state for Chunked"))?;
        let rows = input.num_rows();
        if *offset >= rows {
            *offset = 0;
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let len = self.chunk.min(rows - *offset);
        let out = input.slice(*offset, len)?;
        *offset += len;
        Ok(OperatorOutput::HaveMoreOutput(out))
    }
}

/// Spins for three stall thresholds of CPU time in one `execute` call: the negative
/// control showing the stall detector trips on an unchunked operator over
/// a large batch.
struct Spin;
/// Spins until this thread has run for `d` of CPU time. The stall detector
/// charges a poll its slice minus the thread's run-queue wait, so a spin
/// timed by the wall clock holds the CPU for less than `d` on a loaded
/// machine (the thread is preempted for part of it) and may not trip it.
#[cfg(unix)]
fn spin_on_cpu(d: Duration) {
    let cpu = || {
        let t = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
        Duration::new(
            t.tv_sec.unsigned_abs(),
            u32::try_from(t.tv_nsec).unwrap_or(0),
        )
    };
    let start = cpu();
    while cpu().saturating_sub(start) < d {
        std::hint::spin_loop();
    }
}

/// Targets without `rustix::time` (Windows) time the spin by the wall clock:
/// the test compiles there, and is only as reliable as a wall-clock spin.
#[cfg(not(unix))]
fn spin_on_cpu(d: Duration) {
    let start = std::time::Instant::now();
    while start.elapsed() < d {
        std::hint::spin_loop();
    }
}

impl Operator for Spin {
    fn schema(&self) -> SchemaRef {
        int_schema()
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        spin_on_cpu(stall_threshold() * 3);
        Ok(OperatorOutput::NeedMoreInput(Some(input.clone())))
    }
}

fn big_batch(rows: usize) -> (SchemaRef, Batch) {
    let schema = int_schema();
    let values = Int64Array::from_iter_values(0..i64::try_from(rows).unwrap());
    let data = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(values) as Arc<dyn Array>],
    )
    .unwrap();
    (schema, Batch::new(data))
}

/// One very large batch is processed in chunks: the query succeeds with no
/// stalls, and the yield count (not time) shows the driver yielded between
/// slices.
#[tokio::test]
async fn a_very_large_batch_is_processed_in_chunks() {
    const ROWS: usize = 200_000;
    const CHUNK: usize = 1_000;
    // 200 outputs; the driver yields once per source batch and every 8
    // extra outputs.
    let runner = QueryRunner::new(executor(2));
    let pool = root(1 << 30);
    let q = query(&pool, 1);
    let (schema, batch) = big_batch(ROWS);
    let scan = Arc::new(Scan(krishiv_exec::testing::Batches::new(
        schema,
        vec![batch],
    )));
    let plan = Arc::new(krishiv_exec::testing::Pipe(
        scan,
        Arc::new(Chunked { chunk: CHUNK }),
    ));
    let out = run(&runner, plan, &q).await.unwrap();
    assert_eq!(rows(&out), ROWS);
    assert_no_stalls(&q, "the chunked operator");
    let yields = q.metrics().drivers.yields;
    assert!(
        yields >= 20,
        "the driver yielded between slices: {yields} yields"
    );
    assert_released(&pool);
}

/// Without chunking, one long `execute` call trips the stall detector: the
/// query fails and the counters record the stall.
#[tokio::test]
async fn an_unchunked_operator_trips_the_stall_detector() {
    let runner = QueryRunner::new(executor(1));
    let pool = root(1 << 30);
    let q = query(&pool, 1);
    let (schema, batch) = big_batch(1_000);
    let scan = Arc::new(Scan(krishiv_exec::testing::Batches::new(
        schema,
        vec![batch],
    )));
    let plan: Arc<dyn ExecNode> = Arc::new(krishiv_exec::testing::Pipe(scan, Arc::new(Spin)));
    let e = run(&runner, plan, &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("without yielding"), "{e}");
    assert!(q.metrics().drivers.stalls >= 1);
    assert_released(&pool);
}
