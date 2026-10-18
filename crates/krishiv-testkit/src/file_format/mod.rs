// SPDX-License-Identifier: Apache-2.0
//! The generic [`FileFormat`] conformance suite (P2-T117): every format
//! (the mock here; Parquet in P2-T080a/T082a; CSV, JSON, Avro, Vortex later)
//! must pass it. One macro call per format:
//!
//! ```ignore
//! krishiv_testkit::file_format_conformance!(parquet, ParquetFixture);
//! ```
//!
//! Checks (each a function over a [`FormatFixture`]):
//! 1. [`round_trip`]: every Spark type incl. nested and a Variant
//!    placeholder, nulls, an empty file, a zero-column (count) scan.
//! 2. [`projection`]: random column subsets and orders, nested paths.
//! 3. [`pruning_soundness`]: random data × random predicates — the pushed
//!    scan returns every matching row (NaN, -0.0, nulls, collations,
//!    out-of-range literals after re-typing, inexact statistics).
//! 4. [`selection_and_row_index`]: `RowSelection` and `row_index`.
//! 5. [`split_invariance`]: random byte-range splits read each row once.
//! 6. [`schema_evolution`]: missing (top-level and nested), widened,
//!    mismatched, case-insensitive, ambiguous, case-sensitive, field ids.
//! 7. [`corrupt_inputs`]: truncated, garbage and flipped bytes are typed
//!    errors, never panics; ignore-corrupt / ignore-missing; a file changed
//!    mid-read.
//! 8. [`memory_accounting`]: a tiny pool completes or fails with a memory
//!    error; nothing is held at the end.
//! 9. [`cancellation`]: dropping a scan or a write at every await leaks
//!    nothing and leaves no multipart upload open.
//! 10. [`io_behaviour`]: transient failures retried, one tail read per
//!     footer (then the cache), counters match the store, no request
//!     bypasses the scheduler.
//! 11. [`options`]: unknown-option warning and strict mode, invalid values,
//!     conf fallbacks, a secret never in `Debug` or cache keys.
//! 12. [`writer`]: Spark file names, `maxRecordsPerFile` and target-size
//!     rollover, writer statistics = reader statistics, abort leaves
//!     nothing visible.
//!
//! Split-planning parity with Spark is format-independent and checked in
//! `krishiv-io` against Spark 4.2 goldens.

pub mod data;
pub mod drive;
pub mod mock;
pub mod parquet;

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Waker};
use std::time::Duration;

use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use futures::StreamExt;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::OptionValue;
use krishiv_io::io::cache::CacheKey;
use krishiv_io::io::{FaultStore, IoCounters};
use krishiv_io::{
    ByteRange, ColumnPath, FIELD_ID_KEY, FieldMapping, FileFormat, FileRef, FormatOptions,
    IoConfig, IoContext, IoScheduler, MetadataNeeds, OptionKind, OptionWarning, ROW_INDEX_COLUMN,
    RowSelection, ScanOptions, ScanRequest, ScanStream, StoreId, WriteOptions,
};
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;

use self::data::Rng;

/// Result of a check: `Err` holds the failure.
pub type SuiteResult<T = ()> = Result<T, String>;

macro_rules! ensure {
    ($cond:expr, $($msg:tt)+) => {
        if !$cond {
            return Err(format!($($msg)+));
        }
    };
}

/// A format under test.
pub trait FormatFixture: Send + Sync {
    /// The format.
    fn format(&self) -> Arc<dyn FileFormat>;
    /// Options for every read and write (e.g. a decoder variant).
    fn options(&self) -> Vec<(String, String)> {
        Vec::new()
    }
    /// Write options that put about `rows` rows in each unit (row group),
    /// so files have several units.
    fn unit_rows(&self, rows: usize) -> Vec<(String, String)>;
    /// Whether the format stores `dt` (the round trip skips the others).
    fn supports(&self, dt: &DataType) -> bool {
        let _ = dt;
        true
    }
    /// Named write-option sets that produce format-specific statistics edge
    /// cases (truncated bounds, legacy-writer statistics, INT96): pruning
    /// must stay sound on them.
    fn statistics_edge_cases(&self) -> Vec<(String, Vec<(String, String)>)> {
        Vec::new()
    }
    /// A secret option that decodes metadata (a footer key), if the format
    /// has one.
    fn metadata_key_option(&self) -> Option<&'static str> {
        None
    }
    /// How scans and writes run: the framework's loops ([`drive::Direct`],
    /// the default) or the executor ([`drive::ThroughExecutor`]).
    fn driver(&self) -> Arc<dyn drive::Driver> {
        Arc::new(drive::Direct)
    }
}

/// The harness: an in-memory store behind a fault injector, a scheduler, a
/// memory pool and a current-thread runtime with paused time.
pub struct Env {
    rt: tokio::runtime::Runtime,
    /// The scheduler.
    pub sched: Arc<IoScheduler>,
    /// The query's context.
    pub io: IoContext,
    /// The fault injector in front of the store.
    pub store: Arc<FaultStore>,
    /// The store itself (for setup outside the scheduler).
    pub inner: Arc<InMemory>,
    /// The memory pool root.
    pub root: Arc<TreePool>,
    /// The query pool.
    pub query: Arc<dyn MemoryPool>,
    id: StoreId,
    session: Config,
}

impl std::fmt::Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env")
            .field("sched", &self.sched)
            .finish_non_exhaustive()
    }
}

impl Env {
    /// A harness with a memory limit.
    ///
    /// # Errors
    /// If the runtime cannot be built.
    pub fn new(memory: Option<usize>) -> SuiteResult<Self> {
        Self::with_config(memory, IoConfig::default())
    }

    /// A harness with a memory limit and I/O settings.
    ///
    /// # Errors
    /// If the runtime cannot be built.
    pub fn with_config(memory: Option<usize>, config: IoConfig) -> SuiteResult<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .map_err(|e| e.to_string())?;
        let inner = Arc::new(InMemory::new());
        let store = FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
        let sched = IoScheduler::new(config);
        let id = StoreId::new("memory://suite");
        sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
        let root = TreePool::new(&MemoryConfig {
            limit: memory,
            ..MemoryConfig::default()
        });
        let query = root.child("query", None);
        // The metadata cache is process memory: its own pool, so the query
        // pool is empty after every scan.
        let process: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        sched.attach_memory(&process);
        Ok(Self {
            rt,
            io: sched.context().with_memory(Arc::clone(&query)),
            sched,
            store,
            inner,
            root,
            query,
            id,
            session: Config::new(),
        })
    }

    /// Runs `f` to completion on the harness runtime.
    pub fn block_on<F: Future>(&self, f: F) -> F::Output {
        self.rt.block_on(f)
    }

    /// The session config (for conf fallbacks).
    pub fn session_mut(&mut self) -> &mut Config {
        &mut self.session
    }

    /// Parsed options: the fixture's plus `extra`.
    ///
    /// # Errors
    /// Option errors.
    pub fn options(
        &self,
        fx: &dyn FormatFixture,
        extra: &[(String, String)],
    ) -> SuiteResult<FormatOptions> {
        let mut user = BTreeMap::new();
        for (k, v) in fx.options().iter().chain(extra) {
            user.insert(k.clone(), OptionValue::from_user(k, v.clone()));
        }
        fx.format()
            .options()
            .parse(&user, &self.session)
            .map(|(o, _)| o)
            .map_err(|e| e.to_string())
    }

    /// A reservation of the query pool.
    #[must_use]
    pub fn reservation(&self, name: &str) -> Reservation {
        let id = self.query.register(name, None);
        Reservation::new(Arc::clone(&self.query), id)
    }

    /// Writes `batches` as one file `name` with the format's encoder.
    ///
    /// # Errors
    /// Encode or upload errors.
    pub fn write(
        &self,
        fx: &dyn FormatFixture,
        name: &str,
        schema: &SchemaRef,
        batches: &[RecordBatch],
        extra: &[(String, String)],
    ) -> SuiteResult<FileRef> {
        let o = self.options(fx, extra)?;
        let format = fx.format();
        let path = Path::from(name);
        self.block_on(async {
            let mut enc = format
                .writer(&self.io, Arc::clone(schema), &o)
                .await
                .map_err(|e| e.to_string())?;
            let mut upload = self
                .io
                .upload(&self.id, path.clone())
                .map_err(|e| e.to_string())?;
            for b in batches {
                enc.write(b).map_err(|e| e.to_string())?;
                while let Some(bytes) = enc.take_bytes() {
                    upload.write(bytes).await.map_err(|e| e.to_string())?;
                }
            }
            let done = enc.finish().map_err(|e| e.to_string())?;
            upload.write(done.tail).await.map_err(|e| e.to_string())?;
            upload.finish().await.map_err(|e| e.to_string())?;
            self.file(&path).await
        })
    }

    /// The listed file at `path`.
    ///
    /// # Errors
    /// If it does not exist.
    pub async fn file(&self, path: &Path) -> SuiteResult<FileRef> {
        let meta = object_store::ObjectStoreExt::head(self.inner.as_ref(), path)
            .await
            .map_err(|e| e.to_string())?;
        Ok(FileRef::from_meta(self.id.clone(), &meta))
    }

    /// Replaces the bytes at `path` (outside the scheduler: test setup).
    ///
    /// # Errors
    /// Store errors.
    pub fn put_raw(&self, path: &Path, bytes: Vec<u8>) -> SuiteResult<FileRef> {
        self.block_on(async {
            object_store::ObjectStoreExt::put(self.inner.as_ref(), path, bytes.into())
                .await
                .map_err(|e| e.to_string())?;
            self.file(path).await
        })
    }

    /// The bytes at `path` (test setup).
    ///
    /// # Errors
    /// Store errors.
    pub fn get_raw(&self, path: &Path) -> SuiteResult<Bytes> {
        self.block_on(async {
            object_store::ObjectStoreExt::get(self.inner.as_ref(), path)
                .await
                .map_err(|e| e.to_string())?
                .bytes()
                .await
                .map_err(|e| e.to_string())
        })
    }

    /// Scans `split` of `file` through the fixture's driver; the error is
    /// the engine error.
    ///
    /// # Errors
    /// Any scan error.
    pub fn scan(
        &self,
        fx: &dyn FormatFixture,
        file: &FileRef,
        req: &ScanRequest,
        split: Option<ByteRange>,
        scan: ScanOptions,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        let o = self
            .options(fx, &[])
            .map_err(|e| KrishivError::from(krishiv_common::error::InternalError { message: e }))?;
        let split = split.unwrap_or(ByteRange::whole(file.size));
        self.block_on(fx.driver().scan(
            self,
            fx.format(),
            file.clone(),
            req.clone(),
            split,
            o,
            scan,
        ))
    }
}

fn concat(schema: &SchemaRef, batches: &[RecordBatch]) -> SuiteResult<RecordBatch> {
    arrow_select::concat::concat_batches(schema, batches).map_err(|e| e.to_string())
}

/// `batch` with every struct's nulls pushed into its children (through
/// lists and maps too): a value under a null struct is not observable, and
/// formats such as Parquet cannot store one. Batches equal after this are
/// equal for every reader.
fn logical(batch: &RecordBatch) -> SuiteResult<RecordBatch> {
    let columns = batch
        .columns()
        .iter()
        .map(push_nulls)
        .collect::<SuiteResult<Vec<_>>>()?;
    RecordBatch::try_new_with_options(
        batch.schema(),
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| e.to_string())
}

fn push_nulls(a: &arrow_array::ArrayRef) -> SuiteResult<arrow_array::ArrayRef> {
    use arrow_array::cast::AsArray;
    let bad = |e: arrow_schema::ArrowError| e.to_string();
    Ok(match a.data_type() {
        DataType::Struct(fields) => {
            let s = a.as_struct();
            let children = s
                .columns()
                .iter()
                .map(|c| {
                    let c = push_nulls(c)?;
                    let nulls = arrow_buffer::NullBuffer::union(s.nulls(), c.nulls());
                    let data = c
                        .to_data()
                        .into_builder()
                        .nulls(nulls)
                        .build()
                        .map_err(bad)?;
                    Ok(arrow_array::make_array(data))
                })
                .collect::<SuiteResult<Vec<_>>>()?;
            Arc::new(
                arrow_array::StructArray::try_new_with_length(
                    fields.clone(),
                    children,
                    s.nulls().cloned(),
                    s.len(),
                )
                .map_err(bad)?,
            )
        }
        DataType::List(f) => {
            let l = a.as_list::<i32>();
            Arc::new(
                arrow_array::ListArray::try_new(
                    Arc::clone(f),
                    l.offsets().clone(),
                    push_nulls(l.values())?,
                    l.nulls().cloned(),
                )
                .map_err(bad)?,
            )
        }
        DataType::Map(f, sorted) => {
            let m = a.as_map();
            let entries: arrow_array::ArrayRef = Arc::new(m.entries().clone());
            let entries = push_nulls(&entries)?;
            Arc::new(
                arrow_array::MapArray::try_new(
                    Arc::clone(f),
                    m.offsets().clone(),
                    entries.as_struct().clone(),
                    m.nulls().cloned(),
                    *sorted,
                )
                .map_err(bad)?,
            )
        }
        _ => Arc::clone(a),
    })
}

/// Whether two batches hold the same rows: same schema, and every column
/// equal, as a whole or row by row (Arrow's whole-array comparison of
/// nested view arrays can differ for equal rows, by buffer layout).
fn same_rows(a: &RecordBatch, b: &RecordBatch) -> bool {
    a.schema() == b.schema()
        && a.num_rows() == b.num_rows()
        && a.columns().iter().zip(b.columns()).all(|(x, y)| {
            x.to_data() == y.to_data()
                || (0..x.len()).all(|r| x.slice(r, 1).to_data() == y.slice(r, 1).to_data())
        })
}

fn rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

fn class(e: &KrishivError) -> &'static str {
    e.error_class().name()
}

fn s(k: &str, v: &dyn std::fmt::Display) -> (String, String) {
    (k.to_owned(), v.to_string())
}

/// `schema` with every field nullable (what a mapped scan returns).
fn nullable(schema: &SchemaRef) -> SchemaRef {
    let req = ScanRequest::all(Arc::clone(schema));
    krishiv_io::SchemaMapper::new(&req, Arc::clone(schema), "x")
        .map(|m| Arc::clone(m.output_schema()))
        .unwrap_or_else(|_| Arc::clone(schema))
}

/// Check 1.
///
/// # Errors
/// The first difference found.
pub fn round_trip(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let full = data::all_types_schema();
    let fields: Vec<Field> = full
        .fields()
        .iter()
        .filter(|f| fx.supports(f.data_type()))
        .map(|f| f.as_ref().clone())
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batches = vec![
        data::batch(&schema, 1700, 1)?,
        data::batch(&schema, 0, 2)?,
        data::batch(&schema, 900, 3)?,
    ];
    let file = env.write(fx, "rt/all", &schema, &batches, &fx.unit_rows(500))?;
    let got = env
        .scan(
            fx,
            &file,
            &ScanRequest::all(Arc::clone(&schema)),
            None,
            ScanOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    let out_schema = nullable(&schema);
    let want = concat(
        &out_schema,
        &batches
            .iter()
            .map(|b| b.clone().with_schema(Arc::clone(&out_schema)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
    )?;
    let got = concat(&out_schema, &got)?;
    ensure!(
        same_rows(&logical(&got)?, &logical(&want)?),
        "round trip differs"
    );
    let md = env
        .block_on(env.io.read_metadata(
            fx.format().as_ref(),
            &file,
            MetadataNeeds::default(),
            &env.options(fx, &[])?,
        ))
        .map_err(|e| e.to_string())?;
    ensure!(
        md.schema().fields().len() == schema.fields().len(),
        "metadata schema has other fields"
    );
    let variant = md
        .schema()
        .field_with_name("variant")
        .map(|f| f.metadata().clone())
        .ok();
    ensure!(
        variant
            .is_none_or(|m| m.get("ARROW:extension:name").map(String::as_str)
                == Some("arrow.parquet.variant")),
        "the Variant extension tag was lost"
    );
    // Zero columns: row counts only.
    let count = env
        .scan(
            fx,
            &file,
            &ScanRequest::new(Arc::clone(&schema), Vec::new()),
            None,
            ScanOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    ensure!(
        rows(&count) == 2600,
        "count scan returned {} rows",
        rows(&count)
    );
    ensure!(
        count.iter().all(|b| b.num_columns() == 0),
        "count scan returned columns"
    );
    // An empty file.
    let empty = env.write(fx, "rt/empty", &schema, &[], &[])?;
    let got = env
        .scan(
            fx,
            &empty,
            &ScanRequest::all(Arc::clone(&schema)),
            None,
            ScanOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    ensure!(rows(&got) == 0, "empty file returned rows");
    ensure!(env.root.reserved() == 0, "memory held after scans");
    Ok(())
}

/// Check 2.
///
/// # Errors
/// The first difference found.
pub fn projection(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = data::all_types_schema();
    let schema = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .filter(|f| fx.supports(f.data_type()))
            .map(|f| f.as_ref().clone())
            .collect::<Vec<_>>(),
    ));
    let input = data::batch(&schema, 1200, 9)?;
    let file = env.write(
        fx,
        "proj/f",
        &schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(400),
    )?;
    let mut r = Rng::new(42);
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    for case in 0..12 {
        let mut paths: Vec<ColumnPath> = Vec::new();
        for _ in 0..=r.index(5) {
            let n = &names[r.index(names.len())];
            let p = ColumnPath::new(n.clone());
            if !paths.contains(&p) {
                paths.push(p);
            }
        }
        if case % 3 == 0 && schema.field_with_name("st").is_ok() {
            paths.push(ColumnPath::nested(["st", "b", "c"]));
        }
        let req = ScanRequest::new(Arc::clone(&schema), paths.clone());
        let got = env
            .scan(fx, &file, &req, None, ScanOptions::default())
            .map_err(|e| format!("case {case}: {e}"))?;
        let mapper = krishiv_io::SchemaMapper::new(&req, Arc::clone(&schema), "x")
            .map_err(|e| e.to_string())?;
        let file_req = mapper.file_request(&req);
        let want_file = project_batch(&input, &file_req)?;
        let want = mapper.map_batch(&want_file).map_err(|e| e.to_string())?;
        let got = concat(mapper.output_schema(), &got)?;
        ensure!(
            same_rows(&logical(&got)?, &logical(&want)?),
            "projection {paths:?} differs"
        );
    }
    Ok(())
}

/// `batch` projected as the file-level `req` asks (the expected decoder output).
fn project_batch(batch: &RecordBatch, req: &ScanRequest) -> SuiteResult<RecordBatch> {
    let out = req.output_schema().map_err(|e| e.to_string())?;
    let mut cols = Vec::new();
    for f in out.fields() {
        let c = batch
            .column_by_name(f.name())
            .ok_or_else(|| format!("no column {}", f.name()))?;
        cols.push(prune_array(c, f.data_type())?);
    }
    RecordBatch::try_new(out, cols).map_err(|e| e.to_string())
}

fn prune_array(a: &arrow_array::ArrayRef, want: &DataType) -> SuiteResult<arrow_array::ArrayRef> {
    use arrow_array::cast::AsArray;
    match (a.data_type(), want) {
        (DataType::Struct(_), DataType::Struct(wf)) if a.data_type() != want => {
            let sa = a.as_struct();
            let cols = wf
                .iter()
                .map(|f| {
                    sa.column_by_name(f.name())
                        .ok_or_else(|| format!("no field {}", f.name()))
                        .and_then(|c| prune_array(c, f.data_type()))
                })
                .collect::<SuiteResult<Vec<_>>>()?;
            Ok(Arc::new(
                arrow_array::StructArray::try_new(wf.clone(), cols, sa.nulls().cloned())
                    .map_err(|e| e.to_string())?,
            ))
        }
        _ => Ok(Arc::clone(a)),
    }
}

fn pruning_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("l", DataType::Int64, true),
        Field::new("d", DataType::Float64, true),
        Field::new("s", DataType::Utf8View, true),
        Field::new("dt", DataType::Date32, true),
    ]))
}

fn sorted_batch(schema: &SchemaRef, rows: usize, seed: u64) -> SuiteResult<RecordBatch> {
    // Sorted on `i` so units have narrow ranges and pruning has work to do.
    let b = data::batch(schema, rows, seed)?;
    let idx = arrow_ord_sort(b.column(0))?;
    let cols = b
        .columns()
        .iter()
        .map(|c| arrow_select::take::take(c.as_ref(), &idx, None).map_err(|e| e.to_string()))
        .collect::<SuiteResult<Vec<_>>>()?;
    RecordBatch::try_new(Arc::clone(schema), cols).map_err(|e| e.to_string())
}

fn arrow_ord_sort(a: &arrow_array::ArrayRef) -> SuiteResult<arrow_array::UInt32Array> {
    use arrow_array::cast::AsArray;
    let v = a.as_primitive::<arrow_array::types::Int32Type>();
    let mut idx: Vec<u32> = (0..u32::try_from(v.len()).map_err(|e| e.to_string())?).collect();
    idx.sort_by_key(|&i| {
        let i = i as usize;
        if v.is_null(i) {
            (1, 0)
        } else {
            (0, v.value(i))
        }
    });
    Ok(arrow_array::UInt32Array::from(idx))
}

/// Check 3.
///
/// # Errors
/// A pushed scan that dropped a matching row.
pub fn pruning_soundness(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = sorted_batch(&schema, 3000, 5)?;
    let mut variants = vec![("plain".to_owned(), fx.unit_rows(100))];
    for (name, mut opts) in fx.statistics_edge_cases() {
        opts.extend(fx.unit_rows(100));
        variants.push((name, opts));
    }
    // Table types wider than the file's: literals are re-typed to the file.
    let widened = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, true),
        Field::new("l", DataType::Int64, true),
        Field::new("d", DataType::Float64, true),
        Field::new("s", DataType::Utf8View, true),
        Field::new(
            "dt",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            true,
        ),
    ]));
    let wide_input = krishiv_io::SchemaMapper::new(
        &ScanRequest::all(Arc::clone(&widened)),
        Arc::clone(&schema),
        "x",
    )
    .and_then(|m| m.map_batch(&input))
    .map_err(|e| e.to_string())?;
    let mut r = Rng::new(7);
    let mut returned_rows = 0;
    for (vi, (name, opts)) in variants.iter().enumerate() {
        let file = env.write(
            fx,
            &format!("prune/{vi}"),
            &schema,
            std::slice::from_ref(&input),
            opts,
        )?;
        for case in 0..150 {
            let (table, oracle) = if case % 3 == 2 {
                (&widened, &wide_input)
            } else {
                (&schema, &input)
            };
            let expr = data::predicate(oracle, &["i", "l", "d", "s", "dt"], &mut r, 2);
            let mut req = ScanRequest::all(Arc::clone(table));
            req.predicate = Some(expr.clone());
            req.metadata_cols.row_index = true;
            let got = env
                .scan(fx, &file, &req, None, ScanOptions::default())
                .map_err(|e| e.to_string())?;
            let mut returned = std::collections::HashSet::new();
            if vi == 0 {
                returned_rows += rows(&got);
            }
            for b in &got {
                let idx = b.column_by_name(ROW_INDEX_COLUMN).ok_or("no row index")?;
                let idx = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
                    idx.as_ref(),
                );
                returned.extend(idx.values().iter().copied());
            }
            for row in 0..oracle.num_rows() {
                if data::eval(&expr, oracle, row) == Some(true) {
                    let row_i = i64::try_from(row).map_err(|e| e.to_string())?;
                    ensure!(
                        returned.contains(&row_i),
                        "{name}: pushed {expr:?} dropped matching row {row}"
                    );
                }
            }
        }
    }
    // Not vacuous: statistics pruned some units.
    ensure!(
        returned_rows < 150 * input.num_rows(),
        "no predicate pruned anything"
    );
    nan_behind_float_bounds(&env, fx)
}

/// NaN rows a float unit's bounds do not show (the Parquet specification
/// keeps NaN out of min/max): `{1.0, NaN}` has bounds `1.0..1.0`, yet in
/// Spark NaN is greater than, and unequal to, every other value.
fn nan_behind_float_bounds(env: &Env, fx: &dyn FormatFixture) -> SuiteResult {
    use krishiv_io::{CmpOp, PushdownExpr, ScalarValue};
    let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Float64, true)]));
    let input = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Float64Array::from(vec![
            1.0,
            f64::NAN,
            10.0,
            11.0,
        ]))],
    )
    .map_err(|e| e.to_string())?;
    let file = env.write(
        fx,
        "prune/nan",
        &schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(2),
    )?;
    for (op, v) in [(CmpOp::Gt, 50.0), (CmpOp::GtEq, 50.0), (CmpOp::NotEq, 1.0)] {
        let mut req = ScanRequest::all(Arc::clone(&schema));
        req.predicate = Some(PushdownExpr::cmp(
            ColumnPath::new("d"),
            op,
            ScalarValue::Float64(v),
        ));
        let got = env
            .scan(fx, &file, &req, None, ScanOptions::default())
            .map_err(|e| e.to_string())?;
        let nans: usize = got
            .iter()
            .map(|b| {
                arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Float64Type>(
                    b.column(0).as_ref(),
                )
                .iter()
                .flatten()
                .filter(|x| x.is_nan())
                .count()
            })
            .sum();
        ensure!(nans == 1, "pushed d {op:?} {v} dropped the NaN row");
    }
    Ok(())
}

/// Check 4.
///
/// # Errors
/// Wrong rows or row indexes.
pub fn selection_and_row_index(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = data::batch(&schema, 1000, 11)?;
    let file = env.write(
        fx,
        "sel/f",
        &schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(128),
    )?;
    let o = env.options(fx, &[])?;
    if !fx.format().capabilities(&o).row_index {
        return Ok(());
    }
    let deleted: Vec<u64> = (0..1000)
        .filter(|i| i % 5 == 0 || (300..420).contains(i))
        .collect();
    let mut req = ScanRequest::all(Arc::clone(&schema));
    req.metadata_cols.row_index = true;
    if fx.format().capabilities(&o).selection {
        req.selection = Some(RowSelection::excluding(1000, deleted.iter().copied()));
    }
    let got = env
        .scan(fx, &file, &req, None, ScanOptions::default())
        .map_err(|e| e.to_string())?;
    let mut seen = Vec::new();
    for b in &got {
        let idx = b.column_by_name(ROW_INDEX_COLUMN).ok_or("no row index")?;
        let idx =
            arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(idx.as_ref());
        for (k, &ri) in idx.values().iter().enumerate() {
            let row = usize::try_from(ri).map_err(|e| e.to_string())?;
            // The row's values are those of row `ri` of the input.
            let want = input.slice(row, 1);
            let have = b.slice(k, 1);
            for c in 0..schema.fields().len() {
                ensure!(
                    have.column(c).to_data() == want.column(c).to_data(),
                    "row_index {ri} points at other values"
                );
            }
            seen.push(ri);
        }
    }
    let want: Vec<i64> = if req.selection.is_some() {
        (0..1000_u64)
            .filter(|i| !deleted.contains(i))
            .map(|i| i64::try_from(i).unwrap_or(0))
            .collect()
    } else {
        (0..1000).collect()
    };
    ensure!(
        seen == want,
        "selected rows differ ({} vs {})",
        seen.len(),
        want.len()
    );
    Ok(())
}

/// Check 5.
///
/// # Errors
/// A row read twice or never.
pub fn split_invariance(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = data::batch(&schema, 4000, 13)?;
    let file = env.write(
        fx,
        "split/f",
        &schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(97),
    )?;
    let mut r = Rng::new(99);
    let mut req = ScanRequest::all(Arc::clone(&schema));
    req.metadata_cols.row_index = true;
    for _ in 0..20 {
        let mut cuts: Vec<u64> = (0..r.index(8)).map(|_| r.below(file.size)).collect();
        cuts.push(0);
        cuts.push(file.size);
        cuts.sort_unstable();
        cuts.dedup();
        let mut seen = Vec::new();
        for w in cuts.windows(2) {
            let got = env
                .scan(
                    fx,
                    &file,
                    &req,
                    Some(ByteRange::new(w[0], w[1])),
                    ScanOptions::default(),
                )
                .map_err(|e| e.to_string())?;
            for b in &got {
                let idx = b.column_by_name(ROW_INDEX_COLUMN).ok_or("no row index")?;
                let idx = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
                    idx.as_ref(),
                );
                seen.extend(idx.values().iter().copied());
            }
        }
        seen.sort_unstable();
        ensure!(
            seen == (0..4000).collect::<Vec<i64>>(),
            "splits {cuts:?} did not read each row once"
        );
    }
    Ok(())
}

/// Check 6.
///
/// # Errors
/// A wrong mapping or error class.
pub fn schema_evolution(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    if !fx
        .format()
        .capabilities(&env.options(fx, &[])?)
        .self_describing
    {
        // Files without a schema are read with the table's: nothing to evolve.
        return Ok(());
    }
    let st = DataType::Struct(vec![Field::new("x", DataType::Int32, true)].into());
    let file_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true).with_metadata(id_meta("1")),
        Field::new("Name", DataType::Utf8View, true).with_metadata(id_meta("2")),
        Field::new("st", st, true),
    ]));
    let input = data::batch(&file_schema, 300, 21)?;
    let file = env.write(
        fx,
        "evo/f",
        &file_schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(100),
    )?;
    let read = |table: Schema, mapping: FieldMapping| -> Result<RecordBatch, KrishivError> {
        let table = Arc::new(table);
        let mut req = ScanRequest::all(Arc::clone(&table));
        req.mapping = mapping;
        let got = env.scan(fx, &file, &req, None, ScanOptions::default())?;
        let out = nullable(&table);
        concat(&out, &got)
            .map_err(|e| KrishivError::from(krishiv_common::error::InternalError { message: e }))
    };
    let insensitive = FieldMapping::ByName {
        case_sensitive: false,
    };
    // Missing top-level and nested; widened int -> bigint.
    let st2 = DataType::Struct(
        vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Utf8View, true),
        ]
        .into(),
    );
    let b = read(
        Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("z", DataType::Utf8View, true),
            Field::new("st", st2, true),
        ]),
        insensitive.clone(),
    )
    .map_err(|e| e.to_string())?;
    ensure!(
        b.column(1).null_count() == 300,
        "missing column is not null"
    );
    let sa = arrow_array::cast::AsArray::as_struct(b.column(2).as_ref());
    ensure!(
        sa.column(1).null_count() == 300,
        "missing nested field is not null"
    );
    let a = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
        b.column(0).as_ref(),
    );
    let a0 = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int32Type>(
        input.column(0).as_ref(),
    );
    ensure!(
        (0..300).all(|i| a.is_null(i) == a0.is_null(i)
            && (a.is_null(i) || a.value(i) == i64::from(a0.value(i)))),
        "widened values differ"
    );
    // Mismatch.
    let e = read(
        Schema::new(vec![Field::new("Name", DataType::Int64, true)]),
        insensitive.clone(),
    )
    .err()
    .ok_or("string read as bigint did not fail")?;
    ensure!(
        e.error_class() == ErrorClass::FailedReadFileParquetColumnDataTypeMismatch,
        "mismatch class {}",
        class(&e)
    );
    // Case rules.
    let b = read(
        Schema::new(vec![Field::new("name", DataType::Utf8View, true)]),
        insensitive,
    )
    .map_err(|e| e.to_string())?;
    ensure!(
        b.column(0).null_count() < 300,
        "case-insensitive match read nulls"
    );
    let b = read(
        Schema::new(vec![Field::new("name", DataType::Utf8View, true)]),
        FieldMapping::ByName {
            case_sensitive: true,
        },
    )
    .map_err(|e| e.to_string())?;
    ensure!(
        b.column(0).null_count() == 300,
        "case-sensitive mode matched another case"
    );
    // Field ids: renamed columns.
    let b = read(
        Schema::new(vec![
            Field::new("renamed", DataType::Utf8View, true).with_metadata(id_meta("2")),
        ]),
        FieldMapping::FieldId,
    )
    .map_err(|e| e.to_string())?;
    ensure!(
        b.column(0).to_data() == input.column(1).to_data(),
        "field-id mapping read other data"
    );
    // Ambiguous.
    let dup = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("A", DataType::Int32, true),
    ]));
    let dfile = env.write(fx, "evo/dup", &dup, &[data::batch(&dup, 10, 1)?], &[])?;
    let req = ScanRequest::all(Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Int32,
        true,
    )])));
    let e = env
        .scan(fx, &dfile, &req, None, ScanOptions::default())
        .err()
        .ok_or("ambiguous match did not fail")?;
    // Spark 4.2: FAILED_READ_FILE.NO_HINT caused by _LEGACY_ERROR_TEMP_2093.
    ensure!(
        e.error_class() == ErrorClass::FailedReadFileNoHint
            && e.cause().map(KrishivError::error_class) == Some(ErrorClass::LegacyErrorTemp2093),
        "ambiguity class {} (cause {:?})",
        class(&e),
        e.cause().map(KrishivError::error_class)
    );
    Ok(())
}

fn id_meta(id: &str) -> std::collections::HashMap<String, String> {
    std::collections::HashMap::from([(FIELD_ID_KEY.to_owned(), id.to_owned())])
}

/// Check 7.
///
/// # Errors
/// A panic, an untyped error, or wrong ignore semantics.
pub fn corrupt_inputs(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = data::batch(&schema, 800, 31)?;
    let file = env.write(
        fx,
        "bad/good",
        &schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(200),
    )?;
    let good = env.get_raw(&file.path)?.to_vec();
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
    let n = good.len();
    for cut in [0, 1, 7, n / 3, n / 2, n - 9, n - 5, n - 1] {
        cases.push((format!("truncated at {cut}"), good[..cut].to_vec()));
    }
    let mut r = Rng::new(3);
    for k in 0..24 {
        let mut b = good.clone();
        let at = if k % 2 == 0 {
            n - 1 - r.index(n.min(200))
        } else {
            r.index(n)
        };
        b[at] ^= 1 << r.index(8);
        cases.push((format!("bit flip at {at}"), b));
    }
    cases.push((
        "garbage".into(),
        (0..n)
            .map(|i| u8::try_from(i * 31 % 251).unwrap_or(0))
            .collect(),
    ));
    let req = ScanRequest::all(Arc::clone(&schema));
    let typed = [
        ErrorClass::FailedReadFileNoHint,
        ErrorClass::FailedReadFileCannotReadFileFooter,
    ];
    for (i, (name, bytes)) in cases.into_iter().enumerate() {
        let path = Path::from(format!("bad/{i}"));
        let f = env.put_raw(&path, bytes)?;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            env.scan(fx, &f, &req, None, ScanOptions::default())
        }));
        let res = res.map_err(|_| format!("{name}: panicked"))?;
        if let Err(e) = &res {
            ensure!(
                typed.contains(&e.error_class()),
                "{name}: untyped error {}",
                class(e)
            );
        }
        let ignore = ScanOptions {
            ignore_corrupt_files: true,
            ..ScanOptions::default()
        };
        let got = env
            .scan(fx, &f, &req, None, ignore)
            .map_err(|e| format!("{name}: not ignored: {e}"))?;
        ensure!(rows(&got) <= 800, "{name}: more rows than written");
    }
    // Missing file.
    let gone = FileRef {
        path: Path::from("bad/missing"),
        ..file
    };
    let e = env
        .scan(fx, &gone, &req, None, ScanOptions::default())
        .err()
        .ok_or("missing file read")?;
    ensure!(
        e.error_class() == ErrorClass::FailedReadFileFileNotExist,
        "missing class {}",
        class(&e)
    );
    let ignore = ScanOptions {
        ignore_missing_files: true,
        ..ScanOptions::default()
    };
    ensure!(
        env.scan(fx, &gone, &req, None, ignore)
            .map_err(|e| e.to_string())?
            .is_empty(),
        "missing file not ignored"
    );
    // A file replaced after listing.
    let path = Path::from("bad/changing");
    let listed = env.put_raw(&path, good.clone())?;
    let mut changed = good.clone();
    changed[4] ^= 0xFF;
    env.put_raw(&path, changed)?;
    let e = env
        .scan(fx, &listed, &req, None, ScanOptions::default())
        .err()
        .ok_or("changed file read")?;
    ensure!(
        e.error_class() == ErrorClass::KrishivFileChangedDuringRead,
        "changed class {}",
        class(&e)
    );
    // Changed between the footer read and the data reads.
    let listed = env.put_raw(&path, good.clone())?;
    let o = env.options(fx, &[])?;
    let format = fx.format();
    let res: Result<(), KrishivError> = env.block_on(async {
        let mut s = ScanStream::open(
            &env.io,
            format.as_ref(),
            listed.clone(),
            ByteRange::whole(listed.size),
            &req,
            &o,
            env.reservation("s"),
            ScanOptions::default(),
        )
        .await?;
        object_store::ObjectStoreExt::put(
            env.inner.as_ref(),
            &path,
            good.iter().map(|b| b ^ 1).collect::<Vec<u8>>().into(),
        )
        .await
        .map_err(|e| krishiv_common::error::InternalError {
            message: e.to_string(),
        })?;
        while s.next_batch().await?.is_some() {}
        Ok(())
    });
    let e = res.err().ok_or("mid-read change not detected")?;
    ensure!(
        e.error_class() == ErrorClass::KrishivFileChangedDuringRead,
        "mid-read class {}",
        class(&e)
    );
    ensure!(env.root.reserved() == 0, "memory held after failed scans");
    Ok(())
}

/// Check 8.
///
/// # Errors
/// Memory held at the end, or an error other than a memory error.
pub fn memory_accounting(fx: &dyn FormatFixture) -> SuiteResult {
    let schema = pruning_schema();
    let input = data::batch(&schema, 3000, 41)?;
    for limit in [None, Some(16 << 10), Some(1 << 20)] {
        let env = Env::new(limit)?;
        // The data is written before the limit matters (setup pool only).
        let file = env.write(
            fx,
            "mem/f",
            &schema,
            std::slice::from_ref(&input),
            &fx.unit_rows(300),
        )?;
        let req = ScanRequest::all(Arc::clone(&schema));
        match env.scan(
            fx,
            &file,
            &req,
            None,
            ScanOptions {
                prefetch_bytes: 1 << 20,
                ..ScanOptions::default()
            },
        ) {
            Ok(b) => ensure!(rows(&b) == 3000, "rows lost under a memory limit"),
            Err(e) => ensure!(
                matches!(
                    e.error_class(),
                    ErrorClass::UnableToAcquireMemory | ErrorClass::KrishivQueryAbortedOutOfMemory
                ),
                "limit {limit:?}: not a memory error: {e}"
            ),
        }
        ensure!(
            env.root.reserved() == 0,
            "limit {limit:?}: {} bytes held at the end",
            env.root.reserved()
        );
    }
    Ok(())
}

/// Polls `f` `n` times (time advancing between polls) and drops it.
fn poll_n<F: Future>(env: &Env, f: F, n: usize) -> bool {
    env.block_on(async move {
        let mut f = Box::pin(f);
        let mut cx = Context::from_waker(Waker::noop());
        for _ in 0..n {
            if f.as_mut().poll(&mut cx).is_ready() {
                return true;
            }
            tokio::time::advance(Duration::from_millis(3)).await;
        }
        // Dropped here, inside the runtime (timers need it).
        drop(f);
        false
    })
}

/// Check 9.
///
/// # Errors
/// Memory held or a multipart upload left open after a cancel.
pub fn cancellation(fx: &dyn FormatFixture) -> SuiteResult {
    let schema = pruning_schema();
    let input = data::batch(&schema, 2000, 51)?;
    for n in 0..40 {
        let env = Env::new(None)?;
        let file = env.write(
            fx,
            "cancel/f",
            &schema,
            std::slice::from_ref(&input),
            &fx.unit_rows(200),
        )?;
        env.store.set_latency(Duration::from_millis(5));
        let o = env.options(fx, &[])?;
        let req = ScanRequest::all(Arc::clone(&schema));
        let driver = fx.driver();
        let scan = driver.scan(
            &env,
            fx.format(),
            file.clone(),
            req,
            ByteRange::whole(file.size),
            o,
            ScanOptions::default(),
        );
        let finished = poll_n(&env, scan, n);
        // Let tasks the scan left behind (executor drivers) end.
        env.block_on(async { tokio::time::sleep(Duration::from_secs(1)).await });
        ensure!(
            env.root.reserved() == 0,
            "scan dropped after {n} polls holds memory"
        );
        if finished {
            break;
        }
    }
    // Writes: a multipart upload cancelled at every await is aborted.
    let big_schema = Arc::new(Schema::new(vec![Field::new(
        "b",
        DataType::BinaryView,
        true,
    )]));
    let blob: Vec<Option<Vec<u8>>> = (0..24)
        .map(|i| Some(vec![u8::try_from(i).unwrap_or(0); 256 << 10]))
        .collect();
    let big = RecordBatch::try_new(
        Arc::clone(&big_schema),
        vec![Arc::new(arrow_array::BinaryViewArray::from(
            blob.iter().map(|v| v.as_deref()).collect::<Vec<_>>(),
        ))],
    )
    .map_err(|e| e.to_string())?;
    let mut aborted = 0;
    for n in [1, 4, 8, 16] {
        // The smallest part size, so a 6 MiB file takes a multipart upload
        // whatever the format's overhead.
        let env = Env::with_config(
            None,
            IoConfig {
                multipart_part_bytes: krishiv_io::io::upload::MIN_PART_BYTES,
                ..IoConfig::default()
            },
        )?;
        env.store.set_latency(Duration::from_millis(5));
        let o = env.options(fx, &fx.unit_rows(8))?;
        let driver = fx.driver();
        let write = driver.write(
            &env,
            fx.format(),
            Arc::clone(&big_schema),
            o,
            WriteOptions {
                store: StoreId::new("memory://suite"),
                dir: Path::from("cancelw"),
                task: 0,
                job_uuid: "j".into(),
                max_records_per_file: 0,
                target_file_bytes: 0,
            },
            vec![big.clone()],
            false,
        );
        poll_n(&env, write, n);
        // Let background aborts run.
        env.block_on(async { tokio::time::sleep(Duration::from_secs(1)).await });
        let st = env.store.stats();
        ensure!(
            st.uploads_started == st.uploads_completed + st.uploads_aborted,
            "dangling multipart upload after cancelling at poll {n}: {st:?}"
        );
        ensure!(
            env.root.reserved() == 0,
            "write dropped at poll {n} holds memory"
        );
        aborted += st.uploads_aborted;
    }
    ensure!(aborted > 0, "no cancellation hit an open multipart upload");
    Ok(())
}

/// Check 10.
///
/// # Errors
/// A request outside the scheduler, a footer read twice, or counters that
/// do not match the store.
pub fn io_behaviour(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = data::batch(&schema, 1500, 61)?;
    let file = env.write(
        fx,
        "io/f",
        &schema,
        std::slice::from_ref(&input),
        &fx.unit_rows(500),
    )?;
    let before = env.store.stats();
    env.store
        .fail_next_gets(2, Some(Duration::from_millis(200)));
    let req = ScanRequest::all(Arc::clone(&schema));
    let got = env
        .scan(fx, &file, &req, None, ScanOptions::default())
        .map_err(|e| e.to_string())?;
    ensure!(rows(&got) == 1500, "rows lost with retried requests");
    let c = env.io.counters();
    ensure!(
        IoCounters::get(&c.retries) >= 2,
        "transient failures were not retried"
    );
    let after = env.store.stats();
    ensure!(
        after.tail_gets - before.tail_gets == 1,
        "footer took {} tail reads",
        after.tail_gets - before.tail_gets
    );
    let gets = after.gets - before.gets;
    let reads = IoCounters::get(&c.requests) - (before.puts + before.uploads_started);
    ensure!(
        gets <= reads,
        "store saw {gets} GETs, the scheduler counted {reads} requests"
    );
    ensure!(
        after.unscheduled == 0,
        "{} requests bypassed the scheduler",
        after.unscheduled
    );
    // The second scan finds the footer in the cache.
    env.scan(fx, &file, &req, None, ScanOptions::default())
        .map_err(|e| e.to_string())?;
    ensure!(
        env.store.stats().tail_gets == after.tail_gets,
        "cached footer was read again"
    );
    ensure!(IoCounters::get(&c.cache_hits) >= 1, "no cache hit");
    // Bytes counted are bytes served.
    ensure!(
        IoCounters::get(&c.bytes_read) == env.store.stats().bytes - before.bytes,
        "byte counters differ"
    );
    Ok(())
}

/// Check 11.
///
/// # Errors
/// Wrong option handling or a leaked secret.
pub fn options(fx: &dyn FormatFixture) -> SuiteResult {
    let mut env = Env::new(None)?;
    let format = fx.format();
    let schema = format.options();
    let Some(first) = schema.options.iter().find(|o| o.kind != OptionKind::Secret) else {
        return Ok(());
    };
    // A misspelling: accepted with a did-you-mean warning, listed as ignored.
    let typo = format!("{}x", first.name);
    let user = BTreeMap::from([(typo.clone(), OptionValue::from_user(&typo, "1"))]);
    let (o, warnings) = schema
        .parse(&user, &Config::new())
        .map_err(|e| e.to_string())?;
    ensure!(
        warnings
            == vec![OptionWarning::Unknown {
                key: typo.clone(),
                suggestion: Some(first.name)
            }],
        "warning {warnings:?}"
    );
    ensure!(
        o.ignored() == [typo.clone()],
        "ignored keys {:?}",
        o.ignored()
    );
    // Framework options are not reported; keys are case-insensitive.
    let user = BTreeMap::from([
        (
            "pathGlobFilter".to_owned(),
            OptionValue::from_user("pathGlobFilter", "*.x"),
        ),
        (
            first.name.to_uppercase(),
            OptionValue::from_user(first.name, valid_text(first)),
        ),
    ]);
    let (_, warnings) = schema
        .parse(&user, &Config::new())
        .map_err(|e| e.to_string())?;
    ensure!(warnings.is_empty(), "known options warned: {warnings:?}");
    // Strict mode.
    let (strict, _) =
        Config::from_pairs([("krishiv.io.strictOptions", "true")]).map_err(|e| e.to_string())?;
    let user = BTreeMap::from([(typo.clone(), OptionValue::from_user(&typo, "1"))]);
    let e = schema
        .parse(&user, &strict)
        .err()
        .ok_or("strict mode accepted an unknown option")?;
    ensure!(
        e.error_class() == ErrorClass::KrishivUnknownFormatOption,
        "strict class {}",
        class(&e)
    );
    ensure!(e.to_string().contains(first.name), "no suggestion in {e}");
    // Defaults are typed.
    let d = FormatOptions::defaults(schema);
    for spec in schema.options {
        let has =
            d.bool(spec.name).is_some() || d.int(spec.name).is_some() || d.str(spec.name).is_some();
        ensure!(
            has == (spec.default != krishiv_io::OptionDefault::None),
            "default of {}",
            spec.name
        );
    }
    // A wrongly typed value.
    if let Some(b) = schema
        .options
        .iter()
        .find(|o| matches!(o.kind, OptionKind::Bool | OptionKind::Int { .. }))
    {
        let user = BTreeMap::from([(
            b.name.to_owned(),
            OptionValue::from_user(b.name, "not-a-value"),
        )]);
        let e = schema
            .parse(&user, &Config::new())
            .err()
            .ok_or("bad value accepted")?;
        ensure!(
            e.error_class() == ErrorClass::KrishivInvalidFormatOptionValue,
            "value class {}",
            class(&e)
        );
    }
    // A session-conf fallback.
    if let Some(spec) = schema
        .options
        .iter()
        .find(|o| o.conf_fallback.is_some() && o.kind == OptionKind::Bool)
    {
        let key = spec.conf_fallback.unwrap_or_default();
        let flipped = !matches!(spec.default, krishiv_io::OptionDefault::Bool(true));
        env.session_mut()
            .set(key, &flipped.to_string())
            .map_err(|e| e.to_string())?;
        let o = env.options(fx, &[])?;
        ensure!(
            o.bool(spec.name) == Some(flipped),
            "conf fallback {key} ignored"
        );
    }
    // Secrets: never printed or in cache keys; another key never hits.
    if let Some(key_opt) = fx.metadata_key_option() {
        let canary = "canary-7f3a9b";
        let opts = vec![s(key_opt, &canary)];
        let o = env.options(fx, &opts)?;
        ensure!(
            !format!("{o:?}").contains(canary),
            "secret in Debug of options"
        );
        let pschema = pruning_schema();
        let file = env.write(
            fx,
            "opt/secret",
            &pschema,
            &[data::batch(&pschema, 50, 1)?],
            &opts,
        )?;
        let fp = env
            .block_on(o.metadata_fingerprint(env.io.secrets()))
            .map_err(|e| e.to_string())?;
        let key = CacheKey::new(format.name(), &file, fp);
        ensure!(
            !format!("{key:?}").contains(canary),
            "secret in a cache key"
        );
        env.block_on(
            env.io
                .read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o),
        )
        .map_err(|e| e.to_string())?;
        let wrong = env.options(fx, &[s(key_opt, &"another-key")])?;
        let e = env
            .block_on(env.io.read_metadata(
                format.as_ref(),
                &file,
                MetadataNeeds::default(),
                &wrong,
            ))
            .err()
            .ok_or("metadata decoded with the wrong key (cache hit across keys?)")?;
        ensure!(!e.to_string().contains(canary), "secret in an error");
        ensure!(
            e.error_class() == ErrorClass::FailedReadFileCannotReadFileFooter,
            "wrong-key class {}",
            class(&e)
        );
    }
    Ok(())
}

/// A value `spec` accepts.
fn valid_text(spec: &krishiv_io::OptionSpec) -> String {
    match spec.kind {
        OptionKind::Bool => "true".to_owned(),
        OptionKind::Int { min, .. } => min.to_string(),
        OptionKind::Enum(values) => values.first().map_or_else(String::new, |v| (*v).to_owned()),
        OptionKind::String | OptionKind::Secret => "x".to_owned(),
    }
}

/// Check 12.
///
/// # Errors
/// Wrong files, statistics or abort behaviour.
pub fn writer(fx: &dyn FormatFixture) -> SuiteResult {
    let env = Env::new(None)?;
    let schema = pruning_schema();
    let input = data::batch(&schema, 2500, 71)?;
    let format = fx.format();
    let driver = fx.driver();
    let write = |dir: &str, max_records: u64, target: u64, batches: Vec<RecordBatch>, fail| {
        let o = env.options(fx, &fx.unit_rows(300))?;
        env.block_on(driver.write(
            &env,
            Arc::clone(&format),
            Arc::clone(&schema),
            o,
            WriteOptions {
                store: StoreId::new("memory://suite"),
                dir: Path::from(dir),
                task: 7,
                job_uuid: "8c1e".into(),
                max_records_per_file: max_records,
                target_file_bytes: target,
            },
            batches,
            fail,
        ))
        .map_err(|e| e.to_string())
    };
    // maxRecordsPerFile.
    let files = write(
        "w1",
        1000,
        0,
        vec![input.slice(0, 1700), input.slice(1700, 800)],
        false,
    )?;
    let names: Vec<String> = files
        .iter()
        .map(|f| f.path.filename().unwrap_or_default().to_owned())
        .collect();
    let suffix = format.file_suffix(&env.options(fx, &[])?);
    let want: Vec<String> = (0..3)
        .map(|i| krishiv_io::spark_file_name(7, "8c1e", i, &suffix))
        .collect();
    ensure!(names == want, "file names {names:?}");
    ensure!(
        names[0] == format!("part-00007-8c1e-c000{suffix}"),
        "Spark naming"
    );
    ensure!(
        files.iter().map(|f| f.rows).collect::<Vec<_>>() == [1000, 1000, 500],
        "rows per file"
    );
    let mut back = Vec::new();
    let o = env.options(fx, &[])?;
    for f in &files {
        let file = env.block_on(env.file(&f.path))?;
        ensure!(file.size == f.size, "reported size differs");
        let md = env
            .block_on(
                env.io
                    .read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o),
            )
            .map_err(|e| e.to_string())?;
        ensure!(
            md.statistics() == f.statistics,
            "writer statistics differ from reader statistics"
        );
        back.extend(
            env.scan(
                fx,
                &file,
                &ScanRequest::all(Arc::clone(&schema)),
                None,
                ScanOptions::default(),
            )
            .map_err(|e| e.to_string())?,
        );
    }
    let out = nullable(&schema);
    ensure!(
        concat(&out, &back)? == input.clone().with_schema(out).map_err(|e| e.to_string())?,
        "written data differs"
    );
    // Target-size rollover.
    let files = write("w2", 0, 1, vec![input.clone()], false)?;
    ensure!(files.len() > 1, "no rollover at the target size");
    ensure!(
        files.iter().map(|f| f.rows).sum::<u64>() == 2500,
        "rows lost on rollover"
    );
    // A failed write leaves nothing, not even the files it rolled over.
    ensure!(
        write("w3", 1000, 0, vec![input], true).is_err(),
        "the failing write succeeded"
    );
    let left = env.block_on(env.inner.list(Some(&Path::from("w3"))).collect::<Vec<_>>());
    ensure!(left.is_empty(), "abort left {} objects", left.len());
    ensure!(env.root.reserved() == 0, "writer memory held");
    Ok(())
}

/// Instantiates the conformance suite for a fixture:
/// `file_format_conformance!(module_name, fixture_expression);`
#[macro_export]
macro_rules! file_format_conformance {
    ($name:ident, $fixture:expr) => {
        mod $name {
            #[allow(
                unused_imports,
                reason = "the fixture expression may name items of the caller"
            )]
            use super::*;
            use $crate::file_format as suite;
            fn run(check: fn(&dyn suite::FormatFixture) -> suite::SuiteResult) {
                let fx = $fixture;
                if let Err(e) = check(&fx) {
                    panic!("{e}");
                }
            }
            #[test]
            fn round_trip() {
                run(suite::round_trip)
            }
            #[test]
            fn projection() {
                run(suite::projection)
            }
            #[test]
            fn pruning_soundness() {
                run(suite::pruning_soundness)
            }
            #[test]
            fn selection_and_row_index() {
                run(suite::selection_and_row_index)
            }
            #[test]
            fn split_invariance() {
                run(suite::split_invariance)
            }
            #[test]
            fn schema_evolution() {
                run(suite::schema_evolution)
            }
            #[test]
            fn corrupt_inputs() {
                run(suite::corrupt_inputs)
            }
            #[test]
            fn memory_accounting() {
                run(suite::memory_accounting)
            }
            #[test]
            fn cancellation() {
                run(suite::cancellation)
            }
            #[test]
            fn io_behaviour() {
                run(suite::io_behaviour)
            }
            #[test]
            fn options() {
                run(suite::options)
            }
            #[test]
            fn writer() {
                run(suite::writer)
            }
        }
    };
}

/// The mock format's fixture.
#[derive(Debug, Clone, Copy, Default)]
pub struct MockFixture {
    /// Decode through the async read-at bridge.
    pub async_decoder: bool,
}

impl FormatFixture for MockFixture {
    fn format(&self) -> Arc<dyn FileFormat> {
        Arc::new(mock::MockFormat)
    }
    fn options(&self) -> Vec<(String, String)> {
        if self.async_decoder {
            vec![s("asyncDecoder", &"true")]
        } else {
            Vec::new()
        }
    }
    fn unit_rows(&self, rows: usize) -> Vec<(String, String)> {
        vec![s("batchRows", &rows)]
    }
    fn statistics_edge_cases(&self) -> Vec<(String, Vec<(String, String)>)> {
        vec![
            (
                "truncated string bounds".into(),
                vec![s("statsTruncateLength", &1)],
            ),
            (
                "legacy-writer statistics".into(),
                vec![s("legacyStats", &"true")],
            ),
        ]
    }
    fn metadata_key_option(&self) -> Option<&'static str> {
        Some("encryptionKey")
    }
}
