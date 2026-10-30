// SPDX-License-Identifier: Apache-2.0
//! How the suite reads and writes files: directly through the format
//! framework's [`ScanStream`] / [`FileWriter`] ([`Direct`]), or end to end
//! through the executor's file scan and write ([`ThroughExecutor`]: a
//! [`FileScan`] source and a [`FileWriteSink`] in a query run by the
//! deterministic executor on the harness runtime).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use futures::future::BoxFuture;
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::{
    ExecConfig, ExecNode, FileScan, FileSelections, FileWrite, FileWriteSink, FnPredicate,
    GlobalState, LocalState, PipelineBuilder, PushedFilter, QueryContext, QueryId, QueryRunner,
    ScanPushdown, ScanPushdowns, Source, TaskContext, TokioExecutor, WorkloadClass,
};
use krishiv_io::split::{FilePartition, FileSplit, SplitConfig};
use krishiv_io::{
    ByteRange, FileFormat, FileRef, FileWriter, FormatOptions, ROW_INDEX_COLUMN, RowSelection,
    ScanOptions, ScanRequest, ScanStream, WriteOptions, WrittenFileInfo,
};
use krishiv_types::batch::Batch;

use super::{Env, FormatFixture};

/// A scan of one split of one file, or a write of batches.
pub trait Driver: Send + Sync + std::fmt::Debug {
    /// Reads `split` of `file` for the table-level `req`; every batch, or
    /// the first error.
    #[expect(clippy::too_many_arguments, reason = "one scan, all inputs distinct")]
    fn scan<'a>(
        &'a self,
        env: &'a Env,
        format: Arc<dyn FileFormat>,
        file: FileRef,
        req: ScanRequest,
        split: ByteRange,
        o: FormatOptions,
        scan: ScanOptions,
    ) -> BoxFuture<'a, Result<Vec<RecordBatch>, KrishivError>>;

    /// Writes `batches` of `schema` as one task with `options`; with
    /// `fail`, the write fails after its last batch (nothing must stay
    /// visible).
    #[expect(clippy::too_many_arguments, reason = "one write, all inputs distinct")]
    fn write<'a>(
        &'a self,
        env: &'a Env,
        format: Arc<dyn FileFormat>,
        schema: SchemaRef,
        o: FormatOptions,
        options: WriteOptions,
        batches: Vec<RecordBatch>,
        fail: bool,
    ) -> BoxFuture<'a, Result<Vec<WrittenFileInfo>, KrishivError>>;
}

fn suite_error(message: impl Into<String>) -> KrishivError {
    InternalError {
        message: message.into(),
    }
    .into()
}

/// The format framework's own read and write loops.
#[derive(Debug, Clone, Copy, Default)]
pub struct Direct;

impl Driver for Direct {
    fn scan<'a>(
        &'a self,
        env: &'a Env,
        format: Arc<dyn FileFormat>,
        file: FileRef,
        req: ScanRequest,
        split: ByteRange,
        o: FormatOptions,
        scan: ScanOptions,
    ) -> BoxFuture<'a, Result<Vec<RecordBatch>, KrishivError>> {
        Box::pin(async move {
            let mut s = ScanStream::open(
                &env.io,
                format.as_ref(),
                file,
                split,
                &req,
                &o,
                env.reservation("scan"),
                scan,
            )
            .await?;
            let mut out = Vec::new();
            while let Some(b) = s.next_batch().await? {
                out.push(b);
            }
            Ok(out)
        })
    }

    fn write<'a>(
        &'a self,
        env: &'a Env,
        format: Arc<dyn FileFormat>,
        schema: SchemaRef,
        o: FormatOptions,
        options: WriteOptions,
        batches: Vec<RecordBatch>,
        fail: bool,
    ) -> BoxFuture<'a, Result<Vec<WrittenFileInfo>, KrishivError>> {
        Box::pin(async move {
            let mut w =
                FileWriter::new(&env.io, format, &schema, o, options, env.reservation("w"))?;
            for b in &batches {
                w.write(b).await?;
            }
            if fail {
                w.abort().await;
                return Err(suite_error("the write was made to fail"));
            }
            w.finish().await
        })
    }
}

/// Reads and writes through the executor (see the module docs).
#[derive(Debug, Clone, Copy, Default)]
pub struct Executor {
    /// Drivers per pipeline.
    pub slots: usize,
}

/// The selection of the request, for its one file.
#[derive(Debug)]
struct Only(Option<RowSelection>);

impl FileSelections for Only {
    fn selection(&self, _file: &FileRef) -> Option<RowSelection> {
        self.0.clone()
    }
}

/// The pushdowns that make the executor's scan read `req`.
fn pushdowns(req: &ScanRequest) -> ScanPushdowns {
    // The file scan prunes with the typed predicate and never evaluates
    // rows, so the row evaluator is a stand-in.
    let filters = req
        .predicate
        .clone()
        .map(|expr| {
            let keep_all = FnPredicate(|b: &RecordBatch| {
                Ok(arrow_array::BooleanArray::from(vec![true; b.num_rows()]))
            });
            PushedFilter::typed(Arc::new(keep_all), expr)
        })
        .into_iter()
        .collect();
    ScanPushdowns {
        projection: Some(req.projection.clone()),
        filters,
        limit: req.limit,
        dynamic: krishiv_exec::RuntimeFilters::new(
            req.dynamic
                .iter()
                .map(|d| std::sync::Arc::clone(d.filter()))
                .collect(),
        ),
        selection: req
            .selection
            .clone()
            .map(|s| Arc::new(Only(Some(s))) as Arc<dyn FileSelections>),
        mapping: req.mapping.clone(),
        metadata: req.metadata_cols,
    }
}

/// `batch` with the `_metadata.row_index` field as the framework's
/// top-level row-index column (what the suite expects).
fn unwrap_row_index(batch: &RecordBatch) -> Result<RecordBatch, KrishivError> {
    let schema = batch.schema();
    let Some(last) = schema.fields().last() else {
        return Ok(batch.clone());
    };
    if last.name() != krishiv_exec::METADATA_COLUMN {
        return Ok(batch.clone());
    }
    let meta = batch.column(batch.num_columns() - 1).as_struct();
    let row_index = meta
        .column_by_name("row_index")
        .ok_or_else(|| suite_error("_metadata without row_index"))?;
    let mut fields: Vec<Field> = schema.fields()[..schema.fields().len() - 1]
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new(ROW_INDEX_COLUMN, DataType::Int64, false));
    let mut columns = batch.columns()[..batch.num_columns() - 1].to_vec();
    columns.push(Arc::clone(row_index));
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| suite_error(e.to_string()))
}

struct Leaf(Arc<dyn Source>);

impl ExecNode for Leaf {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> krishiv_exec::Result<()> {
        b.set_source(Arc::clone(&self.0)).map(drop)
    }
}

/// Batches of one morsel, then an error if `fail`.
#[derive(Debug)]
struct Batches {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    fail: bool,
}

impl Source for Batches {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
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
        Ok(Box::new(AtomicUsize::new(0)))
    }
    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        _global: &'a GlobalState,
    ) -> BoxFuture<'a, krishiv_exec::Result<Option<Batch>>> {
        let next = local
            .downcast_ref::<AtomicUsize>()
            .map_or(usize::MAX, |n| n.fetch_add(1, Ordering::Relaxed));
        let out = match self.batches.get(next) {
            Some(b) => Ok(Some(Batch::new(b.clone()))),
            None if self.fail && next == self.batches.len() => {
                Err(suite_error("the write was made to fail"))
            }
            None => Ok(None),
        };
        Box::pin(std::future::ready(out))
    }
}

impl Executor {
    async fn run(
        &self,
        env: &Env,
        plan: Arc<dyn ExecNode>,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        let executor = TokioExecutor::deterministic(self.slots.max(1))?;
        let runner = QueryRunner::new(executor);
        let query = QueryContext::new(
            QueryId(1),
            WorkloadClass::Interactive,
            ExecConfig::default(),
            Arc::clone(&env.query),
        );
        let mut stream = runner.execute(plan, Arc::clone(&query)).await?;
        let mut out = Vec::new();
        let mut first_error = None;
        while let Some(b) = stream.next().await {
            match b {
                Ok(b) => out.push(b.into_parts().0),
                Err(e) => {
                    first_error.get_or_insert(e);
                }
            }
        }
        query.finished().await;
        first_error.map_or(Ok(out), Err)
    }
}

impl Driver for Executor {
    fn scan<'a>(
        &'a self,
        env: &'a Env,
        format: Arc<dyn FileFormat>,
        file: FileRef,
        req: ScanRequest,
        split: ByteRange,
        o: FormatOptions,
        scan: ScanOptions,
    ) -> BoxFuture<'a, Result<Vec<RecordBatch>, KrishivError>> {
        Box::pin(async move {
            let partition = FilePartition {
                index: 0,
                splits: vec![FileSplit {
                    file: 0,
                    range: split,
                }],
            };
            let source = FileScan::new(
                format,
                env.io.clone(),
                Arc::clone(&req.table_schema),
                vec![file],
                o,
                scan,
                SplitConfig::from_config(&krishiv_common::config::Config::new(), 1),
            )
            .with_partitions(vec![partition])
            .build(pushdowns(&req))?;
            let batches = self.run(env, Arc::new(Leaf(source))).await?;
            batches.iter().map(unwrap_row_index).collect()
        })
    }

    fn write<'a>(
        &'a self,
        env: &'a Env,
        format: Arc<dyn FileFormat>,
        schema: SchemaRef,
        o: FormatOptions,
        options: WriteOptions,
        batches: Vec<RecordBatch>,
        fail: bool,
    ) -> BoxFuture<'a, Result<Vec<WrittenFileInfo>, KrishivError>> {
        Box::pin(async move {
            let sink = Arc::new(FileWriteSink::try_new(
                format,
                env.io.clone(),
                Arc::clone(&schema),
                o,
                options,
            )?);
            let input = Arc::new(Leaf(Arc::new(Batches {
                schema,
                batches,
                fail,
            })));
            let plan = Arc::new(FileWrite {
                input,
                sink: Arc::clone(&sink),
            });
            let out = self.run(env, plan).await?;
            let files = sink.files();
            let listed: usize = out.iter().map(RecordBatch::num_rows).sum();
            if listed != files.len() {
                return Err(suite_error(format!(
                    "the write returned {listed} rows for {} files",
                    files.len()
                )));
            }
            Ok(files)
        })
    }
}

/// A fixture read and written through the executor ([`Executor`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct ThroughExecutor<F>(pub F);

impl<F: FormatFixture> FormatFixture for ThroughExecutor<F> {
    fn format(&self) -> Arc<dyn FileFormat> {
        self.0.format()
    }
    fn options(&self) -> Vec<(String, String)> {
        self.0.options()
    }
    fn unit_rows(&self, rows: usize) -> Vec<(String, String)> {
        self.0.unit_rows(rows)
    }
    fn supports(&self, dt: &DataType) -> bool {
        self.0.supports(dt)
    }
    fn footerless(&self) -> bool {
        self.0.footerless()
    }
    fn supports_field_ids(&self) -> bool {
        self.0.supports_field_ids()
    }
    fn statistics_edge_cases(&self) -> Vec<(String, Vec<(String, String)>)> {
        self.0.statistics_edge_cases()
    }
    fn metadata_key_option(&self) -> Option<&'static str> {
        self.0.metadata_key_option()
    }
    fn driver(&self) -> Arc<dyn Driver> {
        Arc::new(Executor { slots: 2 })
    }
}
