// SPDX-License-Identifier: Apache-2.0
//! `LateFetch`: the deferred columns of a top-N's rows, read from their files
//! once the rows are known (design P2-T063 §3).
//!
//! A breaker. The sink holds the rows (at most the top-N's `k`, charged to a
//! reservation). The source groups them by file, reads each file's rows by
//! position through [`ScanPushdown::take`] and emits the rows in the order
//! they came, with the fetched columns between theirs. A read gives back every
//! row asked for or fails, so the result is never short. It yields between
//! reads, as a scan's reads do between files.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_io::{ColumnPath, FileRef, ROW_INDEX_COLUMN, RowFile, ScanGate, TakeStream};
use krishiv_memory::{ConsumerHandle, Reservation};
use krishiv_types::batch::Batch;

use super::file_scan::FileScanMetrics;
use super::scan::{ScanPushdown, ScanPushdowns, TakeRows};
use super::sort::finalized;
use super::{global_ref, local_mut};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};

/// Where a column of a [`LateFetch`]'s output comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchSlot {
    /// That column of the input.
    Input(usize),
    /// That column of the fetched projection.
    Fetched(usize),
}

/// Reads `projection` of the rows of its input from their files. The input's
/// `file_col` and `row_col` are the row id ([`ScanPushdowns::row_id`]).
pub struct LateFetch {
    schema: SchemaRef,
    scan: Arc<dyn ScanPushdown>,
    projection: Vec<ColumnPath>,
    file_col: usize,
    row_col: usize,
    output: Vec<FetchSlot>,
    metrics: Mutex<FileScanMetrics>,
}

impl std::fmt::Debug for LateFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LateFetch")
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .finish_non_exhaustive()
    }
}

impl LateFetch {
    /// The fetch of `projection` through `scan` over batches of `input`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `file_col` or `row_col` is not a row id column
    /// or a slot is out of range.
    pub fn new(
        input: &SchemaRef,
        scan: Arc<dyn ScanPushdown>,
        projection: Vec<ColumnPath>,
        file_col: usize,
        row_col: usize,
        output: Vec<FetchSlot>,
    ) -> Result<Self> {
        let type_of = |i: usize| input.fields().get(i).map(|f| f.data_type());
        if type_of(file_col) != Some(&DataType::BinaryView)
            || type_of(row_col) != Some(&DataType::Int64)
        {
            return Err(internal(format!(
                "a late fetch whose row id is not columns {file_col} and {row_col} of its input"
            )));
        }
        let fetched = scan.output_schema(&ScanPushdowns {
            projection: Some(projection.clone()),
            ..ScanPushdowns::default()
        })?;
        let fields = output
            .iter()
            .map(|slot| {
                match *slot {
                    FetchSlot::Input(i) => input.fields().get(i),
                    FetchSlot::Fetched(i) => fetched.fields().get(i),
                }
                .cloned()
                .ok_or_else(|| internal(format!("late fetch slot {slot:?} out of range")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            schema: Arc::new(Schema::new_with_metadata(fields, input.metadata().clone())),
            scan,
            projection,
            file_col,
            row_col,
            output,
            metrics: Mutex::new(FileScanMetrics::default()),
        })
    }

    /// The rows of `batches`, by file. Files come in first-seen order.
    fn plan(&self, batches: &[Batch]) -> Result<Wanted> {
        let mut wanted = Wanted::default();
        let mut seen: HashMap<&[u8], usize> = HashMap::new();
        for batch in batches {
            let files = batch.data().column(self.file_col).as_binary_view();
            let rows = batch
                .data()
                .column(self.row_col)
                .as_primitive::<Int64Type>();
            for (file, &row) in files.iter().zip(rows.values()) {
                let file = file.ok_or_else(|| internal("a row without a file".to_owned()))?;
                let row =
                    u64::try_from(row).map_err(|_| internal(format!("row position {row}")))?;
                let index = match seen.get(file) {
                    Some(&index) => index,
                    None => {
                        let index = wanted.files.len();
                        seen.insert(file, index);
                        wanted.files.push((RowFile::decode(file)?, Vec::new()));
                        index
                    }
                };
                wanted.files[index].1.push(row);
                wanted.rows.push((index, row));
            }
        }
        Ok(wanted)
    }

    /// The next batch of `l`'s rows, once everything is read.
    fn emit(&self, l: &mut FetchSource) -> Result<Option<Batch>> {
        if l.emitted >= l.wanted.rows.len() {
            // Done: the rows and their memory go now.
            l.fetched.clear();
            l.input = None;
            l.memory = None;
            l.held.clear();
            return Ok(None);
        }
        let n = l.batch_rows.min(l.wanted.rows.len() - l.emitted);
        let at: Vec<(usize, usize)> = l.wanted.rows[l.emitted..l.emitted + n]
            .iter()
            .map(|key| {
                l.located
                    .get(key)
                    .copied()
                    .ok_or_else(|| internal("a fetched row is missing".to_owned()))
            })
            .collect::<Result<_>>()?;
        let input = l
            .input
            .as_ref()
            .ok_or_else(|| internal("late fetch lost its rows".to_owned()))?
            .slice(l.emitted, n);
        let columns = self
            .output
            .iter()
            .map(|slot| -> Result<ArrayRef> {
                Ok(match *slot {
                    FetchSlot::Input(i) => Arc::clone(input.column(i)),
                    FetchSlot::Fetched(i) => {
                        let arrays: Vec<&dyn Array> =
                            l.fetched.iter().map(|b| b.column(i).as_ref()).collect();
                        arrow_select::interleave::interleave(&arrays, &at)
                            .map_err(|e| internal(format!("late fetch: {e}")))?
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        l.emitted += n;
        let out = RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(n)),
        )
        .map_err(|e| internal(format!("invalid late fetch batch: {e}")))?;
        Ok(Some(Batch::new(out)))
    }

    /// Reads on: one fetched batch, or the next file opened. `true`: all of
    /// it is read.
    async fn read(&self, l: &mut FetchSource) -> Result<bool> {
        let Some((file, stream)) = &mut l.stream else {
            let Some((file, rows)) = l.next_file() else {
                return Ok(true);
            };
            let stream = self
                .scan
                .take(TakeRows {
                    file,
                    rows,
                    projection: self.projection.clone(),
                    mem: l.consumer.reservation(),
                    gate: Arc::clone(&l.gate),
                })
                .await?;
            l.stream = Some((l.next - 1, stream));
            return Ok(false);
        };
        let Some(batch) = stream.next_batch().await? else {
            if let Some(s) = stream.stream() {
                self.metrics
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .add(&s.metrics(), s.overcommitted());
            }
            l.stream = None;
            return Ok(false);
        };
        let file = *file;
        let memory = match &mut l.memory {
            Some(memory) => memory,
            None => l.memory.insert(l.consumer.reservation()),
        };
        memory.grow(batch.get_array_memory_size()).await?;
        let index = l.fetched.len();
        let positions = batch
            .column_by_name(ROW_INDEX_COLUMN)
            .ok_or_else(|| internal("take returned no row index".to_owned()))?
            .as_primitive::<Int64Type>();
        for (row, &position) in positions.values().iter().enumerate() {
            let position =
                u64::try_from(position).map_err(|_| internal(format!("row {position}")))?;
            l.located.insert((file, position), (index, row));
        }
        l.fetched.push(batch);
        Ok(false)
    }
}

/// Which rows to read.
#[derive(Debug, Default)]
struct Wanted {
    /// Each file with its rows' positions.
    files: Vec<(FileRef, Vec<u64>)>,
    /// Each row in order: the index of its file, its position.
    rows: Vec<(usize, u64)>,
}

#[derive(Debug, Default)]
struct Held {
    batches: Vec<Batch>,
    reservations: Vec<Reservation>,
}

#[derive(Debug, Default)]
struct FetchGlobal(Mutex<Held>);

struct FetchSink {
    batches: Vec<Batch>,
    reservation: Reservation,
}

/// The source side.
struct FetchSource {
    started: bool,
    input: Option<RecordBatch>,
    /// Keeps `input` charged until it is emitted.
    held: Vec<Reservation>,
    wanted: Wanted,
    /// The next file of `wanted` to open.
    next: usize,
    stream: Option<(usize, TakeStream)>,
    /// (file, position) to (batch of `fetched`, row).
    located: HashMap<(usize, u64), (usize, usize)>,
    fetched: Vec<RecordBatch>,
    /// Pays for `fetched`.
    memory: Option<Reservation>,
    emitted: usize,
    batch_rows: usize,
    consumer: Arc<ConsumerHandle>,
    gate: Arc<ScanGate>,
}

impl FetchSource {
    fn next_file(&mut self) -> Option<(FileRef, Vec<u64>)> {
        let (file, rows) = self.wanted.files.get_mut(self.next)?;
        self.next += 1;
        Some((file.clone(), std::mem::take(rows)))
    }
}

impl Sink for LateFetch {
    fn name(&self) -> &'static str {
        "LateFetch"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(FetchGlobal::default()))
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(FetchSink {
            batches: Vec::new(),
            reservation: ctx.consumer().reservation(),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        let l = local_mut::<FetchSink>(local, "LateFetch")?;
        let bytes = input.data().get_array_memory_size();
        if l.reservation.try_grow(bytes).is_err() {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        l.batches.push(input);
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<FetchGlobal>(global, "LateFetch")?;
        let l = local
            .downcast::<FetchSink>()
            .map_err(|_| internal("wrong local state for LateFetch".to_owned()))?;
        let mut held = g.0.lock().unwrap_or_else(PoisonError::into_inner);
        held.batches.extend(l.batches);
        held.reservations.push(l.reservation);
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

impl Source for LateFetch {
    fn name(&self) -> &'static str {
        "LateFetch"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        finalized::<FetchGlobal>(ctx, "LateFetch")
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(FetchSource {
            started: false,
            input: None,
            held: Vec::new(),
            wanted: Wanted::default(),
            next: 0,
            stream: None,
            located: HashMap::new(),
            fetched: Vec::new(),
            memory: None,
            emitted: 0,
            batch_rows: ctx.query().config().batch_size.max(1),
            consumer: Arc::clone(ctx.consumer().consumer()),
            gate: Arc::clone(ctx.query().scan_gate()),
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let l = local_mut::<FetchSource>(local, "LateFetch")?;
            if !l.started {
                l.started = true;
                let g = global_ref::<FetchGlobal>(global, "LateFetch")?;
                let held = std::mem::take(&mut *g.0.lock().unwrap_or_else(PoisonError::into_inner));
                l.wanted = self.plan(&held.batches)?;
                l.held = held.reservations;
                let parts: Vec<&RecordBatch> = held.batches.iter().map(Batch::data).collect();
                // The copy rides on the sink's reservations, which are sized
                // for what it copies; both exist only until this block ends,
                // and at most `krishiv.exec.lateFetch.maxRows` rows are held.
                if let Some(first) = parts.first() {
                    let all = arrow_select::concat::concat_batches(&first.schema(), parts.clone())
                        .map_err(|e| internal(format!("late fetch: {e}")))?;
                    l.input = Some(all);
                }
            }
            while !self.read(l).await? {
                tokio::task::yield_now().await;
            }
            self.emit(l)
        })
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        self.metrics
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extra()
    }
}
