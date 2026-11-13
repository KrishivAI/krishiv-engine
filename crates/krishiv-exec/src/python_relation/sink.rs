// SPDX-License-Identifier: Apache-2.0
//! The sink of the Python relation nodes (`PythonMap`, `PythonGroupMap`,
//! `PythonCoGroupMap`, `PythonAggregate`; design P3-T085 §3.1-§3.2).
//!
//! **Shape.** As `PythonEval`'s (P2-T059): a pipeline break whose sink sends
//! the input to the runtime and returns at once, and whose results are kept
//! in a shared plan's list (`crate::shared`: memory under the driver's
//! consumer, spilled under pressure) that a `SharedSource` reads after
//! `finalize`. Compute workers never run Python and never wait inside
//! `sink`: each driver opens one [`UdfTask`] per lane (one UDF; an
//! aggregate has a lane per call) at its first frame, queues frames, and
//! while a lane's bounded window of frames in flight is full returns
//! [`SinkStatus::Blocked`], whose future moves frames and results until
//! the window has room. Unlike `PythonEval`, output rows are not aligned to
//! input rows: a task's batches are stored as it yields them.
//!
//! **Groups.** The planner sorts a grouped input by its keys after a hash
//! exchange on them (one partition without keys), so each driver sees a
//! key's rows together; the sink cuts groups where the key changes
//! (compared as the sort compares them, floats canonical) and ends each
//! with [`Frame::EndGroup`]. A cogroup's input is both relations in one
//! stream sorted by key then side: a key's left rows, then its right rows.
//! Rows are held until a frame is full (`maxRecordsPerBatch`, and
//! `maxBytesPerBatch` when set) or its group or side ends, as Spark cuts a
//! group's Arrow stream; a group is never held whole.
//!
//! **Memory.** Held rows and queued frames are reserved in the driver's
//! consumer (a refused batch returns `NeedMemory` before anything changes,
//! so the retry is exact); a frame's reservation is released once the task
//! took it. Results wait in the lane's queue until the next `sink` call
//! stores them; an aggregate keeps one key row per group until `finalize`.
//!
//! **Errors and cancellation.** The first failure of a task fails the
//! query. Dropping the state (query end, cancel, error) drops every task,
//! which cancels it.

use std::collections::VecDeque;
use std::fmt;
use std::future::poll_fn;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_row::OwnedRow;
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use futures::task::noop_waker_ref;
use krishiv_common::error::KrishivError;
use krishiv_memory::{Reservation, SpillScope};
use krishiv_types::batch::Batch;

use super::{Frame, PreparedRelationUdf, UdfTask};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::operators::{KeyEncoder, ScalarExpr, SortKey, global_ref, local_mut};
use crate::shared::SharedSink;
use crate::sync::{Mutex, lock};

/// How a relation node's input reaches its tasks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationShape {
    /// Every row of the partition, in batches, then [`Frame::End`]
    /// (`mapInPandas`, `mapInArrow`).
    Map,
    /// Groups of equal `keys` (input columns), each ended by
    /// [`Frame::EndGroup`]; no keys is one group.
    Grouped {
        /// The key columns.
        keys: Vec<usize>,
    },
    /// Two relations in one stream sorted by `keys` then the `side` column
    /// (0 left, 1 right): per key, the left rows then the right rows.
    CoGrouped {
        /// The key columns.
        keys: Vec<usize>,
        /// The column (`INT`) naming each row's relation.
        side: usize,
    },
    /// As [`Self::Grouped`]; each lane's task yields one value per group,
    /// and the output is the group's keys then one column per lane.
    Aggregate {
        /// The key columns.
        keys: Vec<usize>,
    },
}

impl RelationShape {
    fn keys(&self) -> Option<&[usize]> {
        match self {
            Self::Map => None,
            Self::Grouped { keys } | Self::CoGrouped { keys, .. } | Self::Aggregate { keys } => {
                Some(keys)
            }
        }
    }
}

/// One UDF of a relation node: the prepared call site and the columns of its
/// frames, per side (expressions over the input; a column read for a map or
/// grouped kind, an argument for an aggregate).
pub struct RelationLane {
    /// The prepared call site.
    pub udf: Arc<dyn PreparedRelationUdf>,
    /// Each side's frame columns, evaluated over the input.
    pub sides: Vec<Vec<Arc<dyn ScalarExpr>>>,
    /// Each side's frame column names.
    pub names: Vec<Vec<String>>,
    /// The Arrow types the task's batches must have (client forms), one per
    /// result column.
    pub client: Vec<ArrowType>,
}

impl fmt::Debug for RelationLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelationLane")
            .field("sides", &self.sides)
            .field("names", &self.names)
            .field("client", &self.client)
            .finish_non_exhaustive()
    }
}

/// Frames in flight per lane when nothing else is configured.
pub const DEFAULT_WINDOW: usize = 4;

/// The sink side of a Python relation node.
pub struct RelationSink {
    shape: RelationShape,
    lanes: Vec<RelationLane>,
    encoder: Option<Arc<KeyEncoder>>,
    /// The output (canonical carriers).
    schema: SchemaRef,
    /// An aggregate's key columns, as their rows are kept.
    key_schema: Option<SchemaRef>,
    input: SchemaRef,
    max_rows: usize,
    max_bytes: usize,
    window: usize,
    store: Arc<SharedSink>,
}

impl fmt::Debug for RelationSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelationSink")
            .field("shape", &self.shape)
            .field("lanes", &self.lanes)
            .field("max_rows", &self.max_rows)
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

/// Batching and buffering limits of a [`RelationSink`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationLimits {
    /// Most rows per frame; 0 is no limit.
    pub max_rows: usize,
    /// Most bytes per frame (cut between rows); 0 is no limit.
    pub max_bytes: usize,
    /// Frames in flight per lane before the driver blocks.
    pub window: usize,
}

impl RelationSink {
    /// A sink over input of `input` running `lanes` as `shape` says; its
    /// rows have `schema` (canonical carriers). Finished rows spill to
    /// `scope`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for lanes that do not fit the shape;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a key type that cannot be compared.
    pub fn try_new(
        input: SchemaRef,
        shape: RelationShape,
        lanes: Vec<RelationLane>,
        schema: SchemaRef,
        limits: RelationLimits,
        scope: Option<SpillScope>,
    ) -> Result<Arc<Self>> {
        let sides = match shape {
            RelationShape::CoGrouped { .. } => 2,
            _ => 1,
        };
        if lanes.is_empty()
            || lanes
                .iter()
                .any(|l| l.sides.len() != sides || l.names.len() != sides)
        {
            return Err(internal(
                "Python relation lanes do not fit their shape".to_owned(),
            ));
        }
        let encoder = match shape.keys() {
            Some(keys) => {
                let keys: Vec<SortKey> = keys.iter().map(|&k| SortKey::asc(k)).collect();
                Some(KeyEncoder::new(&input, &keys)?)
            }
            None => None,
        };
        let key_schema = match &shape {
            RelationShape::Aggregate { keys } => {
                let fields: Vec<Field> = keys
                    .iter()
                    .map(|&k| {
                        input
                            .fields()
                            .get(k)
                            .map(|f| (**f).clone())
                            .ok_or_else(|| internal(format!("key column {k} out of range")))
                    })
                    .collect::<Result<_>>()?;
                Some(Arc::new(Schema::new(fields)))
            }
            _ => None,
        };
        Ok(Arc::new(Self {
            shape,
            lanes,
            encoder,
            store: SharedSink::new(Arc::clone(&schema), scope),
            schema,
            key_schema,
            input,
            max_rows: limits.max_rows,
            max_bytes: limits.max_bytes,
            window: limits.window.max(1),
        }))
    }

    /// The output schema.
    #[must_use]
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// A lane's queues, shared by the driver and the lane's pump.
#[derive(Default)]
struct LaneIo {
    /// Frames not taken by the task yet, each with its reservation.
    inbound: VecDeque<(Frame, Reservation)>,
    /// The reservation of the frame the task is taking.
    sending: Option<Reservation>,
    /// Batches the task yielded, not stored yet.
    outputs: VecDeque<RecordBatch>,
}

impl LaneIo {
    fn in_flight(&self) -> usize {
        self.inbound.len() + usize::from(self.sending.is_some())
    }
}

/// The future that moves a lane's frames to its task and its results back.
enum Pump {
    /// No frame yet: the task is not open.
    Idle,
    Running(BoxFuture<'static, Result<()>>),
    Done(Result<()>),
}

/// One lane of one driver.
#[derive(Clone)]
struct Lane {
    io: Arc<Mutex<LaneIo>>,
    pump: Arc<Mutex<Pump>>,
}

impl Lane {
    fn new() -> Self {
        Self {
            io: Arc::new(Mutex::new(LaneIo::default())),
            pump: Arc::new(Mutex::new(Pump::Idle)),
        }
    }

    /// Polls the pump; its outcome once it ended (`None` while it runs or
    /// before the task opened).
    fn poll(pump: &Mutex<Pump>, cx: &mut Context<'_>) -> Option<Result<()>> {
        let mut p = lock(pump);
        if let Pump::Running(f) = &mut *p
            && let Poll::Ready(r) = f.as_mut().poll(cx)
        {
            *p = Pump::Done(r);
        }
        match &*p {
            Pump::Done(r) => Some(r.clone()),
            Pump::Idle | Pump::Running(_) => None,
        }
    }

    fn opened(&self) -> bool {
        !matches!(*lock(&self.pump), Pump::Idle)
    }

    /// Queues `frame`, opening the task first if it is not.
    fn queue(&self, udf: &dyn PreparedRelationUdf, frame: Frame, r: Reservation) -> Result<()> {
        {
            let mut p = lock(&self.pump);
            if matches!(*p, Pump::Idle) {
                let task = udf.open()?;
                *p = Pump::Running(pump(task, Arc::clone(&self.io)));
            }
        }
        lock(&self.io).inbound.push_back((frame, r));
        Ok(())
    }
}

/// Sends the queued frames to `task` while it reads the task's results.
fn pump(task: Box<dyn UdfTask>, io: Arc<Mutex<LaneIo>>) -> BoxFuture<'static, Result<()>> {
    Box::pin(async move {
        let task = task;
        let send = async {
            loop {
                // Never woken: whoever queues a frame polls the pump next.
                let frame = poll_fn(|_| {
                    let mut q = lock(&io);
                    match q.inbound.pop_front() {
                        Some((frame, r)) => {
                            q.sending = Some(r);
                            Poll::Ready(frame)
                        }
                        None => Poll::Pending,
                    }
                })
                .await;
                let end = matches!(frame, Frame::End);
                task.send(frame).await?;
                lock(&io).sending = None;
                if end {
                    return Ok::<(), KrishivError>(());
                }
            }
        };
        let receive = async {
            while let Some(batch) = task.next().await? {
                lock(&io).outputs.push_back(batch);
            }
            Ok(())
        };
        futures::future::try_join(send, receive).await.map(drop)
    })
}

/// One driver's state.
struct RelLocal {
    store: LocalState,
    lanes: Vec<Lane>,
    /// A finished batch the store could not take yet (it asked for memory).
    ready: Option<Batch>,
    partition: u32,
    /// The key of the open group's last row seen.
    current: Option<OwnedRow>,
    /// A group is open (grouped shapes).
    group_open: bool,
    /// Rows of the open group (and side) not sent yet.
    held: Vec<RecordBatch>,
    held_rows: usize,
    held_side: u8,
    /// The reservation of the held rows' bytes.
    held_bytes: Reservation,
    /// An aggregate's group keys, one row per group, in group order.
    key_rows: Vec<RecordBatch>,
    key_bytes: Vec<Reservation>,
}

/// What a driver left at `combine`.
struct Leftover {
    lanes: Vec<Lane>,
    ready: Option<Batch>,
    key_rows: Vec<RecordBatch>,
    _key_bytes: Vec<Reservation>,
}

/// The global state: the store's, and what drivers left.
struct RelGlobal {
    store: GlobalState,
    leftover: Mutex<Vec<Leftover>>,
}

/// A run of input rows within one group and side.
struct Run {
    start: usize,
    len: usize,
    /// It starts a group (its first row's key differs from the open one).
    new_group: bool,
    side: u8,
}

impl RelationSink {
    /// The runs of `batch` and the key of its last row: rows of one group
    /// and side, in order.
    fn runs(
        &self,
        batch: &RecordBatch,
        current: Option<&OwnedRow>,
    ) -> Result<(Vec<Run>, Option<OwnedRow>)> {
        let n = batch.num_rows();
        let sides: Option<&arrow_array::Int32Array> = match &self.shape {
            RelationShape::CoGrouped { side, .. } => Some(
                batch
                    .column(*side)
                    .as_any()
                    .downcast_ref()
                    .ok_or_else(|| internal("a cogroup side column is not INT".to_owned()))?,
            ),
            _ => None,
        };
        let side_of = |i: usize| -> u8 { sides.map_or(0, |s| u8::from(s.value(i) != 0)) };
        let Some(encoder) = &self.encoder else {
            let run = Run {
                start: 0,
                len: n,
                new_group: false,
                side: 0,
            };
            return Ok((vec![run], None));
        };
        let rows = encoder.encode(batch)?;
        let mut runs: Vec<Run> = Vec::new();
        for i in 0..n {
            let new_group = if i == 0 {
                current.is_none_or(|c| c.row() != rows.row(0))
            } else {
                rows.row(i) != rows.row(i - 1)
            };
            let side = side_of(i);
            match runs.last_mut() {
                Some(r) if !new_group && r.side == side => r.len += 1,
                _ => runs.push(Run {
                    start: i,
                    len: 1,
                    new_group,
                    side,
                }),
            }
        }
        let last = n.checked_sub(1).map(|i| rows.row(i).owned());
        Ok((runs, last.or_else(|| current.cloned())))
    }

    /// The frame of `lane` over `rows` (one side): its columns evaluated,
    /// in their client forms.
    fn frame(
        &self,
        lane: &RelationLane,
        side: u8,
        rows: &RecordBatch,
        partition: u32,
    ) -> Result<Frame> {
        let exprs = &lane.sides[usize::from(side)];
        let names = &lane.names[usize::from(side)];
        let mut fields = Vec::with_capacity(exprs.len());
        let mut columns = Vec::with_capacity(exprs.len());
        for (e, name) in exprs.iter().zip(names) {
            let a = e.evaluate_in(rows, partition)?;
            let client = krishiv_types::arrow::client_arrow_type(a.data_type());
            let a = crate::carrier::recarried_array(&a, &client)?;
            fields.push(Field::new(name, client, e.nullable()));
            columns.push(a);
        }
        let options = RecordBatchOptions::new().with_row_count(Some(rows.num_rows()));
        let batch =
            RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)
                .map_err(|e| internal(format!("Python UDF frame: {e}")))?;
        Ok(Frame::Batch { side, batch })
    }

    /// Sends the held rows in frames of at most `maxRecordsPerBatch` rows
    /// (and `maxBytesPerBatch` bytes); with `all` the rows left over too
    /// (the end of a group or side).
    fn send_held(&self, l: &mut RelLocal, all: bool) -> Result<()> {
        if l.held_rows == 0 {
            return Ok(());
        }
        let held = std::mem::take(&mut l.held);
        let rows = std::mem::take(&mut l.held_rows);
        let data = if held.len() == 1 {
            held.into_iter()
                .next()
                .ok_or_else(|| internal("no held rows".to_owned()))?
        } else {
            arrow_select::concat::concat_batches(&self.input, &held)
                .map_err(|e| internal(format!("held Python UDF rows: {e}")))?
        };
        let mut step = if self.max_rows == 0 {
            rows
        } else {
            self.max_rows
        };
        if self.max_bytes > 0 {
            let bytes = data.get_array_memory_size().max(1);
            let fit = (self.max_bytes.saturating_mul(rows) / bytes).max(1);
            step = step.min(fit);
        }
        let mut offset = 0;
        while offset < rows {
            let len = step.min(rows - offset);
            if len < step && !all {
                // Too few for a full frame: held for the next input.
                l.held.push(data.slice(offset, len));
                l.held_rows = len;
                break;
            }
            let chunk = data.slice(offset, len);
            let share = l.held_bytes.size() * len / (rows - offset);
            let mut part = Some(l.held_bytes.split(share));
            for (lane, state) in self.lanes.iter().zip(&l.lanes) {
                let frame = self.frame(lane, l.held_side, &chunk, l.partition)?;
                // The chunk's bytes go with the first lane's frame.
                let r = match part.take() {
                    Some(r) => r,
                    None => l.held_bytes.split(0),
                };
                state.queue(lane.udf.as_ref(), frame, r)?;
            }
            offset += len;
        }
        Ok(())
    }

    /// Ends the open group: its held rows, then `EndGroup` to every lane.
    fn end_group(&self, l: &mut RelLocal) -> Result<()> {
        self.send_held(l, true)?;
        for (lane, state) in self.lanes.iter().zip(&l.lanes) {
            let r = l.held_bytes.split(0);
            state.queue(lane.udf.as_ref(), Frame::EndGroup, r)?;
        }
        Ok(())
    }

    /// Feeds `batch` (reserved in `held_bytes`) to the lanes.
    fn feed(
        &self,
        l: &mut RelLocal,
        batch: &RecordBatch,
        runs: Vec<Run>,
        keys: Vec<RecordBatch>,
    ) -> Result<()> {
        let grouped = self.encoder.is_some();
        let mut keys = keys.into_iter();
        for run in runs {
            if run.new_group {
                if l.group_open {
                    self.end_group(l)?;
                }
                l.group_open = true;
                if let Some(k) = keys.next() {
                    l.key_rows.push(k);
                }
            } else if grouped && !l.group_open {
                return Err(internal("Python UDF rows before their group".to_owned()));
            }
            if run.side != l.held_side {
                self.send_held(l, true)?;
                l.held_side = run.side;
            }
            l.held.push(batch.slice(run.start, run.len));
            l.held_rows += run.len;
            self.send_held(l, false)?;
        }
        Ok(())
    }

    /// The copied key row of each run starting a group (an aggregate's).
    fn key_rows(&self, batch: &RecordBatch, runs: &[Run]) -> Result<Vec<RecordBatch>> {
        let (RelationShape::Aggregate { keys }, Some(schema)) = (&self.shape, &self.key_schema)
        else {
            return Ok(Vec::new());
        };
        let projected = batch
            .project(keys)
            .map_err(|e| internal(format!("aggregate keys: {e}")))?;
        let projected = RecordBatch::try_new_with_options(
            Arc::clone(schema),
            projected.columns().to_vec(),
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )
        .map_err(|e| internal(format!("aggregate keys: {e}")))?;
        runs.iter()
            .filter(|r| r.new_group)
            .map(|r| {
                arrow_select::concat::concat_batches(schema, [&projected.slice(r.start, 1)])
                    .map_err(|e| internal(format!("aggregate keys: {e}")))
            })
            .collect()
    }

    /// A task's batch as an output batch: its columns checked against the
    /// declared client types, then moved to the canonical carriers. The
    /// runtime checks the result against the declared schema first; this is
    /// the engine's own guard.
    fn output(&self, lane: &RelationLane, batch: &RecordBatch) -> Result<Batch> {
        if batch.num_columns() != lane.client.len() {
            return Err(internal(format!(
                "a Python UDF returned {} columns where {} were declared",
                batch.num_columns(),
                lane.client.len()
            )));
        }
        let mut columns = Vec::with_capacity(batch.num_columns());
        for ((c, client), field) in batch
            .columns()
            .iter()
            .zip(&lane.client)
            .zip(self.schema.fields())
        {
            columns.push(recarried(c, client, field.data_type())?);
        }
        let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
        let out = RecordBatch::try_new_with_options(Arc::clone(&self.schema), columns, &options)
            .map_err(|e| internal(format!("a Python UDF result does not fit its schema: {e}")))?;
        Ok(Batch::new(out))
    }

    /// An aggregate driver's rows: each group's keys, then each lane's value
    /// of it.
    fn aggregate_rows(&self, key_rows: &[RecordBatch], lanes: &[Lane]) -> Result<Option<Batch>> {
        let Some(schema) = &self.key_schema else {
            return Ok(None);
        };
        if key_rows.is_empty() {
            return Ok(None);
        }
        let groups = key_rows.iter().map(RecordBatch::num_rows).sum::<usize>();
        let keys = arrow_select::concat::concat_batches(schema, key_rows)
            .map_err(|e| internal(format!("aggregate keys: {e}")))?;
        let mut columns: Vec<ArrayRef> = keys.columns().to_vec();
        let width = columns.len();
        for (i, (lane, state)) in self.lanes.iter().zip(lanes).enumerate() {
            let outputs: Vec<RecordBatch> = lock(&state.io).outputs.drain(..).collect();
            let values: Vec<&ArrayRef> = outputs
                .iter()
                .map(|b| {
                    if b.num_columns() == 1 {
                        Ok(b.column(0))
                    } else {
                        Err(internal(format!(
                            "a Python aggregate UDF returned {} columns",
                            b.num_columns()
                        )))
                    }
                })
                .collect::<Result<_>>()?;
            let refs: Vec<&dyn Array> = values.iter().map(|a| AsRef::as_ref(*a)).collect();
            let client = lane
                .client
                .first()
                .ok_or_else(|| internal("an aggregate lane without a type".to_owned()))?;
            let column = if refs.is_empty() {
                arrow_array::new_empty_array(client)
            } else {
                arrow_select::concat::concat(&refs)
                    .map_err(|e| internal(format!("aggregate results: {e}")))?
            };
            if column.len() != groups {
                return Err(internal(format!(
                    "a Python aggregate UDF returned {} rows for {groups} groups",
                    column.len()
                )));
            }
            let field = self
                .schema
                .fields()
                .get(width + i)
                .ok_or_else(|| internal("an aggregate result out of its schema".to_owned()))?;
            columns.push(recarried(&column, client, field.data_type())?);
        }
        let options = RecordBatchOptions::new().with_row_count(Some(groups));
        let out = RecordBatch::try_new_with_options(Arc::clone(&self.schema), columns, &options)
            .map_err(|e| internal(format!("a Python aggregate result: {e}")))?;
        Ok(Some(Batch::new(out)))
    }

    /// Stores `batch` in the list, or keeps it for the retry after the
    /// memory the store asked for.
    fn store(&self, l: &mut RelLocal, batch: Batch, g: &RelGlobal) -> Result<Option<SinkStatus>> {
        match self.store.sink(batch.clone(), &mut l.store, &g.store)? {
            SinkStatus::NeedMemory { bytes } => {
                l.ready = Some(batch);
                Ok(Some(SinkStatus::NeedMemory { bytes }))
            }
            _ => Ok(None),
        }
    }

    /// Moves each lane's frames and results without waiting; the first
    /// failure of a lane.
    fn progress(l: &RelLocal) -> Result<()> {
        let mut cx = Context::from_waker(noop_waker_ref());
        for lane in &l.lanes {
            if let Some(Err(e)) = Lane::poll(&lane.pump, &mut cx) {
                return Err(e);
            }
        }
        Ok(())
    }
}

/// `c` checked to be `client`, moved to the canonical `canonical`.
fn recarried(c: &ArrayRef, client: &ArrowType, canonical: &ArrowType) -> Result<ArrayRef> {
    if c.data_type() != client {
        return Err(internal(format!(
            "a Python UDF returned {} where {client} was declared",
            c.data_type()
        )));
    }
    crate::carrier::recarried_array(c, canonical)
}

impl Sink for RelationSink {
    fn name(&self) -> &'static str {
        "PythonRelation"
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(RelGlobal {
            store: self.store.global_state(ctx)?,
            leftover: Mutex::new(Vec::new()),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let g = global_ref::<RelGlobal>(global, "PythonRelation")?;
        let consumer = ctx.consumer().consumer();
        Ok(Box::new(RelLocal {
            store: self.store.local_state(ctx, &g.store)?,
            lanes: self.lanes.iter().map(|_| Lane::new()).collect(),
            ready: None,
            partition: ctx.partition(),
            current: None,
            group_open: false,
            held: Vec::new(),
            held_rows: 0,
            held_side: 0,
            held_bytes: consumer.reservation(),
            key_rows: Vec::new(),
            key_bytes: Vec::new(),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<SinkStatus> {
        let g = global_ref::<RelGlobal>(global, "PythonRelation")?;
        let l = local_mut::<RelLocal>(local, "PythonRelation")?;
        // Store what finished first: a `NeedMemory` here leaves the input
        // unsent, so the retry with the same batch is exact.
        if let Some(ready) = l.ready.take()
            && let Some(status) = self.store(l, ready, g)?
        {
            return Ok(status);
        }
        Self::progress(l)?;
        if !matches!(self.shape, RelationShape::Aggregate { .. }) {
            loop {
                let next = l.lanes.iter().zip(&self.lanes).find_map(|(state, lane)| {
                    lock(&state.io).outputs.pop_front().map(|b| (lane, b))
                });
                let Some((lane, batch)) = next else { break };
                let batch = self.output(lane, &batch)?;
                if let Some(status) = self.store(l, batch, g)? {
                    return Ok(status);
                }
            }
        }
        if input.num_rows() > 0 {
            let input = crate::operators::owned_rows(input)?;
            let data = input.data();
            let (runs, last) = self.runs(data, l.current.as_ref())?;
            let keys = self.key_rows(data, &runs)?;
            let key_bytes: usize = keys.iter().map(RecordBatch::get_array_memory_size).sum();
            let bytes = data.get_array_memory_size();
            if l.held_bytes.try_grow(bytes + key_bytes).is_err() {
                return Ok(SinkStatus::NeedMemory {
                    bytes: bytes + key_bytes,
                });
            }
            let part = l.held_bytes.split(key_bytes);
            l.key_bytes.push(part);
            self.feed(l, data, runs, keys)?;
            l.current = last;
            Self::progress(l)?;
        }
        let full = l
            .lanes
            .iter()
            .any(|lane| lock(&lane.io).in_flight() >= self.window);
        if full {
            let lanes: Vec<Lane> = l.lanes.clone();
            let window = self.window;
            return Ok(SinkStatus::Blocked(Box::pin(poll_fn(move |cx| {
                let mut room = true;
                for lane in &lanes {
                    // A lane that ended (or failed) frees the driver: the
                    // next `sink` call returns its failure.
                    if Lane::poll(&lane.pump, cx).is_some() {
                        return Poll::Ready(());
                    }
                    room &= lock(&lane.io).in_flight() < window;
                }
                if room { Poll::Ready(()) } else { Poll::Pending }
            }))));
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<RelGlobal>(global, "PythonRelation")?;
        let l = local
            .downcast::<RelLocal>()
            .map_err(|_| internal("wrong local state for PythonRelation".to_owned()))?;
        let mut l = *l;
        // The input ended: the open group ends, then the partition.
        if l.group_open {
            self.end_group(&mut l)?;
        } else {
            self.send_held(&mut l, true)?;
        }
        for (lane, state) in self.lanes.iter().zip(&l.lanes) {
            if state.opened() {
                let r = l.held_bytes.split(0);
                state.queue(lane.udf.as_ref(), Frame::End, r)?;
            }
        }
        RelationSink::progress(&l)?;
        let RelLocal {
            store,
            lanes,
            ready,
            key_rows,
            key_bytes,
            ..
        } = l;
        lock(&g.leftover).push(Leftover {
            lanes,
            ready,
            key_rows,
            _key_bytes: key_bytes,
        });
        self.store.combine(store, &g.store)
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global_ref::<RelGlobal>(global, "PythonRelation")?;
            let leftover = std::mem::take(&mut *lock(&g.leftover));
            // Every task at once: the first failure (or a cancel) ends the
            // wait, and the error drops the other tasks, which cancels them.
            let mut cancelled = std::pin::pin!(ctx.query().cancelled());
            poll_fn(|cx| {
                if cancelled.as_mut().poll(cx).is_ready() {
                    return Poll::Ready(Err(ctx.query().cancellation()));
                }
                let mut all = true;
                for lane in leftover.iter().flat_map(|d| &d.lanes) {
                    match Lane::poll(&lane.pump, cx) {
                        Some(Err(e)) => return Poll::Ready(Err(e)),
                        Some(Ok(())) => {}
                        None => all &= !lane.opened(),
                    }
                }
                if all {
                    Poll::Ready(Ok(()))
                } else {
                    Poll::Pending
                }
            })
            .await?;
            for driver in leftover {
                ctx.query().check_cancelled()?;
                let mut batches: Vec<Batch> = driver.ready.into_iter().collect();
                if matches!(self.shape, RelationShape::Aggregate { .. }) {
                    batches.extend(self.aggregate_rows(&driver.key_rows, &driver.lanes)?);
                } else {
                    for (lane, state) in self.lanes.iter().zip(&driver.lanes) {
                        let outputs: Vec<RecordBatch> = lock(&state.io).outputs.drain(..).collect();
                        for b in &outputs {
                            batches.push(self.output(lane, b)?);
                        }
                    }
                }
                if batches.is_empty() {
                    continue;
                }
                let bytes = batches
                    .iter()
                    .map(|b| b.data().get_array_memory_size())
                    .sum();
                let mut reservation = ctx.consumer().reservation();
                reservation.grow(bytes).await.map_err(KrishivError::from)?;
                self.store.add_part(&g.store, batches, reservation)?;
            }
            self.store.finalize(&g.store, ctx).await
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = local_mut::<RelLocal>(local, "PythonRelation")?;
            self.store.spill(&mut l.store, target, ctx).await
        })
    }
}

/// The global state a relation node's reader takes its list from: the
/// store's own; `None` for a state of another sink.
pub(crate) fn store_state(global: &GlobalState) -> Option<GlobalState> {
    global
        .downcast_ref::<RelGlobal>()
        .map(|g| Arc::clone(&g.store))
}

/// A column of the input, as a frame reads it.
#[derive(Debug)]
pub struct InputColumn {
    /// The column.
    pub index: usize,
    /// Its type.
    pub data_type: ArrowType,
    /// Whether it may hold nulls.
    pub nullable: bool,
}

impl ScalarExpr for InputColumn {
    fn data_type(&self) -> ArrowType {
        self.data_type.clone()
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        batch
            .columns()
            .get(self.index)
            .cloned()
            .ok_or_else(|| internal(format!("input column {} out of range", self.index)))
    }
}
