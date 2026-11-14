// SPDX-License-Identifier: Apache-2.0
//! The sink of the Python relation nodes (`PythonMap`, `PythonGroupMap`,
//! `PythonCoGroupMap`, `PythonAggregate`; design P3-T085 §3.1-§3.2).
//!
//! **Shape.** As `PythonEval`'s (P2-T059): a pipeline break whose sink sends
//! the input to the runtime and returns at once, and whose results are kept
//! in a shared plan's list (`crate::shared`: memory under the driver's
//! consumer, spilled under pressure) that a `SharedSource` reads after
//! `finalize`. Compute workers never run Python and never wait inside
//! `sink`: each driver opens one [`UdfTask`](super::UdfTask) per lane (one
//! UDF; an aggregate has a lane per call) at its first frame, queues frames,
//! and while a lane's bounded window of frames in flight is full returns
//! [`SinkStatus::Blocked`], whose future moves frames and results until the
//! window has room. Unlike `PythonEval`, output rows are not aligned to
//! input rows: a task's batches are stored as it yields them, by `sink` and
//! then by `finalize` while the tasks finish.
//!
//! **Groups.** The planner sorts a grouped input by its keys after a hash
//! exchange on them (one partition without keys), so each driver sees a
//! key's rows together; the sink cuts groups where the key changes
//! (compared as the sort compares them, floats canonical) and ends each
//! with [`Frame::EndGroup`]. A cogroup's input is both relations in one
//! stream sorted by key then side: a key's left rows, then its right rows;
//! a side without rows for a key gets one zero-row frame.
//! Rows are cut into frames of `maxRecordsPerBatch` rows (and
//! `maxBytesPerBatch` bytes when set) or at the end of their group or side,
//! as Spark cuts a group's Arrow stream; a group is never held whole.
//!
//! **Window.** The window bounds the frames queued per lane and is checked
//! frame by frame: input rows cut but not framed yet wait in the driver's
//! backlog (raw rows, reserved), are framed as the window frees, and the
//! driver is `Blocked` while a lane's window is full.
//!
//! **Memory (I11).** Every byte the sink holds is reserved in the driver's
//! consumer: input rows held or in the backlog, queued frames (the chunk's
//! rows shared by the lanes' frames, released by the last lane's task to
//! take it, and the copies made for the client forms), the results a task
//! yielded until they are stored (`lane.rs`: a refused result stops the
//! task's reads until reserved), and an aggregate's pending group keys
//! (`pair.rs`). A refusal in `sink` returns `NeedMemory` before the input
//! is taken, so the retry is exact. In `finalize` a refused map or grouped
//! result is stored as the others, written to a spill file when the pool
//! is full; an aggregate's refused value waits for its memory while the
//! other lanes move. A stored result's bytes pass to the store's
//! reservation through the consumer's credit, never counted twice.
//!
//! **Errors and cancellation.** The first failure of a task fails the
//! query. Dropping the state (query end, cancel, error) drops every task,
//! which cancels it, and every reservation.

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
use krishiv_common::error::{KrishivError, KrishivPythonUdfResultMismatch};
use krishiv_memory::{ConsumerHandle, Reservation, SpillScope};
use krishiv_types::batch::Batch;

use super::lane::{FrameMem, Lane};
use super::pair::{self, Pending};
use super::{Frame, PreparedRelationUdf};
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
    /// and the output is the values of `key_values` in the group's first
    /// row, then one column per lane.
    Aggregate {
        /// The columns groups are compared by (a collated key's comparison
        /// key).
        keys: Vec<usize>,
        /// The key columns the output shows, as many as `keys`.
        key_values: Vec<usize>,
    },
}

impl RelationShape {
    fn keys(&self) -> Option<&[usize]> {
        match self {
            Self::Map => None,
            Self::Grouped { keys }
            | Self::CoGrouped { keys, .. }
            | Self::Aggregate { keys, .. } => Some(keys),
        }
    }
}

/// One UDF of a relation node: the prepared call site and the columns of its
/// frames, per side (expressions over the input; a column read for a map or
/// grouped kind, an argument for an aggregate).
pub struct RelationLane {
    /// The UDF's name, for errors.
    pub name: String,
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
            .field("name", &self.name)
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
            RelationShape::Aggregate { keys, key_values } => {
                if key_values.len() != keys.len() {
                    return Err(internal(
                        "a Python aggregate's key values do not match its keys".to_owned(),
                    ));
                }
                let fields: Vec<Field> = key_values
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

/// Input rows cut into what a lane's task gets, not framed yet.
enum Cut {
    /// Rows of one side (zero for a cogroup side without rows), with the
    /// reservation of their bytes.
    Rows {
        side: u8,
        rows: RecordBatch,
        memory: Reservation,
    },
    EndGroup,
    End,
}

/// One driver's state (and, after `combine`, what `finalize` finishes).
struct Driver {
    lanes: Vec<Lane>,
    /// A finished batch the store could not take yet (it asked for
    /// memory), with the memory of the results it came from.
    ready: Option<(Batch, Vec<Reservation>)>,
    partition: u32,
    /// The key of the open group's last row seen.
    current: Option<OwnedRow>,
    /// A group is open (grouped shapes).
    group_open: bool,
    /// Rows of the open group (and side) not cut yet.
    held: Vec<RecordBatch>,
    held_rows: usize,
    held_side: u8,
    /// The cogroup sides the open group has rows on.
    sides_seen: [bool; 2],
    /// The reservation of the bytes of the input rows taken and not cut
    /// yet: the held rows and the rest of the batch being fed.
    held_bytes: Reservation,
    /// The rows `held_bytes` covers: each cut takes its share by rows of
    /// all of them, not of the held rows only (a batch's later groups keep
    /// theirs while its first group is cut).
    held_bytes_rows: usize,
    /// Cut, waiting for room in the lanes' windows.
    backlog: VecDeque<Cut>,
    /// Memory reserved for the next frame's copies (`finalize` waits for
    /// it; `sink` asks the driver for it).
    spare: Reservation,
    /// In `finalize`, the bytes the next frame's copies wait for (0: none).
    wants: usize,
    /// An aggregate's groups waiting for their values.
    pending: Pending,
}

/// What [`RelationSink::flush`] stopped at.
#[derive(Clone, Copy)]
enum Flushed {
    /// The backlog is empty.
    All,
    /// A lane's window is full.
    Window,
    /// The next frame's copies need this many more bytes.
    Memory(usize),
}

/// One driver's sink state.
struct RelLocal {
    store: LocalState,
    driver: Driver,
    /// The driver's consumer, whose credit hands results to the store.
    consumer: Arc<ConsumerHandle>,
}

/// The global state: the store's, and what drivers left.
struct RelGlobal {
    store: GlobalState,
    leftover: Mutex<Vec<Driver>>,
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
    /// in their client forms; and the bytes of the columns copied to reach
    /// them.
    fn frame(
        lane: &RelationLane,
        side: u8,
        rows: &RecordBatch,
        partition: u32,
    ) -> Result<(Frame, usize)> {
        let exprs = &lane.sides[usize::from(side)];
        let names = &lane.names[usize::from(side)];
        let mut fields = Vec::with_capacity(exprs.len());
        let mut columns = Vec::with_capacity(exprs.len());
        let mut copied = 0;
        for (e, name) in exprs.iter().zip(names) {
            let a = e.evaluate_in(rows, partition)?;
            let client = krishiv_types::arrow::client_arrow_type(a.data_type());
            let a = if a.data_type() == &client {
                a
            } else {
                let moved = crate::carrier::recarried_array(&a, &client)?;
                copied += moved.get_array_memory_size();
                moved
            };
            fields.push(Field::new(name, client, e.nullable()));
            columns.push(a);
        }
        let options = RecordBatchOptions::new().with_row_count(Some(rows.num_rows()));
        let batch =
            RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)
                .map_err(|e| internal(format!("Python UDF frame: {e}")))?;
        Ok((Frame::Batch { side, batch }, copied))
    }

    /// Cuts the held rows into frames of at most `maxRecordsPerBatch` rows
    /// (and `maxBytesPerBatch` bytes); with `all` the rows left over too
    /// (the end of a group or side).
    fn cut_held(&self, d: &mut Driver, all: bool) -> Result<()> {
        if d.held_rows == 0 {
            return Ok(());
        }
        let held = std::mem::take(&mut d.held);
        let rows = std::mem::take(&mut d.held_rows);
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
                d.held.push(data.slice(offset, len));
                d.held_rows = len;
                break;
            }
            let share = d.held_bytes.size() * len / d.held_bytes_rows.max(len);
            d.held_bytes_rows = d.held_bytes_rows.saturating_sub(len);
            d.backlog.push_back(Cut::Rows {
                side: d.held_side,
                rows: data.slice(offset, len),
                memory: d.held_bytes.split(share),
            });
            offset += len;
        }
        Ok(())
    }

    /// A zero-row frame of `side`: a cogroup key without rows on that side
    /// (Spark gives the UDF an empty DataFrame for it). The side counts as
    /// seen, so a key whose rows span input batches gets it once.
    fn empty_side(&self, d: &mut Driver, side: u8) {
        d.sides_seen[usize::from(side)] = true;
        d.backlog.push_back(Cut::Rows {
            side,
            rows: RecordBatch::new_empty(Arc::clone(&self.input)),
            memory: d.held_bytes.split(0),
        });
    }

    fn cogrouped(&self) -> bool {
        matches!(self.shape, RelationShape::CoGrouped { .. })
    }

    /// Ends the open group: its held rows (a cogroup's empty right side),
    /// then `EndGroup`.
    fn end_group(&self, d: &mut Driver) -> Result<()> {
        self.cut_held(d, true)?;
        if self.cogrouped() && !d.sides_seen[1] {
            self.empty_side(d, 1);
        }
        d.backlog.push_back(Cut::EndGroup);
        Ok(())
    }

    /// Cuts `batch` (reserved in `held_bytes`) into the backlog.
    fn feed(&self, d: &mut Driver, batch: &RecordBatch, runs: Vec<Run>) -> Result<()> {
        let grouped = self.encoder.is_some();
        for run in runs {
            if run.new_group {
                if d.group_open {
                    self.end_group(d)?;
                }
                d.group_open = true;
                d.sides_seen = [false; 2];
            } else if grouped && !d.group_open {
                return Err(internal("Python UDF rows before their group".to_owned()));
            }
            if run.side != d.held_side {
                self.cut_held(d, true)?;
                d.held_side = run.side;
            }
            if self.cogrouped() && run.side == 1 && !d.sides_seen[0] {
                // A key on the right only: its empty left side first.
                self.empty_side(d, 0);
            }
            d.sides_seen[usize::from(run.side)] = true;
            d.held.push(batch.slice(run.start, run.len));
            d.held_rows += run.len;
            self.cut_held(d, false)?;
        }
        Ok(())
    }

    /// Frames the backlog into the lanes while every lane's window has
    /// room. Nothing changes for a frame whose copies the pool refuses.
    fn flush(&self, d: &mut Driver) -> Result<Flushed> {
        loop {
            if d.lanes
                .iter()
                .any(|l| lock(&l.io).in_flight() >= self.window)
            {
                return Ok(Flushed::Window);
            }
            let Some(cut) = d.backlog.front() else {
                return Ok(Flushed::All);
            };
            match cut {
                Cut::Rows { side, rows, .. } => {
                    let mut frames = Vec::with_capacity(self.lanes.len());
                    for lane in &self.lanes {
                        frames.push(Self::frame(lane, *side, rows, d.partition)?);
                    }
                    let copied: usize = frames.iter().map(|(_, c)| c).sum();
                    let more = copied.saturating_sub(d.spare.size());
                    if more > 0 && d.spare.try_grow(more).is_err() {
                        return Ok(Flushed::Memory(more));
                    }
                    let Some(Cut::Rows { memory, .. }) = d.backlog.pop_front() else {
                        return Err(internal("the Python UDF backlog changed".to_owned()));
                    };
                    let shared = Arc::new(Mutex::new(memory));
                    for ((lane, state), (frame, copied)) in
                        self.lanes.iter().zip(&d.lanes).zip(frames)
                    {
                        let memory = FrameMem {
                            _shared: Some(Arc::clone(&shared)),
                            _own: d.spare.split(copied),
                        };
                        state.queue(lane.udf.as_ref(), frame, memory)?;
                    }
                }
                Cut::EndGroup | Cut::End => {
                    let end = matches!(cut, Cut::End);
                    d.backlog.pop_front();
                    for (lane, state) in self.lanes.iter().zip(&d.lanes) {
                        // A task never opened (no rows) gets no `End`.
                        if end && !state.opened() {
                            continue;
                        }
                        let frame = if end { Frame::End } else { Frame::EndGroup };
                        let memory = FrameMem {
                            _shared: None,
                            _own: d.spare.split(0),
                        };
                        state.queue(lane.udf.as_ref(), frame, memory)?;
                    }
                }
            }
        }
    }

    /// A task's batch as an output batch: its columns checked against the
    /// declared client types, then moved to the canonical carriers. The
    /// runtime checks the result against the declared schema first; this is
    /// the engine's own guard.
    fn output(&self, lane: &RelationLane, batch: &RecordBatch) -> Result<Batch> {
        if batch.num_columns() != lane.client.len() {
            return Err(mismatch(
                lane,
                format!(
                    "{} columns where {} were declared",
                    batch.num_columns(),
                    lane.client.len()
                ),
            ));
        }
        let mut columns = Vec::with_capacity(batch.num_columns());
        for ((c, client), field) in batch
            .columns()
            .iter()
            .zip(&lane.client)
            .zip(self.schema.fields())
        {
            columns.push(recarried(lane, c, client, field.data_type())?);
        }
        let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
        let out = RecordBatch::try_new_with_options(Arc::clone(&self.schema), columns, &options)
            .map_err(|e| mismatch(lane, format!("{e}")))?;
        Ok(Batch::new(out))
    }

    /// The next finished output batch of a driver, with the memory of the
    /// results it came from: a lane's next result (map and grouped shapes)
    /// or the groups every lane has a value for (aggregate).
    fn next_output(&self, d: &mut Driver) -> Result<Option<(Batch, Vec<Reservation>)>> {
        if let (RelationShape::Aggregate { .. }, Some(key_schema)) = (&self.shape, &self.key_schema)
        {
            return pair::joined(
                &mut d.pending,
                key_schema,
                &self.schema,
                &self.lanes,
                &d.lanes,
            );
        }
        for (lane, state) in self.lanes.iter().zip(&d.lanes) {
            let next = lock(&state.io).outputs.pop_front();
            if let Some((batch, memory)) = next {
                return Ok(Some((self.output(lane, &batch)?, vec![memory])));
            }
        }
        Ok(None)
    }

    /// Stores `batch` in the list, its results' bytes (`memory`) handed to
    /// the store's reservation; or keeps it, reserved again, for the retry
    /// after the memory the store asked for.
    fn store(
        &self,
        l: &mut RelLocal,
        batch: Batch,
        memory: Vec<Reservation>,
        g: &RelGlobal,
    ) -> Result<Option<SinkStatus>> {
        // As the store keeps it, so the bytes lent are the bytes it takes.
        let batch = crate::operators::owned_rows(batch)?;
        let lent = lend(memory, &l.consumer, batch.data().get_array_memory_size());
        match self.store.sink(batch.clone(), &mut l.store, &g.store)? {
            SinkStatus::NeedMemory { bytes } => {
                let back = l.consumer.reservation_from_credit(lent);
                l.driver.ready = Some((batch, vec![back]));
                Ok(Some(SinkStatus::NeedMemory {
                    bytes: bytes.saturating_sub(lent).max(1),
                }))
            }
            _ => Ok(None),
        }
    }

    /// Moves each lane's frames and results without waiting; the first
    /// failure of a lane.
    fn progress(d: &Driver) -> Result<()> {
        let mut cx = Context::from_waker(noop_waker_ref());
        for lane in &d.lanes {
            if let Some(Err(e)) = lane.poll(&mut cx) {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Stores what the driver's tasks yielded, reserving refused results
    /// first; a status to return if the pool or the store asked for memory.
    fn store_results(&self, l: &mut RelLocal, g: &RelGlobal) -> Result<Option<SinkStatus>> {
        if let Some((ready, memory)) = l.driver.ready.take()
            && let Some(status) = self.store(l, ready, memory, g)?
        {
            return Ok(Some(status));
        }
        Self::progress(&l.driver)?;
        while let Some((batch, memory)) = self.next_output(&mut l.driver)? {
            if let Some(status) = self.store(l, batch, memory, g)? {
                return Ok(Some(status));
            }
        }
        for lane in &l.driver.lanes {
            if let Some(bytes) = lane.try_take_refused() {
                return Ok(Some(SinkStatus::NeedMemory { bytes }));
            }
        }
        // The pumps read on past a result just reserved.
        Self::progress(&l.driver)?;
        Ok(None)
    }

    /// Takes `input` into the backlog: `NeedMemory` (nothing changed) if
    /// the pool refuses its bytes.
    fn take_input(&self, d: &mut Driver, input: Batch) -> Result<Option<SinkStatus>> {
        if !input.is_unweighted() {
            // A retracting row sent to Python would come back as an insert.
            return Err(krishiv_common::error::KrishivUnsupportedFeature {
                feature: "a weighted (change-stream) input to a Python UDF relation".to_owned(),
            }
            .into());
        }
        let input = crate::operators::owned_rows(input)?;
        let data = input.data();
        let (runs, last) = self.runs(data, d.current.as_ref())?;
        let keys = match (&self.shape, &self.key_schema) {
            (RelationShape::Aggregate { key_values, .. }, Some(schema)) => {
                let starts: Vec<u32> = runs
                    .iter()
                    .filter(|r| r.new_group)
                    .map(|r| u32::try_from(r.start))
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|_| internal("a batch over u32 rows".to_owned()))?;
                pair::key_rows(data, key_values, schema, &starts)?
            }
            _ => None,
        };
        let key_bytes = keys.as_ref().map_or(0, RecordBatch::get_array_memory_size);
        let bytes = data.get_array_memory_size() + key_bytes;
        if d.held_bytes.try_grow(bytes).is_err() {
            return Ok(Some(SinkStatus::NeedMemory { bytes }));
        }
        if let Some(keys) = keys {
            let memory = d.held_bytes.split(key_bytes);
            d.pending.push(keys, memory);
        }
        d.held_bytes_rows += data.num_rows();
        self.feed(d, data, runs)?;
        d.current = last;
        Ok(None)
    }
}

/// A result that does not match what the UDF declared.
pub(super) fn mismatch(lane: &RelationLane, reason: String) -> KrishivError {
    KrishivPythonUdfResultMismatch {
        name: lane.name.clone(),
        reason,
    }
    .into()
}

/// `c` checked to be `client`, moved to the canonical `canonical`.
pub(super) fn recarried(
    lane: &RelationLane,
    c: &ArrayRef,
    client: &ArrowType,
    canonical: &ArrowType,
) -> Result<ArrayRef> {
    if c.data_type() != client {
        return Err(mismatch(
            lane,
            format!("{} where {client} was declared", c.data_type()),
        ));
    }
    crate::carrier::recarried_array(c, canonical)
}

/// Lends the bytes `parts` hold (results being stored) to `handle`'s grant
/// credit, up to `bytes`, so the reservation that stores the same rows takes
/// them without asking the pool again; the rest is freed. The bytes lent:
/// what to take back with `reservation_from_credit` if that reservation is
/// refused.
fn lend(parts: Vec<Reservation>, handle: &Arc<ConsumerHandle>, bytes: usize) -> usize {
    let mut lent = 0;
    for mut part in parts {
        if part.consumer() != handle.id() && part.transfer_to(handle).is_err() {
            // Freed as it drops.
            continue;
        }
        let take = part.size().min(bytes - lent);
        part.shrink_to_credit(take);
        lent += take;
    }
    lent
}

/// The future of a `Blocked` driver: ready once every lane's window has
/// room, a lane ended (its failure is the next call's), or a result waits
/// for memory (the next call asks for it).
fn blocked(lanes: Vec<Lane>, window: usize) -> BoxFuture<'static, ()> {
    Box::pin(poll_fn(move |cx| {
        let mut room = true;
        for lane in &lanes {
            if lane.poll(cx).is_some() || lock(&lane.io).refused.is_some() {
                return Poll::Ready(());
            }
            room &= lock(&lane.io).in_flight() < window;
        }
        if room { Poll::Ready(()) } else { Poll::Pending }
    }))
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
            consumer: Arc::clone(consumer),
            store: self.store.local_state(ctx, &g.store)?,
            driver: Driver {
                lanes: self
                    .lanes
                    .iter()
                    .map(|_| Lane::new(consumer.reservation()))
                    .collect(),
                ready: None,
                partition: ctx.partition(),
                current: None,
                group_open: false,
                held: Vec::new(),
                held_rows: 0,
                held_side: 0,
                sides_seen: [false; 2],
                held_bytes: consumer.reservation(),
                held_bytes_rows: 0,
                backlog: VecDeque::new(),
                spare: consumer.reservation(),
                wants: 0,
                pending: Pending::default(),
            },
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
        // Results and the backlog first: a `NeedMemory` here leaves the
        // input untaken, so the retry with the same batch is exact.
        if let Some(status) = self.store_results(l, g)? {
            return Ok(status);
        }
        if let Flushed::Memory(bytes) = self.flush(&mut l.driver)? {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        if input.num_rows() > 0 {
            if let Some(status) = self.take_input(&mut l.driver, input)? {
                return Ok(status);
            }
            // A refusal here waits for the next call, which asks for it.
            self.flush(&mut l.driver)?;
            Self::progress(&l.driver)?;
        }
        let d = &l.driver;
        if d.lanes
            .iter()
            .any(|lane| lock(&lane.io).in_flight() >= self.window)
        {
            return Ok(SinkStatus::Blocked(blocked(d.lanes.clone(), self.window)));
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<RelGlobal>(global, "PythonRelation")?;
        let l = local
            .downcast::<RelLocal>()
            .map_err(|_| internal("wrong local state for PythonRelation".to_owned()))?;
        let RelLocal {
            store, mut driver, ..
        } = *l;
        // The input ended: the open group ends, then the partition. What the
        // windows do not take now, `finalize` frames.
        if driver.group_open {
            self.end_group(&mut driver)?;
        } else {
            self.cut_held(&mut driver, true)?;
        }
        driver.backlog.push_back(Cut::End);
        self.flush(&mut driver)?;
        Self::progress(&driver)?;
        lock(&g.leftover).push(driver);
        self.store.combine(store, &g.store)
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global_ref::<RelGlobal>(global, "PythonRelation")?;
            let mut drivers = std::mem::take(&mut *lock(&g.leftover));
            // Every task at once; results are stored as they arrive. The
            // first failure (or a cancel) ends the phase, and the error
            // drops the other tasks, which cancels them.
            loop {
                ctx.query().check_cancelled()?;
                let mut done = true;
                for d in &mut drivers {
                    done &= self.finish_step(d, g, ctx).await?;
                }
                if done {
                    break;
                }
                // A frame waiting for its copies' memory: every driver's
                // results are stored by now (written out where a frame
                // waits), so only the pool can give it.
                if let Some(d) = drivers.iter_mut().find(|d| d.wants > 0) {
                    let bytes = std::mem::take(&mut d.wants);
                    d.spare.grow(bytes).await.map_err(KrishivError::from)?;
                    continue;
                }
                self.wait(&drivers, ctx).await?;
            }
            if matches!(self.shape, RelationShape::Aggregate { .. }) {
                for d in &drivers {
                    pair::check_complete(&d.pending, &self.lanes, &d.lanes)?;
                }
            }
            drop(drivers);
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

impl RelationSink {
    /// One step of a finished driver in `finalize`: the backlog framed as
    /// far as the windows and the pool allow, the results stored, a refused
    /// map or grouped result with them (written to a spill file when the
    /// pool has no room, or when a frame waits for its copies' memory: the
    /// results' memory is what it can get), the backlog framed again, then
    /// an aggregate's refused values reserved if the pool has room now. A
    /// frame still refused leaves its bytes in `wants`, which `finalize`
    /// waits for once every driver stored its results. Whether the driver
    /// is done: no backlog, every task ended, every result stored (an
    /// aggregate's values left over are `check_complete`'s error).
    async fn finish_step(&self, d: &mut Driver, g: &RelGlobal, ctx: &TaskContext) -> Result<bool> {
        d.wants = wanted(self.flush(d)?);
        let mut batches = Vec::new();
        let mut held = Vec::new();
        if let Some((batch, memory)) = d.ready.take() {
            batches.push(batch);
            held.extend(memory);
        }
        while let Some((batch, memory)) = self.next_output(d)? {
            batches.push(batch);
            held.extend(memory);
        }
        let aggregate = matches!(self.shape, RelationShape::Aggregate { .. });
        if !aggregate {
            // After the lane's reserved results, in its order.
            for (lane, state) in self.lanes.iter().zip(&d.lanes) {
                if let Some(batch) = state.take_refused_unreserved() {
                    batches.push(self.output(lane, &batch)?);
                }
            }
        }
        if !batches.is_empty() {
            let bytes = batches
                .iter()
                .map(|b| b.data().get_array_memory_size())
                .sum();
            // Kept in memory if the pool has room, else written out: the
            // list can spill only once the store finalized.
            let handle = ctx.consumer().consumer();
            let lent = lend(std::mem::take(&mut held), handle, bytes);
            let mut reservation = ctx.consumer().reservation();
            let free = d.wants > 0 && self.store.spills();
            if !free && reservation.try_grow(bytes).is_ok() {
                self.store.add_part(&g.store, batches, reservation)?;
            } else {
                drop((reservation, handle.reservation_from_credit(lent)));
                self.store.add_spilled_part(&g.store, batches, ctx).await?;
            }
        }
        if d.wants > 0 {
            d.wants = wanted(self.flush(d)?);
        }
        if aggregate {
            // The stored results' memory is free again; a value still
            // refused waits in `wait` while the other lanes move.
            for lane in &d.lanes {
                let _missing = lane.try_take_refused();
            }
        }
        Ok(d.backlog.is_empty()
            && d.ready.is_none()
            && d.lanes
                .iter()
                .all(|l| l.finished() && (aggregate || !lock(&l.io).has_results())))
    }

    /// Waits until a driver can move: a lane yielded (for an aggregate:
    /// every lane, as `pair::joined` needs, or a refused value got its
    /// memory), failed, ended or freed room in its window for a backlog; or
    /// the query was cancelled.
    async fn wait(&self, drivers: &[Driver], ctx: &TaskContext) -> Result<()> {
        let aggregate = matches!(self.shape, RelationShape::Aggregate { .. });
        let mut reserving: Vec<BoxFuture<'static, Result<()>>> = if aggregate {
            drivers
                .iter()
                .flat_map(|d| d.lanes.iter().filter_map(Lane::reserve_refused))
                .collect()
        } else {
            Vec::new()
        };
        let mut cancelled = std::pin::pin!(ctx.query().cancelled());
        poll_fn(|cx| {
            if cancelled.as_mut().poll(cx).is_ready() {
                return Poll::Ready(Err(ctx.query().cancellation()));
            }
            for f in &mut reserving {
                if let Poll::Ready(r) = f.as_mut().poll(cx) {
                    return Poll::Ready(r);
                }
            }
            let mut ready = false;
            let mut all_done = true;
            for d in drivers {
                let mut room = true;
                let mut joinable = aggregate;
                for lane in &d.lanes {
                    if let Some(Err(e)) = lane.poll(cx) {
                        return Poll::Ready(Err(e));
                    }
                    all_done &= lane.finished();
                    let q = lock(&lane.io);
                    if aggregate {
                        // Only rows can be joined: a lane holding nothing but
                        // empty batches would make `finalize` spin.
                        joinable &= q.outputs.iter().any(|(b, _)| b.num_rows() > 0);
                    } else {
                        ready |= q.has_results();
                    }
                    room &= q.in_flight() < self.window;
                }
                ready |= joinable || (room && !d.backlog.is_empty());
                all_done &= d.backlog.is_empty() && d.ready.is_none();
            }
            if ready || all_done {
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        })
        .await
    }
}

/// The bytes a [`Flushed`] frame waits for (0: none).
fn wanted(flushed: Flushed) -> usize {
    match flushed {
        Flushed::Memory(bytes) => bytes,
        Flushed::All | Flushed::Window => 0,
    }
}

/// Per lane the frames in flight, and the cuts in the backlog, of a
/// driver's state.
#[cfg(test)]
pub(super) fn queued(local: &mut LocalState) -> Result<(Vec<usize>, usize)> {
    let l = local_mut::<RelLocal>(local, "PythonRelation")?;
    let d = &l.driver;
    let lanes = d.lanes.iter().map(|l| lock(&l.io).in_flight()).collect();
    Ok((lanes, d.backlog.len()))
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
