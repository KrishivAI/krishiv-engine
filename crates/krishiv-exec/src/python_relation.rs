// SPDX-License-Identifier: Apache-2.0
//! The runtime seam of the Python UDF relation kinds (design P3-T085 §3.2):
//! map, grouped map, cogrouped map, grouped aggregate (and, later, window
//! aggregate and UDTF).
//!
//! The scalar kinds keep [`PreparedUdf::call`](crate::python_eval::PreparedUdf)
//! (a column per argument in, a column out). A relation kind sees a
//! relation, so its call is a **task fed by frames**: the worker prepares
//! the call site once ([`UdfRuntime::prepare_relation`]), opens one
//! [`UdfTask`] per driver (a partition read by several drivers opens one
//! task per driver, an empty one none), sends it the driver's rows as
//! [`Frame`]s and reads whatever batches it yields. Output rows are not
//! aligned to input rows.
//!
//! [`Frame`] mirrors Spark's grouped wire form (`PythonArrowInput.scala`
//! L251-300 at v4.2.0): a group's batches, cut by
//! `spark.sql.execution.arrow.maxRecordsPerBatch` (and
//! `maxBytesPerBatch`), then a marker. A map task gets batches then
//! [`Frame::End`]; a grouped task ends each group with
//! [`Frame::EndGroup`]; a cogrouped task gets a key's left rows (side 0),
//! then its right rows (side 1), then `EndGroup`. The embedded runtime
//! (PR-48) and the worker pool (PR-26c) carry the same frames: one
//! protocol, two transports.
//!
//! **Keys.** A grouped or cogrouped kind's frames lead with the
//! [`RelationCallSpec::key_count`] key columns (computed keys included),
//! then the side's data columns: Spark's layout
//! (`PandasGroupUtils.resolveArgOffsets`), from which the Python side builds
//! `key` for `f(key, pdf)` and the frame from the rest. Every row of a
//! group has the same key values.
//!
//! **Cogroup sides.** Every key gets a frame for both sides: a side without
//! rows for a key gets one zero-row [`Frame::Batch`] (its columns' names and
//! types in the batch's schema, as [`RelationCallSpec::input_names`] and
//! [`RelationCallSpec::input_types`] list them), so the Python side builds
//! Spark's empty DataFrame for it. The side order stays left then right.
//!
//! [`UdfRuntime::prepare_relation`] has a default that refuses with
//! `KRISHIV_UDF_NOT_AVAILABLE`, so a runtime that implements only the
//! scalar kinds keeps compiling and a plan calling a relation kind fails
//! on it as on a worker without a runtime.

use std::collections::BTreeMap;
use std::fmt;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_plan::PythonEvalType;
use krishiv_types::DataType;

use crate::error::Result;

pub(crate) mod sink;

pub use sink::{
    DEFAULT_WINDOW, InputColumn, RelationLane, RelationLimits, RelationShape, RelationSink,
};

/// What a runtime prepares one relation call site from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationCallSpec {
    /// The UDF's canonical (lower-case) name.
    pub name: String,
    /// The registration version the plan was analyzed with.
    pub version: u32,
    /// How it is converted and called (a relation kind).
    pub eval_type: PythonEvalType,
    /// Whether tasks may run concurrently.
    pub thread_safe: bool,
    /// The Spark types of the columns of each side's frames, in frame
    /// order: one side, two for a cogroup. A grouped kind's frames lead
    /// with its [`Self::key_count`] key columns, then the side's data
    /// columns.
    pub input_types: Vec<Vec<DataType>>,
    /// The names of the columns of each side's frames, as
    /// [`Self::input_types`] orders them: what an empty side's DataFrame is
    /// built from.
    pub input_names: Vec<Vec<String>>,
    /// The grouping keys (0 for a map kind and an aggregate's arguments):
    /// the leading columns of every side's frames. With [`Self::arity`]
    /// what the Python side needs to call `f(pdf)` or `f(key, pdf)`.
    pub key_count: usize,
    /// The data columns of a side's frames, keys excluded (the first
    /// side's for a cogroup; the other's is its `input_types` length less
    /// `key_count`).
    pub arity: usize,
    /// The declared result: names and Spark types, one per column.
    pub output_names: Vec<String>,
    /// The declared result's Spark types.
    pub output_types: Vec<DataType>,
    /// The Arrow schema the task's batches must have: the declared types'
    /// client forms (decimals on `Decimal128`, strings and binaries as
    /// views), as [`UdfCallSpec::return_arrow`](crate::python_eval::UdfCallSpec).
    pub output_arrow: SchemaRef,
    /// The runner configuration (Spark's `PythonRunner` conf): the session's
    /// explicit settings the Python side reads.
    pub conf: RunnerConf,
}

/// The settings a Python runner reads (Spark's runner conf).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RunnerConf {
    /// `spark.sql.execution.arrow.maxRecordsPerBatch`: most rows per frame;
    /// 0 is no limit.
    pub max_records_per_batch: usize,
    /// `spark.sql.execution.arrow.maxBytesPerBatch`: most bytes per frame
    /// (cut between rows, at least one row); 0 is no limit.
    pub max_bytes_per_batch: usize,
    /// Every explicit session setting, by key.
    pub settings: BTreeMap<String, String>,
}

/// One message to a [`UdfTask`].
#[derive(Debug, Clone)]
pub enum Frame {
    /// Rows of the current group (of the partition, for a map kind); for
    /// a cogroup, zero rows for a side the key has no rows on.
    Batch {
        /// The relation the rows come from: 0, or 1 for a cogroup's right.
        side: u8,
        /// The rows, every column in its client form.
        batch: RecordBatch,
    },
    /// The current group has no more rows.
    EndGroup,
    /// The partition has no more rows: the last frame.
    End,
}

/// One prepared relation call site.
pub trait PreparedRelationUdf: Send + Sync {
    /// One task per driver: the frames in, the result batches out.
    ///
    /// # Errors
    /// What the runtime raises starting it.
    fn open(&self) -> Result<Box<dyn UdfTask>>;
}

/// A running relation call. The two methods may be awaited at once (the
/// operator sends while it reads, so a runtime may wait for its results to
/// be taken before it takes more frames); each is awaited by one caller at a
/// time. The work runs off the caller, on the runtime's own threads:
/// polling only moves frames and batches. Dropping the task cancels it.
pub trait UdfTask: Send + Sync {
    /// Queues `frame`; waits while the task's bounded input window is full.
    ///
    /// # Errors
    /// The task's failure (a Python error, a crashed worker).
    fn send(&self, frame: Frame) -> BoxFuture<'_, Result<()>>;

    /// The next result batch (the client forms of the declared types);
    /// `None` once [`Frame::End`] is processed and every result was read.
    ///
    /// # Errors
    /// The task's failure.
    fn next(&self) -> BoxFuture<'_, Result<Option<RecordBatch>>>;
}

impl fmt::Debug for dyn UdfTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("UdfTask")
    }
}

#[cfg(test)]
mod tests;
