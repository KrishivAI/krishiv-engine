// SPDX-License-Identifier: Apache-2.0
//! The embedded runtime of the Python relation UDF kinds (design P3-T085 §3.3): map, grouped map, cogrouped
//! map and grouped aggregate, run on the process's pool of pinned Python threads.
//!
//! A registered relation UDF is its **runner** (`krishiv._udf.relations.make_relation`: frames and the key
//! count in, result batches out), stored as the wrapper of its `(name, version)` as the scalar kinds' wrappers
//! are. One [`UdfTask`] is one runner call over a partition: it takes **one pool thread for its whole life**,
//! on which the runner's input iterator pulls the engine's frames from a bounded channel (waiting detaches the
//! thread from Python, so other Python threads run) and each result batch goes back through a bounded channel.
//! Frames and results cross as Arrow capsules through the audited bridge functions the scalar kinds use
//! (`export_column`, and `import_column` with full validation: pyarrow is the only trusted producer).
//!
//! **Pool occupancy.** A task holds its thread until its partition ends, so with a pool of one thread (the
//! GIL default) partitions run one after another. The pool has one **slot** per thread (`slots`), which a
//! driver takes for all its lanes together or waits for, holding none: a driver whose tasks outnumbered the
//! threads would hang, its first tasks waiting for frames that its last, queued task never reads. A driver
//! that needs more slots than threads fails with `KRISHIV_PYTHON_WORKER_LIMIT` naming
//! `krishiv.python.udf.threads`. A task whose runner stopped reading closes its input at once, so frames sent
//! after that are dropped, not waited for.
//!
//! **Cancellation.** Dropping the task drops its channels: a runner waiting for a frame raises, a result
//! waiting for room is abandoned, and a runner inside bytecode is interrupted as a scalar call is
//! (`interrupt::Call`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::executor::block_on;
use futures::future::BoxFuture;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::python_relation::{
    Frame, PreparedRelationUdf, RelationCallSpec, RelationSlot, RelationSlots, UdfTask,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::{MutexExt, PyOnceLock};
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use super::interrupt::{Call, CancelOnDrop};
use super::runtime::{
    PoolCache, PyUdfRuntime, Registered, RunnerConf, RunnerScope, Running, from_pyarrow,
    python_error, to_pyarrow,
};

/// Frames a task's input holds before the sender waits (Spark's JVM side is bounded the same way).
const FRAME_WINDOW: usize = 4;
/// Result batches a task holds before its runner waits for the engine to take them.
const RESULT_WINDOW: usize = 2;

impl PyUdfRuntime {
    /// The relation call site of `spec`.
    ///
    /// # Errors
    /// `KRISHIV_UDF_NOT_AVAILABLE` for a UDF this runtime does not hold; `KRISHIV_UNSUPPORTED_FEATURE` for a
    /// table function (PR-48b) or a UDF that is not `thread_safe`.
    pub(super) fn prepare_relation_call(
        &self,
        spec: &RelationCallSpec,
    ) -> Result<Arc<dyn PreparedRelationUdf>, KrishivError> {
        if spec.udtf.is_some() {
            return Err(KrishivUnsupportedFeature {
                feature: "Python table functions in an embedded session".to_owned(),
            }
            .into());
        }
        // A grouped-aggregate UDF used with OVER carries a window eval type; its window runner is PR-48b, and
        // the grouped-aggregate wrapper would return one value per partition, not per row.
        if matches!(spec.eval_type.code(), 203 | 253) {
            return Err(KrishivUnsupportedFeature {
                feature: "a Python aggregate UDF over a window in an embedded session".to_owned(),
            }
            .into());
        }
        let registered = self.lookup(&spec.name, spec.version)?;
        if registered.lock.is_some() {
            return Err(KrishivUnsupportedFeature {
                feature: "a Python relation UDF with thread_safe=False".to_owned(),
            }
            .into());
        }
        Ok(Arc::new(PreparedRelation {
            registered,
            spec: spec.clone(),
            pool: self.pool(),
            runner: Arc::new(self.runner_conf()),
        }))
    }
}

struct PreparedRelation {
    registered: Arc<Registered>,
    spec: RelationCallSpec,
    pool: Arc<PoolCache>,
    runner: Arc<RunnerConf>,
}

impl PreparedRelationUdf for PreparedRelation {
    fn capacity(&self) -> Option<Arc<dyn RelationSlots>> {
        // One slot per pool thread, taken by a driver for all its lanes together (module doc).
        let pool = self.pool.get().ok()?;
        // The pool's one handle: the operator groups a driver's lanes by its pointer.
        let handle = Arc::clone(pool.slots());
        let slots: Arc<dyn RelationSlots> = handle;
        Some(slots)
    }

    fn open(&self, slot: Option<RelationSlot>) -> krishiv_exec::Result<Box<dyn UdfTask>> {
        let pool = self.pool.get()?;
        let (frames_tx, frames_rx) = mpsc::channel::<Frame>(FRAME_WINDOW);
        let (results_tx, results_rx) =
            mpsc::channel::<Result<RecordBatch, KrishivError>>(RESULT_WINDOW);
        let call = Call::new();
        let cancel = CancelOnDrop(Arc::clone(&call));
        let finished = Arc::new(AtomicBool::new(false));
        let job = Job {
            registered: Arc::clone(&self.registered),
            spec: self.spec.clone(),
            runner: Arc::clone(&self.runner),
            frames: frames_rx,
            results: results_tx,
            call,
            finished: Arc::clone(&finished),
            _slot: slot,
        };
        pool.submit(Box::new(move |py| job.run(py)));
        Ok(Box::new(InProcessTask {
            frames: frames_tx,
            results: futures::lock::Mutex::new(results_rx),
            finished,
            _cancel: cancel,
        }))
    }
}

/// The engine's end of a running task.
struct InProcessTask {
    frames: mpsc::Sender<Frame>,
    results: futures::lock::Mutex<mpsc::Receiver<Result<RecordBatch, KrishivError>>>,
    /// Set by the job when it ran to its end: a closed result channel without it means the job was dropped
    /// unrun (the pool shut down), which is a failure, not an empty rest of the partition.
    finished: Arc<AtomicBool>,
    /// Interrupts a runner still in bytecode when the task is dropped.
    _cancel: CancelOnDrop,
}

impl UdfTask for InProcessTask {
    fn send(&self, frame: Frame) -> BoxFuture<'_, krishiv_exec::Result<()>> {
        let mut tx = self.frames.clone();
        Box::pin(async move {
            // A closed channel is a task that ended (or failed, which `next` reports): the frame is dropped.
            let _ = tx.send(frame).await;
            Ok(())
        })
    }

    fn next(&self) -> BoxFuture<'_, krishiv_exec::Result<Option<RecordBatch>>> {
        Box::pin(async move {
            let mut rx = self.results.lock().await;
            match rx.next().await {
                Some(Ok(batch)) => Ok(Some(batch)),
                Some(Err(e)) => Err(e),
                None if self.finished.load(Ordering::Acquire) => Ok(None),
                None => Err(InternalError {
                    message: "a Python relation UDF task ended before it ran to its end (the UDF pool shut down)"
                        .to_owned(),
                }
                .into()),
            }
        })
    }
}

/// The pool thread's end of a task.
struct Job {
    registered: Arc<Registered>,
    spec: RelationCallSpec,
    runner: Arc<RunnerConf>,
    frames: mpsc::Receiver<Frame>,
    results: mpsc::Sender<Result<RecordBatch, KrishivError>>,
    call: Arc<Call>,
    /// Told to the task when this job ends (see `InProcessTask::finished`).
    finished: Arc<AtomicBool>,
    /// Frees the pool slot when the job ends (or is dropped unrun).
    _slot: Option<RelationSlot>,
}

impl Job {
    fn run(self, py: Python<'_>) {
        let Self {
            registered,
            spec,
            runner,
            frames,
            mut results,
            call,
            finished,
            _slot,
        } = self;
        // A task dropped before it started has nobody to read it.
        if results.is_closed() {
            finished.store(true, Ordering::Release);
            return;
        }
        let _scope = RunnerScope::enter(runner, 0);
        let outcome = drive(py, &registered, &spec, frames, &results, &call);
        if let Err(e) = outcome {
            // The engine may be gone already; then there is nobody to tell.
            let _ = py.detach(|| block_on(results.send(Err(e))));
        }
        finished.store(true, Ordering::Release);
    }
}

/// One runner call: frames in, result batches out, until the runner ends.
fn drive(
    py: Python<'_>,
    registered: &Registered,
    spec: &RelationCallSpec,
    frames: mpsc::Receiver<Frame>,
    results: &mpsc::Sender<Result<RecordBatch, KrishivError>>,
    call: &Arc<Call>,
) -> Result<(), KrishivError> {
    static ARRAY_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static BATCH_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let array_type = ARRAY_TYPE
        .import(py, "pyarrow", "Array")
        .map_err(|e| python_error(py, &e))?;
    let batch_type = BATCH_TYPE
        .import(py, "pyarrow", "RecordBatch")
        .map_err(|e| python_error(py, &e))?;
    let source = Py::new(
        py,
        FrameSource {
            frames: Mutex::new(Some(frames)),
            names: spec.input_names.clone(),
            array_type: array_type.clone().unbind(),
            batch_type: batch_type.clone().unbind(),
        },
    )
    .map_err(|e| python_error(py, &e))?;
    if !call.start(py) {
        return Ok(());
    }
    let running = Running::enter(call, py);
    let produced = (|| -> Result<(), KrishivError> {
        let iterator = registered
            .wrapper
            .bind(py)
            .call1((source.clone_ref(py), spec.key_count, spec.arity))
            .map_err(|e| python_error(py, &e))?
            .try_iter()
            .map_err(|e| python_error(py, &e))?;
        for item in iterator {
            let batch = item.map_err(|e| python_error(py, &e))?;
            let batch = import_batch(py, array_type, batch_type, &batch, &spec.output_arrow)?;
            // Waiting for room detaches: the engine reads on its own threads.
            let mut tx = results.clone();
            let delivered = py.detach(|| block_on(tx.send(Ok(batch))));
            if delivered.is_err() {
                // The task was dropped: stop the runner.
                return Ok(());
            }
        }
        Ok(())
    })();
    // At once: an interrupt that came too late is cleared here.
    drop(running);
    // The input closes now, so a runner that stopped reading does not leave the engine waiting.
    source.bind(py).borrow().close(py);
    produced
}

/// A runner's result batch (a `pyarrow.RecordBatch`) as the engine's, checked against the declared schema.
fn import_batch(
    py: Python<'_>,
    array_type: &Bound<'_, PyType>,
    batch_type: &Bound<'_, PyType>,
    batch: &Bound<'_, PyAny>,
    declared: &SchemaRef,
) -> Result<RecordBatch, KrishivError> {
    // Only pyarrow's batches are imported (§2.5 "Trusted producer").
    if !batch
        .is_instance(batch_type)
        .map_err(|e| python_error(py, &e))?
    {
        return Err(InternalError {
            message: format!(
                "a Python relation UDF produced {} where a pyarrow.RecordBatch was expected",
                batch
                    .get_type()
                    .name()
                    .map_or_else(|_| "an object".to_owned(), |n| n.to_string())
            ),
        }
        .into());
    }
    let rows: usize = batch
        .getattr("num_rows")
        .and_then(|n| n.extract())
        .map_err(|e| python_error(py, &e))?;
    let width: usize = batch
        .getattr("num_columns")
        .and_then(|n| n.extract())
        .map_err(|e| python_error(py, &e))?;
    if width != declared.fields().len() {
        return Err(InternalError {
            message: format!(
                "a Python relation UDF produced {width} columns where its declared result has {}",
                declared.fields().len()
            ),
        }
        .into());
    }
    let mut columns = Vec::with_capacity(width);
    for (i, field) in declared.fields().iter().enumerate() {
        let column = batch
            .call_method1("column", (i,))
            .map_err(|e| python_error(py, &e))?;
        columns.push(from_pyarrow(
            py,
            array_type,
            &column,
            field.data_type(),
            rows,
        )?);
    }
    let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(rows));
    RecordBatch::try_new_with_options(Arc::clone(declared), columns, &options).map_err(|e| {
        InternalError {
            message: format!(
                "a Python relation UDF's result does not fit its declared schema: {e}"
            ),
        }
        .into()
    })
}

/// What a runner reads its frames from (`relations.FrameSource`): `next()` waits, detached, for the engine's
/// next frame.
#[pyclass(name = "_FrameSource", module = "krishiv._native", frozen)]
struct FrameSource {
    frames: Mutex<Option<mpsc::Receiver<Frame>>>,
    /// The column names of each side's frames.
    names: Vec<Vec<String>>,
    array_type: Py<PyType>,
    batch_type: Py<PyType>,
}

impl FrameSource {
    fn close(&self, py: Python<'_>) {
        // Dropping the receiver makes every later send fail, which the engine side takes as "ended". The lock
        // is taken detached while it waits: another Python thread may hold it inside `next`, waiting for a
        // frame.
        drop(
            self.frames
                .lock_py_attached(py)
                .unwrap_or_else(PoisonError::into_inner)
                .take(),
        );
    }

    fn batch<'py>(
        &self,
        py: Python<'py>,
        side: u8,
        batch: &RecordBatch,
    ) -> Result<Bound<'py, PyAny>, KrishivError> {
        let array_type = self.array_type.bind(py);
        let mut columns = Vec::with_capacity(batch.num_columns());
        for column in batch.columns() {
            columns.push(to_pyarrow(py, array_type, column)?);
        }
        let names = self.names.get(usize::from(side)).ok_or_else(|| {
            KrishivError::from(InternalError {
                message: format!(
                    "a frame for side {side} of a call with {} sides",
                    self.names.len()
                ),
            })
        })?;
        if columns.is_empty() {
            // `from_arrays([], [])` has no length to carry: a zero-column input keeps its row count as a
            // struct array of empty structs.
            let pa = py.import("pyarrow").map_err(|e| python_error(py, &e))?;
            let empty_type = pa
                .call_method1("struct", (PyList::empty(py),))
                .map_err(|e| python_error(py, &e))?;
            let rows = PyList::new(py, (0..batch.num_rows()).map(|_| PyDict::new(py)))
                .map_err(|e| python_error(py, &e))?;
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("type", empty_type)
                .map_err(|e| python_error(py, &e))?;
            let array = pa
                .call_method("array", (rows,), Some(&kwargs))
                .map_err(|e| python_error(py, &e))?;
            return self
                .batch_type
                .bind(py)
                .call_method1("from_struct_array", (array,))
                .map_err(|e| python_error(py, &e));
        }
        let columns = PyList::new(py, columns).map_err(|e| python_error(py, &e))?;
        self.batch_type
            .bind(py)
            .call_method1("from_arrays", (columns, names.clone()))
            .map_err(|e| python_error(py, &e))
    }
}

#[pymethods]
impl FrameSource {
    /// The next frame as `("batch", side, batch)`, `("end_group",)` or `("end",)`.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        // Another Python thread calling `next` waits for the lock detached (never attached, which would stop
        // the thread holding it from re-attaching).
        let mut guard = self
            .frames
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner);
        let Some(frames) = guard.as_mut() else {
            return Err(PyRuntimeError::new_err(
                "the Python relation task has ended",
            ));
        };
        let frame = py.detach(|| block_on(frames.next()));
        drop(guard);
        let to_err = |e: KrishivError| PyRuntimeError::new_err(e.to_string());
        match frame {
            // The engine dropped the task: no more frames will come.
            None => Err(PyRuntimeError::new_err(
                "the Python relation task was cancelled",
            )),
            Some(Frame::Batch { side, batch }) => {
                let batch = self.batch(py, side, &batch).map_err(to_err)?;
                PyTuple::new(
                    py,
                    [
                        intern_str(py, "batch")?,
                        side.into_pyobject(py)?.into_any(),
                        batch,
                    ],
                )
            }
            Some(Frame::EndGroup) => PyTuple::new(py, [intern_str(py, "end_group")?]),
            Some(Frame::End) => PyTuple::new(py, [intern_str(py, "end")?]),
        }
    }
}

fn intern_str<'py>(py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(pyo3::types::PyString::new(py, s).into_any())
}
