// SPDX-License-Identifier: Apache-2.0
//! The embedded Python UDF runtime (design P2-T059 §2.5, unit 2b): the
//! [`UdfRuntime`] a Python session gives its workers.
//!
//! A registered UDF is its Python wrapper (built by `krishiv.sql`: one
//! `pyarrow.Array` per argument in, one `pyarrow.Array` of the declared type
//! out), keyed by `(name, version)` as plans name it. A call is one job on
//! the process's current pool ([`super::current`], fetched per call so a
//! forked child uses its own): the arguments cross as Arrow capsules (`export_column`), the
//! wrapper runs, and its result comes back through `import_column`, checked
//! against the declared type and the row count while its capsules are
//! alive. The engine side only awaits the job's result.
//!
//! An unregistered UDF (a DataFrame-side `udf(...)`, design unit 2c (a))
//! lives in a process-wide table under a version with the top bit set, so
//! any session's runtime runs it. A call's runner conf (the SQL conf keys
//! Spark's worker reads) is read from the session's conf when the query
//! prepares the call, and handed to the wrapper per batch through
//! `_native._udf_runner_conf()`.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError, RwLock};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use krishiv_common::error::{InternalError, KrishivError, KrishivUdfNotAvailable, PythonException};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::{MutexExt, PyOnceLock};
use pyo3::types::{PyCapsule, PyDict, PyTuple, PyType};

use super::interrupt::{Call, CancelOnDrop};
use super::pool::Pool;
use crate::stream::capsule_pointer;
use crate::stream::ffi::{
    ARRAY_CAPSULE, SCHEMA_CAPSULE, export_column, import_column, owned_tight_copy,
};

thread_local! {
    /// Set on a pool thread while a UDF's Python code runs: a call back into
    /// the engine from there fails ([`refuse_inside_udf`]) instead of waiting
    /// for the pool thread it holds (§2.5 "Reentrancy", T098b B1).
    static IN_UDF: Cell<bool> = const { Cell::new(false) };
    /// The runner conf of the call running on this pool thread.
    static RUNNER: std::cell::RefCell<Option<Arc<RunnerConf>>> = const { std::cell::RefCell::new(None) };
    /// The row count of the call running on this pool thread (a call
    /// without arguments has no array to read it from).
    static ROWS: Cell<usize> = const { Cell::new(0) };
}

/// The SQL conf keys Spark's Python worker reads for these eval types
/// (PySpark 4.2.0 `worker.py:113-184`, `RunnerConf`); the wrappers apply
/// Spark's defaults to the unset ones.
const RUNNER_KEYS: [&str; 8] = [
    "spark.sql.legacy.execution.pandas.groupedMap.assignColumnsByName",
    "spark.sql.execution.arrow.useLargeVarTypes",
    "spark.sql.execution.pandas.convertToArrowArraySafely",
    "spark.sql.execution.pythonUDF.pandas.intToDecimalCoercionEnabled",
    "spark.sql.execution.pyspark.binaryAsBytes",
    "spark.sql.session.timeZone",
    "spark.sql.execution.pythonUDF.pandas.preferIntExtensionDtype",
    "spark.sql.execution.pythonUDF.arrow.concurrency.level",
];

/// The values of [`RUNNER_KEYS`] one query prepared its calls with.
type RunnerConf = Vec<(&'static str, String)>;

/// The process's unregistered UDFs, by version (top bit set).
static ANONYMOUS: LazyLock<RwLock<HashMap<u32, Arc<Registered>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// The last unregistered UDF's number (its version without the top bit).
static ANONYMOUS_COUNT: AtomicU32 = AtomicU32::new(0);

/// Adds an unregistered UDF's wrapper to the process's table; its version.
///
/// # Errors
/// After 2^31 - 1 of them in one process.
pub(crate) fn insert_anonymous(wrapper: Py<PyAny>, thread_safe: bool) -> PyResult<u32> {
    let n = ANONYMOUS_COUNT
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1);
    if n & krishiv_session::ANONYMOUS_VERSION_BIT != 0 {
        return Err(pyo3::exceptions::PyOverflowError::new_err(
            "too many unregistered Python UDFs in this process",
        ));
    }
    let version = n | krishiv_session::ANONYMOUS_VERSION_BIT;
    let lock = (!thread_safe).then(|| Arc::new(Mutex::new(())));
    ANONYMOUS
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(version, Arc::new(Registered { wrapper, lock }));
    Ok(version)
}

/// `_native._udf_runner_conf()`: the runner conf of the UDF call running on
/// this thread, as `{key: value}` for the keys the session sets or
/// defaults, plus `krishiv.python.udf.rows` (the call's row count); empty
/// outside a call.
#[pyfunction]
pub(crate) fn _udf_runner_conf(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    let Some(conf) = RUNNER.with(|r| r.borrow().clone()) else {
        return Ok(out);
    };
    for (key, value) in conf.iter() {
        out.set_item(*key, value)?;
    }
    out.set_item("krishiv.python.udf.rows", ROWS.with(Cell::get).to_string())?;
    Ok(out)
}

/// Raises PySpark's `CONTEXT_ONLY_VALID_ON_DRIVER` (a
/// `PySparkRuntimeError`) when called from inside a UDF: the engine call
/// would wait on the pool thread the UDF holds (a 1-thread pool, the GIL
/// default, or its own `thread_safe=false` lock). PySpark raises the same
/// class for a session or context used in code running on a worker
/// (`pyspark/core/context.py` 4.2.0 lines 2636-2647, `_assert_on_driver`;
/// lines 491-496, pickling the context).
pub(crate) fn refuse_inside_udf(py: Python<'_>) -> PyResult<()> {
    if !IN_UDF.with(Cell::get) {
        return Ok(());
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("errorClass", "CONTEXT_ONLY_VALID_ON_DRIVER")?;
    kwargs.set_item("messageParameters", PyDict::new(py))?;
    let err = py
        .import("krishiv.errors")?
        .getattr("PySparkRuntimeError")?
        .call((), Some(&kwargs))?;
    Err(PyErr::from_value(err))
}

/// The runner conf and row count of the call running on this thread, set
/// for the scope's life; the previous ones come back on drop.
struct RunnerScope {
    previous: Option<Arc<RunnerConf>>,
    previous_rows: usize,
}

impl RunnerScope {
    fn enter(conf: Arc<RunnerConf>, rows: usize) -> Self {
        Self {
            previous: RUNNER.with(|r| r.borrow_mut().replace(conf)),
            previous_rows: ROWS.with(|c| c.replace(rows)),
        }
    }
}

impl Drop for RunnerScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        RUNNER.with(|r| *r.borrow_mut() = previous);
        ROWS.with(|c| c.set(self.previous_rows));
    }
}

/// One registered UDF.
struct Registered {
    wrapper: Py<PyAny>,
    /// `thread_safe=false`: the lock its calls take, so at most one runs.
    lock: Option<Arc<Mutex<()>>>,
}

/// The Python UDFs of one session and the size of the pool their calls run
/// on.
pub(crate) struct PyUdfRuntime {
    udfs: RwLock<HashMap<(String, u32), Arc<Registered>>>,
    pool: Arc<PoolCache>,
    /// The session's conf, read when a query prepares a call.
    conf: krishiv_session::RuntimeConf,
}

/// The pool this runtime's calls last used, so a call takes the process's
/// pool lock only when there is none yet or it is a fork's copy (review N1:
/// that lock, inherited held by a fork, would hang the child; and every
/// batch would wait on it).
struct PoolCache {
    size: usize,
    cached: Mutex<Option<Arc<Pool>>>,
}

impl PoolCache {
    fn get(&self) -> Result<Arc<Pool>, KrishivError> {
        let mut cached = self.cached.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(p) = cached.as_ref()
            && p.is_local()
        {
            return Ok(Arc::clone(p));
        }
        let p = super::current(self.size)?;
        *cached = Some(Arc::clone(&p));
        Ok(p)
    }
}

impl fmt::Debug for PyUdfRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyUdfRuntime").finish_non_exhaustive()
    }
}

impl PyUdfRuntime {
    /// A runtime whose calls run on the process's pool, started with
    /// `pool_size` threads when there is none.
    pub(crate) fn new(pool_size: usize, conf: krishiv_session::RuntimeConf) -> Self {
        Self {
            udfs: RwLock::new(HashMap::new()),
            pool: Arc::new(PoolCache {
                size: pool_size,
                cached: Mutex::new(None),
            }),
            conf,
        }
    }

    /// Adds the wrapper of `(name, version)` (the name as registered, as
    /// plans name it); an older version of the name stays callable by the plans that
    /// name it.
    pub(crate) fn insert(&self, name: String, version: u32, wrapper: Py<PyAny>, thread_safe: bool) {
        let lock = (!thread_safe).then(|| Arc::new(Mutex::new(())));
        self.udfs
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert((name, version), Arc::new(Registered { wrapper, lock }));
    }
}

impl UdfRuntime for PyUdfRuntime {
    fn prepare(&self, spec: &UdfCallSpec) -> krishiv_exec::Result<Arc<dyn PreparedUdf>> {
        let registered = if spec.version & krishiv_session::ANONYMOUS_VERSION_BIT == 0 {
            self.udfs
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&(spec.name.clone(), spec.version))
                .cloned()
        } else {
            ANONYMOUS
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&spec.version)
                .cloned()
        };
        let registered = registered.ok_or_else(|| {
            KrishivError::from(KrishivUdfNotAvailable {
                name: spec.name.clone(),
                version: spec.version.to_string(),
            })
        })?;
        let runner: RunnerConf = RUNNER_KEYS
            .iter()
            .filter_map(|k| self.conf.get(k).map(|v| (*k, v)))
            .collect();
        Ok(Arc::new(Prepared {
            registered,
            declared: spec.return_arrow.clone(),
            pool: Arc::clone(&self.pool),
            runner: Arc::new(runner),
        }))
    }
}

struct Prepared {
    registered: Arc<Registered>,
    declared: DataType,
    pool: Arc<PoolCache>,
    runner: Arc<RunnerConf>,
}

impl PreparedUdf for Prepared {
    fn call(&self, args: RecordBatch) -> BoxFuture<'static, krishiv_exec::Result<ArrayRef>> {
        let pool = match self.pool.get() {
            Ok(p) => p,
            Err(e) => return Box::pin(std::future::ready(Err(e))),
        };
        let (done, result) = oneshot::channel();
        let registered = Arc::clone(&self.registered);
        let declared = self.declared.clone();
        let runner = Arc::clone(&self.runner);
        let call = Call::new();
        let cancel = CancelOnDrop(Arc::clone(&call));
        pool.submit(Box::new(move |py| {
            // A call whose future was dropped (its query cancelled or
            // failed) is skipped: it would hold the pool thread and its input
            // for a result nobody reads (T098b I2).
            if done.is_canceled() {
                return;
            }
            // Restores the outer call's conf and rows when dropped, also on
            // unwinding: a nested call on this thread (a UDF running a query)
            // must not leave the outer call reading another's (review 2c M2).
            let _scope = RunnerScope::enter(runner, args.num_rows());
            let result = invoke(py, &registered, &args, &declared, &call);
            let _ = done.send(result);
        }));
        Box::pin(async move {
            // Dropped with this future: a call still running is interrupted.
            let _cancel = cancel;
            result.await.unwrap_or_else(|_| {
                Err(InternalError {
                    message: "a Python UDF call ended without a result (its pool thread panicked or stopped)"
                        .to_owned(),
                }
                .into())
            })
        })
    }
}

/// A call's Python code running on this pool thread: ends it when dropped,
/// also when the call unwinds (review 2026-11-09 I1). The in-UDF flag is
/// cleared and the call marked finished (an interrupt that came too late is
/// cleared with it), so neither reaches the thread's next call; a panic left
/// between `start` and `finish` would otherwise let a later cancel interrupt
/// that next call.
struct Running<'a, 'py> {
    call: &'a Call,
    py: Python<'py>,
}

impl<'a, 'py> Running<'a, 'py> {
    fn enter(call: &'a Call, py: Python<'py>) -> Self {
        IN_UDF.with(|f| f.set(true));
        Self { call, py }
    }
}

impl Drop for Running<'_, '_> {
    fn drop(&mut self) {
        IN_UDF.with(|f| f.set(false));
        self.call.finish(self.py);
    }
}

/// One call on a pool thread, attached.
fn invoke(
    py: Python<'_>,
    registered: &Registered,
    args: &RecordBatch,
    declared: &DataType,
    call: &Call,
) -> Result<ArrayRef, KrishivError> {
    // Imported once per process (design P2-T059 §6.3 profile): an import per
    // call takes the `sys.modules` lock, which pool threads contend on. A
    // forked child keeps the parent's object, the same type in its copied
    // interpreter; the reference is never released (one at exit).
    static ARRAY_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let array_type = ARRAY_TYPE
        .import(py, "pyarrow", "Array")
        .map_err(|e| python_error(py, &e))?;
    let mut columns = Vec::with_capacity(args.num_columns());
    for column in args.columns() {
        let (schema, array) = export_column(column)?;
        let imported = PyCapsule::new_with_value(py, schema, SCHEMA_CAPSULE)
            .and_then(|s| {
                PyCapsule::new_with_value(py, array, ARRAY_CAPSULE).and_then(|a| {
                    array_type.call_method1(intern!(py, "_import_from_c_capsule"), (s, a))
                })
            })
            .map_err(|e| python_error(py, &e))?;
        columns.push(imported);
    }
    let call_args = PyTuple::new(py, columns).map_err(|e| python_error(py, &e))?;
    let returned = {
        // Waited for detached (review B1): the holder needs the GIL to
        // finish its call.
        let _serial = registered.lock.as_ref().map(|l| {
            l.lock_py_attached(py)
                .unwrap_or_else(PoisonError::into_inner)
        });
        // Dropped while it waited for the lock: nobody reads its result.
        if !call.start(py) {
            return Err(InternalError {
                message: "a cancelled Python UDF call".to_owned(),
            }
            .into());
        }
        let running = Running::enter(call, py);
        let r = registered.wrapper.bind(py).call1(call_args);
        // At once: an interrupt that came too late is cleared here.
        drop(running);
        r
    }
    .map_err(|e| python_error(py, &e))?;
    // Only pyarrow's arrays are imported (§2.5 "Trusted producer").
    if !returned
        .is_instance(array_type)
        .map_err(|e| python_error(py, &e))?
    {
        return Err(InternalError {
            message: format!(
                "a Python UDF wrapper returned {} where a pyarrow.Array was expected",
                returned
                    .get_type()
                    .name()
                    .map_or_else(|_| "an object".to_owned(), |n| n.to_string())
            ),
        }
        .into());
    }
    let (schema, array): (Bound<'_, PyAny>, Bound<'_, PyAny>) = returned
        .call_method0(intern!(py, "__arrow_c_array__"))
        .and_then(|c| c.extract())
        .map_err(|e| python_error(py, &e))?;
    let schema_ptr = capsule_pointer(&schema, SCHEMA_CAPSULE).map_err(|e| python_error(py, &e))?;
    let array_ptr = capsule_pointer(&array, ARRAY_CAPSULE).map_err(|e| python_error(py, &e))?;
    // SAFETY: the pointers of the live `arrow_schema` and `arrow_array`
    // capsules `schema` and `array`, which this frame holds until the import
    // returns; the schema is only borrowed, the array moved out of once (the
    // capsule's destructor then sees a released array).
    let imported = unsafe {
        import_column(
            schema_ptr.cast(),
            array_ptr.cast(),
            declared,
            args.num_rows(),
        )
    }?;
    // The engine gets Rust-owned buffers; `imported` (and so the producer's
    // release, which may take the GIL for numpy-owned memory) is dropped here,
    // on this attached pool thread (design P2-T059 unit 3b, review M3).
    let owned = owned_tight_copy(&imported.to_data())?;
    drop(imported);
    crate::stream::ffi::catch_panic(|| arrow_array::make_array(owned))
}

/// A Python exception raised by a UDF (or by the conversion around it) as
/// Spark's `PYTHON_EXCEPTION` (SQLSTATE 38000) for every eval type, its
/// `traceback` parameter the Python traceback ending with the exception
/// line, as Spark 4.2.0 gives it (`research/spark-notes/python-udf-errors.md`:
/// `PythonRunner.scala:661-670`).
fn python_error(py: Python<'_>, err: &PyErr) -> KrishivError {
    let traceback = err
        .traceback(py)
        .and_then(|t| t.format().ok())
        .unwrap_or_default();
    PythonException {
        msg: "An exception was thrown from the Python worker".to_owned(),
        traceback: format!("{traceback}{err}"),
    }
    .into()
}
