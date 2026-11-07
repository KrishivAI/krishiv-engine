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

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use krishiv_common::error::{InternalError, KrishivError, KrishivUdfNotAvailable, PythonException};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyCapsule, PyTuple};

use super::pool::Pool;
use crate::stream::capsule_pointer;
use crate::stream::ffi::{ARRAY_CAPSULE, SCHEMA_CAPSULE, export_column, import_column};

thread_local! {
    /// Set on a pool thread while a UDF's Python code runs: a call back into
    /// the engine from there must fail, not wait for itself (§2.5
    /// "Reentrancy"). Read by the session's query entry points once Spark's
    /// error class for it is checked (pending; until then a reentrant call
    /// runs, and can only deadlock if it waits on its own pool thread).
    static IN_UDF: Cell<bool> = const { Cell::new(false) };
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
    pub(crate) fn new(pool_size: usize) -> Self {
        Self {
            udfs: RwLock::new(HashMap::new()),
            pool: Arc::new(PoolCache {
                size: pool_size,
                cached: Mutex::new(None),
            }),
        }
    }

    /// Adds the wrapper of `(name, version)` (name lower-case, as plans name
    /// it); an older version of the name stays callable by the plans that
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
        let registered = self
            .udfs
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(spec.name.clone(), spec.version))
            .cloned()
            .ok_or_else(|| {
                KrishivError::from(KrishivUdfNotAvailable {
                    name: spec.name.clone(),
                    version: spec.version.to_string(),
                })
            })?;
        Ok(Arc::new(Prepared {
            registered,
            declared: spec.return_arrow.clone(),
            pool: Arc::clone(&self.pool),
        }))
    }
}

struct Prepared {
    registered: Arc<Registered>,
    declared: DataType,
    pool: Arc<PoolCache>,
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
        pool.submit(Box::new(move |py| {
            let _ = done.send(invoke(py, &registered, &args, &declared));
        }));
        Box::pin(async move {
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

/// One call on a pool thread, attached.
fn invoke(
    py: Python<'_>,
    registered: &Registered,
    args: &RecordBatch,
    declared: &DataType,
) -> Result<ArrayRef, KrishivError> {
    let array_type = py
        .import("pyarrow")
        .and_then(|pa| pa.getattr("Array"))
        .map_err(|e| python_error(py, &e))?;
    let mut columns = Vec::with_capacity(args.num_columns());
    for column in args.columns() {
        let (schema, array) = export_column(column)?;
        let imported = PyCapsule::new_with_value(py, schema, SCHEMA_CAPSULE)
            .and_then(|s| {
                PyCapsule::new_with_value(py, array, ARRAY_CAPSULE)
                    .and_then(|a| array_type.call_method1("_import_from_c_capsule", (s, a)))
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
        IN_UDF.with(|f| f.set(true));
        let r = registered.wrapper.bind(py).call1(call_args);
        IN_UDF.with(|f| f.set(false));
        r
    }
    .map_err(|e| python_error(py, &e))?;
    // Only pyarrow's arrays are imported (§2.5 "Trusted producer").
    if !returned
        .is_instance(&array_type)
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
        .call_method0("__arrow_c_array__")
        .and_then(|c| c.extract())
        .map_err(|e| python_error(py, &e))?;
    let schema_ptr = capsule_pointer(&schema, SCHEMA_CAPSULE).map_err(|e| python_error(py, &e))?;
    let array_ptr = capsule_pointer(&array, ARRAY_CAPSULE).map_err(|e| python_error(py, &e))?;
    // SAFETY: the pointers of the live `arrow_schema` and `arrow_array`
    // capsules `schema` and `array`, which this frame holds until the import
    // returns; the schema is only borrowed, the array moved out of once (the
    // capsule's destructor then sees a released array).
    unsafe {
        import_column(
            schema_ptr.cast(),
            array_ptr.cast(),
            declared,
            args.num_rows(),
        )
    }
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
