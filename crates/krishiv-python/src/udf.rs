// SPDX-License-Identifier: Apache-2.0
//! Embedded Python UDFs (design P2-T059 §2.5, unit 2b): the process's pool
//! of pinned threads, the per-session [`runtime::PyUdfRuntime`], and the
//! exit hook `krishiv` registers at import (`atexit`).
//!
//! One pool per process, started by the first UDF call and sized by the
//! session making it (`krishiv.python.udf.threads`, read when the session
//! is created; 0 is the session executor's compute thread count on a
//! free-threaded Python, else 1). A call asks for the current pool ([`current`]) when its runtime
//! has none or holds a fork's copy: after `os.fork` the child's copy of the
//! parent's pool has no threads (they stayed in the parent), so it is left alone, never
//! stopped or joined, and the child starts a pool of its own of the same
//! size (review B2). After the exit hook ran no pool starts again, so no
//! Python runs during finalization (review M2).

pub(crate) mod interrupt;
pub(crate) mod pool;
pub(crate) mod runtime;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use krishiv_common::error::{InternalError, KrishivError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use pool::Pool;

/// The process's pool, once started.
static POOL: Mutex<Option<Arc<Pool>>> = Mutex::new(None);

/// Set by the exit hook: no pool starts after it.
static SHUT: AtomicBool = AtomicBool::new(false);

/// How long the exit hook waits for the pool's threads.
const STOP_LIMIT: Duration = Duration::from_secs(5);

/// The size of the pool for `threads` (0: chosen as above, from the
/// executor's `executor_threads`; review M5). Raises after the exit hook ran.
pub(crate) fn pool_size(
    py: Python<'_>,
    threads: usize,
    executor_threads: usize,
) -> PyResult<usize> {
    if SHUT.load(Ordering::SeqCst) {
        return Err(PyRuntimeError::new_err(
            "Python UDFs cannot be registered while the interpreter exits",
        ));
    }
    if threads > 0 {
        return Ok(threads);
    }
    Ok(if gil_enabled(py)? {
        1
    } else {
        executor_threads.max(1)
    })
}

/// This process's pool, started with `size` threads if there is none (or
/// only the copy a fork left).
///
/// # Errors
/// After the exit hook ran, or when a thread cannot be spawned.
pub(crate) fn current(size: usize) -> Result<Arc<Pool>, KrishivError> {
    let unavailable = |message: String| -> KrishivError { InternalError { message }.into() };
    let mut slot = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    // Read under the lock the exit hook takes: no pool starts after it.
    if SHUT.load(Ordering::SeqCst) {
        return Err(unavailable(
            "the Python UDF pool has stopped: the interpreter is exiting".to_owned(),
        ));
    }
    if let Some(p) = slot.as_ref()
        && p.is_local()
    {
        return Ok(Arc::clone(p));
    }
    if let Some(inherited) = slot.take() {
        // A fork's copy: its handles name the parent's threads. Never stop
        // or join it; leak it.
        let _leaked = std::mem::ManuallyDrop::new(inherited);
    }
    let started = Arc::new(
        Pool::start(size)
            .map_err(|e| unavailable(format!("cannot start the Python UDF pool: {e}")))?,
    );
    *slot = Some(Arc::clone(&started));
    Ok(started)
}

/// Whether this interpreter runs with the GIL: `sys._is_gil_enabled()`
/// (3.13+); an interpreter without it has the GIL.
fn gil_enabled(py: Python<'_>) -> PyResult<bool> {
    let sys = py.import("sys")?;
    if !sys.hasattr("_is_gil_enabled")? {
        return Ok(true);
    }
    sys.call_method0("_is_gil_enabled")?.extract()
}

/// The `atexit` hook: stops this process's pool (waiting at most 5 s, with
/// this thread detached so the pool threads can release their pins) and
/// returns how many threads were still inside a UDF (left running, their
/// pins leaked, logged by the caller). A fork's copy is left alone. Later
/// calls fail instead of running Python during finalization.
#[pyfunction]
pub(crate) fn _udf_pool_shutdown(py: Python<'_>) -> usize {
    let taken = {
        let mut slot = POOL.lock().unwrap_or_else(PoisonError::into_inner);
        SHUT.store(true, Ordering::SeqCst);
        slot.take()
    };
    match taken {
        Some(pool) if pool.is_local() => py.detach(|| pool.stop(STOP_LIMIT)),
        Some(inherited) => {
            let _leaked = std::mem::ManuallyDrop::new(inherited);
            0
        }
        None => 0,
    }
}

/// `_native.AnonymousPythonUdf`: an unregistered Python UDF (a DataFrame-side
/// `udf(...)`, design P2-T059 unit 2c (a)), its wrapper in the process's
/// table. Its calls ([`Self::call`]) resolve in any session and are never
/// reachable by name from SQL; they show `name(args)`.
#[pyclass(name = "AnonymousPythonUdf", module = "krishiv._native", frozen)]
#[derive(Debug)]
pub(crate) struct PyAnonymousUdf {
    def: krishiv_session::PythonUdfDef,
    version: u32,
}

#[pymethods]
impl PyAnonymousUdf {
    /// Adds `wrapper` (one `pyarrow.Array` per argument in, one of
    /// `return_type`, a Spark DDL type, out) under a new version.
    #[new]
    #[pyo3(signature = (name, eval_type, return_type, deterministic, thread_safe, wrapper))]
    fn new(
        py: Python<'_>,
        name: &str,
        eval_type: i32,
        return_type: &str,
        deterministic: bool,
        thread_safe: bool,
        wrapper: Py<PyAny>,
    ) -> PyResult<Self> {
        if SHUT.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "Python UDFs cannot be created while the interpreter exits",
            ));
        }
        let eval_type = krishiv_plan::PythonEvalType::from_code(eval_type).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported Python UDF eval type {eval_type}"
            ))
        })?;
        let return_type = krishiv_types::parse_data_type(return_type)
            .map_err(|e| crate::errors::to_py(py, &e))?;
        if name.contains('\0') {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a Python UDF name cannot hold a NUL character",
            ));
        }
        let version = runtime::insert_anonymous(wrapper, thread_safe)?;
        Ok(Self {
            def: krishiv_session::PythonUdfDef {
                name: name.to_owned(),
                eval_type,
                return_type,
                deterministic,
                thread_safe,
            },
            version,
        })
    }

    /// The version plans name this UDF by (top bit set).
    #[getter]
    fn version(&self) -> u32 {
        self.version
    }

    /// A call of this UDF on `args`.
    fn call(
        &self,
        py: Python<'_>,
        args: Vec<crate::column::PyColumn>,
    ) -> PyResult<crate::column::PyColumn> {
        krishiv_session::functions::python_udf_call(
            &self.def,
            self.version,
            args.into_iter().map(|c| c.inner),
        )
        .map(crate::column::PyColumn::wrap)
        .map_err(|e| crate::errors::to_py(py, &e))
    }
}
