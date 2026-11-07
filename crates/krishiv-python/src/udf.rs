// SPDX-License-Identifier: Apache-2.0
//! Embedded Python UDFs (design P2-T059 §2.5, unit 2b): the process's pool
//! of pinned threads, the per-session [`runtime::PyUdfRuntime`], and the
//! hooks `krishiv.sql` registers at import (`atexit`, `os.register_at_fork`).
//!
//! One pool per process, started by the first call that needs it and sized
//! then (`krishiv.python.udf.threads`; 0 is the machine's thread count on a
//! free-threaded Python, else 1). After `os.fork` the child marks the pool it
//! inherited as gone and starts its own on first use: the inherited threads
//! and pins do not exist in the child and are never touched.

pub(crate) mod pool;
pub(crate) mod runtime;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use pool::Pool;

/// The process's pool, once started.
static POOL: Mutex<Option<Arc<Pool>>> = Mutex::new(None);

/// Set in a forked child: the pool in [`POOL`] is the parent's.
static FORKED: AtomicBool = AtomicBool::new(false);

/// How long the exit hook waits for the pool's threads.
const STOP_LIMIT: Duration = Duration::from_secs(5);

/// The pool, started with `threads` threads (0: chosen as above) if there
/// is none yet.
pub(crate) fn pool(py: Python<'_>, threads: usize) -> PyResult<Arc<Pool>> {
    let mut slot = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    if FORKED.swap(false, Ordering::SeqCst)
        && let Some(inherited) = slot.take()
    {
        // The parent's threads do not exist here: never stop or join them.
        let _leaked = std::mem::ManuallyDrop::new(inherited);
    }
    if let Some(p) = slot.as_ref() {
        return Ok(Arc::clone(p));
    }
    let size = if threads > 0 {
        threads
    } else if gil_enabled(py)? {
        1
    } else {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    };
    let started =
        Arc::new(Pool::start(size).map_err(|e| {
            PyRuntimeError::new_err(format!("cannot start the Python UDF pool: {e}"))
        })?);
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

/// The `atexit` hook: stops the pool (waiting at most 5 s, with this thread
/// detached so the pool threads can release their pins) and returns how
/// many threads were still inside a UDF (left running, logged by the
/// caller).
#[pyfunction]
pub(crate) fn _udf_pool_shutdown(py: Python<'_>) -> usize {
    let taken = POOL.lock().unwrap_or_else(PoisonError::into_inner).take();
    let Some(pool) = taken else { return 0 };
    // Sessions alive at exit still hold the pool; stopping it makes their
    // later calls fail instead of running Python during finalization.
    py.detach(|| pool.stop(STOP_LIMIT))
}

/// The `os.register_at_fork(after_in_child=...)` hook.
#[pyfunction]
pub(crate) fn _udf_pool_forked() {
    FORKED.store(true, Ordering::SeqCst);
}
