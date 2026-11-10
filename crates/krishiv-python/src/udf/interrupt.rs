// SPDX-License-Identifier: Apache-2.0
//! Cancelling a Python UDF call that is running (design P2-T059 §2.5
//! "Cancel", T098b I2): when the engine drops a call's future (its query was
//! cancelled or failed), the call's Python code gets an asynchronous
//! `KeyboardInterrupt` (`PyThreadState_SetAsyncExc`), raised at its next
//! bytecode. A call inside one C function (a sleep, a pyarrow kernel) is
//! interrupted only when it returns to Python; one that never returns keeps
//! its pool thread, which the exit hook then leaks and logs.
//!
//! The engine's compute workers never attach to Python: the interrupt is
//! delivered by one interrupter thread of the process, started on first use.
//! A call's [`Call`] state is the one place the pool thread and the
//! interrupter meet. The interrupter raises only while the state says
//! `Running` on that thread, and the pool thread, once the UDF returns, marks
//! the call done under the same lock and then takes an interrupt that came
//! too late to be raised ([`drain`]), so an interrupt never reaches the
//! thread's next call. A call inside an import is interrupted only once it is out of
//! importlib ([`importing`]): raised there, the interrupt can leave the
//! process's import lock held.
//!
//! The exit hook sets `SHUT` and then waits ([`quiesce`]) until the
//! interrupter is between calls, so the interrupter never attaches once
//! finalization may have begun (review 2026-11-09 I2). On a build with a GIL,
//! a UDF stuck in C while holding it keeps the interrupter waiting to attach,
//! so later interrupts are delayed behind it, not lost.

use std::ffi::{c_long, c_ulong};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::sync::{Arc, LazyLock, Mutex, PoisonError, TryLockError};
use std::time::{Duration, Instant};

use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

/// Where one call is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Queued, or waiting for its `thread_safe=false` lock.
    Waiting,
    /// Its Python code runs on the thread with this identifier.
    Running(c_ulong),
    /// An interrupt was set on that thread while it ran.
    Interrupted(c_ulong),
    /// Its Python code returned, or it was dropped before it started.
    Done,
}

/// One call's state, shared by the pool thread running it and the future
/// awaiting it.
#[derive(Debug)]
pub(crate) struct Call(Mutex<State>);

impl Call {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(State::Waiting)))
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// On the pool thread, attached, just before the UDF's Python code runs:
    /// `false` when the call was dropped meanwhile (skip it).
    pub(crate) fn start(&self, _py: Python<'_>) -> bool {
        let mut s = self.state();
        if *s == State::Done {
            return false;
        }
        // SAFETY: no precondition; the identifier of the calling thread.
        *s = State::Running(unsafe { ffi::PyThread_get_thread_ident() });
        true
    }

    /// On the pool thread, attached, as soon as the UDF's Python code
    /// returned: an interrupt set too late to be raised is taken here
    /// ([`drain`]), so it cannot reach the thread's next call.
    pub(crate) fn finish(&self, py: Python<'_>) {
        let interrupted = {
            let mut s = self.state();
            let was = matches!(*s, State::Interrupted(_));
            // `Done` first: no interrupt is set from here on.
            *s = State::Done;
            was
        };
        if interrupted {
            drain(py);
        }
    }

    /// The call's future was dropped: a waiting call is skipped, a running
    /// one interrupted (by the interrupter thread).
    pub(crate) fn cancel(self: &Arc<Self>) {
        let mut s = self.state();
        match *s {
            State::Waiting => *s = State::Done,
            State::Running(_) => {
                drop(s);
                if let Some(tx) = INTERRUPTER.as_ref() {
                    // A send fails only once the thread is gone (it never
                    // ends): nothing more can be done for this call.
                    let _ = lock(tx).send(Arc::clone(self));
                }
            }
            State::Interrupted(_) | State::Done => {}
        }
    }

    /// The interrupter's step, attached: raises in the thread running the
    /// call, if it still runs it. `true` when the call is inside an import
    /// and must be tried again later ([`importing`]).
    fn interrupt(&self, py: Python<'_>) -> bool {
        let Some(id) = self.running() else {
            return false;
        };
        // Checked before the state's lock is taken: reading the frames calls
        // into Python, and the pool thread takes that lock attached
        // (`finish`). No bytecode runs between the check and the raise, so
        // on a build with a GIL the thread cannot move meanwhile.
        if importing(py, id) {
            return true;
        }
        let mut s = self.state();
        if *s == State::Running(id) {
            // SAFETY: attached (`py`); `PyExc_KeyboardInterrupt` is a
            // built-in exception type, valid while the interpreter runs.
            // The state is `Running(id)` under the lock that `finish` takes,
            // so thread `id` is inside this call's Python code.
            unsafe { ffi::PyThreadState_SetAsyncExc(ident(id), ffi::PyExc_KeyboardInterrupt) };
            *s = State::Interrupted(id);
        }
        false
    }

    /// The thread running the call, if it runs it now.
    fn running(&self) -> Option<c_ulong> {
        match *self.state() {
            State::Running(id) => Some(id),
            _ => None,
        }
    }
}

/// Raises, and drops, an interrupt still pending on this thread (one that
/// came after the UDF's last check): a Python function checks for one on
/// entry, so calling an empty one raises it here instead of in the thread's
/// next call. With none pending it does nothing.
///
/// Not `PyThreadState_SetAsyncExc(id, NULL)`: on CPython 3.11 that raises the
/// interpreter's signal for an asynchronous exception even when it clears
/// one, and only a thread that takes an exception lowers it again. With a
/// trace function set (coverage, a debugger), a thread at a function's entry
/// then handles the signal and runs the same entry again, forever (seen as a
/// main thread spinning on the first line of `SparkSession.sql`).
fn drain(py: Python<'_>) {
    static NOOP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let noop = NOOP.get_or_try_init(py, || {
        py.eval(c"lambda: None", None, None).map(Bound::unbind)
    });
    if let Ok(noop) = noop {
        // Its error is the interrupt being dropped.
        let _ = noop.call0(py);
    }
}

/// Whether thread `id` is inside importlib's bootstrap (a frame of it
/// anywhere on the thread's stack, so also while a module's body runs).
///
/// An asynchronous exception raised there can land between
/// `_imp.acquire_lock()` and the `try` that releases it (CPython 3.11 checks
/// for one after every call): the process's import lock then stays held by a
/// thread that left, and every later import, on any thread, waits for it
/// forever. Outside importlib the exception is raised at the thread's next
/// check, at the latest when it enters a Python function, so an import begun
/// after this check fails at its first frame, before it takes a lock. On a
/// free-threaded build the thread runs meanwhile, which leaves a window of
/// the few instructions between this read and the raise.
///
/// Frames that cannot be read count as not importing (the interrupt is raised
/// as before).
fn importing(py: Python<'_>, id: c_ulong) -> bool {
    /// Frames walked at most: deeper stacks are taken as not importing.
    const MAX_DEPTH: usize = 2048;
    let frames = || -> PyResult<bool> {
        let sys = py.import(intern!(py, "sys"))?;
        let all = sys.call_method0(intern!(py, "_current_frames"))?;
        let Ok(mut frame) = all.get_item(id) else {
            return Ok(false);
        };
        for _ in 0..MAX_DEPTH {
            if frame.is_none() {
                break;
            }
            let file = frame
                .getattr(intern!(py, "f_code"))?
                .getattr(intern!(py, "co_filename"))?;
            if file.extract::<&str>()?.starts_with("<frozen importlib") {
                return Ok(true);
            }
            frame = frame.getattr(intern!(py, "f_back"))?;
        }
        Ok(false)
    };
    frames().unwrap_or(false)
}

/// The identifier as `PyThreadState_SetAsyncExc` takes it: CPython declares
/// it `unsigned long` since 3.7, the binding `long`; the bits are the same.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the same bits: the C function's parameter is unsigned long"
)]
fn ident(id: c_ulong) -> c_long {
    id as c_long
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Held by the interrupter while it checks `SHUT` and is attached.
static BUSY: Mutex<()> = Mutex::new(());

/// The process the interrupter thread was started in (0: none yet): a
/// forked child has the parent's queue but not its thread.
static STARTED_IN: AtomicU32 = AtomicU32::new(0);

/// Waits, at most `limit`, until the interrupter is between calls, after
/// the exit hook set `SHUT`: from then on it never attaches. Called
/// detached (it may wait for the interrupter to attach and finish). `false`
/// when it was still busy at the deadline (attached to a stuck interpreter,
/// as a stuck pool thread is; left to the process's exit). Returns at once
/// when no interrupter runs in this process.
pub(crate) fn quiesce(limit: Duration) -> bool {
    if STARTED_IN.load(Ordering::SeqCst) != std::process::id() {
        return true;
    }
    let deadline = Instant::now() + limit;
    loop {
        match BUSY.try_lock() {
            Ok(_) | Err(TryLockError::Poisoned(_)) => return true,
            Err(TryLockError::WouldBlock) if Instant::now() >= deadline => return false,
            Err(TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
}

/// The first wait before a call inside an import is tried again.
const RETRY_MIN: Duration = Duration::from_millis(1);
/// The longest wait between tries of a call that stays inside an import.
const RETRY_MAX: Duration = Duration::from_millis(50);

/// The interrupter thread's queue; `None` if it could not be spawned (a
/// cancelled call then runs to its end, as before this existed).
static INTERRUPTER: LazyLock<Option<Mutex<Sender<Arc<Call>>>>> = LazyLock::new(|| {
    let (tx, rx) = channel::<Arc<Call>>();
    std::thread::Builder::new()
        .name("krishiv-python-udf-interrupt".to_owned())
        .spawn(move || {
            // Calls inside an import, tried again after `wait` (doubled while
            // they stay there, up to `RETRY_MAX`).
            let mut deferred: Vec<Arc<Call>> = Vec::new();
            let mut wait = RETRY_MIN;
            loop {
                let next = if deferred.is_empty() {
                    rx.recv().map_err(|_| RecvTimeoutError::Disconnected)
                } else {
                    rx.recv_timeout(wait)
                };
                let mut calls = std::mem::take(&mut deferred);
                match next {
                    Ok(call) => calls.push(call),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
                calls.extend(rx.try_iter());
                // Never attach once the exit hook ran: the interpreter may be
                // finalizing. A call still running then is the hook's to leak.
                // The check and the attach are one step under `BUSY`, which
                // the hook waits for after setting `SHUT` ([`quiesce`]).
                let _busy = lock(&BUSY);
                if super::SHUT.load(Ordering::SeqCst) {
                    continue;
                }
                Python::attach(|py| {
                    deferred = calls.into_iter().filter(|c| c.interrupt(py)).collect();
                });
                wait = if deferred.is_empty() {
                    RETRY_MIN
                } else {
                    (wait * 2).min(RETRY_MAX)
                };
            }
        })
        .ok()
        .map(|_| {
            STARTED_IN.store(std::process::id(), Ordering::SeqCst);
            Mutex::new(tx)
        })
});

/// Cancels its call when dropped: the guard a call's future holds, so a
/// dropped future (a cancelled or failed query) interrupts the call. A
/// finished call ignores it.
pub(crate) struct CancelOnDrop(pub(crate) Arc<Call>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
