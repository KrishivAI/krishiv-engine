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
//! `Running` on that thread, and the pool thread, once the UDF returns,
//! clears an interrupt that came too late to be raised. Both steps happen
//! under the state's lock, so an interrupt never reaches the thread's next
//! call.
//!
//! The exit hook sets `SHUT` and then waits ([`quiesce`]) until the
//! interrupter is between calls, so the interrupter never attaches once
//! finalization may have begun (review 2026-11-09 I2). On a build with a GIL,
//! a UDF stuck in C while holding it keeps the interrupter waiting to attach,
//! so later interrupts are delayed behind it, not lost.

use std::ffi::{c_long, c_ulong};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, LazyLock, Mutex, PoisonError, TryLockError};
use std::time::{Duration, Instant};

use pyo3::ffi;
use pyo3::prelude::*;

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
    /// returned: an interrupt set too late to be raised is cleared, so it
    /// cannot reach the thread's next call.
    pub(crate) fn finish(&self, _py: Python<'_>) {
        let mut s = self.state();
        if let State::Interrupted(id) = *s {
            // SAFETY: attached (`_py`); a null exception clears the pending
            // asynchronous exception of thread `id`, this thread.
            unsafe { ffi::PyThreadState_SetAsyncExc(ident(id), std::ptr::null_mut()) };
        }
        *s = State::Done;
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
    /// call, if it still runs it.
    fn interrupt(&self, _py: Python<'_>) {
        let mut s = self.state();
        if let State::Running(id) = *s {
            // SAFETY: attached (`_py`); `PyExc_KeyboardInterrupt` is a
            // built-in exception type, valid while the interpreter runs.
            // The state is `Running(id)` under the lock that `finish` takes,
            // so thread `id` is inside this call's Python code.
            unsafe { ffi::PyThreadState_SetAsyncExc(ident(id), ffi::PyExc_KeyboardInterrupt) };
            *s = State::Interrupted(id);
        }
    }
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

/// The interrupter thread's queue; `None` if it could not be spawned (a
/// cancelled call then runs to its end, as before this existed).
static INTERRUPTER: LazyLock<Option<Mutex<Sender<Arc<Call>>>>> = LazyLock::new(|| {
    let (tx, rx) = channel::<Arc<Call>>();
    std::thread::Builder::new()
        .name("krishiv-python-udf-interrupt".to_owned())
        .spawn(move || {
            for call in rx {
                // Never attach once the exit hook ran: the interpreter may be
                // finalizing. A call still running then is the hook's to leak.
                // The check and the attach are one step under `BUSY`, which
                // the hook waits for after setting `SHUT` ([`quiesce`]).
                let _busy = lock(&BUSY);
                if super::SHUT.load(Ordering::SeqCst) {
                    continue;
                }
                Python::attach(|py| call.interrupt(py));
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
