// SPDX-License-Identifier: Apache-2.0
//! The threads Python UDF calls run on (design P2-T059 §2.5, unit 2b).
//!
//! The engine's compute workers never run Python: a call is a job sent to
//! this pool. Each pool thread pins a Python thread state once, when it
//! starts (`PyGILState_Ensure`, then `PyEval_SaveThread`), so every job is a
//! cheap re-attach of that state (`Python::attach`). The pin is released on
//! the same thread only by [`Pool::stop`], which the module's `atexit` hook
//! calls before the interpreter starts finalizing; a pool still running at
//! finalization (the hook did not run, or a thread did not stop in time)
//! keeps its pins, leaked, never released after finalization began (under
//! abi3-py311 `Py_IsFinalizing` is not available, so the pin is never
//! released anywhere else).
//!
//! A panic inside a job is caught at the job boundary: it never unwinds into
//! Python or out of the thread, and the job's result channel is dropped, so
//! the caller sees the call fail.

use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pyo3::ffi;
use pyo3::prelude::*;

/// One unit of work, run attached to Python on a pool thread.
pub(crate) type Job = Box<dyn FnOnce(Python<'_>) + Send>;

enum Message {
    Run(Job),
    Stop,
}

/// A pool thread's Python thread state, kept from the thread's start to its
/// stop. Not `Send`: it is released on the thread that took it.
struct Pin {
    gil: ffi::PyGILState_STATE,
    state: *mut ffi::PyThreadState,
    _not_send: PhantomData<*mut ()>,
}

impl Pin {
    /// Pins the calling thread, which must not hold a Python thread state.
    fn new() -> Self {
        // SAFETY: called once, at the start of a pool thread that has never
        // been attached to Python, while the interpreter runs (the pool is
        // started from a Python call, before `atexit`); it creates and
        // attaches this thread's state.
        let gil = unsafe { ffi::PyGILState_Ensure() };
        // SAFETY: this thread holds the state `PyGILState_Ensure` just gave
        // it; saving it detaches the thread (releasing the GIL) while keeping
        // the state, which later `Python::attach` calls on this thread reuse.
        let state = unsafe { ffi::PyEval_SaveThread() };
        Self {
            gil,
            state,
            _not_send: PhantomData,
        }
    }

    /// Releases the pin. Only from [`Pool::stop`]'s path, on the thread
    /// that pinned, before the interpreter finalizes.
    fn release(self) {
        // SAFETY: `state` is this thread's own saved state (taken in `new`
        // on this thread, which is detached now); restoring re-attaches it.
        unsafe { ffi::PyEval_RestoreThread(self.state) };
        // SAFETY: pairs the `PyGILState_Ensure` of `new` on the same thread,
        // with the thread attached (just restored).
        unsafe { ffi::PyGILState_Release(self.gil) };
    }
}

/// A fixed set of pinned threads taking jobs from one queue.
pub(crate) struct Pool {
    jobs: Sender<Message>,
    /// Taken by [`Pool::stop`].
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Pool {
    /// Starts `size` pinned threads (at least one). Called with the caller
    /// attached to Python; the new threads wait for the caller to detach
    /// before they can pin.
    ///
    /// # Errors
    /// A thread that cannot be spawned (the pool started so far is stopped).
    pub(crate) fn start(size: usize) -> std::io::Result<Self> {
        let (jobs, queue) = channel::<Message>();
        let queue = Arc::new(Mutex::new(queue));
        let mut threads = Vec::with_capacity(size.max(1));
        for i in 0..size.max(1) {
            let queue = Arc::clone(&queue);
            let spawned = std::thread::Builder::new()
                .name(format!("krishiv-python-udf-{i}"))
                .spawn(move || worker(&queue));
            match spawned {
                Ok(t) => threads.push(t),
                Err(e) => {
                    // The threads started wait for jobs; tell them to end.
                    for _ in &threads {
                        let _ = jobs.send(Message::Stop);
                    }
                    return Err(e);
                }
            }
        }
        Ok(Self {
            jobs,
            threads: Mutex::new(threads),
        })
    }

    /// Queues `job`. A pool that has stopped drops it (its result channel
    /// with it, so the caller sees the call fail): its threads have ended,
    /// so the queue has no receiver left.
    pub(crate) fn submit(&self, job: Job) {
        let _ = self.jobs.send(Message::Run(job));
    }

    /// Stops every thread, which releases its pin; waits at most `limit`
    /// for them. Called detached from Python (the threads must attach to
    /// release their pins). The number of threads that did not stop in
    /// time (still inside a UDF): they are left running, their pins leaked.
    pub(crate) fn stop(&self, limit: Duration) -> usize {
        let threads =
            std::mem::take(&mut *self.threads.lock().unwrap_or_else(PoisonError::into_inner));
        for _ in &threads {
            let _ = self.jobs.send(Message::Stop);
        }
        let deadline = Instant::now() + limit;
        let mut stuck = 0;
        for t in threads {
            while !t.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if t.is_finished() {
                let _ = t.join();
            } else {
                stuck += 1;
            }
        }
        stuck
    }
}

/// A pool thread: pins, runs jobs until told to stop, releases the pin.
fn worker(queue: &Mutex<Receiver<Message>>) {
    let pin = Pin::new();
    loop {
        let next = queue.lock().unwrap_or_else(PoisonError::into_inner).recv();
        match next {
            Ok(Message::Run(job)) => {
                // A panicking job is contained: the thread keeps serving.
                let _ = catch_unwind(AssertUnwindSafe(|| Python::attach(job)));
            }
            Ok(Message::Stop) | Err(_) => break,
        }
    }
    pin.release();
}
