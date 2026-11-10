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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pyo3::ffi;
use pyo3::intern;
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
        // been attached to Python; it creates and attaches this thread's
        // state. The interpreter is running: a pool starts only before the
        // exit hook (`udf::current` reads `SHUT` under the `POOL` lock the
        // hook takes), and the hook waits, detached, for its threads to pin
        // and stop. Limit (T098b M1): a thread not yet pinned when the hook's
        // 5 s deadline passes can still get here during finalization.
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
    /// The process the threads run in: after `os.fork` the child's copy of
    /// this pool has no threads (review B2).
    pid: u32,
    /// Set by [`Pool::stop`] when a thread did not stop in time: it then
    /// keeps its pin instead of releasing it, because the interpreter may be
    /// finalizing by the time it gets there (review M1).
    abandoned: Arc<AtomicBool>,
}

impl Pool {
    /// Starts `size` pinned threads (at least one). Called attached or not;
    /// a new thread pins once the GIL (if any) is free.
    ///
    /// # Errors
    /// A thread that cannot be spawned (the pool started so far is stopped).
    pub(crate) fn start(size: usize) -> std::io::Result<Self> {
        let (jobs, queue) = channel::<Message>();
        let queue = Arc::new(Mutex::new(queue));
        let abandoned = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::with_capacity(size.max(1));
        for i in 0..size.max(1) {
            let (queue, abandoned) = (Arc::clone(&queue), Arc::clone(&abandoned));
            let spawned = std::thread::Builder::new()
                .name(format!("krishiv-python-udf-{i}"))
                .spawn(move || worker(&queue, &abandoned));
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
            pid: std::process::id(),
            abandoned,
        })
    }

    /// Whether this pool's threads exist in the calling process (false in a
    /// forked child).
    pub(crate) fn is_local(&self) -> bool {
        self.pid == std::process::id()
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
                // Tell it to keep its pin; set before the next check so a
                // thread finishing now still sees it.
                self.abandoned.store(true, Ordering::SeqCst);
                if t.is_finished() {
                    let _ = t.join();
                } else {
                    stuck += 1;
                }
            }
        }
        stuck
    }
}

/// What a `threading.Thread` does as it starts (`Thread._bootstrap_inner`):
/// the trace and profile functions set with `threading.settrace` and
/// `threading.setprofile` (coverage, a debugger, a profiler) are installed on
/// this thread too, so UDF code is traced as code on any Python thread is.
/// The pinned thread state keeps them for every later call. Nothing is set
/// when none is; a failure leaves the thread untraced.
fn inherit_thread_hooks(py: Python<'_>) {
    let install = || -> PyResult<()> {
        let threading = py.import(intern!(py, "threading"))?;
        let sys = py.import(intern!(py, "sys"))?;
        for (get, set) in [("gettrace", "settrace"), ("getprofile", "setprofile")] {
            let hook = threading.call_method0(get)?;
            if !hook.is_none() {
                sys.call_method1(set, (hook,))?;
            }
        }
        Ok(())
    };
    // A failure is the thread left untraced: never a failed pool.
    let _ = install();
}

/// A pool thread: pins, runs jobs until told to stop, releases the pin.
fn worker(queue: &Mutex<Receiver<Message>>, abandoned: &AtomicBool) {
    let pin = Pin::new();
    Python::attach(inherit_thread_hooks);
    loop {
        let next = queue.lock().unwrap_or_else(PoisonError::into_inner).recv();
        match next {
            // Past the exit deadline no job starts: the interpreter may be
            // finalizing (T098b I2); dropping it fails its call.
            Ok(Message::Run(_)) if abandoned.load(Ordering::SeqCst) => break,
            Ok(Message::Run(job)) => {
                // A panicking job is contained: the thread keeps serving.
                let _ = catch_unwind(AssertUnwindSafe(|| Python::attach(job)));
            }
            Ok(Message::Stop) | Err(_) => break,
        }
    }
    if abandoned.load(Ordering::SeqCst) {
        // Past the exit deadline the interpreter may be finalizing: keep the
        // thread state (leaked) rather than touch it. A thread that read the
        // flag just before it was set can still release after finalization
        // began: the window is the release's two calls (review M1, narrowed).
        let _kept = std::mem::ManuallyDrop::new(pin);
    } else {
        pin.release();
    }
}
