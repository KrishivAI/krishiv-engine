// SPDX-License-Identifier: Apache-2.0
//! The interpreters' startup inspection, run on a thread of its own so that a server listens before it ends
//! (an interpreter with `pyspark`, `pyarrow` and `pandas` takes about two seconds to import them). The first
//! worker start, and every caller of [`crate::Pool::interpreters`] or [`crate::Pool::refusal`], waits for it.
//! The pool's drop stops it and waits for its thread, so no inspection process outlives the pool (its
//! timeout is kept by that thread, which ends with the process).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use krishiv_common::error::{InternalError, KrishivError};

use crate::interpreter::Interpreter;

/// What the inspection found: the usable interpreters and, when the pool cannot run workers at all, why.
#[derive(Debug)]
pub(crate) struct Inspected {
    /// The usable interpreters, in the listed order.
    pub(crate) interpreters: Vec<Interpreter>,
    /// Why every call is refused: no usable interpreter, else the sandbox the kernel cannot enforce.
    pub(crate) refusal: Option<KrishivError>,
}

/// The inspection of one pool: set once by its thread, read by everyone after.
#[derive(Debug, Default)]
pub(crate) struct Inspection {
    done: OnceLock<Inspected>,
    lock: Mutex<()>,
    ready: Condvar,
    /// Set by [`Inspection::stop`]: the running inspection kills its process and the rest are skipped.
    stop: Arc<AtomicBool>,
    /// The inspection's thread, joined by [`Inspection::stop`].
    thread: Mutex<Option<JoinHandle<()>>>,
}

/// How often a cancellable wait looks at its cancel flag.
const POLL: Duration = Duration::from_millis(100);

impl Inspection {
    /// Starts inspecting `executables` on a thread of its own. `sandbox_refusal` is the pool's refusal when
    /// the interpreters are fine: no usable interpreter is the first reason given.
    ///
    /// # Errors
    /// When the thread cannot be started.
    pub(crate) fn start(
        executables: Vec<String>,
        sandbox_refusal: Option<KrishivError>,
    ) -> Result<Arc<Self>, KrishivError> {
        Self::start_with(sandbox_refusal, move |stop| {
            executables
                .iter()
                .map(|exe| Interpreter::inspect_until(exe, stop))
                .collect()
        })
    }

    /// [`Inspection::start`] with the inspection itself given (`inspect` returns one result per listed
    /// interpreter, in order, and ends early once its flag is set; tests give a slow or a failing one).
    pub(crate) fn start_with(
        sandbox_refusal: Option<KrishivError>,
        inspect: impl FnOnce(&AtomicBool) -> Vec<Result<Interpreter, String>> + Send + 'static,
    ) -> Result<Arc<Self>, KrishivError> {
        let this = Arc::new(Self::default());
        let mine = Arc::clone(&this);
        let stop = Arc::clone(&this.stop);
        let thread = std::thread::Builder::new()
            .name("krishiv-udf-inspection".to_owned())
            .spawn(move || {
                // Set even if the inspection panics, so no waiter waits for ever.
                let _unfinished = Unfinished(&mine);
                mine.finish(conclude(inspect(&stop), sandbox_refusal));
            })
            .map_err(|e| {
                internal(format!(
                    "cannot start the Python interpreter inspection: {e}"
                ))
            })?;
        *this.thread.lock().unwrap_or_else(PoisonError::into_inner) = Some(thread);
        Ok(this)
    }

    /// Stops the inspection, if it still runs, and waits for its thread: its process group is killed within
    /// about 10 ms. Waiters then get a refusal (every interpreter failed as stopped).
    pub(crate) fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let thread = self
            .thread
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(thread) = thread {
            // A panic in it already set the result (`Unfinished`); nothing more to do with it.
            let _ = thread.join();
        }
    }

    /// The result, if the inspection has ended.
    pub(crate) fn get(&self) -> Option<&Inspected> {
        self.done.get()
    }

    /// The result, waiting for the inspection to end.
    pub(crate) fn wait(&self) -> &Inspected {
        // `wait_until` returns `None` only when cancelled, which this never is: one pass. The loop avoids an
        // `expect` in library code.
        loop {
            if let Some(found) = self.wait_until(&|| false) {
                return found;
            }
        }
    }

    /// The result, waiting for the inspection to end; `None` once `cancelled` says so (looked at every
    /// 100 ms).
    pub(crate) fn wait_until(&self, cancelled: &dyn Fn() -> bool) -> Option<&Inspected> {
        let mut held = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(found) = self.done.get() {
                return Some(found);
            }
            if cancelled() {
                return None;
            }
            held = self
                .ready
                .wait_timeout(held, POLL)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    fn finish(&self, found: Inspected) {
        let _held = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        // Only the inspection's thread sets it, once.
        let _ = self.done.set(found);
        self.ready.notify_all();
    }
}

/// Ends the inspection with an internal error when its thread unwinds before it gave a result; does nothing
/// once a result is set.
struct Unfinished<'a>(&'a Inspection);

impl Drop for Unfinished<'_> {
    fn drop(&mut self) {
        if self.0.done.get().is_some() {
            return;
        }
        self.0.finish(Inspected {
            interpreters: Vec::new(),
            refusal: Some(internal(
                "the Python interpreter inspection ended without a result".to_owned(),
            )),
        });
    }
}

/// The usable interpreters of `results` and the pool's refusal; logs each interpreter not used.
fn conclude(
    results: Vec<Result<Interpreter, String>>,
    sandbox_refusal: Option<KrishivError>,
) -> Inspected {
    let mut interpreters = Vec::new();
    let mut rejected = Vec::new();
    for result in results {
        match result {
            Ok(i) => match i.unusable_reason() {
                None => interpreters.push(i),
                Some(why) => rejected.push(why),
            },
            Err(why) => rejected.push(why),
        }
    }
    for why in &rejected {
        tracing::warn!("Python UDF worker interpreter not used: {why}");
    }
    let refusal = interpreters
        .is_empty()
        .then(|| {
            internal(format!(
                "no usable Python interpreter for UDF workers (\"krishiv.python.worker.executables\"): {}",
                if rejected.is_empty() { "none listed".to_owned() } else { rejected.join("; ") }
            ))
        })
        .or(sandbox_refusal);
    if let Some(refusal) = &refusal {
        tracing::warn!("Python UDFs are refused: {refusal}");
    }
    Inspected {
        interpreters,
        refusal,
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;

    use super::*;

    fn sandbox() -> KrishivError {
        internal("sandbox".to_owned())
    }

    #[test]
    fn the_start_returns_before_the_inspection_ends() {
        let (release, hold) = mpsc::channel::<()>();
        let inspection = Inspection::start_with(None, move |_| {
            let _ = hold.recv();
            vec![Err("slow".to_owned())]
        })
        .expect("start");
        assert!(
            inspection.get().is_none(),
            "the inspection waited for nothing"
        );
        let stop = AtomicBool::new(false);
        stop.store(true, Ordering::SeqCst);
        assert!(
            inspection
                .wait_until(&|| stop.load(Ordering::SeqCst))
                .is_none(),
            "a cancelled wait still waited"
        );
        release.send(()).expect("release");
        let found = inspection.wait();
        assert!(found.interpreters.is_empty());
        let why = found.refusal.as_ref().expect("refused").to_string();
        assert!(why.contains("slow"), "{why}");
    }

    #[test]
    fn no_interpreter_is_the_first_reason_then_the_sandbox() {
        let none = Inspection::start_with(Some(sandbox()), |_| Vec::new()).expect("start");
        let why = none.wait().refusal.as_ref().expect("refused").to_string();
        assert!(why.contains("none listed"), "{why}");

        let failed = Inspection::start_with(Some(sandbox()), |_| vec![Err("gone".to_owned())])
            .expect("start");
        let why = failed.wait().refusal.as_ref().expect("refused").to_string();
        assert!(why.contains("gone") && !why.contains("sandbox"), "{why}");
    }

    #[test]
    fn a_panicking_inspection_refuses_rather_than_hangs() {
        let inspection = Inspection::start_with(None, |_| -> Vec<Result<Interpreter, String>> {
            panic!("inspection broke")
        })
        .expect("start");
        let why = inspection
            .wait()
            .refusal
            .as_ref()
            .expect("refused")
            .to_string();
        assert!(why.contains("without a result"), "{why}");
    }
}
