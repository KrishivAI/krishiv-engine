// SPDX-License-Identifier: Apache-2.0
//! The result of a running query as an Arrow reader (pyo3-free, so that
//! `tests.rs` can compile it and check cancel-on-release).

use std::sync::{Arc, Mutex};

use arrow_array::{RecordBatch, RecordBatchOptions, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use futures::StreamExt as _;
use krishiv_common::error::KrishivError;
use krishiv_session::RecordBatchStream;

/// Where a reader leaves the engine error that ended it, so the caller can raise
/// the original (Spark-classed) exception instead of the Arrow one the C stream carries.
pub(crate) type ErrorSlot = Arc<Mutex<Option<KrishivError>>>;

/// The result of a running query as an Arrow reader. It never touches Python,
/// so a consumer may call it with the GIL held or from a foreign thread.
/// Dropping it drops the engine stream, which cancels the query without
/// waiting for it.
pub(crate) struct QueryReader {
    schema: SchemaRef,
    /// The batch read to learn the schema, handed out first.
    first: Option<RecordBatch>,
    stream: RecordBatchStream,
    error: ErrorSlot,
}

impl Iterator for QueryReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.first.take() {
            Some(batch) => batch,
            None => match wait_for(self.stream.next())? {
                Ok(batch) => batch,
                Err(e) => {
                    let arrow = ArrowError::ExternalError(e.to_string().into());
                    if let Ok(mut slot) = self.error.lock() {
                        *slot = Some(e);
                    }
                    return Some(Err(arrow));
                }
            },
        };
        // The schema was fixed by the first batch: a batch that differs is an
        // engine bug, reported instead of exported under the wrong schema.
        // The row count is kept: a batch without columns has no other source for it.
        Some(RecordBatch::try_new_with_options(
            SchemaRef::clone(&self.schema),
            batch.columns().to_vec(),
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        ))
    }
}

impl RecordBatchReader for QueryReader {
    fn schema(&self) -> SchemaRef {
        SchemaRef::clone(&self.schema)
    }
}

impl QueryReader {
    /// A reader of `stream` under `schema`; `first` is a batch already taken
    /// from it, handed out first.
    pub(crate) fn new(
        schema: SchemaRef,
        first: Option<RecordBatch>,
        stream: RecordBatchStream,
    ) -> Self {
        Self {
            schema,
            first,
            stream,
            error: ErrorSlot::default(),
        }
    }

    /// The slot the engine error that ends this reader is left in.
    pub(crate) fn error_slot(&self) -> ErrorSlot {
        ErrorSlot::clone(&self.error)
    }
}

/// Runs `work` to completion on the calling thread, waiting on a
/// `Mutex`/`Condvar` pair between polls.
///
/// Not `futures::executor::block_on`: that parks the thread with
/// `std::thread::park`, which on Apple platforms waits on a libdispatch
/// semaphore created with the thread's handle, a Mach port. Ports are not
/// inherited by `fork`, so the main thread of a forked child (a parent that ran
/// a query, then `os.fork()`) traps (`SIGTRAP`) the first time it parks. The
/// pthread mutex and condition variable behind `Mutex` and `Condvar` live in
/// ordinary memory, and each call makes fresh ones, so a child's first query
/// runs as on Linux.
pub(crate) fn wait_for<F: std::future::Future>(work: F) -> F::Output {
    let mut waiter = Waiter::new();
    let mut work = std::pin::pin!(work);
    loop {
        if let Some(out) = waiter.step(work.as_mut(), None) {
            return out;
        }
    }
}

/// Set by a wake, cleared by the waiting thread before it polls again.
struct Signal {
    woken: std::sync::Mutex<bool>,
    ready: std::sync::Condvar,
}

impl std::task::Wake for Signal {
    fn wake(self: std::sync::Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        *self
            .woken
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        self.ready.notify_one();
    }
}

/// The waker and wake flag one wait uses.
pub(crate) struct Waiter {
    signal: std::sync::Arc<Signal>,
    waker: std::task::Waker,
}

impl Waiter {
    pub(crate) fn new() -> Self {
        let signal = std::sync::Arc::new(Signal {
            woken: std::sync::Mutex::new(false),
            ready: std::sync::Condvar::new(),
        });
        let waker = std::task::Waker::from(std::sync::Arc::clone(&signal));
        Self { signal, waker }
    }

    /// Polls `work` once; while it is pending, waits for a wake (at most `limit`, if given).
    /// `Some` is its output. `None` means it is still pending: the wait ran out (`limit`), or a
    /// wake came and the next call polls again.
    pub(crate) fn step<F: std::future::Future>(
        &mut self,
        work: std::pin::Pin<&mut F>,
        limit: Option<std::time::Duration>,
    ) -> Option<F::Output> {
        use std::sync::PoisonError;
        use std::task::{Context, Poll};

        let mut cx = Context::from_waker(&self.waker);
        if let Poll::Ready(out) = work.poll(&mut cx) {
            return Some(out);
        }
        let mut woken = self
            .signal
            .woken
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let deadline = limit.map(|l| std::time::Instant::now() + l);
        while !*woken {
            match deadline {
                None => {
                    woken = self
                        .signal
                        .ready
                        .wait(woken)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(std::time::Instant::now());
                    if left.is_zero() {
                        return None;
                    }
                    woken = self
                        .signal
                        .ready
                        .wait_timeout(woken, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
            }
        }
        *woken = false;
        None
    }
}
