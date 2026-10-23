// SPDX-License-Identifier: Apache-2.0
//! A breaker's finished output held in memory, spillable while no driver
//! owns it.
//!
//! Between a breaker's `finalize` and the end of its source side, its
//! in-memory result (an aggregate's final tables, a sort's buffered
//! batches) belongs to no running driver: the finalizing driver has ended,
//! and the reading driver may not have started, or may be one batch into a
//! long read. Charged to the finalizing driver's consumer, whose spill hook
//! went with it, the result could not be spilled, so whether another
//! request in the same pool was granted or failed depended on how far the
//! reader had got: a scheduling accident, not a property of the query.
//!
//! [`OutputSpill`] gives such an output a consumer of its own. Its spill
//! hook runs on a task of its own: it takes the part no reader has claimed
//! yet, writes it to a spill file on the blocking pool and frees its
//! memory; readers continue from the file, in the same order. This is the
//! spillable-iterator pattern of Spark's `UnsafeExternalSorter`
//! (`SpillableIterator.spill`) and `BytesToBytesMap` (`MapIterator.spill`):
//! the records not yet returned move to disk and the iterator switches to
//! reading them back.
//!
//! The output is immutable data owned by the breaker's global state, so
//! the hook never touches a running driver's operator state; a reader that
//! holds part of it (a claimed chunk, a merge step in progress) keeps that
//! part until it is done with it. The state stays partition-local: one
//! output per breaker instance, on the worker that finalized it.

use std::sync::{Arc, OnceLock, Weak};

use futures::future::BoxFuture;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_memory::{ConsumerHandle, OwnerId, SpillSignal, SpillTicket, Spillable};

use crate::context::{QueryContext, TaskContext};
use crate::error::Result;
use crate::executor::Executor;
use crate::sync::{Mutex, lock};

/// The spill side of a finished output.
pub(crate) trait SpillUnread: Send + Sync + 'static {
    /// Moves the unread in-memory part to disk and frees its memory;
    /// returns the bytes freed (0 if nothing was left in memory). Runs on
    /// the hook's own task; blocking I/O goes to the blocking pool.
    fn spill_unread(self: Arc<Self>) -> BoxFuture<'static, Result<usize>>;
}

/// A spill's outcome when its write failed: a refused scratch reservation
/// frees nothing and is not an error (the arbitrator moves on); any other
/// error is.
pub(crate) fn refused(e: KrishivError) -> Result<usize> {
    if e.error_class() == ErrorClass::UnableToAcquireMemory {
        Ok(0)
    } else {
        Err(e)
    }
}

/// The memory consumer of a finished output and its spill hook (see the
/// [module docs](self)).
pub(crate) struct OutputSpill {
    me: Weak<Self>,
    signal: SpillSignal,
    output: OnceLock<Weak<dyn SpillUnread>>,
    /// A spill task is running: requests join it. A lock, not an atomic:
    /// the task's last check for a pending request and a request's check
    /// of this flag are ordered by it, so no request is left unserved.
    running: Mutex<bool>,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
}

impl OutputSpill {
    /// A hook and the consumer it covers, registered for `ctx`'s operator
    /// (its spill files count with the operator's). Attach the output with
    /// [`attach`](Self::attach) and move its reservations to the consumer.
    pub(crate) fn register(ctx: &TaskContext, name: &str) -> (Arc<Self>, Arc<ConsumerHandle>) {
        let hook = Arc::new_cyclic(|me| Self {
            me: Weak::clone(me),
            signal: SpillSignal::new(),
            output: OnceLock::new(),
            running: Mutex::new(false),
            query: Arc::clone(ctx.query()),
            executor: Arc::clone(ctx.executor()),
        });
        let consumer = ctx.register_consumer(name, Arc::clone(&hook) as Arc<dyn Spillable>);
        (hook, consumer)
    }

    /// The output this hook spills (held weakly: the output owns the hook).
    pub(crate) fn attach(&self, output: Weak<dyn SpillUnread>) {
        let _ = self.output.set(output);
    }

    /// Publishes the bytes a spill would free.
    pub(crate) fn set_spillable_bytes(&self, bytes: usize) {
        self.signal.set_spillable_bytes(bytes);
    }

    /// Starts the spill task unless one is running.
    fn start(&self) {
        let mut running = lock(&self.running);
        if *running {
            return;
        }
        // Always alive here: `self` is borrowed from the `Arc`.
        if let Some(me) = self.me.upgrade() {
            *running = true;
            drop(
                self.executor
                    .spawn(self.query.class(), Box::pin(me.serve())),
            );
        }
    }

    /// The spill task: serves requests until none is pending. It counts as
    /// the query's blocking work for its whole run, so the query does not
    /// end while it holds the output, its memory or a file.
    async fn serve(self: Arc<Self>) {
        let _in_flight = self.query.blocking_started();
        loop {
            self.service().await;
            let mut running = lock(&self.running);
            // A request that arrived after the last one completed starts a
            // new generation: serve it too.
            if self.signal.spill_requested().is_none() {
                *running = false;
                return;
            }
        }
    }

    /// Serves the pending request, if any: spills the output's unread part
    /// and completes every ticket of the request with the bytes freed.
    async fn service(&self) {
        let Some(service) = self.signal.begin() else {
            return;
        };
        let output = self.output.get().and_then(Weak::upgrade);
        let freed = match output {
            // A cancelled query stops waiting (for a reader's step that
            // will not end): the output goes with the query.
            Some(o) => tokio::select! {
                spilled = o.spill_unread() => spilled.unwrap_or_else(|e| {
                    // A failed spill write fails the query, as it does on
                    // a driver.
                    self.query.fail(e);
                    0
                }),
                () = self.query.cancelled() => 0,
            },
            None => 0,
        };
        service.complete(freed);
    }
}

impl Spillable for OutputSpill {
    fn spillable_bytes(&self) -> usize {
        self.signal.spillable_bytes()
    }

    fn request_spill(&self, target: usize) -> SpillTicket {
        let ticket = self.signal.request_spill(target);
        if target > 0 {
            self.start();
        }
        ticket
    }

    fn spill_requested(&self) -> Option<usize> {
        self.signal.spill_requested()
    }

    fn owner(&self) -> Option<OwnerId> {
        None
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use krishiv_memory::{MemoryConfig, MemoryError, MemoryPool, SpillOwner, TreePool};

    use super::*;
    use crate::context::{ExecConfig, OperatorId, QueryId};
    use crate::error::internal;
    use crate::executor::{TokioExecutor, WorkloadClass};

    /// An output whose spill frees `bytes`, fails, or waits until `gate`
    /// opens; counts its spills.
    struct Fake {
        spills: AtomicUsize,
        result: Result<usize>,
        gate: Option<tokio::sync::Notify>,
    }

    impl SpillUnread for Fake {
        fn spill_unread(self: Arc<Self>) -> BoxFuture<'static, Result<usize>> {
            Box::pin(async move {
                self.spills.fetch_add(1, Ordering::Relaxed);
                if let Some(gate) = &self.gate {
                    gate.notified().await;
                }
                self.result.clone()
            })
        }
    }

    fn setup(fake: Option<Fake>) -> (Arc<OutputSpill>, Option<Arc<Fake>>, Arc<QueryContext>) {
        let pool = TreePool::new(&MemoryConfig::default());
        let query = QueryContext::new(
            QueryId(1),
            WorkloadClass::Batch,
            ExecConfig::default(),
            pool.child("q", None),
        );
        let executor = TokioExecutor::deterministic(1).unwrap_or_else(|e| unreachable!("{e}"));
        let ctx = TaskContext::new(
            Arc::clone(&query),
            executor,
            SpillOwner::new(),
            OperatorId(1),
            "output",
        );
        let (hook, _consumer) = OutputSpill::register(&ctx, "output");
        let fake = fake.map(Arc::new);
        if let Some(f) = &fake {
            let weak: Weak<dyn SpillUnread> = Arc::downgrade(f) as Weak<Fake>;
            hook.attach(weak);
        }
        hook.set_spillable_bytes(100);
        (hook, fake, query)
    }

    fn fake(result: Result<usize>, gated: bool) -> Fake {
        Fake {
            spills: AtomicUsize::new(0),
            result,
            gate: gated.then(tokio::sync::Notify::new),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn requests_coalesce_into_one_spill() {
        let (hook, fake, _q) = setup(Some(fake(Ok(100), true)));
        let fake = fake.unwrap_or_else(|| unreachable!());
        assert_eq!(hook.spillable_bytes(), 100);
        assert_eq!(hook.owner(), None);
        // A zero target asks for nothing.
        assert_eq!(hook.request_spill(0).await, 0);
        let first = hook.request_spill(10);
        assert_eq!(hook.spill_requested(), Some(10));
        // The second joins the first (the task is running).
        let second = hook.request_spill(50);
        while fake.spills.load(Ordering::Relaxed) == 0 {
            tokio::task::yield_now().await;
        }
        fake.gate
            .as_ref()
            .unwrap_or_else(|| unreachable!())
            .notify_one();
        assert_eq!((first.await, second.await), (100, 100));
        assert_eq!(fake.spills.load(Ordering::Relaxed), 1);
        assert_eq!(hook.spill_requested(), None);
    }

    #[test]
    fn a_refused_scratch_frees_nothing_and_is_no_error() {
        let refusal = KrishivError::from(MemoryError::Exhausted {
            requested: 10,
            available: 0,
        });
        assert_eq!(refused(refusal).ok(), Some(0));
        let other = refused(internal("disk".to_owned())).err();
        assert_eq!(
            other.map(|e| e.error_class()),
            Some(ErrorClass::InternalError)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn without_an_output_a_request_frees_nothing() {
        let (hook, _, q) = setup(None);
        assert_eq!(hook.request_spill(10).await, 0);
        assert!(q.error().is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_spill_fails_the_query() {
        let (hook, _fake, q) = setup(Some(fake(Err(internal("disk full".to_owned())), false)));
        assert_eq!(hook.request_spill(10).await, 0);
        let e = q
            .error()
            .unwrap_or_else(|| unreachable!("the query failed"));
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        assert!(e.to_string().contains("disk full"), "{e}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_cancelled_query_stops_a_waiting_spill() {
        let (hook, fake, q) = setup(Some(fake(Ok(100), true)));
        let fake = fake.unwrap_or_else(|| unreachable!());
        let ticket = hook.request_spill(10);
        while fake.spills.load(Ordering::Relaxed) == 0 {
            tokio::task::yield_now().await;
        }
        q.cancel();
        assert_eq!(ticket.await, 0);
    }
}
