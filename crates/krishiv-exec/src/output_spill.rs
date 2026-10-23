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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use futures::future::BoxFuture;
use krishiv_memory::{ConsumerHandle, OwnerId, SpillSignal, SpillTicket, Spillable};

use crate::context::{QueryContext, TaskContext};
use crate::error::Result;
use crate::executor::Executor;

/// The spill side of a finished output.
pub(crate) trait SpillUnread: Send + Sync + 'static {
    /// Moves the unread in-memory part to disk and frees its memory;
    /// returns the bytes freed (0 if nothing was left in memory). Runs on
    /// the hook's own task; blocking I/O goes to the blocking pool.
    fn spill_unread(self: Arc<Self>) -> BoxFuture<'static, Result<usize>>;
}

/// The memory consumer of a finished output and its spill hook (see the
/// [module docs](self)).
pub(crate) struct OutputSpill {
    me: Weak<Self>,
    signal: SpillSignal,
    output: OnceLock<Weak<dyn SpillUnread>>,
    /// A spill task is running: requests join it.
    running: AtomicBool,
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
            running: AtomicBool::new(false),
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
        if self.running.swap(true, Ordering::AcqRel) {
            return;
        }
        let Some(me) = self.me.upgrade() else {
            self.running.store(false, Ordering::Release);
            return;
        };
        let task = async move {
            loop {
                me.service().await;
                me.running.store(false, Ordering::Release);
                // A request that arrived after the last one completed
                // starts a new generation: serve it too.
                if me.signal.spill_requested().is_none() || me.running.swap(true, Ordering::AcqRel)
                {
                    return;
                }
            }
        };
        drop(self.executor.spawn(self.query.class(), Box::pin(task)));
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
