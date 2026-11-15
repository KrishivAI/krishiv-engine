// SPDX-License-Identifier: Apache-2.0
//! The tables finished drivers combined, spillable while no driver owns
//! them.
//!
//! A driver whose input ended hands its table to the global state
//! (`combine`) and ends; its spill hook goes with it. Until `finalize`, the
//! combined tables belong to no running driver. A running driver's own
//! spill writes them too, but only that driver's self-spill asks it to:
//! charged to the ended driver's consumer, an arbitration round could not
//! reach them, and a driver that had spilled just before another one
//! combined found the pool full of memory nothing could free. Whether the
//! query got its memory or failed with `UNABLE_TO_ACQUIRE_MEMORY` then
//! depended on which driver combined first: a scheduling accident.
//!
//! With a spill scope, [`CombinedSpill`] gives the combined tables a
//! consumer of their own whose hook ([`crate::output_spill`]) writes them
//! as spill runs on a task of its own, as a running driver's spill would.
//! Once `finalize` begins ([`Inner::finalizing`]) the hook frees nothing:
//! the finalizing driver's own hook serves the tables from then on. A
//! `finalize` waits for a hook write in progress ([`Inner::hook_writing`]),
//! so it sees the tables either in memory or as runs, never neither.

use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, Weak};

use futures::future::BoxFuture;
use krishiv_memory::ConsumerHandle;

use super::held::lend_headroom;
use super::{Global, Spiller};
use crate::context::TaskContext;
use crate::error::Result;
use crate::executor::{Executor, run_blocking};
use crate::output_spill::{OutputSpill, SpillUnread};
use crate::sync::lock;

/// The consumer of the combined tables and its spill hook.
pub(super) struct Combined {
    pub(super) hook: Arc<OutputSpill>,
    pub(super) consumer: Arc<ConsumerHandle>,
    /// Kept alive here: the hook holds it weakly.
    _spill: Arc<CombinedSpill>,
}

impl Combined {
    /// Registers the consumer for `ctx`'s operator; `spiller` makes the
    /// writer of the runs over it (`None` without a spill scope).
    pub(super) fn register(
        ctx: &TaskContext,
        global: Weak<Global>,
        spiller: impl FnOnce(Arc<ConsumerHandle>) -> Option<Spiller>,
    ) -> Option<Self> {
        let (hook, consumer) = OutputSpill::register(ctx, "HashAggregate combined");
        let spiller = spiller(Arc::clone(&consumer))?;
        let spill = Arc::new(CombinedSpill {
            global,
            spiller,
            executor: Arc::clone(ctx.executor()),
        });
        let weak: Weak<dyn SpillUnread> = Arc::downgrade(&spill) as Weak<dyn SpillUnread>;
        hook.attach(weak);
        Some(Self {
            hook,
            consumer,
            _spill: spill,
        })
    }
}

/// The spill side of the combined tables.
struct CombinedSpill {
    global: Weak<Global>,
    spiller: Spiller,
    executor: Arc<dyn Executor>,
}

/// Clears [`Inner::hook_writing`] and wakes a waiting `finalize`, on every
/// path (a cancelled query drops the spill mid-write).
struct Writing(Arc<Global>);

impl Drop for Writing {
    fn drop(&mut self) {
        lock(&self.0.inner).hook_writing = false;
        self.0.changed.notify();
    }
}

impl SpillUnread for CombinedSpill {
    fn spill_unread(self: Arc<Self>, _target: usize) -> BoxFuture<'static, Result<usize>> {
        Box::pin(async move {
            let Some(g) = self.global.upgrade() else {
                return Ok(0);
            };
            let tables = {
                let mut inner = lock(&g.inner);
                if inner.finalizing || inner.tables.is_empty() {
                    return Ok(0);
                }
                inner.hook_writing = true;
                g.combined.store(0, AtomicOrdering::Relaxed);
                std::mem::take(&mut inner.tables)
            };
            let writing = Writing(Arc::clone(&g));
            if let Some(c) = &g.spill_combined {
                c.hook.set_spillable_bytes(0);
            }
            let freed: usize = tables.iter().map(|(_, _, r)| r.size()).sum();
            let s = self.spiller.clone();
            let guard = s.query.blocking_started();
            // Until the memory written is freed, it counts as about to be
            // released (`ConsumerHandle::spilling`). No arbitration inside:
            // the writes run on the blocking pool, their scratch a sync grow.
            let _spilling = s.consumer.spilling();
            let runs = run_blocking(&*self.executor, move || {
                let _guard = guard;
                let mut runs = Vec::with_capacity(tables.len());
                for (p, t, mut r) in tables {
                    lend_headroom(&mut r);
                    runs.push(s.write_run(&t, 0, 0, p)?);
                }
                Ok(runs)
            })
            .await?;
            lock(&g.inner).runs.extend(runs);
            drop(writing);
            Ok(freed)
        })
    }
}
