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
//! Once `finalize` begins (`Inner::finalizing`) the hook frees nothing:
//! the finalizing driver's own hook serves the tables from then on. A
//! `finalize` waits for a hook write in progress (`Inner::hook_writing`),
//! so it sees the tables either in memory or as runs, never neither.
//!
//! A write the pool refuses its scratch frees nothing and is no error (as
//! any spill's, [`crate::output_spill::refused`]): the tables not written
//! go back, in driver-partition order, and the hook offers them again; the
//! bytes it reports freed are only those of the runs it wrote.

use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, Weak};

use futures::future::BoxFuture;
use krishiv_memory::{ConsumerHandle, Reservation};

use super::held::{lend_headroom_above, regrow};
use super::{Global, Run, Spiller, Table, held};
use crate::context::TaskContext;
use crate::error::Result;
use crate::executor::{Executor, run_blocking};
use crate::output_spill::{OutputSpill, SpillUnread, is_refusal};
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

/// What a hook write did: the runs written, the bytes their tables held,
/// and the tables kept in memory because the pool refused a run's scratch.
#[derive(Default)]
struct Written {
    runs: Vec<Run>,
    freed: usize,
    kept: Vec<(u32, Table, Reservation)>,
}

/// Clears `Inner::hook_writing` and wakes a waiting `finalize`, on every
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
                // The finalize state owns the tables once `finalize` began
                // (or a free job or the disk phase took over).
                if inner.finalizing || inner.busy || inner.disk || inner.tables.is_empty() {
                    return Ok(0);
                }
                inner.hook_writing = true;
                g.combined.store(0, AtomicOrdering::Relaxed);
                // Under the lock: a `combine` after it publishes its own.
                if let Some(c) = &g.spill_combined {
                    c.hook.set_spillable_bytes(0);
                }
                std::mem::take(&mut inner.tables)
            };
            let writing = Writing(Arc::clone(&g));
            let s = self.spiller.clone();
            let guard = s.query.blocking_started();
            // Until the memory written is freed, it counts as about to be
            // released (`ConsumerHandle::spilling`). No arbitration inside:
            // the writes run on the blocking pool, their scratch a sync grow.
            let _spilling = s.consumer.spilling();
            let query = Arc::clone(&s.query);
            let written = run_blocking(&*self.executor, move || {
                let mut out = Written::default();
                let mut tables = tables.into_iter();
                while let Some((p, t, mut r)) = tables.next() {
                    let size = r.size();
                    // Only the headroom above the table: a table kept by an
                    // earlier refused write may have lost it already.
                    let lent = lend_headroom_above(&mut r, t.memory_size());
                    match s.write_run(&t, 0, 0, p) {
                        Ok(run) => {
                            out.runs.push(run);
                            out.freed += size;
                        }
                        Err(e) if is_refusal(&e) => {
                            // A refused scratch frees nothing and is no
                            // error (as `output_spill::refused`): this
                            // table and the rest go back, still in memory.
                            regrow(&mut r, lent);
                            out.kept.push((p, t, r));
                            out.kept.extend(tables);
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                }
                // The guard drops after the kept tables (tuple order): the
                // query's blocking count stays up until their memory is
                // back with the hook's state, or freed.
                Ok((out, guard))
            })
            .await;
            let (out, _guard) = match written {
                Ok(written) => written,
                Err(e) => {
                    // The tables are gone: the query fails before `finalize`
                    // (waiting on this write) can run on what is left.
                    query.fail(e.clone());
                    return Err(e);
                }
            };
            {
                let mut inner = lock(&g.inner);
                inner.runs.extend(out.runs);
                if !out.kept.is_empty() {
                    // `finalize` waits for this write, so it has not begun:
                    // the kept tables are the hook's to offer again. Tables
                    // merge in driver-partition order.
                    for (p, t, r) in out.kept {
                        g.combined.fetch_add(r.size(), AtomicOrdering::Relaxed);
                        let at = inner.tables.partition_point(|(q, _, _)| *q < p);
                        inner.tables.insert(at, (p, t, r));
                    }
                    if let Some(c) = &g.spill_combined {
                        c.hook.set_spillable_bytes(held(&inner));
                    }
                }
            }
            let freed = out.freed;
            drop(writing);
            Ok(freed)
        })
    }
}
