// SPDX-License-Identifier: Apache-2.0
//! The pool an [`IoContext`](super::IoContext) reserves metadata reads from: a base consumer
//! (the session's pool while a statement is analyzed and planned) and, once the query runs,
//! the query's own consumer ([`QueryIoMemory`]), shared by every clone of the context.

use std::sync::{Arc, OnceLock, RwLock, Weak};

use krishiv_memory::{ConsumerHandle, MemoryPool};

/// The consumer name metadata reads are charged under.
pub(super) const CONSUMER: &str = "file-metadata-reads";

/// Shared by the clones of one context: binding the query's pool reaches the file tables
/// the planner opened with a clone, so their footer reads are the query's.
#[derive(Debug, Default)]
pub(super) struct MemorySlot {
    /// The pool charged when no running query is bound; its consumer registers on the first
    /// reservation, so a context that never reads metadata shows nothing in `system.memory`.
    base: Option<Arc<dyn MemoryPool>>,
    base_handle: OnceLock<Arc<ConsumerHandle>>,
    /// The running queries' consumers, oldest first. Weak: a context kept past its query (a
    /// cached plan's file tables) neither keeps a query's pool alive nor charges a query that
    /// is over. Several executions of one analyzed frame may overlap: the newest still alive
    /// is charged, and when it ends the earlier one is again, not the base.
    queries: RwLock<Vec<Weak<ConsumerHandle>>>,
}

impl MemorySlot {
    /// A slot charging `pool`.
    pub(super) fn over(pool: Arc<dyn MemoryPool>) -> Arc<Self> {
        Arc::new(Self {
            base: Some(pool),
            base_handle: OnceLock::new(),
            queries: RwLock::default(),
        })
    }

    /// The consumer reads are charged to now: the newest live bound query's, else the base.
    pub(super) fn current(&self) -> Option<Arc<ConsumerHandle>> {
        let live = self
            .queries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .rev()
            .find_map(Weak::upgrade);
        live.or_else(|| {
            let pool = self.base.as_ref()?;
            Some(Arc::clone(self.base_handle.get_or_init(|| {
                ConsumerHandle::register(Arc::clone(pool), CONSUMER, None)
            })))
        })
    }

    /// Binds `pool` as a running query's; reads charge it while the returned handle lives (and
    /// no newer bound query is alive).
    pub(super) fn bind(&self, pool: Arc<dyn MemoryPool>) -> QueryIoMemory {
        let handle = ConsumerHandle::register(pool, CONSUMER, None);
        let mut queries = self
            .queries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        queries.retain(|w| w.strong_count() > 0);
        queries.push(Arc::downgrade(&handle));
        QueryIoMemory(handle)
    }
}

/// The running query's consumer of metadata reads ([`IoContext::bind_query_memory`]): the
/// context's clones charge the query's pool while it lives. Keep it for the query's life
/// and drop it when the query ends.
///
/// [`IoContext::bind_query_memory`]: super::IoContext::bind_query_memory
#[derive(Debug)]
#[must_use = "the query's pool is charged only while this lives"]
pub struct QueryIoMemory(
    #[expect(dead_code, reason = "held for its registration")] Arc<ConsumerHandle>,
);
