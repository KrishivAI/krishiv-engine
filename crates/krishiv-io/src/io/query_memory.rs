// SPDX-License-Identifier: Apache-2.0
//! The pool an [`IoContext`](super::IoContext) reserves metadata reads from: a base consumer
//! (the session's pool while a statement is analyzed and planned) and, once the query runs,
//! the query's own consumer ([`QueryIoMemory`]), shared by every clone of the context.

use std::sync::{Arc, RwLock, Weak};

use krishiv_memory::{ConsumerHandle, MemoryPool};

/// The consumer name metadata reads are charged under.
pub(super) const CONSUMER: &str = "file-metadata-reads";

/// Shared by the clones of one context: binding the query's pool reaches the file tables
/// the planner opened with a clone, so their footer reads are the query's.
#[derive(Debug, Default)]
pub(super) struct MemorySlot {
    /// Charged when no running query is bound.
    base: Option<Arc<ConsumerHandle>>,
    /// The running query's consumer. Weak: a context kept past its query (a cached plan's
    /// file tables) neither keeps the query's pool alive nor charges a query that is over;
    /// it falls back to `base`.
    query: RwLock<Weak<ConsumerHandle>>,
}

impl MemorySlot {
    /// A slot charging `pool`.
    pub(super) fn over(pool: Arc<dyn MemoryPool>) -> Arc<Self> {
        Arc::new(Self {
            base: Some(ConsumerHandle::register(pool, CONSUMER, None)),
            query: RwLock::default(),
        })
    }

    /// The consumer reads are charged to now: the bound query's while it runs, else the base.
    pub(super) fn current(&self) -> Option<Arc<ConsumerHandle>> {
        self.query
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .upgrade()
            .or_else(|| self.base.clone())
    }

    /// Binds `pool` as the query's; reads charge it while the returned handle lives.
    pub(super) fn bind(&self, pool: Arc<dyn MemoryPool>) -> QueryIoMemory {
        let handle = ConsumerHandle::register(pool, CONSUMER, None);
        *self
            .query
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::downgrade(&handle);
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
