// SPDX-License-Identifier: Apache-2.0
//! The pool tree and its lock-free accounting.
//!
//! Every node keeps `reserved`, the bytes granted to consumers in its
//! subtree, and enforces `reserved <= limit` with a compare-and-swap. A grant
//! walks from the consumer's node to the root, adding to each node; if a node
//! refuses, the nodes below it are rolled back and the refusing node is
//! reported as the *binding* pool for arbitration. Each consumer's
//! `used`/`granted` pair sits behind its own mutex, held across the walk, so
//! operations on one consumer are serialised and node counters always equal
//! the sum of the grants below them once no walk is in flight.
//!
//! A read-only pass checks the whole chain first, so a full pool refuses
//! without touching the counters above it. Arbitration (the slow path)
//! builds on this fast path.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use krishiv_common::config::{
    Config, ConfigValue, MEMORY_ABORT_RELEASE_TIMEOUT, MEMORY_LIMIT, MEMORY_RESERVATION_QUANTUM,
    MEMORY_SPILL_WAIT_TIMEOUT,
};

use crate::spill::Spillable;
use crate::sync::{AtomicUsize, Mutex, Ordering, lock};
use crate::{BoxFuture, ConsumerId, MemoryError, MemoryPool};

/// Largest byte count a pool can hold.
const MAX_BYTES: usize = usize::MAX >> 1;
/// A consumer's quantum is at most this fraction of the smallest limit above
/// it, so slack stays small under small limits.
const QUANTUM_LIMIT_DIVISOR: usize = 64;

/// Settings of a pool tree, normally read from the `krishiv.memory.*` config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryConfig {
    /// Limit of the root pool; `None` is unbounded. From
    /// `krishiv.memory.limit`, else 80% of the effective memory limit.
    pub limit: Option<usize>,
    /// Reservation granularity (`krishiv.memory.reservationQuantum`).
    pub quantum: usize,
    /// How long a round waits for requested spills
    /// (`krishiv.memory.spillWaitTimeout`).
    pub spill_wait_timeout: Duration,
    /// How long a request waits for aborted queries to release their memory
    /// (`krishiv.memory.abortReleaseTimeout`).
    pub abort_release_timeout: Duration,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::from_config(&Config::new())
    }
}

impl MemoryConfig {
    /// Reads the `krishiv.memory.*` entries, falling back to their defaults;
    /// an unset limit becomes [`crate::default_limit`].
    pub fn from_config(config: &Config) -> Self {
        let size = |v: Option<&ConfigValue>| {
            v.and_then(ConfigValue::as_size)
                .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
        };
        let duration = |v: Option<&ConfigValue>, default| {
            v.and_then(ConfigValue::as_duration).unwrap_or(default)
        };
        Self {
            limit: size(config.get(&MEMORY_LIMIT)).or_else(crate::default_limit),
            quantum: size(config.get(&MEMORY_RESERVATION_QUANTUM)).unwrap_or(1 << 20),
            spill_wait_timeout: duration(
                config.get(&MEMORY_SPILL_WAIT_TIMEOUT),
                Duration::from_secs(2),
            ),
            abort_release_timeout: duration(
                config.get(&MEMORY_ABORT_RELEASE_TIMEOUT),
                Duration::from_secs(30),
            ),
        }
    }
}

/// State shared by every node of one tree.
pub(crate) struct Shared {
    quantum: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Usage {
    /// Bytes reserved by the consumer's callers.
    pub(crate) used: usize,
    /// Bytes taken from the pools: `used <= granted <= round_up(used)`.
    pub(crate) granted: usize,
    /// Unregistered: an in-flight reservation must not grant any more.
    closed: bool,
}

/// A registered consumer.
pub(crate) struct Consumer {
    pub(crate) id: ConsumerId,
    name: String,
    /// Held weakly: a dropped hook means the consumer cannot be asked.
    spill: Option<std::sync::Weak<dyn Spillable>>,
    pub(crate) quantum: usize,
    pub(crate) usage: Mutex<Usage>,
}

impl fmt::Debug for Consumer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Consumer")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("spillable", &self.spill.is_some())
            .finish_non_exhaustive()
    }
}

/// Why a fast-path reservation was refused.
pub(crate) enum Denied {
    /// Not retryable.
    Error(MemoryError),
    /// `binding`'s limit refused the bytes.
    Limit { binding: Arc<Node> },
}

/// A pool in the tree.
pub(crate) struct Node {
    name: String,
    limit: Option<usize>,
    /// Bytes held by the subtree.
    word: AtomicUsize,
    parent: Option<Arc<Node>>,
    /// 0 = process root, 1 = query, deeper = task etc.
    depth: usize,
    consumers: Mutex<HashMap<ConsumerId, Arc<Consumer>>>,
    pub(crate) shared: Arc<Shared>,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("name", &self.name)
            .field("depth", &self.depth)
            .field("limit", &self.limit)
            .field("reserved", &self.reserved())
            .finish_non_exhaustive()
    }
}

/// `n` rounded up to a multiple of `quantum` (`n` itself on overflow).
pub(crate) fn round_up(n: usize, quantum: usize) -> usize {
    n.checked_next_multiple_of(quantum.max(1)).unwrap_or(n)
}

impl Node {
    pub(crate) fn reserved(&self) -> usize {
        self.load_word()
    }

    /// Reads the counter word. Every decision taken from this read is
    /// re-validated by the compare-and-swap in [`Node::try_add`], so a stale
    /// value only costs a retry or a trip to the slow path.
    fn load_word(&self) -> usize {
        self.word.load(Ordering::Acquire)
    }

    /// Free bytes under this node's own limit.
    pub(crate) fn available(&self) -> usize {
        self.limit
            .map_or(MAX_BYTES, |l| l.saturating_sub(self.reserved()))
    }

    pub(crate) fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Adds `n` unless the limit refuses.
    fn try_add(&self, n: usize) -> bool {
        let limit = self.limit.unwrap_or(MAX_BYTES);
        let mut cur = self.load_word();
        loop {
            let Some(next) = cur.checked_add(n).filter(|t| *t <= limit) else {
                return false;
            };
            match self
                .word
                .compare_exchange(cur, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return true,
                Err(actual) => cur = actual,
            }
        }
    }

    fn sub(&self, n: usize) {
        let prev = self.word.fetch_sub(n, Ordering::AcqRel);
        // Callers only return what they were granted; a wrap here would be
        // an accounting bug, which the proptest and loom models check.
        debug_assert!(prev >= n, "pool {} released more than reserved", self.name);
    }

    /// Whether this node would refuse `n` now (a read-only check).
    fn refuses(&self, n: usize) -> bool {
        self.load_word()
            .checked_add(n)
            .is_none_or(|t| t > self.limit.unwrap_or(MAX_BYTES))
    }

    /// Adds `n` to this node and every ancestor, or to none of them.
    fn add_chain(self: &Arc<Self>, n: usize) -> Result<(), Arc<Node>> {
        // Read-only pass first: a full pool refuses without any store, so
        // retries never churn the shared counters above it.
        let mut cur = Some(self);
        while let Some(node) = cur {
            if node.refuses(n) {
                return Err(Arc::clone(node));
            }
            cur = node.parent.as_ref();
        }
        let mut cur = Some(self);
        while let Some(node) = cur {
            if !node.try_add(n) {
                let mut back = Some(self);
                while let Some(b) = back {
                    if Arc::ptr_eq(b, node) {
                        break;
                    }
                    b.sub(n);
                    back = b.parent.as_ref();
                }
                return Err(Arc::clone(node));
            }
            cur = node.parent.as_ref();
        }
        Ok(())
    }

    /// Subtracts `n` from this node and every ancestor.
    pub(crate) fn sub_chain(&self, n: usize) {
        if n == 0 {
            return;
        }
        let mut cur = Some(self);
        while let Some(node) = cur {
            node.sub(n);
            cur = node.parent.as_deref();
        }
    }

    pub(crate) fn consumer(&self, id: ConsumerId) -> Result<Arc<Consumer>, MemoryError> {
        lock(&self.consumers)
            .get(&id)
            .cloned()
            .ok_or(MemoryError::UnknownConsumer { id })
    }

    /// The fast path: no arbitration, no await.
    pub(crate) fn try_reserve_for(
        self: &Arc<Self>,
        consumer: &Consumer,
        bytes: usize,
    ) -> Result<(), Denied> {
        let mut u = lock(&consumer.usage);
        if u.closed {
            return Err(Denied::Error(MemoryError::UnknownConsumer {
                id: consumer.id,
            }));
        }
        let exhausted = |available| {
            Denied::Error(MemoryError::Exhausted {
                requested: bytes,
                available,
            })
        };
        let new_used = u
            .used
            .checked_add(bytes)
            .filter(|n| *n <= MAX_BYTES)
            .ok_or_else(|| exhausted(0))?;
        if new_used <= u.granted {
            u.used = new_used;
            return Ok(());
        }
        let quantised = round_up(new_used, consumer.quantum).min(MAX_BYTES);
        let mut result = self.add_chain(quantised - u.granted).map(|()| quantised);
        if result.is_err() && quantised > new_used {
            // The rounded amount does not fit; the exact amount might.
            result = self.add_chain(new_used - u.granted).map(|()| new_used);
        }
        match result {
            Ok(granted) => {
                u.used = new_used;
                u.granted = granted;
                Ok(())
            }
            Err(binding) => Err(Denied::Limit { binding }),
        }
    }

    pub(crate) fn release_for(&self, consumer: &Consumer, bytes: usize) {
        let mut u = lock(&consumer.usage);
        u.used -= bytes.min(u.used);
        let granted = u.granted.min(round_up(u.used, consumer.quantum));
        self.sub_chain(u.granted - granted);
        u.granted = granted;
    }

    /// The quantum for a consumer registered here.
    fn consumer_quantum(&self) -> usize {
        let mut q = self.shared.quantum;
        let mut cur = Some(self);
        while let Some(n) = cur {
            if let Some(l) = n.limit {
                q = q.min(l / QUANTUM_LIMIT_DIVISOR);
            }
            cur = n.parent.as_deref();
        }
        q.max(1)
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // Children hold their parent, so they are gone and have already
        // returned their grants: what is left belongs to our own consumers.
        let own = self.reserved();
        if let Some(p) = &self.parent {
            p.sub_chain(own);
        }
    }
}

static NEXT_CONSUMER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The pool-tree implementation of [`MemoryPool`]. The root's children are
/// queries ([`MemoryPool::child`]); below a query, any depth of task,
/// pipeline and operator pools.
#[derive(Debug)]
pub struct TreePool {
    node: Arc<Node>,
}

impl TreePool {
    /// A new process root.
    pub fn new(config: &MemoryConfig) -> Arc<Self> {
        let shared = Arc::new(Shared {
            quantum: config.quantum.max(1),
        });
        Arc::new(Self {
            node: Arc::new(Node {
                name: "root".to_owned(),
                limit: config.limit.map(|l| l.min(MAX_BYTES)),
                word: AtomicUsize::new(0),
                parent: None,
                depth: 0,
                consumers: Mutex::new(HashMap::new()),
                shared,
            }),
        })
    }

    fn new_child(parent: &Arc<Node>, name: &str, limit: Option<usize>) -> Arc<Self> {
        Arc::new(Self {
            node: Arc::new(Node {
                name: name.to_owned(),
                limit: limit.map(|l| l.min(MAX_BYTES)),
                word: AtomicUsize::new(0),
                parent: Some(Arc::clone(parent)),
                depth: parent.depth + 1,
                consumers: Mutex::new(HashMap::new()),
                shared: Arc::clone(&parent.shared),
            }),
        })
    }

    /// [`MemoryPool::child`], returning the concrete type.
    pub(crate) fn child_pool(&self, name: &str, limit: Option<usize>) -> Arc<TreePool> {
        Self::new_child(&self.node, name, limit)
    }

    #[cfg(test)]
    pub(crate) fn node(&self) -> &Arc<Node> {
        &self.node
    }
}

impl MemoryPool for TreePool {
    fn child(&self, name: &str, limit: Option<usize>) -> Arc<dyn MemoryPool> {
        self.child_pool(name, limit)
    }

    fn register(&self, name: &str, spill: Option<Arc<dyn Spillable>>) -> ConsumerId {
        let id = ConsumerId(NEXT_CONSUMER.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        let consumer = Arc::new(Consumer {
            id,
            name: name.to_owned(),
            spill: spill.as_ref().map(Arc::downgrade),
            quantum: self.node.consumer_quantum(),
            usage: Mutex::new(Usage::default()),
        });
        lock(&self.node.consumers).insert(id, consumer);
        id
    }

    fn unregister(&self, id: ConsumerId) {
        let removed = lock(&self.node.consumers).remove(&id);
        if let Some(c) = removed {
            let mut u = lock(&c.usage);
            self.node.sub_chain(u.granted);
            *u = Usage {
                closed: true,
                ..Usage::default()
            };
        }
    }

    fn try_reserve(&self, id: ConsumerId, bytes: usize) -> Result<(), MemoryError> {
        let consumer = self.node.consumer(id)?;
        self.node
            .try_reserve_for(&consumer, bytes)
            .map_err(|d| match d {
                Denied::Error(e) => e,
                Denied::Limit { binding } => MemoryError::Exhausted {
                    requested: bytes,
                    available: binding.available(),
                },
            })
    }

    fn reserve<'a>(
        &'a self,
        id: ConsumerId,
        bytes: usize,
    ) -> BoxFuture<'a, Result<(), MemoryError>> {
        // The fast path until the arbitrator lands.
        Box::pin(std::future::ready(self.try_reserve(id, bytes)))
    }

    fn release(&self, id: ConsumerId, bytes: usize) {
        if let Ok(c) = self.node.consumer(id) {
            self.node.release_for(&c, bytes);
        }
    }

    fn reserved(&self) -> usize {
        self.node.reserved()
    }

    fn limit(&self) -> Option<usize> {
        self.node.limit()
    }
}
