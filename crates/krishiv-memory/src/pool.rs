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
//! A query pool's counter word also carries an abort bit: once set, every
//! grant through that query fails in the same compare-and-swap, so the
//! query's held bytes can only go down after an abort.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Weak};
use std::time::Duration;

use krishiv_common::config::{
    Config, ConfigValue, MEMORY_ABORT_RELEASE_TIMEOUT, MEMORY_LIMIT, MEMORY_RESERVATION_QUANTUM,
    MEMORY_SPILL_WAIT_TIMEOUT,
};

use crate::event::Event;
use crate::rounds::Rounds;
use crate::spill::{OwnerId, Spillable};
use crate::sync::{AtomicBool, AtomicU64, AtomicUsize, Mutex, Ordering, fence, lock};
use crate::{BoxFuture, ConsumerId, MemoryError, MemoryPool};

/// The abort bit of a query pool's counter word.
const ABORTED: usize = 1 << (usize::BITS - 1);
/// Largest byte count a pool can hold (below the abort bit).
const MAX_BYTES: usize = ABORTED - 1;
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

/// Counters of arbitration activity for one pool tree (monotonic).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArbitrationStats {
    /// Arbitration rounds run.
    pub rounds: u64,
    /// Spill requests sent to consumers.
    pub spill_requests: u64,
    /// Queries aborted.
    pub aborts: u64,
}

#[derive(Debug, Default)]
pub(crate) struct Stats {
    pub(crate) rounds: AtomicU64,
    pub(crate) spill_requests: AtomicU64,
    pub(crate) aborts: AtomicU64,
}

/// Wakes arbitration rounds waiting for memory to be released. Releases
/// only touch the event while someone waits (the common case is one load).
#[derive(Debug, Default)]
pub(crate) struct ReleaseSignal {
    waiters: AtomicUsize,
    event: Event,
}

impl ReleaseSignal {
    /// Releaser side, after the counters went down.
    pub(crate) fn released(&self) {
        // Pairs with the fence in `WaitGuard::listen`: either the waiter's
        // re-check sees this release, or this load sees the waiter.
        fence(Ordering::SeqCst);
        if self.waiters.load(Ordering::SeqCst) > 0 {
            self.event.notify();
        }
    }

    /// Registers a waiter until the guard drops.
    pub(crate) fn wait(&self) -> WaitGuard<'_> {
        self.waiters.fetch_add(1, Ordering::SeqCst);
        WaitGuard(self)
    }
}

/// A registered release waiter.
pub(crate) struct WaitGuard<'a>(&'a ReleaseSignal);

impl WaitGuard<'_> {
    /// A listener for the next release; re-check the condition after this.
    pub(crate) fn listen(&self) -> crate::event::Listener<'_> {
        let l = self.0.event.listen();
        fence(Ordering::SeqCst);
        l
    }
}

impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        self.0.waiters.fetch_sub(1, Ordering::SeqCst);
    }
}

/// State shared by every node of one tree.
pub(crate) struct Shared {
    quantum: usize,
    pub(crate) spill_wait_timeout: Duration,
    pub(crate) abort_release_timeout: Duration,
    /// Every live consumer of the tree, for victim selection. Weak: nodes own
    /// their consumers, so a dropped subtree disappears from here too.
    registry: Mutex<HashMap<ConsumerId, Weak<Consumer>>>,
    pub(crate) rounds: Rounds,
    pub(crate) release: ReleaseSignal,
    pub(crate) stats: Stats,
}

impl Shared {
    /// Live consumers (snapshot; the registry lock is not held on return).
    pub(crate) fn consumers(&self) -> Vec<Arc<Consumer>> {
        lock(&self.registry)
            .values()
            .filter_map(Weak::upgrade)
            .collect()
    }
}

/// What a depth-1 subtree is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GroupKind {
    /// A query: arbitrates, can be aborted.
    Query,
    /// Capped headroom for external consumers: fast path only, may be asked
    /// to shed, never aborted and never the cause of an abort.
    External,
}

/// Abort state of one depth-1 subtree, shared by all of its nodes.
#[derive(Debug)]
pub(crate) struct Group {
    pub(crate) kind: GroupKind,
    aborted: AtomicBool,
    error: Mutex<Option<MemoryError>>,
    pub(crate) abort_event: Event,
}

impl Group {
    fn new(kind: GroupKind) -> Self {
        Self {
            kind,
            aborted: AtomicBool::new(false),
            error: Mutex::new(None),
            abort_event: Event::default(),
        }
    }

    /// One atomic load while not aborted.
    pub(crate) fn check(&self) -> Result<(), MemoryError> {
        if self.aborted.load(Ordering::Acquire)
            && let Some(e) = lock(&self.error).clone()
        {
            return Err(e);
        }
        Ok(())
    }

    pub(crate) fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }
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
    spill: Option<Weak<dyn Spillable>>,
    pub(crate) owner: Option<OwnerId>,
    pub(crate) quantum: usize,
    pub(crate) usage: Mutex<Usage>,
    pub(crate) node: Weak<Node>,
}

impl fmt::Debug for Consumer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Consumer")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

impl Consumer {
    /// The live spill hook, or `None` if the consumer cannot spill or its
    /// owner has gone.
    pub(crate) fn spillable(&self) -> Option<Arc<dyn Spillable>> {
        self.spill.as_ref().and_then(Weak::upgrade)
    }

    pub(crate) fn used(&self) -> usize {
        lock(&self.usage).used
    }

    /// Returns the quantisation slack (`granted - used`) to the pools.
    pub(crate) fn trim(&self, node: &Node) -> usize {
        let mut u = lock(&self.usage);
        let slack = u.granted - u.used;
        if slack > 0 {
            node.sub_chain(slack);
            u.granted = u.used;
        }
        slack
    }
}

/// Why a fast-path reservation was refused.
pub(crate) enum Denied {
    /// Not retryable.
    Error(MemoryError),
    /// `binding`'s limit refused `need` more bytes.
    Limit { binding: Arc<Node>, need: usize },
}

/// A pool in the tree.
pub(crate) struct Node {
    name: String,
    limit: Option<usize>,
    /// Bytes held by the subtree; on a query node, plus the abort bit.
    word: AtomicUsize,
    parent: Option<Arc<Node>>,
    /// 0 = process root, 1 = query or external pool, deeper = task etc.
    pub(crate) depth: usize,
    /// `None` for the root.
    pub(crate) group: Option<Arc<Group>>,
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

/// Why a node refused an addition.
enum Refusal {
    Limit,
    Aborted,
}

impl Node {
    pub(crate) fn reserved(&self) -> usize {
        self.load_word() & MAX_BYTES
    }

    /// Reads the counter word. Every decision taken from this read is
    /// re-validated by the compare-and-swap in [`Node::try_add`], so a stale
    /// value only costs a retry or a trip to the slow path.
    #[cfg(not(all(test, loom)))]
    fn load_word(&self) -> usize {
        self.word.load(Ordering::Acquire)
    }

    /// Under loom 0.7 a plain load that returns a superseded store can trip
    /// loom's own store-history assertion (`match_load_to_stores`, "TODO:
    /// this sometimes fails"), so the models read the counter with an RMW,
    /// which returns the latest value. Stale counter reads are still covered
    /// by real races on the CAS.
    #[cfg(all(test, loom))]
    fn load_word(&self) -> usize {
        self.word.fetch_add(0, Ordering::AcqRel)
    }

    /// Free bytes under this node's own limit.
    pub(crate) fn available(&self) -> usize {
        self.limit
            .map_or(MAX_BYTES, |l| l.saturating_sub(self.reserved()))
    }

    pub(crate) fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Adds `n` unless the limit or the abort bit refuses.
    fn try_add(&self, n: usize) -> Result<(), Refusal> {
        let limit = self.limit.unwrap_or(MAX_BYTES);
        let mut cur = self.load_word();
        loop {
            if cur & ABORTED != 0 {
                return Err(Refusal::Aborted);
            }
            let Some(next) = cur.checked_add(n).filter(|t| *t <= limit) else {
                return Err(Refusal::Limit);
            };
            match self
                .word
                .compare_exchange(cur, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Ok(()),
                Err(actual) => cur = actual,
            }
        }
    }

    fn sub(&self, n: usize) {
        let prev = self.word.fetch_sub(n, Ordering::AcqRel);
        // Callers only return what they were granted; a wrap here would be
        // an accounting bug, which the proptest and loom models check.
        debug_assert!(
            prev & MAX_BYTES >= n,
            "pool {} released more than reserved",
            self.name
        );
    }

    /// Why this node would refuse `n` now (a read-only check).
    fn refusal(&self, n: usize) -> Option<Refusal> {
        let cur = self.load_word();
        if cur & ABORTED != 0 {
            Some(Refusal::Aborted)
        } else if cur
            .checked_add(n)
            .is_none_or(|t| t > self.limit.unwrap_or(MAX_BYTES))
        {
            Some(Refusal::Limit)
        } else {
            None
        }
    }

    /// Adds `n` to this node and every ancestor, or to none of them.
    fn add_chain(self: &Arc<Self>, n: usize) -> Result<(), (Arc<Node>, Refusal)> {
        // Read-only pass first: a full pool refuses without any store, so
        // retries never churn the shared counters above it.
        let mut cur = Some(self);
        while let Some(node) = cur {
            if let Some(why) = node.refusal(n) {
                return Err((Arc::clone(node), why));
            }
            cur = node.parent.as_ref();
        }
        let mut cur = Some(self);
        while let Some(node) = cur {
            if let Err(why) = node.try_add(n) {
                let mut back = Some(self);
                while let Some(b) = back {
                    if Arc::ptr_eq(b, node) {
                        break;
                    }
                    b.sub(n);
                    back = b.parent.as_ref();
                }
                return Err((Arc::clone(node), why));
            }
            cur = node.parent.as_ref();
        }
        Ok(())
    }

    /// Subtracts `n` from this node and every ancestor, then wakes rounds
    /// waiting for memory.
    pub(crate) fn sub_chain(&self, n: usize) {
        if n == 0 {
            return;
        }
        let mut cur = Some(self);
        while let Some(node) = cur {
            node.sub(n);
            cur = node.parent.as_deref();
        }
        self.shared.release.released();
    }

    /// Whether `self` is `ancestor` or below it.
    pub(crate) fn is_within(&self, ancestor: &Node) -> bool {
        let mut cur = Some(self);
        while let Some(n) = cur {
            if std::ptr::eq(n, ancestor) {
                return true;
            }
            cur = n.parent.as_deref();
        }
        false
    }

    /// The depth-1 node (query or external pool) this node belongs to.
    pub(crate) fn group_node(self: &Arc<Self>) -> Option<Arc<Node>> {
        let mut cur = self;
        loop {
            match cur.depth {
                0 => return None,
                1 => return Some(Arc::clone(cur)),
                _ => cur = cur.parent.as_ref()?,
            }
        }
    }

    /// Aborts this query node: later grants through it fail (abort bit, in
    /// the same word as its counter), drivers see `error` in
    /// `check_aborted`, and waiters are woken. The first error wins.
    pub(crate) fn abort(&self, error: MemoryError) {
        let Some(group) = self.group.as_ref().filter(|g| g.kind == GroupKind::Query) else {
            return;
        };
        {
            let mut e = lock(&group.error);
            if e.is_none() {
                *e = Some(error);
            }
        }
        self.word.fetch_or(ABORTED, Ordering::AcqRel);
        group.aborted.store(true, Ordering::Release);
        group.abort_event.notify();
    }

    pub(crate) fn consumer(&self, id: ConsumerId) -> Result<Arc<Consumer>, MemoryError> {
        lock(&self.consumers)
            .get(&id)
            .cloned()
            .ok_or(MemoryError::UnknownConsumer { id })
    }

    fn group_error(&self) -> MemoryError {
        self.group
            .as_ref()
            .and_then(|g| g.check().err())
            .unwrap_or(MemoryError::Aborted { held: 0 })
    }

    /// The fast path: no arbitration, no await.
    pub(crate) fn try_reserve_for(
        self: &Arc<Self>,
        consumer: &Consumer,
        bytes: usize,
    ) -> Result<(), Denied> {
        if let Some(g) = &self.group {
            g.check().map_err(Denied::Error)?;
        }
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
        if matches!(result, Err((_, Refusal::Limit))) && quantised > new_used {
            // The rounded amount does not fit; the exact amount might.
            result = self.add_chain(new_used - u.granted).map(|()| new_used);
        }
        match result {
            Ok(granted) => {
                u.used = new_used;
                u.granted = granted;
                Ok(())
            }
            Err((_, Refusal::Aborted)) => Err(Denied::Error(self.group_error())),
            Err((binding, Refusal::Limit)) => Err(Denied::Limit {
                need: new_used - u.granted,
                binding,
            }),
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
        let mut registry = lock(&self.shared.registry);
        for id in lock(&self.consumers).keys() {
            registry.remove(id);
        }
        drop(registry);
        let own = self.reserved();
        if let Some(p) = &self.parent {
            p.sub_chain(own);
        }
    }
}

static NEXT_CONSUMER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The pool-tree implementation of [`MemoryPool`]. The root's children are
/// queries ([`MemoryPool::child`]) or capped pools for external consumers
/// ([`TreePool::external`]); below a query, any depth of task, pipeline and
/// operator pools.
#[derive(Debug)]
pub struct TreePool {
    node: Arc<Node>,
}

impl TreePool {
    /// A new process root.
    pub fn new(config: &MemoryConfig) -> Arc<Self> {
        let shared = Arc::new(Shared {
            quantum: config.quantum.max(1),
            spill_wait_timeout: config.spill_wait_timeout,
            abort_release_timeout: config.abort_release_timeout,
            registry: Mutex::new(HashMap::new()),
            rounds: Rounds::default(),
            release: ReleaseSignal::default(),
            stats: Stats::default(),
        });
        Arc::new(Self {
            node: Arc::new(Node {
                name: "root".to_owned(),
                limit: config.limit.map(|l| l.min(MAX_BYTES)),
                word: AtomicUsize::new(0),
                parent: None,
                depth: 0,
                group: None,
                consumers: Mutex::new(HashMap::new()),
                shared,
            }),
        })
    }

    fn new_child(
        parent: &Arc<Node>,
        name: &str,
        limit: Option<usize>,
        group: Arc<Group>,
    ) -> Arc<Self> {
        Arc::new(Self {
            node: Arc::new(Node {
                name: name.to_owned(),
                limit: limit.map(|l| l.min(MAX_BYTES)),
                word: AtomicUsize::new(0),
                parent: Some(Arc::clone(parent)),
                depth: parent.depth + 1,
                group: Some(group),
                consumers: Mutex::new(HashMap::new()),
                shared: Arc::clone(&parent.shared),
            }),
        })
    }

    /// [`MemoryPool::child`], returning the concrete type.
    pub(crate) fn child_pool(&self, name: &str, limit: Option<usize>) -> Arc<TreePool> {
        let p = &self.node;
        let group = match &p.group {
            Some(g) => Arc::clone(g),
            None => Arc::new(Group::new(GroupKind::Query)),
        };
        Self::new_child(p, name, limit, group)
    }

    /// A capped pool under the root for external consumers (a UDF worker
    /// pool's RSS, scan prefetch buffers). Its consumers reserve with the
    /// fast path only, even through [`MemoryPool::reserve`]: they can be
    /// asked to shed memory by other rounds but never start one, so they
    /// never cause a query to be aborted, and they are never aborted.
    pub fn external(&self, name: &str, cap: usize) -> Arc<dyn MemoryPool> {
        let mut root = &self.node;
        while let Some(p) = &root.parent {
            root = p;
        }
        Self::new_child(
            root,
            name,
            Some(cap),
            Arc::new(Group::new(GroupKind::External)),
        )
    }

    /// Arbitration counters of the whole tree.
    pub fn stats(&self) -> ArbitrationStats {
        let s = &self.node.shared.stats;
        ArbitrationStats {
            rounds: s.rounds.load(Ordering::Relaxed),
            spill_requests: s.spill_requests.load(Ordering::Relaxed),
            aborts: s.aborts.load(Ordering::Relaxed),
        }
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
        let shared = &self.node.shared;
        let id = ConsumerId(NEXT_CONSUMER.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        let consumer = Arc::new(Consumer {
            id,
            name: name.to_owned(),
            owner: spill.as_ref().and_then(|s| s.owner()),
            spill: spill.as_ref().map(Arc::downgrade),
            quantum: self.node.consumer_quantum(),
            usage: Mutex::new(Usage::default()),
            node: Arc::downgrade(&self.node),
        });
        lock(&shared.registry).insert(id, Arc::downgrade(&consumer));
        lock(&self.node.consumers).insert(id, consumer);
        id
    }

    fn unregister(&self, id: ConsumerId) {
        let removed = lock(&self.node.consumers).remove(&id);
        lock(&self.node.shared.registry).remove(&id);
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
                Denied::Limit { binding, .. } => MemoryError::Exhausted {
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
        Box::pin(crate::arbitrate::reserve(Arc::clone(&self.node), id, bytes))
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

    fn check_aborted(&self) -> Result<(), MemoryError> {
        self.node.group.as_ref().map_or(Ok(()), |g| g.check())
    }
}
