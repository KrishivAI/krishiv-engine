// SPDX-License-Identifier: Apache-2.0
//! The group tree's state and the dispatcher (design `P3-T003-workload-management.md` §4, §5).
//!
//! One `Mutex<Tree>` holds every counter. Nothing is sent or dropped while it is held: a
//! dispatch returns the deliveries, and the caller sends them after unlocking, because a
//! `GroupSlot` that fails to reach a cancelled waiter releases itself (and locks again) when it
//! is dropped.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use krishiv_common::error::{KrishivError, KrishivResourceGroupQuotaExhausted};
use krishiv_exec::WorkloadClass;
use tokio::sync::oneshot;

use super::config::{GroupCfg, Policy, ScanQuota};

/// What a dispatch hands to a waiter: a slot, or the refusal of a quota spent while it waited.
pub(super) type Grant = Result<GroupSlot, KrishivError>;

/// A queued query.
struct Waiter {
    id: u64,
    seq: u64,
    arrived: Instant,
    tx: oneshot::Sender<Grant>,
}

/// Where a group is in its life: a reload turns a group the new file lacks into `Removed`
/// (it takes nothing new and is freed when its last query ends), and a freed index is reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Active,
    Removed,
    Free,
}

/// One group's configuration and counters.
struct Node {
    state: State,
    id: String,
    parent: Option<usize>,
    children: Vec<usize>,
    hard: usize,
    max_queued: usize,
    policy: Policy,
    weight: u64,
    soft_memory: Option<usize>,
    per_query_memory: Option<usize>,
    priority: u8,
    class: WorkloadClass,
    scan_quota: Option<ScanQuota>,
    // Counters, over the group's subtree.
    running: usize,
    queued: usize,
    last_dispatch: u64,
    scan_used: u64,
    window_end_ms: u64,
    // A leaf's queue and the memory its running queries report.
    waiting: VecDeque<Waiter>,
    reserved: Vec<(u64, Arc<AtomicUsize>)>,
}

/// The whole tree.
pub(super) struct Tree {
    nodes: Vec<Node>,
    free: Vec<usize>,
    roots: Vec<usize>,
    seq: u64,
    next_id: u64,
}

/// Why an admission was refused before queueing.
pub(super) enum Refusal {
    /// No active group has the id (a reload removed it).
    Missing,
    Quota(KrishivError),
    QueueFull {
        group: String,
        max_queued: usize,
    },
}

/// What a leaf gives the query it admits.
pub(super) struct Traits {
    pub(super) budget: Option<usize>,
    pub(super) priority: u8,
    pub(super) class: WorkloadClass,
}

/// A grant on its way to a waiter, sent once the lock is released.
pub(super) struct Delivery {
    tx: oneshot::Sender<Grant>,
    grant: Grant,
}

impl Delivery {
    /// Sends the grant; a waiter that is gone drops it, which releases the slot.
    pub(super) fn send(self) {
        drop(self.tx.send(self.grant));
    }
}

/// The shared state and the clock (milliseconds since the Unix epoch).
pub(super) struct Shared {
    tree: Mutex<Tree>,
    now_ms: Arc<dyn Fn() -> u64 + Send + Sync>,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared").finish_non_exhaustive()
    }
}

impl Shared {
    pub(super) fn new(roots: &[GroupCfg], now_ms: Arc<dyn Fn() -> u64 + Send + Sync>) -> Arc<Self> {
        let mut tree = Tree {
            nodes: Vec::new(),
            free: Vec::new(),
            roots: Vec::new(),
            seq: 0,
            next_id: 0,
        };
        let now = now_ms();
        for r in roots {
            let idx = tree.add(r, None, "", 50, WorkloadClass::Interactive, now);
            tree.roots.push(idx);
        }
        Arc::new(Self {
            tree: Mutex::new(tree),
            now_ms,
        })
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, Tree> {
        self.tree.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(super) fn now(&self) -> u64 {
        (self.now_ms)()
    }

    /// Swaps in a new group document (see `Tree::reload`) and hands out what the new limits let
    /// a queued query start.
    pub(super) fn reload(self: &Arc<Self>, roots: &[GroupCfg]) {
        let now = self.now();
        self.lock().reload(roots, now);
        self.dispatch();
    }

    /// Admits at once when the leaf may run another query and nothing waits ahead, else queues.
    ///
    /// The group is looked up by its dotted id under the same lock, so a reload between the
    /// selector and the admission can never leave a stale index: a group the reload removed is
    /// `Refusal::Missing` and the caller selects again.
    pub(super) fn admit(self: &Arc<Self>, group: &str) -> Result<(Admitted, Traits), Refusal> {
        let now = self.now();
        let mut tree = self.lock();
        let Some(leaf) = tree.active(group) else {
            return Err(Refusal::Missing);
        };
        let traits = tree.traits(leaf);
        if let Some(refusal) = tree.quota_refusal(leaf, now) {
            return Err(Refusal::Quota(refusal));
        }
        let immediate =
            tree.can_run(leaf) && tree.nodes.get(leaf).is_some_and(|n| n.waiting.is_empty());
        if immediate {
            return Ok((Admitted::Now(tree.start(self, leaf, 0)), traits));
        }
        // `maxQueued` limits waiting queries; a query that can start never waits.
        if let Some(full) = tree.full_queue(leaf) {
            return Err(Refusal::QueueFull {
                group: full.0,
                max_queued: full.1,
            });
        }
        let (tx, rx) = oneshot::channel();
        let id = tree.next_id();
        let seq = tree.bump();
        tree.enqueue(
            leaf,
            Waiter {
                id,
                seq,
                arrived: Instant::now(),
                tx,
            },
        );
        Ok((
            Admitted::Queued(
                rx,
                QueuedGuard {
                    shared: Arc::clone(self),
                    leaf,
                    id,
                },
            ),
            traits,
        ))
    }

    /// Runs the dispatcher and sends what it granted.
    fn dispatch(self: &Arc<Self>) {
        let now = self.now();
        let deliveries = self.lock().dispatch(self, now);
        for d in deliveries {
            d.send();
        }
    }

    /// One row per group, parents before their children; a removed group still holding queries
    /// shows as `removed` (`system.resource_groups`).
    pub(super) fn rows(&self) -> Vec<GroupRow> {
        let tree = self.lock();
        tree.order()
            .into_iter()
            .filter_map(|i| tree.nodes.get(i).map(|n| (i, n)))
            .map(|(i, n)| GroupRow {
                group_id: n.id.clone(),
                parent_id: n
                    .parent
                    .and_then(|p| tree.nodes.get(p))
                    .map(|p| p.id.clone()),
                state: if n.state == State::Active {
                    "active"
                } else {
                    "removed"
                },
                scheduling_policy: match n.policy {
                    Policy::WeightedFair => "weighted_fair",
                    Policy::QueryPriority => "query_priority",
                },
                scheduling_weight: i32::try_from(n.weight).unwrap_or(i32::MAX),
                priority: i32::from(n.priority),
                workload_class: n.class.as_str(),
                hard_concurrency_limit: i32::try_from(n.hard).unwrap_or(i32::MAX),
                max_queued: i32::try_from(n.max_queued).unwrap_or(i32::MAX),
                soft_memory_limit_bytes: n
                    .soft_memory
                    .map(|b| i64::try_from(b).unwrap_or(i64::MAX)),
                per_query_memory_limit_bytes: n
                    .per_query_memory
                    .map(|b| i64::try_from(b).unwrap_or(i64::MAX)),
                scan_quota_bytes: n
                    .scan_quota
                    .map(|q| i64::try_from(q.bytes).unwrap_or(i64::MAX)),
                scan_quota_period_s: n
                    .scan_quota
                    .map(|q| i64::try_from(q.period_ms / 1000).unwrap_or(i64::MAX)),
                running: i32::try_from(n.running).unwrap_or(i32::MAX),
                queued: i32::try_from(n.queued).unwrap_or(i32::MAX),
                memory_reserved_bytes: i64::try_from(tree.reserved_in(i)).unwrap_or(i64::MAX),
                scan_bytes_in_window: n
                    .scan_quota
                    .map(|_| i64::try_from(n.scan_used).unwrap_or(i64::MAX)),
                scan_window_resets_at_ms: n
                    .scan_quota
                    .map(|_| i64::try_from(n.window_end_ms).unwrap_or(i64::MAX)),
                config_etag: String::new(),
                loaded_at_ms: 0,
            })
            .collect()
    }

    /// `(running, queued)` of a group, by dotted id.
    pub(super) fn counts(&self, id: &str) -> Option<(usize, usize)> {
        self.lock()
            .nodes
            .iter()
            .filter(|n| n.state != State::Free)
            .find(|n| n.id == id)
            .map(|n| (n.running, n.queued))
    }
}

/// One group of the tree as `system.resource_groups` shows it.
#[derive(Debug, Clone)]
pub struct GroupRow {
    pub(crate) group_id: String,
    pub(crate) parent_id: Option<String>,
    pub(crate) state: &'static str,
    pub(crate) scheduling_policy: &'static str,
    pub(crate) scheduling_weight: i32,
    pub(crate) priority: i32,
    pub(crate) workload_class: &'static str,
    pub(crate) hard_concurrency_limit: i32,
    pub(crate) max_queued: i32,
    pub(crate) soft_memory_limit_bytes: Option<i64>,
    pub(crate) per_query_memory_limit_bytes: Option<i64>,
    pub(crate) scan_quota_bytes: Option<i64>,
    pub(crate) scan_quota_period_s: Option<i64>,
    pub(crate) running: i32,
    pub(crate) queued: i32,
    pub(crate) memory_reserved_bytes: i64,
    pub(crate) scan_bytes_in_window: Option<i64>,
    pub(crate) scan_window_resets_at_ms: Option<i64>,
    /// The etag and load time of the document the row came from (filled by `ResourceGroups::rows`).
    pub(crate) config_etag: String,
    pub(crate) loaded_at_ms: i64,
}

/// The result of asking a group for a slot.
pub(super) enum Admitted {
    Now(GroupSlot),
    Queued(oneshot::Receiver<Grant>, QueuedGuard),
}

/// Removes a waiter whose future was dropped while it queued.
pub(super) struct QueuedGuard {
    shared: Arc<Shared>,
    leaf: usize,
    id: u64,
}

impl Drop for QueuedGuard {
    fn drop(&mut self) {
        {
            let mut tree = self.shared.lock();
            tree.remove_waiter(self.leaf, self.id);
        }
        // Removing a waiter can free a removed group the parent's queue was waiting on.
        self.shared.dispatch();
    }
}

/// One running query's place in its group. Dropping it frees the place, charges the scan quota
/// and wakes the next queued query.
pub struct GroupSlot {
    shared: Arc<Shared>,
    leaf: usize,
    slot_id: u64,
    reserved: Arc<AtomicUsize>,
    bytes_read: AtomicU64,
    queued_ns: u64,
}

impl std::fmt::Debug for GroupSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupSlot")
            .field("leaf", &self.leaf)
            .field("queued_ns", &self.queued_ns)
            .finish_non_exhaustive()
    }
}

impl GroupSlot {
    /// Reports the bytes the query reserves now; a group's `softMemoryLimit` sums these.
    pub fn set_reserved(&self, bytes: usize) {
        let before = self.reserved.swap(bytes, Ordering::Relaxed);
        if bytes < before {
            // A lower reservation can bring a group back under its `softMemoryLimit`.
            self.shared.dispatch();
        }
    }

    /// Adds scanned bytes, charged to the scan quota of every group up to the root when the
    /// slot is dropped.
    pub fn add_bytes_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Nanoseconds the query waited in its group's queue.
    #[must_use]
    pub fn queued_ns(&self) -> u64 {
        self.queued_ns
    }
}

impl Drop for GroupSlot {
    fn drop(&mut self) {
        let now = self.shared.now();
        {
            let mut tree = self.shared.lock();
            tree.release(
                self.leaf,
                self.slot_id,
                self.bytes_read.load(Ordering::Relaxed),
                now,
            );
        }
        self.shared.dispatch();
    }
}

impl Node {
    /// Starts a fresh quota window when the current one has ended.
    fn roll_window(&mut self, now: u64) {
        let Some(q) = self.scan_quota else { return };
        if now >= self.window_end_ms {
            self.scan_used = 0;
            let periods = now.saturating_sub(self.window_end_ms) / q.period_ms.max(1) + 1;
            self.window_end_ms = self
                .window_end_ms
                .saturating_add(periods.saturating_mul(q.period_ms));
        }
    }
}

impl Tree {
    fn make_node(
        cfg: &GroupCfg,
        id: String,
        parent: Option<usize>,
        priority: u8,
        class: WorkloadClass,
        now: u64,
    ) -> Node {
        Node {
            state: State::Active,
            id,
            parent,
            children: Vec::new(),
            hard: cfg.hard_concurrency,
            max_queued: cfg.max_queued,
            policy: cfg.policy,
            weight: cfg.weight,
            soft_memory: cfg.soft_memory,
            per_query_memory: cfg.per_query_memory,
            priority,
            class,
            scan_quota: cfg.scan_quota,
            running: 0,
            queued: 0,
            last_dispatch: 0,
            scan_used: 0,
            window_end_ms: cfg
                .scan_quota
                .map_or(0, |q| now.saturating_add(q.period_ms)),
            waiting: VecDeque::new(),
            reserved: Vec::new(),
        }
    }

    /// Puts a node in a freed slot, or at the end.
    fn alloc(&mut self, node: Node) -> usize {
        if let Some((i, slot)) = self
            .free
            .pop()
            .and_then(|i| self.nodes.get_mut(i).map(|slot| (i, slot)))
        {
            *slot = node;
            return i;
        }
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn group_id(prefix: &str, name: &str) -> String {
        if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}.{name}")
        }
    }

    fn add(
        &mut self,
        cfg: &GroupCfg,
        parent: Option<usize>,
        prefix: &str,
        inherited_priority: u8,
        inherited_class: WorkloadClass,
        now: u64,
    ) -> usize {
        let id = Self::group_id(prefix, &cfg.name);
        let priority = cfg.priority.unwrap_or(inherited_priority);
        let class = cfg.class.unwrap_or(inherited_class);
        let idx = self.alloc(Self::make_node(
            cfg,
            id.clone(),
            parent,
            priority,
            class,
            now,
        ));
        for child in &cfg.children {
            let c = self.add(child, Some(idx), &id, priority, class, now);
            if let Some(n) = self.nodes.get_mut(idx) {
                n.children.push(c);
            }
        }
        idx
    }

    /// The index of the active group with this dotted id.
    fn active(&self, id: &str) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| n.state == State::Active && n.id == id)
    }

    fn traits(&self, leaf: usize) -> Traits {
        let n = self.nodes.get(leaf);
        Traits {
            budget: n.and_then(|n| n.per_query_memory),
            priority: n.map_or(50, |n| n.priority),
            class: n.map_or(WorkloadClass::Interactive, |n| n.class),
        }
    }

    /// Every live group, parents before their children.
    fn order(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack: Vec<usize> = self.roots.iter().rev().copied().collect();
        while let Some(i) = stack.pop() {
            out.push(i);
            if let Some(n) = self.nodes.get(i) {
                stack.extend(n.children.iter().rev().copied());
            }
        }
        out
    }

    /// Swaps in a new group document. A group the document still has keeps its counters, queue
    /// and running queries (matched by dotted id) and takes the new limits; a new group starts
    /// empty; a group the document lacks takes nothing new and is freed with its last query.
    pub(super) fn reload(&mut self, roots: &[GroupCfg], now: u64) {
        let mut seen = std::collections::HashSet::new();
        for r in roots {
            let idx = self.sync(r, None, "", 50, WorkloadClass::Interactive, now, &mut seen);
            if !self.roots.contains(&idx) {
                self.roots.push(idx);
            }
        }
        for (i, n) in self.nodes.iter_mut().enumerate() {
            if n.state == State::Active && !seen.contains(&i) {
                n.state = State::Removed;
            }
        }
        self.sweep();
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the recursion carries the inherited values"
    )]
    fn sync(
        &mut self,
        cfg: &GroupCfg,
        parent: Option<usize>,
        prefix: &str,
        inherited_priority: u8,
        inherited_class: WorkloadClass,
        now: u64,
        seen: &mut std::collections::HashSet<usize>,
    ) -> usize {
        let id = Self::group_id(prefix, &cfg.name);
        let priority = cfg.priority.unwrap_or(inherited_priority);
        let class = cfg.class.unwrap_or(inherited_class);
        let existing = self
            .nodes
            .iter()
            .position(|n| n.state != State::Free && n.parent == parent && n.id == id);
        let idx = if let Some(i) = existing {
            if let Some(n) = self.nodes.get_mut(i) {
                n.state = State::Active;
                n.hard = cfg.hard_concurrency;
                n.max_queued = cfg.max_queued;
                n.policy = cfg.policy;
                n.weight = cfg.weight;
                n.soft_memory = cfg.soft_memory;
                n.per_query_memory = cfg.per_query_memory;
                n.priority = priority;
                n.class = class;
                // The window is realigned when the period changes or the quota is new; the
                // bytes counted stay while the window does.
                match (n.scan_quota, cfg.scan_quota) {
                    (_, None) => {
                        n.scan_used = 0;
                        n.window_end_ms = 0;
                    }
                    (Some(old), Some(new)) if old.period_ms == new.period_ms => {}
                    (_, Some(new)) => {
                        n.scan_used = 0;
                        n.window_end_ms = now.saturating_add(new.period_ms);
                    }
                }
                n.scan_quota = cfg.scan_quota;
            }
            i
        } else {
            let i = self.alloc(Self::make_node(
                cfg,
                id.clone(),
                parent,
                priority,
                class,
                now,
            ));
            if let Some(p) = parent.and_then(|p| self.nodes.get_mut(p)) {
                p.children.push(i);
            }
            i
        };
        seen.insert(idx);
        for child in &cfg.children {
            self.sync(child, Some(idx), &id, priority, class, now, seen);
        }
        idx
    }

    /// Frees every removed group that holds no query and has no group below it.
    fn sweep(&mut self) {
        while let Some(i) = self.nodes.iter().position(|n| {
            n.state == State::Removed && n.running == 0 && n.queued == 0 && n.children.is_empty()
        }) {
            let parent = self.nodes.get(i).and_then(|n| n.parent);
            if let Some(p) = parent.and_then(|p| self.nodes.get_mut(p)) {
                p.children.retain(|c| *c != i);
            }
            self.roots.retain(|r| *r != i);
            if let Some(n) = self.nodes.get_mut(i) {
                n.state = State::Free;
                n.id.clear();
                n.parent = None;
                n.waiting.clear();
                n.reserved.clear();
            }
            self.free.push(i);
        }
    }

    fn next_id(&mut self) -> u64 {
        self.next_id = self.next_id.wrapping_add(1);
        self.next_id
    }

    fn bump(&mut self) -> u64 {
        self.seq = self.seq.wrapping_add(1);
        self.seq
    }

    /// `leaf`, its parent, and so on up to the root.
    fn chain(&self, leaf: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut cur = Some(leaf);
        while let Some(i) = cur {
            out.push(i);
            cur = self.nodes.get(i).and_then(|n| n.parent);
        }
        out
    }

    /// Bytes the subtree's running queries report.
    fn reserved_in(&self, idx: usize) -> usize {
        let Some(n) = self.nodes.get(idx) else {
            return 0;
        };
        let own: usize = n
            .reserved
            .iter()
            .map(|(_, r)| r.load(Ordering::Relaxed))
            .sum();
        own + n
            .children
            .iter()
            .map(|c| self.reserved_in(*c))
            .sum::<usize>()
    }

    /// Every group from the leaf to the root is below its concurrency and memory limits.
    fn can_run(&self, leaf: usize) -> bool {
        self.chain(leaf).into_iter().all(|i| {
            self.nodes.get(i).is_some_and(|n| {
                n.running < n.hard && n.soft_memory.is_none_or(|m| self.reserved_in(i) < m)
            })
        })
    }

    /// The first group of the chain whose queue is full.
    fn full_queue(&self, leaf: usize) -> Option<(String, usize)> {
        self.chain(leaf).into_iter().find_map(|i| {
            self.nodes
                .get(i)
                .filter(|n| n.queued >= n.max_queued)
                .map(|n| (n.id.clone(), n.max_queued))
        })
    }

    /// Starts a window afresh when the old one ended, then reports a spent quota.
    fn quota_refusal(&mut self, leaf: usize, now: u64) -> Option<KrishivError> {
        for i in self.chain(leaf) {
            let n = self.nodes.get_mut(i)?;
            let Some(q) = n.scan_quota else {
                // No quota here; an ancestor may still have one.
                continue;
            };
            n.roll_window(now);
            if n.scan_used >= q.bytes {
                return Some(
                    KrishivResourceGroupQuotaExhausted {
                        group: n.id.clone(),
                        reset_at: utc(n.window_end_ms),
                    }
                    .into(),
                );
            }
        }
        None
    }

    fn enqueue(&mut self, leaf: usize, w: Waiter) {
        for i in self.chain(leaf) {
            if let Some(n) = self.nodes.get_mut(i) {
                n.queued += 1;
            }
        }
        if let Some(n) = self.nodes.get_mut(leaf) {
            n.waiting.push_back(w);
        }
    }

    fn remove_waiter(&mut self, leaf: usize, id: u64) {
        let found = self
            .nodes
            .get_mut(leaf)
            .and_then(|n| n.waiting.iter().position(|w| w.id == id).map(|p| (n, p)))
            .and_then(|(n, p)| n.waiting.remove(p));
        if found.is_some() {
            for i in self.chain(leaf) {
                if let Some(n) = self.nodes.get_mut(i) {
                    n.queued = n.queued.saturating_sub(1);
                }
            }
            self.sweep();
        }
    }

    /// Counts the query as running in every group of the chain.
    fn start(&mut self, shared: &Arc<Shared>, leaf: usize, queued_ns: u64) -> GroupSlot {
        let slot_id = self.next_id();
        let reserved = Arc::new(AtomicUsize::new(0));
        let seq = self.bump();
        for i in self.chain(leaf) {
            if let Some(n) = self.nodes.get_mut(i) {
                n.running += 1;
                n.last_dispatch = seq;
            }
        }
        if let Some(n) = self.nodes.get_mut(leaf) {
            n.reserved.push((slot_id, Arc::clone(&reserved)));
        }
        GroupSlot {
            shared: Arc::clone(shared),
            leaf,
            slot_id,
            reserved,
            bytes_read: AtomicU64::new(0),
            queued_ns,
        }
    }

    fn release(&mut self, leaf: usize, slot_id: u64, bytes_read: u64, now: u64) {
        if let Some(n) = self.nodes.get_mut(leaf) {
            n.reserved.retain(|(id, _)| *id != slot_id);
        }
        for i in self.chain(leaf) {
            if let Some(n) = self.nodes.get_mut(i) {
                n.running = n.running.saturating_sub(1);
                if n.scan_quota.is_some() {
                    // Start the new window first, so bytes scanned across a boundary count.
                    n.roll_window(now);
                    n.scan_used = n.scan_used.saturating_add(bytes_read);
                }
            }
        }
        self.sweep();
    }

    /// Hands every freed slot to a queued query, as far as the groups allow.
    fn dispatch(&mut self, shared: &Arc<Shared>, now: u64) -> Vec<Delivery> {
        let mut out = Vec::new();
        let roots = self.roots.clone();
        for root in roots {
            while let Some(leaf) = self.pick(root) {
                let Some(w) = self.nodes.get_mut(leaf).and_then(|n| n.waiting.pop_front()) else {
                    break;
                };
                for i in self.chain(leaf) {
                    if let Some(n) = self.nodes.get_mut(i) {
                        n.queued = n.queued.saturating_sub(1);
                    }
                }
                if w.tx.is_closed() {
                    continue;
                }
                let grant = match self.quota_refusal(leaf, now) {
                    Some(e) => Err(e),
                    None => {
                        let ns = u64::try_from(w.arrived.elapsed().as_nanos()).unwrap_or(u64::MAX);
                        Ok(self.start(shared, leaf, ns))
                    }
                };
                out.push(Delivery { tx: w.tx, grant });
            }
        }
        self.sweep();
        out
    }

    /// Whether the subtree has a queued query that could start now.
    fn eligible(&self, idx: usize) -> bool {
        let Some(n) = self.nodes.get(idx) else {
            return false;
        };
        // A group that gained subgroups in a reload keeps the queries it queued as a leaf: they
        // are dispatched from it, before the subgroups' own.
        (!n.waiting.is_empty() && self.can_run(idx)) || n.children.iter().any(|c| self.eligible(*c))
    }

    /// The leaf whose oldest query goes next, walking down by each group's policy.
    fn pick(&self, idx: usize) -> Option<usize> {
        let n = self.nodes.get(idx)?;
        if !self.eligible(idx) {
            return None;
        }
        if n.children.is_empty() || (!n.waiting.is_empty() && self.can_run(idx)) {
            return Some(idx);
        }
        match n.policy {
            Policy::WeightedFair => {
                let best = n
                    .children
                    .iter()
                    .filter_map(|c| self.nodes.get(*c).map(|node| (*c, node)))
                    .filter(|(c, _)| self.eligible(*c))
                    .min_by(|(_, a), (_, b)| {
                        // running / weight, compared without division
                        let l = (a.running as u128) * u128::from(b.weight);
                        let r = (b.running as u128) * u128::from(a.weight);
                        l.cmp(&r)
                            .then(a.last_dispatch.cmp(&b.last_dispatch))
                            .then_with(|| a.id.cmp(&b.id))
                    })?;
                self.pick(best.0)
            }
            Policy::QueryPriority => self.pick_by_priority(idx),
        }
    }

    /// The eligible leaf of the subtree with the highest priority, then the oldest head query.
    fn pick_by_priority(&self, idx: usize) -> Option<usize> {
        let mut best: Option<(u8, std::cmp::Reverse<u64>, usize)> = None;
        let mut stack = vec![idx];
        while let Some(i) = stack.pop() {
            let n = self.nodes.get(i)?;
            if let Some(head) = n.waiting.front().filter(|_| self.can_run(i)) {
                let key = (n.priority, std::cmp::Reverse(head.seq), i);
                if best.is_none_or(|b| key > b) {
                    best = Some(key);
                }
            }
            stack.extend(n.children.iter().copied());
        }
        best.map(|b| b.2)
    }
}

/// `2026-11-13T12:00:00Z` for milliseconds since the Unix epoch.
fn utc(ms: u64) -> String {
    let secs = ms / 1000;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil date from days since 1970-01-01 (Hinnant's algorithm).
    let z = i64::try_from(days).unwrap_or(0) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}
