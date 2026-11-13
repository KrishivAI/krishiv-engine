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

/// One group's configuration and counters.
struct Node {
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
    #[expect(
        dead_code,
        reason = "the session wiring reads the class for system.queries"
    )]
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
    roots: Vec<usize>,
    seq: u64,
    next_id: u64,
}

/// Why an admission was refused before queueing.
pub(super) enum Refusal {
    Quota(KrishivError),
    QueueFull { group: String, max_queued: usize },
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

    /// The leaf index of a dotted group id.
    pub(super) fn leaf(&self, id: &str) -> Option<usize> {
        self.lock().nodes.iter().position(|n| n.id == id)
    }

    /// Admits at once when the leaf may run another query and nothing waits ahead, else queues.
    pub(super) fn admit(self: &Arc<Self>, leaf: usize) -> Result<Admitted, Refusal> {
        let now = self.now();
        let mut tree = self.lock();
        if let Some(refusal) = tree.quota_refusal(leaf, now) {
            return Err(Refusal::Quota(refusal));
        }
        let immediate =
            tree.can_run(leaf) && tree.nodes.get(leaf).is_some_and(|n| n.waiting.is_empty());
        if immediate {
            return Ok(Admitted::Now(tree.start(self, leaf, 0)));
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
        Ok(Admitted::Queued(
            rx,
            QueuedGuard {
                shared: Arc::clone(self),
                leaf,
                id,
            },
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

    /// The per-query memory budget of a leaf.
    pub(super) fn per_query_memory(&self, leaf: usize) -> Option<usize> {
        self.lock().nodes.get(leaf).and_then(|n| n.per_query_memory)
    }

    /// `(running, queued)` of a group, by dotted id.
    pub(super) fn counts(&self, id: &str) -> Option<(usize, usize)> {
        self.lock()
            .nodes
            .iter()
            .find(|n| n.id == id)
            .map(|n| (n.running, n.queued))
    }
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
        let mut tree = self.shared.lock();
        tree.remove_waiter(self.leaf, self.id);
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
    fn add(
        &mut self,
        cfg: &GroupCfg,
        parent: Option<usize>,
        prefix: &str,
        inherited_priority: u8,
        inherited_class: WorkloadClass,
        now: u64,
    ) -> usize {
        let id = if prefix.is_empty() {
            cfg.name.clone()
        } else {
            format!("{prefix}.{}", cfg.name)
        };
        let priority = cfg.priority.unwrap_or(inherited_priority);
        let class = cfg.class.unwrap_or(inherited_class);
        let idx = self.nodes.len();
        self.nodes.push(Node {
            id: id.clone(),
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
        });
        for child in &cfg.children {
            let c = self.add(child, Some(idx), &id, priority, class, now);
            if let Some(n) = self.nodes.get_mut(idx) {
                n.children.push(c);
            }
        }
        idx
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
        out
    }

    /// Whether the subtree has a queued query that could start now.
    fn eligible(&self, idx: usize) -> bool {
        let Some(n) = self.nodes.get(idx) else {
            return false;
        };
        if n.children.is_empty() {
            !n.waiting.is_empty() && self.can_run(idx)
        } else {
            n.children.iter().any(|c| self.eligible(*c))
        }
    }

    /// The leaf whose oldest query goes next, walking down by each group's policy.
    fn pick(&self, idx: usize) -> Option<usize> {
        let n = self.nodes.get(idx)?;
        if !self.eligible(idx) {
            return None;
        }
        if n.children.is_empty() {
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
            if n.children.is_empty() {
                if let Some(head) = n.waiting.front().filter(|_| self.can_run(i)) {
                    let key = (n.priority, std::cmp::Reverse(head.seq), i);
                    if best.is_none_or(|b| key > b) {
                        best = Some(key);
                    }
                }
            } else {
                stack.extend(n.children.iter().copied());
            }
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
