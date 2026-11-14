// SPDX-License-Identifier: Apache-2.0
//! The slow path: cooperative arbitration (ADR 0007 update, spec 13 §3).
//!
//! Only consumers inside a query pool arbitrate; external pools and the
//! root use the fast path only. A round is admitted per *binding* pool, the
//! pool whose limit refused the request (see [`crate::rounds`]), and works
//! inside that pool's subtree:
//! 1. **Reclaim** quantisation slack from every consumer.
//! 2. **Spill.** Request spills until the targets cover the shortfall that
//!    pending releases (memory of already-aborted queries) do not. Victims
//!    are ranked by how far their query holds memory above its fair share
//!    (`limit / active queries`), then by spillable bytes; last-resort
//!    consumers ([`Spillable::last_resort`](crate::Spillable::last_resort):
//!    reservations given up while the memory stays in use) come after all
//!    others, idle finished outputs ([`Spillable::idle`](crate::Spillable::idle))
//!    before the rest of their rank; the requester's
//!    own consumer is never asked, and the other consumers of its driver
//!    ([`crate::SpillOwner`]) only after every other victim (the driver
//!    services them while it waits; a caller that does not is given up on
//!    at `spillWaitTimeout`, as any victim that never begins). Tickets are awaited together, each completed victim
//!    is trimmed so its exact freed bytes count, and the request is retried
//!    on every release. `spillWaitTimeout` bounds the wait for a victim to
//!    *begin* its spill; a spill in progress is awaited to its end.
//!    Last-resort victims not reached by then (a pass that freed nothing
//!    ends the spilling) are asked once more before anything else.
//! 3. **Abort.** While pending releases cover the shortfall, wait for them
//!    (no new abort). The same while they and other drivers' unsettled grants
//!    ([`crate::GrantCredit::unsettled`]: granted for a retry that has not
//!    ended, so about to be used or returned) cover it, re-planning whenever
//!    memory is released or a grant settles: a consumer whose driver waits in
//!    a round of its own, and the requester's own driver, do not count.
//!    Spill writes in progress count too: the memory of another driver's
//!    consumer while it holds a [`crate::ConsumerHandle::spilling`] guard
//!    (taken only across a write that ends without help: the blocking job
//!    and the release of the credit it freed) will be released, so the
//!    round waits for it (re-planning when a write ends) instead of failing
//!    a request while no spill can be asked of anyone. Cohort members
//!    holding count the same way: another driver's consumer with a seat
//!    holding ([`crate::ConsumerHandle::join_cohort`]: a sort's merge budget,
//!    a nested loop's chunk, freed without help) of the requester's query
//!    will release its memory, unless its driver waits in a round of its
//!    own; a requester or member registered without a driver
//!    ([`crate::SpillOwner::signal`]) never waits this way. The consumer's
//!    whole use counts, as for a write.
//!    Otherwise abort the query furthest above its fair share. The requester's own query fails the request with
//!    `UNABLE_TO_ACQUIRE_MEMORY` and its other tasks see
//!    `KRISHIV_QUERY_ABORTED_OUT_OF_MEMORY`; another victim is aborted and
//!    the round waits for its release. The phase is bounded by
//!    `abortReleaseTimeout`. A task- or operator-level limit (below the top
//!    of a query) fails only the request, never the whole query; a tenant
//!    pool's limit ([`crate::TreePool::tenant`]) aborts one of its queries. A request made with
//!    [`Escalation::SpillOnly`] waits for those releases, grants and writes
//!    the same way but never aborts: it fails instead, since its caller has
//!    a cheaper fallback than aborting a query.

use std::future::{Future, poll_fn};
use std::pin::{Pin, pin};
use std::sync::Arc;
use std::task::Poll;

use crate::pool::{Class, Consumer, Denied, Group, GroupKind, Node};
use crate::spill::{OwnerId, SpillTicket};
use std::time::Instant;

use crate::sync::{AtomicBool, Ordering};
use crate::{ConsumerId, MemoryError};

/// Spill requests a round sends while each pass frees memory, before it
/// escalates.
const MAX_SPILL_PASSES: usize = 8;

/// How far a round may go to make room.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Escalation {
    /// Reclaim, spill, then abort queries.
    Abort,
    /// Reclaim and spill only.
    SpillOnly,
}

pub(crate) async fn reserve(
    node: Arc<Node>,
    id: ConsumerId,
    bytes: usize,
    escalation: Escalation,
) -> Result<(), MemoryError> {
    let consumer = node.consumer(id)?;
    let _waiting = Waiting::enter(Arc::clone(&consumer), &node.shared);
    let round = Round {
        node,
        consumer,
        bytes,
        escalation,
        unguarded: AtomicBool::new(false),
    };
    let Some((mut binding, need)) = round.attempt()? else {
        return Ok(());
    };
    let arbitrates = round
        .node
        .group
        .as_ref()
        .is_some_and(|g| g.kind == GroupKind::Query);
    if !arbitrates || binding.grant_limit().is_some_and(|l| need > l) {
        // External or root consumers never start a round; a request larger
        // than the limit could never fit, whatever is freed.
        return Err(round.exhausted(&binding));
    }
    let shared = Arc::clone(&round.node.shared);
    // Before anything is spilled or aborted: a round must be able to time out.
    shared.clock.prepare()?;
    loop {
        let _slot = race_abort(round.group(), shared.rounds.enter(Arc::clone(&binding))).await?;
        shared.stats.rounds.fetch_add(1, Ordering::Relaxed);
        match round.run(&binding).await? {
            Step::Granted => return Ok(()),
            Step::Rebind(b) | Step::Short(b, _) => binding = b,
        }
    }
}

/// [`MemoryPool::spill_idle`](crate::MemoryPool::spill_idle) of a tree
/// pool: idle consumers of the requester's query, largest first, asked for
/// `bytes` together. No round is entered: the requester's own consumers
/// have not spilled yet, so this asks nothing of anyone else's driver.
pub(crate) async fn spill_idle(node: Arc<Node>, id: ConsumerId, bytes: usize) -> usize {
    if bytes == 0 {
        return 0;
    }
    let Some(query) = node
        .consumer(id)
        .ok()
        .and_then(|c| c.node.upgrade())
        .and_then(|n| n.group_node())
        .filter(|q| q.group.as_ref().is_some_and(|g| g.kind == GroupKind::Query))
    else {
        return 0;
    };
    let mut idle: Vec<_> = node
        .shared
        .consumers()
        .into_iter()
        .filter(|c| c.id != id)
        .filter_map(|c| {
            let n = c.node.upgrade()?;
            let spill = c.spillable().filter(|s| s.idle())?;
            let held = spill.spillable_bytes().min(c.used());
            (n.is_within(&query) && !aborted(&n) && held > 0).then_some((held, spill))
        })
        .collect();
    idle.sort_by_key(|(held, _)| std::cmp::Reverse(*held));
    let mut remaining = bytes;
    let mut tickets = Vec::new();
    for (held, spill) in idle {
        if remaining == 0 {
            break;
        }
        let target = held.min(remaining);
        remaining -= target;
        node.shared
            .stats
            .spill_requests
            .fetch_add(1, Ordering::Relaxed);
        tickets.push(spill.request_spill(target));
    }
    let mut freed = 0usize;
    for t in tickets {
        freed = freed.saturating_add(t.await);
    }
    freed
}

/// Marks a consumer as waiting in arbitration while it lives: its unsettled
/// grant is not counted as in flight, and its driver's holding cohort
/// members are not waited for, since its driver is blocked here. Entering
/// wakes rounds that may be waiting for either, so they re-plan.
struct Waiting(Arc<Consumer>);

impl Waiting {
    fn enter(consumer: Arc<Consumer>, shared: &crate::pool::Shared) -> Self {
        consumer.waiting.store(true, Ordering::Release);
        // Unconditional: another consumer of this driver may hold a seat.
        shared.release.released();
        Self(consumer)
    }
}

impl Drop for Waiting {
    fn drop(&mut self) {
        self.0.waiting.store(false, Ordering::Release);
    }
}

/// Awaits `fut`, unless the group is aborted first.
pub(crate) async fn race_abort<F: Future>(
    group: Option<&Group>,
    fut: F,
) -> Result<F::Output, MemoryError> {
    let Some(group) = group else {
        return Ok(fut.await);
    };
    let mut fut = pin!(fut);
    loop {
        let mut aborted = group.abort_event.listen();
        group.check()?;
        let done = poll_fn(|cx| {
            if let Poll::Ready(v) = fut.as_mut().poll(cx) {
                return Poll::Ready(Some(v));
            }
            Pin::new(&mut aborted).poll(cx).map(|()| None)
        })
        .await;
        if let Some(v) = done {
            return Ok(v);
        }
    }
}

/// Where a retry left the request.
enum Step {
    Granted,
    /// A pool outside the round's subtree refused: re-enter there.
    Rebind(Arc<Node>),
    /// `binding` (inside the round's subtree) refused `need` bytes.
    Short(Arc<Node>, usize),
}

/// How a wait ended, short of a grant.
enum Waited {
    Step(Step),
    /// All tickets completed ([`Until::Tickets`]), or something changed
    /// ([`Until::Pending`], [`Until::Change`]): re-plan.
    TicketsDone,
    Deadline,
}

/// When a [`Round::wait`] ends, short of a grant or its deadline.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Until {
    /// Once every ticket completed.
    Tickets,
    /// While pending releases cover the shortfall: at the first release or
    /// abort the caller re-plans. A release reaches the root before the
    /// aborted query's own counter, so `pending` can count bytes the binding
    /// already got back; re-planning then finds the rest of the shortfall
    /// still to fund (another abort) instead of waiting out the deadline.
    Pending,
    /// At the first release, credit taken, ticket or abort: the caller
    /// re-plans.
    Change,
}

/// The groups (queries, external pools) involved in a round and their fair
/// shares.
struct Plan {
    groups: Vec<Arc<Node>>,
    fair_share: usize,
    /// Bytes held by aborted queries in the subtree: they will be released.
    pending: usize,
    /// Unsettled grants of other drivers' consumers in the subtree (see
    /// [`Consumer::unsettled`]): about to be used or returned.
    credit: usize,
    /// Bytes of other consumers with a spill write in progress
    /// ([`crate::ConsumerHandle::spilling`]): about to be released.
    writing: usize,
    /// Bytes of other drivers' consumers of the requester's query with a
    /// cohort seat holding ([`crate::ConsumerHandle::join_cohort`]): memory
    /// the member frees on its own (a sort's merge budget, a nested loop's
    /// chunk). A driver waiting in a round of its own is left out: its
    /// member may need that round's memory before it frees anything.
    holding: usize,
}

impl Plan {
    /// Pending releases, unsettled grants, spill writes in progress and
    /// holding cohort members cover `shortfall`, and some of it is grants,
    /// writes or members (a wait for pending releases alone is the abort
    /// phase's own).
    fn covered_by_releases(&self, shortfall: usize) -> bool {
        (self.credit > 0 || self.writing > 0 || self.holding > 0)
            && self
                .pending
                .saturating_add(self.credit)
                .saturating_add(self.writing)
                .saturating_add(self.holding)
                >= shortfall
    }
    /// Releases of already-aborted queries alone cover `shortfall`.
    fn covered_by_pending(&self, shortfall: usize) -> bool {
        self.pending > 0 && self.pending >= shortfall
    }
    fn excess(&self, group: Option<&Arc<Node>>) -> usize {
        group.map_or(0, |g| g.reserved().saturating_sub(self.fair_share))
    }
}

struct Round {
    node: Arc<Node>,
    consumer: Arc<Consumer>,
    bytes: usize,
    escalation: Escalation,
    /// Set once spilling is exhausted: the round then asks without the
    /// resident-set guard's cut, so the cut never fails or aborts an arbitrated request.
    unguarded: AtomicBool,
}

fn aborted(n: &Node) -> bool {
    n.group.as_ref().is_some_and(|g| g.is_aborted())
}

impl Round {
    fn group(&self) -> Option<&Group> {
        self.node.group.as_deref()
    }

    /// `None` when granted, else the binding pool and the bytes it refused.
    fn attempt(&self) -> Result<Option<(Arc<Node>, usize)>, MemoryError> {
        match self
            .node
            .try_reserve_for(&self.consumer, self.bytes, self.class())
        {
            Ok(()) => Ok(None),
            Err(Denied::Error(e)) => Err(e),
            Err(Denied::Limit { binding, need }) => Ok(Some((binding, need))),
        }
    }

    /// The ceiling the round's requests are checked against.
    fn class(&self) -> Class {
        if self.unguarded.load(Ordering::Acquire) {
            Class::Unguarded
        } else {
            Class::Ordinary
        }
    }

    fn step(&self, slot: &Node) -> Result<Step, MemoryError> {
        Ok(match self.attempt()? {
            None => Step::Granted,
            Some((b, _)) if !b.is_within(slot) => Step::Rebind(b),
            Some((b, need)) => Step::Short(b, need),
        })
    }

    fn exhausted(&self, binding: &Node) -> MemoryError {
        if tracing::enabled!(tracing::Level::DEBUG) {
            self.trace_exhausted(binding);
        }
        MemoryError::Exhausted {
            requested: self.bytes,
            available: binding.available(),
        }
    }

    /// What held `binding` when this round gave up: one debug event with
    /// every live consumer below it (used and spillable bytes, unsettled
    /// grant, whether its driver waits in a round), for diagnosing a refusal
    /// no spill could cover.
    fn trace_exhausted(&self, binding: &Node) {
        use std::fmt::Write as _;
        let mut held = String::new();
        for (c, _) in self.consumers_within(binding) {
            let spillable = c.spillable().map(|s| s.spillable_bytes());
            let _ = write!(
                held,
                "[{} used={} spillable={spillable:?} unsettled={} waiting={} writing={} holding={}] ",
                c.name(),
                c.used(),
                c.unsettled(),
                c.waiting.load(Ordering::Acquire),
                c.writing(),
                c.cohort_holding(),
            );
        }
        tracing::debug!(
            requester = self.consumer.name(),
            requested = self.bytes,
            binding_reserved = binding.reserved(),
            binding_available = binding.available(),
            %held,
            "memory arbitration exhausted"
        );
    }

    /// Live consumers in `binding`'s subtree with their nodes.
    fn consumers_within(&self, binding: &Node) -> Vec<(Arc<Consumer>, Arc<Node>)> {
        self.node
            .shared
            .consumers()
            .into_iter()
            .filter_map(|c| {
                let n = c.node.upgrade()?;
                n.is_within(binding).then_some((c, n))
            })
            .collect()
    }

    fn plan(&self, binding: &Arc<Node>, consumers: &[(Arc<Consumer>, Arc<Node>)]) -> Plan {
        let mut groups: Vec<Arc<Node>> = Vec::new();
        let candidates = consumers
            .iter()
            .filter_map(|(_, n)| n.group_node())
            .chain(self.node.group_node());
        for g in candidates {
            if g.is_within(binding) && !groups.iter().any(|x| Arc::ptr_eq(x, &g)) {
                groups.push(g);
            }
        }
        let active = groups
            .iter()
            .filter(|g| !aborted(g) && g.reserved() > 0)
            .count()
            .max(1);
        let me = &self.consumer;
        let credit = consumers
            .iter()
            .filter(|(c, n)| {
                c.id != me.id
                    && !(c.owner.is_some() && c.owner == me.owner)
                    && !c.waiting.load(Ordering::Acquire)
                    && !aborted(n)
            })
            .map(|(c, _)| c.unsettled())
            .sum();
        // A write ends without help, even one its driver started while it
        // waits in a round of its own (a self-spill).
        let writing: usize = consumers
            .iter()
            .filter(|(c, n)| {
                c.id != me.id
                    && !(c.owner.is_some() && c.owner == me.owner)
                    && c.writing()
                    && !aborted(n)
            })
            .map(|(c, _)| c.used())
            .sum();
        // A cohort member frees its memory on its own, unless its driver
        // waits in a round (which may be this round's wait, the other way)
        // or is the requester's (which pulls it). Both drivers must be
        // known: a consumer registered without one could be either. Cohorts
        // are per query: only the requester's query's members count.
        let holding: usize = match (me.owner, self.node.group_node()) {
            (Some(own), Some(query)) => {
                // Drivers waiting anywhere, not only in `binding`: a member's
                // driver may wait in a round for a consumer of a sibling pool.
                let waiting: Vec<OwnerId> = self
                    .node
                    .shared
                    .consumers()
                    .into_iter()
                    .filter(|c| c.waiting.load(Ordering::Acquire))
                    .filter_map(|c| c.owner)
                    .collect();
                consumers
                    .iter()
                    .filter(|(c, n)| {
                        c.owner.is_some_and(|o| o != own && !waiting.contains(&o))
                            && c.cohort_holding()
                            // Counted as a write already.
                            && !c.writing()
                            && !aborted(n)
                            && n.group_node().is_some_and(|g| Arc::ptr_eq(&g, &query))
                    })
                    .map(|(c, _)| c.used())
                    .sum()
            }
            _ => 0,
        };
        Plan {
            credit,
            writing,
            holding,
            fair_share: binding.limit().unwrap_or(usize::MAX) / active,
            pending: groups
                .iter()
                .filter(|g| aborted(g))
                .map(|g| g.reserved())
                .sum(),
            groups,
        }
    }

    async fn run(&self, slot: &Arc<Node>) -> Result<Step, MemoryError> {
        let shared = Arc::clone(&self.node.shared);

        // 1. Reclaim quantisation slack.
        let binding = match self.step(slot)? {
            Step::Short(b, _) => b,
            s => return Ok(s),
        };
        for (c, n) in self.consumers_within(&binding) {
            c.trim(&n);
        }

        // 2. Spill what pending releases do not cover. Bytes a spill frees
        // return to the pool, where other consumers' fast path may take
        // them before this request does: while spills keep freeing memory,
        // ask again rather than escalate (a bounded number of times).
        for _ in 0..MAX_SPILL_PASSES {
            let (binding, need) = match self.step(slot)? {
                Step::Short(b, need) => (b, need),
                s => return Ok(s),
            };
            let consumers = self.consumers_within(&binding);
            let plan = self.plan(&binding, &consumers);
            let shortfall = binding
                .shortfall(need, self.class())
                .saturating_sub(plan.pending);
            let tickets = self.request_spills(&consumers, &plan, shortfall, false);
            if tickets.is_empty() {
                break;
            }
            let all: Vec<SpillTicket> = tickets.iter().map(|(t, ..)| t.clone()).collect();
            let deadline = shared.clock.now() + shared.spill_wait_timeout;
            if let Waited::Step(s) = self.wait(slot, deadline, tickets, Until::Tickets).await? {
                return Ok(s);
            }
            if all.iter().filter_map(SpillTicket::try_get).sum::<usize>() == 0 {
                break;
            }
        }

        // A pass ends when spills freed nothing, which can come before the
        // last-resort victims were reached: ask them now, before failing or
        // aborting anything.
        let (binding, need) = match self.step(slot)? {
            Step::Short(b, need) => (b, need),
            s => return Ok(s),
        };
        let consumers = self.consumers_within(&binding);
        let plan = self.plan(&binding, &consumers);
        let shortfall = binding
            .shortfall(need, self.class())
            .saturating_sub(plan.pending);
        let tickets = self.request_spills(&consumers, &plan, shortfall, true);
        if !tickets.is_empty() {
            let deadline = shared.clock.now() + shared.spill_wait_timeout;
            if let Waited::Step(s) = self.wait(slot, deadline, tickets, Until::Tickets).await? {
                return Ok(s);
            }
        }

        // Spilling is exhausted. The resident-set guard's cut only makes
        // queries spill: from here on, ask without it.
        self.unguarded.store(true, Ordering::Release);

        // 3. Abort, never more than the shortfall needs (a request made
        // with `SpillOnly` waits the same way, then fails instead).
        let deadline = shared.clock.now() + shared.abort_release_timeout;
        loop {
            let (binding, need) = match self.step(slot)? {
                Step::Short(b, need) => (b, need),
                s => return Ok(s),
            };
            let consumers = self.consumers_within(&binding);
            let plan = self.plan(&binding, &consumers);
            let shortfall = binding.shortfall(need, self.class());
            if shortfall == 0 {
                // Released since the refused attempt (the victim's release
                // may have funded it): ask again before aborting anything.
                // Each pass re-reads the counters only; a shortfall of 0
                // means room was free at that read, so this cannot spin
                // without other rounds taking that room each time.
                continue;
            }
            if plan.covered_by_pending(shortfall) {
                match self
                    .wait(slot, deadline, Vec::new(), Until::Pending)
                    .await?
                {
                    Waited::Step(s) => return Ok(s),
                    Waited::Deadline => return Err(self.exhausted(&binding)),
                    Waited::TicketsDone => continue,
                }
            }
            // Other drivers' grants are about to be used or returned, or
            // their spill writes about to free memory: wait for them,
            // re-planning on every change, rather than fail or abort while
            // a retry or a write is about to free it.
            if plan.covered_by_releases(shortfall) {
                match self.wait(slot, deadline, Vec::new(), Until::Change).await? {
                    Waited::Step(s) => return Ok(s),
                    Waited::Deadline => return Err(self.exhausted(&binding)),
                    Waited::TicketsDone => continue,
                }
            }
            if self.escalation == Escalation::SpillOnly || binding.is_inside_group() {
                return Err(self.exhausted(&binding));
            }
            let own = self.node.group_node();
            let Some(victim) = self.abort_victim(&plan, own.as_ref()) else {
                return Err(self.exhausted(&binding));
            };
            shared.stats.aborts.fetch_add(1, Ordering::Relaxed);
            victim.abort(MemoryError::Aborted {
                held: victim.reserved(),
            });
            if own.as_ref().is_some_and(|o| Arc::ptr_eq(o, &victim)) {
                return Err(self.exhausted(&binding));
            }
        }
    }

    /// The query furthest above its fair share that is not aborted and holds
    /// memory; ties go to the requester's own query.
    fn abort_victim(&self, plan: &Plan, own: Option<&Arc<Node>>) -> Option<Arc<Node>> {
        let key = |g: &Arc<Node>| {
            let is_own = own.is_some_and(|o| Arc::ptr_eq(o, g));
            (plan.excess(Some(g)), g.reserved(), is_own)
        };
        plan.groups
            .iter()
            .filter(|g| {
                g.group.as_ref().is_some_and(|x| x.kind == GroupKind::Query)
                    && !aborted(g)
                    && g.reserved() > 0
            })
            .max_by_key(|g| key(g))
            .cloned()
    }

    fn request_spills(
        &self,
        consumers: &[(Arc<Consumer>, Arc<Node>)],
        plan: &Plan,
        shortfall: usize,
        last_resort_only: bool,
    ) -> Vec<(SpillTicket, Arc<Consumer>, Arc<Node>)> {
        let me = &self.consumer;
        let mut victims: Vec<_> = consumers
            .iter()
            .filter(|(c, n)| c.id != me.id && !aborted(n))
            .filter_map(|(c, n)| {
                let spill = c
                    .spillable()
                    .filter(|s| !last_resort_only || s.last_resort())?;
                let bytes = spill.spillable_bytes().min(c.used());
                let excess = plan.excess(n.group_node().as_ref());
                // The requester's own driver services requests while it
                // waits, but has already spilled what `SpillOwner::reserve`
                // asked of it: its other consumers are asked last.
                let own = c.owner.is_some() && c.owner == me.owner;
                // Consumers that only give up reservations (memory still in
                // use) come after every real spill.
                // An idle finished output first (`Spillable::idle`).
                let rank = (spill.last_resort(), own, !spill.idle());
                (bytes > 0).then(|| (rank, excess, bytes, spill, Arc::clone(c), Arc::clone(n)))
            })
            .collect();
        victims
            .sort_by_key(|(rank, excess, bytes, ..)| (*rank, std::cmp::Reverse((*excess, *bytes))));
        let mut remaining = shortfall;
        let mut tickets = Vec::new();
        for (_, _, bytes, spill, c, n) in victims {
            if remaining == 0 {
                break;
            }
            let target = bytes.min(remaining);
            remaining -= target;
            self.node
                .shared
                .stats
                .spill_requests
                .fetch_add(1, Ordering::Relaxed);
            tickets.push((spill.request_spill(target), c, n));
        }
        tickets
    }

    /// Retries on every release, ticket completion or abort until granted,
    /// `deadline`, or as `until` says (all tickets completed, or any
    /// change). The deadline only stops waiting for victims that have not begun their
    /// spill ([`SpillTicket::started`]): at the deadline those tickets are
    /// dropped, and the round keeps waiting, with no deadline, for the
    /// spills already in progress.
    async fn wait(
        &self,
        slot: &Node,
        deadline: Instant,
        mut tickets: Vec<(SpillTicket, Arc<Consumer>, Arc<Node>)>,
        until: Until,
    ) -> Result<Waited, MemoryError> {
        let until_tickets_done = until == Until::Tickets;
        let shared = &self.node.shared;
        let waiter = shared.release.wait();
        let mut sleep = Some(shared.clock.sleep_until(deadline)?);
        loop {
            let mut released = waiter.listen();
            let mut aborted = self.group().map(|g| g.abort_event.listen());
            let short = match self.step(slot)? {
                Step::Short(b, need) => (b, need),
                s => return Ok(Waited::Step(s)),
            };
            // The caller planned before this listener existed: a grant
            // settled in between notified no one, so check it again now.
            if matches!(until, Until::Pending | Until::Change) {
                let (binding, need) = short;
                let plan = self.plan(&binding, &self.consumers_within(&binding));
                let shortfall = binding.shortfall(need, self.class());
                let covered = if until == Until::Pending {
                    plan.covered_by_pending(shortfall)
                } else {
                    plan.covered_by_releases(shortfall)
                };
                if !covered {
                    return Ok(Waited::TicketsDone);
                }
            }
            if until_tickets_done && tickets.is_empty() {
                return Ok(Waited::TicketsDone);
            }
            let woke = poll_fn(|cx| {
                let mut ready = false;
                let mut i = 0;
                while i < tickets.len() {
                    if Pin::new(&mut tickets[i].0).poll(cx).is_ready() {
                        let (_, c, n) = tickets.swap_remove(i);
                        // Count exactly what the spill freed, sub-quantum too.
                        c.trim(&n);
                        ready = true;
                    } else {
                        i += 1;
                    }
                }
                ready |= Pin::new(&mut released).poll(cx).is_ready();
                if let Some(a) = aborted.as_mut() {
                    ready |= Pin::new(a).poll(cx).is_ready();
                }
                if ready {
                    return Poll::Ready(true);
                }
                match sleep.as_mut() {
                    Some(s) => s.as_mut().poll(cx).map(|()| false),
                    None => Poll::Pending,
                }
            })
            .await;
            if woke && matches!(until, Until::Pending | Until::Change) {
                return Ok(match self.step(slot)? {
                    Step::Short(..) => Waited::TicketsDone,
                    s => Waited::Step(s),
                });
            }
            if woke {
                continue;
            }
            // The deadline: give up on victims that never began.
            sleep = None;
            tickets.retain(|(t, ..)| t.started());
            if tickets.is_empty() || !until_tickets_done {
                return Ok(match self.step(slot)? {
                    Step::Short(..) => Waited::Deadline,
                    s => Waited::Step(s),
                });
            }
        }
    }
}
