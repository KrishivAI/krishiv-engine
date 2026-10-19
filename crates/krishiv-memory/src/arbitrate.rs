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
//!    (`limit / active queries`), then by spillable bytes; the requester's
//!    own consumer and every consumer of its driver ([`crate::SpillOwner`])
//!    are never asked. Tickets are awaited together, each completed victim
//!    is trimmed so its exact freed bytes count, and the request is retried
//!    on every release. `spillWaitTimeout` bounds the wait for a victim to
//!    *begin* its spill; a spill in progress is awaited to its end.
//! 3. **Abort.** While pending releases cover the shortfall, wait for them
//!    (no new abort). Otherwise abort the query furthest above its fair
//!    share. The requester's own query fails the request with
//!    `UNABLE_TO_ACQUIRE_MEMORY` and its other tasks see
//!    `KRISHIV_QUERY_ABORTED_OUT_OF_MEMORY`; another victim is aborted and
//!    the round waits for its release. The phase is bounded by
//!    `abortReleaseTimeout`. A task- or operator-level limit (depth ≥ 2)
//!    fails only the request, never the whole query. A request made with
//!    [`Escalation::SpillOnly`] skips this phase and fails instead: its
//!    caller has a cheaper fallback than aborting a query.

use std::future::{Future, poll_fn};
use std::pin::{Pin, pin};
use std::sync::Arc;
use std::task::Poll;

use crate::pool::{Class, Consumer, Denied, Group, GroupKind, Node};
use crate::spill::SpillTicket;
use std::time::Instant;

use crate::sync::Ordering;
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
    let round = Round {
        node,
        consumer,
        bytes,
        escalation,
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
    TicketsDone,
    Deadline,
}

/// The depth-1 subtrees involved in a round and their fair shares.
struct Plan {
    groups: Vec<Arc<Node>>,
    fair_share: usize,
    /// Bytes held by aborted queries in the subtree: they will be released.
    pending: usize,
}

impl Plan {
    fn excess(&self, group: Option<&Arc<Node>>) -> usize {
        group.map_or(0, |g| g.reserved().saturating_sub(self.fair_share))
    }
}

struct Round {
    node: Arc<Node>,
    consumer: Arc<Consumer>,
    bytes: usize,
    escalation: Escalation,
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
            .try_reserve_for(&self.consumer, self.bytes, Class::Ordinary)
        {
            Ok(()) => Ok(None),
            Err(Denied::Error(e)) => Err(e),
            Err(Denied::Limit { binding, need }) => Ok(Some((binding, need))),
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
        MemoryError::Exhausted {
            requested: self.bytes,
            available: binding.available(),
        }
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
        Plan {
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
            let shortfall = need
                .saturating_sub(binding.available())
                .saturating_sub(plan.pending);
            let tickets = self.request_spills(&consumers, &plan, shortfall);
            if tickets.is_empty() {
                break;
            }
            let all: Vec<SpillTicket> = tickets.iter().map(|(t, ..)| t.clone()).collect();
            let deadline = shared.clock.now() + shared.spill_wait_timeout;
            if let Waited::Step(s) = self.wait(slot, deadline, tickets, true).await? {
                return Ok(s);
            }
            if all.iter().filter_map(SpillTicket::try_get).sum::<usize>() == 0 {
                break;
            }
        }

        if self.escalation == Escalation::SpillOnly {
            return match self.step(slot)? {
                Step::Short(b, _) => Err(self.exhausted(&b)),
                s => Ok(s),
            };
        }

        // 3. Abort, never more than the shortfall needs.
        let deadline = shared.clock.now() + shared.abort_release_timeout;
        loop {
            let (binding, need) = match self.step(slot)? {
                Step::Short(b, need) => (b, need),
                s => return Ok(s),
            };
            let consumers = self.consumers_within(&binding);
            let plan = self.plan(&binding, &consumers);
            let shortfall = need.saturating_sub(binding.available());
            if plan.pending > 0 && plan.pending >= shortfall {
                match self.wait(slot, deadline, Vec::new(), false).await? {
                    Waited::Step(s) => return Ok(s),
                    Waited::Deadline => return Err(self.exhausted(&binding)),
                    Waited::TicketsDone => continue,
                }
            }
            if binding.depth >= 2 {
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
    ) -> Vec<(SpillTicket, Arc<Consumer>, Arc<Node>)> {
        let me = &self.consumer;
        let mut victims: Vec<_> = consumers
            .iter()
            .filter(|(c, n)| {
                c.id != me.id && (c.owner.is_none() || c.owner != me.owner) && !aborted(n)
            })
            .filter_map(|(c, n)| {
                let spill = c.spillable()?;
                let bytes = spill.spillable_bytes().min(c.used());
                let excess = plan.excess(n.group_node().as_ref());
                (bytes > 0).then(|| (excess, bytes, spill, Arc::clone(c), Arc::clone(n)))
            })
            .collect();
        victims.sort_by_key(|(excess, bytes, ..)| std::cmp::Reverse((*excess, *bytes)));
        let mut remaining = shortfall;
        let mut tickets = Vec::new();
        for (_, bytes, spill, c, n) in victims {
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
    /// `deadline`, or (with `until_tickets_done`) all tickets completed. The
    /// deadline only stops waiting for victims that have not begun their
    /// spill ([`SpillTicket::started`]): at the deadline those tickets are
    /// dropped, and the round keeps waiting, with no deadline, for the
    /// spills already in progress.
    async fn wait(
        &self,
        slot: &Node,
        deadline: Instant,
        mut tickets: Vec<(SpillTicket, Arc<Consumer>, Arc<Node>)>,
        until_tickets_done: bool,
    ) -> Result<Waited, MemoryError> {
        let shared = &self.node.shared;
        let waiter = shared.release.wait();
        let mut sleep = Some(shared.clock.sleep_until(deadline)?);
        loop {
            let mut released = waiter.listen();
            let mut aborted = self.group().map(|g| g.abort_event.listen());
            match self.step(slot)? {
                Step::Short(..) => {}
                s => return Ok(Waited::Step(s)),
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
