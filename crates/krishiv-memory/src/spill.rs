// SPDX-License-Identifier: Apache-2.0
//! The cooperative spill protocol (ADR 0007 update, spec 13 §3).
//!
//! The arbitrator never touches another task's operator state. It calls
//! [`Spillable::request_spill`], which only sets a flag, wakes the owning
//! driver and hands back a [`SpillTicket`]. The driver services the request
//! on its own task: at its next batch boundary, or at once if it is waiting
//! for memory itself ([`SpillOwner::reserve`]). It spills, shrinks its
//! reservations and completes the ticket.
//!
//! **Self-spill contract.** The arbitrator never asks the requester's own
//! driver to spill (that driver is busy waiting). A driver whose operator
//! needs memory therefore spills its *own* consumers first, then asks the
//! arbitrator: [`SpillOwner::reserve`] does both. Sync operator code never
//! spills or waits inline; on a failed `try_grow` it returns `NeedMemory`
//! with its state unchanged, and the driver calls [`SpillOwner::reserve`].

use std::fmt;
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::sync::{Arc, Weak};
use std::task::{Context, Poll, Waker};

use crate::event::Event;
use crate::sync::{AtomicUsize, Mutex, Ordering, lock};
use crate::{ConsumerHandle, ConsumerId, MemoryError, MemoryPool};

/// A memory consumer that can give memory back on request: hash join, hash
/// aggregate, sort and window, and external consumers such as a UDF worker
/// pool. Every method is callable from any task and must not block; the
/// spill itself runs later, on the owner's task.
pub trait Spillable: Send + Sync {
    /// Bytes the owner could free by spilling: an atomic snapshot the owner
    /// maintains.
    fn spillable_bytes(&self) -> usize;
    /// Asks the owner to free at least `target` bytes. Non-blocking.
    /// Requests that arrive before the owner services the first coalesce:
    /// the largest target wins and every caller's ticket completes together.
    fn request_spill(&self, target: usize) -> SpillTicket;
    /// The pending target, if any. Drivers call this at every batch
    /// boundary, so it must be cheap (one atomic load for [`SpillSignal`]).
    fn spill_requested(&self) -> Option<usize>;
    /// The driver that services this consumer's requests, if any. The
    /// arbitrator never sends requests to the requester's own driver.
    fn owner(&self) -> Option<OwnerId> {
        None
    }
}

#[derive(Debug, Default)]
struct TicketState {
    freed: Option<usize>,
    wakers: Vec<Waker>,
}

/// Completion slot shared by every ticket of one request generation.
#[derive(Debug, Default)]
struct TicketCell {
    state: Mutex<TicketState>,
}

impl TicketCell {
    /// Called once per cell: [`SpillSignal::complete`] takes the cell out of
    /// the signal, and [`SpillTicket::ready`] completes a fresh one.
    fn complete(&self, freed: usize) {
        let wakers = {
            let mut s = lock(&self.state);
            s.freed = Some(freed);
            std::mem::take(&mut s.wakers)
        };
        for w in wakers {
            w.wake();
        }
    }
}

/// Resolves to the bytes the owner freed for a spill request. Dropping it
/// withdraws nothing (the owner still spills); it only stops waiting.
#[derive(Clone)]
pub struct SpillTicket {
    cell: Arc<TicketCell>,
}

impl SpillTicket {
    /// A ticket that is already complete, for consumers that free memory
    /// synchronously inside `request_spill` or have nothing to free.
    pub fn ready(freed: usize) -> Self {
        let cell = Arc::new(TicketCell::default());
        cell.complete(freed);
        Self { cell }
    }

    /// The bytes freed, if the owner has completed the request.
    pub fn try_get(&self) -> Option<usize> {
        lock(&self.cell.state).freed
    }
}

impl fmt::Debug for SpillTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillTicket")
            .field("freed", &self.try_get())
            .finish()
    }
}

impl Future for SpillTicket {
    type Output = usize;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<usize> {
        let mut s = lock(&self.cell.state);
        if let Some(freed) = s.freed {
            return Poll::Ready(freed);
        }
        if !s.wakers.iter().any(|w| w.will_wake(cx.waker())) {
            s.wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

#[derive(Debug, Default)]
struct Pending {
    target: usize,
    cell: Option<Arc<TicketCell>>,
}

/// A ready-made [`Spillable`] for operators. Create it with
/// [`SpillOwner::signal`] so requests wake the driver; the owner keeps
/// [`set_spillable_bytes`](Self::set_spillable_bytes) current and calls
/// [`complete`](Self::complete) after spilling.
///
/// The pool holds it weakly: once the operator drops it, the consumer is
/// never asked again, and an outstanding request completes with 0 bytes.
#[derive(Debug, Default)]
pub struct SpillSignal {
    spillable: AtomicUsize,
    /// Mirror of `pending.target` for the lock-free check; 0 = no request.
    requested: AtomicUsize,
    pending: Mutex<Pending>,
    owner: Option<(OwnerId, Weak<SpillOwner>)>,
}

impl SpillSignal {
    /// A signal with no owning driver: requests are only seen by polling
    /// [`spill_requested`](Spillable::spill_requested).
    pub fn new() -> Self {
        Self::default()
    }

    /// Owner side: publishes how many bytes a spill could free.
    pub fn set_spillable_bytes(&self, bytes: usize) {
        self.spillable.store(bytes, Ordering::Release);
    }

    /// Owner side: the pending request is done (`freed` may be 0 if nothing
    /// could be spilled). Clears the request and completes every ticket of
    /// it; a request that arrives afterwards starts a new generation.
    pub fn complete(&self, freed: usize) {
        let cell = {
            let mut p = lock(&self.pending);
            p.target = 0;
            self.requested.store(0, Ordering::Release);
            p.cell.take()
        };
        if let Some(cell) = cell {
            cell.complete(freed);
        }
    }
}

impl Spillable for SpillSignal {
    fn spillable_bytes(&self) -> usize {
        self.spillable.load(Ordering::Acquire)
    }

    fn request_spill(&self, target: usize) -> SpillTicket {
        if target == 0 {
            return SpillTicket::ready(0);
        }
        let ticket = {
            let mut p = lock(&self.pending);
            p.target = p.target.max(target);
            self.requested.store(p.target, Ordering::Release);
            let cell = Arc::clone(p.cell.get_or_insert_with(Arc::default));
            SpillTicket { cell }
        };
        if let Some(owner) = self.owner.as_ref().and_then(|(_, o)| o.upgrade()) {
            owner.requested.notify();
        }
        ticket
    }

    fn spill_requested(&self) -> Option<usize> {
        let target = self.requested.load(Ordering::Acquire);
        (target != 0).then_some(target)
    }

    fn owner(&self) -> Option<OwnerId> {
        self.owner.as_ref().map(|(id, _)| *id)
    }
}

impl Drop for SpillSignal {
    fn drop(&mut self) {
        self.complete(0);
    }
}

/// Identifies a [`SpillOwner`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OwnerId(u64);

static NEXT_OWNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The driver side of the protocol: one per pipeline driver. It creates the
/// [`SpillSignal`]s of the consumers the driver owns, is woken when any of
/// them receives a request, and runs the self-spill-then-arbitrate sequence.
///
/// Driver loop (at every batch boundary):
/// ```text
/// pool.check_aborted()?;                       // one atomic load
/// if owner.pending() { spill_flagged_consumers().await }   // one load per signal
/// match operator.execute(batch, state)? {
///     NeedMemory { bytes } => owner.reserve(&*pool, id, bytes, || spill_flagged_consumers()).await?,
///     ...
/// }
/// ```
/// where `spill_flagged_consumers` spills every own signal whose
/// [`spill_requested`](Spillable::spill_requested) is set, shrinks its
/// reservations and calls [`SpillSignal::complete`].
#[derive(Debug)]
pub struct SpillOwner {
    id: OwnerId,
    requested: Event,
    signals: Mutex<Vec<Weak<SpillSignal>>>,
}

impl SpillOwner {
    /// A new owner (driver).
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            id: OwnerId(NEXT_OWNER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
            requested: Event::default(),
            signals: Mutex::new(Vec::new()),
        })
    }

    /// This owner's id.
    pub fn id(&self) -> OwnerId {
        self.id
    }

    /// A signal for a consumer this driver owns. Register every consumer of
    /// the driver with one (a consumer that cannot spill simply never sets
    /// spillable bytes), so the arbitrator knows which driver it belongs to.
    pub fn signal(self: &Arc<Self>) -> Arc<SpillSignal> {
        let signal = Arc::new(SpillSignal {
            spillable: AtomicUsize::new(0),
            requested: AtomicUsize::new(0),
            pending: Mutex::new(Pending::default()),
            owner: Some((self.id, Arc::downgrade(self))),
        });
        let mut signals = lock(&self.signals);
        signals.retain(|s| s.strong_count() > 0);
        signals.push(Arc::downgrade(&signal));
        signal
    }

    fn live_signals(&self) -> Vec<Arc<SpillSignal>> {
        lock(&self.signals)
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }

    /// Whether any of this owner's signals has a pending request.
    pub fn pending(&self) -> bool {
        // No allocation: drivers call this at every batch boundary.
        lock(&self.signals)
            .iter()
            .any(|s| s.upgrade().is_some_and(|s| s.spill_requested().is_some()))
    }

    /// Completes after the next request to any of this owner's signals.
    /// Create it before checking [`pending`](Self::pending) so a request in
    /// between is not missed.
    pub fn requested(&self) -> impl Future<Output = ()> + Send + '_ {
        self.requested.listen()
    }

    /// [`reserve`](Self::reserve) for a [`ConsumerHandle`], parking the
    /// granted bytes as its grant credit in the same step, so a caller
    /// cancelled after the grant never leaves bytes untracked.
    ///
    /// # Errors
    /// As [`MemoryPool::reserve`].
    pub async fn reserve_for<F, Fut>(
        &self,
        consumer: &ConsumerHandle,
        bytes: usize,
        service: F,
    ) -> Result<(), MemoryError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = ()>,
    {
        self.reserve(&**consumer.pool(), consumer.id(), bytes, service)
            .await?;
        consumer.add_credit(bytes);
        Ok(())
    }

    /// Reserves `bytes` for consumer `id` of `pool` the cooperative way:
    /// 1. the fast path;
    /// 2. **self-spill**: flag this driver's own consumers (largest
    ///    spillable first) to cover the shortfall, run `service`, retry;
    /// 3. arbitration ([`MemoryPool::reserve`]), running `service` whenever
    ///    another round asks one of this driver's consumers to spill.
    ///
    /// `service` spills every own consumer whose request flag is set and
    /// completes its signal; it runs on the caller's task.
    ///
    /// # Errors
    /// As [`MemoryPool::reserve`].
    pub async fn reserve<F, Fut>(
        &self,
        pool: &dyn MemoryPool,
        id: ConsumerId,
        bytes: usize,
        mut service: F,
    ) -> Result<(), MemoryError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = ()>,
    {
        let shortfall = match pool.try_reserve(id, bytes) {
            Ok(()) => return Ok(()),
            Err(MemoryError::Exhausted { available, .. }) => bytes.saturating_sub(available),
            Err(e) => return Err(e),
        };
        let mut own = self.live_signals();
        own.sort_by_key(|s| std::cmp::Reverse(s.spillable_bytes()));
        let mut remaining = shortfall;
        for s in &own {
            if remaining == 0 {
                break;
            }
            let target = s.spillable_bytes().min(remaining);
            if target > 0 {
                remaining -= target;
                drop(s.request_spill(target));
            }
        }
        drop(own);
        if remaining < shortfall {
            service().await;
            if pool.try_reserve(id, bytes).is_ok() {
                return Ok(());
            }
        }
        let mut arbitration = pool.reserve(id, bytes);
        loop {
            let mut requested = self.requested.listen();
            if self.pending() {
                service().await;
                continue;
            }
            let done = poll_fn(|cx| {
                if let Poll::Ready(r) = arbitration.as_mut().poll(cx) {
                    return Poll::Ready(Some(r));
                }
                Pin::new(&mut requested).poll(cx).map(|()| None)
            })
            .await;
            match done {
                Some(r) => return r,
                None => service().await,
            }
        }
    }
}
