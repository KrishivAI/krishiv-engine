// SPDX-License-Identifier: Apache-2.0
//! [`ScanGate`]: the decode working sets of one query's scan streams
//! (ADR 0007).
//!
//! A stream's [`Reservation`](krishiv_memory::Reservation) covers its
//! prefetch. What its decoder holds, the bytes it buffers and the unit it
//! decodes ([`Decoder::working_bytes`](crate::Decoder::working_bytes): a
//! Parquet row group's chunks handed to its reader, decompressed pages, the
//! batch being built), is the stream's *share* of the gate, reserved
//! through one consumer of the query's pool that the gate registers:
//!
//! - **Admission.** A stream grows its share for a unit before the unit's
//!   bytes are fetched. It is granted at once while the shares stay within the
//!   gate's budget (a part of the query's memory limit) and the pool has
//!   room. Otherwise, while another stream holds a share, the stream waits
//!   for a share to be returned: the query's decode concurrency shrinks to
//!   what its memory allows. A stream with no other to wait for asks the
//!   pool with spilling but no abort
//!   ([`MemoryPool::reserve_without_abort`]), and if that cannot make room
//!   it goes on without the bytes, counted as over-committed.
//! - **Revocation.** The shares are a last-resort spill victim
//!   ([`Spillable::last_resort`]): when spilling everything else cannot
//!   fund a request, the arbitrator revokes them, at once and without I/O,
//!   and they become over-committed until their units end. A share never
//!   makes the arbitrator abort a query: it gives way first.
//! - **End.** A share is returned at its unit's end, and waiting streams
//!   are woken.
//!
//! Over-committed bytes are what the pool does not see. Admission keeps
//! them to about the budget, or one unit when a single unit is larger (a
//! stream that already holds a unit's bytes grows its share without
//! waiting), and they are reported ([`ScanGate::overcommitted`]). The
//! budget is a share of the query's limit rather than a per-driver
//! constant, so it scales with the limit.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use krishiv_memory::{ConsumerId, MemoryError, MemoryPool, SpillTicket, Spillable};

/// One stream's share: reserved in the pool and over-committed bytes.
#[derive(Debug, Default, Clone, Copy)]
struct Share {
    reserved: usize,
    over: usize,
    /// The stream asks the pool for a share with other streams waiting
    /// meanwhile.
    claim: bool,
}

impl Share {
    fn bytes(self) -> usize {
        self.reserved.saturating_add(self.over)
    }

    fn active(self) -> bool {
        self.bytes() > 0 || self.claim
    }
}

#[derive(Debug, Default)]
struct State {
    shares: BTreeMap<u64, Share>,
    /// Streams with an active share.
    active: usize,
    /// Bytes of all shares.
    bytes: usize,
}

impl State {
    /// Replaces stream `id`'s share, keeping the totals.
    fn set(&mut self, id: u64, share: Share) {
        let old = self.shares.get(&id).copied().unwrap_or_default();
        self.bytes = self
            .bytes
            .saturating_sub(old.bytes())
            .saturating_add(share.bytes());
        match (old.active(), share.active()) {
            (false, true) => self.active += 1,
            (true, false) => self.active -= 1,
            _ => {}
        }
        if share.active() {
            self.shares.insert(id, share);
        } else {
            self.shares.remove(&id);
        }
    }

    fn get(&self, id: u64) -> Share {
        self.shares.get(&id).copied().unwrap_or_default()
    }
}

/// The gate's state, which its spill hook reaches weakly.
#[derive(Debug)]
struct Inner {
    /// The query's pool, and the gate's consumer in it.
    pool: Arc<dyn MemoryPool>,
    id: ConsumerId,
    /// The consumer's spill hook (the pool holds it weakly).
    _hook: Arc<Revoke>,
    budget: usize,
    state: Mutex<State>,
    /// Bytes reserved in the pool for all shares (the spill hook's
    /// snapshot).
    reserved: AtomicUsize,
    released: tokio::sync::Notify,
    overcommitted: AtomicU64,
    revoked: AtomicU64,
}

impl Inner {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn try_reserve(&self, bytes: usize) -> bool {
        self.pool.try_reserve(self.id, bytes).is_ok()
    }

    fn release(&self, bytes: usize) {
        if bytes > 0 {
            self.pool.release(self.id, bytes);
        }
    }

    fn over_committed(&self, bytes: usize) {
        self.overcommitted
            .fetch_add(u64::try_from(bytes).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    /// Sets `id`'s share, keeping the spill hook's snapshot, and wakes the
    /// waiting streams if bytes were returned.
    fn set(&self, st: &mut State, id: u64, share: Share) {
        let old = st.get(id);
        st.set(id, share);
        let reserved = st.shares.values().map(|s| s.reserved).sum();
        self.reserved.store(reserved, Ordering::Release);
        if share.bytes() < old.bytes() || (old.active() && !share.active()) {
            self.released.notify_waiters();
        }
    }
}

/// The spill hook: revokes reserved shares (module docs).
#[derive(Debug)]
struct Revoke(Weak<Inner>);

impl Spillable for Revoke {
    fn spillable_bytes(&self) -> usize {
        self.0
            .upgrade()
            .map_or(0, |i| i.reserved.load(Ordering::Acquire))
    }

    fn request_spill(&self, target: usize) -> SpillTicket {
        let Some(inner) = self.0.upgrade() else {
            return SpillTicket::ready(0);
        };
        let mut st = inner.state();
        let mut ids: Vec<(usize, u64)> = st
            .shares
            .iter()
            .filter(|(_, s)| s.reserved > 0)
            .map(|(&id, s)| (s.reserved, id))
            .collect();
        // Largest first: the fewest streams over-commit.
        ids.sort_unstable_by(|a, b| b.cmp(a));
        let mut freed = 0usize;
        for (_, id) in ids {
            if freed >= target {
                break;
            }
            let mut share = st.get(id);
            let take = share.reserved;
            share.reserved = 0;
            share.over = share.over.saturating_add(take);
            inner.set(&mut st, id, share);
            freed = freed.saturating_add(take);
        }
        drop(st);
        inner.release(freed);
        inner.over_committed(freed);
        inner
            .revoked
            .fetch_add(u64::try_from(freed).unwrap_or(u64::MAX), Ordering::Relaxed);
        SpillTicket::ready(freed)
    }

    fn spill_requested(&self) -> Option<usize> {
        None
    }

    fn last_resort(&self) -> bool {
        true
    }
}

/// A stream's claim while it asks the pool for a share; dropping it
/// withdraws the claim (the request was cancelled).
struct Claim<'a> {
    inner: &'a Inner,
    id: u64,
    armed: bool,
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut st = self.inner.state();
        let mut share = st.get(self.id);
        share.claim = false;
        self.inner.set(&mut st, self.id, share);
    }
}

/// See the module docs.
#[derive(Debug)]
pub struct ScanGate {
    inner: Arc<Inner>,
    next_id: AtomicU64,
}

impl Drop for ScanGate {
    fn drop(&mut self) {
        self.inner.pool.unregister(self.inner.id);
    }
}

impl ScanGate {
    /// A gate whose shares are reserved in `pool` and together stay within
    /// `budget` bytes while more than one stream holds one.
    #[must_use]
    pub fn new(pool: &Arc<dyn MemoryPool>, budget: usize) -> Self {
        let inner = Arc::new_cyclic(|me| {
            let hook = Arc::new(Revoke(Weak::clone(me)));
            let id = pool.register(
                "FileScanDecode",
                Some(Arc::clone(&hook) as Arc<dyn Spillable>),
            );
            Inner {
                pool: Arc::clone(pool),
                id,
                _hook: hook,
                budget,
                state: Mutex::new(State::default()),
                reserved: AtomicUsize::new(0),
                released: tokio::sync::Notify::new(),
                overcommitted: AtomicU64::new(0),
                revoked: AtomicU64::new(0),
            }
        });
        Self {
            inner,
            next_id: AtomicU64::new(0),
        }
    }

    /// A new stream's id for [`acquire`](Self::acquire) and friends.
    pub(crate) fn stream_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Grows stream `id`'s share by `bytes` (module docs, admission), and
    /// returns how many of them are over-committed.
    ///
    /// # Errors
    /// [`MemoryError`] if the query was aborted.
    pub(crate) async fn acquire(&self, id: u64, bytes: usize) -> Result<usize, MemoryError> {
        let inner = &self.inner;
        loop {
            // Listen before looking, so a release in between still wakes us.
            let released = inner.released.notified();
            tokio::pin!(released);
            released.as_mut().enable();
            let claimed = {
                let mut st = inner.state();
                let mut share = st.get(id);
                let others = st.active - usize::from(share.active());
                let within = st.bytes.saturating_add(bytes) <= inner.budget;
                if (within || others == 0) && inner.try_reserve(bytes) {
                    share.reserved = share.reserved.saturating_add(bytes);
                    inner.set(&mut st, id, share);
                    return Ok(0);
                }
                if others > 0 && share.bytes() == 0 {
                    false
                } else {
                    // Active while it asks, so the other streams wait.
                    share.claim = true;
                    inner.set(&mut st, id, share);
                    true
                }
            };
            if !claimed {
                released.await;
                continue;
            }
            // Withdraws the claim if this future is dropped while it waits
            // for spills.
            let mut claim = Claim {
                inner,
                id,
                armed: true,
            };
            let got = inner.pool.reserve_without_abort(inner.id, bytes).await;
            claim.armed = false;
            let mut st = inner.state();
            // Revocation may have moved this share's bytes meanwhile.
            let mut now = st.get(id);
            now.claim = false;
            let result = match got {
                Ok(()) => {
                    now.reserved = now.reserved.saturating_add(bytes);
                    Ok(0)
                }
                Err(MemoryError::Exhausted { .. }) => {
                    now.over = now.over.saturating_add(bytes);
                    inner.over_committed(bytes);
                    Ok(bytes)
                }
                Err(e) => Err(e),
            };
            inner.set(&mut st, id, now);
            return result;
        }
    }

    /// Shrinks stream `id`'s share to `bytes`, over-committed bytes first,
    /// returns the reserved rest to the pool, and returns how many
    /// over-committed bytes it dropped (granted without a reservation, or
    /// revoked), so a byte is counted once whichever way it got there.
    pub(crate) fn shrink(&self, id: u64, bytes: usize) -> usize {
        let inner = &self.inner;
        let mut st = inner.state();
        let mut share = st.get(id);
        let mut excess = share.bytes().saturating_sub(bytes);
        let over = excess.min(share.over);
        share.over -= over;
        excess -= over;
        share.reserved -= excess.min(share.reserved);
        inner.set(&mut st, id, share);
        drop(st);
        inner.release(excess);
        over
    }

    /// Stream `id`'s share in bytes (reserved and over-committed).
    pub(crate) fn share(&self, id: u64) -> usize {
        self.inner.state().get(id).bytes()
    }

    /// Bytes of all shares.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.inner.state().bytes
    }

    /// Streams holding a share.
    #[must_use]
    pub fn active(&self) -> usize {
        self.inner.state().active
    }

    /// Bytes of the shares reserved in the pool.
    #[must_use]
    pub fn reserved(&self) -> usize {
        self.inner.reserved.load(Ordering::Acquire)
    }

    /// Bytes of shares held without a reservation so far (taken while the
    /// pool had no room, or revoked), in total.
    #[must_use]
    pub fn overcommitted(&self) -> u64 {
        self.inner.overcommitted.load(Ordering::Relaxed)
    }

    /// Bytes of shares the arbitrator revoked so far, in total.
    #[must_use]
    pub fn revoked(&self) -> u64 {
        self.inner.revoked.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests;
