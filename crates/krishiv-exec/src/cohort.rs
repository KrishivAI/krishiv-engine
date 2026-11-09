// SPDX-License-Identifier: Apache-2.0
//! Waiting in a [`krishiv_memory::Cohort`]: a member's request the pool
//! refuses waits while another member holds memory it frees on its own
//! (the nested loop's build chunks, sorts' merge budgets), instead of
//! failing or aborting a query.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_common::error::KrishivError;
use krishiv_memory::{MemoryError, Reservation, Seat, SeatState};

use crate::error::Result;

/// Grows `r` by `bytes`, memory a member of cohort `what` needs to go on (a
/// nested-loop driver holding no chunk, a sort's merge budget). With no
/// seat, as [`Reservation::grow`]. With one, the seat is
/// [`SeatState::Reserving`] while it asks: arbitration spills without
/// aborting; if that is not enough and another member of the cohort is
/// active (holds memory it frees on its own, or is asking), the seat stands
/// by ([`Seat::stand_by`]: idle without waking anyone, so two askers never
/// wait for each other and waiters do not wake each other), waits for a
/// member to go idle and asks again, counting the wait in `waits`. With no
/// other member active, the full arbitration runs, which may fail the
/// request or abort a query. The seat ends in `then` once granted, idle on
/// an error or if the future is dropped (which wakes the cohort).
pub(crate) async fn reserve_in_cohort(
    r: &mut Reservation,
    bytes: usize,
    seat: Option<(&mut Seat, &AtomicU64)>,
    then: SeatState,
    what: &'static str,
) -> Result<()> {
    /// Leaves the seat idle (waking the cohort) unless it ended otherwise.
    struct Asking<'a>(&'a mut Seat);
    impl Drop for Asking<'_> {
        fn drop(&mut self) {
            if self.0.state() == SeatState::Reserving {
                self.0.set(SeatState::Idle);
            }
        }
    }
    let Some((seat, waits)) = seat else {
        return r.grow(bytes).await.map_err(KrishivError::from);
    };
    debug_assert_eq!(seat.state(), SeatState::Idle, "a seat asks from idle");
    let cohort = Arc::clone(seat.cohort());
    let asking = Asking(seat);
    let out = loop {
        asking.0.set(SeatState::Reserving);
        match r.grow_without_abort(bytes).await {
            Ok(()) => break Ok(()),
            Err(MemoryError::Exhausted { .. }) => {}
            Err(e) => break Err(e),
        }
        asking.0.stand_by();
        // Listen before checking: a member going idle meanwhile wakes us.
        let changed = cohort.changed();
        if !cohort.others_active(asking.0) {
            asking.0.set(SeatState::Reserving);
            let out = r.grow(bytes).await;
            if out.is_err() {
                tracing::debug!(
                    bytes,
                    members = cohort.members(),
                    cohort = what,
                    "cohort: no other member active, request refused"
                );
            }
            break out;
        }
        waits.fetch_add(1, Ordering::Relaxed);
        changed.await;
    };
    // Granted: `then`; failed: idle (the guard, waking the cohort).
    if out.is_ok() {
        asking.0.set(then);
    }
    drop(asking);
    out.map_err(KrishivError::from)
}

#[cfg(all(test, not(loom)))]
mod tests {
    #![expect(
        clippy::unwrap_used,
        clippy::disallowed_methods,
        reason = "tests: #[tokio::test] blocks on its own runtime"
    )]

    use std::time::Duration;

    use krishiv_common::error::ErrorClass;
    use krishiv_memory::{
        ConsumerId, MemoryConfig, MemoryPool, SpillSignal, Spillable as _, TreePool,
    };

    use super::*;

    const TEST: &str = "test";

    const KIB: usize = 1 << 10;

    /// A query pool under a 256 KiB limit; a peer member of the chunk cohort
    /// holding 240 KiB as its chunk (seat holding); the seat and an empty
    /// reservation of another consumer of the query.
    fn setup() -> (Arc<dyn MemoryPool>, ConsumerId, Seat, Seat, Reservation) {
        let root = TreePool::new(&MemoryConfig {
            limit: Some(256 * KIB),
            quantum: 4 * KIB,
            spill_scratch_reserve: 0,
            ..MemoryConfig::default()
        });
        let q = root.child("q", None);
        let cohort = q.cohort(TEST).unwrap();
        let mut peer = cohort.join();
        peer.set(SeatState::Holding);
        let held = q.register("peer", None);
        q.try_reserve(held, 240 * KIB).unwrap();
        let id = q.register("chunk", None);
        let r = Reservation::new(Arc::clone(&q), id);
        (q, held, peer, cohort.join(), r)
    }

    /// While another member holds its chunk, a batch that does not fit waits
    /// (no failure, no abort) and is granted once the member frees its
    /// chunk and goes idle; the seat then holds.
    #[tokio::test]
    async fn a_batch_waits_for_an_active_member() {
        let (q, held, mut peer, mut seat, mut r) = setup();
        let waits = AtomicU64::new(0);
        {
            let mut ask = Box::pin(reserve_in_cohort(
                &mut r,
                32 * KIB,
                Some((&mut seat, &waits)),
                SeatState::Holding,
                TEST,
            ));
            let early = tokio::time::timeout(Duration::from_millis(200), ask.as_mut()).await;
            assert!(early.is_err(), "granted while the peer held its chunk");
            // As `Disk::drop_chunk`: the memory goes, then the seat goes idle.
            q.release(held, 240 * KIB);
            peer.set(SeatState::Idle);
            tokio::time::timeout(Duration::from_secs(10), ask)
                .await
                .unwrap()
                .unwrap();
        }
        assert_eq!(seat.state(), SeatState::Holding);
        assert_eq!(r.size(), 32 * KIB);
        assert!(waits.load(Ordering::Relaxed) >= 1);
        assert!(q.check_aborted().is_ok());
        r.shrink(32 * KIB);
        q.unregister(held);
    }

    /// Two batches waiting on one holder do not wake each other: each
    /// waits a bounded number of times (backing off does not notify), and
    /// both are granted once the holder lets go.
    #[tokio::test]
    async fn two_waiters_do_not_wake_each_other() {
        let (q, held, mut peer, mut a, mut ra) = setup();
        let mut b = a.cohort().join();
        let mut rb = Reservation::new(Arc::clone(&q), q.register("chunk2", None));
        let (wa, wb) = (AtomicU64::new(0), AtomicU64::new(0));
        {
            let ask_a = reserve_in_cohort(
                &mut ra,
                20 * KIB,
                Some((&mut a, &wa)),
                SeatState::Holding,
                TEST,
            );
            let ask_b = reserve_in_cohort(
                &mut rb,
                20 * KIB,
                Some((&mut b, &wb)),
                SeatState::Holding,
                TEST,
            );
            let both = async { tokio::join!(ask_a, ask_b) };
            let mut both = Box::pin(both);
            let early = tokio::time::timeout(Duration::from_millis(300), both.as_mut()).await;
            assert!(early.is_err(), "granted while the peer held its chunk");
            assert!(
                wa.load(Ordering::Relaxed) <= 2 && wb.load(Ordering::Relaxed) <= 2,
                "waiters woke each other: {wa:?} {wb:?}"
            );
            q.release(held, 240 * KIB);
            peer.set(SeatState::Idle);
            let (x, y) = tokio::time::timeout(Duration::from_secs(10), both)
                .await
                .unwrap();
            x.unwrap();
            y.unwrap();
        }
        assert!(wa.load(Ordering::Relaxed) <= 3 && wb.load(Ordering::Relaxed) <= 3);
        assert_eq!(
            (a.state(), b.state()),
            (SeatState::Holding, SeatState::Holding)
        );
        ra.shrink(20 * KIB);
        rb.shrink(20 * KIB);
        q.unregister(held);
    }

    /// A request dropped while it waits leaves its seat idle.
    #[tokio::test]
    async fn a_dropped_request_leaves_its_seat_idle() {
        let (q, held, _peer, mut seat, mut r) = setup();
        let waits = AtomicU64::new(0);
        {
            let ask = reserve_in_cohort(
                &mut r,
                32 * KIB,
                Some((&mut seat, &waits)),
                SeatState::Holding,
                TEST,
            );
            let early = tokio::time::timeout(Duration::from_millis(200), ask).await;
            assert!(early.is_err());
        }
        assert_eq!(seat.state(), SeatState::Idle);
        q.release(held, 240 * KIB);
        q.unregister(held);
    }

    /// A request dropped while it is still asking (its arbitration waits on
    /// a spill nobody services) leaves its seat idle and wakes the cohort:
    /// the seat would otherwise count as active for good.
    #[tokio::test]
    async fn a_request_dropped_while_asking_leaves_its_seat_idle() {
        let (q, held, mut peer, mut seat, mut r) = setup();
        // The peer's chunk is gone; a spillable consumer holds the memory
        // instead, and its spill is requested but never serviced.
        q.release(held, 240 * KIB);
        q.unregister(held);
        peer.set(SeatState::Idle);
        let signal = Arc::new(SpillSignal::new());
        signal.set_spillable_bytes(240 * KIB);
        let victim = q.register("victim", Some(Arc::clone(&signal) as _));
        q.try_reserve(victim, 240 * KIB).unwrap();
        let waits = AtomicU64::new(0);
        let woken = peer.cohort().changed();
        {
            let ask = reserve_in_cohort(
                &mut r,
                32 * KIB,
                Some((&mut seat, &waits)),
                SeatState::Holding,
                TEST,
            );
            let early = tokio::time::timeout(Duration::from_millis(200), ask).await;
            assert!(early.is_err(), "the request did not wait on the spill");
        }
        assert!(signal.spill_requested().is_some(), "no spill was asked for");
        assert_eq!(waits.load(Ordering::Relaxed), 0, "it stood by instead");
        assert_eq!(seat.state(), SeatState::Idle);
        assert!(!peer.cohort().others_active(&peer), "the seat still counts");
        tokio::time::timeout(Duration::from_secs(1), woken)
            .await
            .expect("the drop did not wake the cohort");
        q.release(victim, 240 * KIB);
        q.unregister(victim);
    }

    /// With no other member active, nothing will free memory: the request
    /// runs the full arbitration as before and fails, and the seat is idle.
    #[tokio::test]
    async fn with_no_active_member_a_batch_fails_as_before() {
        let (q, held, mut peer, mut seat, mut r) = setup();
        peer.set(SeatState::Idle);
        let waits = AtomicU64::new(0);
        let e = tokio::time::timeout(
            Duration::from_secs(10),
            reserve_in_cohort(
                &mut r,
                32 * KIB,
                Some((&mut seat, &waits)),
                SeatState::Holding,
                TEST,
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
        assert_eq!(seat.state(), SeatState::Idle);
        assert_eq!(waits.load(Ordering::Relaxed), 0);
        q.release(held, 240 * KIB);
        q.unregister(held);
    }
}
