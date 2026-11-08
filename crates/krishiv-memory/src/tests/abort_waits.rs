// SPDX-License-Identifier: Apache-2.0
//! Abort-phase waits: a round waits for another driver's spill write or holding cohort member, and fails at once
//! when the awaited consumer frees nothing (ADR 0007 update 2026-11-08).

use super::*;

/// Another driver's consumer is writing a spill (its rows are lent to the
/// writer, nothing is spillable): a request that needs those bytes waits for
/// the write to end and release them, instead of failing at once while no
/// spill can be asked of anyone.
#[tokio::test(start_paused = true)]
async fn a_round_waits_for_another_consumers_spill_write() {
    let root = root(Some(4 * Q), Q);
    let q = root.child_pool("q", None);
    let (a, b) = (SpillOwner::new(), SpillOwner::new());
    let ha = ConsumerHandle::register(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        "a",
        Some(a.signal() as Arc<dyn Spillable>),
    );
    let hb = ConsumerHandle::register(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        "b",
        Some(b.signal() as Arc<dyn Spillable>),
    );
    let mut rows = hb.reservation();
    rows.try_grow(3 * Q).unwrap();
    let writing = hb.spilling();
    let started = tokio::time::Instant::now();
    let want = tokio::spawn({
        let ha = Arc::clone(&ha);
        async move { a.reserve_for(&ha, 2 * Q, || async {}).await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(
        !want.is_finished(),
        "the request failed while b's spill write was in progress"
    );
    // The write ends: the rows' bytes go back, then the count.
    rows.shrink(3 * Q);
    drop(writing);
    want.await.unwrap().unwrap();
    assert_eq!(ha.credit(), 2 * Q);
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(root.stats().aborts, 0);
}

/// A spill write that ends without freeing anything (it failed, the rows
/// stay) ends the wait at once: the request fails then, not at
/// `abortReleaseTimeout`.
#[tokio::test(start_paused = true)]
async fn a_spill_write_that_frees_nothing_fails_the_wait_at_once() {
    let root = root(Some(4 * Q), Q);
    let q = root.child_pool("q", Some(4 * Q));
    let (a, b) = (SpillOwner::new(), SpillOwner::new());
    let ha = ConsumerHandle::register(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        "a",
        Some(a.signal() as Arc<dyn Spillable>),
    );
    let hb = ConsumerHandle::register(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        "b",
        Some(b.signal() as Arc<dyn Spillable>),
    );
    let mut rows = hb.reservation();
    rows.try_grow(3 * Q).unwrap();
    let writing = hb.spilling();
    let started = tokio::time::Instant::now();
    let want = tokio::spawn({
        let ha = Arc::clone(&ha);
        async move { a.reserve_for(&ha, 2 * Q, || async {}).await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!want.is_finished());
    drop(writing);
    let e = want.await.unwrap().unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "waited {:?} after the write ended",
        started.elapsed()
    );
    drop(rows);
}

/// Another driver's consumer holds memory as a member of a query cohort
/// (its seat holding: a sort's merge budget, freed without help), nothing is
/// spillable: a request that needs those bytes waits for the member to free
/// them and go idle, instead of aborting the query.
#[tokio::test(start_paused = true)]
async fn a_round_waits_for_another_drivers_holding_cohort_member() {
    let root = root(Some(4 * Q), Q);
    let q = root.child_pool("q", None);
    let (a, b) = (SpillOwner::new(), SpillOwner::new());
    let ha = ConsumerHandle::register(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        "a",
        Some(a.signal() as Arc<dyn Spillable>),
    );
    let hb = ConsumerHandle::register(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        "b",
        Some(b.signal() as Arc<dyn Spillable>),
    );
    let mut seat = hb.join_cohort("merges").unwrap();
    seat.set(SeatState::Holding);
    let mut budget = hb.reservation();
    budget.try_grow(3 * Q).unwrap();
    let started = tokio::time::Instant::now();
    let want = tokio::spawn({
        let ha = Arc::clone(&ha);
        async move { a.reserve_for(&ha, 2 * Q, || async {}).await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(
        !want.is_finished(),
        "the request ended while b's cohort member held its budget"
    );
    // The merge ends: the budget goes, then the seat.
    budget.shrink(3 * Q);
    seat.set(SeatState::Idle);
    want.await.unwrap().unwrap();
    assert_eq!(ha.credit(), 2 * Q);
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(root.stats().aborts, 0);
}

/// A cohort member that goes idle still holding its memory (it was not the
/// member's to free) ends the wait at once: the request fails then, not at
/// `abortReleaseTimeout`.
#[tokio::test(start_paused = true)]
async fn a_cohort_member_that_frees_nothing_fails_the_wait_at_once() {
    let root = root(Some(4 * Q), Q);
    let q = root.child_pool("q", Some(4 * Q));
    let (a, b) = (SpillOwner::new(), SpillOwner::new());
    let ha = ConsumerHandle::register(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        "a",
        Some(a.signal() as Arc<dyn Spillable>),
    );
    let hb = ConsumerHandle::register(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        "b",
        Some(b.signal() as Arc<dyn Spillable>),
    );
    let mut rows = hb.reservation();
    rows.try_grow(3 * Q).unwrap();
    let mut seat = hb.join_cohort("merges").unwrap();
    seat.set(SeatState::Holding);
    let started = tokio::time::Instant::now();
    let want = tokio::spawn({
        let ha = Arc::clone(&ha);
        async move { a.reserve_for(&ha, 2 * Q, || async {}).await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!want.is_finished());
    seat.set(SeatState::Idle);
    let e = want.await.unwrap().unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "waited {:?} after the member went idle",
        started.elapsed()
    );
    drop(rows);
}

/// Holding cohort members a round must not wait for end it at once (the
/// request fails or aborts as without them, never at `abortReleaseTimeout`):
/// one of the requester's own driver (which pulls it), one of a consumer
/// registered without a driver, any for a requester without one, and one of
/// another query.
#[tokio::test(start_paused = true)]
async fn holding_members_a_round_cannot_wait_for_end_it_at_once() {
    /// Asks `bytes` for `h` through `owner` (or straight from the pool
    /// without one) and returns the error class and the time it took.
    async fn ask(
        owner: Option<&Arc<SpillOwner>>,
        h: &Arc<ConsumerHandle>,
    ) -> (ErrorClass, Duration) {
        let started = tokio::time::Instant::now();
        let e = match owner {
            Some(o) => o.reserve_for(h, 2 * Q, || async {}).await.unwrap_err(),
            None => h.reservation().grow(2 * Q).await.unwrap_err(),
        };
        (e.error_class(), started.elapsed())
    }
    let register = |q: &Arc<TreePool>, owner: Option<&Arc<SpillOwner>>| {
        ConsumerHandle::register(
            Arc::clone(q) as Arc<dyn MemoryPool>,
            "c",
            owner.map(|o| o.signal() as Arc<dyn Spillable>),
        )
    };
    let fast = Duration::from_secs(1);
    // (holder's owner is the requester's, holder has an owner, requester
    // has an owner)
    for (same, holder_owned, requester_owned) in [
        (true, true, true),
        (false, false, true),
        (false, true, false),
    ] {
        let root = root(Some(4 * Q), Q);
        let q = root.child_pool("q", Some(4 * Q));
        let (a, b) = (SpillOwner::new(), SpillOwner::new());
        let holder_owner = if same { &a } else { &b };
        let hb = register(&q, holder_owned.then_some(holder_owner));
        let mut seat = hb.join_cohort("merges").unwrap();
        seat.set(SeatState::Holding);
        let mut held = hb.reservation();
        held.try_grow(3 * Q).unwrap();
        let ha = register(&q, requester_owned.then_some(&a));
        let (class, took) = ask(requester_owned.then_some(&a), &ha).await;
        assert_eq!(class, ErrorClass::UnableToAcquireMemory);
        assert!(
            took < fast,
            "same={same} holder_owned={holder_owned} requester_owned={requester_owned}: waited {took:?}"
        );
        drop((held, seat));
    }
    // Another query's member: the round aborts that query (the furthest
    // above its share) instead of waiting for it.
    let root = root(Some(4 * Q), Q);
    let (q1, q2) = (root.child_pool("q1", None), root.child_pool("q2", None));
    let (a, b) = (SpillOwner::new(), SpillOwner::new());
    let hb = register(&q2, Some(&b));
    let mut seat = hb.join_cohort("merges").unwrap();
    seat.set(SeatState::Holding);
    let mut held = hb.reservation();
    held.try_grow(3 * Q).unwrap();
    let ha = register(&q1, Some(&a));
    let started = tokio::time::Instant::now();
    let granted = tokio::spawn({
        let ha = Arc::clone(&ha);
        async move { a.reserve_for(&ha, 2 * Q, || async {}).await }
    });
    // The abort waits for the aborted query's release.
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(
        root.stats().aborts,
        1,
        "another query's member was waited for"
    );
    drop(held);
    granted.await.unwrap().unwrap();
    assert!(started.elapsed() < fast);
    drop(seat);
}

/// Two drivers, each with a holding cohort member, both ask for memory only
/// the other's member would free: neither waits for the other (each one's
/// driver waits in a round), so the requests end at once instead of at
/// `abortReleaseTimeout`.
#[tokio::test(start_paused = true)]
async fn two_holding_drivers_asking_do_not_wait_for_each_other() {
    let root = root(Some(4 * Q), Q);
    let q = root.child_pool("q", Some(4 * Q));
    let owners = [SpillOwner::new(), SpillOwner::new()];
    let mut kept = Vec::new();
    let mut asks = Vec::new();
    for o in &owners {
        let member = ConsumerHandle::register(
            Arc::clone(&q) as Arc<dyn MemoryPool>,
            "member",
            Some(o.signal() as Arc<dyn Spillable>),
        );
        let mut seat = member.join_cohort("merges").unwrap();
        seat.set(SeatState::Holding);
        let mut held = member.reservation();
        held.try_grow(2 * Q).unwrap();
        kept.push((held, seat, member));
        let asker = ConsumerHandle::register(
            Arc::clone(&q) as Arc<dyn MemoryPool>,
            "asker",
            Some(o.signal() as Arc<dyn Spillable>),
        );
        let o = Arc::clone(o);
        asks.push(tokio::spawn(async move {
            o.reserve_for(&asker, Q, || async {}).await
        }));
    }
    let started = tokio::time::Instant::now();
    for ask in asks {
        assert!(ask.await.unwrap().is_err());
    }
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "waited {:?}",
        started.elapsed()
    );
    drop(kept);
}
