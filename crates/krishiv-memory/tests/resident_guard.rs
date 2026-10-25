// SPDX-License-Identifier: Apache-2.0
//! The resident-set guard end to end: a probe reports the process holding
//! more than the pools reserved plus the allowance, the root's ordinary
//! grants shrink by that excess, a round spills it, and the cut goes away
//! with the excess. The probe is process-wide, so this binary holds one test.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use krishiv_memory::{
    MemoryConfig, MemoryError, MemoryPool, Reservation, SAMPLE_INTERVAL, SpillOwner, Spillable,
    TreePool, allowance, set_resident_probe,
};

const MIB: usize = 1 << 20;
const LIMIT: usize = 64 * MIB;

static ROOT: OnceLock<Arc<TreePool>> = OnceLock::new();
/// What the fake process holds beyond the root's reservations.
static OVERHEAD: AtomicUsize = AtomicUsize::new(0);
/// When not 0: the resident set, whatever the reservations ([`UNREADABLE`]:
/// the probe fails).
static FIXED: AtomicUsize = AtomicUsize::new(0);
const UNREADABLE: usize = usize::MAX;

/// A resident set that follows the root's reservations plus [`OVERHEAD`],
/// as a process's does, or [`FIXED`].
fn probe() -> Option<usize> {
    match FIXED.load(Ordering::SeqCst) {
        0 => Some(ROOT.get()?.reserved() + OVERHEAD.load(Ordering::SeqCst)),
        UNREADABLE => None,
        fixed => Some(fixed),
    }
}

/// Lets the next grant that needs the pools take a new sample.
fn next_sample() {
    std::thread::sleep(SAMPLE_INTERVAL * 3);
}

#[tokio::test(flavor = "current_thread")]
async fn the_guard_spills_what_the_process_holds_beyond_the_allowance() {
    assert!(set_resident_probe(probe));
    assert!(!set_resident_probe(probe), "one probe per process");
    let root = ROOT.get_or_init(|| {
        TreePool::new(&MemoryConfig {
            limit: Some(LIMIT),
            quantum: MIB,
            spill_wait_timeout: Duration::from_secs(5),
            abort_release_timeout: Duration::from_secs(5),
            spill_scratch_reserve: 0,
            resident_guard: true,
        })
    });

    // Within the allowance: the whole limit is available.
    OVERHEAD.store(allowance(LIMIT), Ordering::SeqCst);
    let q1 = root.child("q1", None);
    let owner = SpillOwner::new();
    let signal = owner.signal();
    let held = q1.register("held", Some(Arc::clone(&signal) as Arc<dyn Spillable>));
    let mut held = Reservation::new(Arc::clone(&q1), held);
    next_sample();
    held.try_grow(LIMIT).unwrap();
    held.shrink(LIMIT - 40 * MIB);
    signal.set_spillable_bytes(held.size());

    // 16 MiB beyond the allowance: ordinary grants stop 16 MiB below the limit.
    OVERHEAD.store(allowance(LIMIT) + 16 * MIB, Ordering::SeqCst);
    let q2 = root.child("q2", None);
    let other = q2.register("other", None);
    next_sample();
    q2.try_reserve(other, MIB).unwrap();
    assert!(matches!(
        q2.try_reserve(other, 8 * MIB),
        Err(MemoryError::Exhausted { .. })
    ));
    assert_eq!(root.limit(), Some(LIMIT), "the limit itself is unchanged");

    // A round spills what the reservations hold above the cut ceiling.
    let q2c = Arc::clone(&q2);
    let request = tokio::spawn(async move { q2c.reserve(other, 8 * MIB).await });
    let target = loop {
        if let Some(t) = signal.spill_requested() {
            break t;
        }
        tokio::task::yield_now().await;
    };
    // 41 MiB held + 8 MiB asked, against 64 - 16 MiB.
    assert_eq!(target, 41 * MIB + 8 * MIB - 48 * MIB);
    held.shrink(target);
    signal.set_spillable_bytes(held.size());
    signal.complete(target);
    request.await.unwrap().unwrap();
    assert!(root.reserved() <= 48 * MIB, "{}", root.reserved());

    // The excess is gone: so is the cut.
    OVERHEAD.store(allowance(LIMIT), Ordering::SeqCst);
    next_sample();
    q2.try_reserve(other, LIMIT - root.reserved()).unwrap();
    assert_eq!(root.reserved(), LIMIT);
    assert!(q1.check_aborted().is_ok() && q2.check_aborted().is_ok());
    q2.release(other, LIMIT);

    // A probe that cannot read the resident set clears the cut.
    OVERHEAD.store(allowance(LIMIT) + 16 * MIB, Ordering::SeqCst);
    next_sample();
    q2.try_reserve(other, MIB).unwrap();
    assert!(q2.try_reserve(other, LIMIT - root.reserved()).is_err());
    FIXED.store(UNREADABLE, Ordering::SeqCst);
    next_sample();
    q2.try_reserve(other, LIMIT - root.reserved()).unwrap();
    q2.release(other, LIMIT);

    // Spilling during an excess does not raise it: the process stays at 8 MiB
    // over the allowance of 40 MiB reserved while the reservations drop to
    // 21 MiB (the allocator keeps the freed pages). Measured from 21 MiB the
    // excess would be 27 MiB, cut to 16; from the high-water it stays 8.
    assert_eq!(root.reserved(), held.size());
    held.shrink(held.size().saturating_sub(40 * MIB));
    held.try_grow(40 * MIB - held.size()).unwrap();
    FIXED.store(40 * MIB + allowance(LIMIT) + 8 * MIB, Ordering::SeqCst);
    next_sample();
    q2.try_reserve(other, MIB).unwrap(); // samples: excess 8 MiB from 40
    held.shrink(20 * MIB);
    next_sample();
    q2.try_reserve(other, MIB).unwrap(); // samples at 21 MiB reserved
    assert_eq!(root.reserved(), 22 * MIB);
    // The ceiling is 64 - 8 MiB (it would be 48 with a cut of 16).
    assert!(q2.try_reserve(other, 35 * MIB).is_err(), "the cut applies");
    q2.try_reserve(other, 34 * MIB).unwrap();

    // `krishiv.memory.residentGuard=false`: a root that is never cut.
    let off = TreePool::new(&MemoryConfig {
        limit: Some(8 * MIB),
        quantum: MIB,
        spill_scratch_reserve: 0,
        resident_guard: false,
        ..MemoryConfig::default()
    });
    let q = off.child("q", None);
    let id = q.register("op", None);
    FIXED.store(usize::MAX / 2, Ordering::SeqCst);
    next_sample();
    // Sampling: the guarded root is cut by a quarter, below what it holds.
    assert!(q2.try_reserve(other, MIB).is_err());
    q.try_reserve(id, 8 * MIB).unwrap();
}
