// SPDX-License-Identifier: Apache-2.0
//! With the resident-set guard off, an installed probe starts no pressure
//! epoch, however large the resident set: no root takes part, so there is
//! no excess to release. The probe and the release hook are process-wide,
//! so this binary holds one test.

use std::time::Duration;

use krishiv_memory::{
    EPOCH_INTERVAL, MemoryConfig, MemoryPool, SAMPLE_INTERVAL, TreePool, pressure_epoch,
    set_release_hook, set_resident_probe,
};

const MIB: usize = 1 << 20;

/// A process far above any limit.
fn probe() -> Option<usize> {
    Some(1 << 40)
}

fn hook() {}

#[test]
fn a_disabled_guard_starts_no_pressure_epoch() {
    assert!(set_resident_probe(probe));
    assert!(set_release_hook(hook));
    let root = TreePool::new(&MemoryConfig {
        limit: Some(64 * MIB),
        quantum: MIB,
        spill_wait_timeout: Duration::from_secs(5),
        abort_release_timeout: Duration::from_secs(5),
        spill_scratch_reserve: 0,
        resident_guard: false,
    });
    let q = root.child("q", None);
    let c = q.register("c", None);
    let before = pressure_epoch();
    for _ in 0..3 {
        std::thread::sleep(SAMPLE_INTERVAL.max(EPOCH_INTERVAL) * 2);
        q.try_reserve(c, MIB).unwrap();
        q.release(c, MIB);
    }
    assert_eq!(pressure_epoch(), before);
}
