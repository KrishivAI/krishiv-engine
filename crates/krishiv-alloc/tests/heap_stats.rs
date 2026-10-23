// SPDX-License-Identifier: Apache-2.0
//! `krishiv_alloc::heap` as the global allocator: it counts the live heap
//! and its peak.
#![cfg(feature = "heap-stats")]

use krishiv_alloc::heap::{Counting, live, peak, reset_peak};

#[global_allocator]
static GLOBAL: Counting<std::alloc::System> = Counting(std::alloc::System);

const MIB: usize = 1 << 20;

#[test]
fn counts_live_bytes_and_their_peak() {
    reset_peak();
    let base = live();
    let mut a: Vec<u8> = Vec::with_capacity(MIB);
    let b: Vec<u8> = vec![0; MIB];
    assert!(live() >= base + 2 * MIB, "{} {base}", live());
    // Grown in place or moved: the old size is no longer counted.
    a.reserve_exact(3 * MIB);
    let grown = live();
    assert!(grown >= base + 4 * MIB, "{grown} {base}");
    assert!(grown < base + 6 * MIB, "{grown} {base}");
    drop(a);
    drop(b);
    assert!(live() < base + MIB, "{} {base}", live());
    assert!(peak() >= base + 4 * MIB);
    reset_peak();
    assert!(peak() < base + MIB);
}
