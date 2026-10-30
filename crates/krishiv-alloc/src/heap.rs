// SPDX-License-Identifier: Apache-2.0
//! Live-heap counting for memory measurements (feature `heap-stats`): a
//! global allocator wrapper that counts the bytes allocated and not yet
//! freed, their high-water mark, and the number of allocations. A
//! measurement build only (two atomic adds per allocation, one per free):
//! it tells the memory pool's view
//! (reserved bytes) from the heap the process really holds, and that from
//! what the allocator retains on top (the resident set).

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// `A`, counting its live bytes ([`live`], [`peak`]).
#[derive(Debug, Default)]
pub struct Counting<A>(pub A);

fn grew(bytes: usize) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    let now = LIVE
        .fetch_add(bytes, Ordering::Relaxed)
        .saturating_add(bytes);
    PEAK.fetch_max(now, Ordering::Relaxed);
}

fn shrank(bytes: usize) {
    LIVE.fetch_sub(bytes, Ordering::Relaxed);
}

// SAFETY: every call is forwarded to `A` with the caller's arguments, which
// meet `GlobalAlloc`'s contract because the caller's do; the counters only
// observe sizes and never touch the memory.
unsafe impl<A: GlobalAlloc> GlobalAlloc for Counting<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged (see the impl).
        let p = unsafe { self.0.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged (see the impl).
        let p = unsafe { self.0.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        shrank(layout.size());
        // SAFETY: forwarded unchanged (see the impl).
        unsafe { self.0.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged (see the impl).
        let p = unsafe { self.0.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            shrank(layout.size());
            grew(new_size);
        }
        p
    }
}

/// Allocations (and reallocations) made through [`Counting`] since the
/// process started; the difference of two reads counts those in between.
pub fn allocations() -> usize {
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// Bytes allocated and not freed through [`Counting`].
pub fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// The most [`live`] has been since the last [`reset_peak`].
pub fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Starts a new high-water mark at the current live bytes.
///
/// A compare-exchange loop rather than a plain store: if a concurrent `grew`
/// raises the peak between the read and the write, the exchange fails and
/// the retry re-reads live bytes (which already include that growth), so a
/// newer peak is never overwritten with a stale value. Growth that stays
/// below the old peak leaves it untouched, so a successful exchange is
/// reconciled with freshly read live bytes (a max, which cannot erase a
/// newer peak landing in between).
pub fn reset_peak() {
    reset_peak_racing(|| {});
}

/// [`reset_peak`] with `raced` run after the live bytes are read and before
/// the exchange, the window in which a concurrent `grew` can interleave.
fn reset_peak_racing(mut raced: impl FnMut()) {
    let mut seen = PEAK.load(Ordering::Relaxed);
    loop {
        let target = live();
        raced();
        match PEAK.compare_exchange_weak(seen, target, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => {
                // A concurrent `grew` that stays below the old peak leaves
                // it untouched, so the exchange above can land below the
                // current live bytes. Re-read live and take the max: either
                // value's growth is covered by its own `fetch_max`, and the
                // max never lowers the peak, so a newer peak landing in
                // between survives either way.
                PEAK.fetch_max(live(), Ordering::Relaxed);
                return;
            }
            Err(now) => seen = now,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Both tests below program the shared counters directly, so they must
    // not run in parallel with each other.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn reset_keeps_a_peak_recorded_during_the_reset() {
        let _guard = SERIAL.lock().unwrap();
        // This unit-test binary does not install `Counting`, so the counters
        // are set directly; no other test in it touches them.
        LIVE.store(10, Ordering::Relaxed);
        PEAK.store(100, Ordering::Relaxed);
        let mut raced = false;
        reset_peak_racing(|| {
            if !raced {
                raced = true;
                // A concurrent allocation of 200 bytes that sets a new peak.
                grew(200);
            }
        });
        assert_eq!(live(), 210);
        assert_eq!(peak(), 210);
    }

    #[test]
    fn every_growth_counts_one_allocation() {
        let _guard = SERIAL.lock().unwrap();
        let before = allocations();
        grew(8);
        grew(0);
        shrank(8);
        assert_eq!(allocations() - before, 2);
    }

    #[test]
    fn reset_never_lands_below_live_when_growth_stays_below_the_old_peak() {
        let _guard = SERIAL.lock().unwrap();
        LIVE.store(10, Ordering::Relaxed);
        PEAK.store(1_000, Ordering::Relaxed);
        let mut raced = false;
        reset_peak_racing(|| {
            if !raced {
                raced = true;
                // Concurrent growth that stays below the old peak: `grew`
                // leaves `PEAK` at 1,000, so the exchange would land at 10,
                // below the live 210, without the reconcile.
                grew(200);
            }
        });
        assert_eq!(live(), 210);
        assert_eq!(peak(), 210);
    }
}
