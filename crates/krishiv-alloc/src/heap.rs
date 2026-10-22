// SPDX-License-Identifier: Apache-2.0
//! Live-heap counting for memory measurements (feature `heap-stats`): a
//! global allocator wrapper that counts the bytes allocated and not yet
//! freed, and their high-water mark. A measurement build only (one atomic
//! add per allocation and free): it tells the memory pool's view
//! (reserved bytes) from the heap the process really holds, and that from
//! what the allocator retains on top (the resident set).

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// `A`, counting its live bytes ([`live`], [`peak`]).
#[derive(Debug, Default)]
pub struct Counting<A>(pub A);

fn grew(bytes: usize) {
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

/// Bytes allocated and not freed through [`Counting`].
pub fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// The most [`live`] has been since the last [`reset_peak`].
pub fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Starts a new high-water mark at the current live bytes.
pub fn reset_peak() {
    PEAK.store(live(), Ordering::Relaxed);
}
