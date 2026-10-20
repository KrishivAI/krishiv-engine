// SPDX-License-Identifier: Apache-2.0
//! Returning freed memory to the OS under memory pressure.
//!
//! An allocator keeps memory it has freed (mimalloc purges it only after a
//! delay, and not at all from huge pages backing its arenas), so a process
//! can hold far more than its memory pools have reserved. Purging eagerly
//! costs page faults when the memory is used again, so it is done only when
//! it matters: whenever a pool with a limit refuses a grant, the process
//! enters a new *pressure epoch* (at most one per [`EPOCH_INTERVAL`]), and
//! each pipeline driver that sees a new epoch at a batch boundary calls the
//! host's release hook ([`set_release_hook`]) on its own thread. Unlimited
//! runs never refuse a grant, so they never pay for it.
//!
//! The engine never sets a global allocator (a library cannot: its host
//! owns it), so the hook comes from the binary that chose the allocator,
//! e.g. one calling `mi_collect(true)` for mimalloc. Without a hook, memory
//! the allocator retains is outside the pools' control: see the memory
//! bound's documented limitation.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The shortest time between two pressure epochs.
pub const EPOCH_INTERVAL: Duration = Duration::from_millis(1);

static HOOK: OnceLock<fn()> = OnceLock::new();
static EPOCH: AtomicU64 = AtomicU64::new(0);
/// Milliseconds since [`start`] at the last epoch (0: none yet).
static LAST: AtomicU64 = AtomicU64::new(0);

fn start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Installs the process's release hook: it returns the calling thread's
/// freed memory to the OS (for mimalloc: `mi_collect(true)`). Called by the
/// program that owns the global allocator, once; returns `false` if a hook
/// was set already.
pub fn set_release_hook(hook: fn()) -> bool {
    HOOK.set(hook).is_ok()
}

/// The current pressure epoch.
pub fn pressure_epoch() -> u64 {
    EPOCH.load(Ordering::Acquire)
}

/// A pool with a limit refused a grant: starts a new epoch unless one
/// started within [`EPOCH_INTERVAL`].
pub(crate) fn note_pressure() {
    if HOOK.get().is_none() {
        return;
    }
    let now = u64::try_from(start().elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    let last = LAST.load(Ordering::Relaxed);
    let interval = u64::try_from(EPOCH_INTERVAL.as_millis()).unwrap_or(u64::MAX);
    if (last == 0 || now.saturating_sub(last) >= interval)
        && LAST
            .compare_exchange(last, now, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    {
        EPOCH.fetch_add(1, Ordering::AcqRel);
    }
}

/// Calls the release hook on this thread if a pressure epoch started since
/// `seen` (then updates `seen`). Returns whether it called the hook. Cheap
/// otherwise: one atomic load.
pub fn release_if_pressured(seen: &mut u64) -> bool {
    let epoch = pressure_epoch();
    if epoch == *seen {
        return false;
    }
    *seen = epoch;
    match HOOK.get() {
        Some(hook) => {
            hook();
            true
        }
        None => false,
    }
}
