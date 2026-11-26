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
//! An *idle epoch* ([`note_idle`]) starts when the process has stayed without
//! a running query for a while (the executor's idle round decides; design
//! `0.4-memory-query-end-release`). Threads that check with
//! [`release_if_due_here`] (an idle worker before it parks, a pool thread
//! after its closure) then release once for it, so memory a finished query
//! freed does not stay in the allocator of every thread that ran it. Running
//! drivers check only pressure epochs ([`release_if_pressured_here`]), so a
//! query never releases mid-task for an idle epoch.
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
static IDLE_EPOCH: AtomicU64 = AtomicU64::new(0);
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

/// Whether a release hook is installed ([`set_release_hook`]): pressure
/// epochs start only then.
#[must_use]
pub fn release_hook_installed() -> bool {
    HOOK.get().is_some()
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

/// The current idle epoch.
pub fn idle_epoch() -> u64 {
    IDLE_EPOCH.load(Ordering::Acquire)
}

/// The process has stayed idle (no query running): starts a new idle epoch.
/// Called by the executor once its debounce confirmed idleness, never per
/// query end (a burst of statements must not make threads release between
/// them).
pub fn note_idle() {
    IDLE_EPOCH.fetch_add(1, Ordering::AcqRel);
}

thread_local! {
    /// The pressure epoch this thread last released memory for.
    static SEEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The idle epoch this thread last released memory for.
    static SEEN_IDLE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Calls the release hook on this thread once if an idle epoch
/// ([`note_idle`]) started since it last released for one; pressure epochs
/// are left to the batch-boundary checks. For a worker about to park: it
/// parks often under load, and a pressure release there would add collects
/// on threads that never took part in them (I/O workers). Cheap otherwise.
pub fn release_if_idle_here() -> bool {
    let Some(hook) = HOOK.get() else {
        return false;
    };
    let idle = idle_epoch();
    let due = SEEN_IDLE.with(|s| s.replace(idle) != idle);
    if due {
        hook();
    }
    due
}

/// Calls the release hook on this thread once if a pressure epoch or an idle
/// epoch ([`note_idle`]) started since it last released; returns whether it
/// called the hook. Cheap otherwise: a few atomic and thread-local loads. For
/// threads between tasks (a worker about to park, a pool thread after its
/// closure), not for running drivers.
pub fn release_if_due_here() -> bool {
    let Some(hook) = HOOK.get() else {
        return false;
    };
    let pressure = pressure_epoch();
    let idle = idle_epoch();
    let due = SEEN.with(|s| s.replace(pressure) != pressure)
        | SEEN_IDLE.with(|s| s.replace(idle) != idle);
    if due {
        hook();
    }
    due
}

/// [`release_if_pressured`] with this thread's own record of the epochs it
/// released for: the hook frees the *calling thread's* allocator caches, and
/// a task can move between threads, so the record belongs to the thread.
pub fn release_if_pressured_here() -> bool {
    SEEN.with(|seen| {
        let mut s = seen.get();
        let called = release_if_pressured(&mut s);
        seen.set(s);
        called
    })
}

/// Calls the release hook on this thread if a pressure epoch started since
/// `seen` (then updates `seen`). Returns whether it called the hook. Cheap
/// otherwise: one atomic load.
pub fn release_if_pressured(seen: &mut u64) -> bool {
    // Epochs only start once a hook is installed.
    let Some(hook) = HOOK.get() else {
        return false;
    };
    let epoch = pressure_epoch();
    if epoch == *seen {
        return false;
    }
    *seen = epoch;
    hook();
    true
}
