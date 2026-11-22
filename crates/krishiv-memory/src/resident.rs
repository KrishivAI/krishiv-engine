// SPDX-License-Identifier: Apache-2.0
//! The resident-set guard: what the process holds beyond the pools'
//! reservations stays within an allowance.
//!
//! A limited root pool bounds the bytes its consumers reserve. The process
//! holds more: memory the allocator keeps after it was freed (blocks freed
//! while other blocks of their allocator page are still live, which no
//! purge can return), memory decoded without a reservation, and the binary,
//! stacks and runtime. That overhead follows the workload's allocation
//! pattern rather than the limit: a spilling TPC-H Q21 at SF10 with 12
//! threads holds about 400–500 MiB above its reservations both at a 2 GiB
//! and at a 1.75 GiB limit, while scans that do not spill hold far less. A
//! fixed share of the limit set aside for it is either too small for one
//! query or wasted on the others.
//!
//! So a limited root measures it. With a probe installed
//! ([`set_resident_probe`]) and `krishiv.memory.residentGuard` on, the
//! process's resident set is sampled at most once per [`SAMPLE_INTERVAL`],
//! on a grant that needs more bytes from the pools. What it holds above the
//! reservations of every guarded root plus their allowances ([`allowance`]:
//! an eighth of the limit plus [`ALLOWANCE_BASE`]) is the *excess*. While
//! an excess lasts, the reservations count at their high-water since it
//! began: spilling lowers them, but the allocator may keep the freed pages,
//! and the excess must not grow because of the spill it caused. Each root's
//! share of the excess, in proportion to its limit and at most a quarter of
//! the limit, is cut from what ordinary reservations may take there, so
//! consumers spill, and threads return their allocator caches (a pressure
//! epoch, see [`set_release_hook`](crate::set_release_hook)). A probe that
//! cannot read the resident set clears the cuts.
//!
//! **What is cut, and what is not.** The cut applies to growth that has a
//! fallback: an operator whose `try_grow` is refused spills or returns
//! `NeedMemory`, and the driver's arbitration round spills the shortfall.
//! It never applies to spill scratch, to spill readers and writers, or to
//! reservations marked [`Reservation::unguarded`](crate::Reservation::unguarded)
//! (growth whose refusal would fail the query), and an arbitration round
//! whose spills are exhausted asks without it. So the cut on its own does
//! not abort a query or fail an arbitrated request; a sync `try_grow` that
//! is neither marked unguarded nor followed by a fallback can still be
//! refused by it, as by any full pool. The cut does not change a pool's
//! [`limit`](crate::MemoryPool::limit), so plans and budgets sized from the
//! limit stay the same.
//!
//! **Scope.** The resident set covers the whole process, not one query:
//! allocator overhead, but also memory used without a reservation (I/O and
//! exchange buffers, decode working sets over their gate), the binary and
//! the runtime. The cut is root-wide, so one query's overhead can make
//! another query spill. That is the intent: a cgroup that kills the process
//! kills every query in it. A process with several guarded roots splits the
//! excess between them by limit.
//!
//! The probe belongs to the program that owns the allocator, like the
//! release hook. The resident set tracks the pools' reservations closely
//! when the allocator returns freed memory promptly (`krishiv_alloc`'s
//! settings under a limit below its generous threshold). An allocator that
//! keeps freed memory for a second (mimalloc's defaults, at a generous
//! limit or in the Python module) makes the guard cut for memory about to be
//! returned; the cut also starts a pressure epoch, whose release hook
//! returns it. An excess the release cannot lower (in the Python module the
//! resident set also holds the program's own data) keeps its cut, but
//! starts pressure epochs only with a back-off (`Backoff`). The `krishiv`
//! CLI and the Python module install the probe at every limit anyway (on
//! Linux; [`process_resident_bytes`] is `None` elsewhere): the default
//! limit is 80% of the container, and with no guard memory the pools do not
//! track took a process over its cgroup (acceptance SQL-34); a spill costs
//! less than a kill. At the default limit the guard holds the process to
//! about 0.9 x the container + 64 MiB, which is inside it from 640 MiB up;
//! a smaller container needs a lower limit. Without a probe nothing is
//! sampled or cut.

use std::sync::atomic::{AtomicU64, Ordering as StdOrdering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use crate::pool::Node;
use crate::sync::Ordering;

/// The shortest time between two samples of the resident set.
pub const SAMPLE_INTERVAL: Duration = Duration::from_millis(1);

/// The fixed part of a root's allowance: the binary, thread stacks and
/// runtime structures that do not grow with the limit.
pub const ALLOWANCE_BASE: usize = 64 << 20;

/// A root's allowance is this fraction of its limit plus [`ALLOWANCE_BASE`].
const ALLOWANCE_DIVISOR: usize = 8;

/// The cut is at most this fraction of a root's limit.
const MAX_CUT_DIVISOR: usize = 4;

/// A change of the excess by at least this much counts as a rise or a fall
/// for the guard's release back-off ([`Backoff`]); smaller changes are the
/// resident set's ordinary noise.
const EXCESS_STEP: usize = 16 << 20;

/// The longest wait, in milliseconds, between two pressure epochs the guard
/// starts while the excess neither rises nor falls.
const MAX_BACKOFF_MS: u64 = 1024;

static PROBE: OnceLock<fn() -> Option<usize>> = OnceLock::new();

/// Milliseconds since [`start`] at the last sample (0: none yet).
static LAST: AtomicU64 = AtomicU64::new(0);

/// The guard's process-wide state.
struct Guard {
    /// Every limited root pool that takes part (pruned when sampled).
    roots: Vec<Weak<Node>>,
    /// While an excess lasts: the most the roots reserved since it began.
    /// The excess is measured against it, so memory that spilling frees
    /// from the reservations, but that stays resident as allocator
    /// overhead, does not raise the cut further.
    high_water: Option<usize>,
    /// When the guard may start its next pressure epoch.
    backoff: Backoff,
}

/// When an excess starts a pressure epoch. The release hook can return only
/// what the allocator keeps; an excess it cannot lower (in the Python
/// module: the program's own data, which the resident set includes) would
/// otherwise start an epoch, and a release on every pipeline thread, at
/// every sample for as long as it lasts. So an excess starts one at once
/// only when it is new, or reaches a new high or a new low by at least
/// [`EXCESS_STEP`] beyond the range its epochs have covered since it began;
/// otherwise the next waits for a back-off that doubles, up to
/// [`MAX_BACKOFF_MS`], and restarts at 1 ms after an epoch at once. The
/// range, not the last epoch's excess, is the reference: a release purges
/// what the allocator kept, so the next sample is lower, and the running
/// query then refills the same pages; judged against the last epoch, that
/// purge and refill would read as a fall and a rise and start epochs at the
/// refill rate. The range moves only at epochs, so a slow rise between
/// them adds up until it crosses the step. The cut does not depend on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Backoff {
    /// The lowest and highest excess at the guard's epochs since the excess
    /// began (`None`: no epoch in this excess yet).
    range: Option<(usize, usize)>,
    /// The current wait between epochs, in milliseconds (0: none yet).
    step_ms: u64,
    /// Milliseconds since [`start`] before which no epoch starts unless the
    /// excess leaves its range.
    next_at: u64,
}

impl Backoff {
    const NONE: Self = Self {
        range: None,
        step_ms: 0,
        next_at: 0,
    };

    /// Whether a sample at `now` with `excess` starts a pressure epoch;
    /// updates the back-off.
    fn should_release(&mut self, excess: usize, now: u64) -> bool {
        if excess == 0 {
            *self = Self::NONE;
            return false;
        }
        let outside = self.range.is_none_or(|(lowest, highest)| {
            excess >= highest.saturating_add(EXCESS_STEP)
                || excess.saturating_add(EXCESS_STEP) <= lowest
        });
        let step = if outside {
            1
        } else if now < self.next_at {
            return false;
        } else {
            self.step_ms.saturating_mul(2).clamp(1, MAX_BACKOFF_MS)
        };
        let range = self.range.map_or((excess, excess), |(lo, hi)| {
            (lo.min(excess), hi.max(excess))
        });
        *self = Self {
            range: Some(range),
            step_ms: step,
            next_at: now.saturating_add(step),
        };
        true
    }
}

static GUARD: std::sync::Mutex<Guard> = std::sync::Mutex::new(Guard {
    roots: Vec::new(),
    high_water: None,
    backoff: Backoff::NONE,
});

fn guard() -> std::sync::MutexGuard<'static, Guard> {
    GUARD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Installs the process's resident-set probe: it returns the bytes the
/// process holds in memory, `None` when unknown. Called once by the program
/// that owns the global allocator (see the module docs); returns `false` if
/// a probe was set already.
pub fn set_resident_probe(probe: fn() -> Option<usize>) -> bool {
    PROBE.set(probe).is_ok()
}

/// Whether a resident-set probe is installed ([`set_resident_probe`]): the
/// guard samples and cuts only then.
#[must_use]
pub fn resident_probe_installed() -> bool {
    PROBE.get().is_some()
}

/// The allowance of a root pool with `limit`: what the process may hold
/// beyond the root's reservations before the guard cuts into the limit.
#[must_use]
pub fn allowance(limit: usize) -> usize {
    (limit / ALLOWANCE_DIVISOR).saturating_add(ALLOWANCE_BASE)
}

/// The process's resident set in bytes (`VmRSS` of `/proc/self/status`),
/// for [`set_resident_probe`]. `None` off Linux or if it cannot be read.
/// Allocates nothing: it can run while memory is short.
#[must_use]
pub fn process_resident_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::FileExt;
        static STATUS: OnceLock<Option<std::fs::File>> = OnceLock::new();
        let file = STATUS
            .get_or_init(|| std::fs::File::open("/proc/self/status").ok())
            .as_ref()?;
        let mut buf = [0u8; 4096];
        let n = file.read_at(&mut buf, 0).ok()?;
        vm_rss(&buf[..n])
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The `VmRSS:` value of a `/proc/<pid>/status` text, in bytes.
fn vm_rss(status: &[u8]) -> Option<usize> {
    const KEY: &[u8] = b"VmRSS:";
    let at = status.windows(KEY.len()).position(|w| w == KEY)?;
    let rest = &status[at + KEY.len()..];
    let digits = rest
        .iter()
        .skip_while(|b| b.is_ascii_whitespace())
        .take_while(|b| b.is_ascii_digit());
    let mut kib: usize = 0;
    let mut any = false;
    for &d in digits {
        any = true;
        kib = kib.checked_mul(10)?.checked_add(usize::from(d - b'0'))?;
    }
    any.then(|| kib.checked_mul(1024)).flatten()
}

/// Adds a limited root to the roots the guard samples.
pub(crate) fn register(root: &Arc<Node>) {
    let mut g = guard();
    g.roots.retain(|r| r.strong_count() > 0);
    g.roots.push(Arc::downgrade(root));
}

/// Samples the resident set and updates every root's cut, unless no probe
/// is installed or a sample was taken within [`SAMPLE_INTERVAL`]. A probe
/// that cannot read the resident set clears the cuts.
pub(crate) fn sample() {
    let Some(probe) = PROBE.get() else {
        return;
    };
    let now = u64::try_from(start().elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    let last = LAST.load(StdOrdering::Relaxed);
    let interval = u64::try_from(SAMPLE_INTERVAL.as_millis()).unwrap_or(u64::MAX);
    if last != 0 && now.saturating_sub(last) < interval {
        return;
    }
    if LAST
        .compare_exchange(last, now, StdOrdering::AcqRel, StdOrdering::Relaxed)
        .is_err()
    {
        return;
    }
    let resident = probe();
    let mut g = guard();
    g.roots.retain(|r| r.strong_count() > 0);
    let limited: Vec<(Arc<Node>, usize)> = g
        .roots
        .iter()
        .filter_map(Weak::upgrade)
        .filter_map(|r| r.limit().map(|l| (r, l)))
        .collect();
    let Some(resident) = resident else {
        // Unknown: no cut rather than a stale one.
        g.high_water = None;
        g.backoff = Backoff::NONE;
        for (root, _) in &limited {
            root.shared.cut.store(0, Ordering::Release);
        }
        return;
    };
    let reserved = limited
        .iter()
        .map(|(r, _)| r.reserved())
        .fold(0, usize::saturating_add);
    let allowances = limited
        .iter()
        .map(|(_, l)| allowance(*l))
        .fold(0, usize::saturating_add);
    let (excess, high_water) = excess(resident, reserved, allowances, g.high_water);
    g.high_water = high_water;
    let total_limit = limited
        .iter()
        .map(|(_, l)| *l)
        .fold(0, usize::saturating_add);
    for (root, limit) in &limited {
        root.shared
            .cut
            .store(cut(excess, *limit, total_limit), Ordering::Release);
    }
    let release = g.backoff.should_release(excess, now);
    drop(g);
    if release {
        crate::release::note_pressure();
    }
}

/// The excess of `resident` over the roots' reservations plus their
/// `allowances`, and the reservation high-water to keep for the next
/// sample. During an excess (`high_water` set) the reservations count at
/// their high-water since it began, so spilling, which lowers the
/// reservations while the allocator may keep the freed pages, cannot raise
/// the excess; once there is none, the high-water is dropped.
fn excess(
    resident: usize,
    reserved: usize,
    allowances: usize,
    high_water: Option<usize>,
) -> (usize, Option<usize>) {
    let base = high_water.map_or(reserved, |h| h.max(reserved));
    let excess = resident.saturating_sub(base.saturating_add(allowances));
    (excess, (excess > 0).then_some(base))
}

/// A root's cut: its share of `excess` in proportion to its `limit` out of
/// `total_limit`, at most a quarter of its limit.
fn cut(excess: usize, limit: usize, total_limit: usize) -> usize {
    if total_limit == 0 {
        return 0;
    }
    let share = (excess as u128) * (limit as u128) / (total_limit as u128);
    usize::try_from(share)
        .unwrap_or(usize::MAX)
        .min(limit / MAX_CUT_DIVISOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vm_rss_reads_the_status_line_in_bytes() {
        let status = b"Name:\tx\nVmHWM:\t  9000 kB\nVmRSS:\t  1234 kB\nRssAnon:\t 1 kB\n";
        assert_eq!(vm_rss(status), Some(1234 * 1024));
        assert_eq!(vm_rss(b"Name:\tx\n"), None);
        assert_eq!(vm_rss(b"VmRSS:\t kB\n"), None);
        assert_eq!(vm_rss(b"VmRSS:\t 99999999999999999999999 kB\n"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_process_resident_set_is_read() {
        let rss = process_resident_bytes().unwrap();
        // At least the test binary's own pages, and below a terabyte.
        assert!(rss > 1 << 20 && rss < 1 << 40, "{rss}");
    }

    #[test]
    fn the_allowance_is_an_eighth_of_the_limit_plus_the_base() {
        assert_eq!(allowance(2 << 30), (256 << 20) + ALLOWANCE_BASE);
        assert_eq!(allowance(0), ALLOWANCE_BASE);
        assert_eq!(allowance(usize::MAX), usize::MAX / 8 + ALLOWANCE_BASE);
    }

    #[test]
    fn the_excess_is_what_reservations_and_allowances_do_not_cover() {
        let (gib, al) = (1usize << 30, allowance(2 << 30));
        assert_eq!(excess(gib + al + 5, gib, al, None), (5, Some(gib)));
        assert_eq!(excess(gib + al, gib, al, None), (0, None));
        assert_eq!(excess(10, gib, al, None), (0, None));
        assert_eq!(excess(usize::MAX, usize::MAX, al, None), (0, None));
    }

    #[test]
    fn a_constant_excess_starts_few_epochs() {
        let mut b = Backoff::NONE;
        let excess = 1usize << 30;
        // One sample a millisecond for ten seconds, the excess unchanged
        // (give or take noise): 1, 2, 4, ... 1024 ms apart, then every
        // 1024 ms, instead of 10,000 epochs.
        let epochs = (1..=10_000u64)
            .filter(|&ms| b.should_release(excess + usize::try_from(ms % 7).unwrap() * 4096, ms))
            .count();
        assert!(epochs <= 20, "{epochs}");
        assert!(epochs >= 10, "{epochs}");
    }

    #[test]
    fn a_rising_or_falling_excess_releases_at_once() {
        let mut b = Backoff::NONE;
        let mib = 1usize << 20;
        assert!(b.should_release(100 * mib, 1));
        // Within the back-off and unchanged: no epoch.
        assert!(!b.should_release(100 * mib, 1));
        for _ in 0..12 {
            let at = b.next_at;
            assert!(b.should_release(100 * mib, at));
        }
        assert_eq!(b.step_ms, MAX_BACKOFF_MS);
        // A rise past the step: at once, and the back-off restarts.
        let now = b.next_at - 1;
        assert!(b.should_release(100 * mib + EXCESS_STEP, now));
        assert_eq!(b.step_ms, 1);
        // A small rise is noise.
        assert!(!b.should_release(100 * mib + EXCESS_STEP + mib, now));
        // The release helped (a new low past the step): at once again.
        assert!(b.should_release(80 * mib, now));
        // Back inside the range covered so far: the back-off decides.
        assert!(!b.should_release(100 * mib, now));
        // No excess: the next one starts fresh.
        assert!(!b.should_release(0, now + 1));
        assert_eq!(b, Backoff::NONE);
        assert!(b.should_release(5 * mib, now + 1));
    }

    /// The guard's own release feeding back: an excess no release can lower
    /// (`base`) plus `kept` that the allocator holds, which each release
    /// purges by the next sample and the query refills at `refill` a
    /// sample, up to `kept`. Ten seconds at one sample a millisecond.
    fn epochs_with_feedback(kept: usize, refill: usize) -> usize {
        let base = 1300usize << 20;
        let mut b = Backoff::NONE;
        let mut held = kept;
        let mut epochs = 0;
        for ms in 1..=10_000u64 {
            if b.should_release(base + held, ms) {
                epochs += 1;
                held = 0;
            } else {
                held = (held + refill).min(kept);
            }
        }
        epochs
    }

    #[test]
    fn the_guards_own_purge_and_refill_start_few_epochs() {
        let mib = 1usize << 20;
        for kept in [16 * mib, 64 * mib, 256 * mib] {
            for refill in [mib / 4, mib, 4 * mib, 16 * mib] {
                let epochs = epochs_with_feedback(kept, refill);
                assert!(epochs <= 25, "kept {kept} refill {refill}: {epochs}");
            }
        }
    }

    #[test]
    fn a_real_rise_still_releases_at_once() {
        let (mib, base) = (1usize << 20, 100usize << 20);
        let mut b = Backoff::NONE;
        // The excess grows by 1 MiB a millisecond for ten seconds and no
        // release lowers it: an epoch at least every 16 ms (each new high).
        let epochs = (1..=10_000u64)
            .filter(|&ms| b.should_release(base + usize::try_from(ms).unwrap() * mib, ms))
            .count();
        assert!(epochs >= 10_000 / 16, "{epochs}");
    }

    #[test]
    fn spilling_during_an_excess_does_not_raise_it() {
        let (gib, mib, al) = (1usize << 30, 1usize << 20, allowance(2 << 30));
        // 100 MiB over: the episode starts at 1 GiB reserved.
        let (e, hw) = excess(gib + al + 100 * mib, gib, al, None);
        assert_eq!((e, hw), (100 * mib, Some(gib)));
        // Spilling frees 300 MiB of reservations, the pages stay resident:
        // the excess stays at 100 MiB instead of growing to 400.
        let (e, hw) = excess(gib + al + 100 * mib, gib - 300 * mib, al, hw);
        assert_eq!((e, hw), (100 * mib, Some(gib)));
        // The allocator returns 60 MiB: the excess shrinks.
        let (e, hw) = excess(gib + al + 40 * mib, gib - 300 * mib, al, hw);
        assert_eq!((e, hw), (40 * mib, Some(gib)));
        // Reservations above the high-water raise it.
        let (e, hw) = excess(gib + al + 140 * mib, gib + 100 * mib, al, hw);
        assert_eq!((e, hw), (40 * mib, Some(gib + 100 * mib)));
        // Back within the allowance: the episode ends.
        let (e, hw) = excess(gib, gib - 300 * mib, al, hw);
        assert_eq!((e, hw), (0, None));
        // A new episode measures from the current reservations.
        let (e, _) = excess(gib + al, gib - 300 * mib, al, hw);
        assert_eq!(e, 300 * mib);
    }

    #[test]
    fn the_cut_is_a_share_by_limit_capped_at_a_quarter() {
        assert_eq!(cut(100 << 20, 2 << 30, 2 << 30), 100 << 20);
        assert_eq!(cut(1 << 40, 2 << 30, 2 << 30), (2 << 30) / 4);
        // Two roots, 3:1 by limit.
        assert_eq!(cut(400 << 20, 3 << 30, 4 << 30), 300 << 20);
        assert_eq!(cut(400 << 20, 1 << 30, 4 << 30), 100 << 20);
        assert_eq!(cut(5, 1 << 30, 0), 0);
        assert_eq!(cut(0, 1 << 30, 1 << 30), 0);
    }
}
