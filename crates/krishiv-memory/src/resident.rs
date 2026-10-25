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
//! ([`set_resident_probe`]), the process's resident set is sampled at most
//! once per [`SAMPLE_INTERVAL`], on a grant that needs more bytes from the
//! pools. What it holds above the reservations of every limited root plus
//! their allowances ([`allowance`]: an eighth of the limit plus
//! [`ALLOWANCE_BASE`]) is the *excess*. Each root's share of it, in
//! proportion to its limit and at most a quarter of the limit, is cut from
//! what ordinary reservations may take there, so consumers spill until the
//! process is back within the reservations plus the allowance, and threads
//! return their allocator caches (a pressure epoch, see
//! [`set_release_hook`](crate::set_release_hook)). Spill scratch keeps the
//! whole limit, and a request that only the cut refuses is granted before
//! arbitration would abort a query or report it out of memory: the guard
//! makes queries spill, never fail. The cut does not change a pool's
//! [`limit`](crate::MemoryPool::limit), so plans and budgets sized from the
//! limit stay the same.
//!
//! The probe belongs to the program that owns the allocator, like the
//! release hook: the resident set only tracks the pools' reservations when
//! the allocator returns freed memory promptly (`krishiv_alloc`'s settings
//! under a limit below its generous threshold). An allocator that keeps
//! freed memory for a second would make the guard spill for memory that is
//! about to be returned. Without a probe nothing is sampled or cut.

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

static PROBE: OnceLock<fn() -> Option<usize>> = OnceLock::new();

/// Milliseconds since [`start`] at the last sample (0: none yet).
static LAST: AtomicU64 = AtomicU64::new(0);

/// Every limited root pool of the process (pruned when sampled).
static ROOTS: std::sync::Mutex<Vec<Weak<Node>>> = std::sync::Mutex::new(Vec::new());

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
    let mut roots = ROOTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    roots.retain(|r| r.strong_count() > 0);
    roots.push(Arc::downgrade(root));
}

/// Samples the resident set and updates every root's cut, unless no probe
/// is installed or a sample was taken within [`SAMPLE_INTERVAL`].
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
    let Some(resident) = probe() else {
        return;
    };
    let roots: Vec<Arc<Node>> = {
        let mut roots = ROOTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        roots.retain(|r| r.strong_count() > 0);
        roots.iter().filter_map(Weak::upgrade).collect()
    };
    let limited: Vec<(&Arc<Node>, usize)> = roots
        .iter()
        .filter_map(|r| r.limit().map(|l| (r, l)))
        .collect();
    let excess = excess(resident, limited.iter().map(|(r, l)| (r.reserved(), *l)));
    let total_limit: usize = limited
        .iter()
        .map(|(_, l)| *l)
        .fold(0, usize::saturating_add);
    for (root, limit) in &limited {
        root.shared
            .cut
            .store(cut(excess, *limit, total_limit), Ordering::Release);
    }
    if excess > 0 {
        crate::release::note_pressure();
    }
}

/// What `resident` holds above the roots' reservations plus allowances
/// (`roots`: each root's reserved bytes and limit).
fn excess(resident: usize, roots: impl Iterator<Item = (usize, usize)>) -> usize {
    let covered = roots.fold(0usize, |acc, (reserved, limit)| {
        acc.saturating_add(reserved)
            .saturating_add(allowance(limit))
    });
    resident.saturating_sub(covered)
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
        let limit = 2 << 30;
        let covered = (1 << 30) + allowance(limit);
        assert_eq!(excess(covered + 5, [(1 << 30, limit)].into_iter()), 5);
        assert_eq!(excess(covered, [(1 << 30, limit)].into_iter()), 0);
        assert_eq!(excess(10, [(1 << 30, limit)].into_iter()), 0);
        // Two roots: both reservations and both allowances count.
        let two = (1 << 30) + (1 << 20) + allowance(limit) + allowance(1 << 30);
        assert_eq!(
            excess(two + 7, [(1 << 30, limit), (1 << 20, 1 << 30)].into_iter()),
            7
        );
        assert_eq!(excess(usize::MAX, [(usize::MAX, limit)].into_iter()), 0);
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
