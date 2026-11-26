// SPDX-License-Identifier: Apache-2.0
//! The default memory limit: 80% of the effective memory limit of this
//! process, lowered so the resident-set guard's ceiling stays at least
//! [`MIN_HEADROOM`] below it (but never under half of it),
//! so a query engine embedded in a notebook or container spills instead of
//! being OOM-killed.
//!
//! The effective limit is the smallest of the cgroup limits on the
//! process's cgroup path (v2 `memory.max`, else v1
//! `memory.limit_in_bytes`, each walked up to the hierarchy root) and the
//! physical memory (`MemTotal` in `/proc/meminfo`). On other platforms, or if
//! none of this can be read, the default is [`FALLBACK_LIMIT`]: a
//! conservative fixed size, documented on `krishiv.memory.limit`.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::resident::ALLOWANCE_BASE;

/// Percentage of the effective memory limit used by default.
pub const DEFAULT_LIMIT_PERCENT: u64 = 80;

/// Memory the default limit leaves below the effective limit beyond the
/// resident-set guard's ceiling (the limit plus its
/// [`allowance`](crate::allowance): decode and I/O buffers, allocator
/// retention), at least: the page cache, dirty spill pages and kernel memory
/// a cgroup also counts. 20% alone is too little on a small container
/// (0.3.0 acceptance QA2-12): in a 1 GiB cgroup an 819 MiB pool let the
/// process hold up to 985 MiB, and a 41M-row self-join or a high-cardinality
/// group-by was OOM-killed; in a 2 GiB cgroup a 1.5 GiB pool (a 1.75 GiB
/// ceiling) filled the cgroup on every run of that group-by and was killed
/// once, while a 1.39 GiB pool peaked at 1.5-1.7 GiB.
pub const MIN_HEADROOM: u64 = 384 << 20;

/// Default limit when the effective memory limit cannot be determined.
pub const FALLBACK_LIMIT: usize = 2 << 30;

/// cgroup v1 reports "no limit" as a huge page-aligned value.
const V1_UNLIMITED_FROM: u64 = 1 << 62;

/// The default root-pool limit: [`DEFAULT_LIMIT_PERCENT`] of
/// [`effective_memory_limit`], lowered so the guard's ceiling stays
/// [`MIN_HEADROOM`] below it but not under half of it, else
/// [`FALLBACK_LIMIT`]. Always `Some`; pass
/// `limit: None` in [`crate::MemoryConfig`] for an unbounded pool.
pub fn default_limit() -> Option<usize> {
    Some(effective_memory_limit().map_or(FALLBACK_LIMIT, share_of))
}

/// The pool's share of an effective limit of `bytes`:
/// [`DEFAULT_LIMIT_PERCENT`] of it, lowered so that the limit `L` plus the
/// guard's allowance (`L / 8` + [`ALLOWANCE_BASE`]) leaves [`MIN_HEADROOM`]
/// (from about 4.4 GiB up the percentage already does), but never under
/// half.
fn share_of(bytes: u64) -> usize {
    let percent = bytes / 100 * DEFAULT_LIMIT_PERCENT;
    let base = u64::try_from(ALLOWANCE_BASE).unwrap_or(u64::MAX);
    // L + L / 8 + base <= bytes - MIN_HEADROOM.
    let guarded = bytes.saturating_sub(MIN_HEADROOM).saturating_sub(base) / 9 * 8;
    let share = percent.min(guarded).max(bytes / 2);
    usize::try_from(share).unwrap_or(usize::MAX)
}

/// The root-pool limit for an explicit `krishiv.memory.limit` of `explicit`
/// bytes: `explicit` itself, unless it is above [`effective_memory_limit`].
/// Then the pool could grow past what the kernel allows and the process would
/// be OOM-killed without an error before the engine ever spills, so the limit
/// is [`default_limit`] instead, with a warning (logged once per requested
/// value, not on every call: the setting is read again by each admin check).
///
/// A value at or just under the effective limit stands as given. It leaves
/// no room for what the process holds outside the pool (the allocator's
/// overhead, memory used without a reservation), so the kernel can still
/// kill such a process; `krishiv.memory.limit` says so.
pub fn explicit_limit(explicit: usize) -> usize {
    let fitted = fitted_limit(explicit, effective_memory_limit());
    if fitted != explicit && first_warning(explicit) {
        tracing::warn!(
            target: "krishiv::memory",
            requested = explicit,
            limit = fitted,
            "krishiv.memory.limit is above the memory this process may use (cgroup limit or \
             physical memory); using the default share of that instead"
        );
    }
    fitted
}

/// The requested limit the over-limit warning was last logged for (0: none).
static LAST_WARNED: AtomicUsize = AtomicUsize::new(0);

/// Whether the over-limit warning for `requested` is not a repeat of the
/// last one logged; records `requested` as logged.
fn first_warning(requested: usize) -> bool {
    LAST_WARNED.swap(requested, Ordering::Relaxed) != requested
}

/// [`explicit_limit`] against an `effective` memory limit (tests).
fn fitted_limit(explicit: usize, effective: Option<u64>) -> usize {
    match effective {
        Some(bytes) if explicit as u64 > bytes => share_of(bytes),
        _ => explicit,
    }
}

/// The memory this process may use: the tightest cgroup limit, else the
/// physical memory. `None` if neither can be read (non-Linux platforms).
pub fn effective_memory_limit() -> Option<u64> {
    effective_memory_limit_at(Path::new("/"))
}

/// [`effective_memory_limit`] with `/proc` and `/sys` under `root` (tests).
pub(crate) fn effective_memory_limit_at(root: &Path) -> Option<u64> {
    let read = |p: &str| std::fs::read_to_string(root.join(p)).ok();
    let physical = read("proc/meminfo").as_deref().and_then(meminfo_total);
    let cgroup = read("proc/self/cgroup")
        .as_deref()
        .and_then(|text| cgroup_limit(root, text));
    match (cgroup, physical) {
        (Some(c), Some(p)) => Some(c.min(p)),
        (c, p) => c.or(p),
    }
}

/// `MemTotal` in bytes.
fn meminfo_total(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let mut parts = line.split_whitespace().skip(1);
    let value: u64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some("kB") => value.checked_mul(1024),
        None => Some(value),
        Some(_) => None,
    }
}

/// The tightest limit on the process's cgroup path, v2 first, then v1.
fn cgroup_limit(root: &Path, proc_self_cgroup: &str) -> Option<u64> {
    for line in proc_self_cgroup.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if id == "0" && controllers.is_empty() {
            if let Some(l) = walk_limits(&root.join("sys/fs/cgroup"), path, "memory.max", u64::MAX)
            {
                return Some(l);
            }
        } else if controllers.split(',').any(|c| c == "memory")
            && let Some(l) = walk_limits(
                &root.join("sys/fs/cgroup/memory"),
                path,
                "memory.limit_in_bytes",
                V1_UNLIMITED_FROM,
            )
        {
            return Some(l);
        }
    }
    None
}

/// The smallest numeric `file` from `mount/path` up to `mount`; `max` or
/// values at or above `unlimited_from` mean no limit.
fn walk_limits(mount: &Path, path: &str, file: &str, unlimited_from: u64) -> Option<u64> {
    let mut dir = mount.join(path.trim_start_matches('/'));
    let mut best: Option<u64> = None;
    loop {
        if let Ok(text) = std::fs::read_to_string(dir.join(file))
            && let Ok(v) = text.trim().parse::<u64>()
            && v < unlimited_from
        {
            best = Some(best.map_or(v, |b| b.min(v)));
        }
        if dir == mount || !dir.pop() || !dir.starts_with(mount) {
            return best;
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn the_over_limit_warning_is_logged_once_per_requested_value() {
        // No other test logs for these values, and an over-limit value on a real host is far larger.
        assert!(first_warning(7));
        assert!(!first_warning(7));
        assert!(first_warning(9));
        assert!(first_warning(7));
    }

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/memory-limit")
            .join(name)
    }

    #[test]
    fn cgroup_v2_takes_the_tightest_limit_on_the_path() {
        // pod memory.max = 2 GiB, parent = 1 GiB, MemTotal = 16 GiB.
        assert_eq!(
            effective_memory_limit_at(&fixture("cgroup-v2")),
            Some(1 << 30)
        );
    }

    #[test]
    fn cgroup_v2_max_falls_back_to_physical_memory() {
        assert_eq!(
            effective_memory_limit_at(&fixture("cgroup-v2-unlimited")),
            Some(8 << 30)
        );
    }

    #[test]
    fn cgroup_v1_limit_and_unlimited_marker() {
        assert_eq!(
            effective_memory_limit_at(&fixture("cgroup-v1")),
            Some(3 << 30)
        );
        assert_eq!(
            effective_memory_limit_at(&fixture("cgroup-v1-unlimited")),
            Some(4 << 30)
        );
    }

    #[test]
    fn physical_memory_only_and_nothing_readable() {
        assert_eq!(
            effective_memory_limit_at(&fixture("meminfo-only")),
            Some(16 << 30)
        );
        assert_eq!(effective_memory_limit_at(&fixture("does-not-exist")), None);
    }

    #[test]
    fn the_share_leaves_headroom_on_small_limits() {
        let mib = 1_usize << 20;
        let gib = 1_u64 << 30;
        // The guard's ceiling (the limit, an eighth more and 64 MiB) stays
        // at least the headroom below the effective limit.
        let ceiling = |l: usize| l + crate::allowance(l);
        for b in [gib, 2 * gib, 5 * gib / 2, 4 * gib, 16 * gib] {
            let room = usize::try_from(b - (MIN_HEADROOM)).unwrap();
            assert!(ceiling(share_of(b)) <= room, "{b}");
        }
        // 1 GiB (QA2-12): 512 MiB, not 819 MiB.
        assert_eq!(share_of(gib), 512 * mib);
        // 2 GiB: about 1.39 GiB, not 1.6 GiB; 2.5 GiB: about 1.83 GiB.
        assert_eq!(share_of(2 * gib), (1600 * mib) / 9 * 8);
        assert_eq!(share_of(5 * gib / 2), (2112 * mib) / 9 * 8);
        // From about 4.4 GiB the 20% is at least the guard's room: 80% as
        // before.
        let eighty = |b: u64| usize::try_from(b / 100 * 80).unwrap();
        assert!(share_of(4 * gib) < eighty(4 * gib));
        assert_eq!(share_of(5 * gib), eighty(5 * gib));
        assert_eq!(share_of(16 * gib), eighty(16 * gib));
        // Small limits keep half rather than nothing.
        assert_eq!(share_of(512 << 20), 256 * mib);
        assert_eq!(share_of(768 << 20), 384 * mib);
        assert_eq!(share_of(0), 0);
    }

    #[test]
    fn meminfo_parsing() {
        assert_eq!(
            meminfo_total("MemTotal:  1024 kB\nMemFree: 1 kB"),
            Some(1 << 20)
        );
        assert_eq!(meminfo_total("MemFree: 1 kB"), None);
        assert_eq!(meminfo_total("MemTotal: 12 MB"), None);
    }

    /// db-benchmark 1e8 q10: `krishiv.memory.limit=24g` in a 20 GiB cgroup let
    /// the pool grow until the kernel killed the process without an error.
    #[test]
    fn an_explicit_limit_above_the_effective_limit_is_fitted_to_it() {
        let gib = 1_usize << 30;
        let cap = 20_u64 << 30;
        assert_eq!(fitted_limit(24 * gib, Some(cap)), share_of(cap));
        assert!(share_of(cap) > 15 * gib && share_of(cap) <= 16 * gib);
        // At or below the effective limit the setting stands, even above 80%.
        assert_eq!(fitted_limit(18 * gib, Some(cap)), 18 * gib);
        assert_eq!(fitted_limit(20 * gib, Some(cap)), 20 * gib);
        // Nothing readable: the setting stands.
        assert_eq!(fitted_limit(24 * gib, None), 24 * gib);
    }

    #[test]
    fn default_limit_is_the_share_or_the_fallback() {
        let limit = default_limit().unwrap_or(0);
        match effective_memory_limit() {
            Some(b) => assert_eq!(limit, share_of(b)),
            None => assert_eq!(limit, FALLBACK_LIMIT),
        }
    }
}
