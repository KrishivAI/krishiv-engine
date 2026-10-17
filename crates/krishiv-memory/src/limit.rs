// SPDX-License-Identifier: Apache-2.0
//! The default memory limit: 80% of the effective memory limit of this
//! process, so a query engine embedded in a notebook or container spills
//! instead of being OOM-killed.
//!
//! The effective limit is the smallest of the cgroup limits on the
//! process's cgroup path (v2 `memory.max`, else v1
//! `memory.limit_in_bytes`, each walked up to the hierarchy root) and the
//! physical memory (`MemTotal` in `/proc/meminfo`). On other platforms, or if
//! none of this can be read, the default is [`FALLBACK_LIMIT`]: a
//! conservative fixed size, documented on `krishiv.memory.limit`.

use std::path::Path;

/// Percentage of the effective memory limit used by default.
pub const DEFAULT_LIMIT_PERCENT: u64 = 80;

/// Default limit when the effective memory limit cannot be determined.
pub const FALLBACK_LIMIT: usize = 2 << 30;

/// cgroup v1 reports "no limit" as a huge page-aligned value.
const V1_UNLIMITED_FROM: u64 = 1 << 62;

/// The default root-pool limit: [`DEFAULT_LIMIT_PERCENT`] of
/// [`effective_memory_limit`], else [`FALLBACK_LIMIT`]. Always `Some`; pass
/// `limit: None` in [`crate::MemoryConfig`] for an unbounded pool.
pub fn default_limit() -> Option<usize> {
    Some(effective_memory_limit().map_or(FALLBACK_LIMIT, |bytes| {
        let limit = bytes / 100 * DEFAULT_LIMIT_PERCENT;
        usize::try_from(limit).unwrap_or(usize::MAX)
    }))
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
    fn meminfo_parsing() {
        assert_eq!(
            meminfo_total("MemTotal:  1024 kB\nMemFree: 1 kB"),
            Some(1 << 20)
        );
        assert_eq!(meminfo_total("MemFree: 1 kB"), None);
        assert_eq!(meminfo_total("MemTotal: 12 MB"), None);
    }

    #[test]
    fn default_limit_is_eighty_percent_or_the_fallback() {
        let limit = default_limit().unwrap_or(0);
        match effective_memory_limit() {
            Some(b) => assert_eq!(limit as u64, b / 100 * DEFAULT_LIMIT_PERCENT),
            None => assert_eq!(limit, FALLBACK_LIMIT),
        }
    }
}
