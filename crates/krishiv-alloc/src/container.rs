// SPDX-License-Identifier: Apache-2.0
//! The memory this process may use, read without allocating, so the
//! `.init_array` constructor can size the allocator policy from it before
//! mimalloc reserves its first arena.
//!
//! The same rule as `krishiv_memory::effective_memory_limit` (which the
//! default `krishiv.memory.limit` is a share of): the tightest cgroup limit
//! on the process's cgroup path (v2 `memory.max`, else v1
//! `memory.limit_in_bytes`, each walked up to the hierarchy root), and the
//! physical memory (`MemTotal`), whichever is smaller. That crate cannot be
//! used here: it is a higher layer, and it reads with `String`s.
//!
//! Files are read through a callback into stack buffers; a path longer than
//! [`PATH_MAX`] or a file longer than its buffer is cut short (a cut
//! `/proc/self/cgroup` is read up to its last whole line), never allocated.

use std::ffi::CStr;

/// The longest path built, NUL included; a longer one is not read.
const PATH_MAX: usize = 512;

/// The bytes read of `/proc/self/cgroup` and `/proc/meminfo` (`MemTotal`
/// is its first line).
const TEXT_MAX: usize = 4096;

/// cgroup v1 reports "no limit" as a huge page-aligned value.
const V1_UNLIMITED_FROM: u64 = 1 << 62;

/// Reads the file at a path into a buffer: the bytes read, `None` if it
/// cannot be opened or read.
pub(crate) type Read<'a> = &'a mut dyn FnMut(&CStr, &mut [u8]) -> Option<usize>;

/// The default `krishiv.memory.limit` is 80% of the effective limit at most
/// (`krishiv_memory`'s `share_of`; less only below about 4.4 GiB), and
/// exactly 80% from there up, so it is generous ([`crate::is_generous`])
/// exactly when 80% of the effective limit is.
pub(crate) fn default_is_generous(effective: u64) -> bool {
    crate::is_generous(effective / 100 * 80)
}

/// The memory this process may use, `None` if neither a cgroup limit nor
/// the physical memory can be read.
pub(crate) fn effective_limit(read: Read<'_>) -> Option<u64> {
    let mut text = [0u8; TEXT_MAX];
    let physical = read_text(read, b"/proc/meminfo", &mut text).and_then(meminfo_total);
    let cgroup =
        read_text(read, b"/proc/self/cgroup", &mut text).and_then(|t| cgroup_limit(read, t));
    match (cgroup, physical) {
        (Some(c), Some(p)) => Some(c.min(p)),
        (c, p) => c.or(p),
    }
}

/// The text of `path` up to its last whole line (all of it when it fits).
fn read_text<'b>(read: Read<'_>, path: &[u8], buf: &'b mut [u8]) -> Option<&'b [u8]> {
    let mut p = PathBuf::new();
    p.push(path)?;
    let n = read(p.as_cstr()?, buf)?.min(buf.len());
    let text = &buf[..n];
    if n < buf.len() {
        return Some(text);
    }
    let end = text.iter().rposition(|&b| b == b'\n')?;
    Some(&text[..end])
}

/// `MemTotal` in bytes.
fn meminfo_total(text: &[u8]) -> Option<u64> {
    let line = text
        .split(|&b| b == b'\n')
        .find_map(|l| l.strip_prefix(b"MemTotal:"))?;
    let mut parts = line
        .split(u8::is_ascii_whitespace)
        .filter(|p| !p.is_empty());
    let value = number(parts.next()?)?;
    match parts.next() {
        Some(b"kB") => value.checked_mul(1024),
        None => Some(value),
        Some(_) => None,
    }
}

/// The tightest limit on the process's cgroup path, v2 first, then v1.
fn cgroup_limit(read: Read<'_>, proc_self_cgroup: &[u8]) -> Option<u64> {
    for line in proc_self_cgroup.split(|&b| b == b'\n') {
        let mut fields = line.splitn(3, |&b| b == b':');
        let (Some(id), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if id == b"0" && controllers.is_empty() {
            if let Some(l) = walk_limits(read, b"/sys/fs/cgroup", path, b"memory.max", u64::MAX) {
                return Some(l);
            }
        } else if controllers.split(|&b| b == b',').any(|c| c == b"memory")
            && let Some(l) = walk_limits(
                read,
                b"/sys/fs/cgroup/memory",
                path,
                b"memory.limit_in_bytes",
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
fn walk_limits(
    read: Read<'_>,
    mount: &[u8],
    path: &[u8],
    file: &[u8],
    unlimited_from: u64,
) -> Option<u64> {
    let mut dir = PathBuf::new();
    dir.push(mount)?;
    for part in path.split(|&b| b == b'/').filter(|p| !p.is_empty()) {
        dir.push(b"/")?;
        dir.push(part)?;
    }
    let mut best: Option<u64> = None;
    let mut value = [0u8; 64];
    loop {
        let mut f = dir;
        if f.push(b"/").and_then(|()| f.push(file)).is_some()
            && let Some(c) = f.as_cstr()
            && let Some(n) = read(c, &mut value)
            && let Some(v) = number(value[..n.min(value.len())].trim_ascii())
            && v < unlimited_from
        {
            best = Some(best.map_or(v, |b| b.min(v)));
        }
        if dir.len <= mount.len() {
            return best;
        }
        // Up one directory, never above the mount.
        let Some(cut) = dir.bytes[..dir.len].iter().rposition(|&b| b == b'/') else {
            return best;
        };
        dir.len = cut.max(mount.len());
    }
}

/// A decimal `u64`; `None` for anything else (`max`, empty, overflow).
fn number(digits: &[u8]) -> Option<u64> {
    if digits.is_empty() {
        return None;
    }
    let mut n: u64 = 0;
    for &d in digits {
        if !d.is_ascii_digit() {
            return None;
        }
        n = n.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
    }
    Some(n)
}

/// A path on the stack.
#[derive(Clone, Copy)]
struct PathBuf {
    bytes: [u8; PATH_MAX],
    len: usize,
}

impl PathBuf {
    fn new() -> Self {
        Self {
            bytes: [0; PATH_MAX],
            len: 0,
        }
    }

    /// Appends `part`; `None` if the path (with its NUL) would not fit.
    fn push(&mut self, part: &[u8]) -> Option<()> {
        let end = self.len.checked_add(part.len())?;
        if end >= PATH_MAX || part.contains(&0) {
            return None;
        }
        self.bytes[self.len..end].copy_from_slice(part);
        self.len = end;
        Some(())
    }

    /// The path, NUL-terminated in place.
    fn as_cstr(&mut self) -> Option<&CStr> {
        *self.bytes.get_mut(self.len)? = 0;
        CStr::from_bytes_with_nul(&self.bytes[..=self.len]).ok()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// [`effective_limit`] over in-memory `files`.
    fn limit(files: &[(&str, &str)]) -> Option<u64> {
        let files: HashMap<&str, &str> = files.iter().copied().collect();
        let mut read = |p: &CStr, buf: &mut [u8]| {
            let text = files.get(p.to_str().ok()?)?.as_bytes();
            let n = text.len().min(buf.len());
            buf[..n].copy_from_slice(&text[..n]);
            Some(n)
        };
        effective_limit(&mut read)
    }

    const MEMINFO_16G: (&str, &str) = (
        "/proc/meminfo",
        "MemTotal:       16777216 kB\nMemFree: 1 kB\n",
    );

    #[test]
    fn cgroup_v2_takes_the_tightest_limit_on_the_path() {
        let got = limit(&[
            MEMINFO_16G,
            ("/proc/self/cgroup", "0::/kubepods/pod1\n"),
            ("/sys/fs/cgroup/memory.max", "max\n"),
            ("/sys/fs/cgroup/kubepods/memory.max", "1073741824\n"),
            ("/sys/fs/cgroup/kubepods/pod1/memory.max", "2147483648\n"),
        ]);
        assert_eq!(got, Some(1 << 30));
    }

    #[test]
    fn cgroup_v2_max_falls_back_to_physical_memory() {
        let got = limit(&[
            MEMINFO_16G,
            ("/proc/self/cgroup", "0::/\n"),
            ("/sys/fs/cgroup/memory.max", "max\n"),
        ]);
        assert_eq!(got, Some(16 << 30));
        // A limit above the physical memory: the physical memory.
        let got = limit(&[
            MEMINFO_16G,
            ("/proc/self/cgroup", "0::/a.scope"),
            ("/sys/fs/cgroup/a.scope/memory.max", "34359738368"),
        ]);
        assert_eq!(got, Some(16 << 30));
    }

    #[test]
    fn cgroup_v1_reads_the_memory_controller() {
        let got = limit(&[
            MEMINFO_16G,
            (
                "/proc/self/cgroup",
                "12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n",
            ),
            (
                "/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
                "3221225472\n",
            ),
        ]);
        assert_eq!(got, Some(3 << 30));
        let unlimited = limit(&[
            MEMINFO_16G,
            ("/proc/self/cgroup", "4:memory:/\n"),
            (
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "9223372036854771712\n",
            ),
        ]);
        assert_eq!(unlimited, Some(16 << 30));
    }

    #[test]
    fn nothing_readable_or_only_one_source() {
        assert_eq!(limit(&[]), None);
        assert_eq!(limit(&[MEMINFO_16G]), Some(16 << 30));
        assert_eq!(
            limit(&[
                ("/proc/self/cgroup", "0::/s\n"),
                ("/sys/fs/cgroup/s/memory.max", "2147483648\n"),
            ]),
            Some(2 << 30)
        );
        assert_eq!(meminfo_total(b"MemTotal: 12 MB\n"), None);
        assert_eq!(meminfo_total(b"MemTotal: 1024\n"), Some(1024));
    }

    #[test]
    fn a_path_too_long_is_not_read() {
        let long = format!("0::/{}\n", "a".repeat(PATH_MAX));
        let got = limit(&[
            MEMINFO_16G,
            ("/proc/self/cgroup", long.as_str()),
            ("/sys/fs/cgroup/memory.max", "1073741824"),
        ]);
        assert_eq!(got, Some(16 << 30));
    }

    #[test]
    fn a_cut_text_keeps_its_whole_lines() {
        let mut text = "9:cpu:/x\n".repeat(TEXT_MAX / 9);
        text.push_str("0::/s\n");
        let got = limit(&[
            MEMINFO_16G,
            ("/proc/self/cgroup", text.as_str()),
            ("/sys/fs/cgroup/s/memory.max", "1073741824"),
        ]);
        // The v2 line is past the buffer: physical memory.
        assert_eq!(got, Some(16 << 30));
    }

    #[test]
    fn the_default_is_generous_from_about_5_gib() {
        assert!(!default_is_generous(2 << 30));
        assert!(!default_is_generous(4 << 30));
        assert!(!default_is_generous((5 << 30) - (64 << 20)));
        // 80% of 5 GiB rounds below 4 GiB, as `share_of` computes it.
        assert!(!default_is_generous(5 << 30));
        assert!(default_is_generous((5 << 30) + (1 << 20)));
        assert!(default_is_generous(64 << 30));
    }
}
