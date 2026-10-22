// SPDX-License-Identifier: Apache-2.0
//! mimalloc policy for the binaries whose global allocator is mimalloc
//! (`krishiv` CLI, `krishiv-bench-runner`). Libraries, the Python module and
//! embedding hosts never depend on this crate: the host owns its allocator
//! and its policy (P2-T012), and can install its own release hook with
//! `krishiv_memory::set_release_hook`.
//!
//! **Why.** mimalloc 3 keeps freed memory committed for `purge_delay`
//! (1,000 ms by default) and backs its arenas with transparent huge pages,
//! so a query engine's freed memory stays resident: under a 200 MiB limit
//! the resident set ran 2.5-3.3x above the live heap (TPC-H SF1) and
//! processes were OOM-killed under a cgroup cap at the memory bound.
//!
//! **What, only under a tight memory limit.** [`LIMITED_OPTIONS`]: purge
//! after 10 ms (mimalloc 2's default), no huge pages, purge ranges from
//! 64 KiB. They cost speed (TPC-H SF1, 12 threads, ABBA: the 10 ms purge
//! delay 1.11x, no huge pages 1.05x, the 64 KiB minimum none; 1.16x
//! together), so they apply only where the limit needs them: a limit below
//! [`GENEROUS_MIN`], or one whose worst case with mimalloc's defaults does
//! not fit the memory the process may use ([`is_generous`]). A generous
//! limit keeps mimalloc's defaults and relies on [`release`] under
//! pressure, as an unlimited process does.
//!
//! The options must be in force before mimalloc reserves its first arena,
//! at the process's first allocation, before `main` (set from `main` they
//! did not prevent the OOM kills, measured). So they apply only when the
//! limit is known at process start: on Linux with glibc an ELF
//! `.init_array` constructor reads the limit from `KRISHIV_MEMORY_LIMIT`
//! (the documented, reliable way) or the arguments ([`limit_in_args`]:
//! `--memory-limit`, `krishiv.memory.limit=`), and sets each option the
//! user has not set through its `MIMALLOC_*` variable. A limit from a
//! configuration file or a runtime `SET` is still enforced by the memory
//! pools and uses [`release`], but without these options. On other targets
//! (musl, macOS) there is no constructor: set the `MIMALLOC_*` variables of
//! [`LIMITED_OPTIONS`] in the environment instead. [`release`] is the
//! pressure release hook (`mi_collect(true)`).

use std::ffi::{CStr, c_int, c_long};

/// `(environment variable, mimalloc option number, value)` set under a
/// memory limit. The numbers are mimalloc 3's `mi_option_e` positions
/// (`mi_option_purge_delay` = 15, `mi_option_allow_thp` = 43,
/// `mi_option_minimal_purge_size` = 44, in KiB); a test pins them to the
/// linked mimalloc.
pub const LIMITED_OPTIONS: [(&CStr, c_int, c_long); 3] = [
    (c"MIMALLOC_PURGE_DELAY", 15, 10),
    (c"MIMALLOC_ALLOW_THP", 43, 0),
    (c"MIMALLOC_MINIMAL_PURGE_SIZE", 44, 64),
];

/// The smallest memory limit that keeps mimalloc's defaults (4 GiB).
///
/// Below it the process holds the resident-set bound of the memory gate
/// (`1.25 x limit` plus a floor, `krishiv_bench_runner::rss_bound`), which
/// mimalloc's defaults break at any size (TPC-H SF1, 12 threads: peak RSS
/// 1,950 MiB at a full 1 GiB pool, against 1,119 MiB with
/// [`LIMITED_OPTIONS`] and a bound of 1,520 MiB). The memory gate runs at
/// 2 GiB; 4 GiB keeps it, and every limit near it, on the bounded path.
pub const GENEROUS_MIN: u64 = 4 << 30;

/// What mimalloc's defaults may hold beyond twice the limit (1 GiB): the
/// constant part of [`is_generous`]'s worst case.
pub const RETENTION_FLOOR: u64 = 1 << 30;

/// Whether `limit` (bytes) is generous enough to keep mimalloc's defaults:
/// at least [`GENEROUS_MIN`], and with the defaults' worst case,
/// `2 x limit + RETENTION_FLOOR`, within `ceiling`, the memory the process
/// may use ([`memory_ceiling`] on Linux; `None`, unknown, is not
/// generous). The worst case is measured: with the defaults, peak RSS
/// stayed within it at every limit tried (TPC-H SF1, 12 threads: 911 MiB
/// at 200 MiB, 1,600 MiB at 512 MiB, 1,950 MiB at 1 GiB, 2,345 MiB at 2 and
/// 4 GiB). `ceiling` is only called for a limit of at least
/// [`GENEROUS_MIN`].
pub fn is_generous(limit: u64, ceiling: impl FnOnce() -> Option<u64>) -> bool {
    limit >= GENEROUS_MIN
        && ceiling().is_some_and(|c| {
            limit
                .checked_mul(2)
                .and_then(|w| w.checked_add(RETENTION_FLOOR))
                .is_some_and(|worst| worst <= c)
        })
}

/// A size as `krishiv.memory.limit` takes it (`2g`, `512MB`, `1048576`:
/// digits, then an optional unit `b`, `k`, `kb`, `m`, `mb`, `g`, `gb`, `t`,
/// `tb`, `p` or `pb` in any case), in bytes; `None` if it is not one.
/// Allocates nothing, so the constructor can use it.
pub fn parse_size(value: &[u8]) -> Option<u64> {
    let value = value.trim_ascii();
    let split = value
        .iter()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(value.len());
    let (digits, unit) = value.split_at(split);
    if digits.is_empty() {
        return None;
    }
    let shift = unit_shift(unit)?;
    let mut n: u64 = 0;
    for &d in digits {
        n = n.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
    }
    n.checked_mul(1 << shift)
}

/// The power of two a size unit ([`parse_size`]) multiplies by.
fn unit_shift(unit: &[u8]) -> Option<u32> {
    let Some((letter, rest)) = unit.split_first() else {
        return Some(0);
    };
    let shift = match letter.to_ascii_lowercase() {
        b'b' => return rest.is_empty().then_some(0),
        b'k' => 10,
        b'm' => 20,
        b'g' => 30,
        b't' => 40,
        b'p' => 50,
        _ => return None,
    };
    matches!(rest, [] | [b'b' | b'B']).then_some(shift)
}

/// The memory limit program arguments configure, in bytes: a
/// `--memory-limit <size>` or `--memory-limit=<size>` flag (the bench
/// runner) or a `krishiv.memory.limit=<size>` setting (`--conf`). With
/// several, the smallest; a value that is not a size ([`parse_size`]) or a
/// flag without one counts as 0, so it never makes a limit generous.
/// `None` without a limit.
pub fn limit_in_args<'a>(args: impl IntoIterator<Item = &'a [u8]>) -> Option<u64> {
    const KEY: &[u8] = b"krishiv.memory.limit=";
    const FLAG: &[u8] = b"--memory-limit";
    let size = |v: Option<&[u8]>| v.and_then(parse_size).unwrap_or(0);
    let mut smallest: Option<u64> = None;
    let mut args = args.into_iter();
    while let Some(a) = args.next() {
        let found = if a == FLAG {
            Some(size(args.next()))
        } else if let Some(v) = a.strip_prefix(FLAG).and_then(|r| r.strip_prefix(b"=")) {
            Some(size(Some(v)))
        } else {
            a.windows(KEY.len())
                .position(|w| w == KEY)
                .map(|i| size(Some(&a[i + KEY.len()..])))
        };
        if let Some(l) = found {
            smallest = Some(smallest.map_or(l, |s| s.min(l)));
        }
    }
    smallest
}

/// The environment variable that configures a memory limit for the
/// allocator settings (module docs).
pub const LIMIT_VAR: &CStr = c"KRISHIV_MEMORY_LIMIT";

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod init {
    use std::ffi::{CStr, c_char, c_int};

    use super::{
        LIMIT_VAR, LIMITED_OPTIONS, is_generous, limit_in_args, memory_ceiling, parse_size,
    };

    unsafe extern "C" {
        fn getenv(name: *const c_char) -> *const c_char;
    }

    /// Calls `f` with the value of `var` if it is set to anything
    /// non-empty, else with `None`.
    fn with_var<R>(var: &CStr, f: impl FnOnce(Option<&[u8]>) -> R) -> R {
        // SAFETY: `var` is NUL-terminated, and a non-null `getenv` result is
        // a NUL-terminated string; constructors run before any other thread,
        // so nothing modifies the environment concurrently.
        let value = unsafe {
            let value = getenv(var.as_ptr());
            (!value.is_null()).then(|| CStr::from_ptr(value))
        };
        f(value.map(CStr::to_bytes).filter(|v| !v.is_empty()))
    }

    /// Whether `var` is set (to anything non-empty).
    fn is_set(var: &CStr) -> bool {
        with_var(var, |v| v.is_some())
    }

    /// The constructor; allocates nothing (the options must be set before
    /// the first allocation).
    extern "C" fn init(argc: c_int, argv: *const *const c_char, _envp: *const *const c_char) {
        let count = if argv.is_null() {
            0
        } else {
            usize::try_from(argc).unwrap_or(0)
        };
        // SAFETY: glibc (and only glibc, hence the `cfg`) calls
        // `.init_array` functions with the program's `argc`, `argv` and
        // `envp`: `argv` holds `argc` pointers to NUL-terminated strings
        // that live for the whole process.
        let args = (0..count).filter_map(|i| unsafe {
            let p = *argv.add(i);
            (!p.is_null()).then(|| CStr::from_ptr(p).to_bytes())
        });
        // A variable that is not a size counts as 0, as in `limit_in_args`.
        let from_var = with_var(LIMIT_VAR, |v| v.map(|v| parse_size(v).unwrap_or(0)));
        let limit = match (from_var, limit_in_args(args)) {
            (Some(a), Some(b)) => a.min(b),
            (a, b) => match a.or(b) {
                Some(l) => l,
                None => return,
            },
        };
        if is_generous(limit, memory_ceiling) {
            return;
        }
        for &(var, option, value) in &LIMITED_OPTIONS {
            if !is_set(var) {
                // SAFETY: only stores the value in mimalloc's option table;
                // no other thread exists yet.
                unsafe { libmimalloc_sys::mi_option_set(option, value) };
            }
        }
    }

    /// Runs [`init`] before the Rust runtime's first allocation (a reference
    /// from [`super::install`] keeps it linked).
    #[used]
    #[unsafe(link_section = ".init_array")]
    pub(super) static INIT: extern "C" fn(c_int, *const *const c_char, *const *const c_char) = init;

    #[cfg(test)]
    mod tests {
        /// Without arguments (a null `argv`) and without the variable, the
        /// constructor sets nothing and does not read `argv`.
        #[test]
        fn a_null_argv_sets_nothing() {
            // (The test process has no `KRISHIV_MEMORY_LIMIT`, as the other
            // tests that read mimalloc's defaults assume.)
            super::init(0, std::ptr::null(), std::ptr::null());
            // SAFETY: plain getters.
            let got = unsafe {
                (
                    libmimalloc_sys::mi_option_get(15),
                    libmimalloc_sys::mi_option_get(43),
                    libmimalloc_sys::mi_option_get(44),
                )
            };
            assert_eq!(got, (1000, 1, 0));
        }
    }
}

#[cfg(target_os = "linux")]
mod ceiling {
    use super::parse_size;

    pub(crate) fn memory_ceiling() -> Option<u64> {
        let mut buf = [0u8; 256];
        let physical = meminfo_total(read_file(b"/proc/meminfo", &mut buf, false)?)?;
        let mut cgroups = [0u8; 1024];
        let path = cgroup_v2_path(read_file(b"/proc/self/cgroup", &mut cgroups, true)?);
        let cgroup = match path {
            Some(path) => cgroup_limit(b"/sys/fs/cgroup", path)?,
            None => None,
        };
        Some(cgroup.map_or(physical, |c| c.min(physical)))
    }

    /// The longest path [`read_file`] opens: the standard library builds a
    /// shorter one's C string on the stack (it allocates from 384 bytes).
    const MAX_PATH: usize = 320;

    /// Reads `path` into `buf`, returning the bytes read: the start of the
    /// file, or with `whole` the whole file (`None` if it fills `buf`).
    fn read_file<'b>(path: &[u8], buf: &'b mut [u8], whole: bool) -> Option<&'b [u8]> {
        use std::io::Read;
        use std::os::unix::ffi::OsStrExt;
        if path.len() > MAX_PATH {
            return None;
        }
        let mut file = std::fs::File::open(std::ffi::OsStr::from_bytes(path)).ok()?;
        let mut n = 0;
        while n < buf.len() {
            match file.read(&mut buf[n..]) {
                Ok(0) => break,
                Ok(k) => n += k,
                Err(_) => return None,
            }
        }
        if whole && n == buf.len() {
            return None;
        }
        Some(&buf[..n])
    }

    /// `MemTotal` of a `/proc/meminfo` text, in bytes.
    fn meminfo_total(text: &[u8]) -> Option<u64> {
        let line = text
            .split(|&b| b == b'\n')
            .find_map(|l| l.strip_prefix(b"MemTotal:"))?;
        let kib = line.trim_ascii().strip_suffix(b"kB")?;
        parse_size(kib)?.checked_mul(1024)
    }

    /// The cgroup v2 path (`0::<path>`) of a `/proc/self/cgroup` text.
    fn cgroup_v2_path(text: &[u8]) -> Option<&[u8]> {
        text.split(|&b| b == b'\n')
            .find_map(|l| l.strip_prefix(b"0::"))
            .filter(|p| p.starts_with(b"/"))
    }

    /// The tightest `memory.max` of `path` and its ancestors under `root`:
    /// `Some(None)` if none is set, `None` if a path is too long to read.
    fn cgroup_limit(root: &[u8], path: &[u8]) -> Option<Option<u64>> {
        const FILE: &[u8] = b"/memory.max";
        let mut tightest: Option<u64> = None;
        let mut dir = path.strip_suffix(b"/").unwrap_or(path);
        loop {
            let len = root.len() + dir.len() + FILE.len();
            if len > MAX_PATH {
                return None;
            }
            let mut name = [0u8; MAX_PATH];
            name[..root.len()].copy_from_slice(root);
            name[root.len()..root.len() + dir.len()].copy_from_slice(dir);
            name[root.len() + dir.len()..len].copy_from_slice(FILE);
            let mut buf = [0u8; 32];
            // `max` (no limit) and a missing file (the root) are not sizes.
            if let Some(l) = read_file(&name[..len], &mut buf, true).and_then(parse_size) {
                tightest = Some(tightest.map_or(l, |t| t.min(l)));
            }
            match dir.iter().rposition(|&b| b == b'/') {
                Some(i) => dir = &dir[..i],
                None => return Some(tightest),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn meminfo_total_is_the_kib_line_in_bytes() {
            let text = b"MemTotal:       65843204 kB\nMemFree:  1 kB\n";
            assert_eq!(meminfo_total(text), Some(65_843_204 * 1024));
            assert_eq!(meminfo_total(b"MemFree: 1 kB\nMemTotal: 2 kB"), Some(2048));
            assert_eq!(meminfo_total(b"MemTotal: 12 MB\n"), None);
            assert_eq!(meminfo_total(b"MemTotal: lots kB\n"), None);
            assert_eq!(meminfo_total(b"MemFree: 1 kB\n"), None);
        }

        #[test]
        fn the_cgroup_path_is_the_v2_line() {
            let hybrid = b"12:memory:/user.slice\n0::/user.slice/app.scope\n";
            assert_eq!(cgroup_v2_path(hybrid), Some(&b"/user.slice/app.scope"[..]));
            assert_eq!(cgroup_v2_path(b"0::/\n"), Some(&b"/"[..]));
            assert_eq!(cgroup_v2_path(b"12:memory:/user.slice\n"), None);
            assert_eq!(cgroup_v2_path(b"0::relative\n"), None);
        }

        /// A cgroup tree under a temporary directory, removed on drop.
        struct Tree(std::path::PathBuf);

        impl Tree {
            fn new(name: &str) -> Self {
                let root = std::env::temp_dir()
                    .join(format!("krishiv-alloc-{}-{name}", std::process::id()));
                let _ = std::fs::remove_dir_all(&root);
                std::fs::create_dir_all(root.join("a/b/c")).unwrap();
                Self(root)
            }

            fn set(&self, dir: &str, max: &str) {
                std::fs::write(self.0.join(dir).join("memory.max"), max).unwrap();
            }

            fn limit(&self, path: &[u8]) -> Option<Option<u64>> {
                use std::os::unix::ffi::OsStrExt;
                cgroup_limit(self.0.as_os_str().as_bytes(), path)
            }
        }

        impl Drop for Tree {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        #[test]
        fn the_cgroup_limit_is_the_tightest_on_the_path() {
            let tree = Tree::new("tightest");
            assert_eq!(tree.limit(b"/a/b/c"), Some(None));
            tree.set("a/b/c", "max\n");
            tree.set("a/b", "3000000000\n");
            tree.set("a", "4000000000\n");
            assert_eq!(tree.limit(b"/a/b/c"), Some(Some(3_000_000_000)));
            assert_eq!(tree.limit(b"/a/b/c/"), Some(Some(3_000_000_000)));
            assert_eq!(tree.limit(b"/a"), Some(Some(4_000_000_000)));
            tree.set("a", "1000\n");
            assert_eq!(tree.limit(b"/a/b/c"), Some(Some(1000)));
            // The root has no `memory.max`; a file at the root is read too.
            assert_eq!(tree.limit(b"/"), Some(None));
            tree.set("", "500\n");
            assert_eq!(tree.limit(b"/"), Some(Some(500)));
        }

        #[test]
        fn an_unreadable_cgroup_limit_is_unknown() {
            let tree = Tree::new("unreadable");
            // A path too long to open without allocating.
            let long = [b'/'; MAX_PATH];
            assert_eq!(tree.limit(&long), None);
            // A file that fills the buffer is not a whole value.
            tree.set("a", &"9".repeat(40));
            assert_eq!(tree.limit(b"/a"), Some(None));
            let mut buf = [0u8; 4];
            let path = tree.0.join("a/memory.max");
            let path = std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str()).to_vec();
            assert_eq!(read_file(&path, &mut buf, false), Some(&b"9999"[..]));
            assert_eq!(read_file(&path, &mut buf, true), None);
            assert_eq!(read_file(b"/nonexistent/krishiv", &mut buf, false), None);
            assert_eq!(read_file(&[b'/'; MAX_PATH + 1], &mut buf, false), None);
        }

        /// The host's ceiling: at most physical memory, and not below
        /// this process's own resident set.
        #[test]
        fn the_host_ceiling_is_within_physical_memory() {
            let mut buf = [0u8; 256];
            let physical =
                meminfo_total(read_file(b"/proc/meminfo", &mut buf, false).unwrap()).unwrap();
            let ceiling = memory_ceiling().unwrap();
            assert!(ceiling <= physical, "{ceiling} > {physical}");
            assert!(ceiling > 16 << 20, "{ceiling}");
        }
    }
}

/// The memory this process may use, in bytes: the smaller of physical
/// memory (`MemTotal` in `/proc/meminfo`) and the tightest cgroup v2
/// `memory.max` on the process's cgroup path. `None` if physical memory or
/// the cgroup path cannot be read. cgroup v1 limits are not read (the
/// pools' own default, `krishiv_memory::effective_memory_limit`, reads
/// them). Allocates nothing, so the constructor can call it: the files are
/// read into stack buffers, and paths stay under the length up to which
/// the standard library builds the C path on the stack.
#[cfg(target_os = "linux")]
pub fn memory_ceiling() -> Option<u64> {
    ceiling::memory_ceiling()
}

/// Without `/proc` there is nothing to read: unknown.
#[cfg(not(target_os = "linux"))]
pub fn memory_ceiling() -> Option<u64> {
    None
}

/// Returns the calling thread's freed memory to the OS: the pressure
/// release hook.
pub fn release() {
    // SAFETY: `mi_collect` has no preconditions; it only touches mimalloc's
    // state for the calling thread and the arenas.
    unsafe { libmimalloc_sys::mi_collect(true) };
}

/// The binary's half of the policy, called first thing in `main`: keeps
/// the constructor linked and returns [`release`], the hook to install with
/// `krishiv_memory::set_release_hook`.
pub fn install() -> fn() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    std::hint::black_box(&init::INIT);
    release
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_limit_comes_from_the_flag_or_the_conf_key() {
        let limit = |a: &[&str]| limit_in_args(a.iter().map(|s| s.as_bytes()));
        assert_eq!(
            limit(&["runner", "--data", "d", "--memory-limit", "2g"]),
            Some(2 << 30)
        );
        assert_eq!(limit(&["runner", "--memory-limit=512m"]), Some(512 << 20));
        assert_eq!(
            limit(&["krishiv", "sql", "--conf", "krishiv.memory.limit=24g"]),
            Some(24 << 30)
        );
        assert_eq!(
            limit(&["krishiv", "sql", "--conf=krishiv.memory.limit=1t"]),
            Some(1 << 40)
        );
        assert_eq!(limit(&["runner", "--data", "d", "--threads", "4"]), None);
        assert_eq!(
            limit(&["krishiv", "sql", "--conf", "krishiv.io.prefetchBytes=1m"]),
            None
        );
        assert_eq!(limit(&["krishiv", "sql", "--memory-limits"]), None);
        // Several: the smallest. Not a size, or no value: 0.
        assert_eq!(
            limit(&["--memory-limit", "24g", "--conf", "krishiv.memory.limit=2g"]),
            Some(2 << 30)
        );
        assert_eq!(limit(&["--memory-limit", "lots"]), Some(0));
        assert_eq!(limit(&["--memory-limit"]), Some(0));
        assert_eq!(limit(&["--memory-limit="]), Some(0));
        assert_eq!(limit(&["--memory-limit", "24g", "--memory-limit"]), Some(0));
    }

    #[test]
    fn sizes_parse_as_the_memory_limit_setting_does() {
        let size = |v: &str| parse_size(v.as_bytes());
        assert_eq!(size("1048576"), Some(1 << 20));
        assert_eq!(size("100b"), Some(100));
        assert_eq!(size("64k"), Some(64 << 10));
        assert_eq!(size("64KB"), Some(64 << 10));
        assert_eq!(size("200m"), Some(200 << 20));
        assert_eq!(size("512Mb"), Some(512 << 20));
        assert_eq!(size(" 2g "), Some(2 << 30));
        assert_eq!(size("24GB"), Some(24 << 30));
        assert_eq!(size("3t"), Some(3 << 40));
        assert_eq!(size("1tb"), Some(1 << 40));
        assert_eq!(size("1p"), Some(1 << 50));
        assert_eq!(size("2PB"), Some(2 << 50));
        assert_eq!(size("0"), Some(0));
        for bad in [
            "", "g", "lots", "2x", "2gbb", "2bb", "2 g", "-1g", "1.5g", "16384p",
        ] {
            assert_eq!(size(bad), None, "{bad:?}");
        }
        assert_eq!(size("18446744073709551616"), None);
    }

    #[test]
    fn a_limit_is_generous_from_4_gib_with_room_for_twice_it() {
        const GIB: u64 = 1 << 30;
        let ceiling = |c: u64| move || Some(c);
        // The memory gate's 2 GiB is never generous, whatever the machine.
        assert!(!is_generous(2 * GIB, ceiling(u64::MAX)));
        assert!(!is_generous(GENEROUS_MIN - 1, ceiling(u64::MAX)));
        // 24 GiB on a 64 GiB machine: 2 x 24 + 1 = 49 GiB fits.
        assert!(is_generous(24 * GIB, ceiling(64 * GIB)));
        assert!(is_generous(4 * GIB, ceiling(9 * GIB)));
        assert!(!is_generous(4 * GIB, ceiling(9 * GIB - 1)));
        // 12 GiB on a 16 GiB machine or in a 16 GiB container: no room.
        assert!(!is_generous(12 * GIB, ceiling(16 * GIB)));
        // Unknown room, or a limit too large to double: not generous.
        assert!(!is_generous(24 * GIB, || None));
        assert!(!is_generous(u64::MAX, ceiling(u64::MAX)));
        assert!(!is_generous(u64::MAX / 2, ceiling(u64::MAX)));
        // Below the minimum the ceiling is not read.
        assert!(!is_generous(GIB, || unreachable!("ceiling read")));
    }

    /// The option numbers belong to the linked mimalloc (version 3); this
    /// test process has no limit, so mimalloc's defaults are in force.
    #[test]
    fn the_option_numbers_are_the_linked_mimalloc_s() {
        assert_eq!(read(), (1000, 1, 0));
        // SAFETY: plain getter.
        let version = unsafe { libmimalloc_sys::mi_version() };
        // `major * 10000 + minor * 100 + patch` (3.3.2 is 30302).
        assert!((30_000..40_000).contains(&version), "mimalloc {version}");
        assert_eq!(install() as usize, release as fn() as usize);
        release();
    }

    /// `(purge_delay, allow_thp, minimal_purge_size)` in force.
    fn read() -> (c_long, c_long, c_long) {
        // SAFETY: plain getters.
        unsafe {
            (
                libmimalloc_sys::mi_option_get(15),
                libmimalloc_sys::mi_option_get(43),
                libmimalloc_sys::mi_option_get(44),
            )
        }
    }

    /// In a child of this test binary: the options the constructor set.
    #[test]
    fn child_reports_options() {
        if std::env::var_os("KRISHIV_ALLOC_CHILD").is_some() {
            // Whether huge pages are off for the process: mimalloc calls
            // `PR_SET_THP_DISABLE` at its own initialisation, so this shows
            // the constructor ran before it (not only that the option is set).
            println!("OPTIONS {:?} THP {}", read(), thp_enabled());
        }
    }

    /// `THP_enabled` of this process (`?` if the kernel does not report it).
    fn thp_enabled() -> String {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        status
            .lines()
            .find_map(|l| l.strip_prefix("THP_enabled:"))
            .map_or("?", str::trim)
            .to_owned()
    }

    /// Runs `child_reports_options` in a child process with `arg` and `env`.
    fn child(arg: Option<&str>, env: &[(&str, &str)]) -> String {
        let mut c = std::process::Command::new(std::env::current_exe().unwrap());
        c.args(["tests::child_reports_options", "--exact", "--nocapture"])
            .env("KRISHIV_ALLOC_CHILD", "1")
            .env_remove("KRISHIV_MEMORY_LIMIT");
        for (k, ..) in LIMITED_OPTIONS {
            c.env_remove(k.to_str().unwrap());
        }
        // A second filter: libtest runs tests matching either; the
        // constructor sees it in `argv`.
        c.args(arg);
        c.envs(env.iter().copied());
        let out = c.output().unwrap();
        assert!(out.status.success(), "{out:?}");
        let text = String::from_utf8(out.stdout).unwrap();
        text.lines()
            .find_map(|l| l.strip_prefix("OPTIONS "))
            .unwrap()
            .to_owned()
    }

    /// The constructor applies the options exactly when the process starts
    /// with a limit (argument or environment), and the user's variables win.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn a_limit_at_start_applies_the_options() {
        // Without a limit the child has this (unlimited) process's huge-page
        // state: 1, or 0 on a kernel without transparent huge pages. With a
        // limit it is 0 either way.
        let unlimited = format!("(1000, 1, 0) THP {}", thp_enabled());
        assert_eq!(child(None, &[]), unlimited);
        assert_eq!(
            child(Some("krishiv.memory.limit=1g"), &[]),
            "(10, 0, 64) THP 0"
        );
        assert_eq!(
            child(None, &[("KRISHIV_MEMORY_LIMIT", "1g")]),
            "(10, 0, 64) THP 0"
        );
        assert_eq!(
            child(
                None,
                &[
                    ("KRISHIV_MEMORY_LIMIT", "1g"),
                    ("MIMALLOC_PURGE_DELAY", "50")
                ]
            ),
            "(50, 0, 64) THP 0"
        );
        assert_eq!(child(None, &[("KRISHIV_MEMORY_LIMIT", "")]), unlimited);
        // The smaller of the variable and the arguments decides.
        assert_eq!(
            child(
                Some("krishiv.memory.limit=1g"),
                &[("KRISHIV_MEMORY_LIMIT", "1p")]
            ),
            "(10, 0, 64) THP 0"
        );
    }

    /// A generous limit leaves mimalloc's defaults, huge pages included;
    /// one too large for the machine applies the options, after reading the
    /// machine's memory before mimalloc starts (the huge-page state shows
    /// that reading it allocated nothing).
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn a_generous_limit_leaves_the_defaults() {
        let unlimited = format!("(1000, 1, 0) THP {}", thp_enabled());
        let limited = "(10, 0, 64) THP 0".to_owned();
        // 4 GiB needs 9 GiB of room: developer machines and CI runners
        // have it, so this checks the defaults path there.
        let room = memory_ceiling().unwrap();
        assert!(room >= 9 << 30, "too little memory to check: {room}");
        assert!(is_generous(GENEROUS_MIN, memory_ceiling));
        assert_eq!(child(Some("krishiv.memory.limit=4g"), &[]), unlimited);
        assert_eq!(child(None, &[("KRISHIV_MEMORY_LIMIT", "4096m")]), unlimited);
        // More than the machine has (2 x 1 PiB): the options apply.
        assert_eq!(child(None, &[("KRISHIV_MEMORY_LIMIT", "1p")]), limited);
        assert_eq!(child(Some("krishiv.memory.limit=1p"), &[]), limited);
        // A limit that is not a size counts as 0.
        assert_eq!(child(None, &[("KRISHIV_MEMORY_LIMIT", "lots")]), limited);
    }
}
