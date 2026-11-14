// SPDX-License-Identifier: Apache-2.0
//! The worker package the server ships inside its own binary, and the bootstrap that starts it.
//!
//! The files are `python/krishiv/_udf/*.py`, the same ones the wheel ships (`worker/krishiv/_udf` is a
//! symbolic link to that directory, so there is one copy). They are written under a stub `krishiv/__init__.py` that
//! marks the pool worker (`_host.py`), so worker interpreters need `pyspark-client`, `pyarrow` and `pandas`
//! but no `krishiv` wheel (design `P3-T014-connect-udfs-sandbox.md` §3.1).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// `(path below the package root, contents)` of every file of the worker package.
const FILES: &[(&str, &str)] = &[
    ("krishiv/__init__.py", STUB_INIT),
    (
        "krishiv/_udf/__init__.py",
        include_str!("../worker/krishiv/_udf/__init__.py"),
    ),
    (
        "krishiv/_udf/_host.py",
        include_str!("../worker/krishiv/_udf/_host.py"),
    ),
    (
        "krishiv/_udf/conversion.py",
        include_str!("../worker/krishiv/_udf/conversion.py"),
    ),
    (
        "krishiv/_udf/loose_version.py",
        include_str!("../worker/krishiv/_udf/loose_version.py"),
    ),
    (
        "krishiv/_udf/pandas_types.py",
        include_str!("../worker/krishiv/_udf/pandas_types.py"),
    ),
    (
        "krishiv/_udf/pandas_utils.py",
        include_str!("../worker/krishiv/_udf/pandas_utils.py"),
    ),
    (
        "krishiv/_udf/pool_worker.py",
        include_str!("../worker/krishiv/_udf/pool_worker.py"),
    ),
    (
        "krishiv/_udf/types.py",
        include_str!("../worker/krishiv/_udf/types.py"),
    ),
    (
        "krishiv/_udf/worker.py",
        include_str!("../worker/krishiv/_udf/worker.py"),
    ),
];

/// The stub package root: it marks the pool worker and imports nothing (`krishiv/_udf/_host.py`).
const STUB_INIT: &str = "# SPDX-License-Identifier: Apache-2.0\n\
\"\"\"Krishiv's Python UDF worker package (written by the server; not the krishiv wheel).\"\"\"\n\
\n\
_UDF_WORKER = True\n";

/// The directories of the package made read-only, parents last.
const DIRS: [&str; 2] = ["krishiv/_udf", "krishiv"];

/// The extracted worker package. Dropping it makes its directories writable again: an unprivileged owner
/// cannot remove the entries of a read-only directory, so the pool's temporary directory would otherwise stay
/// behind (the files themselves stay read-only; unlinking needs only the directory's write bit).
#[derive(Debug)]
pub(crate) struct Package {
    root: PathBuf,
}

impl Package {
    /// The directory to put first on the worker's `sys.path`.
    pub(crate) fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for Package {
    fn drop(&mut self) {
        for dir in DIRS {
            let _ = set_writable(&self.root.join(dir));
        }
    }
}

/// Writes the worker package under `root` (which must exist and be empty or absent) and makes it read-only.
pub(crate) fn extract(root: &Path) -> io::Result<Package> {
    fs::create_dir_all(root)?;
    // From here on a failure still restores the directories (`Drop`).
    let package = Package {
        root: root.to_path_buf(),
    };
    for (rel, contents) in FILES {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, contents)?;
        set_read_only(&path)?;
    }
    for dir in DIRS {
        set_read_only(&root.join(dir))?;
    }
    Ok(package)
}

#[cfg(unix)]
fn set_read_only(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = if path.is_dir() { 0o555 } else { 0o444 };
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn set_read_only(path: &Path) -> io::Result<()> {
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_readonly(true);
    fs::set_permissions(path, perms)
}

#[cfg(unix)]
fn set_writable(dir: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(dir, fs::Permissions::from_mode(0o755))
}

#[cfg(not(unix))]
fn set_writable(dir: &Path) -> io::Result<()> {
    let mut perms = fs::metadata(dir)?.permissions();
    #[expect(
        clippy::permissions_set_readonly_false,
        reason = "the package's own directory, made writable for its removal"
    )]
    perms.set_readonly(false);
    fs::set_permissions(dir, perms)
}

/// The `close_range` system call number: the same on every Linux architecture with the generic table
/// (x86-64 and aarch64 both use 436).
const CLOSE_RANGE: u32 = 436;

/// The `-c` program the interpreter runs (with `-I`). Its first statements close every descriptor above 2
/// (the sandbox crate's invariant: before `sys.path` changes and before any client code), then it puts
/// `package` first on `sys.path`, applies `limits` when the sandbox is off (the trampoline applies them
/// otherwise), and runs the worker.
///
/// With `deny_exec` (the sandbox is on and `allow_fork` is off), the next statements stack a second seccomp
/// filter that makes `execve` and `execveat` `EPERM`, before `sys.path` changes and before any client code (the
/// sandbox crate's invariant). The parent-death signal the trampoline arms is per thread and new threads start
/// without it, so an `execve` from a second thread would otherwise give an image that outlives the server; the
/// trampoline's own filter cannot refuse that call, since its own `exec` uses it. A failure exits the worker
/// before it greets the server (fail closed).
///
/// `close_range` goes through `ctypes`: some CPython builds (python-build-standalone, which `uv` installs)
/// implement `os.closerange` as a loop over every number, which takes minutes up to `2**31`. Where the call
/// is missing (Linux before 5.9), `os.closerange` runs up to `fd_limit`, the server's descriptor limit, which
/// bounds every descriptor the server could have opened.
pub(crate) fn bootstrap(
    package: &Path,
    fd_limit: u64,
    limits: Option<&BootstrapLimits>,
    deny_exec: bool,
) -> String {
    let mut lines = vec![
        "import os".to_owned(),
        "try:".to_owned(),
        "    import ctypes".to_owned(),
        format!(
            "    _r = ctypes.CDLL(None, use_errno=True).syscall({CLOSE_RANGE}, 3, 0xFFFFFFFF, 0)"
        ),
        "except Exception:".to_owned(),
        "    _r = -1".to_owned(),
        "if _r != 0:".to_owned(),
        format!("    os.closerange(3, {fd_limit})"),
    ];
    if deny_exec {
        lines.extend([
            "import ctypes".to_owned(),
            format!("_f = bytes.fromhex('{}')", hex(&exec_deny_filter())),
            "_b = ctypes.create_string_buffer(_f, len(_f))".to_owned(),
            "class _Prog(ctypes.Structure):".to_owned(),
            "    _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.c_void_p)]".to_owned(),
            "_p = _Prog(len(_f) // 8, ctypes.cast(_b, ctypes.c_void_p))".to_owned(),
            "_c = ctypes.CDLL(None, use_errno=True)".to_owned(),
            // prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog): no_new_privs is already set by the trampoline.
            "if _c.prctl(22, 2, ctypes.byref(_p), 0, 0) != 0:".to_owned(),
            "    raise SystemExit('krishiv: the worker could not deny exec: errno ' + str(ctypes.get_errno()))"
                .to_owned(),
            "del _f, _b, _p, _c, _Prog".to_owned(),
        ]);
    }
    if let Some(limits) = limits {
        // An unprivileged process may only lower its hard limit, so never ask above a finite inherited one
        // (as the sandbox's own `set_limits` does for sandboxed workers).
        lines.extend([
            "import resource".to_owned(),
            "def _limit(r, n):".to_owned(),
            "    h = resource.getrlimit(r)[1]".to_owned(),
            "    if h != resource.RLIM_INFINITY and h < n:".to_owned(),
            "        n = h".to_owned(),
            "    resource.setrlimit(r, (n, n))".to_owned(),
        ]);
        if let Some(n) = limits.address_space {
            lines.push(format!("_limit(resource.RLIMIT_AS, {n})"));
        }
        if let Some(n) = limits.file_size {
            lines.push(format!("_limit(resource.RLIMIT_FSIZE, {n})"));
        }
        lines.push(format!(
            "_limit(resource.RLIMIT_NOFILE, {})",
            limits.open_files
        ));
        lines.push("del _limit".to_owned());
    }
    lines.extend([
        "import sys".to_owned(),
        format!("sys.path.insert(0, {})", python_str(package)),
        "from krishiv._udf.pool_worker import main".to_owned(),
        "main()".to_owned(),
    ]);
    lines.join("\n")
}

/// `AUDIT_ARCH_*` of this target and its `execve`/`execveat` numbers.
#[cfg(target_arch = "x86_64")]
const EXEC_ARCH: (u32, u32, u32) = (0xC000_003E, 59, 322);
#[cfg(target_arch = "aarch64")]
const EXEC_ARCH: (u32, u32, u32) = (0xC000_00B7, 221, 281);
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const EXEC_ARCH: (u32, u32, u32) = (0, u32::MAX, u32::MAX);

/// A classic BPF seccomp program: another architecture is killed (the trampoline's filter already refuses
/// x32 numbers), `execve` and `execveat` are `EPERM`, everything else passes on to the earlier filter.
/// Each instruction is `struct sock_filter` (`u16 code, u8 jt, u8 jf, u32 k`), native endian.
fn exec_deny_filter() -> Vec<u8> {
    const LD_W_ABS: u16 = 0x20;
    const JEQ_K: u16 = 0x15;
    const RET_K: u16 = 0x06;
    const RET_KILL_PROCESS: u32 = 0x8000_0000;
    const RET_ALLOW: u32 = 0x7FFF_0000;
    const RET_EPERM: u32 = 0x0005_0000 | 1;
    let (arch, execve, execveat) = EXEC_ARCH;
    let program: [(u16, u8, u8, u32); 8] = [
        (LD_W_ABS, 0, 0, 4), // seccomp_data.arch
        (JEQ_K, 1, 0, arch),
        (RET_K, 0, 0, RET_KILL_PROCESS),
        (LD_W_ABS, 0, 0, 0), // seccomp_data.nr
        (JEQ_K, 2, 0, execve),
        (JEQ_K, 1, 0, execveat),
        (RET_K, 0, 0, RET_ALLOW),
        (RET_K, 0, 0, RET_EPERM),
    ];
    let mut out = Vec::with_capacity(program.len() * 8);
    for (code, jt, jf, k) in program {
        out.extend_from_slice(&code.to_ne_bytes());
        out.push(jt);
        out.push(jf);
        out.extend_from_slice(&k.to_ne_bytes());
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// The limits the bootstrap applies itself when the sandbox is off.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BootstrapLimits {
    pub(crate) address_space: Option<u64>,
    pub(crate) file_size: Option<u64>,
    pub(crate) open_files: u64,
}

/// `path` as a Python string literal (non-UTF-8 bytes become `surrogateescape` escapes, as `os.fsdecode` does).
fn python_str(path: &Path) -> String {
    use std::fmt::Write as _;
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    };
    #[cfg(not(unix))]
    let bytes = path.to_string_lossy().into_owned().into_bytes();
    let mut out = String::from("os.fsdecode(b'");
    for b in bytes {
        if b.is_ascii_alphanumeric() || b"/._-+".contains(&b) {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "\\x{b:02x}");
        }
    }
    out.push_str("')");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_file_is_embedded_and_marks_the_worker() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let package = extract(&dir.path().join("pkg")).unwrap_or_else(|e| panic!("{e}"));
        let root = package.path();
        let init =
            fs::read_to_string(root.join("krishiv/__init__.py")).unwrap_or_else(|e| panic!("{e}"));
        assert!(init.contains("_UDF_WORKER = True"));
        for (rel, _) in FILES {
            let meta = fs::metadata(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
            assert!(meta.permissions().readonly(), "{rel} is writable");
        }
        assert!(
            fs::read_to_string(root.join("krishiv/_udf/pool_worker.py"))
                .unwrap_or_default()
                .contains("def main(")
        );
    }

    #[test]
    fn bootstrap_closes_descriptors_first() {
        let code = bootstrap(Path::new("/srv/p k\u{e9}"), 1024, None, false);
        let first_path = code.find("sys.path").unwrap_or(usize::MAX);
        assert!(code.starts_with("import os\ntry:\n    import ctypes"));
        assert!(code.find("os.closerange(3, 1024)").unwrap_or(usize::MAX) < first_path);
        assert!(code.contains("os.fsdecode(b'/srv/p\\x20k\\xc3\\xa9')"));
        assert!(!code.contains("resource"));
        let limited = bootstrap(
            Path::new("/p"),
            8,
            Some(&BootstrapLimits {
                address_space: Some(1 << 30),
                file_size: None,
                open_files: 256,
            }),
            false,
        );
        assert!(limited.contains("_limit(resource.RLIMIT_AS, 1073741824)"));
        assert!(limited.contains("_limit(resource.RLIMIT_NOFILE, 256)"));
        assert!(!limited.contains("RLIMIT_FSIZE"));
        assert!(!code.contains("prctl"));
    }

    /// With the sandbox off, the bootstrap never asks above an inherited finite hard limit (an unprivileged
    /// process cannot raise it): run its limit statements under hard limits below the requested ones.
    #[cfg(target_os = "linux")]
    #[test]
    fn sandbox_off_limits_are_clamped_to_the_inherited_hard_limits() {
        let python =
            std::env::var_os("KRISHIV_TEST_UDF_PYTHON").unwrap_or_else(|| "python3".into());
        let code = bootstrap(
            Path::new("/p"),
            8,
            Some(&BootstrapLimits {
                address_space: Some(1 << 40),
                file_size: Some(1 << 40),
                open_files: 256,
            }),
            false,
        );
        let limits_only = code
            .split("\nimport sys\nsys.path.insert")
            .next()
            .unwrap_or_default();
        let program = format!(
            "import resource\n\
             _kinds = (resource.RLIMIT_AS, resource.RLIMIT_FSIZE, resource.RLIMIT_NOFILE)\n\
             _hard = [resource.getrlimit(r)[1] for r in _kinds]\n\
             {limits_only}\n\
             assert [resource.getrlimit(r) for r in _kinds] == [(h, h) for h in _hard], \
             [resource.getrlimit(r) for r in _kinds]\n"
        );
        let out = std::process::Command::new("/bin/sh")
            .args([
                "-c",
                "ulimit -n 128 && ulimit -v 8388608 && ulimit -f 2097152 && exec \"$0\" -I -c \"$1\"",
            ])
            .arg(&python)
            .arg(&program)
            .output();
        let Ok(out) = out else { return }; // no `sh`: nothing to run the check with
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn bootstrap_denies_exec_before_the_path_changes() {
        let code = bootstrap(Path::new("/p"), 8, None, true);
        let prctl = code.find("_c.prctl(22, 2").unwrap_or(usize::MAX);
        assert!(code.find("os.closerange(3, 8)").unwrap_or(usize::MAX) < prctl);
        assert!(prctl < code.find("sys.path").unwrap_or(0));
        let filter = exec_deny_filter();
        assert_eq!(filter.len(), 64);
        assert!(code.contains(&hex(&filter)));
    }
}
