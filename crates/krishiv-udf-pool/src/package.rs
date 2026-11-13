// SPDX-License-Identifier: Apache-2.0
//! The worker package the server ships inside its own binary, and the bootstrap that starts it.
//!
//! The files are `python/krishiv/_udf/*.py`, the same ones the wheel ships (`worker/krishiv/_udf/` holds
//! symbolic links to them, so there is one copy). They are written under a stub `krishiv/__init__.py` that
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

/// Writes the worker package under `root` (which must exist and be empty or absent) and makes it read-only.
/// Returns `root`, the directory to put first on the worker's `sys.path`.
pub(crate) fn extract(root: &Path) -> io::Result<PathBuf> {
    fs::create_dir_all(root)?;
    for (rel, contents) in FILES {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, contents)?;
        set_read_only(&path)?;
    }
    for dir in ["krishiv/_udf", "krishiv"] {
        set_read_only(&root.join(dir))?;
    }
    Ok(root.to_path_buf())
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

/// The `close_range` system call number: the same on every Linux architecture with the generic table
/// (x86-64 and aarch64 both use 436).
const CLOSE_RANGE: u32 = 436;

/// The `-c` program the interpreter runs (with `-I`). Its first statements close every descriptor above 2
/// (the sandbox crate's invariant: before `sys.path` changes and before any client code), then it puts
/// `package` first on `sys.path`, applies `limits` when the sandbox is off (the trampoline applies them
/// otherwise), and runs the worker.
///
/// `close_range` goes through `ctypes`: some CPython builds (python-build-standalone, which `uv` installs)
/// implement `os.closerange` as a loop over every number, which takes minutes up to `2**31`. Where the call
/// is missing (Linux before 5.9), `os.closerange` runs up to `fd_limit`, the server's descriptor limit, which
/// bounds every descriptor the server could have opened.
pub(crate) fn bootstrap(package: &Path, fd_limit: u64, limits: Option<&BootstrapLimits>) -> String {
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
    if let Some(limits) = limits {
        lines.push("import resource".to_owned());
        if let Some(n) = limits.address_space {
            lines.push(format!(
                "resource.setrlimit(resource.RLIMIT_AS, ({n}, {n}))"
            ));
        }
        if let Some(n) = limits.file_size {
            lines.push(format!(
                "resource.setrlimit(resource.RLIMIT_FSIZE, ({n}, {n}))"
            ));
        }
        let n = limits.open_files;
        lines.push(format!(
            "resource.setrlimit(resource.RLIMIT_NOFILE, ({n}, {n}))"
        ));
    }
    lines.extend([
        "import sys".to_owned(),
        format!("sys.path.insert(0, {})", python_str(package)),
        "from krishiv._udf.pool_worker import main".to_owned(),
        "main()".to_owned(),
    ]);
    lines.join("\n")
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
        let root = extract(&dir.path().join("pkg")).unwrap_or_else(|e| panic!("{e}"));
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
                .contains("def main()")
        );
    }

    #[test]
    fn bootstrap_closes_descriptors_first() {
        let code = bootstrap(Path::new("/srv/p k\u{e9}"), 1024, None);
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
        );
        assert!(limited.contains("resource.RLIMIT_AS, (1073741824, 1073741824)"));
        assert!(limited.contains("resource.RLIMIT_NOFILE, (256, 256)"));
        assert!(!limited.contains("RLIMIT_FSIZE"));
    }
}
