// SPDX-License-Identifier: Apache-2.0
//! The interpreters an administrator lists (`krishiv.python.worker.executables`), recorded once at startup
//! (design `P3-T014-connect-udfs-sandbox.md` §3.5).
//!
//! Each is run once, **unsandboxed but with an empty environment**, to record its version, prefixes,
//! `sys.path` and packages. The record gives the sandbox's read paths for that interpreter and the choice of
//! interpreter per request. One without `pyspark` or `pyarrow` is logged and not used.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

/// What the inspection prints: one JSON object.
const INSPECT: &str = r#"
import json, sys
def version(name):
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "?"))
    except Exception:
        return None
print(json.dumps({
    "version": list(sys.version_info[:3]),
    "prefixes": sorted({sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}),
    "path": [p for p in sys.path if p],
    "pyspark": version("pyspark"),
    "pyarrow": version("pyarrow"),
    "pandas": version("pandas"),
}))
"#;

/// How long the inspection of one interpreter may take: a hanging interpreter must not hold up server startup.
const INSPECT_TIMEOUT: Duration = Duration::from_secs(60);

/// The oldest Python a worker may run: the 4.3 client's floor (design §2, "Python floor").
const MIN_VERSION: (u32, u32) = (3, 11);

/// One listed interpreter and what it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interpreter {
    /// Its absolute path, as the worker executes it.
    pub path: PathBuf,
    /// `major.minor.micro`.
    pub version: (u32, u32, u32),
    /// The paths a worker of this interpreter must read: its prefixes, its `sys.path` and its own
    /// directory.
    pub read_paths: Vec<PathBuf>,
    /// The installed `pyspark` (or `pyspark-client`) version.
    pub pyspark: Option<String>,
    /// The installed `pyarrow` version.
    pub pyarrow: Option<String>,
    /// The installed `pandas` version (needed by pandas UDFs only).
    pub pandas: Option<String>,
}

impl Interpreter {
    /// Runs `executable` (an absolute path, or a name looked up on `PATH`) once and records it.
    ///
    /// # Errors
    /// The reason, in words, when it cannot be run, prints no record, or is older than Python 3.11.
    pub fn inspect(executable: &str) -> Result<Self, String> {
        let path = resolve(executable)
            .ok_or_else(|| format!("{executable}: not found (no such file, or not on PATH)"))?;
        let out = run_with_timeout(
            Command::new(&path)
                .args(["-I", "-c", INSPECT])
                .env_clear()
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped()),
            INSPECT_TIMEOUT,
        )
        .map_err(|e| format!("{}: cannot run it: {e}", path.display()))?;
        if !out.status.success() {
            return Err(format!(
                "{}: the inspection failed ({}): {}",
                path.display(),
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        let record: Value = serde_json::from_slice(&out.stdout)
            .map_err(|e| format!("{}: unreadable inspection output: {e}", path.display()))?;
        Self::from_record(path, &record)
    }

    fn from_record(path: PathBuf, record: &Value) -> Result<Self, String> {
        let number = |i: usize| {
            record
                .get("version")
                .and_then(|v| v.get(i))
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
        };
        let (Some(major), Some(minor), Some(micro)) = (number(0), number(1), number(2)) else {
            return Err(format!("{}: no version in the record", path.display()));
        };
        if (major, minor) < MIN_VERSION {
            return Err(format!(
                "{}: Python {major}.{minor} is older than {}.{}, the oldest a worker may run",
                path.display(),
                MIN_VERSION.0,
                MIN_VERSION.1
            ));
        }
        let strings = |key: &str| -> Vec<PathBuf> {
            record
                .get(key)
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(PathBuf::from)
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut read_paths = strings("prefixes");
        read_paths.extend(strings("path"));
        if let Some(dir) = path.parent() {
            read_paths.push(dir.to_path_buf());
        }
        // A venv's interpreter is a link into the base installation: the target's directory too.
        if let Ok(real) = std::fs::canonicalize(&path)
            && let Some(dir) = real.parent()
        {
            read_paths.push(dir.to_path_buf());
        }
        read_paths.sort();
        read_paths.dedup();
        let package = |key: &str| record.get(key).and_then(Value::as_str).map(str::to_owned);
        Ok(Interpreter {
            path,
            version: (major, minor, micro),
            read_paths,
            pyspark: package("pyspark"),
            pyarrow: package("pyarrow"),
            pandas: package("pandas"),
        })
    }

    /// `major.minor`, as a client's `python_ver` names it.
    pub fn minor_version(&self) -> String {
        format!("{}.{}", self.version.0, self.version.1)
    }

    /// Why this interpreter cannot run workers, if it cannot.
    pub fn unusable_reason(&self) -> Option<String> {
        let missing: Vec<&str> = [("pyspark", &self.pyspark), ("pyarrow", &self.pyarrow)]
            .into_iter()
            .filter(|(_, v)| v.is_none())
            .map(|(name, _)| name)
            .collect();
        (!missing.is_empty()).then(|| {
            format!(
                "{}: missing {} (install pyspark-client of the clients' version and pyarrow)",
                self.path.display(),
                missing.join(" and ")
            )
        })
    }
}

/// The interpreter for a request whose client runs `python_ver` (`major.minor`): the first listed one of that
/// version, else the first one (whose worker then raises Spark's version mismatch, design §3.1). `None` only
/// for an empty list.
pub fn choose<'a>(
    interpreters: &'a [Interpreter],
    python_ver: &str,
) -> Option<(usize, &'a Interpreter)> {
    interpreters
        .iter()
        .enumerate()
        .find(|(_, i)| i.minor_version() == python_ver)
        .or_else(|| interpreters.first().map(|i| (0, i)))
}

/// Runs `command` (stdout and stderr piped) and collects its output, killing it after `timeout`. The
/// inspection prints a few kilobytes, well inside a pipe's buffer, so reading after the exit cannot block it.
fn run_with_timeout(command: &mut Command, timeout: Duration) -> std::io::Result<Output> {
    let mut child = command.spawn()?;
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("no answer within {} s", timeout.as_secs()),
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_end(&mut stdout)?;
    }
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_end(&mut stderr)?;
    }
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

/// `name` as an absolute path: itself when it has a directory part, else the first match on `PATH`.
fn resolve(name: &str) -> Option<PathBuf> {
    let candidate = Path::new(name);
    if candidate.components().count() > 1 {
        return candidate
            .is_absolute()
            .then(|| candidate.to_path_buf())
            .filter(|p| p.is_file());
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(version: [u32; 3], pyspark: Option<&str>) -> Value {
        json!({
            "version": version,
            "prefixes": ["/opt/py", "/usr"],
            "path": ["/opt/py/lib/python3.12", "/opt/py/lib/python3.12/site-packages"],
            "pyspark": pyspark,
            "pyarrow": "26.0.1",
            "pandas": null,
        })
    }

    #[test]
    fn a_record_gives_read_paths_and_packages() {
        let i = Interpreter::from_record(
            "/opt/py/bin/python3".into(),
            &record([3, 12, 1], Some("4.2.1")),
        )
        .expect("record");
        assert_eq!(i.minor_version(), "3.12");
        assert!(
            i.read_paths
                .contains(&PathBuf::from("/opt/py/lib/python3.12/site-packages"))
        );
        assert!(i.read_paths.contains(&PathBuf::from("/opt/py/bin")));
        assert_eq!(i.unusable_reason(), None);
        let without = Interpreter::from_record("/x/python3".into(), &record([3, 12, 1], None))
            .expect("record");
        assert!(
            without
                .unusable_reason()
                .unwrap_or_default()
                .contains("missing pyspark")
        );
    }

    #[test]
    fn old_pythons_and_bad_records_are_refused() {
        let old = Interpreter::from_record("/x".into(), &record([3, 10, 9], Some("4.2.1")));
        assert!(old.unwrap_err().contains("older than 3.11"));
        assert!(Interpreter::from_record("/x".into(), &json!({})).is_err());
    }

    #[test]
    fn the_clients_minor_version_wins() {
        let a =
            Interpreter::from_record("/a".into(), &record([3, 11, 9], Some("4.2.1"))).expect("a");
        let b =
            Interpreter::from_record("/b".into(), &record([3, 14, 0], Some("4.2.1"))).expect("b");
        let list = [a, b];
        assert_eq!(choose(&list, "3.14").map(|(i, _)| i), Some(1));
        assert_eq!(choose(&list, "3.11").map(|(i, _)| i), Some(0));
        assert_eq!(
            choose(&list, "3.12").map(|(i, _)| i),
            Some(0),
            "else the first"
        );
        assert!(choose(&[], "3.12").is_none());
    }

    #[test]
    fn a_hanging_interpreter_times_out() {
        let err = run_with_timeout(
            Command::new("/bin/sh").args(["-c", "exec sleep 30"]),
            Duration::from_millis(200),
        )
        .expect_err("timed out");
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
        let out = run_with_timeout(
            Command::new("/bin/sh")
                .args(["-c", "echo hi"])
                .stdout(Stdio::piped()),
            Duration::from_secs(10),
        )
        .expect("runs");
        assert_eq!(out.stdout, b"hi\n");
    }

    #[test]
    fn names_resolve_on_path_and_relative_paths_do_not() {
        assert!(resolve("no-such-interpreter-krishiv").is_none());
        assert!(resolve("bin/sh").is_none());
        assert_eq!(resolve("/bin/sh").as_deref(), Some(Path::new("/bin/sh")));
    }
}
