// SPDX-License-Identifier: Apache-2.0
//! The interpreters an administrator lists (`krishiv.python.worker.executables`), recorded once at startup
//! (design `P3-T014-connect-udfs-sandbox.md` §3.5).
//!
//! Each is run once, **unsandboxed but with an empty environment**, to record its version, prefixes,
//! `sys.path` and packages. The record gives the sandbox's read paths for that interpreter and the choice of
//! interpreter per request. One without `pyspark` or `pyarrow` is logged and not used.

use std::io::{self, Read};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use rustix::process::{Pid, Signal, WaitId, WaitIdOptions};
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

/// The most the inspection keeps of each of standard output and standard error: its record is a few
/// kilobytes, so more is refused rather than buffered.
const INSPECT_OUTPUT_CAP: usize = 1 << 20;

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
            INSPECT_OUTPUT_CAP,
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

/// Runs `command` in a process group of its own and collects its standard output and error (those piped),
/// read while it runs, at most `cap` bytes each. `timeout` covers the run and the reads. On the timeout, or on
/// output over the cap, the process group is killed and the run fails. When the process exits, the rest of
/// its group is killed too: a grandchild that kept a pipe open (a `.pth` hook starting a daemon) must not hold
/// up the reads (review 2026-11-13 n9). A grandchild that left the group and holds a pipe only delays the
/// failure to the timeout; its reader thread ends when the pipe closes.
fn run_with_timeout(command: &mut Command, timeout: Duration, cap: usize) -> io::Result<Output> {
    let deadline = Instant::now() + timeout;
    let mut child = command.process_group(0).spawn()?;
    let pid = Pid::from_child(&child);
    // The process is not reaped until `child.wait`, so until then its group id names no other group.
    let kill_group = || {
        let _ = rustix::process::kill_process_group(pid, Signal::KILL);
    };
    let fail = |child: &mut std::process::Child, error: io::Error| {
        kill_group();
        let _ = child.kill();
        let _ = child.wait();
        Err(error)
    };
    let (sender, outputs) = mpsc::channel::<(bool, io::Result<Vec<u8>>)>();
    let pipes: [(bool, Option<Box<dyn Read + Send>>); 2] = [
        (
            true,
            child
                .stdout
                .take()
                .map(|p| Box::new(p) as Box<dyn Read + Send>),
        ),
        (
            false,
            child
                .stderr
                .take()
                .map(|p| Box::new(p) as Box<dyn Read + Send>),
        ),
    ];
    let mut readers = 0_usize;
    for (is_stdout, pipe) in pipes {
        let Some(pipe) = pipe else { continue };
        if let Err(e) = read_capped(pipe, cap, is_stdout, sender.clone()) {
            return fail(&mut child, e);
        }
        readers += 1;
    }
    drop(sender);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut status = None;
    loop {
        let wait = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(10));
        match outputs.recv_timeout(wait) {
            Ok((is_stdout, read)) => {
                readers = readers.saturating_sub(1);
                let bytes = match read {
                    Ok(bytes) if bytes.len() > cap => {
                        let which = if is_stdout { "output" } else { "error output" };
                        return fail(
                            &mut child,
                            io::Error::other(format!("more than {cap} bytes of {which}")),
                        );
                    }
                    Ok(bytes) => bytes,
                    Err(e) => return fail(&mut child, e),
                };
                if is_stdout {
                    stdout = bytes;
                } else {
                    stderr = bytes;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            // Every reader is done: only the exit is awaited.
            Err(mpsc::RecvTimeoutError::Disconnected) => std::thread::sleep(wait),
        }
        if status.is_none() && exited(pid) {
            // Its own exit: whatever it left running in its group goes too, then it is reaped.
            kill_group();
            status = Some(child.wait()?);
        }
        if let (Some(status), 0) = (status, readers) {
            return Ok(Output {
                status,
                stdout,
                stderr,
            });
        }
        if Instant::now() >= deadline {
            let error = io::Error::new(
                io::ErrorKind::TimedOut,
                format!("no answer within {} s", timeout.as_secs()),
            );
            if status.is_some() {
                // Reaped: its group id may name another group now, so only the readers are left behind.
                return Err(error);
            }
            return fail(&mut child, error);
        }
    }
}

/// Whether the child `pid` has exited, without reaping it.
fn exited(pid: Pid) -> bool {
    matches!(
        rustix::process::waitid(
            WaitId::Pid(pid),
            WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
        ),
        Ok(Some(_))
    )
}

/// Reads `pipe` to its end, or to `cap + 1` bytes, on a thread of its own, and sends what it read.
fn read_capped(
    pipe: Box<dyn Read + Send>,
    cap: usize,
    is_stdout: bool,
    sender: mpsc::Sender<(bool, io::Result<Vec<u8>>)>,
) -> io::Result<()> {
    std::thread::Builder::new()
        .name("krishiv-udf-inspect".to_owned())
        .spawn(move || {
            let mut bytes = Vec::new();
            let limit = u64::try_from(cap).unwrap_or(u64::MAX).saturating_add(1);
            let read = pipe.take(limit).read_to_end(&mut bytes).map(|_| bytes);
            let _ = sender.send((is_stdout, read));
        })
        .map(drop)
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

    fn sh(script: &str) -> Command {
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", script])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }

    #[test]
    fn a_hanging_interpreter_times_out() {
        let err = run_with_timeout(&mut sh("exec sleep 30"), Duration::from_millis(200), 1024)
            .expect_err("timed out");
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
        let out = run_with_timeout(
            Command::new("/bin/sh")
                .args(["-c", "echo hi; echo err >&2"])
                .stdout(Stdio::piped()),
            Duration::from_secs(10),
            1024,
        )
        .expect("runs");
        assert_eq!(out.stdout, b"hi\n");
        assert!(out.stderr.is_empty(), "not piped, not read");
        let out = run_with_timeout(
            &mut sh("echo hi; echo err >&2; exit 3"),
            Duration::from_secs(10),
            1024,
        )
        .expect("runs");
        assert_eq!(
            (out.stdout.as_slice(), out.stderr.as_slice()),
            (&b"hi\n"[..], &b"err\n"[..])
        );
        assert_eq!(out.status.code(), Some(3));
    }

    #[test]
    fn output_over_the_cap_fails_at_once() {
        // Review 2026-11-13 n9: more than a pipe's buffer used to block the interpreter until the timeout.
        for script in ["exec yes", "exec yes >&2"] {
            let started = Instant::now();
            let err = run_with_timeout(&mut sh(script), Duration::from_secs(30), 64 * 1024)
                .expect_err("too much");
            assert_ne!(err.kind(), std::io::ErrorKind::TimedOut, "{script}");
            assert!(
                err.to_string().contains("more than 65536 bytes"),
                "{script}: {err}"
            );
            assert!(started.elapsed() < Duration::from_secs(10), "{script}");
        }
    }

    #[test]
    fn a_grandchild_holding_the_pipe_does_not_hold_up_the_inspection() {
        // In its group: killed when the interpreter exits.
        let started = Instant::now();
        let out = run_with_timeout(&mut sh("sleep 30 & echo hi"), Duration::from_secs(20), 1024)
            .expect("runs");
        assert_eq!(out.stdout, b"hi\n");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
        // Out of its group: the timeout still covers the reads.
        if Path::new("/usr/bin/setsid").is_file() {
            let started = Instant::now();
            let err = run_with_timeout(
                &mut sh("/usr/bin/setsid sleep 5 & sleep 0.2; echo hi"),
                Duration::from_secs(1),
                1024,
            )
            .expect_err("the pipe stays open");
            assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
            assert!(
                started.elapsed() < Duration::from_secs(4),
                "{:?}",
                started.elapsed()
            );
        }
    }

    #[test]
    fn names_resolve_on_path_and_relative_paths_do_not() {
        assert!(resolve("no-such-interpreter-krishiv").is_none());
        assert!(resolve("bin/sh").is_none());
        assert_eq!(resolve("/bin/sh").as_deref(), Some(Path::new("/bin/sh")));
    }
}
