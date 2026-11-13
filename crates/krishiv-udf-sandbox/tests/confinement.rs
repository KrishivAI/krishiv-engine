// SPDX-License-Identifier: Apache-2.0
//! T014b acceptance: a real Python interpreter, started through the trampoline, is confined.
//!
//! Each test re-runs this test binary with [`CHILD_VAR`] set; the child entry ([`trampoline_child`]) calls
//! `trampoline_main` exactly as the server's hidden subcommand will, so the interpreter starts confined. The
//! Python snippet prints one `R:<check>:<outcome>` line per check, and the parent asserts on those lines.
//!
//! These tests need Linux with Landlock and seccomp and a `python3` on `PATH` (or `KRISHIV_TEST_PYTHON`).
//! They fail, never skip, when either is missing: the sandbox must not pass with nothing enforced
//! (`probe_matches_the_kernel` also asserts ABI 6 or later on CI).
#![cfg(target_os = "linux")]
#![expect(
    clippy::panic,
    clippy::print_stdout,
    reason = "test helpers sit outside #[test] fns; the child reports results on stdout"
)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use krishiv_udf_sandbox::{
    EXIT_SANDBOX_FAILED, Network, SandboxSpec, confine_current_thread, probe, trampoline_main,
};

/// Set in the child: the trampoline arguments, joined with [`SEP`].
const CHILD_VAR: &str = "KRISHIV_SANDBOX_TEST_ARGS";
const SEP: char = '\u{1f}';

/// The child entry. Without [`CHILD_VAR`] it does nothing and passes.
#[test]
fn trampoline_child() {
    let Some(encoded) = std::env::var_os(CHILD_VAR) else {
        return;
    };
    let encoded = encoded.to_string_lossy().into_owned();
    let args = encoded.split(SEP).map(OsString::from);
    if std::env::var_os(NO_EXEC_VAR).is_some() {
        rust_side_checks(args);
        return;
    }
    // Returns only on failure. The test then returns normally rather than exiting from this (possibly
    // confined) thread, so libtest's unconfined main thread exits and coverage data is written.
    let code = trampoline_main(args);
    let failed = code == ExitCode::from(EXIT_SANDBOX_FAILED);
    println!(
        "\nR:trampoline_exit:{}",
        if failed { "failed" } else { "other" }
    );
}

/// Set with [`CHILD_VAR`]: confine this thread with `confine` (no `exec`) and check the layers from Rust.
const NO_EXEC_VAR: &str = "KRISHIV_SANDBOX_TEST_NO_EXEC";

fn report(name: &str, result: std::io::Result<()>) {
    let outcome = match result {
        Ok(()) => "ok".to_owned(),
        Err(e) => format!("err:{}", e.raw_os_error().unwrap_or(-1)),
    };
    println!("\nR:{name}:{outcome}");
}

fn rust_side_checks(args: impl Iterator<Item = OsString>) {
    let spec = match SandboxSpec::from_args(args) {
        Ok(spec) => spec,
        Err(e) => return println!("\nR:confine:{e}"),
    };
    if let Err(e) = confine_current_thread(&spec) {
        return println!("\nR:confine:{e}");
    }
    println!("\nR:confine:ok");
    report("passwd", std::fs::read("/etc/passwd").map(drop));
    report(
        "private_write",
        std::fs::write(spec.private_dir.join("rust.txt"), "x"),
    );
    let port = std::env::var("KRISHIV_SANDBOX_TEST_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(9);
    report(
        "tcp",
        std::net::TcpStream::connect(("127.0.0.1", port)).map(drop),
    );
    report(
        "spawn",
        Command::new(&spec.interpreter)
            .arg("-c")
            .arg("0")
            .status()
            .map(drop),
    );
    report(
        "thread",
        std::thread::spawn(|| ())
            .join()
            .map_err(|_| std::io::Error::other("join")),
    );
}

fn python() -> PathBuf {
    if let Some(p) = std::env::var_os("KRISHIV_TEST_PYTHON") {
        return PathBuf::from(p);
    }
    std::env::var_os("PATH")
        .iter()
        .flat_map(std::env::split_paths)
        .map(|dir| dir.join("python3"))
        .find(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from("/usr/bin/python3"))
}

/// The interpreter's own read paths, asked of it unsandboxed (as the pool does at startup, note §3.5).
fn interpreter_paths(python: &Path) -> Vec<PathBuf> {
    let out = Command::new(python)
        .args([
            "-I",
            "-c",
            "import sys\nfor p in {sys.prefix, sys.base_prefix, sys.exec_prefix, *sys.path}: print(p)",
        ])
        .output();
    let Ok(out) = out else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect()
}

struct Dirs {
    _root: tempfile::TempDir,
    private: PathBuf,
    readable: PathBuf,
    outside: PathBuf,
}

fn dirs() -> Dirs {
    let root = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
    let mk = |name: &str| {
        let p = root.path().join(name);
        std::fs::create_dir(&p).unwrap_or_else(|e| panic!("mkdir {name}: {e}"));
        std::fs::write(p.join("file.txt"), "content")
            .unwrap_or_else(|e| panic!("write {name}: {e}"));
        p
    };
    let private = mk("private");
    let readable = mk("readable");
    let outside = mk("outside");
    Dirs {
        _root: root,
        private,
        readable,
        outside,
    }
}

struct Outcome {
    exit: Option<i32>,
    results: BTreeMap<String, String>,
    stderr: String,
}

impl Outcome {
    fn get(&self, check: &str) -> &str {
        self.results.get(check).map_or_else(
            || {
                panic!(
                    "no result for {check}: {:?}\nstderr: {}",
                    self.results, self.stderr
                )
            },
            String::as_str,
        )
    }
}

/// Python helpers prepended to every snippet: `check(name, f)` prints `R:name:ok` or
/// `R:name:<exception class>:<errno>`.
const PRELUDE: &str = r#"
import os, sys
def check(name, f):
    try:
        f()
        print(f"R:{name}:ok", flush=True)
    except BaseException as e:
        print(f"R:{name}:{type(e).__name__}:{getattr(e, 'errno', '')}", flush=True)
"#;

fn run(dirs: &Dirs, code: &str, configure: impl FnOnce(&mut SandboxSpec)) -> Outcome {
    let python = python();
    let mut spec = SandboxSpec::new(&python, &dirs.private);
    spec.args = vec![
        "-I".into(),
        "-c".into(),
        format!("{PRELUDE}\n{code}").into(),
    ];
    spec.env = vec![("KRISHIV_MARK".into(), "1".into())];
    spec.read_paths = SandboxSpec::system_read_paths();
    spec.read_paths.extend(interpreter_paths(&python));
    spec.read_paths.push(dirs.readable.clone());
    spec.parent_pid = Some(std::process::id());
    configure(&mut spec);
    run_args(&spec.to_args())
}

fn run_args(args: &[OsString]) -> Outcome {
    run_child(args, &[])
}

fn run_child(args: &[OsString], extra_env: &[(&str, &str)]) -> Outcome {
    let joined = args
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(&SEP.to_string());
    let exe = std::env::current_exe().unwrap_or_else(|e| panic!("current_exe: {e}"));
    let out = Command::new(exe)
        .args([
            "trampoline_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_VAR, joined)
        .env("SERVER_SECRET", "must-not-reach-the-worker")
        .envs(extra_env.iter().copied())
        .output()
        .unwrap_or_else(|e| panic!("spawning the trampoline: {e}"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let results = stdout
        .lines()
        // libtest prints `test trampoline_child ... ` without a newline before the interpreter starts.
        .filter_map(|l| l.find("R:").map(|at| &l[at + 2..]))
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    Outcome {
        exit: out.status.code(),
        results,
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

const EPERM: &str = "PermissionError:1";
const EACCES: &str = "PermissionError:13";

#[test]
fn the_kernel_can_sandbox() {
    let caps = probe();
    assert!(caps.landlock_abi >= 1 && caps.seccomp, "{caps:?}");
}

#[test]
fn udf_cannot_read_outside_allowed_roots() {
    let d = dirs();
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_owned());
    let code = format!(
        r#"
check("passwd", lambda: open("/etc/passwd").read())
check("outside", lambda: open({outside:?}).read())
check("home", lambda: os.listdir({home:?}))
check("readable", lambda: open({readable:?}).read())
check("stdlib", lambda: __import__("json").dumps([__import__("decimal").Decimal("1.5")], default=str))
"#,
        outside = d.outside.join("file.txt"),
        readable = d.readable.join("file.txt"),
    );
    let o = run(&d, &code, |_| {});
    assert_eq!(o.exit, Some(0), "stderr: {}", o.stderr);
    assert_eq!(o.get("passwd"), EACCES);
    assert_eq!(o.get("outside"), EACCES);
    assert_eq!(o.get("home"), EACCES);
    assert_eq!(o.get("readable"), "ok");
    assert_eq!(o.get("stdlib"), "ok");
}

#[test]
fn udf_cannot_write_outside_tmp() {
    let d = dirs();
    let code = format!(
        r#"
check("private", lambda: open({private:?}, "w").write("x"))
check("readable", lambda: open({readable:?}, "w").write("x"))
check("tmp", lambda: open("/tmp/krishiv-sandbox-escape", "w").write("x"))
check("cwd_is_private", lambda: None if os.getcwd() == {dir:?} else 1/0)
"#,
        private = d.private.join("new.txt"),
        readable = d.readable.join("new.txt"),
        dir = d.private,
    );
    let o = run(&d, &code, |_| {});
    assert_eq!(o.get("private"), "ok", "stderr: {}", o.stderr);
    assert_eq!(o.get("readable"), EACCES);
    assert_eq!(o.get("tmp"), EACCES);
    assert_eq!(o.get("cwd_is_private"), "ok");
}

#[test]
fn udf_fsize_quota_and_address_space() {
    let d = dirs();
    let code = format!(
        r#"
check("big_file", lambda: open({path:?}, "wb").write(b"x" * (2 << 20)))
check("small_file", lambda: open({path2:?}, "wb").write(b"x" * 1024))
check("huge_alloc", lambda: bytearray(1 << 31))
"#,
        path = d.private.join("big"),
        path2 = d.private.join("small"),
    );
    let o = run(&d, &code, |spec| {
        spec.limits.file_size = Some(1 << 20);
        spec.limits.address_space = Some(1 << 30);
    });
    assert_eq!(
        o.get("big_file"),
        "OSError:27",
        "EFBIG; stderr: {}",
        o.stderr
    );
    assert_eq!(o.get("small_file"), "ok");
    assert_eq!(o.get("huge_alloc"), "MemoryError:");
}

#[test]
fn udf_no_socket() {
    let d = dirs();
    let code = r#"
import socket
check("inet", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM))
check("inet6", lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM))
check("udp", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
check("docker", lambda: socket.socket(socket.AF_UNIX).connect("/var/run/docker.sock"))
check("socketpair", lambda: socket.socketpair())
"#;
    let o = run(&d, code, |_| {});
    for name in ["inet", "inet6", "udp", "docker"] {
        assert_eq!(o.get(name), EPERM, "{name}; stderr: {}", o.stderr);
    }
    assert_eq!(o.get("socketpair"), "ok");
}

#[test]
fn udf_no_fork_no_exec_threads_ok() {
    let d = dirs();
    let code = r#"
import subprocess, threading, multiprocessing
check("fork", lambda: os.fork())
check("subprocess", lambda: subprocess.run(["/bin/true"]))
check("multiprocessing", lambda: multiprocessing.get_context("fork").Process(target=print).start())
check("exec", lambda: os.execv("/bin/sh", ["sh", "-c", "true"]))
def threads():
    out = []
    ts = [threading.Thread(target=out.append, args=(i,)) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sorted(out) == [0, 1, 2, 3]
check("threads", threads)
"#;
    let o = run(&d, code, |_| {});
    assert_eq!(o.get("fork"), EPERM, "stderr: {}", o.stderr);
    assert!(
        o.get("subprocess").starts_with("PermissionError"),
        "{}",
        o.get("subprocess")
    );
    assert_eq!(o.get("multiprocessing"), EPERM);
    assert_eq!(o.get("exec"), EACCES);
    assert_eq!(o.get("threads"), "ok");
}

#[test]
fn allow_fork_keeps_the_other_layers() {
    let d = dirs();
    let code = r#"
import socket
def fork_child():
    pid = os.fork()
    if pid == 0:
        os._exit(7)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 7
check("fork", fork_child)
check("socket", lambda: socket.socket())
check("passwd", lambda: open("/etc/passwd").read())
"#;
    let o = run(&d, code, |spec| spec.allow_fork = true);
    assert_eq!(o.get("fork"), "ok", "stderr: {}", o.stderr);
    assert_eq!(o.get("socket"), EPERM);
    assert_eq!(o.get("passwd"), EACCES);
}

#[test]
fn udf_no_ptrace_and_kill_only_self() {
    let d = dirs();
    let code = r#"
import ctypes, signal
libc = ctypes.CDLL(None, use_errno=True)
def ptrace():
    if libc.ptrace(0, 0, None, None) != 0:
        raise OSError(ctypes.get_errno(), "ptrace")
check("ptrace", ptrace)
check("kill_parent", lambda: os.kill(os.getppid(), 0))
check("kill_init", lambda: os.kill(1, 0))
check("kill_self", lambda: os.kill(os.getpid(), 0))
"#;
    let o = run(&d, code, |_| {});
    assert_eq!(o.get("ptrace"), EPERM, "stderr: {}", o.stderr);
    assert_eq!(o.get("kill_parent"), EPERM);
    assert_eq!(o.get("kill_init"), EPERM);
    assert_eq!(o.get("kill_self"), "ok");
}

#[test]
fn udf_clean_env() {
    let d = dirs();
    // Python may add LC_CTYPE itself (PEP 538 locale coercion); nothing of the server's environment arrives.
    let code = r#"
check("env", lambda: None if set(os.environ) <= {"KRISHIV_MARK", "LC_CTYPE"} and os.environ["KRISHIV_MARK"] == "1" else 1/0)
check("isolated", lambda: None if sys.flags.isolated else 1/0)
"#;
    let o = run(&d, code, |_| {});
    assert_eq!(o.get("env"), "ok", "stderr: {}", o.stderr);
    assert_eq!(o.get("isolated"), "ok");
}

#[test]
fn tcp_allowlist_opens_only_its_ports() {
    let open = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|e| panic!("bind: {e}"));
    let closed = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|e| panic!("bind: {e}"));
    let open_port = open.local_addr().map(|a| a.port()).unwrap_or(0);
    let closed_port = closed.local_addr().map(|a| a.port()).unwrap_or(0);
    let d = dirs();
    let code = format!(
        r#"
import socket
check("allowed", lambda: socket.create_connection(("127.0.0.1", {open_port}), timeout=5).close())
check("other_port", lambda: socket.create_connection(("127.0.0.1", {closed_port}), timeout=5).close())
check("bind", lambda: socket.socket().bind(("127.0.0.1", 0)))
check("udp", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
check("unix", lambda: socket.socket(socket.AF_UNIX))
# Landlock's TCP rules do not apply to MPTCP or SCTP stream sockets, so seccomp refuses those protocols.
check("mptcp", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM, 262))
check("sctp", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM, 132))
check("tcp_protocol", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP).close())
"#
    );
    let o = run(&d, &code, |spec| {
        spec.network = Network::TcpConnect(vec![open_port]);
    });
    if probe().landlock_abi < 4 {
        // An allowlist the kernel cannot enforce is refused, never dropped.
        assert_eq!(o.get("trampoline_exit"), "failed", "{}", o.stderr);
        assert!(
            o.stderr.contains("KRISHIV_UDF_SANDBOX_UNAVAILABLE"),
            "{}",
            o.stderr
        );
        return;
    }
    assert_eq!(o.get("allowed"), "ok", "stderr: {}", o.stderr);
    assert_eq!(o.get("other_port"), EACCES);
    assert_eq!(o.get("bind"), EACCES);
    assert_eq!(o.get("udp"), EPERM);
    assert_eq!(o.get("unix"), EPERM);
    assert_eq!(o.get("mptcp"), EPERM);
    assert_eq!(o.get("sctp"), EPERM);
    assert_eq!(o.get("tcp_protocol"), "ok");
}

#[test]
fn failures_exit_with_the_class_on_stderr() {
    let o = run_args(&["--bogus".into(), "x".into()]);
    assert_eq!(o.get("trampoline_exit"), "failed", "{}", o.stderr);
    assert!(o.stderr.contains("INTERNAL_ERROR"), "{}", o.stderr);

    let d = dirs();
    let spec = SandboxSpec::new("/nonexistent/python3", &d.private);
    let o = run_args(&spec.to_args());
    assert_eq!(o.get("trampoline_exit"), "failed", "{}", o.stderr);
    assert!(
        o.stderr.contains("KRISHIV_UDF_SANDBOX_UNAVAILABLE"),
        "{}",
        o.stderr
    );
    assert!(o.stderr.contains("/nonexistent/python3"), "{}", o.stderr);

    let mut spec = SandboxSpec::new(python(), &d.private);
    spec.parent_pid = Some(1);
    let o = run_args(&spec.to_args());
    assert_eq!(o.get("trampoline_exit"), "failed", "{}", o.stderr);
    assert!(
        o.stderr.contains("exited before the worker started"),
        "{}",
        o.stderr
    );
}

#[test]
fn confine_without_exec_binds_the_thread() {
    let d = dirs();
    let mut spec = SandboxSpec::new(python(), &d.private);
    spec.read_paths = SandboxSpec::system_read_paths();
    let o = run_child(&spec.to_args(), &[(NO_EXEC_VAR, "1")]);
    assert_eq!(o.get("confine"), "ok", "stderr: {}", o.stderr);
    assert_eq!(o.get("passwd"), "err:13");
    assert_eq!(o.get("private_write"), "ok");
    assert_eq!(o.get("tcp"), "err:1");
    assert_eq!(o.get("spawn"), "err:1");
    assert_eq!(o.get("thread"), "ok");
    // The trampoline's own refusal of a spec the kernel cannot enforce is checked in `probe`; a missing
    // private directory is refused before anything is applied.
    let spec = SandboxSpec::new(python(), d.private.join("missing"));
    let o = run_child(&spec.to_args(), &[(NO_EXEC_VAR, "1")]);
    assert!(
        o.get("confine").contains("KRISHIV_UDF_SANDBOX_UNAVAILABLE"),
        "{}",
        o.get("confine")
    );
    assert!(
        o.get("confine").contains("private directory"),
        "{}",
        o.get("confine")
    );
}

#[test]
fn an_interpreter_that_cannot_run_reports_the_exec_error() {
    let d = dirs();
    // Readable and allowed by Landlock, but not executable: `execve` itself fails.
    let not_executable = d.readable.join("file.txt");
    let spec = SandboxSpec::new(&not_executable, &d.private);
    let o = run_args(&spec.to_args());
    assert_eq!(o.get("trampoline_exit"), "failed", "{}", o.stderr);
    assert!(
        o.stderr.contains("KRISHIV_UDF_SANDBOX_UNAVAILABLE"),
        "{}",
        o.stderr
    );
    assert!(o.stderr.contains("in the sandbox failed"), "{}", o.stderr);
}

#[test]
fn confine_without_exec_with_a_tcp_allowlist() {
    if probe().landlock_abi < 4 {
        return; // `tcp_allowlist_opens_only_its_ports` checks the refusal on such kernels.
    }
    let open = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|e| panic!("bind: {e}"));
    let port = open.local_addr().map(|a| a.port()).unwrap_or(0);
    let d = dirs();
    let mut spec = SandboxSpec::new(python(), &d.private);
    spec.network = Network::TcpConnect(vec![port]);
    let port_text = port.to_string();
    let o = run_child(
        &spec.to_args(),
        &[
            (NO_EXEC_VAR, "1"),
            ("KRISHIV_SANDBOX_TEST_PORT", &port_text),
        ],
    );
    assert_eq!(o.get("confine"), "ok", "stderr: {}", o.stderr);
    assert_eq!(o.get("tcp"), "ok");
    let other = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|e| panic!("bind: {e}"));
    let other_text = other
        .local_addr()
        .map(|a| a.port())
        .unwrap_or(0)
        .to_string();
    let o = run_child(
        &spec.to_args(),
        &[
            (NO_EXEC_VAR, "1"),
            ("KRISHIV_SANDBOX_TEST_PORT", &other_text),
        ],
    );
    assert_eq!(o.get("tcp"), "err:13", "stderr: {}", o.stderr);
}

/// A same-uid sibling process (as the server is) cannot be reconfigured: its limits, priority and CPU
/// affinity stay as they were. The worker can still change its own.
#[test]
fn udf_cannot_touch_other_processes() {
    let mut sibling = Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap_or_else(|e| panic!("spawning sleep: {e}"));
    let pid = sibling.id();
    let limits_before = std::fs::read_to_string(format!("/proc/{pid}/limits")).unwrap_or_default();
    let d = dirs();
    let code = format!(
        r#"
import resource
check("prlimit_other", lambda: resource.prlimit({pid}, resource.RLIMIT_CPU, (1, 1)))
check("setpriority_other", lambda: os.setpriority(os.PRIO_PROCESS, {pid}, 19))
check("setpriority_user", lambda: os.setpriority(os.PRIO_USER, 0, 19))
check("setpriority_group", lambda: os.setpriority(os.PRIO_PGRP, 0, 19))
check("affinity_other", lambda: os.sched_setaffinity({pid}, {{0}}))
check("prlimit_self", lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
check("setpriority_self", lambda: os.setpriority(os.PRIO_PROCESS, 0, 5))
check("affinity_self", lambda: os.sched_setaffinity(0, os.sched_getaffinity(0)))
"#
    );
    let o = run(&d, &code, |_| {});
    let limits_after = std::fs::read_to_string(format!("/proc/{pid}/limits")).unwrap_or_default();
    let _ = sibling.kill();
    let _ = sibling.wait();
    for name in [
        "prlimit_other",
        "setpriority_other",
        "setpriority_user",
        "setpriority_group",
        "affinity_other",
    ] {
        assert_eq!(o.get(name), EPERM, "{name}; stderr: {}", o.stderr);
    }
    for name in ["prlimit_self", "setpriority_self", "affinity_self"] {
        assert_eq!(o.get(name), "ok", "{name}; stderr: {}", o.stderr);
    }
    assert_eq!(limits_before, limits_after);
}

/// x32 system calls, path truncation and terminal input injection are refused; the worker runs in a session
/// of its own.
#[test]
fn x32_truncate_terminal_and_session() {
    let d = dirs();
    let code = format!(
        r#"
import ctypes, fcntl, termios
libc = ctypes.CDLL(None, use_errno=True)
def x32_getpid():
    if libc.syscall(0x40000000 | 39) == -1:
        raise OSError(ctypes.get_errno(), "x32")
check("x32", x32_getpid)
check("truncate_path", lambda: os.truncate({readable:?}, 0))
check("truncate_private", lambda: os.truncate({private:?}, 0))
def ftruncate_private():
    with open({private:?}, "r+b") as f:
        os.ftruncate(f.fileno(), 1)
check("ftruncate_private", ftruncate_private)
check("tiocsti", lambda: fcntl.ioctl(0, termios.TIOCSTI, b"x"))
check("own_session", lambda: None if os.getsid(0) == os.getpid() else 1/0)
"#,
        readable = d.readable.join("file.txt"),
        private = d.private.join("file.txt"),
    );
    let o = run(&d, &code, |_| {});
    assert_eq!(o.get("x32"), "OSError:38", "ENOSYS; stderr: {}", o.stderr);
    assert_eq!(o.get("truncate_path"), EPERM);
    assert_eq!(o.get("truncate_private"), EPERM);
    assert_eq!(o.get("ftruncate_private"), "ok");
    assert_eq!(o.get("tiocsti"), EPERM);
    assert_eq!(o.get("own_session"), "ok");
}
