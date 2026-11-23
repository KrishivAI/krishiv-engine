// SPDX-License-Identifier: Apache-2.0
//! The pool with real worker processes (design `P3-T014-connect-udfs-sandbox.md` §5, the PR-26b part).
//!
//! The interpreter comes from `KRISHIV_TEST_UDF_PYTHON`: Python 3.11 or later with `pyspark-client` (or
//! `pyspark`), `pyarrow` and `pandas`. Without it every test passes after a note on standard error, unless
//! `KRISHIV_UDF_POOL_REQUIRED` is set (the CI job that installs the interpreter sets it), when they fail.
//!
//! Sandboxed workers are started through this test binary: the launcher runs it with the libtest arguments
//! `worker_child --exact --nocapture --`, and [`worker_child`] hands the arguments after `--` to the
//! sandbox trampoline, exactly as the server's hidden subcommand does.
#![cfg(target_os = "linux")]
#![expect(
    clippy::print_stderr,
    clippy::expect_used,
    reason = "the skip note when no interpreter is configured; test helpers sit outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::python_eval::{UdfCallSpec, UdfRuntime};
use krishiv_plan::function::PythonEvalType;
use krishiv_types::DataType;
use krishiv_udf_pool::{Launcher, Owner, Pool, PoolConfig, PythonPayload, SessionUdfs};
use krishiv_udf_sandbox::Capabilities;

/// The child entry: run as the trampoline when started with `--` and spec arguments.
#[test]
fn worker_child() {
    let args: Vec<OsString> = std::env::args_os().collect();
    let Some(split) = args.iter().position(|a| a == "--") else {
        return;
    };
    let spec: Vec<OsString> = args.get(split + 1..).unwrap_or_default().to_vec();
    if spec.is_empty() {
        return;
    }
    // Returns only on failure; the exit status then tells the pool (EXIT_SANDBOX_FAILED).
    let code = krishiv_udf_sandbox::trampoline_main(spec);
    let failed = code == ExitCode::from(krishiv_udf_sandbox::EXIT_SANDBOX_FAILED);
    std::process::exit(if failed { 125 } else { 1 });
}

fn python() -> Option<PathBuf> {
    match std::env::var_os("KRISHIV_TEST_UDF_PYTHON") {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            assert!(
                std::env::var_os("KRISHIV_UDF_POOL_REQUIRED").is_none(),
                "KRISHIV_UDF_POOL_REQUIRED is set but KRISHIV_TEST_UDF_PYTHON is not"
            );
            eprintln!("note: KRISHIV_TEST_UDF_PYTHON is not set; the pool tests did not run");
            None
        }
    }
}

fn config(python: &std::path::Path) -> PoolConfig {
    let mut config = PoolConfig::default();
    config.executables = vec![python.display().to_string()];
    config.launcher = Launcher {
        program: std::env::current_exe().expect("test binary"),
        args: ["worker_child", "--exact", "--nocapture", "--"]
            .map(OsString::from)
            .to_vec(),
    };
    config
}

fn owner(session: &str) -> Owner {
    Owner {
        principal: "alice".to_owned(),
        session: session.to_owned(),
        tenant: None,
        capped: true,
    }
}

/// Spark's `command` for `body` (a Python function `f`) and `return_type` (a pyspark type expression),
/// made by the interpreter's own cloudpickle, as a client makes it.
fn command(python: &std::path::Path, body: &str, return_type: &str) -> PythonPayload {
    let code = format!(
        "import sys\nfrom pyspark import cloudpickle\nfrom pyspark.sql.types import *\n{body}\n\
         sys.stdout.buffer.write(cloudpickle.dumps((f, {return_type})))"
    );
    let out = Command::new(python)
        .args(["-c", &code])
        .output()
        .expect("run the interpreter");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let version = Command::new(python)
        .args([
            "-c",
            "import sys; print('%d.%d' % sys.version_info[:2], end='')",
        ])
        .output()
        .expect("version");
    PythonPayload {
        command: Arc::from(out.stdout),
        eval_type: 101,
        python_ver: String::from_utf8_lossy(&version.stdout).into_owned(),
    }
}

fn spec(name: &str, eval_type: PythonEvalType, return_arrow: ArrowType) -> UdfCallSpec {
    let return_type = match return_arrow {
        ArrowType::Utf8View => DataType::String,
        _ => DataType::Long,
    };
    UdfCallSpec {
        name: name.to_owned(),
        version: 1,
        eval_type,
        thread_safe: true,
        arg_types: vec![DataType::Long],
        return_type,
        return_arrow,
    }
}

fn longs(values: &[Option<i64>]) -> RecordBatch {
    RecordBatch::try_from_iter([("a", Arc::new(Int64Array::from(values.to_vec())) as ArrayRef)])
        .expect("batch")
}

/// Registers `body` as `name`, prepares it and calls it over `args`.
fn call(
    udfs: &SessionUdfs,
    payload: PythonPayload,
    eval_type: (i32, PythonEvalType),
    return_arrow: ArrowType,
    args: RecordBatch,
) -> Result<ArrayRef, KrishivError> {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let name = format!(
        "f{}",
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    udfs.register(
        name.clone(),
        1,
        PythonPayload {
            eval_type: eval_type.0,
            ..payload
        },
    );
    let prepared = udfs.prepare(&spec(&name, eval_type.1, return_arrow))?;
    block_on(prepared.call(args))
}

const ARROW_BATCHED: (i32, PythonEvalType) = (101, PythonEvalType::ArrowBatched);
const PANDAS: (i32, PythonEvalType) = (200, PythonEvalType::ScalarPandas);
const ARROW: (i32, PythonEvalType) = (250, PythonEvalType::ScalarArrow);

fn as_longs(a: &ArrayRef) -> Vec<Option<i64>> {
    a.as_any()
        .downcast_ref::<Int64Array>()
        .expect("longs")
        .iter()
        .collect()
}

#[test]
fn scalar_kinds_run_in_a_sandboxed_worker() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    assert!(pool.refusal().is_none(), "{:?}", pool.refusal());
    let udfs = SessionUdfs::new(pool.clone(), owner("s1"));
    let doubled = call(
        &udfs,
        command(
            &py,
            "def f(x):\n    return None if x is None else x * 2",
            "LongType()",
        ),
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1), None, Some(3)]),
    )
    .expect("@udf");
    assert_eq!(as_longs(&doubled), [Some(2), None, Some(6)]);
    // A pandas UDF using numpy and threads (the sandbox allows threads).
    let pandas = call(
        &udfs,
        command(
            &py,
            "import numpy as np\nfrom concurrent.futures import ThreadPoolExecutor\n\
             def f(s):\n    with ThreadPoolExecutor(2) as ex:\n        \
             parts = list(ex.map(lambda v: v + 1, [s]))\n    return parts[0].astype(np.int64)",
            "LongType()",
        ),
        PANDAS,
        ArrowType::Int64,
        longs(&[Some(10), Some(20)]),
    )
    .expect("pandas_udf");
    assert_eq!(as_longs(&pandas), [Some(11), Some(21)]);
    let strings = call(
        &udfs,
        command(
            &py,
            "import pyarrow as pa, pyarrow.compute as pc\n\
             def f(a):\n    return pc.cast(a, pa.string())",
            "StringType()",
        ),
        ARROW,
        ArrowType::Utf8View,
        longs(&[Some(7)]),
    )
    .expect("arrow_udf");
    let strings = strings
        .as_any()
        .downcast_ref::<StringViewArray>()
        .expect("string view");
    assert_eq!(strings.value(0), "7");
    assert_eq!(pool.live_workers(), 1, "one worker served all three calls");
}

#[test]
fn a_snapshot_lists_the_sessions_worker_and_a_monitor_outlives_the_pool_without_keeping_it() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    let monitor = pool.monitor();
    let empty = pool.snapshot();
    assert_eq!((empty.live, empty.idle, empty.owners.len()), (0, 0, 0));
    let udfs = SessionUdfs::new(pool.clone(), owner("s1"));
    call(
        &udfs,
        command(&py, "def f(x):\n    return x", "LongType()"),
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(5)]),
    )
    .expect("call");
    // The call is over: its worker waits for the next one.
    let after = monitor.snapshot().expect("the pool is alive");
    assert_eq!((after.live, after.idle), (1, 1));
    assert_eq!(after.max_total, pool.snapshot().max_total);
    assert_eq!(after.owners.len(), 1);
    assert_eq!(after.owners[0].owner, owner("s1"));
    assert_eq!((after.owners[0].live, after.owners[0].idle), (1, 1));
    drop(udfs);
    drop(pool);
    // The reaper thread holds the pool for the length of one pass, so the pool may outlive the last
    // handle by that much; the monitor never keeps it.
    let deadline = Instant::now() + Duration::from_secs(10);
    while monitor.snapshot().is_some() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        monitor.snapshot(),
        None,
        "the monitor does not keep the pool"
    );
}

#[test]
fn a_python_error_is_python_exception_and_the_worker_stays() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    let udfs = SessionUdfs::new(pool.clone(), owner("s1"));
    let err = call(
        &udfs,
        command(&py, "def f(x):\n    return 1 // 0", "LongType()"),
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect_err("raises");
    assert_eq!(err.error_class(), ErrorClass::PythonException);
    assert!(err.to_string().contains("ZeroDivisionError"), "{err}");
    let ok = call(
        &udfs,
        command(&py, "def f(x):\n    return x", "LongType()"),
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(5)]),
    )
    .expect("the worker is reused");
    assert_eq!(as_longs(&ok), [Some(5)]);
    assert_eq!(pool.live_workers(), 1);
}

#[test]
fn python_version_mismatch_is_reported() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    let udfs = SessionUdfs::new(pool, owner("s1"));
    let mut payload = command(&py, "def f(x):\n    return x", "LongType()");
    payload.python_ver = "3.9".to_owned();
    let err = call(
        &udfs,
        payload,
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect_err("mismatch");
    assert_eq!(err.error_class(), ErrorClass::PythonException);
    assert!(
        err.to_string()
            .contains("[PYTHON_VERSION_MISMATCH] Python in worker has different version"),
        "{err}"
    );
}

#[test]
fn worker_crash_respawns() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    let udfs = SessionUdfs::new(pool.clone(), owner("s1"));
    for (body, how) in [
        ("import os\ndef f(x):\n    os._exit(3)", "exit code 3"),
        // The trampoline's reserved status, from the UDF after the greeting: an ordinary exit, not the sandbox.
        ("import os\ndef f(x):\n    os._exit(125)", "exit code 125"),
        (
            "import os, signal\ndef f(x):\n    os.kill(os.getpid(), signal.SIGSEGV)",
            "signal 11",
        ),
    ] {
        let err = call(
            &udfs,
            command(&py, body, "LongType()"),
            ARROW_BATCHED,
            ArrowType::Int64,
            longs(&[Some(1)]),
        )
        .expect_err("crashes");
        assert_eq!(err.error_class(), ErrorClass::KrishivPythonWorkerExited);
        assert!(err.to_string().contains(how), "{err}");
        assert_eq!(pool.live_workers(), 0, "the crashed worker is forgotten");
        let ok = call(
            &udfs,
            command(&py, "def f(x):\n    return x + 1", "LongType()"),
            ARROW_BATCHED,
            ArrowType::Int64,
            longs(&[Some(1)]),
        )
        .expect("a new worker");
        assert_eq!(as_longs(&ok), [Some(2)]);
    }
}

#[test]
fn udf_clean_env_and_fds() {
    let Some(py) = python() else { return };
    // A descriptor the server opens without close-on-exec must not reach a UDF (T-U1).
    let leaked = rustix::fs::open(
        "/dev/null",
        rustix::fs::OFlags::RDONLY,
        rustix::fs::Mode::empty(),
    )
    .expect("open without O_CLOEXEC");
    let fd_no = std::os::fd::AsRawFd::as_raw_fd(&leaked);
    let pool = Pool::new(config(&py)).expect("pool");
    let udfs = SessionUdfs::new(pool, owner("s1"));
    let body = "import os\n\
        def f(x):\n    \
        fds, inputs = [], set()\n    \
        for fd in range(3, 4096):\n        \
        try:\n            os.fstat(fd)\n        except OSError:\n            continue\n        \
        target = os.readlink(f'/proc/self/fd/{fd}')\n        \
        if target.startswith('/memfd:krishiv-udf-input'):\n            inputs.add(os.fstat(fd).st_ino)\n        \
        else:\n            fds.append(f'{fd}={target}')\n    \
        env = sorted(os.environ)\n    \
        return ' '.join(fds) + '|' + ','.join(env) + f'|{len(inputs)}'";
    let out = call(
        &udfs,
        command(&py, body, "StringType()"),
        ARROW_BATCHED,
        ArrowType::Utf8View,
        longs(&[Some(1)]),
    )
    .expect("runs");
    let out = out
        .as_any()
        .downcast_ref::<StringViewArray>()
        .expect("string")
        .value(0)
        .to_owned();
    // While the UDF runs, the only memfd is the batch's own input (open twice: the received descriptor and
    // the one `mmap` duplicates); any other descriptor, a memfd too, would be inherited.
    assert_eq!(
        out, "|HOME,LC_ALL,TMPDIR|1",
        "inherited fd {fd_no} or server env leaked"
    );
    drop(leaked);
}

#[test]
fn the_sandbox_confines_udfs() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    let udfs = SessionUdfs::new(pool, owner("s1"));
    let body = "import os, socket\n\
        def f(x):\n    \
        out = []\n    \
        for name, op in [('passwd', lambda: open('/etc/passwd').read()), \
        ('socket', lambda: socket.socket()), ('fork', lambda: os.fork())]:\n        \
        try:\n            op()\n            out.append(name + '=open')\n        \
        except OSError:\n            out.append(name + '=denied')\n    \
        return ' '.join(out)";
    let out = call(
        &udfs,
        command(&py, body, "StringType()"),
        ARROW_BATCHED,
        ArrowType::Utf8View,
        longs(&[Some(1)]),
    )
    .expect("runs");
    let out = out
        .as_any()
        .downcast_ref::<StringViewArray>()
        .expect("string")
        .value(0)
        .to_owned();
    assert_eq!(out, "passwd=denied socket=denied fork=denied");
}

/// The parent-death signal is per thread and new threads start without it, so an `exec` from a second thread
/// would give an image that outlives the server. The bootstrap's second filter refuses `execve` (and
/// `execveat`) in every thread.
#[test]
fn udf_cannot_outlive_the_server_by_exec_from_a_thread() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    let udfs = SessionUdfs::new(pool, owner("s1"));
    let body = "import os, sys, threading\n\
        def f(x):\n    \
        out = []\n    \
        def attempt():\n        \
        try:\n            os.execv(sys.executable, [sys.executable, '-c', 'pass'])\n        \
        except OSError as e:\n            out.append(f'thread_exec={e.errno}')\n    \
        t = threading.Thread(target=attempt)\n    \
        t.start()\n    t.join()\n    \
        try:\n        os.execv(sys.executable, [sys.executable, '-c', 'pass'])\n    \
        except OSError as e:\n        out.append(f'exec={e.errno}')\n    \
        return ' '.join(out)";
    let out = call(
        &udfs,
        command(&py, body, "StringType()"),
        ARROW_BATCHED,
        ArrowType::Utf8View,
        longs(&[Some(1)]),
    )
    .expect("runs");
    let out = out
        .as_any()
        .downcast_ref::<StringViewArray>()
        .expect("string")
        .value(0)
        .to_owned();
    // EPERM (1) from the bootstrap's filter: UDF calls already run on threads other than the first, so both
    // attempts are the escape the filter closes.
    assert_eq!(out, "thread_exec=1 exec=1");
}

#[test]
fn sandbox_unavailable_refuses() {
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.capabilities = Some(Capabilities::new(0, true));
    let pool = Pool::new(config).expect("pool");
    let udfs = SessionUdfs::new(pool, owner("s1"));
    let err = call(
        &udfs,
        command(&py, "def f(x):\n    return x", "LongType()"),
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect_err("refused");
    assert_eq!(err.error_class(), ErrorClass::KrishivUdfSandboxUnavailable);
}

#[test]
fn sandbox_off_runs_with_limits_and_hygiene() {
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.sandbox = false;
    config.capabilities = Some(Capabilities::new(0, false));
    config.memory_limit = Some(1 << 30);
    let pool = Pool::new(config).expect("pool");
    assert!(
        pool.refusal().is_none(),
        "off runs even where the kernel cannot sandbox"
    );
    let udfs = SessionUdfs::new(pool, owner("s1"));
    let body = "import os, resource\n\
        def f(x):\n    \
        return f'{resource.getrlimit(resource.RLIMIT_AS)[0]}|' + ','.join(sorted(os.environ))";
    let out = call(
        &udfs,
        command(&py, body, "StringType()"),
        ARROW_BATCHED,
        ArrowType::Utf8View,
        longs(&[Some(1)]),
    )
    .expect("runs");
    let out = out
        .as_any()
        .downcast_ref::<StringViewArray>()
        .expect("string")
        .value(0)
        .to_owned();
    assert_eq!(out, format!("{}|HOME,LC_ALL,TMPDIR", 1_u64 << 30));
}

#[test]
fn a_udf_over_its_memory_limit_fails_the_call_not_the_server() {
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.memory_limit = Some(1 << 30);
    let pool = Pool::new(config).expect("pool");
    let udfs = SessionUdfs::new(pool, owner("s1"));
    let err = call(
        &udfs,
        command(
            &py,
            "def f(x):\n    return len(bytearray(2 << 30))",
            "LongType()",
        ),
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect_err("over the limit");
    assert_eq!(err.error_class(), ErrorClass::PythonException);
    assert!(err.to_string().contains("MemoryError"), "{err}");
}

#[test]
fn interrupt_stops_udf_within_1s() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    let udfs = SessionUdfs::new(pool.clone(), owner("s1"));
    let payload = command(
        &py,
        "import time\ndef f(x):\n    time.sleep(60)\n    return x",
        "LongType()",
    );
    udfs.register("slow", 1, payload);
    let prepared = udfs
        .prepare(&spec(
            "slow",
            PythonEvalType::ArrowBatched,
            ArrowType::Int64,
        ))
        .expect("prepare");
    // Warm the worker first, so the cancelled call finds it started and is mid-UDF when dropped.
    call(
        &udfs,
        command(&py, "def f(x):\n    return x", "LongType()"),
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect("warm");
    assert_eq!(pool.live_workers(), 1);
    let mut fut = prepared.call(longs(&[Some(1)]));
    assert!(futures::FutureExt::now_or_never(&mut fut).is_none());
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        futures::FutureExt::now_or_never(&mut fut).is_none(),
        "still sleeping"
    );
    let cancelled = Instant::now();
    drop(fut);
    while pool.live_workers() > 0 && cancelled.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(pool.live_workers(), 0, "the worker was not stopped");
    assert!(
        cancelled.elapsed() < Duration::from_secs(1),
        "{:?}",
        cancelled.elapsed()
    );
}

#[test]
fn workers_isolated_by_session_and_reused_within_one() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    // Module state lives in a real module: each payload is unpickled with globals of its own.
    let body = "import os, sys\n\
        def f(x):\n    state = sys.__dict__.setdefault('_krishiv_test_state', [])\n    \
        state.append(1)\n    return os.getpid() * 1000 + len(state)";
    let run = |session: &str| {
        let udfs = SessionUdfs::new(pool.clone(), owner(session));
        let out = call(
            &udfs,
            command(&py, body, "LongType()"),
            ARROW_BATCHED,
            ArrowType::Int64,
            longs(&[Some(1)]),
        )
        .expect("runs");
        as_longs(&out)[0].expect("value")
    };
    let a1 = run("a");
    let a2 = run("a");
    let b1 = run("b");
    assert_eq!(a1 / 1000, a2 / 1000, "one session reuses its worker");
    assert_eq!(a2 % 1000, a1 % 1000 + 1, "and its module state");
    assert_ne!(a1 / 1000, b1 / 1000, "another session gets its own worker");
    assert_eq!(b1 % 1000, 1, "and never sees the first's state");
}

#[test]
fn idle_workers_stop_and_released_sessions_free_theirs() {
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.idle_timeout = Duration::from_millis(300);
    let pool = Pool::new(config).expect("pool");
    let udfs = SessionUdfs::new(pool.clone(), owner("s1"));
    let payload = command(&py, "def f(x):\n    return x", "LongType()");
    call(
        &udfs,
        payload.clone(),
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect("runs");
    assert_eq!(pool.live_workers(), 1);
    let waited = Instant::now();
    while pool.live_workers() > 0 && waited.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(pool.live_workers(), 0, "the idle timeout stops it");

    let mut config = self::config(&py);
    config.idle_timeout = Duration::from_secs(600);
    let pool = Pool::new(config).expect("pool");
    let udfs = SessionUdfs::new(pool.clone(), owner("s2"));
    call(
        &udfs,
        payload,
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect("runs");
    assert_eq!(pool.live_workers(), 1);
    udfs.release();
    // The workers stop on a release thread, and their slots are freed once they are gone.
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while pool.live_workers() > 0 && Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        pool.live_workers(),
        0,
        "release stops the session's idle workers"
    );
}

#[test]
fn calls_wait_at_the_session_cap() {
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.max_per_session = 1;
    let pool = Pool::new(config).expect("pool");
    let udfs = Arc::new(SessionUdfs::new(pool.clone(), owner("s1")));
    let payload = command(
        &py,
        "import time\ndef f(x):\n    time.sleep(0.2)\n    return x",
        "LongType()",
    );
    udfs.register("nap", 1, payload);
    let prepared = udfs
        .prepare(&spec("nap", PythonEvalType::ArrowBatched, ArrowType::Int64))
        .expect("prepare");
    let calls: Vec<_> = (0..3).map(|i| prepared.call(longs(&[Some(i)]))).collect();
    let results = block_on(futures::future::join_all(calls));
    for (i, r) in results.into_iter().enumerate() {
        assert_eq!(
            as_longs(&r.expect("runs")),
            [Some(i64::try_from(i).expect("small"))]
        );
    }
    assert_eq!(pool.live_workers(), 1, "never more than the cap");
}

#[test]
fn an_unregistered_udf_is_not_available() {
    let Some(py) = python() else { return };
    let pool = Pool::new(config(&py)).expect("pool");
    let udfs = SessionUdfs::new(pool, owner("s1"));
    udfs.set_runner_conf(BTreeMap::new());
    let err = udfs
        .prepare(&spec(
            "nope",
            PythonEvalType::ArrowBatched,
            ArrowType::Int64,
        ))
        .err()
        .expect("not registered");
    assert_eq!(err.error_class(), ErrorClass::KrishivUdfNotAvailable);
}

/// A pool whose "trampoline" is `/bin/sh -c script`: the script plays a worker that fails at start.
fn fake_worker_pool(py: &std::path::Path, script: &str) -> Pool {
    let mut config = config(py);
    config.launcher = Launcher {
        program: PathBuf::from("/bin/sh"),
        // The spec's own arguments follow as the script's positional parameters.
        args: vec!["-c".into(), script.into(), "sh".into()],
    };
    Pool::new(config).expect("pool")
}

fn start_error(pool: Pool) -> KrishivError {
    let udfs = SessionUdfs::new(pool, owner("s1"));
    call(
        &udfs,
        PythonPayload {
            command: Arc::from(&b"unused"[..]),
            eval_type: 101,
            python_ver: "3.12".to_owned(),
        },
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect_err("the worker does not start")
}

#[test]
fn workers_that_fail_at_start() {
    let Some(py) = python() else { return };
    let cases: [(&str, ErrorClass, &str); 5] = [
        (
            "echo 'the kernel said no' >&2; exit 125",
            ErrorClass::KrishivUdfSandboxUnavailable,
            "the kernel said no",
        ),
        (
            "echo 'cannot import' >&2; exit 3",
            ErrorClass::KrishivPythonWorkerExited,
            "exit code 3",
        ),
        (
            r#"printf '{"t":"hello","wire":9}' >&0; sleep 5"#,
            ErrorClass::KrishivPythonWorkerExited,
            "wire version",
        ),
        (
            r#"printf '{"t":"what"}' >&0; sleep 5"#,
            ErrorClass::KrishivPythonWorkerExited,
            "unexpected greeting",
        ),
        (
            r#"printf '{"t":"error","kind":"python","message":"No module named pyarrow"}' >&0; sleep 5"#,
            ErrorClass::PythonException,
            "No module named pyarrow",
        ),
    ];
    for (script, class, text) in cases {
        let pool = fake_worker_pool(&py, script);
        let err = start_error(pool.clone());
        assert_eq!(err.error_class(), class, "{script}: {err}");
        assert!(err.to_string().contains(text), "{script}: {err}");
        assert_eq!(
            pool.live_workers(),
            0,
            "{script}: the failed start is forgotten"
        );
    }
}

#[test]
fn the_total_cap_stops_another_sessions_idle_worker() {
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.max_total = 1;
    let pool = Pool::new(config).expect("pool");
    assert!(format!("{pool:?}").contains("Pool"));
    let payload = command(
        &py,
        "import os\ndef f(x):\n    return os.getpid()",
        "LongType()",
    );
    let pid_of = |session: &str| {
        let udfs = SessionUdfs::new(pool.clone(), owner(session));
        assert!(format!("{udfs:?}").contains(session));
        let out = call(
            &udfs,
            payload.clone(),
            ARROW_BATCHED,
            ArrowType::Int64,
            longs(&[Some(1)]),
        )
        .expect("runs");
        as_longs(&out)[0].expect("pid")
    };
    let a = pid_of("a");
    let b = pid_of("b");
    assert_ne!(a, b);
    assert_eq!(pool.live_workers(), 1, "a's idle worker made room for b's");
}

#[test]
fn a_call_cancelled_while_its_worker_starts_ends_the_start() {
    // Review 2026-11-13 n3: the start used to run on until the greeting timeout (120 s).
    let Some(py) = python() else { return };
    let pool = fake_worker_pool(&py, "exec sleep 60");
    let udfs = SessionUdfs::new(pool.clone(), owner("s1"));
    udfs.register(
        "never",
        1,
        PythonPayload {
            command: Arc::from(&b"unused"[..]),
            eval_type: 101,
            python_ver: "3.12".to_owned(),
        },
    );
    let prepared = udfs
        .prepare(&spec(
            "never",
            PythonEvalType::ArrowBatched,
            ArrowType::Int64,
        ))
        .expect("prepare");
    let mut fut = prepared.call(longs(&[Some(1)]));
    let started = Instant::now();
    while pool.live_workers() == 0 && started.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        futures::FutureExt::now_or_never(&mut fut).is_none(),
        "still starting"
    );
    assert_eq!(pool.live_workers(), 1, "the worker is starting");
    let cancelled = Instant::now();
    drop(fut);
    while pool.live_workers() > 0 && cancelled.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        pool.live_workers(),
        0,
        "the starting worker was not stopped"
    );
    assert!(
        cancelled.elapsed() < Duration::from_secs(2),
        "{:?}",
        cancelled.elapsed()
    );
}

/// The pool's call threads in this process (`krishiv-udf-call`, cut to 15 bytes by the kernel).
fn call_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("tasks")
        .filter_map(Result::ok)
        .filter(|t| {
            std::fs::read_to_string(t.path().join("comm"))
                .is_ok_and(|name| name.starts_with("krishiv-udf-cal"))
        })
        .count()
}

/// The call threads added since `before`, once every thread has its own name: a worker starts on a call thread,
/// which spawns the worker's `krishiv-udf-stderr` reader, and on Linux a new thread carries its parent's name until
/// it sets its own, so for a moment `/proc` counts the reader as a call thread (merge-queue flake 2026-11-16, run
/// 43858094536). Waits up to 2 s for at most `limit`; N calls on N threads would still fail.
fn added_call_threads(before: usize, limit: usize) -> usize {
    let deadline = Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let added = call_threads().saturating_sub(before);
        if added <= limit || Instant::now() >= deadline {
            return added;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn concurrent_calls_share_the_pools_threads() {
    // Review 2026-11-13 n2: N concurrent calls used to make N threads.
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.max_total = 2;
    config.max_per_session = 2;
    let pool = Pool::new(config).expect("pool");
    let udfs = SessionUdfs::new(pool.clone(), owner("s1"));
    udfs.register(
        "nap",
        1,
        command(
            &py,
            "import time\ndef f(x):\n    time.sleep(0.05)\n    return x",
            "LongType()",
        ),
    );
    let prepared = udfs
        .prepare(&spec("nap", PythonEvalType::ArrowBatched, ArrowType::Int64))
        .expect("prepare");
    // The count is of the whole process: exact only when this test has it to itself (nextest's default);
    // under `cargo test` other tests' pools start and stop call threads meanwhile, so only the results count.
    let alone = std::env::var("NEXTEST_EXECUTION_MODE").is_ok_and(|m| m == "process-per-test");
    let before = call_threads();
    let calls: Vec<_> = (0..12).map(|i| prepared.call(longs(&[Some(i)]))).collect();
    let added = added_call_threads(before, 2);
    assert!(!alone || added <= 2, "{added} call threads");
    let results = block_on(futures::future::join_all(calls));
    for (i, r) in results.into_iter().enumerate() {
        assert_eq!(
            as_longs(&r.expect("runs")),
            [Some(i64::try_from(i).expect("small"))]
        );
    }
    let added = added_call_threads(before, 2);
    assert!(!alone || added <= 2, "{added} call threads");
    assert!(pool.live_workers() <= 2);
}

#[test]
fn the_session_cap_stops_an_idle_worker_of_another_interpreter() {
    // Review 2026-11-13 m2, tested with two interpreters: an interpreter is one entry of the list, chosen by
    // the client's Python version, so the second is a wrapper of the first whose inspection reports 3.99.
    let Some(py) = python() else { return };
    let dir = tempfile::tempdir().expect("dir");
    let wrapper = dir.path().join("python3.99");
    let real = py.display().to_string();
    std::fs::write(
        &wrapper,
        format!(
            "#!/bin/sh\ncase \"$*\" in\n*base_exec_prefix*) '{real}' \"$@\" | '{real}' -I -c \
             'import json, sys; r = json.load(sys.stdin); r[\"version\"] = [3, 99, 0]; print(json.dumps(r))' ;;\n\
             *) exec '{real}' \"$@\" ;;\nesac\n"
        ),
    )
    .expect("wrapper");
    std::fs::set_permissions(
        &wrapper,
        std::os::unix::fs::PermissionsExt::from_mode(0o755),
    )
    .expect("chmod");
    let mut config = config(&py);
    config.executables = vec![real, wrapper.display().to_string()];
    config.sandbox = false;
    config.max_per_session = 1;
    config.idle_timeout = Duration::from_secs(600);
    let pool = Pool::new(config).expect("pool");
    assert_eq!(pool.interpreters().len(), 2, "{:?}", pool.refusal());
    assert_eq!(pool.interpreters()[1].minor_version(), "3.99");
    let udfs = SessionUdfs::new(pool.clone(), owner("s1"));
    let payload = command(
        &py,
        "import os\ndef f(x):\n    return os.getpid()",
        "LongType()",
    );
    let first = call(
        &udfs,
        payload.clone(),
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect("runs");
    let first = as_longs(&first)[0].expect("pid");
    assert_eq!(pool.live_workers(), 1);
    // The second interpreter's worker: the session is at its cap with an idle worker of the first, which is
    // stopped rather than waited for (the idle timeout is 10 minutes). The worker is the real interpreter, so
    // it reports the version mismatch.
    let started = Instant::now();
    let err = call(
        &udfs,
        PythonPayload {
            python_ver: "3.99".to_owned(),
            ..payload
        },
        ARROW_BATCHED,
        ArrowType::Int64,
        longs(&[Some(1)]),
    )
    .expect_err("mismatch");
    assert!(err.to_string().contains("PYTHON_VERSION_MISMATCH"), "{err}");
    assert!(
        started.elapsed() < Duration::from_secs(60),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(pool.live_workers(), 1, "never more than the cap");
    assert!(
        !PathBuf::from(format!("/proc/{first}")).exists(),
        "the first interpreter's idle worker was stopped"
    );
}

fn tenant_owner(tenant: &str, session: &str) -> Owner {
    Owner {
        tenant: Some(tenant.to_owned()),
        capped: true,
        ..owner(session)
    }
}

/// `pid|RLIMIT_AS` as the worker sees them.
fn pid_and_limit(udfs: &SessionUdfs, py: &std::path::Path) -> String {
    let body = "import os, resource\n\
        def f(x):\n    \
        return f'{os.getpid()}|{resource.getrlimit(resource.RLIMIT_AS)[0]}'";
    let out = call(
        udfs,
        command(py, body, "StringType()"),
        ARROW_BATCHED,
        ArrowType::Utf8View,
        longs(&[Some(1)]),
    )
    .expect("runs");
    out.as_any()
        .downcast_ref::<StringViewArray>()
        .expect("string")
        .value(0)
        .to_owned()
}

#[test]
fn two_tenants_never_share_a_worker_and_each_has_its_memory_limit() {
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.memory_limit = Some(4 << 30);
    config.tenant_memory_limit = Some(2 << 30);
    let pool = Pool::new(config).expect("pool");
    let acme = SessionUdfs::new(pool.clone(), tenant_owner("acme", "s1"));
    let acme2 = SessionUdfs::new(pool.clone(), tenant_owner("acme", "s2"));
    let globex = SessionUdfs::new(pool.clone(), tenant_owner("globex", "s1"));
    let plain = SessionUdfs::new(pool, owner("s3"));
    let results: Vec<String> = [&acme, &acme2, &globex, &plain]
        .iter()
        .map(|u| pid_and_limit(u, &py))
        .collect();
    let pids: std::collections::HashSet<&str> = results
        .iter()
        .map(|r| r.split('|').next().expect("pid"))
        .collect();
    assert_eq!(pids.len(), 4, "four owners, four workers: {results:?}");
    let limit = |r: &String| r.split('|').nth(1).expect("limit").to_owned();
    assert_eq!(limit(&results[0]), (2_u64 << 30).to_string());
    assert_eq!(limit(&results[2]), (2_u64 << 30).to_string());
    assert_eq!(limit(&results[3]), (4_u64 << 30).to_string(), "no tenant");
    // The tenant's worker is reused by its own session.
    assert_eq!(pid_and_limit(&acme, &py), results[0]);
}

#[test]
fn a_tenant_at_its_worker_cap_replaces_its_idle_worker_and_never_another_tenants() {
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.max_per_tenant = 1;
    let pool = Pool::new(config).expect("pool");
    let a1 = SessionUdfs::new(pool.clone(), tenant_owner("acme", "s1"));
    let a2 = SessionUdfs::new(pool.clone(), tenant_owner("acme", "s2"));
    let g1 = SessionUdfs::new(pool.clone(), tenant_owner("globex", "s1"));
    let first = pid_and_limit(&a1, &py);
    let other = pid_and_limit(&g1, &py);
    assert_eq!(pool.live_workers(), 2);
    // acme's second session finds its tenant at the cap: acme's idle worker goes, globex's stays.
    let second = pid_and_limit(&a2, &py);
    assert_ne!(first, second);
    assert_eq!(pool.live_workers(), 2, "one per tenant");
    assert_eq!(pid_and_limit(&g1, &py), other, "globex's worker was kept");
}

#[test]
fn a_principal_without_a_tenant_is_capped_like_a_tenant() {
    let Some(py) = python() else { return };
    let mut config = config(&py);
    config.max_per_tenant = 1;
    let pool = Pool::new(config).expect("pool");
    let principal = |name: &str, session: &str| Owner {
        principal: name.to_owned(),
        session: session.to_owned(),
        tenant: None,
        capped: true,
    };
    let m1 = SessionUdfs::new(pool.clone(), principal("mallory", "s1"));
    let m2 = SessionUdfs::new(pool.clone(), principal("mallory", "s2"));
    let a1 = SessionUdfs::new(pool.clone(), principal("alice", "s1"));
    let first = pid_and_limit(&m1, &py);
    let other = pid_and_limit(&a1, &py);
    assert_eq!(pool.live_workers(), 2);
    // mallory's second session finds her at the cap: her idle worker goes, alice's stays.
    assert_ne!(first, pid_and_limit(&m2, &py));
    assert_eq!(pool.live_workers(), 2, "one per principal");
    assert_eq!(pid_and_limit(&a1, &py), other, "alice's worker was kept");
}
