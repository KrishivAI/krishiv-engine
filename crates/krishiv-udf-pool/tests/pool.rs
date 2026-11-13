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
