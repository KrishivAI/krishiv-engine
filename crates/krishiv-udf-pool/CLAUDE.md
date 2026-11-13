# krishiv-udf-pool

**Layer:** L5 (`cargo xtask layering`): implements `krishiv-exec`'s `UdfRuntime`; depends on `krishiv-udf-sandbox`
(L1). Hosted by the server binary (`krishiv-cli` with `connect`, which also dispatches the trampoline subcommand);
never by the PyO3 module (`krishiv-python` keeps the in-process runtime, and the server must not need libpython).

## Purpose
The isolated Python UDF worker pool of server mode (P3-T014, PR-26b; design `design/P3-T014-connect-udfs-sandbox.md`
§3 in krishiv-internal). Workers are separate Python processes, one principal and one session each, started through
the sandbox trampoline, talking wire `krishiv.udf.wire.v1` over a `SOCK_SEQPACKET` socket with data in memfds.

## Key files
Paths are relative to this crate.
- `src/lib.rs` public surface and `run_trampoline_if_requested` (the binary's `__udf-worker` dispatch)
- `src/pool.rs` `Pool`, `PoolConfig` (from the `krishiv.python.worker.*` and `krishiv.udf.sandbox` keys), `Launcher`,
  `Owner`; the spawner thread, the caps, idle stop, `release`, and `Lease` (a worker handed to one call)
- `src/worker.rs` one worker: handshake, `ensure_loaded` (a `udf` message), `run_batch` (a `batch` message), the
  error mapping (`PYTHON_EXCEPTION`, `KRISHIV_PYTHON_WORKER_EXITED`, `KRISHIV_UDF_SANDBOX_UNAVAILABLE` for a trampoline
  that exits 125) and the validating result reader `decode_result`
- `src/wire.rs` control messages (JSON, `SCM_RIGHTS`), memfds (`sealed_memfd`, `read_untrusted`)
- `src/runtime.rs` `SessionUdfs`: a session's payload store keyed by `(name, version)` as a `UdfRuntime`; each call
  runs on a thread of its own and is cancelled (worker killed) when its future is dropped
- `src/interpreter.rs` the startup record of each listed interpreter (version, read paths, packages) and `choose`
- `src/package.rs` the embedded worker package and the `-c` bootstrap
- `worker/krishiv/_udf` a symbolic link to `python/krishiv/_udf` (one copy of the Python code; `cargo package`
  copies the files through it; one link rather than one per file, so ruff, which lints them under `python/`, does
  not lint them again without that config). The worker program is `python/krishiv/_udf/pool_worker.py`; `_host.py` switches the
  wrappers' names between the `krishiv` facade (in-process) and `pyspark` (here)
- `tests/pool.rs` real workers: needs `KRISHIV_TEST_UDF_PYTHON` (Python >= 3.11 with `pyspark-client`, `pyarrow`,
  `pandas`); without it the tests pass with a note unless `KRISHIV_UDF_POOL_REQUIRED` is set (the CI job sets both)

## Extension points
- A new message: both sides (`pool_worker.py` and `worker.rs`) and the module doc of `pool_worker.py`. A change a
  worker of the previous build cannot read raises `WIRE_VERSION` (the worker is part of the same binary, so the
  version guards mixed packages, not stored data).
- A new file in the worker package: an entry in `package::FILES` (the file itself lives in `python/krishiv/_udf/`).
- Relation kinds, UDTF analyze and archives (PR-26c), Connect decode and registered functions (PR-26d) build on
  `SessionUdfs` and `Worker`.

## Invariants
- `#![forbid(unsafe_code)]`: the server never maps a memfd. Input is written and sealed; results are read with
  `pread` into server memory, capped (`max_result_bytes`), their framing checked by `check_frames` (verified
  flatbuffer headers, bodies and buffers in bounds, **no compression**: arrow-ipc 60 allocates a compressed
  buffer's declared length up front, so a 16 KiB stream could abort the server), then decoded by `arrow-ipc`'s
  validating reader and checked for one column of the declared type, one batch and the input's row count. Worker
  output is untrusted; fuzz target `fuzz/fuzz_targets/udf_result_ipc.rs` runs `decode_result` nightly.
- Every worker is started by the one `krishiv-udf-spawner` thread (the parent-death signal follows the starting
  thread), never in a process group of its own. Every signal goes through the worker's **pidfd** (opened before
  the child can be reaped), never a pid, so a late kill (a cancel racing the worker's own exit, or `Drop` after
  `exited` reaped it) reaches no other process. A call's cancel kills only while the call holds the worker
  (`CallState.running`, cleared under its lock before the worker is given back).
- A worker that breaks the protocol (user code writing to fd 0, a result changed after it was sent) is
  `KRISHIV_PYTHON_WORKER_EXITED`; `INTERNAL_ERROR` is only for the server's own failures (memfd, socket calls).
- The bootstrap closes every descriptor above 2 before anything else (`close_range` through `ctypes`, or
  `os.closerange` up to the server's `RLIMIT_NOFILE`), and the interpreter runs with `-I` and an environment of
  `HOME`, `TMPDIR` and `LC_ALL` only. With the sandbox off the bootstrap applies the rlimits itself.
- One worker serves one `Owner` (principal and session) and one interpreter; reuse only within that key.
- Blocking work (spawn, socket I/O, waits) runs on the pool's threads, never on a compute worker.
- No `unwrap`/`expect`/`panic` in library code; errors carry an error class and SQLSTATE.

## Not here yet (design note §9 records why)
Per-query RSS accounting and `system.queries` visibility, the timing trailer, traceback shaping by the three
`spark.sql.execution.pyspark.udf.*Traceback*` flags, the memfd buffer ring, artifacts and the in-process opt-in:
PR-26c/26d, where the pool meets the session and the query. Also open (review 2026-11-13 nits): the result is
copied twice (`pread` into a `Vec`, then `StreamReader`; `StreamDecoder` over the buffer would not copy), one
thread per call, a cancelled call still waits for its worker's start, and the default-Docker CI run of the
sandbox suite.

## ADRs (in `../krishiv-internal/adr/`)
0022 (UDF isolation; process pool is the server default), 0046 (N1: client code is admin-only in server mode until
the sandbox is in use), 0029 (`unsafe` placement), 0028 (dependencies)

## Tests
`cargo nextest run -p krishiv-udf-pool --no-tests=pass` (unit tests need nothing; set `KRISHIV_TEST_UDF_PYTHON` for
`tests/pool.rs`) · `cargo clippy -p krishiv-udf-pool --all-targets -- -D warnings`.
