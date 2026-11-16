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
- `src/worker.rs` one worker: handshake, `ensure_loaded` (a `udf` message), `run_batch` (a `batch` message), a
  relation task's messages (`start_task`, `task_event`, `send_frame`, `abandon`), the
  error mapping (`PYTHON_EXCEPTION`, `KRISHIV_PYTHON_WORKER_EXITED`, `KRISHIV_UDF_SANDBOX_UNAVAILABLE` for a trampoline
  that exits 125 before the worker greets the server; afterwards, or with the sandbox off, 125 is an ordinary exit) and the validating result readers `decode_result` (scalar) and `decode_batch` (a relation task's result, given the declared schema)
- `src/wire.rs` control messages (JSON, `SCM_RIGHTS`), memfds (`sealed_memfd`, `read_untrusted`)
- `src/runtime.rs` `SessionUdfs`: a session's payload store keyed by `(name, version)` as a `UdfRuntime`; each call
  runs on a pool call thread and is cancelled (worker killed, even one still starting) when its future is dropped
  (`with_worker`, shared with the relation tasks)
- `src/relation.rs` the relation kinds (PR-26c-2: map 205/207, grouped map 201/209/215/216, cogrouped map 206/210,
  grouped aggregate 202/217/252/254): `prepare_relation` checks the payload's eval type, `open` starts one task per
  driver on a call thread that holds one worker for the task (wire `task`: frames on the worker's `want`, results
  as they come, `done`); `send`/`next` are two bounded channels (`send` resolves once the thread took the frame);
  dropping the task cancels it; a thread that ends without `done` or a failure (a panic, a job dropped unrun) is
  `INTERNAL_ERROR` (`Ending`), never a clean end; a `want` after `end` is a protocol break. `PoolSlots`: the
  session's relation slots, the call sites' capacity hint (design P3-T085 §3.3a)
- `src/validate.rs` Spark 4.1+'s result-schema check of the map kinds under
  `spark.sql.execution.arrow.pyspark.validateSchema.enabled` (`ARROW_TYPE_MISMATCH`, compatible nullability)
- `src/calls.rs` `CallThreads`: the bounded call threads (at most `maxTotal` threads, at most `maxPerSession`
  running calls per owner; the rest wait in a FIFO queue, not on a thread), and the relation task slots: at most
  `maxPerSession - 1` per owner (all of them when it is 1), taken all at once or waited for with a waker
  (`take_slots`, never queued), a task with its slot submitted ahead of the queue (`submit_task`); with
  `maxPerSession` 1 a scalar call beside an open task is `KRISHIV_PYTHON_WORKER_LIMIT`
- `src/interpreter.rs` the startup record of each listed interpreter (version, read paths, packages) and `choose`;
  the inspection runs in a process group of its own, its output read while it runs (capped, under the timeout)
- `src/package.rs` the embedded worker package and the `-c` bootstrap
- `worker/krishiv/_udf` a symbolic link to `python/krishiv/_udf` (one copy of the Python code; `cargo package`
  copies the files through it; one link rather than one per file, so ruff, which lints them under `python/`, does
  not lint them again without that config). The worker program is `python/krishiv/_udf/pool_worker.py`; `_host.py` switches the
  wrappers' names between the `krishiv` facade (in-process) and `pyspark` (here)
- `python/krishiv/_udf/relations.py` (through the link) the relation kinds' runners, Spark 4.2.0's per-eval-type
  loops of `worker.py` over the engine's frames (host-agnostic: the in-process runtime of PR-48 can reuse them)
- `tests/relations.rs` each relation kind through the pool with real workers (same interpreter rule as below)
- `tests/pool.rs` real workers: needs `KRISHIV_TEST_UDF_PYTHON` (Python >= 3.11 with `pyspark-client`, `pyarrow`,
  `pandas`); without it the tests pass with a note unless `KRISHIV_UDF_POOL_REQUIRED` is set (the CI job sets both)

## Extension points
- A new message: both sides (`pool_worker.py` and `worker.rs`) and the module doc of `pool_worker.py`. A change a
  worker of the previous build cannot read raises `WIRE_VERSION` (the worker is part of the same binary, so the
  version guards mixed packages, not stored data).
- A new file in the worker package: an entry in `package::FILES` (the file itself lives in `python/krishiv/_udf/`).
- Window aggregates, UDTF analyze and archives (PR-26c-3), Connect decode and registered functions (PR-26d) build on
  `SessionUdfs` and `Worker`.

## Invariants
- `#![forbid(unsafe_code)]`: the server never maps a memfd. Input is written and sealed; results are read with
  `pread` into one 64-byte-aligned Arrow buffer, capped (`max_result_bytes`), their framing checked by `check_frames` (verified
  flatbuffer headers, bodies and buffers in bounds, **no compression**: arrow-ipc 60 allocates a compressed
  buffer's declared length up front, so a 16 KiB stream could abort the server), then decoded by `arrow-ipc`'s
  validating `StreamDecoder`, whose columns are slices of that buffer (no second copy), and checked for one column of the declared type, one batch and the input's row count (a relation task's result: one batch with the declared schema's types, given the declared schema). Worker
  output is untrusted; fuzz target `fuzz/fuzz_targets/udf_result_ipc.rs` runs `decode_result` and `decode_batch`
  nightly.
- **No relation task waits behind its owner's own tasks** (design P3-T085 §3.3a, review 2026-11-16 finding 1): a
  task holds a worker while idle in `want`, so a queued one could hang its driver. The operator takes a slot per
  lane at once (`PoolSlots`) or waits holding none; a scalar call of the owner always has a worker left (the
  reserve), or with `maxPerSession` 1 fails fast. Tests: `calls.rs`, `relation.rs` unit tests, `tests/relations.rs`
  `*_beside_an_open_task_*`.
- **Memory outside the operator's accounting** (as on the scalar path): a frame's memfd copy while the worker reads
  it, and up to two decoded result batches per task (one in the channel, one the thread waits to hand over).
- **Divergence: map-kind results unlike the declared schema.** With
  `spark.sql.execution.arrow.pyspark.validateSchema.enabled` off (Spark's default; 3.5 and 4.0 have no such
  setting) Spark does not check a `mapInPandas`/`mapInArrow` result's types; the pool refuses it with
  `PYTHON_EXCEPTION` (and a null in a non-nullable declared column). Recorded in `docs/concepts/known-differences.md`.
  No Spark golden backs either form yet (the build host has no Spark).
- Every worker is started by the one `krishiv-udf-spawner` thread (the parent-death signal follows the starting
  thread), never in a process group of its own. Every signal goes through the worker's **pidfd** (opened before
  the child can be reaped), never a pid, so a late kill (a cancel racing the worker's own exit, or `Drop` after
  `exited` reaped it) reaches no other process. pidfds need Linux 5.3, so that is the pool's floor even with
  `krishiv.udf.sandbox=off` (the sandbox itself needs 5.13). A call's cancel kills only while the call holds the worker
  (`CallState.running`, cleared under its lock before the worker is given back). A starting worker's pidfd is
  set there before the greeting is awaited (`CallWatch::starting`), so a cancel kills it and the start ends at once.
- A worker that breaks the protocol (user code writing to fd 0, a result changed after it was sent) is
  `KRISHIV_PYTHON_WORKER_EXITED`; `INTERNAL_ERROR` is only for the server's own failures (memfd, socket calls).
- The bootstrap closes every descriptor above 2 before anything else (`close_range` through `ctypes`, or
  `os.closerange` up to the server's `RLIMIT_NOFILE`), and the interpreter runs with `-I` and an environment of
  `HOME`, `TMPDIR` and `LC_ALL` only. With the sandbox off the bootstrap applies the rlimits itself.
- With the sandbox on and `allow_fork` off, the bootstrap's next statements stack a second seccomp filter that makes
  `execve`/`execveat` `EPERM` (`package::exec_deny_filter`), before `sys.path` changes: the parent-death signal is
  per thread, so an `exec` from any thread but the first (UDF calls run on such threads) would otherwise outlive
  the server. Test `udf_cannot_outlive_the_server_by_exec_from_a_thread`.
- One worker serves one `Owner` (principal and session) and one interpreter; reuse only within that key.
- Blocking work (spawn, socket I/O, waits) runs on the pool's threads, never on a compute worker.
- A relation task's server side writes to the worker only after the worker's `want`, so neither side blocks on a
  full socket while the other writes; the worker is given back before the task's results end.
- No `unwrap`/`expect`/`panic` in library code; errors carry an error class and SQLSTATE.

## Not here yet (design note §9 records why)
Per-query RSS accounting and `system.queries` visibility, the timing trailer, traceback shaping by the three
`spark.sql.execution.pyspark.udf.*Traceback*` flags, the memfd buffer ring, artifacts and the in-process opt-in:
PR-26c/26d, where the pool meets the session and the query. Also open (review 2026-11-13): the default-Docker
CI run of the sandbox suite.

## ADRs (in `../krishiv-internal/adr/`)
0022 (UDF isolation; process pool is the server default), 0046 (N1: client code is admin-only in server mode until
the sandbox is in use), 0029 (`unsafe` placement), 0028 (dependencies)

## Tests
`cargo nextest run -p krishiv-udf-pool --no-tests=pass` (unit tests need nothing; set `KRISHIV_TEST_UDF_PYTHON` for
`tests/pool.rs`) · `cargo clippy -p krishiv-udf-pool --all-targets -- -D warnings`.
