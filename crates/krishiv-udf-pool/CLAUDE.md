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
  `Owner`; the spawner thread, the caps, idle stop, `release`, and `Lease` (a worker handed to one call);
  `Pool::snapshot` (`PoolSnapshot`: live and idle workers, the three caps, `OwnerWorkers` per owner with a live
  worker; one short lock hold) and `Pool::monitor` (`PoolMonitor`, a weak view for the console's Resources page:
  `None` once the pool is gone, so a page never keeps workers running)
- `src/worker.rs` one worker: handshake, `ensure_loaded` (a `udf` message), `run_batch` (a `batch` message), a
  relation task's messages (`start_task`, `task_event`, `send_frame`, `abandon`), the
  error mapping (`PYTHON_EXCEPTION`, `KRISHIV_PYTHON_WORKER_EXITED`, `KRISHIV_UDF_SANDBOX_UNAVAILABLE` for a trampoline
  that exits 125 before the worker greets the server; afterwards, or with the sandbox off, 125 is an ordinary exit) and the validating result readers `decode_result` (scalar) and `decode_batch` (a relation task's result, given the declared schema)
- `src/wire.rs` control messages (JSON, `SCM_RIGHTS`), memfds (`sealed_memfd`, `read_untrusted`)
- `src/runtime.rs` `SessionUdfs`: a session's payload store keyed by `(name, version)` as a `UdfRuntime`; its runner conf read at each `prepare` and `prepare_relation` from a `ConfSource` (`with_conf_source`, the session's live conf) or set with `set_runner_conf`; a call of another kind than the payload's eval type (the store finds payloads by name and version only) is `KRISHIV_PYTHON_UDF_EVAL_TYPE_MISMATCH` before any worker starts (`check_eval_type`; a scalar call takes any scalar payload, `runs_as_scalar`: 100 or a `PythonEvalType::is_scalar` code, pinned to `worker.py`'s wrappers by a test, and the worker runs the payload's own eval type on purpose, since Connect registers a plain `@udf` as 100 and plans it as 101; a relation call needs its exact code, but for a grouped aggregate over a window, 202 as 203 and 252 as 253, `relation.rs` `runs_as`); each call
  runs on a pool call thread and is cancelled (worker killed, even one still starting) when its future is dropped
  (`with_worker`, shared with the relation tasks)
- `src/relation.rs` the relation kinds (PR-26c-2: map 205/207, grouped map 201/209/215/216, cogrouped map 206/210,
  grouped aggregate 202/217/252/254; PR-26c-3: window aggregate 203/253, run over a 202/252 payload as Spark runs
  one over a window, and scalar iterator 204/251; part B: the UDTFs 300/301/302, whose task message carries the
  call's `udtf` part, `udtf_message`: argument offsets and names, partition child indexes, `TABLE` offsets, the
  declared struct as Spark's JSON and `analyze`'s pickled result in hex; a UDTF payload's command is the
  cloudpickled class alone): `prepare_relation` checks the payload's eval type (the task
  message carries the kind it runs as), `open` starts one task per
  driver on a call thread that holds one worker for the task (wire `task`: frames on the worker's `want`, results
  as they come, `done`); `send`/`next` are two bounded channels (`send` resolves once the thread took the frame);
  dropping the task cancels it; a thread that ends without `done` or a failure (a panic, a job dropped unrun) is
  `INTERNAL_ERROR` (`Ending`), never a clean end; a `want` after `end` is a protocol break. `PoolSlots`: the
  session's relation slots, the call sites' capacity hint (design P3-T085 §3.3a)
- `src/udtf.rs` (PR-26c-3 part B) `UdfRuntime::analyze_udtf` on the pool: a 300/301 payload's static `analyze`
  run on a worker of the owner on a call thread (wire `analyze`/`analyzed`), its answer read as a `UdtfAnalysis`
  (untrusted: an unreadable one fails the analysis); a Python failure is
  `TABLE_VALUED_FUNCTION_FAILED_TO_ANALYZE_IN_PYTHON` with Spark's message (the text after the traceback's last
  `PySparkValueError:`, `worker.rs` `analyze_message`); 302 has no `analyze` (`KRISHIV_PYTHON_UDF_EVAL_TYPE_MISMATCH`).
  The call's constants (a one-row Arrow batch, `analyze_input`) and the result (JSON) travel in memfds, never in
  a control message, the result capped at `MAX_ANALYZE_RESULT` on both sides (the analyze class); the worker
  converts each constant as Spark does (internal value, then `fromInternal`). A UDTF task's call data goes in a
  memfd too (`udtf_len`)
- Spark 4.1+'s result-schema check of the map kinds under
  `spark.sql.execution.arrow.pyspark.validateSchema.enabled` lives in `krishiv-exec`
  `src/python_relation/schema_check.rs` (`ARROW_TYPE_MISMATCH`, compatible nullability), shared by both runtimes
- `src/calls.rs` `CallThreads`: the bounded call threads (at most `maxTotal` threads, at most `maxPerSession`
  running calls per owner; the rest wait in a FIFO queue, not on a thread), and the relation task slots: at most
  `min(maxPerSession, maxTotal) - 1` per owner (all of them when that is 1) and `maxTotal - 1` in all (the threads
  are shared across sessions), taken all at once or waited for in arrival order (a `TaskTicket` per waiter,
  dropped when it stops waiting; no grant passes a waiter ahead of the same owner, or one waiting for room in
  all; with `maxTotal` 1 no task is granted while another owner's scalar call that arrived before it waits for the
  thread: tickets and queued calls share one arrival sequence, so neither kind starves the other).
  `try_take_now` is the no-wait form (no ticket, no waker)
  (`take_slots`, never queued), a task with its slot submitted ahead of the queue (`submit_task`); with
  `maxPerSession` 1 a scalar call beside an open task is `KRISHIV_PYTHON_WORKER_LIMIT`
- `src/artifacts.rs` (PR-26c-3) a session's pyfiles and archives (`SessionUdfs::set_artifacts`, `PythonArchive`
  with its `#fragment` alias): paths checked when set (absolute, lexically canonical, a regular file opened by
  `openat2` with `RESOLVE_NO_SYMLINKS`: no symbolic link anywhere, `INVALID_ARTIFACT_PATH`; an alias plain, not
  empty, hidden or with a `/`, `INVALID_ARTIFACT_PATH`; one name once across pyfiles and archives,
  `ARTIFACT_ALREADY_EXISTS`; the device and inode recorded), prepared once per set by the first worker start into
  a directory of the pool (each file opened again the same way and refused unless it is the file checked; `py/`,
  the files directory, holds the pyfile copies and each archive unpacked into `py/<alias or file name>/`: the
  archive copied to `in/` and that copy unpacked by `unpack.py` with the worker interpreter, sandboxed when the
  sandbox is on, killed through its pidfd on a cancel; `tarfile`'s `data` filter, a hard or symbolic link counted
  at its target's size), within `PoolConfig::artifact_limit` (`KRISHIV_PYTHON_ARTIFACT_REFUSED`; the disk peaks at
  twice the limit while the last archive's copy and its unpacking coexist); only a failure about the artifacts is
  kept for the set (`Failure::lasting`, and the `MAX_TIMEOUTS`th timeout), a cancel, timeout or server I/O failure
  is retried by the next call, and the set's lock is not held while it is made; the directory is the worker's only
  extra read path; `py/` and each zip or egg go on `sys.path` (index 1, Spark's `add_path`) after the worker's own
  imports, an unpacked archive does not (as Spark); each entry of `py/` is linked into the worker's own working
  directory (`pool.rs` `link_files`); the pool key carries the set's id (a worker never serves another set)
- `src/inspection.rs` `Inspection`: the listed interpreters inspected on a thread of their own, started by
  `Pool::new` so the server's listeners open before Python has imported its libraries; `checkout` waits for it
  (cancellably), `Pool::interpreters` and `Pool::refusal` wait for it; a thread that unwinds leaves an
  `INTERNAL_ERROR` refusal, never a waiter waiting for ever; the pool's drop (`Inner`) stops it and joins its
  thread, so no inspection process outlives the pool (its timeout lives on that thread)
- `src/interpreter.rs` the startup record of each listed interpreter (version, read paths, packages) and `choose`;
  the inspection runs in a process group of its own, its output read while it runs (capped, under the timeout)
- `src/package.rs` the embedded worker package and the `-c` bootstrap (`bootstrap_for`: the worker with its
  artifact paths, or `unpack.py` for one archive)
- `worker/krishiv/_udf` a symbolic link to `python/krishiv/_udf` (one copy of the Python code; `cargo package`
  copies the files through it; one link rather than one per file, so ruff, which lints them under `python/`, does
  not lint them again without that config). The worker program is `python/krishiv/_udf/pool_worker.py`; `_host.py` switches the
  wrappers' names between the `krishiv` facade (in-process) and `pyspark` (here)
- `python/krishiv/_udf/relations.py` (through the link) the relation kinds' runners, Spark 4.2.0's per-eval-type
  loops of `worker.py` over the engine's frames (host-agnostic: the in-process runtime of PR-48 can reuse them)
- `tests/relations.rs` each relation kind through the pool with real workers (same interpreter rule as below)
- `tests/udtf.rs` the UDTFs through the pool: `eval`/`terminate` per kind, a partitioned `TABLE` argument, the
  `UDTF_*` errors, `analyze` and its result reaching `__init__`, large and oversize `analyze` data, constants of
  each type; `python/krishiv/_udf/udtf.py` is the Python side (a port of pyspark 4.2.0's `read_udtf` and
  `analyze_udtf.py`)
- `tests/window_iter_goldens.rs` PySpark 4.2.0's goldens for window aggregates and scalar iterators
  (`python/tests/pyspark_udf/window_iter/goldens.json`) replayed as SQL on a `krishiv_session::Session` over the
  pool, and grouped-aggregate UDFs in SQL `GROUP BY` with the shapes Spark refuses
- `tests/udtf_goldens.rs` (PR-26c-4) PySpark 4.2.0's UDTF goldens (`python/tests/pyspark_udf/udtf/goldens.json`)
  replayed as SQL on a `krishiv_session::Session` over the pool: `export_cases.py` (run by
  `KRISHIV_TEST_PYSPARK_PYTHON`, a full PySpark 4.2.0) records each case's UDTFs, cloudpickled, and its SQL;
  `KNOWN_GAPS` and `INCOMPATIBLE` (recorded in known-differences) must still differ. A 300 UDTF's rows are
  converted as Spark's JVM converts them (`udtf.py` `_from_java`); the JVM's own errors come back as the worker
  error kind `spark` (`worker.rs` `spark_error`: `STRUCT_ARRAY_LENGTH_MISMATCH`,
  `NUMERIC_VALUE_OUT_OF_RANGE.WITHOUT_SUGGESTION`)
- `tests/pool.rs` real workers: needs `KRISHIV_TEST_UDF_PYTHON` (Python >= 3.11 with `pyspark-client`, `pyarrow`,
  `pandas`); without it the tests pass with a note unless `KRISHIV_UDF_POOL_REQUIRED` is set (the CI job sets both)

## Extension points
- A new message: both sides (`pool_worker.py` and `worker.rs`) and the module doc of `pool_worker.py`. A change a
  worker of the previous build cannot read raises `WIRE_VERSION` (the worker is part of the same binary, so the
  version guards mixed packages, not stored data).
- A new file in the worker package: an entry in `package::FILES` (the file itself lives in `python/krishiv/_udf/`).
- UDTFs and UDTF analyze (PR-26c-3 part B), Connect decode and registered functions (PR-26d) build on
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
  No Spark golden backs either form yet (the build host has no Spark): a spark-semantics golden is owed for
  `mapInArrow` with a wrong type at 4.1 and 4.2 and for `mapInPandas` with a non-nullable declared column (which
  the v4.2.0 source says Spark refuses under the setting: its pandas serializer makes every field nullable). The
  setting is not gated by Spark version (the pool does not know the client's).
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
  `HOME`, `TMPDIR`, `LC_ALL` and the server's `TZ` (when set; `PoolConfig::time_zone`: a PySpark worker inherits
  the executor's zone) only. With the sandbox off the bootstrap applies the rlimits itself.
- With the sandbox on and `allow_fork` off, the bootstrap's next statements stack a second seccomp filter that makes
  `execve`/`execveat` `EPERM` (`package::exec_deny_filter`), before `sys.path` changes: the parent-death signal is
  per thread, so an `exec` from any thread but the first (UDF calls run on such threads) would otherwise outlive
  the server. Test `udf_cannot_outlive_the_server_by_exec_from_a_thread`.
- **Tenants (P3-T072).** `Owner.tenant` (the principal's tenant, from Connect's `SessionOwner`; `None` in single-tenant
  servers) is part of the key, so a worker never serves two tenants (nor two sessions). The cap group is `Owner::group()`: the tenant, or the principal when it has no tenant and `Owner.capped` (authentication on; with it off every session is the one local principal and is in no group, so only `maxTotal`/`maxPerSession` bound it) (so one principal of a
  single-tenant server cannot take the pool either; threat model T-U4/T-D10). A group's live workers
  (`State::per_tenant`) and running calls (`calls.rs` `running_by_group`/`slots_by_group`, O(1) counters,
  `with_tenant_cap`) are capped by
  `krishiv.python.worker.maxPerTenant`, and its relation slots by one less (`slot_limit_for`), so one tenant cannot
  take the shared call threads and a tenant's tasks never wait behind its own. At the cap a tenant's longest-idle
  worker of another key is stopped, never another tenant's. `tenant_memory_limit` lowers `RLIMIT_AS` of tenant
  workers. A slot waiter of the same group holds back a later request of it (no starvation by the group's other
  sessions). An archive unpack runs under the general `memoryLimit`, not `tenantMemoryLimit`. Tests: `calls.rs` tenant tests, `tests/pool.rs` `*tenant*`.
- One worker serves one `Owner` (principal and session), one interpreter and one artifact set; reuse only within
  that key. Artifacts are reachable from the worker's working directory through links into a read-only directory
  (Spark's worker runs in a writable files directory): a recorded difference.
- Blocking work (spawn, socket I/O, waits) runs on the pool's threads, never on a compute worker.
- A relation task's server side writes to the worker only after the worker's `want`, so neither side blocks on a
  full socket while the other writes; the worker is given back before the task's results end.
- No `unwrap`/`expect`/`panic` in library code; errors carry an error class and SQLSTATE.

## Not here yet (design note §9 records why)
Per-query RSS accounting and `system.queries` visibility, the timing trailer, traceback shaping by the three
`spark.sql.execution.pyspark.udf.*Traceback*` flags, the memfd buffer ring, a `krishiv.*` key for the artifact
limit (the hosting server sets `PoolConfig::artifact_limit`) and the in-process opt-in:
PR-26c/26d, where the pool meets the session and the query. Also open (review 2026-11-13): the default-Docker
CI run of the sandbox suite.

## ADRs (in `../krishiv-internal/adr/`)
0022 (UDF isolation; process pool is the server default), 0046 (N1: client code is admin-only in server mode until
the sandbox is in use), 0029 (`unsafe` placement), 0028 (dependencies)

## Tests
`cargo nextest run -p krishiv-udf-pool --no-tests=pass` (unit tests need nothing; set `KRISHIV_TEST_UDF_PYTHON` for
`tests/pool.rs`) · `cargo clippy -p krishiv-udf-pool --all-targets -- -D warnings`.
