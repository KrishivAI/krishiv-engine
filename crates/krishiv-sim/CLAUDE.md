# krishiv-sim

**Layer:** L7 test infrastructure (`publish = false`; product crates must not depend on it; `cargo xtask layering`).  
**Status:** P2-T176 unit 2: a simulated cluster on turmoil 0.7.2 (ADR 0029).

## Purpose
Deterministic simulation testing (DST): reproducible cluster runs from a seed, with fault injection (a killed map task
now; host crash/restart, partitions, clock skew and object-store errors in P5-T024).

## Key files
Paths are relative to this crate.
- `src/lib.rs` `simulate` (the turmoil world: worker hosts + a coordinator client), `ClusterConfig`, `SimConfig`
- `src/cluster.rs` `Coordinator`: a `krishiv_exec::Scheduler` that ships fragments and task descriptors to workers, retries a failed task as its next attempt, runs the root stage, records a trace, releases each job's map outputs
- `src/worker.rs` `serve` (one worker host: own executor, pool, tables via `HostSetup`, one exchange and one
  `BroadcastTables` per job, so a broadcast join's table is built once per host and job) and `HostSetup`
- `src/exchange.rs` `HostExchange` (writes to the host's `LocalShuffle`, reads map partition `m` from worker `m mod W`, remotely over TCP), `Fault`
- `src/wire.rs` the length-prefixed request/reply messages; `src/task.rs` the `krishiv.task.v1alpha` task descriptor
  (with the file splits the coordinator dealt, `Coordinator::deal_scans`; relation batches are refused)
- `tests/tpch_sim.rs` TPC-H on two workers with dealt scans equals one worker; same seed, same trace; a killed map
  task reruns; a two-worker write names files as Spark does and commits once; a file changed after listing fails

## Extension points
- New scenario: a test calling `simulate` with its own `HostSetup` (which must share nothing between hosts but the object store).
- New fault: a field of `ClusterConfig` honoured by `HostExchange` or the worker.

## Invariants
- Every run is reproducible from its seed: hosts use `TokioExecutor::deterministic` on turmoil's current-thread runtimes and memory pools on `TokioClock`; no wall clock, threads or ambient randomness.
- Nothing live crosses a host: plans as codec bytes, tasks as descriptors, rows as IPC frames.
- `#![forbid(unsafe_code)]`.

## ADRs (in `../krishiv-internal/adr/`)
0029, 0013, 0040

## Tests
`cargo nextest run -p krishiv-sim --no-tests=pass` (SF 0.1: `KRISHIV_SIM_TPCH_DIR=<dir from tests/scripts/gen_tpch_parquet.py 0.1>`) · `cargo clippy -p krishiv-sim --all-targets -- -D warnings`
