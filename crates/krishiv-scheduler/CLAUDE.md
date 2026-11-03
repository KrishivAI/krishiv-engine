# krishiv-scheduler

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).
**Real code since:** P2-T176 unit 1 (fragmenter, in-process stage runner); coordinator, AQE and fault tolerance
follow (P2-T176 unit 2, P5).

## Purpose
Distributed scheduler: plans cut into stages, stages run as tasks, later across workers.

## Key files
Paths are relative to this crate.
- `src/fragment.rs` `fragment()`: cuts a `PhysicalPlan` at every exchange into `Stage`s (dependency order, root
  last); `PhysNode::ExchangeSource` in each exchange's place. Shared-plan and recursive-CTE bodies are not cut and run
  whole; a stage holding one, a `range`, a gather read by its rows, or a build-emitting join over a broadcast is
  `single`. A runtime filter whose producer and scan are in different fragments is dropped on both sides until
  P2-T177 unit 2 carries values across stages. Refuses unpinned hash exchanges and a nondeterministic shared plan read
  by several stages.
- `src/stages.rs` `StageRunner`: a `krishiv_exec::Scheduler` running the stages bottom-up over a `LocalShuffle`
  (on disk in the worker's spill scope), each task its own instantiation from a `TaskSpec`; the root stage is one
  task on the caller's query context. `StageSplits` deals leaf scans to tasks (the session's split planning plugs in
  here).
- `tests/tpch_stages.rs` TPC-H SF0.01 on stages (4 partitions, whole and dealt scans; 8 partitions under a 16 MiB
  pool) equals one worker.

## Invariants
- Tasks get only bytes-shaped inputs: the fragment (plan data) and a `TaskSpec`; nothing live crosses a task but the
  worker's `Exchange`.
- Tested with deterministic simulation (turmoil, ADR 0029) for crashes and partitions from P2-T176 unit 2.
- Hard subsystem: Opus.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0004, 0006, 0013, 0020

## Test
`cargo nextest run -p krishiv-scheduler --no-tests=pass` · `cargo clippy -p krishiv-scheduler --all-targets -- -D warnings`
