# krishiv-exec

**Layer:** L4 (may depend only on crates in L4 or lower; `cargo xtask layering`).  
**Real code arrives:** P2; hash tables/spill: Opus.

## Purpose
Batch execution engine: operators, pipelines, morsel scheduling within a node.

## Layout (P2-T050)
`operator.rs` Source/Operator/Sink traits · `pipeline.rs` ExecNode + PipelineBuilder · `driver.rs` the pipeline
driver (yields, batch-boundary checks, NeedMemory, spill servicing, stall detector) · `runner.rs` QueryRunner /
Scheduler (coordinator, first error wins) · `channel.rs` bounded result stream · `executor.rs` Executor seam +
TokioExecutor · `context.rs` QueryContext/TaskContext/MemoryHandle · `morsel.rs` cursor + finalize countdown ·
`testing.rs` test operators (also used by benches) · `loom_tests.rs`.

## Invariants
- All memory through krishiv-memory reservations; every spill path tested with tiny-memory tests.
- Only the driver yields, awaits memory, spills or checks cancellation; `execute`/`sink` stay sync and bounded.
- At query end (success, error, cancel) the query pool is empty: tests assert it.
- Cancellation-safe, no blocking in async; metrics feed EXPLAIN ANALYZE.
- New operators follow skill `krishiv-operator`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0006, 0007, 0011, 0013

## Test
`cargo nextest run -p krishiv-exec --no-tests=pass` · `cargo clippy -p krishiv-exec --all-targets -- -D warnings` ·
loom models: `RUSTFLAGS="--cfg loom" cargo test -p krishiv-exec --lib --release` (use a separate target dir) ·
benches: `cargo bench -p krishiv-exec --bench pipeline`
