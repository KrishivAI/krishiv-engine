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
`operators/` (P2-T051) stateless operators: `scan.rs` MemoryScan + `ScanPushdown` (pushdowns decided once as an
immutable `ScanPushdowns`; implemented by scan factories: MemoryScan, FileScan) ·
`filter.rs` · `project.rs` · `limit.rs` · `union.rs` (per-branch `UnionAlign`; UNION ALL fan-in is T074, distinct UNION is the aggregate's) ·
`expr.rs` the `Predicate`/`ScalarExpr` evaluator interface the physical planner plugs in ·
`hash.rs` shared hash-table blocks (seeded key hash, key normalization + row encoding, `KeyTable`, spill
partition bits; for the join too) · `aggregate/` hash aggregate (P2-T053a): `mod.rs` spec + Spark result types,
`accumulator.rs` per-group states, `table.rs` one table, `sink.rs` breaker (spill runs, finalize merge,
`spill_global`), `partial.rs` flushable partial phase with adaptive bypass, `node.rs` plan nodes + DISTINCT/UNION ·
(P2-T117b) `file_scan.rs` FileScan/FileScanSource over any `krishiv_io::FileFormat` (Spark partitions as
morsels, footer read-ahead, `_metadata` struct column; filters stay `Inexact`) · `file_write.rs` FileWriteSink
(one writer task per driver behind a bounded channel; nothing visible after a failed or cancelled write) + FileWrite.
End-to-end tests over the mock format: `krishiv-testkit/tests/file_exec.rs`; the conformance suite through the
executor: `mock_exec` in `krishiv-testkit/tests/file_format_mock.rs`.
`join/` hash join (P2-T052a): `mod.rs` API + Spark semantics · `keys.rs` row-format keys, seeded foldhash,
radix partitions · `table.rs` row arena + `JoinTable` (shared `KeyTable`, E02) · `build.rs` build
sink · `probe.rs` probe operator + match cursor · `grace.rs` spill partitions, flush-phase grace join.
Operators with work after their input (`needs_flush`) get `end_input` + a per-pipeline barrier + `flush`.

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
benches: `cargo bench -p krishiv-exec --bench pipeline` · `--bench operators`
