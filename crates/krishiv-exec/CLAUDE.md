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
`testing.rs` test operators (also used by benches) · `output_spill.rs` spill hook of finished breaker output · `loom_model.rs`.
`operators/` (P2-T051) stateless operators: `scan.rs` MemoryScan + `ScanPushdown` (pushdowns decided once as an
immutable `ScanPushdowns`; implemented by scan factories: MemoryScan, FileScan) ·
`filter.rs` · `project.rs` · `limit.rs` · `union.rs` (per-branch `UnionAlign`; UNION ALL fan-in is T074, distinct UNION is the aggregate's) ·
`expr.rs` the `Predicate`/`ScalarExpr` evaluator interface the physical planner plugs in ·
`hash.rs` shared hash-table blocks (seeded key hash, key normalization + row encoding, `KeyTable`, spill
partition bits; for the join too) · `aggregate/` hash aggregate (P2-T053a): `mod.rs` spec + Spark result types,
`accumulator.rs` per-group states, `table.rs` one table, `sink.rs` breaker (spill runs, finalize merge,
`spill_global`), `partial.rs` flushable partial phase with adaptive bypass, `node.rs` plan nodes + DISTINCT/UNION ·
(P2-T117b) `file_scan.rs` FileScan/FileScanSource over any `krishiv_io::FileFormat` (Spark partitions cut into
about four morsels per driver, footer read-ahead, `_metadata` struct column; filters stay `Inexact`) · `file_write.rs` FileWriteSink
(one writer task per driver behind a bounded channel; nothing visible after a failed or cancelled write) + FileWrite.
End-to-end tests over the mock format: `krishiv-testkit/tests/file_exec.rs`; the conformance suite through the
executor: `mock_exec` in `krishiv-testkit/tests/file_format_mock.rs`.
`join/` hash join (P2-T052a): `mod.rs` API + Spark semantics · `keys.rs` row-format keys, seeded foldhash,
radix partitions · `table.rs` row arena + `JoinTable` (shared `KeyTable`, E02) · `build.rs` build
sink (also the runtime filter producer, P2-T163: per-driver partials, published at finalize, `Disabled` on drop) ·
`probe.rs` probe operator + match cursor · `grace.rs` spill partitions, flush-phase grace join.
`values/` (P2-T177) `ValueService` + `LocalValues`: runtime filters, subquery results and broadcast builds as
versioned values (`ValueId`, `Version` are plan data), published as one partial per producer partition and visible
once every producer partition published (count 1 unless the task says otherwise; a retried partition's partial is
ignored); `WorkerContext` holds the service, a fresh `LocalValues` per instantiation by default.
Runtime filters (P2-T163): `operators/runtime_filter.rs` the per-scan `RuntimeFilters` list and the row stage
(adaptive switch-off); `planner/runtime_filter.rs` legal targets (sites) and the cost model. A scan's pipeline
starts after the pipelines publishing the filters worth the wait (`scan_waits`: not when the build reads the scan's
own table, unless it is far smaller) unless they run with or after it (`runner.rs`, `with_filter_order`); no driver
ever waits on a filter's value.
`window/` (P2-T055) the `Window` operator over input sorted by partition then order keys (the planner puts a
`Sort` below it; `planner/window.rs`; over a hash exchange that sort splits its rows into hash buckets merged and read
by several drivers, `operators/sort/sort.rs`): `mod.rs` plan types (`WindowPlan`, plan data) + the operator (buffers one
partition, evaluates complete partitions), `eval.rs` boundaries, ranking functions, frame aggregates (through the
aggregate `Accumulator`, one row at a time in Spark's order; decimal `sum`/`avg` exact in `i256`, checked when
evaluated), `frames.rs` (ROWS frames with offsets and frames to the partition's end: segment tree, removable
cumulative, in-order fold for float/ANSI sums), `spill.rs` (buffered rows spill under pressure; after a spill rows wait for the flush phase, which reads
them back in order; a partition that does not fit is copied to its own file; failed/cancelled steps leave the state
failed, never a short result), `stream.rs` (a partition evaluated from disk: lead and trailing readers, bounded
memory), `group_limit.rs` (Spark's `WindowGroupLimit`: the exact limit operator below the window, the `Ranker`
the sort also uses to prune each sorted batch while that pays; the planner infers it for a filter over the window,
`planner/window.rs`), `tests.rs` (property test against a naive reference; spilled vs in-memory equality).
Operators with work after their input (`needs_flush`) get `end_input` + a per-pipeline barrier + `flush`.
`physical/` (P2-T172) the physical plan as data: `mod.rs` `PhysNode`/`PhysicalPlan`, `data.rs` the sealed
`PlanData` marker, `instantiate.rs` `WorkerContext` + `instantiate` (the operator graph built from a plan on the
worker that runs it; the planner validates nodes with the same builders). `Scheduler::submit` takes a
`PhysicalPlan`; `QueryRunner::execute` runs an already instantiated `ExecNode` graph.
`planner/aggregate.rs` also lowers `PlanNode::PartialAggregate` (P2-T167) onto a partial hash aggregate, and a call over
`ResolvedExpr::StateInput` onto one merging its states; it refuses states the optimizer typed differently.
`planner/properties.rs` (P2-T174) the ADR 0012 property pass: required distributions and provided placements (by
attribute id); `Exchange` nodes (`Hash` on the routing hash, `Single`, `Broadcast`, `RoundRobin`, `Range`) are inserted
only where a requirement is unmet; on one worker instantiation lowers each exchange into the operators around it.

## Key files
Paths are relative to this crate.
- `src/operator.rs` Source, Operator and Sink traits; `src/pipeline.rs` `ExecNode` and `PipelineBuilder`
- `src/driver.rs` the pipeline driver: yields, memory waits, spill servicing, cancellation
- `src/runner.rs` `QueryRunner` and the scheduler (first error wins)
- `src/physical/mod.rs` `PhysNode` and `PhysicalPlan` as serializable data; `src/physical/instantiate.rs` builds operators on the worker
- `src/planner/mod.rs` physical planner (IR to physical plan); `src/planner/runtime_filter.rs` runtime-filter sites
- `src/operators/mod.rs` stateless operators (scan, filter, project, limit, union)
- `src/aggregate/mod.rs` hash aggregate; `src/join/mod.rs` hash join (build, probe, grace spill)
- `src/rows/mod.rs` row-table layout and pages shared by aggregate and join; the stored row hash `h` is a fast seedless
  table hash, never the routing hash `H_kg` (kernels `hash::routing`, used only by exchanges and key groups);
  tables place rows by the seeded `g = finalize(h, seed)`; `rows/flooding_tests.rs` bounds probe runs under crafted and forged `h` collisions
- `src/window/mod.rs` window operator and plan types; `src/window/eval.rs` frames, ranking and frame aggregates; `src/planner/window.rs` plans it over sorted input
- `src/operators/file_scan.rs` and `file_write.rs` file scan and write operators over `krishiv-io`

## Extension points
- New operator: follow skill `krishiv-operator`; stateless ones go in `src/operators/`, stateful ones next to aggregate or join.
- New plan node: a `PhysNode` variant in `src/physical/mod.rs` with its builder in `src/physical/instantiate.rs`; the planner validates it with the same builder.

## Invariants
- All memory through krishiv-memory reservations; every spill path tested with tiny-memory tests.
- Only the driver yields, awaits memory, spills or checks cancellation; `execute`/`sink` stay sync and bounded.
  Exception: a breaker's finished in-memory result, owned by no driver, spills on a task of its own
  (`output_spill.rs`: aggregate and sort results); the coordinator drops a finalized state after its last reader.
- At query end (success, error, cancel) the query pool is empty: tests assert it.
- Cancellation-safe, no blocking in async; metrics feed EXPLAIN ANALYZE.
- New operators follow skill `krishiv-operator`.
- Physical plans are serializable data: plan types in `physical/` hold no `Arc<dyn …>`, locks, atomics or cells
  (`PlanData`, `cargo xtask source-guard`); scan sources, spill scopes, runtime-filter and subquery cells are
  created at instantiation. Review question for every change: does this assume one process?
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0006, 0007, 0011, 0013

## Tests
`cargo nextest run -p krishiv-exec --no-tests=pass` · `cargo clippy -p krishiv-exec --all-targets -- -D warnings` ·
loom models: `RUSTFLAGS="--cfg loom" cargo test -p krishiv-exec --lib --release loom_model::` (use a separate target dir) ·
benches: `cargo bench -p krishiv-exec --bench pipeline` · `--bench operators` · `--bench routing_hash` (per-row
`H_kg` vs the row layout's table hash and seeded foldhash)
