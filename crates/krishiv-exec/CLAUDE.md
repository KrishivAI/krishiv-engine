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
`testing.rs` test operators (also used by benches) · `output_spill.rs` spill hook of finished breaker output · `cohort.rs` `reserve_in_cohort`: a request the pool refuses waits while another member of a query cohort holds memory it frees on its own (nested-loop chunks; sort merge budgets, `operators/sort/budget.rs` `MERGE_COHORT`; grace joins' partition loads and tables, `join/grace.rs` `TABLE_COHORT`, holding seats only) · `loom_model.rs`.
`operators/` (P2-T051) stateless operators: `scan.rs` MemoryScan + `ScanPushdown` (pushdowns decided once as an
immutable `ScanPushdowns`; implemented by scan factories: MemoryScan, FileScan) ·
`filter.rs` · `project.rs` · `limit.rs` · `union.rs` (per-branch `UnionAlign`; UNION ALL fan-in is T074, distinct UNION is the aggregate's) ·
`spilled_scan.rs` `SpilledScan`, one driver streaming a spill file a batch at a time under its own consumer (the scan of `planner/table.rs` `SpilledTable`, a relation cache entry on disk) ·
`expr.rs` the `Predicate`/`ScalarExpr` evaluator interface the physical planner plugs in ·
`select.rs` (design P2-E3 U2) `Selection` (ascending `u32` row ids) and `RowSelect` (a fused predicate and a scan's row stage; after a selective stage the predicate reads a copy of the kept rows, of only its columns when compiled as a `Narrow`) ·
`hash.rs` shared hash-table blocks (seeded key hash, key normalization + row encoding, `KeyTable`, spill
partition bits; for the join too) · `aggregate/` hash aggregate (P2-T053a): `mod.rs` spec + Spark result types,
`accumulator.rs` per-group states (`accumulator/paired.rs`: a decimal `sum(x)` and `avg(x)` of one aggregate folded in one pass where both take the proof path), `table.rs` one table, `sink.rs` breaker (spill runs, finalize merge,
`spill_global`; a batch whose worst case the pool refuses is applied in slices, `slice_tests.rs`, unless a spill of the driver's tables would free the shortfall (then it spills, ADR 0007 addendum 2026-11-08); `sink/disk.rs` the disk phase's partition tasks in parallel, one output file each), `partial.rs` flushable partial phase with adaptive bypass, `node.rs` plan nodes + DISTINCT/UNION, `udf.rs` session aggregate UDFs (ADR 0045: `UdafCall` plan data, `UdafEnv`, the guarded `UdfAcc` accumulator kind) ·
(P2-T117b) `file_scan.rs` FileScan/FileScanSource over any `krishiv_io::FileFormat` (Spark partitions cut into
about four morsels per driver, footer read-ahead, `_metadata` struct column; filters stay `Inexact`) · `file_write.rs` FileWriteSink
(one writer task per driver behind a bounded channel; nothing visible after a failed or cancelled write) + FileWrite.
End-to-end tests over the mock format: `krishiv-testkit/tests/file_exec.rs`; the conformance suite through the
executor: `mock_exec` in `krishiv-testkit/tests/file_format_mock.rs`.
`join/` hash join (P2-T052a): `mod.rs` API + Spark semantics · `keys.rs` row-format keys, seeded foldhash,
radix partitions · `table.rs` row arena + `JoinTable` (shared `KeyTable`, E02) · `build.rs` build
sink (also the runtime filter producer, P2-T163: per-driver partials, published at finalize, `Disabled` on drop) ·
`probe.rs` probe operator + match cursor (its first read of the build, `BuildGlobal::built`, ends the build's idle
window: until then a finished build's tables are an `OutputSpill` written partition by partition, largest first, and
idle after `krishiv.exec.join.idleSpillAfter`, so the arbitrator asks them before a running consumer self-spills; study
30. Each probe driver then holds the build's level-0 tables, `build.rs` `Level0`, whose memory includes the per-partition
reservations moved at the seal; the last probe driver done after the flush barrier releases them, so a downstream
sink's final phase gets that memory; a `Prebuilt` broadcast build is never released) · `grace.rs` spill partitions,
flush-phase grace join · `broadcast.rs`
(P2-T177 unit 3) `BroadcastTables`, a worker resource (`WorkerContext::with_broadcasts`): a broadcast join over a cut
broadcast exchange builds its table once per job and worker (its own query over the resource's pool, no spill, its
reservations never abort a query) and every task probes it through a `PrebuiltSink` that republishes its runtime
filters. A shared build that does not fit (`UNABLE_TO_ACQUIRE_MEMORY`, or past `krishiv.exec.join.sharedBroadcastBytes`, the
worker's cap per query) falls back to one spilling build per task (`SharedOrOwnRows` then reads the broadcast for the
task; owner decision 2026-11-06); each task's build reports `broadcast_shared` or `broadcast_fallback` (count 1).
`join/merge.rs` sort-merge join (P2-T056): `SortMergeJoin` is the source over two `Sort`s in hash buckets of the keys
(`Sort::bucket_reader`, `PipelineBuilder::set_merge_source`), `Merge` the merge of one bucket pair; a key's right rows
are the group, spilled past `krishiv.exec.join.smj.bufferBytes`; `join/fallback.rs` runs a grace partition that will not
split through the same `Merge`.
`values/` (P2-T177) `ValueService` + `LocalValues`: runtime filters, subquery results and broadcast builds as
versioned values (`ValueId`, `Version` are plan data), published as one partial per producer partition and visible
once every producer partition published (count 1 unless the task says otherwise; a retried partition's partial is
ignored); `WorkerContext` holds the service, a fresh `LocalValues` per instantiation by default. A runtime filter's
partials merge with `KeyFilter::merge` under the publishing join's options (`ValueService::filter_options`); a
partitioned join's Bloom size is planned (`KeyFilterOptions::bloom_bytes`, P2-T177 unit 2) and dropped at
instantiation when the filter has one producer partition.
Runtime filters (P2-T163): `operators/runtime_filter.rs` the per-scan `RuntimeFilters` list and the row stage
(adaptive switch-off; a scan can hand its drivers' stages to the hash probe above it, `RuntimeFilters::hand_to_consumer`); `planner/runtime_filter.rs` legal targets (sites) and the cost model; `GroupKeyCode`'s encoding join (a dimension joined by `<=>` with its numbered dictionary, `krishiv_optimizer::is_group_code_dictionary`) publishes none (`planner/join.rs`). A scan's pipeline
starts after the pipelines publishing the filters worth the wait (`scan_waits`: not when the build reads the scan's
own table, unless it is far smaller) unless they run with or after it (`runner.rs`, `with_filter_order`); no driver
ever waits on a filter's value. A `TopN` offers the first key of its last row the same way (`PhysNode::TopN::bound`, `ValueService::tighten`, design P2-T063): never where a filter below may raise or a nondeterministic plan sits below.
`operators/late_fetch.rs` (design P2-T063) `LateFetch`: a breaker above a `TopN` over a narrow scan (`ScanNode::row_id` appends `_krishiv_row_file`, `_krishiv_row_index`) that reads the deferred columns of its rows through `ScanPushdown::take`, in the rows' order. Planned by `Planner::late_fetch` under the rules of the note's §4, up to `krishiv.exec.lateFetch.maxRows`; a partition column stays in the narrow scan, and an expression projection above the scan disables it.
A filter on a partition column skips the files of partitions it drops (dynamic partition pruning, `file_scan.rs`).
`window/` (P2-T055) the `Window` operator over input sorted by partition then order keys (the planner puts a
`Sort` below it; `planner/window.rs`; over a hash exchange that sort splits its rows into hash buckets merged and read
by several drivers, `operators/sort/sort.rs`): `mod.rs` plan types (`WindowPlan`, plan data) + the operator (buffers one
partition, evaluates complete partitions), `eval.rs` boundaries, ranking functions, frame aggregates (through the
aggregate `Accumulator`, one row at a time in Spark's order; decimal `sum`/`avg` exact in `i256`, checked when
evaluated), `frames.rs` (ROWS frames with offsets and frames to the partition's end: segment tree, removable
cumulative, in-order fold for float/ANSI sums), `spill.rs` (buffered rows spill under pressure; after a spill rows wait for the flush phase, which reads
them back in order; a partition that does not fit is copied to its own file; failed/cancelled steps leave the state
failed, never a short result), `stream.rs` (a partition evaluated from disk: lead and trailing readers, bounded
memory), `disk.rs` (frames and `lag`/`lead`/`nth_value` of a partition on disk: forward readers, a frame's batches held or folded from the file), `group_limit.rs` (Spark's `WindowGroupLimit`: the exact limit operator below the window, the `Ranker`
the sort also uses to prune each sorted batch while that pays; the planner infers it for a filter over the window,
`planner/window.rs`), `tests.rs` (property test against a naive reference; spilled vs in-memory equality). `RANGE` value bounds are `FrameBound::RangeInt`/`RangeFloat` (the key's units), `RangeDayTime` (a day-time interval over a `TIMESTAMP`/`TIMESTAMP_NTZ` key: whole days on the zone's clock, then elapsed time) and `RangeMonths` (calendar months over a `DATE` or `TIMESTAMP` key; the `TIMESTAMP` zone is the session's, passed to `Window::new` because the column carries only a marker); `month_tests.rs` (month bounds from disk vs memory in a DST zone).
Operators with work after their input (`needs_flush`) get `end_input` + a per-pipeline barrier + `flush`.
`loops.rs` (T180 unit 5b) `UnionLoopSource`: a recursive CTE (`PhysNode::UnionLoop`/`LoopRef`) as one source. The anchor and each
round of the recursive term run as queries of their own over a child `QueryContext` (child memory pool, cancelled with the
outer query); every batch goes downstream at once and into the round's working table (`shared.rs` `Output::open/append`,
spilled to the query's scope under pressure), which the next round's `LoopRef` reads (`SharedSource::over`). `Detached`
(`physical/instantiate/detached.rs`) instantiates the anchor and each round when it starts, with the loop tables bound; shared plans are built again for each run, their unseeded random calls salted as the outer plan salted them, so an enclosing nondeterministic CTE gives the anchor, round 1 and the outer query one value and round `k` the stream of partition `k - 1`, as in Spark (the runs' own calls take a salt range per loop). Limits (`LoopLimits`) are plan data. Seeded random functions draw round `k`'s stream at partition `k - 1` (`EvalEnv::partition_shift`, as Spark does); a working table of at most 100 rows is read by one driver in order (`Output::read_by_one_driver`), as Spark reads its `LocalRelation`. Not matched: Spark round-robins a larger working table into the shuffle partitions, which Krishiv's drivers do not reproduce. `EXPLAIN ANALYZE` reports the rounds, their rows and the folded compute time and spill bytes of every round, a LIMIT-stopped one included (`add_run_metrics`).
`physical/` (P2-T172) the physical plan as data: `mod.rs` `PhysNode`/`PhysicalPlan`, `data.rs` the sealed
`PlanData` marker, `instantiate.rs` `WorkerContext` + `instantiate` (the operator graph built from a plan on the
worker that runs it; the planner validates nodes with the same builders). `instantiate/distributed.rs`: what only a task of a cut plan
uses (exchange write with its collected filters, exchange reads, `TaskSpec::filters`, broadcasts shared per worker). `PhysicalPlan::shared` (P2-T198): subplans
read by `PhysNode::SharedRead`; a walk over a plan starts at `PhysicalPlan::roots`. `shared.rs` runs them: `SharedSink`
ends the body's pipeline (built once through `PipelineBuilder::shared_pipeline`, so pipeline dependencies are a DAG),
keeps the batches as segments (memory or spill file) under a consumer of its own whose `OutputSpill` hook writes every
batch still in memory, and each `SharedSource` reads the whole list through its own cursor (own `SpillReader` per
spilled segment). The planner (`planner/build.rs`) reads every `CteRef` of a nondeterministic definition from a shared
plan, and a deterministic one (the optimizer's `ShareCte`) only in a second pass for the groups no runtime filter reaches. `Scheduler::submit` takes a
`PhysicalPlan`; `QueryRunner::execute` runs an already instantiated `ExecNode` graph.
`planner/aggregate.rs` also lowers `PlanNode::PartialAggregate` (P2-T167) onto a partial hash aggregate, and a call over
`ResolvedExpr::StateInput` onto one merging its states; it refuses states the optimizer typed differently.
`exchange/` (P2-T176 unit 1) the spec 13 §6 traits `Exchange`/`ExchangeWriter`/`ExchangeReader` (blocking mode,
commit before read, first committed attempt wins), `route` (hash by the plan's `HashScheme`, round-robin by the routing
hash of the whole row's columns it covers (a nested column never fails a row), gathers/broadcasts/ranges to partition 0), `ExchangeSink` (a map task's root: rows routed and
buffered per reduce partition under the driver's consumer, a spill hands them to the writer as frames, commit at
finalize) and `ExchangeRead` (a reduce task's `PhysNode::ExchangeSource`: `(map, reduce)` pairs as morsels, one driver
in map order for a gather). `instantiate_task` writes a root exchange when `TaskSpec::output` is set and reads each
source as `TaskSpec::exchanges` says; `exchange_schemas` gives a fragmenter each exchange's columns. Storage is
`krishiv-shuffle`'s, fragments and stages are `krishiv-scheduler`'s.
`planner/probe_select.rs` (design P2-E3 U2) fuses a `Filter` (nothing may raise: predicate, keys, residual) and a
scan's row stage below a hash probe's column-only projections into `HashJoinPlan::select` (codec feature `fused_probe`);
the probe (`join/probe.rs`) encodes the selected rows' keys and gathers by composed ids; `EXPLAIN` prints `probe filter:` on the join.
`planner/properties.rs` (P2-T174) the ADR 0012 property pass: required distributions and provided placements (by
attribute id); `Exchange` nodes (`Hash` on the routing hash, `Single`, `Broadcast`, `RoundRobin`, `Range`) are inserted
only where a requirement is unmet, plus one hash exchange for every `DISTRIBUTE BY` (`Planner::repartition`: key expressions computed below, dropped above, so they are evaluated as Spark's shuffle evaluates them); on one worker instantiation lowers each exchange into the operators around it.
An aggregate over rows that must move is planned in two phases, `Aggregate{Partial}` -> exchange -> `Aggregate{Final}`
over states (`planner/aggregate.rs`, design P2-T174 unit 2; `DISTINCT`: partial over keys and argument, exchange,
partial merge, final, `physical::distinct_phases`, fused back by `fused_distinct`); `physical::fused_aggregate`
runs the three as one `HashAggregate::fused` breaker on one worker (`WorkerContext::with_fused_aggregates(false)` runs
them as written, for tests), whose drivers record the `ExchangeStats` (`exchange_stats.rs`: rows and bytes per
partition, `Hll(10)` of the keys) that `EXPLAIN ANALYZE` prints as `exchange_*`.

## Key files
Paths are relative to this crate.
- `src/carrier.rs` moves a value between decimal carriers and between the calendar struct and `Interval(MonthDayNano)` (nested ones too); `planner/eval/func.rs` `Recarried` keeps registry functions and UDFs on `Decimal128`
- `src/operator.rs` Source, Operator and Sink traits; `src/pipeline.rs` `ExecNode` and `PipelineBuilder`
- `src/driver.rs` the pipeline driver: yields, memory waits, spill servicing, cancellation
- `src/runner.rs` `QueryRunner` and the scheduler (first error wins)
- `src/physical/mod.rs` `PhysNode` and `PhysicalPlan` as serializable data; `src/physical/instantiate.rs` builds operators on the worker
- `src/planner/mod.rs` physical planner (IR to physical plan); `src/planner/runtime_filter.rs` runtime-filter sites
- `src/planner/build_project.rs` (S4') an exact, narrowing expression above an inner hash join computed on the build input before its table is built; `krishiv.exec.join.buildProjection`; differential `tests/planner/build_projection.rs`
- `src/operators/mod.rs` stateless operators (scan, filter, project, limit, union)
- `src/aggregate/mod.rs` hash aggregate (`rowtable.rs` the row tables, `rowtable/local.rs` the batch-local key map of a low-cardinality batch (E2), `rowtable/direct.rs` direct-address ids of a tiny key domain (D1, `krishiv.exec.agg.directKeys`), `merge_index.rs` the salted index of a large final-phase merge); `src/join/mod.rs` hash join (build, probe, grace spill)
- `src/rows/mod.rs` row-table layout and pages shared by aggregate and join; the stored row hash `h` is a fast seedless
  table hash, never the routing hash `H_kg` (kernels `hash::routing`, used only by exchanges and key groups);
  tables place rows by the seeded `g = finalize(h, seed)`; `rows/flooding_tests.rs` bounds probe runs under crafted and forged `h` collisions
- `src/window/mod.rs` window operator and plan types; `src/window/eval.rs` frames, ranking and frame aggregates; `src/planner/window.rs` plans it over sorted input
- `src/operators/file_scan.rs` and `file_write.rs` file scan and write operators over `krishiv-io`
- `src/planner/write.rs` (P3-T023a) `WriteMode` (append, truncate, overwrite by filter, dynamic overwrite), `WriteRequest` (mode, table name, input schema, options, write id) and `TableCapabilities` (Spark's `TableCapability` write values, checked as Spark v4.0.0's `TableCapabilityCheck`: every mode needs `BATCH_WRITE` or `V1_BATCH_WRITE`, a truncate also `TRUNCATE` or `OVERWRITE_BY_FILTER`, dynamic overwrite `BATCH_WRITE` and `OVERWRITE_DYNAMIC`); `TableProvider::capabilities` / `writer(&WriteRequest)` default to no write modes, and `TableCapabilities::check` raises `UNSUPPORTED_FEATURE.TABLE_OPERATION` ("... in batch mode") before anything is written
- `src/python_relation.rs` (design P3-T085 §3.2) the Python relation seam: `RelationCallSpec`, `PreparedRelationUdf::open`, `UdfTask` fed by `Frame`s (`UdfRuntime::prepare_relation` defaults to `KRISHIV_UDF_NOT_AVAILABLE`); `python_relation/sink.rs` `RelationSink` (map, grouped, cogrouped, aggregate shapes: one task per driver, groups cut where the sorted key changes, frames lead with the keys, a one-sided cogroup key gets a zero-row frame for the other side, the window checked per frame with a reserved backlog, `Blocked` while a lane's frame window is full or cuts wait, its future framing the backlog so no input is taken before it drained; a frame's computed columns reserved, results in the shared plan's list as they arrive), `python_relation/lane.rs` (a lane's frames, its reserved results and the pump; a refused result stops the task's reads), `python_relation/pair.rs` (an aggregate's pending keys joined with the values as they arrive); `planner/python_relation.rs` plans `MapPartitions`/`GroupMap`/`CoGroupMap`/Python-UDAF `Aggregate` onto `PhysNode::PythonMap`/`PythonGroupMap`/`PythonCoGroupMap`/`PythonAggregate` (hash exchange + spillable sort on the keys; a cogroup is one union stream sorted by key then side), refuses barrier mode, Python window aggregates and UDTFs (PR-26c); `physical/instantiate/python_relation.rs` builds them
- `src/planner/ndv_sample.rs` (P2-T164 U6) `FileTable::sample_distinct`: plan-time distinct-value samples of the file columns an equality or `IN` compares, and of the column group keys while `krishiv.optimizer.groupKeyCode` is on (the session picks them, `catalog/ndv_sampling.rs`) (first rows in file order through `ScanStream`, `krishiv.optimizer.ndvSampling.*`), exact for a table the sample read whole, else GEE (`estimate`); cached in `krishiv_io::DistinctSamples`; `FileTable::statistics` reports them as `StatSource::Sampled`; `FileTable::sample_outcomes` says per table what each sample did (`SampleOutcome`: sampled, cached, timed out, no time left), which `EXPLAIN COST` prints as "Distinct-Value Samples" and `EXPLAIN VERBOSE` as a `Statistics` step

## Extension points
- New operator: follow skill `krishiv-operator`; stateless ones go in `src/operators/`, stateful ones next to aggregate or join.
- New plan node: a `PhysNode` variant in `src/physical/mod.rs` with its builder in `src/physical/instantiate.rs`; the planner validates it with the same builder.

## Invariants
- All memory through krishiv-memory reservations; every spill path tested with tiny-memory tests.
- Only the driver yields, awaits memory, spills or checks cancellation; `execute`/`sink` stay sync and bounded.
  Exception: a breaker's finished in-memory result, owned by no driver, spills on a task of its own
  (`output_spill.rs`: aggregate and sort results, and a hash join build's finished tables until the first probe driver
  reads them, `join/build.rs` `IdleBuild`); the coordinator drops a finalized state after its last reader.
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

## Split modules (T193)
Files split to stay under 2,000 lines; the parent file keeps the types and the entry points.
- `aggregate/accumulator/` `fold.rs` (the batch fold match), `output.rs` (state and result columns), `results.rs` (percentile and struct-extremum results), `helpers.rs` (decimal proofs, column readers, decimal text), `tests.rs`, `proof_tests.rs`; `aggregate/rowtable/` `probe.rs` (lookups), `merge.rs` (merging tables, the salted index), `tests.rs`; `aggregate/hll/tables.rs` (bias tables); `planner/build/util.rs` (plan-building helpers); `planner/eval/hof/sort.rs` (the sort-with-comparator function); test modules of `aggregate/tests/`, `window/tests/` in `partN.rs`.
