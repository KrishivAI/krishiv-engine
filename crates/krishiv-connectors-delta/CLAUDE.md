# krishiv-connectors-delta

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P3.

## Purpose
Delta Lake connector on `delta_kernel` (no default features) with our own Engine implementation.

## Layout
- `src/error.rs` kernel errors as Krishiv errors (a Krishiv error raised in a handler travels through the kernel and back).
- `src/engine/schema.rs` kernel `StructType`/`DataType`/`Scalar` to Arrow 60 and back (µs timestamps, unshredded Variant, field ids).
- `src/engine/data.rs` `KrishivEngineData`: the kernel's `EngineData`/`GetData` over a `RecordBatch` (port of the kernel's `arrow_data.rs`/`arrow_get_data.rs`).
- `src/engine/mod.rs` `DeltaEngine`, the kernel's `Engine` (IoContext, StoreId, table root, tokio `Handle`, `DeltaLimits`, the query's `CancellationToken`) and `delta_blocking`, the only entry to kernel calls; `src/engine/pool.rs` `LogPool`, the one process-wide bounded pool of its own OS threads (`krishiv.delta.logThreads`; never the runtime's blocking pool, which the handlers' local I/O needs), cancellable slot wait, `delta.log_pool.wait_ms` (`WaitStats`), the log-pool thread flag the handlers' `block_on` asserts.
- `src/engine/storage.rs` `KrishivStorageHandler` (`list_from` via `IoContext::list_from`, `read_files` lazily per item with the `_last_checkpoint`/`.crc` limits, `put` with `put_if_absent`, `head`, `delete`; `copy_atomic` unsupported); `src/engine/json.rs` `KrishivJsonHandler` (arrow-json; `UTC` timestamps cross arrow-json as `+00:00` since chrono-tz is banned) and the parse limits (`KRISHIV_CATALOG_LIMIT_EXCEEDED`); `src/engine/paths.rs` `TableRoot`, every URL confined to the table root (lexically, and on local roots also with symlinks resolved).
- `src/engine/parquet.rs` `KrishivParquetHandler` over krishiv-io's `ScanStream`/`ParquetFormat` (checkpoints, sidecars, `stats_parsed`, CDF): lazy per-file reads, one batch per pull with cancellation checked before each, the footer length checked against `krishiv.catalog.maxMetadataBytes`, columns by field id then name at every struct level, reader widenings, `row_index`/`_file`; `write_parquet_file` one overwriting put returning the size; `read_parquet_footer` with field ids.
- `src/engine/eval/` `KrishivEvaluationHandler`, the kernel's evaluator over Arrow compute (port of the kernel's `arrow_expression`): `expr.rs` expressions (checked arithmetic, safe casts, `Coalesce`, `Struct`, `StructPatch`, `Array`), `predicate.rs` (NOT pushed down, Kleene AND/OR, null-safe `DISTINCT`, `IN` never NULL; `Unknown`/`Opaque` unsupported), `json.rs` `ParseJson` (bad timestamp/date/decimal cells NULL, an unparseable batch NULL) and `ToJson` (ms, `Z`), `partition.rs` `MapToStruct` (Delta's partition-value encoding), `types.rs` the type check and output conformance; `src/engine/relabel.rs` zone/name relabelling without copies (`UTC` crosses Arrow as `+00:00`).
- Tests: `src/engine/eval/tests.rs` (`delta_eval_parity`, the kernel's evaluator tests as a table), `src/engine/tests.rs` + `parquet_tests.rs` + `oracle_tests.rs` (checkpoint reads of the Spark 4.2 + delta-spark 4.4.1 tables in `tests/fixtures/delta-4.4.1/`: V1, V2 with a sidecar, multi-part; generator `tests/scripts/gen_delta_fixtures.py`; local temp tables: `engine_reports_written_file_size`, handlers, `delta_handler_cancellation`, `catalog_parse_limits`, `delta_log_paths_confined`, a kernel snapshot and scan metadata, with partition pruning and data skipping, through the assembled engine), `schema_tests.rs`, `data_tests.rs`.

Fuzz target `fuzz/fuzz_targets/delta_commit_json.rs` (one commit through `parse_json`, `read_json_files`, the `_last_checkpoint`/`.crc` limits and a kernel snapshot + scan; seeds from the oracle tables' logs).

## Invariants
- `delta_kernel` with `default-features = false`; never `deltalake`.
- Kernel calls run only through `DeltaEngine::delta_blocking` (the log pool), never on a runtime worker; lazy kernel iterators are drained inside that call; puts are never raced against cancellation once started (`block_on_to_completion`); every handler call checks the query's cancellation first and confines its URLs to the table root.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028

## Test
`cargo nextest run -p krishiv-connectors-delta --no-tests=pass` · `cargo clippy -p krishiv-connectors-delta --all-targets -- -D warnings`
