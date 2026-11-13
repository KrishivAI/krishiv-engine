# krishiv-connectors-delta

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P3.

## Purpose
Delta Lake connector on `delta_kernel` (no default features) with our own Engine implementation.

## Layout
- `src/error.rs` kernel errors as Krishiv errors (a Krishiv error raised in a handler travels through the kernel and back).
- `src/engine/schema.rs` kernel `StructType`/`DataType`/`Scalar` to Arrow 60 and back (µs timestamps, unshredded Variant, field ids).
- `src/engine/data.rs` `KrishivEngineData`: the kernel's `EngineData`/`GetData` over a `RecordBatch` (port of the kernel's `arrow_data` and `arrow_get_data` modules).
- `src/engine/mod.rs` `DeltaEngine`, the kernel's `Engine` (IoContext, StoreId, table root, tokio `Handle`, `DeltaLimits`, its cancellation: a token at planning, the running query's `QueryContext` for a scan via `DeltaEngine::for_query`) and `delta_blocking`, the only entry to kernel calls; `src/engine/pool.rs` `LogPool`, the one process-wide bounded pool of its own OS threads (`krishiv.delta.logThreads`; never the runtime's blocking pool, which the handlers' local I/O needs), cancellable slot wait, `delta.log_pool.wait_ms` (`WaitStats`), the log-pool thread flag the handlers' `block_on` asserts.
- `src/engine/storage.rs` `KrishivStorageHandler` (`list_from` via `IoContext::list_from`, `read_files` lazily per item with the `_last_checkpoint`/`.crc` limits, `put` with `put_if_absent`, `head`, `delete`; `copy_atomic` unsupported); `src/engine/json.rs` `KrishivJsonHandler` (arrow-json; `UTC` timestamps cross arrow-json as `+00:00` since chrono-tz is banned) and the parse limits (`KRISHIV_CATALOG_LIMIT_EXCEEDED`); `src/engine/paths.rs` `TableRoot`, every URL confined to the table root (lexically, and on local roots also with symlinks resolved).
- `src/engine/columnar.rs` `KrishivParquetHandler` (the kernel's ParquetHandler; the format comes from `FormatRegistry`, never named) over krishiv-io's `ScanStream`/`ParquetFormat` (checkpoints, sidecars, `stats_parsed`, CDF): lazy per-file reads, one batch per pull with cancellation checked before each, the footer length checked against `krishiv.catalog.maxMetadataBytes`, columns by field id then name at every struct level, reader widenings, `row_index`/`_file`; `write_parquet_file` one overwriting put returning the size (the whole file is buffered outside any reservation: PR-37, checkpoint writes, reserves it or streams a multipart put); reads versioned by the store's `head` (ETag/mtime); `read_parquet_footer` with field ids.
- `src/engine/eval/` `KrishivEvaluationHandler`, the kernel's evaluator over Arrow compute (port of the kernel's `arrow_expression`): `expr.rs` expressions (checked arithmetic, safe casts, `Coalesce`, `Struct`, `StructPatch`, `Array`), `predicate.rs` (NOT pushed down, Kleene AND/OR, null-safe `DISTINCT`, `IN` never NULL; `Unknown`/`Opaque` unsupported), `json.rs` `ParseJson` (bad timestamp/date/decimal cells NULL, an unparseable batch NULL) and `ToJson` (ms, `Z`), `partition.rs` `MapToStruct` (Delta's partition-value encoding), `types.rs` the type check and output conformance; `src/engine/relabel.rs` zone/name relabelling without copies (`UTC` crosses Arrow as `+00:00`).
- `src/features.rs` Krishiv's reader allow-list (design §3.5–3.7) on a `ReadProtocol` (protocol + configuration): unsupported reader features → `DELTA_UNSUPPORTED_FEATURES_FOR_READ` (delta-spark's parameters, `Krishiv <version>`); catalog-managed, coordinated commits (writer feature or `delta.coordinatedCommits.*`) and row-tracking columns → `UNSUPPORTED_FEATURE.TABLE_OPERATION`; `snapshot_error` maps the kernel's own refusals (`MaxCatalogVersion`, "Feature 'x' is not supported") the same way. Tests `src/features/tests.rs` (every row on edited protocols; delta-spark's probe `PROBE-unknown-reader-feature.json`).
- `src/table/mod.rs` `DeltaTable`, the `TableProvider` of one table at one version (`open`/`open_with(engine, Option<u64>, config)`, the pinned version is the time-travel hook; `hadoop_path` is delta-spark's `file:/…` form of `<tableNameOrPath>` for a path table, a catalog table passes its name); `src/table/cache.rs` `SnapshotCache`, the cross-query snapshot cache (ADR 0037; key: root, `StoreId`, credential identity; bounded by count and by estimated bytes, `krishiv.delta.snapshotCache.maxBytes`, a listing over the bound not kept; a hit is checked with one `head` of its last commit file, then brought up to date from the cached snapshot; any failure drops the entry and rebuilds): snapshot on the log pool, the allow-list before the listing, engine Arrow schema (column-mapping metadata kept), `ScanUnit` per live file, statistics = `numRecords` less DV cardinality, `scan_units` = files; kernel-parsed partition values (`partition_scalar`). `src/table/predicate.rs` pushed filters as a conservative kernel `Predicate` (comparisons, IS NULL, IN, AND/OR/NOT; exact types only, no floats or collations). `src/table/scan.rs` `DeltaScan` (`ScanPushdown`: Inexact/Unsupported, the predicate recorded on the source) and `DeltaScanSource` (a pushed predicate re-lists through the kernel once per run; one unit per file through `ScanStream` with `FieldMapping::DeltaColumnMapping`, partition columns from the log, DVs via `DvInfo::get_row_indexes` on the log pool as a `RowSelection`, charged to the driver's reservation; runtime filters applied last to each batch through krishiv-exec's `RuntimeFilters::source_rows`; every kernel call of a run on the engine bound to the query's cancellation).
- Tests: `src/table/tests.rs` (predicate translation, partition values), `src/table/cache_tests.rs` (cache hits, misses, credential scope, `<tableNameOrPath>` forms), `src/table/oracle_tests.rs` (the read-path oracle tables through the executor: `every_readable_table_matches_spark`, `scan_shape_matches_kernel` against the kernel's `Scan::execute`, `delta_type_widening_read`, `catalog_managed_read_refused`, refused and edited protocols, pruning, projections), `src/engine/eval/tests.rs` (`delta_eval_parity`, the kernel's evaluator tests as a table), `src/engine/tests.rs` + `parquet_tests.rs` + `oracle_tests.rs` (checkpoint reads of the Spark 4.2 + delta-spark 4.4.1 tables in `tests/fixtures/delta-4.4.1/`: V1, V2 with a sidecar, multi-part; generator `tests/scripts/gen_delta_fixtures.py`; local temp tables: `engine_reports_written_file_size`, handlers, `delta_handler_cancellation`, `catalog_parse_limits`, `delta_log_paths_confined`, a kernel snapshot and scan metadata, with partition pruning and data skipping, through the assembled engine), `schema_tests.rs`, `data_tests.rs`.

Fuzz target `fuzz/fuzz_targets/delta_commit_json.rs` (one commit through `parse_json`, `read_json_files`, the `_last_checkpoint`/`.crc` limits and a kernel snapshot + scan; seeds from the oracle tables' logs).

## Not here yet (PR-08b part 2 left these; time travel waits for PB-07 `TableVersionSpec`, CDF read is PR-37e)
- Time travel and `@`: `DeltaTable::open_with` already takes a pinned version.
- CDF read (`table_changes`, `readChangeFeed`).
- Delta row-tracking columns: `features::row_tracking_refused` is wired once `MetadataCols` can name them.
- The pushed predicate prunes files only (not row groups: no `ScanRequest::predicate`).
- A catalog table whose log schema changed since it was registered is refused by the session (delta-spark reads
  the snapshot's schema on every query).
- Interrupting a query during analysis does not reach its planning-time Delta opens: the session registers the
  query with its cancel registry only after planning. The session's `DeltaReads` cancels its token when dropped (a
  dropped planning future); hooking it into the interrupt registry is a session-wide follow-up.
- The session names a Delta table in errors as a catalog table when its key holds a `.` (`delta_name`,
  `krishiv-session` `catalog/file_tables.rs`): a temporary view whose back-quoted name holds a `.` is named as a
  catalog table. Deciding from the entry's kind needs `FilesKey` to carry it (review finding 10, a follow-up).
- Arrays and maps under column mapping, `NOT NULL` columns (the analyzer makes file columns nullable) and inline DVs
  have no oracle table.

## Invariants
- `delta_kernel` with `default-features = false`; never `deltalake`.
- Kernel calls run only through `DeltaEngine::delta_blocking` (the log pool), never on a runtime worker; lazy kernel iterators are drained inside that call; puts are never raced against cancellation once started (`block_on_to_completion`); every handler call checks the query's cancellation first and confines its URLs to the table root.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028, 0037

## Test
`cargo nextest run -p krishiv-connectors-delta --no-tests=pass` · `cargo clippy -p krishiv-connectors-delta --all-targets -- -D warnings`
