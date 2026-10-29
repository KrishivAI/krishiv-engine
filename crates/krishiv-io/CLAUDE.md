# krishiv-io

**Layer:** L2 (may depend only on crates in L2 or lower; `cargo xtask layering`).  
**Real code since:** P2-T117 (`FileFormat` framework); the Parquet reader is P2-T080a, the writer P2-T082a.

## Purpose
Object store and file format I/O: the `FileFormat` trait (sans-IO `Decoder`/`Encoder`), the `IoScheduler`
(every byte read or written), the file-metadata cache, Spark's split planner, the shared `SchemaMapper`,
typed format options, and the scan/write drivers (`ScanStream`, `FileWriter`) the executor wraps.

## Layout (P2-T117)
`format.rs` FileFormat/FileMetadata/Decoder/Encoder/Capabilities + `AsyncDecoder`/`PushReadAt` bridge ·
`request.rs` ScanRequest, PushdownExpr, RowSelection, ColumnPath, `project_schema`/`project_batch` · `stats.rs` statistics and
sound `prune` (TriState) · `options.rs` OptionSchema/FormatOptions (unknown-option warning, strict mode,
secrets) · `schema_map.rs` SchemaMapper (case rules, field ids, Delta column mapping, widening, predicate
re-typing) · `split.rs` Spark split planning (goldens: `tests/fixtures/spark-4.2.0/file-splits.json`) ·
`scan.rs` ScanStream · `runtime_filter/` (P2-T163): `RuntimeFilter` (snapshots numbered by generation; the
atomic generation is only a "changed?" hint), `ScanFilter` (lowering cached per scan level), `KeyFilter` and
`KeyFilterBuilder` (min/max, value set, Bloom filter), `normalize_key_bytes` and `widens_to` (the single key
normalization that producers and consumers share), `RuntimeBloom` (Parquet's split-block Bloom filter over hashes),
and a loom model of publication · `write.rs` FileWriter · `commit.rs` `SaveMode` + `DirectoryWrite` (Spark's save modes on a
directory, `_SUCCESS`, the empty-output file; files written under final names, never renamed) · `io/` IoScheduler,
IoContext (reads, uploads, listing), coalescing, metadata cache, cloud store registry (`stores.rs`: S3, GCS and
Azure stores built per session from Spark `fs.*` keys, behind the `cloud` feature),
multipart `Upload`, `FaultStore` (fault injection + unscheduled-request detection), `pace.rs` prefix pacing
(`StorePolicy`, per-store; single-flight footer reads live in `cache.rs`).
`parquet/` (P2-T080a) `ParquetFormat` over arrow-rs's `ParquetPushDecoder`, no custom decoders: `meta.rs` footer
(one tail read, checked against the file size, engine types, exact-only row-group statistics, calendar markers) ·
`decode.rs` the sans-IO decoder (projection mask, row-group selections, row index, unsigned widening, millisecond
timestamps as microseconds, runtime filters re-read at row-group boundaries) · `write.rs` (P2-T082a) the encoder: `ArrowWriter` over a drained buffer, Spark codec names
and file suffixes, `parquet.block.size`/`page.size`/`enable.dictionary`/`writer.version`, page indexes always,
`outputTimestampType` (INT96 refused), `datetimeRebaseModeInWrite`, `org.apache.spark.version` footer marker · `tests.rs`.
Interop fixtures (pyarrow codecs/pages, INT96, unsigned, DuckDB, Spark 4.2): `tests/fixtures/parquet/`
(`tests/scripts/gen_parquet_fixtures.py`, checked by `krishiv-testkit/tests/parquet_interop.rs`); TPC-H SF1 parity
with pyarrow: `tests/scripts/tpch_parquet_checksums.py` + the ignored `tpch_sf1_reads_as_pyarrow_reads_it`; written
files read back by pyarrow, DuckDB and Spark 4.2: `tests/fixtures/parquet/written/`
(`tests/scripts/check_parquet_written.py` + `parquet_written.scala`, checked by
`krishiv-testkit/tests/parquet_write_interop.rs`); fuzz
target `fuzz/fuzz_targets/parquet_read.rs` (seeds: `tests/scripts/fuzz_seed_parquet.py`).

## Key files
Paths are relative to this crate.
- `src/format.rs` `FileFormat`, `Decoder`, `Encoder`, `Capabilities`; `src/request.rs` scan request and pushdown types
- `src/stats.rs` statistics and sound pruning (`TriState`); `src/split.rs` Spark split planner
- `src/scan.rs` `ScanStream`; `src/write.rs` `FileWriter`; `src/commit.rs` save modes and `_SUCCESS`
- `src/io/mod.rs` `IoContext` and `IoScheduler`, the only path to object stores; `src/io/cache.rs` metadata cache
- `src/schema_map.rs` `SchemaMapper`; `src/options.rs` typed format options
- `src/runtime_filter/mod.rs` runtime filters and their key normalization
- `src/parquet/mod.rs` `ParquetFormat`; `src/parquet/decode.rs` and `write.rs` the reader and writer
- `src/scan_gate.rs` `ScanGate`, admission of the decode working sets of a query's scan streams (ADR 0007)
- `src/parquet/thrift_guard.rs` structural check of a footer before arrow-rs decodes it
- `src/csv/mod.rs` CSV format

## Extension points
- New format: implement `FileFormat` in a module like `src/parquet/`, route all bytes through `IoContext`, and instantiate `krishiv_testkit::file_format_conformance!`.

## Invariants
- Formats never touch `object_store`: all I/O goes through `IoContext` (counters, retries, `if_match`/
  `if_unmodified_since`, cache, budget). `ParquetObjectReader/Writer` are banned (`cargo xtask source-guard`).
- `Decoder::step` / `Encoder::write` are sync, bounded ≈ one batch, never block; every await is drop-safe.
- Pruning keeps a superset of matching rows (exact bounds only; NaN/±0/inexact/collated → Unknown).
- Scan memory is reserved (required bytes with waiting, prefetch with `try_grow`); the pool is empty at the end.
- New formats must pass `krishiv_testkit::file_format_conformance!`.
- Strings are valid UTF-8: invalid bytes in a Parquet `STRING` column are `INVALID_UTF8_STRING`, never passed through.
- The parquet crate's `object_store` feature stays off; Parquet bytes come only from `IoContext`.
- object_store and parquet versions from the workspace (Arrow 60 family). `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.
- Row-level drops (late materialisation, runtime filters at row level) use only what
  `ScanRequest::row_level` allows: with `RowLevel::Prefix` (a filter conjunct above may raise) only that
  never-NULL prefix and no runtime filter, so the filter above sees every row it could raise on. Whole units
  (row groups, pages) are still pruned with the full predicate.
- Runtime filters only drop rows their producer would drop: key values are compared after `normalize_key_bytes`
  (lossless casts to the key type only; any other column type is not filtered); consumers never wait for a filter.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028, 0007, 0026, 0037, 0038; the P2-T117 design note

## Tests
`cargo nextest run -p krishiv-io --no-tests=pass` · `cargo clippy -p krishiv-io --all-targets -- -D warnings` ·
conformance: `cargo nextest run -p krishiv-testkit --test file_format_mock` · bench: `cargo bench -p krishiv-io` (targets `io`, `coalesce` and `runtime_filter`)
