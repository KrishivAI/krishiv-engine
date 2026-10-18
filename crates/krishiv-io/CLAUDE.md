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
`scan.rs` ScanStream · `write.rs` FileWriter · `commit.rs` `SaveMode` + `DirectoryWrite` (Spark's save modes on a
directory, `_SUCCESS`, the empty-output file; files written under final names, never renamed) · `io/` IoScheduler,
IoContext (reads, uploads, listing), coalescing, metadata cache,
multipart `Upload`, `FaultStore` (fault injection + unscheduled-request detection), `pace.rs` prefix pacing
(`StorePolicy`, per-store; single-flight footer reads live in `cache.rs`).
`parquet/` (P2-T080a) `ParquetFormat` over arrow-rs's `ParquetPushDecoder`, no custom decoders: `meta.rs` footer
(one tail read, checked against the file size, engine types, exact-only row-group statistics, calendar markers) ·
`decode.rs` the sans-IO decoder (projection mask, row-group selections, row index, unsigned widening, millisecond
timestamps as microseconds) · `write.rs` (P2-T082a) the encoder: `ArrowWriter` over a drained buffer, Spark codec names
and file suffixes, `parquet.block.size`/`page.size`/`enable.dictionary`/`writer.version`, page indexes always,
`outputTimestampType` (INT96 refused), `datetimeRebaseModeInWrite`, `org.apache.spark.version` footer marker · `tests.rs`.
Interop fixtures (pyarrow codecs/pages, INT96, unsigned, DuckDB, Spark 4.2): `tests/fixtures/parquet/`
(`tests/scripts/gen_parquet_fixtures.py`, checked by `krishiv-testkit/tests/parquet_interop.rs`); TPC-H SF1 parity
with pyarrow: `tests/scripts/tpch_parquet_checksums.py` + the ignored `tpch_sf1_reads_as_pyarrow_reads_it`; written
files read back by pyarrow, DuckDB and Spark 4.2: `tests/fixtures/parquet/written/`
(`tests/scripts/check_parquet_written.py` + `parquet_written.scala`, checked by
`krishiv-testkit/tests/parquet_write_interop.rs`); fuzz
target `fuzz/fuzz_targets/parquet_read.rs` (seeds: `tests/scripts/fuzz_seed_parquet.py`).

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

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028, 0007, 0026, 0037, 0038; the P2-T117 design note

## Test
`cargo nextest run -p krishiv-io --no-tests=pass` · `cargo clippy -p krishiv-io --all-targets -- -D warnings` ·
conformance: `cargo nextest run -p krishiv-testkit --test file_format_mock` · bench: `cargo bench -p krishiv-io --bench io`
