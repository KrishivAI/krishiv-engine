# krishiv-io

**Layer:** L2 (may depend only on crates in L2 or lower; `cargo xtask layering`).  
**Real code since:** P2-T117 (`FileFormat` framework); Parquet arrives with P2-T080a/T082a.

## Purpose
Object store and file format I/O: the `FileFormat` trait (sans-IO `Decoder`/`Encoder`), the `IoScheduler`
(every byte read or written), the file-metadata cache, Spark's split planner, the shared `SchemaMapper`,
typed format options, and the scan/write drivers (`ScanStream`, `FileWriter`) the executor wraps.

## Layout (P2-T117)
`format.rs` FileFormat/FileMetadata/Decoder/Encoder/Capabilities + `AsyncDecoder`/`PushReadAt` bridge ·
`request.rs` ScanRequest, PushdownExpr, RowSelection, ColumnPath, `project_schema` · `stats.rs` statistics and
sound `prune` (TriState) · `options.rs` OptionSchema/FormatOptions (unknown-option warning, strict mode,
secrets) · `schema_map.rs` SchemaMapper (case rules, field ids, Delta column mapping, widening, predicate
re-typing) · `split.rs` Spark split planning (goldens: `tests/fixtures/spark-4.2.0/file-splits.json`) ·
`scan.rs` ScanStream · `write.rs` FileWriter · `io/` IoScheduler, IoContext, coalescing, metadata cache,
multipart `Upload`, `FaultStore` (fault injection + unscheduled-request detection).

## Invariants
- Formats never touch `object_store`: all I/O goes through `IoContext` (counters, retries, `if_match`/
  `if_unmodified_since`, cache, budget). `ParquetObjectReader/Writer` are banned (`cargo xtask source-guard`).
- `Decoder::step` / `Encoder::write` are sync, bounded ≈ one batch, never block; every await is drop-safe.
- Pruning keeps a superset of matching rows (exact bounds only; NaN/±0/inexact/collated → Unknown).
- Scan memory is reserved (required bytes with waiting, prefetch with `try_grow`); the pool is empty at the end.
- New formats must pass `krishiv_testkit::file_format_conformance!`.
- object_store and parquet versions from the workspace (Arrow 60 family). `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028, 0007, 0026, 0037, 0038; the P2-T117 design note

## Test
`cargo nextest run -p krishiv-io --no-tests=pass` · `cargo clippy -p krishiv-io --all-targets -- -D warnings` ·
conformance: `cargo nextest run -p krishiv-testkit --test file_format_mock` · bench: `cargo bench -p krishiv-io --bench io`
