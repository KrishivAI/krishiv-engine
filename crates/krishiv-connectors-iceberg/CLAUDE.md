# krishiv-connectors-iceberg

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code since:** P3-T022a (PR-06): iceberg-rust 0.11's file access over Krishiv's I/O, metadata loading, the
Spark type mapping and the bridged `inspect` tables. Catalog clients are PR-17 (T022b); scans and commits T026/T027.

## Purpose
Iceberg table connector. The only crate that depends on `iceberg` (iceberg-rust, Arrow 58).

## Layout
`src/lib.rs` the public items (`StorageRegistration`, `IcebergTable`) · `src/iceberg_v0/` (private; every item
that names `iceberg`): `storage.rs` iceberg's `Storage`/`StorageFactory` (`#[typetag::serde]`, tag `krishiv`)
over `krishiv-io`'s `IoContext`, the process registry of registration names, `iceberg::Runtime` on the
scheduler's I/O runtime, error mapping both ways (a Krishiv error keeps its class through iceberg-rust) ·
`types.rs` Iceberg → Spark types (Iceberg's `TypeToSparkType`, apache-iceberg 1.12.0) · `table.rs`
`IcebergTable` (load from a metadata location, schema, `snapshots`/`manifests` as Arrow 60) · `tests.rs`.
Fixture: `tests/fixtures/iceberg-1.12.0-spark-types.json` (source-only goldens,
`research/p3/t022a-iceberg-spark-types.md` in krishiv-internal).

## Invariants
- Isolated `iceberg` crate (Arrow 58) crossing to Arrow 60 via the C Data Interface; only values iceberg-rust returns
  as Arrow (`inspect` tables) cross, through `krishiv-arrow-bridge`; data batches never do (`design/P3-lakehouse.md`
  §5.2). No `iceberg` or Arrow 58 type in a public signature.
- Every byte iceberg-rust reads or writes goes through `IoContext` (`StorageRegistration`); `opendal` and
  `iceberg-storage-opendal` are banned in `deny.toml`. `StorageConfig` properties from a catalog are ignored: stores
  come from Krishiv's configuration (LH §5.6).
- A serialised `FileIO` carries only the registration name; an unknown name fails at `build`, never panics.
- iceberg-rust runs on the I/O runtime (`IoScheduler::runtime`), never on a compute worker.
- `#![forbid(unsafe_code)]`. No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE
  (iceberg-rust's own failures map to `INTERNAL_ERROR` naming its message until PR-17's catalog classes).

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028 (iceberg `skip-tree`, `anyhow` wrapper), 0029 (the bridge crate)

## Test
`cargo nextest run -p krishiv-connectors-iceberg --no-tests=pass` · `cargo clippy -p krishiv-connectors-iceberg --all-targets -- -D warnings`
