# krishiv-connectors-iceberg

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code since:** P3-T022a (PR-06): iceberg-rust 0.11's file access over Krishiv's I/O, metadata loading, the
Spark type mapping and the bridged `inspect` tables. Scan planning since P3-T026 (PR-25a). Catalog clients are PR-17
(T022b); the `TableProvider` over append-only tables since PR-25a step 2; delete application and caches are
PR-25b–d; commits T027.

## Purpose
Iceberg table connector. The only crate that depends on `iceberg` (iceberg-rust, Arrow 58).

## Layout
`src/lib.rs` the public items (`StorageRegistration`, `IcebergTable`, `IcebergTableProvider`, module `plan`) · `src/iceberg_v0/` (private; every item
that names `iceberg`): `storage.rs` iceberg's `Storage`/`StorageFactory` (`#[typetag::serde]`, tag `krishiv`)
over `krishiv-io`'s `IoContext`, the process registry of registration names, `iceberg::Runtime` on the
scheduler's I/O runtime, error mapping both ways (a Krishiv error keeps its class through iceberg-rust) ·
`types.rs` Iceberg → Spark types (Iceberg's `TypeToSparkType`, apache-iceberg 1.12.0) · `table.rs`
`IcebergTable` (load from a metadata location, schema, `snapshots`/`manifests` as Arrow 60, `snapshot(VersionSpec)`,
`plan`) · `provider.rs` `IcebergTableProvider` (design note §3.1): a pinned snapshot's data files as the file list
of `krishiv-exec`'s `FileScan` (Parquet, Spark split planning), columns by field id with Java's positional fallback
for files without ids (`FieldMapping::FieldIdOr`; refused under a name mapping until 25c), identity-partition
columns constant in every file's spec read from the manifest (`FileConstants`, FC S4), a mixed-spec identity
column that a file of an identity spec lacks refused (footers of those files read at open), a snapshot with delete
files refused (25b), pushed filters pruning files through `Pruner` when the scan is built (still applied above);
`types.rs` also builds the Arrow table schema carrying every Iceberg id · `scan.rs` metadata → `TableSnapshot` (refs read from the metadata's serialisation: iceberg-rust has no
accessor and its `snapshot_for_ref` panics on a dangling ref), manifest list and manifests fetched through the
`FileIO` and parsed with iceberg-rust's public parsers into `plan` values, up to 8 manifests at once, each
metadata file, manifest list and manifest capped by `krishiv.catalog.maxMetadataBytes` before parsing
(`KRISHIV_CATALOG_LIMIT_EXCEEDED`; decoded entries are not charged to memory until 25d), a manifest list parsed
with the format version its Avro header names (snapshots from before a v1 → v2 upgrade), `main` added for
metadata without refs (v1) · `tests.rs`,
`scan_tests.rs` (manifests written by iceberg-rust's `ManifestWriterBuilder`, transforms against iceberg-rust's,
an in-memory-catalog table planned and time-travelled), `provider_tests.rs` (SQL through the executor: identity
partitions from the manifest, files without ids, delete files refused, time travel, the 22 TPC-H queries over
Iceberg equal to the same Parquet files), `table_tests.rs` (`catalog_parse_limits`, v1 refs and lists across an
upgrade, the table's metrics modes in a scan, the mixed-spec refusal), `baseline_tests.rs` (ignored: the §2.3 planning baseline, 1,000
manifests × 100 entries, timed as fetch/parse/map/prune).
`src/plan/` (public, Krishiv-owned types only; design note `P3-T026-T027-iceberg-scan-and-commit.md` §2): `types.rs`
`IcebergType`, `SchemaField` (primitive columns reachable through structs), `PartitionSpecInfo`/`Transform` ·
`snapshot.rs` `VersionSpec` (connector-local until PR-07's `TableVersionSpec`), resolution with Iceberg's error texts
(FC S6), `TableSnapshot` · `entry.rs` manifest-list and manifest entries, `inherit` (FC B2: sequence numbers,
snapshot ids, v3 `first_row_id`), `DataFileEntry`/`DeleteFileEntry`, `FileScanTask`, `ScanPlan`, `ManifestFilter` ·
`bounds.rs` single-value serialisation decoded by stored width (review-1 S10) · `metrics.rs` `WriterMetrics`, the
`write.metadata.metrics.*` modes (`none`/`counts`: bounds untrusted, `truncate(n)`, `full`; default `truncate(16)`) · `transform.rs` partition transforms
of literals (Murmur3 bucket, truncate, year/month/day/hour) · `prune.rs` `Pruner`, the might-match evaluator over
summaries, partition tuples and metrics (`PruneMode::Conservative` for T027 validation, review-1 S11) · `tests.rs`,
`prop_tests.rs` (`prune_never_skips_a_matching_row`).
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
- Pruning keeps every file that may hold a matching row under Spark's semantics: a float column that may hold NaN
  is never pruned by bounds, a string/binary upper bound at the column's truncation length (from the table's
  metrics modes) is a prefix, uuid bounds and a transform of a literal of another type than the source are unused, a bound that does
  not decode is dropped (never the file), unknown transforms and columns keep. Time-travel errors carry Iceberg's
  `IllegalArgumentException` texts (class `KRISHIV_TIME_TRAVEL_NOT_FOUND`, SQLSTATE 42K0E).
- `#![forbid(unsafe_code)]`. No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE
  (iceberg-rust's own failures map to `INTERNAL_ERROR` naming its message until PR-17's catalog classes).

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028 (iceberg `skip-tree`, `anyhow` wrapper), 0029 (the bridge crate)

## Test
`cargo nextest run -p krishiv-connectors-iceberg --no-tests=pass` · `cargo clippy -p krishiv-connectors-iceberg --all-targets -- -D warnings`
