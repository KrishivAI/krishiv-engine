# krishiv-connectors-iceberg

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code since:** P3-T022a (PR-06): iceberg-rust 0.11's file access over Krishiv's I/O, metadata loading, the
Spark type mapping and the bridged `inspect` tables. Scan planning since P3-T026 (PR-25a). Catalog clients are PR-17
(T022b); the `TableProvider` over append-only tables since PR-25a step 2; position deletes and deletion
vectors since PR-25b; equality deletes since PR-25c; deletes after pruning charged to the query, stats, `EXPLAIN` rows
and the manifest cache since PR-25d; commits T027.

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
for files without ids (`FieldMapping::FieldIdOr`; under `schema.name-mapping.default` by the mapping's names instead, `types.rs` `parse_name_mapping`, strict as Java: `names` required, no name twice at a level; PR-25c. Equality delete files ignore the mapping, as Java's delete loader does), identity-partition
columns constant in every file's spec read from the manifest (`FileConstants`, FC S4), a mixed-spec identity
column that a file of an identity spec lacks refused (footers of the files a scan reads, on its first batch), pushed
filters pruning files through `Pruner` when the scan is built (still applied above), deletes assigned (`DeleteIndex::for_data_file`)
to the files that survive pruning when the scan is built, `statistics` (`stats.rs`), `scan_units` and the scan's
`EXPLAIN` text (`Snapshot: <id>, DataFiles: <read>/<total>, DeleteFiles: [position=, dv=, equality=]`, through
`ScanPushdown::explain`, a krishiv-exec hook); `types.rs` also builds the Arrow table schema carrying every Iceberg id ·
`lazy.rs` `DeleteState`: the deletes of one built scan, loaded on its first batch (`DeleteSource` wraps the file
scan's source, takes the query's pool in `global_state`, awaits the load, then applies equality deletes; position
deletes and DVs reach the file scan as `LazySelections`); charged to a `ConsumerHandle` of the query's pool, so a
refused reservation is `UNABLE_TO_ACQUIRE_MEMORY` and a dropped load unregisters · `stats.rs` row count (record counts
less applying position deletes and DVs, each once; equality deletes not subtracted), null counts and min/max from
bounds (no NaN-able float, no possibly truncated string), estimates only · `manifest_cache.rs` decoded manifests as
`Arc<[LiveEntry]>` keyed by path, length and the inherited list fields, LRU under `krishiv.iceberg.manifestCache.bytes`
(0 disables), one per storage registration (`Registered`), in memory only · `deletes.rs` reads each position
delete file once per scan (`file_path IN` row-group pruning, rows of other files dropped, order not trusted) and each
DV by range, into one `RowSelection` per data file (unbounded, so a wrong `record_count` never drops rows; positions
past the end delete nothing, as Java); positions charged per batch · `eq_deletes.rs` reads each equality
delete file once per scan (its `equality_ids` by field id through the read schema, so promoted keys compare at the
table's type) into a key set in Arrow's row format (one `Rows` buffer and a `hashbrown` index hashed with foldhash; null equals null; NaN equals NaN and -0.0 differs from 0.0, as
Java's `Comparators`; a field of a null struct is null); `EqProjection` (applied by `DeleteSource`) drops the
rows whose key is in a set of their data file (a run's keys encoded once per distinct column list and key types), found by the per-file constant column `_krishiv_iceberg_file` (the
file's index in the scan, through `FileConstants`); the equality columns, that column and any `_metadata` the query
did not ask for are projected away, and no limit is pushed below it · `scan.rs` metadata → `TableSnapshot` (refs read from the metadata's serialisation: iceberg-rust has no
accessor and its `snapshot_for_ref` panics on a dangling ref), manifest list and manifests fetched through the
`FileIO` and parsed with iceberg-rust's public parsers into `plan` values, up to 8 manifests at once, each
metadata file, manifest list and manifest capped by `krishiv.catalog.maxMetadataBytes` before parsing
(`KRISHIV_CATALOG_LIMIT_EXCEEDED`; decoded entries are not charged to memory: the cache bounds what is kept), decoded
manifests read from and kept in the manifest cache, a manifest list parsed
with the format version its Avro header names (snapshots from before a v1 → v2 upgrade), `main` added for
metadata without refs (v1), delete files assigned to tasks through `DeleteIndex` · `tests.rs`,
`scan_tests.rs` (manifests written by iceberg-rust's `ManifestWriterBuilder`, transforms against iceberg-rust's,
an in-memory-catalog table planned and time-travelled), `provider_tests.rs` (SQL through the executor: identity
partitions from the manifest, files without ids, a missing delete file, time travel, the 22 TPC-H queries over
Iceberg equal to the same Parquet files), `table_tests.rs` (`catalog_parse_limits`, v1 refs and lists across an
upgrade, the table's metrics modes in a scan, the mixed-spec refusal), `baseline_tests.rs` (ignored: the §2.3 planning baseline, 1,000
manifests × 100 entries, timed as fetch/parse/map/prune), `delete_tests.rs` (snapshots with chosen sequence
numbers: deletes through SQL, `delete_application_matches_model`, DV corruption, missing delete file, Java's
assignment errors, equality deletes by sequence number and partition, null keys, a missing equality delete file;
PR-25d: pruned files read no delete file, the delete state charged and cancel-safe, equality deletes through
`with_file_splits`, the row estimate, `EXPLAIN`, the manifest cache). `evolution_tests.rs`
(snapshots written by hand with one manifest per writing schema, as iceberg-rust 0.11 cannot promote or rename:
`promoted_column_filter_returns_all_rows`, the iceberg-rust #3482 shape, int → long with 4-byte bounds in the old
manifest, plus float → double with NaN and decimal(9,2) → decimal(18,2); `iceberg_schema_evolution_matrix`).
`src/plan/` (public, Krishiv-owned types only; design note `P3-T026-T027-iceberg-scan-and-commit.md` §2): `types.rs`
`IcebergType`, `SchemaField` (primitive columns reachable through structs), `PartitionSpecInfo`/`Transform` ·
`snapshot.rs` `VersionSpec` (connector-local until PR-07's `TableVersionSpec`), resolution with Iceberg's error texts
(FC S6), `TableSnapshot` · `entry.rs` manifest-list and manifest entries, `inherit` (FC B2: sequence numbers,
snapshot ids, v3 `first_row_id`), `DataFileEntry`/`DeleteFileEntry`, `FileScanTask`, `ScanPlan`, `ManifestFilter` ·
`deletes.rs` `DeleteIndex`/`TaskDeletes`, Java's `DeleteFileIndex` rules (a DV replaces position deletes, FC S1;
position deletes by path or by spec and partition, never table-wide, M1; equality deletes strictly newer, global for an
unpartitioned spec) · `dv.rs` the `deletion-vector-v1` blob codec on `roaring` (Java's checks, order and texts, FC S2)
· `bounds.rs` single-value serialisation decoded by stored width (review-1 S10) · `metrics.rs` `WriterMetrics`, the
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
- Deletes are never skipped: a missing or unreadable delete file fails the query; deletes that cannot be assigned or
  a corrupt DV fail with Java's texts (class `KRISHIV_ICEBERG_INVALID_DELETES`, SQLSTATE KD001). An equality delete
  whose partition tuple, or its data file's, cannot be read is refused (`KRISHIV_UNSUPPORTED_FEATURE`), never
  offered to every file of the spec (review N4).
- `#![forbid(unsafe_code)]`. No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE
  (iceberg-rust's own failures map to `INTERNAL_ERROR` naming its message until PR-17's catalog classes).

## Known limits (PR-25d review)
- **Equality-delete narrowing not ported:** Java's `DeleteFileIndex.canContainEqDeletesForFile` (skip an equality
  delete whose key bounds cannot overlap the data file's) is not applied; every strictly newer equality delete of the
  partition (or global) is assigned. Correct (more deletes read, none dropped); deferred to PR-25d step 2, which needs
  the Java source at `apache-iceberg-1.12.0` (it could not be fetched in step 1; not ported from memory).
- **Delete files re-read per built scan:** each `build` loads its own `DeleteState`, so a self-join or CTE reading a
  table twice reads and charges its shared delete files twice, and every distributed task re-reads the global
  (unpartitioned-spec) equality deletes. Design §3.3's "read once per scan" holds per built source. Follow-up: a
  per-provider (`Weak`) or per-worker cache of loaded key sets and selections keyed by delete path.
- **Planning memory:** decoded manifest entries are not charged to a query reservation while a plan runs (`open` has no
  query context); the manifest cache bounds only what is kept.
- `statistics()` takes no filter: the row estimate sums every live file, not the files left after pruning.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028 (iceberg `skip-tree`, `anyhow` wrapper), 0029 (the bridge crate)

## Test
`cargo nextest run -p krishiv-connectors-iceberg --no-tests=pass` · `cargo clippy -p krishiv-connectors-iceberg --all-targets -- -D warnings`
