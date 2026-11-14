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
`plan`; crate-private `with_committer`, the `CommitTable` its providers append through) · `provider.rs` `IcebergTableProvider` (design note §3.1): a pinned snapshot's data files as the file list
of `krishiv-exec`'s `FileScan` (Parquet, Spark split planning), columns by field id with Java's positional fallback
for files without ids (`FieldMapping::FieldIdOr`; under `schema.name-mapping.default` by the mapping's names instead, `types.rs` `parse_name_mapping`, strict as Java: `names` required, no name twice at a level; PR-25c. Equality delete files ignore the mapping, as Java's delete loader does), identity-partition
columns constant in every file's spec read from the manifest (`FileConstants`, FC S4), a mixed-spec identity
column that a file of an identity spec lacks refused (footers of the files a scan reads, on its first batch), pushed
filters pruning files through `Pruner` when the scan is built (still applied above), deletes assigned (`DeleteIndex::for_data_file`)
to the files that survive pruning when the scan is built, `statistics` (`stats.rs`), `scan_units` and the scan's
`EXPLAIN` text (`Snapshot: <id>, DataFiles: <read>/<total>, DeleteFiles: [position=, dv=, equality=]`, through
`ScanPushdown::explain`, a krishiv-exec hook); with a committer and opened at the current version it declares `BATCH_WRITE` (appends only) and `writer` returns `sink.rs`'s sink over a `WriteTarget` (the committer, the metadata pinned at open, `krishiv.iceberg.write.maxOpenFiles` read at open); `types.rs` also builds the Arrow table schema carrying every Iceberg id ·
`lazy.rs` `DeleteState`: the deletes of one built scan, loaded on its first batch (`DeleteSource` wraps the file
scan's source, takes the query's pool in `global_state`, awaits the load, then applies equality deletes; position
deletes and DVs reach the file scan as `LazySelections`); charged to a `ConsumerHandle` of the query's pool, so a
refused reservation is `UNABLE_TO_ACQUIRE_MEMORY` and a dropped load unregisters · `stats.rs` row count (record counts
less applying position deletes and DVs, each once; equality deletes not subtracted), null counts and min/max from
bounds (no NaN-able float, no possibly truncated string), estimates only · `manifest_cache.rs` decoded manifests as
`Arc<[LiveEntry]>` keyed by path, length and the inherited list fields, LRU under `krishiv.iceberg.manifestCache.bytes`
(0 disables), one per storage registration (`Registered`), in memory only · `data_file.rs` (PR-36a) a written
file's manifest entry: `DataLocations` (Java's `LocationProviders` through iceberg-rust's generators: `write.data.path`,
object storage with hash directories; the deprecated path properties honoured, a divergence), `file_name`,
`partition_tuple` (transformed values through `plan::bounds` to iceberg literals), `data_file` (an iceberg `DataFile`
from size, rows and `FileMetrics`; sort order 0 unless claimed); `write_metrics.rs` `WriteMetricsConfig` (Java's
`MetricsConfig.from`); tests `data_file_tests.rs` (Java's partition paths and hash directories, a built file through a
manifest and back) · `append.rs` (PR-36a) an append's metadata, as Java's `SnapshotProducer`/`MergeAppend`:
`write_new_manifests` once per write (one manifest per spec, entries with the write's snapshot id and inherited
sequence numbers; no rolling at the target size), `prepare_attempt` per commit attempt (the list through `merge.rs`'s
`arrange`: new manifests reset to unassigned, then the base's data manifests, then its delete manifests;
sequence number `last-sequence-number` + 1; v3 `first-row-id`/`added-rows`; summary with `total-*`,
`manifests-created`/`-kept`/`-replaced`, `engine-name`, `krishiv.write-id`/`krishiv.query-id`; requirements
`assert-table-uuid`/`assert-ref-snapshot-id`, updates `add-snapshot`/`set-snapshot-ref`); tests `append_tests.rs`
(committed as a catalog applies a request, read back through SQL and iceberg-rust's planning, a stale attempt refused
and rebuilt, v3 lineage) · `merge.rs` (PR-36a, design note §6.2) manifest merging as Java's `MergeAppend`
(`MergingSnapshotProducer.apply`, `ManifestMergeManager`, `BinPacking.ListPacker.packEnd`, apache-iceberg 1.12.0):
`commit.manifest.{target-size-bytes,min-count-to-merge}`, `commit.manifest-merge.enabled` (`MergeConfig`, garbage
`KRISHIV_UNSUPPORTED_FEATURE`); base manifests without added or existing files left out (`shouldKeep`); data and
delete manifests merged apart, by spec id highest first, `pack_end` lookback 1, the bin holding the first manifest
merged only at `min-count-to-merge`, others at 2; merged entries: older `DELETED` dropped, own `ADDED` kept with
inherited sequence numbers, the rest `EXISTING` with explicit sequence numbers, inherited snapshot id and (v3) explicit
`first_row_id`; **first attempt only** (later attempts list the unmerged manifests; Java re-merges every attempt);
merged paths in `NewManifests::merged` for §7.4 cleanup; tests `merge_tests.rs` (100 appends into one manifest,
count/switch/target, packEnd order, two specs, review-1 P2's `DELETED` entry, v3 row ids, empty manifests dropped
and delete manifests merged apart, v1, first attempt only; every merged manifest read back through
`Manifest::parse_avro`)
· `commit.rs` (PR-36a, design note §7, Tier U) the append commit loop `commit_append` over the crate-private
`CommitTable` (`commit(requirements, updates) -> CommitOutcome { Committed, Conflict { stale }, Unknown, Failed }`,
`load(all_snapshots)`; the REST status mapping documented on `CommitOutcome`, implemented by the catalog client
later) and `CommitClock` (time and jitter, injected): an unknown outcome probed with `snapshots=all`
(`commit.status-check.*`), the same attempt resent while the branch stays at its base, "not applied" final only once
the branch moved; own-key search (snapshot id or `krishiv.write-id`) before every rebase; per-request outcomes;
rebase across v2 → v3 only (other format changes and a replaced table `KRISHIV_ICEBERG_COMMIT_CONFLICT`); Iceberg's
`commit.retry.*` budget and Java's `Tasks` backoff; `KRISHIV_ICEBERG_COMMIT_FAILED` / `_STATE_UNKNOWN`; §7.4 cleanup.
No production `CommitTable` yet (PR-17's catalog clients); the production clock is `sink.rs`'s `IoClock`. Tests `commit_tests.rs` (DST: `TestCatalog`
applies requests as `append_tests` does, with scripted faults per request and foreign writers at chosen points,
on virtual time; the §10 36a cases; merged first attempts committed, rebased, failed and left unknown; the sweep
merging in one schedule in three; table invariants and the store's files checked after each case)
· `validate.rs` (PR-36b unit 1, design note §7.3 steps 2–3) commit validation as Java's `MergingSnapshotProducer`
(apache-iceberg 1.12.0): `Validation` (checks, isolation level, conflict filter, files removed / referenced / DV'd,
`max_intervening`), `walk` from the branch head to S0 along parent ids (Java's "Cannot determine history ..." when it
does not end at S0; past `krishiv.iceberg.commit.maxIntervening` a conflict), `read_history` (only the manifests an
intervening snapshot added, `ADDED`/`DELETED` entries; `EXISTING` only in the delete manifests the added delete files
check reads, as Java's `DeleteFileIndex`), `check`: added data files (`append`,
`overwrite`; serializable only), added delete files (`overwrite`, `delete`; live entries with data sequence number above
S0's; an equality delete judged only by its `equality_ids` columns' metrics), added
DVs (`overwrite`, `delete`, `replace`), removed files by path (`overwrite`, `replace`, `delete`, `delete` skippable
for references), through `Pruner` in conservative mode (every unknown a match), Java's message texts; run by
`commit.rs`'s `commit_validated` on every rebase (`commit_append` passes `Validation::none()`, no walk). No caller
validates yet (the operations are unit 2). Tests `validate_tests.rs` (the walk, each check over a hand-built history,
the conservative evaluator, histories written by iceberg-rust's writers, the loop with `commit_tests`' catalog: a
validation read error with outcomes known, after an unknown answer and while unresolved, a second rebase, a stale
answer). Open items (filter bound by name to Sn's schema, the CoW check set, two unverified texts) in `validate.rs`'s
module docs
· `write.rs` (PR-36a) `DataFileWriter`, one task's data files for an append (distribution `none`): batches of the
table's Arrow schema (nested types retyped onto the field-id schema; another shape is `INTERNAL_ERROR`), each row's
partition key from the default spec (`partition_value`; an all-`void` spec does no per-row work), rows taken per key,
one open Parquet file per key (at most `max_open_files`, default 100, oldest-opened closed first), rollover at a
multiple of 1,000 rows once `write.target-file-size-bytes` is reached (Java's `RollingFileWriter`), Parquet settings
from `write.parquet.{compression-codec,row-group-size-bytes,page-size-bytes}` (gzip when unset, Java's fallback; the level is not applied;
`write.format.default` other than parquet, `uuid`/`fixed`/`variant` columns refused), Java's file names and partition
directories, a `WrittenDataFile` per file (`file_metrics`, no `split_offsets`); open uploads and the encoders' memory
charged to one `Reservation`; `finish` failing, `abort` (cancellation-safe: a dropped abort leaves the rest to drop)
and drop delete the finished files; `TaskIds::attempt_id` must be unique per attempt (Java's `taskAttemptId`), so a
retry never reuses a failed attempt's names; tests `write_tests.rs`
(tuples against iceberg-rust's transforms over the rows read back, footers' field ids, rollover, the open-file cap,
an append of the written files read back, failed writes and an injected upload failure leaving no file, a dropped
abort, two attempts, memory pressure and the reservation released, list and map ids from a sliced batch)
· `sink.rs` (PR-36a) `IcebergWriteSink`, an append as `krishiv-exec`'s `FileWriteSink`: one `DataFileWriter` per
driver on a task of its own behind a bounded channel (`Blocked` while full; not counted as the query's blocking work,
whose counter is private to krishiv-exec, so an aborting task may outlive the query briefly), `combine` finishes,
`finalize` gathers every driver's files, tells each task to keep or delete them, then `write_new_manifests` and
`commit_append` against the base pinned at open with `request.write_id` (else a fresh UUID) and the query id; data
files and manifests deleted on any failure before the commit loop starts, its §7.4 rules after; the input schema
checked against the table's columns (`KRISHIV_UNSUPPORTED_FEATURE`), a NULL in a required position
`NOT_NULL_ASSERT_VIOLATION` with Spark's `AssertNotNull` path; attempt ids from `unique_u64` (a per-process counter
under a keyed bijective mix); `IoClock` (wall time, sleeps on the I/O runtime); tests `sink_tests.rs` (appends into an
unpartitioned and a partitioned table by two drivers read back through SQL, a second append, the write id in the
summary, a NULL in a required column and a cancelled write leaving no file, commit or memory, the capability check,
a schema mismatch) · `delete_write.rs` (PR-36c unit 1, design note §8 item 4) merge-on-read delete writers as Java's `BaseDVFileWriter`/`SortingPositionOnlyDeleteWriter`/`PositionDeleteWriter` (apache-iceberg 1.12.0): `PositionDeletes` (positions per data path with spec and partition), `load_previous` (Spark's `shouldRewrite`: v3 every non-equality delete, v2 `file` the file-scoped ones, v2 `partition` none; read by `deletes.rs`' readers), `DeleteFileWriter` (`DeleteFormat` by format version: v3 one Puffin per task of `deletion-vector-v1` blobs, the union with previous deletes, DVs and file-scoped files rewritten; v2 Parquet `file_path`/`pos` sorted, `write.delete.granularity` default `file` as Spark, single-path bounds and no counts, no `referenced_data_file` as Java, rolling at `write.delete.target-file-size-bytes`; v1 refused; Java's `-deletes` names; every file deleted on failure), `DeleteWriteResult`; no caller yet (the DML path); tests `delete_write_tests.rs` (read back through `deletes::load`, `PuffinReader` offsets, manifests, granularities, rolling, codec fallback, refusals, an upload failure leaving no file) · `deletes.rs` reads each position
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
`with_file_splits`, the row estimate, `EXPLAIN`, the manifest cache). `dst_tests.rs` (PR-25d step 2: `Flaky`, a local store whose read faults are a function of a seed, the request and its attempt, so a case replays in any request order: transient failures before or mid-body (manifest list, manifests, data and delete files) retried to the model's rows or failing as `KRISHIV_OBJECT_STORE_REQUEST_FAILED`; a data or delete file vanished after planning fails `FAILED_READ_FILE.FILE_NOT_EXIST` naming it, never wrong rows; `delete_tests`' `history` model under faults). `evolution_tests.rs`
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
unpartitioned spec); an equality delete is skipped for a data file when their column metrics rule out every key (Java's `canContainEqDeletesForFile`, PR-25d step 2, on through `DeleteIndex::with_fields`: metrics trusted as pruning trusts them under the table's `write.metadata.metrics.*` modes, bounds only for exact types, not floats, strings, binaries or uuids; `eq_narrow_tests.rs`; `delete_tests` writes `id` metrics, so the model tests cover it end to end) · `dv.rs` the `deletion-vector-v1` blob codec on `roaring` (Java's checks, order and texts, FC S2)
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
- Commits (`commit.rs`) are **never retried blind**: a request whose outcome is unknown is probed, and only a resend of
  the same attempt (same snapshot id, base and manifest list) or a rebase after the branch moved off its base follows;
  every attempt of a write carries one snapshot id. A write that ends with any outcome unresolved is
  `KRISHIV_ICEBERG_COMMIT_STATE_UNKNOWN` (40003) and **deletes nothing**; a committed write never deletes a data file.
  A write that fails after any of its requests had an unknown answer also deletes nothing (its error says the files
  were kept): an attempt settled as "not applied" because the branch left its base lands if the branch is rolled
  back to that base (review-1 pass 2 P1; on v2+ the sequence number usually refuses it, on v1 nothing does); nor does
  a commit found as a replay under another id (`clean_committed`).
  A 409 leaves the write unresolved until the own-key search has run on a successful load (a duplicated request lands
  and answers 409). A `commit_append` dropped or cancelled after its first request is an unknown outcome too: the
  caller (`sink.rs`'s finalize, whose guard is disarmed before the loop) must not delete the write's files. Only
  v2 → v3 is rebased; Java reuses manifests across any upgrade (a divergence toward refusal).
- Manifest merging (`merge.rs`) runs on a write's first attempt only; a merged manifest is derived from that
  attempt's base and never reused by a later attempt. Its merged manifests are cleaned up with the attempt's files
  (deleted when the committed list does not name them or the write fails with every outcome known, kept otherwise);
  a base manifest is never deleted, only left out of the list. Merged `EXISTING` entries carry explicit sequence
  numbers, the adding snapshot's id and (v3) `first_row_id`, so files keep their sequence numbers and row ids.
- `#![forbid(unsafe_code)]`. No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE
  (iceberg-rust's own failures map to `INTERNAL_ERROR` naming its message until PR-17's catalog classes).

## Known limits (PR-25d review)
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
