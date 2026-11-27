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
`src/lib.rs` the public items (`StorageRegistration`, `IcebergTable` (`from_metadata`: a table whose metadata a catalog returned, no metadata file read; `with_rest_commit`: commits through a REST catalog's transport), `IcebergMetadata`, `IcebergTableProvider`, module `plan`, module `rest_commit` (PR-36c part 2: `RestCommitTransport` (`post_commit(body) -> TransportReply | TransportFailure { sent }`, `load(all_snapshots)`), JSON bytes only, no iceberg type), feature `testing`: `testing::write_table`, a small table for other crates' tests, `write_table_with`, the same table under any warehouse URL through the stores of a caller's path locator, `LocalIo`, `append` (rows through a provider's writer), `apply_commit` (a fake REST server's apply)) · `src/iceberg_v0/` (private; every item
that names `iceberg`): `storage.rs` iceberg's `Storage`/`StorageFactory` (`#[typetag::serde]`, tag `krishiv`)
over `krishiv-io`'s `IoContext`, the process registry of registration names, `iceberg::Runtime` on the
scheduler's I/O runtime, error mapping both ways (a Krishiv error keeps its class through iceberg-rust) ·
`types.rs` Iceberg → Spark types (Iceberg's `TypeToSparkType`, apache-iceberg 1.12.0) · `table.rs`
`IcebergTable` (load from a metadata location, schema, `snapshots`/`manifests` as Arrow 60, `snapshot(VersionSpec)`,
`plan`; crate-private `with_committer`, the `CommitTable` its providers append through) · `rest_commit.rs` (PR-36c part 2) `RestCommitTable`, the production `CommitTable` over a `RestCommitTransport`: the spec's `CommitTableRequest` (`identifier`, `requirements`, `updates` from iceberg's serde types) and the status mapping of `CommitOutcome`'s table (duplicate id `Snapshot already exists for` → unknown; stale texts → stale conflict; 409 → conflict; 500/502/503/504, any other 2xx than a parsable 200 (201, 202, 204: accepted, so it may have landed) and a sent request without answer → unknown, `Retry-After` named in the reason only; never sent, 401/403/404/422/other → `KRISHIV_ICEBERG_COMMIT_FAILED` naming status and type, never the server's message); tests `rest_commit_tests.rs` (one per row; the loop over a wire on `commit_tests`' catalog: plain, 409 rebase, stale 400, lost 200, duplicated, 401 first and after a 503, never sent) · `provider.rs` `IcebergTableProvider` (design note §3.1): a pinned snapshot's data files as the file list
of `krishiv-exec`'s `FileScan` (Parquet, Spark split planning), columns by field id with Java's positional fallback
for files without ids (`FieldMapping::FieldIdOr`; under `schema.name-mapping.default` by the mapping's names instead, `types.rs` `parse_name_mapping`, strict as Java: `names` required, no name twice at a level; PR-25c. Equality delete files ignore the mapping, as Java's delete loader does), identity-partition
columns constant in every file's spec read from the manifest (`FileConstants`, FC S4), a mixed-spec identity
column that a file of an identity spec lacks refused (footers of the files a scan reads, on its first batch), pushed
filters pruning files through `Pruner` when the scan is built (still applied above), deletes assigned (`DeleteIndex::for_data_file`)
to the files that survive pruning when the scan is built, `statistics` (`stats.rs`), `scan_units` and the scan's
`EXPLAIN` text (`Snapshot: <id>, DataFiles: <read>/<total>, DeleteFiles: [position=, dv=, equality=]`, through
`ScanPushdown::explain`, a krishiv-exec hook); with a committer and opened at the current version it declares `BATCH_WRITE`, `TRUNCATE`, `OVERWRITE_BY_FILTER` and `OVERWRITE_DYNAMIC`, and `writer` returns `sink.rs`'s sink (an append) or `overwrite.rs`'s over a `WriteTarget` (the committer, the metadata pinned at open, `krishiv.iceberg.write.maxOpenFiles` and `krishiv.iceberg.commit.maxIntervening` read at open), and `row_level` returns `row_level.rs`'s table; the scan `supports_take` (the file scan's, with the table's field mapping; identity-partition constants, lineage columns refused) and is restricted to the touched files for a rewrite; `types.rs` also builds the Arrow table schema carrying every Iceberg id ·
`lazy.rs` `DeleteState`: the deletes of one built scan, loaded on its first batch (`DeleteSource` wraps the file
scan's source, takes the query's pool in `global_state`, awaits the load, then applies equality deletes; position
deletes and DVs reach the file scan as `LazySelections`); charged to a `ConsumerHandle` of the query's pool, so a
refused reservation is `UNABLE_TO_ACQUIRE_MEMORY` and a dropped load unregisters · `stats.rs` row count (record counts
less applying position deletes and DVs, each once; equality deletes not subtracted), null counts and min/max from
bounds (no NaN-able float, no possibly truncated string), distinct counts (`StatSource::Sketch`) from the `ndv` of the pinned snapshot's Theta blob in the table metadata's statistics file (`puffin_stats.rs`: `ndv_by_field`, no Puffin I/O at plan time; `read_sketch` fetches a blob as a `CompactThetaSketch` for `statistics_file.rs`'s unions; Spark's `compute_table_stats` and Krishiv's appends write them), estimates only · `statistics_file.rs` (PR-54b) statistics on write: `sketch_columns` (top-level primitives but `uuid`, first 100), `TaskSketches` per write task (charged to the driver's consumer, refused → the write has none), `union_tasks` at finalize, `attempt_statistics` per commit attempt (rule 2 of design note `P3-T024-statistics-on-write.md` §2: the write's own sketch when the parent has no live rows, else the union with the parent's blob of the field of the same type, else the column left out; a Puffin file `<metadata dir>/<snapshot id>-<commit uuid>-<attempt>.stats`, blobs uncompressed, `ndv` the truncated estimate; QuickSelect-derived, not Alpha, review B1); never fails a write (`skipped`, a warning and the process counter `statistics_skipped_total`, not yet in the engine's metrics catalogue); switches `krishiv.iceberg.write.statistics.enabled` and the table property `krishiv.write.statistics.enabled`; tests `statistics_tests.rs` (columns, switches, a CTAS and a union through the sink read back by PR-54a's reader and the provider, rule 2, and the commit simulation: rebases, a resend, a refused update, failed Puffin writes, cleanup) · `plan/theta_value.rs` (PR-54a) an Arrow column's values as Theta input: the single-value serialisation of `bounds::encode` hashed as raw bytes (`datasketches`, this crate its only product user; the optimizer's q-error test has it as a dev-dependency (`KRISHIV_QERROR_NDV=theta`)), empty strings/binaries skipped as Java does, `uuid` not sketched · `manifest_cache.rs` decoded manifests as
`Arc<[LiveEntry]>` keyed by path, length and the inherited list fields, LRU under `krishiv.iceberg.manifestCache.bytes`
(0 disables), one per storage registration (`Registered`), in memory only · `data_file.rs` (PR-36a) a written
file's manifest entry: `DataLocations` (Java's `LocationProviders` through iceberg-rust's generators: `write.data.path`,
object storage with hash directories; the deprecated path properties honoured, a divergence), `file_name`,
`partition_tuple` (transformed values through `plan::bounds` to iceberg literals), `data_file` (an iceberg `DataFile`
from size, rows and `FileMetrics`; sort order 0 unless claimed); `write_metrics.rs` `WriteMetricsConfig` (Java's
`MetricsConfig.from`); tests `data_file_tests.rs` (Java's partition paths and hash directories, a built file through a
manifest and back) · `append.rs` (PR-36a) an append's metadata, as Java's `SnapshotProducer`/`MergeAppend`:
`write_new_manifests` once per write (each spec's files through `rolling.rs`, entries with the write's snapshot id and inherited
sequence numbers; each path pushed onto the caller's `written` list before its file exists, so `sink.rs` deletes what a failed or dropped write left), `prepare_attempt` per commit attempt (the list through `merge.rs`'s
`arrange`: new manifests reset to unassigned, then the base's data manifests, then its delete manifests;
sequence number `last-sequence-number` + 1; v3 `first-row-id`/`added-rows`; summary with `total-*`,
`manifests-created`/`-kept`/`-replaced`, `engine-name`, `krishiv.write-id`/`krishiv.query-id`; requirements
`assert-table-uuid`/`assert-ref-snapshot-id`, updates `add-snapshot`/`set-snapshot-ref`, then `set-statistics` for the attempt's own statistics file when `NewManifests::statistics` describes the whole snapshot, PR-54b); tests `append_tests.rs`
(committed as a catalog applies a request, read back through SQL and iceberg-rust's planning, a stale attempt refused
and rebuilt, v3 lineage) · `rolling.rs` (PR-36a, owner 2026-11-15) the new manifests' layout as Java's `writeDataManifests`: groups
`max(1, min(4, round_half_up(n / 10,000)))` of `ceil(n / groups)` (`manifestWriterCount`, `divide`; Java's pool is
the host's worker threads, Krishiv fixes 4), each rolled as `RollingManifestWriter` (every 250 entries, at
`commit.manifest.target-size-bytes`), the length being iceberg-rust's exact encoded size of the entries so far, found
by encoding prefixes in memory (250 entries, then the line's crossing of the target, halving as a fallback; the
settling probe's bytes written as the manifest; Java's length is its flushed, compressed one, a divergence in where
the split falls); tests `rolling_tests.rs` (against a transcription of Java's loop with even and uneven entries, the
14,999/15,000/25,000/45,001 group boundaries, names, a failed later manifest's paths in the written list, rolled
manifests committed and merged; ignored `probe_cost_on_a_10k_group`)
· `merge.rs` (PR-36a, design note §6.2) manifest merging as Java's `MergeAppend`
(`MergingSnapshotProducer.apply`, `ManifestMergeManager`, `BinPacking.ListPacker.packEnd`, apache-iceberg 1.12.0):
`commit.manifest.{target-size-bytes,min-count-to-merge}`, `commit.manifest-merge.enabled` (`MergeConfig`, garbage
`KRISHIV_UNSUPPORTED_FEATURE`); base manifests without added or existing files left out (`shouldKeep`); data and
delete manifests merged apart, by spec id highest first, `pack_end` lookback 1, the bin holding the first manifest
merged only at `min-count-to-merge`, others at 2; merged entries: older `DELETED` dropped, own `ADDED` kept with
inherited sequence numbers, the rest `EXISTING` with explicit sequence numbers, inherited snapshot id and (v3) explicit
`first_row_id`, partition tuples widened to the current type (`filter.rs` `widen_partition`); **first attempt only** (later attempts list the unmerged manifests; Java re-merges every attempt);
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
`commit.retry.*` budget and Java's `Tasks` backoff; `KRISHIV_ICEBERG_COMMIT_FAILED` / `_STATE_UNKNOWN`; §7.4 cleanup (statistics files as lists: kept when the committed metadata names them); a known refusal of a request carrying `set-statistics` resent once at once without it (PR-54b).
The production `CommitTable` is `rest_commit.rs`'s (krishiv-catalogs attaches it to declared REST tables with `option.write = true`); the production clock is `sink.rs`'s `IoClock`. Tests `commit_tests.rs` (DST: `TestCatalog`
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
`commit.rs`'s `commit_validated` on every rebase (`commit_append` passes `Validation::none()`, no walk); new deletes
for replaced data files (`validateNoNewDeletesForDataFiles`, part 2a: per intervening `overwrite`/`delete` snapshot a
`DeleteIndex` of its live delete files above S0's sequence number, each replaced file judged at S0's). Tests `validate_tests.rs` (the walk, each check over a hand-built history,
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
retry never reuses a failed attempt's names; `with_sketches`/`take_sketches` (PR-54b) update the task's Theta sketches per batch; tests `write_tests.rs`
(tuples against iceberg-rust's transforms over the rows read back, footers' field ids, rollover, the open-file cap,
an append of the written files read back, failed writes and an injected upload failure leaving no file, a dropped
abort, two attempts, memory pressure and the reservation released, list and map ids from a sliced batch)
· `lineage.rs` (PR-36b unit 2a, design note §8 item 3) v3 row lineage: `_row_id` (2147483540) and
`_last_updated_sequence_number` (2147483539) as the spec's "Row Lineage" rules: `IcebergTableProvider::lineage_scan`
(crate-private, the seam unit 2's CoW rewrite reads through) appends both to the table schema; its inner file scan
reads the stored values by reserved id (null when the file lacks them), the file's `first_row_id` and data sequence
number as `FileConstants` and the row position as the row id, under the delete-aware source, and `LineageSource`
derives the values (stored non-null wins, else `first_row_id + _pos` / data sequence number; a file without
`first_row_id` reads null for both); `DataFileWriter::with_lineage` writes both columns under their ids, optional longs,
metrics in the table's default mode (Java's `MetricsConfig.columnMode(id)`); v1/v2 tables refuse both
(`KRISHIV_UNSUPPORTED_FEATURE`); tests `lineage_tests.rs` (a rewrite round trip, stored/updated/inserted rows, deletes
first, a pre-upgrade file, a real v2 → v3 upgrade, a file of several row groups in several splits under a pruning filter,
v1/v2, the derivation against a model) · `filter.rs` (PR-36b part 2a, design note §7.3 step 4a.2) an `OverwriteFiles`'s
removals on every attempt's base, as Java's `ManifestFilterManager` (apache-iceberg 1.12.0): data files by path, then
delete files per entry (`DeleteKey`: path, referenced data file, content offset; one Puffin file holds many DVs), DVs of removed data files (`removeDanglingDeletesFor`) and delete files older than every data
file (`dropDeleteFilesOlderThan`); a manifest without a hit kept as is, a hit one rewritten from **live entries only**
(the hit `DELETED` under the write's id, the rest `EXISTING` with their own ids, explicit sequence numbers, v3
`first_row_id`), duplicates counted once (`deleted-duplicate-files`); "Missing required files to delete" for a file no
longer live (the loop's conflict); Java's `UpdateMetrics` for the removed files (a DV sized by its blob) merged with the
added ones by `overwrite_summary` (changed partitions over both, partition summaries in Java's key order); rewritten
paths kept in `NewManifests::merged` for §7.4 cleanup; since part 2b a row filter (`deleteByRowFilter`: a file, data or delete, removed when every row matches by `plan::strict`; a data file matched in part `FilterError::Partial`, raised as `KRISHIV_ICEBERG_OVERWRITE_PARTIAL_FILE` (SQLSTATE 22000, deterministic, not the retryable conflict), Java's "Cannot delete file where some, but not all, rows match filter"; a delete file so matched kept) and dropped partitions (spec id and tuple, compared by value at the spec's current type through `partition_key`, as Java's `PartitionSet`: a promoted column's old 4-byte tuples equal new 8-byte ones, also in `validate.rs`'s partition scope; a rewritten manifest writes old `int`/`float` tuples at the promoted `long`/`double`, `widen_partition`, which `merge.rs`'s `copy_entries` applies to every entry it copies too (review 36b2b round 2, finding 1; test `merge_tests::a_merge_widens_pre_promotion_partition_tuples`)); tests `filter_tests.rs` · `overwrite.rs` (PR-36b part 2b) INSERT OVERWRITE as Spark's `SparkWrite`: truncate = overwrite by `alwaysTrue` (dynamic under the write option `overwrite-mode=dynamic`), overwrite by filter (`WriteRequest::overwrite_filter_pushdown`, exact or refused), dynamic overwrite = `ReplacePartitions` (the written files' partitions, the whole table when its spec has no field, `replace-partitions=true`, operation `overwrite`, an empty write commits nothing); validation only under the write option `isolation-level`, from `validate-from-snapshot-id` (absent: the whole history, as Java, the message naming the option), on every attempt; tests `overwrite_tests.rs` (truncate, identity/bucket/truncate/day static overwrites, a straddling file refused, metrics-only drops, dynamic partitioned/unpartitioned/empty, DVs of replaced partitions, v3 row ids kept, the rebase re-applying the filter without validation, serializable from a snapshot id, the missing-option message) · `row_level.rs` (PR-36b part 2a,
design note §8 item 2) `IcebergRowLevel`, copy-on-write DELETE/UPDATE/MERGE as Spark's `SparkWrite` CoW
(`newOverwrite()`): `RowIdScan`s over the provider's scan (the lineage scan on v3), the rewriter an `IcebergWriteSink`
whose finalize commits an `OverwriteFiles` (the touched files and their DVs removed, `request.write_id`; an empty
change commits an `overwrite` snapshot) validated on every rebase from S0 with the read filter (`RowLevelWrite::read_filter_pushdown`,
none: every row): `serializable` (default of `write.{delete,update,merge}.isolation-level`) added data files and new
deletes for the replaced files, `snapshot` the deletes only; on v3 the rows must carry `_row_id` and
`_last_updated_sequence_number` (else `KRISHIV_UNSUPPORTED_FEATURE`); `SparkMergeCardinality`, no output rows; tests
`cow_tests.rs` (the session planner's find and rewrite queries driven by hand over `sink_tests`' catalog: DELETE on
unpartitioned and partitioned v2, UPDATE, MERGE, v3 lineage kept, an empty DELETE, a DV and an old position delete removed
with their file, a foreign append under both levels, a foreign removal, `rebase_after_foreign_manifest_merge`, a cancelled
statement, `take` (`_metadata` refused), one Puffin file holding two files' DVs, a foreign row delta under both
levels, a v3 rewrite merged on the first attempt keeping row ids); `cow_fault_tests.rs` (the overwrite through
`commit_tests`' scripted catalog: unknown answer then a missing removal, unknown answer then landed, a known failure, a still-unknown answer resending the same attempt, an unknown answer then a rebase re-filtering) · `row_delta.rs` (PR-36c part 1, design note §8 item 4) merge-on-read DELETE/UPDATE/MERGE as Spark's `SparkPositionDeltaWrite` (`RowDelta`, notes 36c): `row_level_mode` (`write.{delete,update,merge}.mode`, default `copy-on-write`, Java's "Unknown row-level operation mode"), `MergeOnReadWrite` (the internal entry PR-37d's merge-on-read contract will adapt to; v1 refused, "Deletes are supported in V2 and above"): per task `MorTask::delete(file reference, position)` and `insert(rows)` (v3: the lineage columns required), `finish` writes the deletes, on v3 one Puffin file of DVs per task absorbing each file's DV or every position delete the scan assigned it, on v2 one position delete file per data file absorbing its file-scoped ones; the file-scoped absorbed files removed through `filter.rs` (a `DELETE` skips one no longer live, `Removals::ignore_missing_deletes`); `commit` merges two tasks' DVs of one file into `merged-dvs-<id>-1.puffin` (Java's `mergeDVs`), operation `delete`/`append`/`overwrite` (`BaseRowDelta.operation`; an empty row delta `overwrite`); `abort(outputs)` deletes finished tasks' files a statement does not commit (Java's `PositionDeltaBatchWrite.abort`), `finish` dropped midway leaves no delete file (an armed `Written`), validation from S0 per notes 36c §1.5 (references removed, added data under serializable, added deletes and missing absorbed files for UPDATE/MERGE, concurrent DVs always); wired to the session by `mor_sink.rs` (below) · `mor_sink.rs` (PR-36c part 2) the `Sink` of a merge-on-read statement: `is_merge_on_read_input` recognizes the rows ending with the row id columns, `IcebergRowLevel::rewriter` routes them here (an `UPDATE`/`MERGE` of a v3 table whose rows lack the lineage columns refused before anything runs), one `MorTask` per driver with a keep/abort handshake (a task told not to keep, or whose `finalize` never comes, deletes its files), `finalize` commits through `MergeOnReadWrite::commit`; an `UPDATE`'s replacement rows get the required-field check (`NOT_NULL_ASSERT_VIOLATION`); tests `mor_sink_tests.rs` (v2/v3 DELETE and UPDATE, the empty statement, a charge over a small pool limit, a failing driver, a cancelled statement, two drivers on one file, the v3 UPDATE refusal) · `delete_write.rs` (PR-36c) the delete files: `dv_puffin` (Java's Puffin bytes: blobs from offset 4, compact JSON footer, `created-by` Krishiv's; byte-equal to the `dvs` fixture), `dv_file` (Puffin entry: blob offset/length, whole-file size, `referenced_data_file`), `write_position_deletes` (Parquet `file_path`/`pos`, sorted, `write.delete.parquet.*` over `write.parquet.*`, equal `file_path` bounds, no counts), `partition_struct`; tests `mor_tests.rs` (v2/v3 DELETE, partitioned locations, one Puffin per task, UPDATE/MERGE incl. `append`, empty, old DV and file-scoped position delete absorbed, partition-scoped kept, the v2 → v3 model case, two tasks' DVs merged, concurrent DV at both levels, foreign append/delete file/overwrite rows of the table, rebase onto v3 refusing position deletes, refusals) and `mor_fault_tests.rs` (known failure deletes the delete and data files, unknown then conflict keeps them, lost response landing, an aborted task, a `finish` dropped at many points, an aborted statement) · `sink.rs` (PR-36a) `IcebergWriteSink`, an append as `krishiv-exec`'s `FileWriteSink`: one `DataFileWriter` per
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
a schema mismatch) · `deletes.rs` reads each position
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
manifest, plus float → double with NaN and decimal(9,2) → decimal(18,2); `iceberg_schema_evolution_matrix`); `azure_tests.rs` (feature `cloud`: a table written to Azurite
over `abfss://` with the account key, loaded and scanned with only a bearer token; a token for another audience
refused; run by `tests/scripts/object_store_emulators.sh`).
`src/plan/` (public, Krishiv-owned types only; design note `P3-T026-T027-iceberg-scan-and-commit.md` §2): `types.rs`
`IcebergType`, `SchemaField` (primitive columns reachable through structs), `PartitionSpecInfo`/`Transform` ·
`snapshot.rs` `VersionSpec` (connector-local until PR-07's `TableVersionSpec`), resolution with Iceberg's error texts
(FC S6), `TableSnapshot` · `entry.rs` manifest-list and manifest entries, `inherit` (FC B2: sequence numbers,
snapshot ids, v3 `first_row_id`), `DataFileEntry`/`DeleteFileEntry`, `FileScanTask`, `ScanPlan`, `ManifestFilter` ·
`deletes.rs` `DeleteIndex`/`TaskDeletes`, Java's `DeleteFileIndex` rules (a DV replaces position deletes, FC S1;
position deletes by path or by spec and partition, never table-wide, M1; equality deletes strictly newer, global for an
unpartitioned spec); an equality delete is skipped for a data file when their column metrics rule out every key (Java's `canContainEqDeletesForFile`, PR-25d step 2, on through `DeleteIndex::with_fields`: metrics trusted as pruning trusts them under the table's `write.metadata.metrics.*` modes, bounds only for exact types, not floats, strings, binaries or uuids; `eq_narrow_tests.rs`; `delete_tests` writes `id` metrics, so the model tests cover it end to end) · `dv.rs` the `deletion-vector-v1` blob codec on `roaring` (Java's checks, order and texts, FC S2; the encoder writes Java's layout: dense keys, run containers, byte-checked for array containers only, the run/array boundary may differ, PR-36c)
· `bounds.rs` single-value serialisation decoded by stored width (review-1 S10) · `metrics.rs` `WriterMetrics`, the
`write.metadata.metrics.*` modes (`none`/`counts`: bounds untrusted, `truncate(n)`, `full`; default `truncate(16)`) · `transform.rs` partition transforms
of literals (Murmur3 bucket, truncate, year/month/day/hour) · `strict.rs` (PR-36b part 2b) the overwrite row filter: `Pred` (Iceberg semantics, from Spark's `PushdownExpr` with `not_null` guards on `<>`/`NOT startswith`), `RowFilter::rows_must_match`/`rows_might_match` on the residual after the partition (strict projections per transform, the pre-1970 time fix) as Java's `StrictMetricsEvaluator`/`ManifestFilterManager`; tests `strict_tests.rs` · `prune.rs` `Pruner`, the might-match evaluator over
summaries, partition tuples and metrics (`PruneMode::Conservative` for T027 validation, review-1 S11) · `tests.rs`,
`prop_tests.rs` (`prune_never_skips_a_matching_row`).
Type goldens: `tests/fixtures/iceberg-1.12.0/types.json` at the repository root, from a real Spark 4.1.0 +
iceberg-spark-runtime 1.12.0 run (Iceberg's `SparkSchemaUtil.convert` per schema, `tests/scripts/gen_iceberg_type_goldens.py`),
checked by `iceberg_v0/type_golden_tests.rs`; column defaults are a documented refusal.

## Write conformance and round trip (PR-53b)
`write_conformance_tests.rs`: two `#[ignore]`d tests run by `tests/scripts/iceberg_write_conformance.sh` (Docker; nightly job
`iceberg-write-conformance`): Krishiv's append sink writes `unpart`/`part`/`v3`/`v3part` into a Hadoop-catalog layout,
Spark 4.1.0 + Iceberg 1.12.0 (`gen_iceberg_write_readback.py`) reads and appends on top, Krishiv compares and reads Spark's
result. `roundtrip_tests.rs`: two `#[ignore]`d steps of `tests/scripts/iceberg_roundtrip.sh` (nightly `iceberg-roundtrip`):
Spark makes `mor2`/`mor3`/`evolved` (position deletes, a DV, a promoted and a renamed column, a spec added), Krishiv
reads them, appends, runs copy-on-write DELETE/UPDATE and, on `mor2`/`mor3`, merge-on-read DELETE (v2 also UPDATE) (`cow_tests`' `Dml`, over `sink_tests`' `Catalog::with_metadata` on
Spark's metadata), Spark lists Krishiv's delete files and time travels to Krishiv's snapshots and runs merge-on-read DML on top, Krishiv reads the
result. Pending, not stubbed: MERGE through SQL. Concurrent writers: `race_k` (two Krishiv writers pinned at one base) and `race_s` (Krishiv pinned before Spark's `DELETE`, committing onto it; `Dml::open_at`), `race_tables`/`check_races`. See
`tests/scripts/iceberg_write_conformance.README.md`.

## Invariants
- Isolated `iceberg` crate (Arrow 58) crossing to Arrow 60 via the C Data Interface; only values iceberg-rust returns
  as Arrow (`inspect` tables) cross, through `krishiv-arrow-bridge`; data batches never do (`design/P3-lakehouse.md`
  §5.2). No `iceberg` or Arrow 58 type in a public signature.
- Every byte iceberg-rust reads or writes goes through `IoContext` (`StorageRegistration`), except the scratch
  `FileIO::new_with_memory` that `rolling.rs` encodes manifest prefixes into (memory only, never a store); `opendal` and
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
  a commit found as a replay under another id (`clean_committed`). A replay found on the base before any
  request (`drive`) is the exception: nothing of it can be in the table, so it deletes the files it wrote and answers
  with the found metadata.
  A 409 leaves the write unresolved until the own-key search has run on a successful load (a duplicated request lands
  and answers 409). A `commit_append` dropped or cancelled after its first request is an unknown outcome too: the
  caller (`sink.rs`'s finalize, whose guard is disarmed before the loop) must not delete the write's files. Only
  v2 → v3 is rebased; Java reuses manifests across any upgrade (a divergence toward refusal).
- An overwrite's removals (`filter.rs`) are re-applied to **every** attempt's base, never reused from an earlier one;
  a rewritten manifest carries only live entries (an older `DELETED` entry copied as `EXISTING` would bring a removed
  file back), and a base manifest is never deleted. A file to remove that is not live in the base is a conflict, never
  skipped. Rewritten manifests are cleaned up as merged ones are.
- An overwrite's row filter removes a file only when every row is proven to match (strict projection or strict
  metrics, `plan::strict`); every unknown keeps the file, which fails the commit for a data file ("some, but not
  all, rows match"), never removes rows the filter keeps. An overwrite filter that does not translate exactly, or
  holds a part the strict side cannot evaluate (`Pred::unsupported`: `LIKE '%a%'`, `x < NaN`, a struct under
  `IS NULL`), is refused; an unknown leaf or a column a rebase's schema lacks may match, never "no row matches".
  **Deliberate divergence from Java** (review 36b2b B2): a missing null count means "may hold nulls" and a missing
  NaN count of a `float`/`double` column "may hold NaN" for `<`, `<=`, `=` and `IN` (Java's `canContainNulls` /
  `canContainNaNs` say no, so Java may delete NaN rows of a file written without NaN counts; Krishiv keeps the file,
  which then fails with the partial-match error).
- A row-level rewrite of a v3 table writes `_row_id` and `_last_updated_sequence_number` for every row it moves; a
  rewrite that does not carry them is refused, never new row ids for untouched rows (design note §8 item 3).
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
- **Manifest probe memory** (`rolling.rs`): a probe holds its cloned entries and the encoded bytes (in the writer and
  the scratch store until deleted), and the search keeps the reaching probe's bytes, so up to about twice the
  target size (8 MiB default) plus the entries on the driver, uncharged, as iceberg-rust's own manifest buffering is.
- **Row lineage (PR-36b unit 2a):** a lineage scan reading a lineage column refuses `_metadata` columns
  (`KRISHIV_UNSUPPORTED_FEATURE`: the row position is read as the row id; so does a row id with `_metadata`); a table
  with a user column named `_row_id` or `_last_updated_sequence_number` is refused (Krishiv's own rule, uncited); filters on the lineage columns are not
  pushed. The Iceberg scan `supports_take` since part 2a; `in_files` stays false, so no late fetch is planned over Iceberg yet. Whether Spark hides or refuses the columns on v1/v2 tables is not verified
  (its `SparkTable` source was not supplied); Krishiv refuses. A file whose manifest leaves its data sequence number
  null reads a null `_last_updated_sequence_number` where none is stored.
- **Row-level DML (PR-36b part 2a):** the session's planner (`krishiv-session` `table_commands/row_level.rs`) does not
  select the lineage columns, so DML on a v3 table through SQL is refused until it carries them (the scans expose them);
  it passes no read filter yet (the conflict filter is every row, wider than Spark's) and has no MERGE (PR-37c). The
  filter reads every manifest with live files (Java prunes by the removed files' partitions first), one at a time
  on every attempt (the scan fetches with a concurrency of 8). An empty change still validates (Spark commits with no
  validation when the optimizer removed the scan, `SparkWrite.java:473-492`), so an empty serializable DELETE fails
  against a concurrent append: a divergence toward refusal.
- **Merge-on-read memory (PR-36c, review 36c1 m6):** `MorTask` charges the `iceberg-mor` consumer (v3 only: kept in `TaskOutput::held` until the commit returns) 8 bytes per deleted
  position (an upper bound of the `RoaringTreemap`s, per batch in `settle`), the absorbed positions, and each delete file
  before it is built (a Puffin file's bytes, a v2 file's positions at 8 bytes each), failing with the memory error class.
  `delete_write.rs` still builds each file whole in memory and writes it with one `put` (no multipart upload), so the
  charge is an estimate, not the allocator's count.
- **Row-level DML through merge-on-read (PR-36c part 2):** the statement reports a hit count of 0 (the count is not known
  before the commit; Iceberg's `output` is `None`); its conflict filter is every row (`RowLevelWrite::new(.., None)`);
  an `UPDATE` of a v3 table through SQL is refused up front (the session does not select the lineage columns).
- **Merge-on-read v2 → v3 first attempt (pre-existing, shared with 36a appends; review 36c1 m5):** the first commit
  attempt is built on the base pinned at open and `check_rebase` runs only on a rebase, so an upgrade to v3 that does
  not move the branch lets a v2 position delete file (or, for UPDATE/MERGE, data files without lineage) land on a v3
  table, a spec "must not" (spec.md:1357). Rows stay correct: v3 readers apply existing position deletes and the next
  v3 merge-on-read absorbs the file-scoped ones. Java narrows the window by refreshing before every attempt. Follow-up:
  load and check the format version before the first attempt (the commit-loop backlog).
- **Merge-on-read drops (review 36c1 M3):** a `MorTask::finish` dropped midway deletes its delete files and finished
  data files in the background, and a `put` dropped in flight deletes its file once the store is done; a data file
  whose close is dropped after the store completed it is `write.rs`'s documented limit (an unreferenced file).
- **Expire cleanup memory (PR-36e, review of #983):** `maintenance/expire.rs` `cleanup` holds the paths every
  removed snapshot reached, every kept snapshot reached and the decoded manifests' paths in memory, uncharged to any
  pool (each manifest is capped by `krishiv.catalog.maxMetadataBytes`, the sum is not), as Java's
  `ReachableFileCleanup` holds its sets. Follow-up: charge the sets to the procedure's pool or walk in bounded batches.
- **`remove_orphan_files` (PR-36e unit 2b, `maintenance/orphans.rs` + `paths.rs`):** `classify` is the pure decision (valid set vs listing, `older_than`, hidden paths, `prefix_mismatch_mode`); `remove` loads with `checked_load_located`, gathers the valid set (`expire::reachable` over every snapshot, metadata files, statistics, `version-hint.text`) and lists through `IcebergTable::list_under`. Not done yet: the nested-table check (`KRISHIV_ICEBERG_ORPHAN_LOCATION_SHARED` is defined, `ProcedureTables` cannot list a namespace); listed keys are the store's key text (`Path`'s own form), not percent-decoded; no S3/MinIO listing test yet; the valid set and listing are held in memory, uncharged (as the expire limit above).
- **Expire and concurrent refs (review of #983):** a snapshot removal asserts every ref as planned, `main` included
  (`maintenance/expire.rs` `request`), so a concurrent rollback or a ref moved onto a removed snapshot conflicts and the
  expire plans again. Two costs: a concurrent append to `main` conflicts too, so a busy table can use up
  `commit.retry.num-retries` (`KRISHIV_ICEBERG_EXPIRE_FAILED`); and a ref created after planning under a new name is not
  covered (no REST requirement names a ref that did not exist), so the server drops that new ref with the snapshot and
  its files are deleted. Iceberg 1.12.0 has the same window: `UpdateRequirements` asserts nothing for
  `remove-snapshots`.
- **Expire and gzip metadata (review of #983):** `maintenance/expire.rs` `checked_load` parses the `metadata.json` an
  answer names as plain JSON, so a table written with `write.metadata.compression-codec=gzip` (`*.gz.metadata.json`) is
  refused with the "snapshot list that differs from its metadata file" text until gzip metadata is read. Refused at
  the first load, nothing is committed or deleted. A table switched to gzip after that load can still take the commit:
  the reload after it is then refused, so the snapshots are gone, no file is deleted (orphans for
  `remove_orphan_files`) and the procedure returns that error.
- `statistics()` takes no filter: the row estimate sums every live file, not the files left after pruning.
- A deletion vector is read at the length its manifest declares (`deletes.rs` `read_dv`) without the
  `krishiv.catalog.maxMetadataBytes` cap the metadata files get; its memory is charged to the query (the decode's transient copy aside; review of the
  PR-25b+c carry, 2026-11-16).

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028 (iceberg `skip-tree`, `anyhow` wrapper), 0029 (the bridge crate)

## Test
`cargo nextest run -p krishiv-connectors-iceberg --no-tests=pass` (`--features cloud` adds the Azurite test: offline
without `KRISHIV_TEST_AZURE_OAUTH_ENDPOINT`, a failure when it is set without the tokens) · `cargo clippy -p krishiv-connectors-iceberg --all-targets -- -D warnings`
- `benches/theta_update.rs` criterion bench of the Theta value path (`plan::bench_estimate`, doc-hidden) per type, 1M values; `cargo bench -p krishiv-connectors-iceberg --bench theta_update`
