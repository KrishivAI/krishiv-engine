# Delta 4.4.1 oracle tables

Written by Spark 4.2.0 + delta-spark 4.4.1 with `tests/scripts/gen_delta_fixtures.py` (the docstring has the Docker
command). Never edit these files by hand; regenerate them. Hadoop's `.*.crc` side files are dropped; Delta's own
`_delta_log/*.crc` files are kept.

| Table | Checkpoint | Versions |
|---|---|---|
| `v1-checkpoint` | classic single file at version 3, `stats_parsed` | 0–4 (a DELETE after the checkpoint) |
| `v2-checkpoint` | `delta.checkpointPolicy = v2`: a JSON top-level checkpoint and one Parquet sidecar | 0–4 |
| `multipart-checkpoint` | classic, six parts (`spark.databricks.delta.checkpoint.partSize = 1`) | 0–3 |

Read-path tables (PR-08b; `gen_delta_fixtures.py /out scan`, run with `-e TZ=UTC` and the host's `/etc/passwd` and
`/etc/group` mounted read-only so Hadoop can name the container user):

| Table | Reader features | What it covers |
|---|---|---|
| `deletion-vectors` | `deletionVectors` | one file of 40,000 rows in five row groups of up to 8,332 rows (`parquet.block.size` 100,000; more rows than Krishiv's 8,192-row batch), then two DELETE statements: the first keeps every 100th row and the rows within three of each row-group and batch edge, the second (rewriting the DV) deletes the rows right at those edges (`row_groups`, `edges`, `deleted_at_edges` in `EXPECTED.json`; regenerated alone with `gen_delta_fixtures.py /out review`); delta-spark wrote on-disk DVs only (`dv_storage_types`: `u`), so inline DVs have no oracle table |
| `column-mapping-name`, `column-mapping-id` | none (reader version 2) | mode `name` / `id`, partitioned by `p`; a rename (`a` to `a2`), a nested rename (`s.x` to `s.x2`), a drop (`b`) and a new column of the dropped name |
| `type-widening` | `typeWidening` | `int` to `bigint` and `float` to `double` after a file was written in the old types |
| `partitioned` | none (1, 2) | partition columns `p string`, `n int`, `d date`, `ts timestamp` with NULLs, an empty string (read as NULL), `/`, `=`, `%`, a space and non-ASCII values |
| `timestamp-ntz` | `timestampNtz` | a `timestamp_ntz` data column and partition column |
| `variant` | `variantType` | an unshredded Variant column with a NULL |
| `vacuum-protocol-check` | `vacuumProtocolCheck` | a reader feature with no reader requirement |
| `variant-shredding` | `variantType`, `variantShredding-preview` | **refused** by Krishiv (no rows; `delta.enableVariantShredding`) |

`REFUSED.json` records the refused reader features delta-spark 4.4.1 cannot create, with its error:
`catalogManaged` (`DELTA_UNSUPPORTED_CATALOG_MANAGED_TABLE_CREATION`), coordinated commits (the `in-memory`
coordinator is not registered outside delta-spark's tests) and `geospatial` (`DELTA_GEOSPATIAL_NOT_SUPPORTED`). Those
rows are covered by protocols edited in unit tests (`src/features.rs`), never by a hand-written table.
`PROBE-unknown-reader-feature.json` is delta-spark's own refusal of an unknown reader feature (a commit adding
`krishivProbeFeature` to a table's protocol): the class, SQLSTATE and parameters Krishiv's refusal matches. `PROBE-two-unknown-reader-features.json` is the same
with two unknown features: delta-spark names them in the protocol's order, joined by `, `.

Each table's `EXPECTED.json` holds the rows Spark reads back (by `id`) and the `_delta_log` listing. The read-path tables also hold
`typed_rows` (each value with its kind: floats by `repr`, bytes as hex, a Variant as its two byte strings, timestamps
as the driver's naive ISO form in UTC), `schema`, `fields` and the last `protocol`. Consumers:
`crates/krishiv-connectors-delta/src/engine/oracle_tests.rs` (checkpoint read through the assembled engine, with the
pre-checkpoint commits removed so only the checkpoint can supply the state) and
`crates/krishiv-connectors-delta/src/table/oracle_tests.rs` (every read-path table through `DeltaTable`, against
`typed_rows` and against the kernel's own `Scan::execute`).

Session probes (PR-08b part 2; `gen_delta_fixtures.py /out session`, run as the read-path tables are, plus
`--conf spark.sql.warehouse.dir=/tmp/wh` so the managed-table probes have a writable warehouse). `SESSION.json`
records, for one partitioned table, what Spark + delta-spark does with each statement Krishiv's session handles or
refuses, with the table path elided as `<path>`:
- `CREATE TABLE d USING delta LOCATION` without a column list, its `DESCRIBE` rows and its `SELECT` rows;
- a column list that matches but omits `PARTITIONED BY` (`DELTA_CREATE_TABLE_WITH_DIFFERENT_PARTITIONING`), one that
  matches with it (accepted), and two that differ (`DELTA_CREATE_TABLE_SCHEME_MISMATCH`, with its parameters);
- `CREATE` over a missing or empty location, `CREATE` without a location, CTAS, `INSERT` and `ALTER` (all succeed in
  Spark; Krishiv refuses them until PR-37a);
- the unknown-reader-feature refusal through a catalog table (`tableNameOrPath` = `default.refused`) and through a path
  (`file:<path>`, a trailing slash dropped), and a corrupt commit file (`FAILED_READ_FILE.NO_HINT`, KD001, with the
  commit's `file://` URL).

`session-table` is that table as the probes found it (before their writes), for the session's tests. `session-appends`
is a table of three commits (a create and two appends); the session's snapshot-cache tests remove its
last commit and put it back while a query runs.
