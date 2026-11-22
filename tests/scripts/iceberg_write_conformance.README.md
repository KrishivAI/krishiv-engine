# Lakehouse write conformance and round trips (P3-T028b, PR-53b)

Krishiv and real Spark on the same tables: Spark 4.1.0 + Iceberg 1.12.0 for Iceberg, Spark 4.2.0 + delta-spark 4.4.1
for Delta (the P3 oracles). Each script needs Docker and network (the runtime jars come from Maven Central through
Spark's `--packages`) and runs from the repository root; each has a nightly job of the same name. The default PR gate
does not run them: their Rust steps are `#[ignore]`d tests. Nothing in them is a hand-written expectation: every
check compares one engine's record (JSON files in `OUT`, never committed) with the other's.

## `iceberg_write_conformance.sh` (Krishiv first)

1. `write_conformance_tests::krishiv_writes_the_warehouse` writes `unpart`, `part` (identity(x)), `v3`, `v3part`
   through the append sink (two appends each, the first by two drivers, one NULL id) into a Hadoop-catalog layout.
2. `gen_iceberg_write_readback.py` (Spark) records rows, snapshots and their summaries, history, schema, files,
   partitions and, for v3, `_row_id` / `_last_updated_sequence_number`; appends two rows; records again
   (`READBACK.json`).
3. `write_conformance_tests::krishiv_agrees_with_spark_readback` checks Krishiv's record against Spark's, that Spark's
   snapshot is a child of Krishiv's last, that Krishiv reads Spark's table, and time travel to Krishiv's snapshot.

## `iceberg_roundtrip.sh` (Spark first, Spark last)

1. `gen_iceberg_roundtrip.py create`: Spark makes `mor2` (v2, merge-on-read, position deletes), `mor3` (v3, a
   deletion vector) and `evolved` (`id` promoted from `int` to `bigint`, `nm` renamed to `name`, unpartitioned then
   `identity(x)`).
2. `roundtrip_tests::krishiv_meets_spark_tables`: Krishiv reads each table as Spark did, then appends (two drivers,
   a NULL id), runs a copy-on-write `DELETE` of a row in a Spark file carrying Spark's deletes, an `UPDATE` of a
   Spark row in another Spark file carrying deletes and of a Krishiv row, then on `mor2`/`mor3` a merge-on-read
   `DELETE` of a Spark row and a Krishiv row (position delete files on v2, DVs on v3) and on `mor2` a merge-on-read
   `UPDATE`, and appends again; each step records its rows and live delete files; its last metadata becomes the
   Hadoop catalog's next version.
3. `gen_iceberg_roundtrip.py readback`: Spark reads Krishiv's result and snapshots, time travels to each Krishiv
   snapshot (Krishiv's merge-on-read ones included), then runs merge-on-read `DELETE`, `UPDATE`, `MERGE` (position deletes and DVs on Krishiv's files) and an
   append.
4. `roundtrip_tests::krishiv_reads_spark_on_top`: Spark read what Krishiv read at every snapshot; Spark's history is
   Spark's snapshots then Krishiv's, with the same parents and counts; Spark's `delete_files` lists exactly Krishiv's
   last snapshot's (content, format, record count); the rewritten files' v3 DVs are gone (v2 position
   delete files stay, as Java keeps them); v3 row
   ids survive Krishiv's rewrite; Krishiv reads Spark's last metadata and time travels back to its own snapshots.

## `delta_roundtrip.sh` (Spark first, Spark last)

1. `gen_delta_roundtrip.py create`: Spark makes `plain`, `part` (partitioned by `x`), `dv` (a deletion vector) and
   `cdf` (change data feed and deletion vectors).
2. `roundtrip_tests::krishiv_commits_on_spark_tables` (krishiv-connectors-delta): Krishiv reads each table as Spark
   did, then commits through its commit loop an append (data files from its Parquet handler, one per partition
   value), a whole-file `DELETE` (partition `x = 2` of `part`; on `dv` Spark's files, one carrying Spark's DV, and its
   own first file; its own first file elsewhere) and a second append.
3. `gen_delta_roundtrip.py readback`: Spark reads Krishiv's result, the history entries of its commits, each Krishiv
   version by time travel and the change feed of Krishiv's versions on `cdf`; then `DELETE`, `UPDATE`, `MERGE` (each on rows in
   Krishiv's files; the `DELETE` leaves a DV on Krishiv's last file on `dv` and `cdf`) and an append, recording each
   commit's `operationMetrics`.
4. `roundtrip_tests::krishiv_reads_spark_on_top`: Krishiv's versions follow Spark's and leave the protocol alone;
   Spark's history names Krishiv's operation, `readVersion`, `isBlindAppend` and `Serializable`; Spark's time travel
   and change feed agree with Krishiv's reads; Spark's DML metrics are non-zero (`numDeletedRows`,
   `numUpdatedRows`, MERGE's deleted and inserted rows, `numDeletionVectorsAdded` on `dv` and `cdf`); Krishiv reads
   Spark's last version and its own versions again.

## Pending (not written, not stubbed)

| Case | Blocked on |
|---|---|
| Iceberg equality deletes written by Krishiv; a merge-on-read `UPDATE` of a v3 table | Krishiv writes no equality deletes; the session does not select the lineage columns (v3 `UPDATE` refused) |
| Iceberg `MERGE` through Krishiv's SQL | PR-37c (the round trip drives the row-level table as the session's planner does) |
| Delta data files from Krishiv's writer (statistics, column mapping) in the round trip | PR-37a part 2 (#892); its own Spark readback is in `tests/fixtures/delta-4.4.1/krishiv-written/` |
| Delta `UPDATE`, `MERGE`, row-level `DELETE` and DVs written by Krishiv | PR-37 |
| Krishiv's change feed read (`table_changes`) | PR-37e |
| `isBlindAppend: false` on Krishiv's non-append Delta commits (Spark's history shows NULL; the check pins it) | delta_kernel writes the field only when true and replaces a connector's value |
| Krishiv-written tables with non-primitive or wide column types, other `write.*` properties | follow-up |
