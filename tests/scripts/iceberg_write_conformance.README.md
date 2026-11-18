# Iceberg write conformance (P3-T028b, PR-53b)

Krishiv writes, real Spark 4.1.0 + Iceberg 1.12.0 reads. Run `tests/scripts/iceberg_write_conformance.sh` from the
repository root (Docker and network; nightly job `iceberg-write-conformance`). The default PR gate does not run it: its
two Rust tests are `#[ignore]`d.

1. `write_conformance_tests::krishiv_writes_the_warehouse` writes `unpart`, `part` (identity(x)), `v3`, `v3part`
   through the append sink (two appends each, the first by two drivers, one NULL id) into a Hadoop-catalog layout.
2. `gen_iceberg_write_readback.py` (Spark) records rows, snapshots and their summaries, history, schema, files,
   partitions and, for v3, `_row_id` / `_last_updated_sequence_number`; appends two rows; records again
   (`READBACK.json`, never committed: everything is regenerated, the expected values are Spark's).
3. `write_conformance_tests::krishiv_agrees_with_spark_readback` checks Krishiv's record against Spark's, that Spark's
   snapshot is a child of Krishiv's last, that Krishiv reads Spark's table, and time travel to Krishiv's snapshot.

## Pending (not written, not stubbed)

| Case | Blocked on |
|---|---|
| Iceberg MERGE / UPDATE / DELETE written by Krishiv (copy-on-write and merge-on-read) | PR-36b units 2-4 (no Krishiv write path on main: `append` only) |
| Iceberg position deletes, deletion vectors, equality deletes written by Krishiv | PR-36b-d |
| Delta MERGE, deletion vectors, change data feed written by Krishiv | PR-37 (#782, open) |
| Krishiv-written tables with non-primitive or wide column types, other `write.*` properties | follow-up |
