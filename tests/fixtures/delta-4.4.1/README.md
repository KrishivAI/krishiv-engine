# Delta 4.4.1 oracle tables

Written by Spark 4.2.0 + delta-spark 4.4.1 with `tests/scripts/gen_delta_fixtures.py` (the docstring has the Docker
command). Never edit these files by hand; regenerate them. Hadoop's `.*.crc` side files are dropped; Delta's own
`_delta_log/*.crc` files are kept.

| Table | Checkpoint | Versions |
|---|---|---|
| `v1-checkpoint` | classic single file at version 3, `stats_parsed` | 0–4 (a DELETE after the checkpoint) |
| `v2-checkpoint` | `delta.checkpointPolicy = v2`: a JSON top-level checkpoint and one Parquet sidecar | 0–4 |
| `multipart-checkpoint` | classic, six parts (`spark.databricks.delta.checkpoint.partSize = 1`) | 0–3 |

Each table's `EXPECTED.json` holds the rows Spark reads back (by `id`) and the `_delta_log` listing. Consumer:
`crates/krishiv-connectors-delta/src/engine/oracle_tests.rs` (checkpoint read through the assembled engine, with the
pre-checkpoint commits removed so only the checkpoint can supply the state).
