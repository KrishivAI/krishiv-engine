# Iceberg tables written by Spark 4.1.0 + Iceberg 1.12.0

Oracle for the read direction of the Iceberg conformance (P3-T028a). `wh/` is the warehouse of a Hadoop catalog that
was mounted at `/wh` in the container, so every path in the metadata is `file:/wh/db/<table>/...`; the test serves
this directory as the root of its file system. `wh/EXPECTED.json` holds the rows Spark read back, the current metadata
file and the snapshot ids per table. Regenerate with `tests/scripts/gen_iceberg_fixtures.py` (the command is in its
docstring), delete the `*.crc` files, and replace this directory's `wh/`.

Tables: `basic`, `partitioned`, `pos_deletes` (position delete files), `dvs` (format version 3 deletion vectors),
`evolved` (add, rename, promote, drop; every snapshot), `nested`. Equality deletes are not written (Spark cannot).

`types.json` (not a table): the Spark schema Iceberg's `SparkSchemaUtil.convert` gives for each Iceberg type, or the
refusal, from the same Spark 4.1.0 + Iceberg 1.12.0 pair. Regenerate with `tests/scripts/gen_iceberg_type_goldens.py`
(command in its docstring; it runs as your user, `--user "$(id -u)"`, so the output directory must be writable by you).
