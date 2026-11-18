# csv-json-corrupt-record (Spark 4.2.0)

Goldens for acceptance rows PY-076 (`UNSUPPORTED_FEATURE.QUERY_ONLY_CORRUPT_RECORD_COLUMN`), PY-067 (row order of one
CSV file that splits) and SQL-30 (a UTF-16 CSV file read as UTF-8).

- Run: `apache/spark:4.2.0` spark-shell, `local[4]`, ANSI on (`ansi_off.*` probes set it off), by
  `tests/scripts/gen_csv_json_corrupt_goldens.py` (+ `.scala`). Every value in `goldens.json` was produced by that run.
- `goldens.json`: `inputs_hex` (the small input files, byte for byte), `big_file` (generator parameters of PY-067's
  file: row `i` is `"<i>,name<i>,"` padded with `x` to 79 bytes + `\n`, 100000 rows; the two-file case: `f1.csv`
  60000 rows prefixed `a`, `f2.csv` 20000 rows prefixed `b`; `spark.sql.files.maxPartitionBytes` = 1048576), and
  `probes`: label to `{status: ok, value}` or `{status: error, exception, condition, sqlstate, parameters, message}`.
  Strings are `\u`-escaped (NUL is `\u0000`, a bad byte `�`); a row prints as Spark's `Row.toString`.
- Source rules (v4.2.0, with line numbers): see the lane notes `~/krishiv-data/tmp/perf-4/b1/goldens-notes.md`.
  `JsonFileFormat.buildReader` lines 90-96 and `CSVFileFormat.buildReader` 104-110 and `CSVScan` 71-75
  (`requiredSchema.length == 1 && head.name == columnNameOfCorruptRecord`).
- Only the PY-076 error cases are error-class goldens; all others are values.
