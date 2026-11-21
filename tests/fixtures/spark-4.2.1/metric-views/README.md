# Metric-view goldens (P3 4.2-MV)

What Spark does for `CREATE VIEW ... WITH METRICS LANGUAGE YAML AS $$ ... $$`, `measure(...)` and the DDL around a
metric view. Every outcome was produced by a real Spark run; nothing is transcribed from source or docs.

## Files

| File | Spark | Image (pinned by digest in the generator) | Cases |
|---|---|---|---|
| `tests/fixtures/spark-4.2.1/metric-views/cases.json` | 4.2.1 | `apache/spark:4.2.1@sha256:0cb70cf0...` | all |
| `tests/fixtures/spark-4.3.0/metric-views/cases.json` | 4.3.0 | `apache/spark:4.3.0@sha256:3dedc5c8...` | all |
| `tests/fixtures/spark-4.1.0/metric-views/gate.json` | 4.1.0 | `apache/spark:4.1.0@sha256:370ada68...` | `gate` cases only |
| `tests/fixtures/spark-4.0.0/metric-views/gate.json` | 4.0.0 | `apache/spark:4.0.0@sha256:a89782d9...` | `gate` cases only |
| `tests/fixtures/spark-{4.2.1,4.3.0}/metric-views/connect.json` | 4.2.1, 4.3.0 | same images | Spark Connect form (see below) |
| `tests/fixtures/spark-3.5.7/metric-views/gate.json` | 3.5.7 | `apache/spark:3.5.7@sha256:936ff39f...` | `gate` cases only |

The 4.1 and 3.5 runs used the images already on the machine (4.1.0, 3.5.7), so their directories are named after
those exact versions.

Each file is written with one case per line (`dump_golden` in both generators), so a diff shows the cases that
changed.

## How to regenerate

From the repo root (needs docker; about four minutes per full release):

    python3 tests/scripts/gen_metric_view_goldens.py            # all four releases
    python3 tests/scripts/gen_metric_view_goldens.py --only 4.3.0
    python3 tests/scripts/gen_metric_view_goldens.py --rematch  # recompute `match` only, no Spark run

Each release runs in a throw-away container (`o2-mv-spark421`, `o2-mv-spark430`, `o2-mv-spark41x`,
`o2-mv-spark35`, removed by `--rm`). The generator file is bind-mounted and runs under `spark-submit` in `--probe`
mode: classic PySpark, `local[2]`, the in-memory session catalog, `spark.sql.session.timeZone=UTC`, ANSI at the
release default. The generator docstring describes the record format.

## Format

`format_version` 1. A case is `{name, group, note_ref, database, target, steps}`. Each case runs in a fresh
`SparkSession.newSession()` and a fresh database `mv_<name>` (plus `mv_<name>_b` where a case needs two), dropped
afterwards. A step is `sql`, `conf` or `api` (DataFrame API: `table_schema`, `table_collect`, `union_self`,
`df_measure`, `list_tables`, `via_temp`), with `outcome`:

- success: `schema` (name, `simpleString` type, nullable, column metadata when not empty) and `rows`.
  `row_order` is `spark` (command output, or a query with `ORDER BY`) or `sorted` (a query without `ORDER BY`; the
  generator sorted the rows). Values: doubles that are not finite are the strings `NaN`, `Infinity`, `-Infinity`;
  decimals are strings; binary is `{"hex": ...}`.
- failure: `error` with `phase` (`sql`: raised by `spark.sql()`, that is parse, analysis or an eager command;
  `collect`: raised while planning or running; `api`), `java_class`, `condition`, `sqlstate`, `parameters`,
  `query_context`, `message` (`getMessage()`, the full text) and `causes` (class and message of the cause chain).
  A JVM exception that is not a `SparkThrowable` has no `condition`/`sqlstate` (for example
  `MetricViewValidationException` for the 1,024 limit and `NoSuchElementException` for `measure(<dimension>)`).
- `pyspark_conversion_error`: PySpark itself failed to convert the JVM error (a null query-context entry); the JVM
  error is then read from the Py4J error PySpark was handling.

`match` per step is how Krishiv compares: `exact`; `exact_before_plan` for analysis errors whose message ends
with the analyzed plan (compare up to and including `; line N pos M` plus the parameters); or
`class_sqlstate_prefix` for SQLSTATE `XX000` errors, for
JVM errors without a condition and for SnakeYAML scanner messages inside `INVALID_METRIC_VIEW_YAML` (design note
section 5). Messages are not normalised; expression ids (`#123`) in messages vary with the case order, which is
one more reason those steps compare by prefix. Normalised: the `Created Time` / `Last Access` values of DESCRIBE
and SHOW TABLE EXTENDED, `created_time`/`last_access` in `DESCRIBE ... AS JSON`, and expression ids and `plan_id`
in EXPLAIN text. `target` is `compatible` for every case.

## Cases that differ by version

- 4.2.1 vs 4.3.0: only the three missing-clause cases (`parse_missing_with_metrics`, `parse_missing_language`,
  `parse_missing_both_clauses`). 4.2.1 raises `INTERNAL_ERROR` (XX000) "Cannot find main error class
  'MISSING_CLAUSES_FOR_OPERATION'"; 4.3.0 raises `MISSING_CLAUSES_FOR_OPERATION` (42601) with operation
  `CREATE METRIC VIEW`. Everything else is the same after normalising expression ids and the version string.
- 4.1.0 and 3.5.7 (`gate.json`): `CREATE VIEW ... WITH METRICS` is a `PARSE_SYNTAX_ERROR`, at `'METRICS'`
  (pos 20, no SQL fragment) on 4.1.0 and at `'WITH'` (pos 15, with the `== SQL ==` fragment) on 3.5.7.
  `measure(x)` is `UNRESOLVED_ROUTINE` (42883) on both, with "routine" (4.1.0) vs "function" (3.5.7) in the text.
  `DROP VIEW <table>` reports `requiredType` `VIEW` (4.1.0, 3.5.7) vs `VIEW or METRIC_VIEW` (4.2.1, 4.3.0); 3.5.7
  has no SQLSTATE for `WRONG_COMMAND_FOR_OBJECT_TYPE`.
- 4.0.0 (`gate.json`, container `o2-mv-g2-spark400`) is a hybrid of the two: `CREATE VIEW ... WITH METRICS` is a
  `PARSE_SYNTAX_ERROR` at `'METRICS'` (pos 20, like 4.1.0) but the message still carries the `== SQL ==` fragment
  (like 3.5.7, unlike 4.1.0); `measure(x)` and `DROP VIEW <table>` are byte-identical to 4.1.0 ("routine").

## Spark Connect form (`connect.json`)

`python3 tests/scripts/gen_metric_view_connect_goldens.py` starts a Connect server per release (containers
`o2-mv-g1-spark421`, `o2-mv-g1-spark430`) and runs a `pyspark-client` of the same version from the host. Same case
ids as `cases.json`; each step carries `vs_classic`. The three errors that are not `SparkThrowable` reach a Connect
client as `UnknownException` (the 1,024 limit and `NoSuchElementException`, message `(<jvm class>) <text>`, no
condition, no SQLSTATE, gRPC `INTERNAL`) or `IllegalArgumentException` (`SHOW CREATE TABLE ... AS SERDE`). All are
raised by `spark.sql()`, as in classic.
