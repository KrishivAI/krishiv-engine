# Spark Connect client smoke runs (Go and Scala)

Report-only smoke runs of two Apache Spark Connect clients against `krishiv server`:
the Go client (`github.com/apache/spark-connect-go/v40`) and the Scala client
(`org.apache.spark:spark-connect-client-jvm_2.13`). They show which client-side calls work
against Krishiv today; they gate nothing (P3 plan rows PR-39 / PR-40, "report-only smoke runs").
The Python client is covered by the Connect goldens and the Python suites; the Apache Rust client
is a separate CI test.

## Run

Docker and python3 are the only tools needed (the clients run in `golang:1.24` and `virtuslab/scala-cli`
containers on the host network).

```text
cargo build -p krishiv-cli --features server
tests/connect-clients/run.sh --server-bin target/debug/krishiv        # starts and stops the server
tests/connect-clients/run.sh --remote sc://localhost:15002 --clients go   # an already running server
```

`--out DIR` (default `target/connect-clients`) receives `go.json`, `scala.json`, `summary.md` and,
when the script starts the server, `server.log`. The exit status is 0 whenever the runs happened;
`python3 report.py DIR --strict` exits 1 if any check did not pass.
The same query list also runs against real Spark (`apache/spark:4.1.0`,
`sbin/start-connect-server.sh`) as a baseline: when a check fails there, the check is wrong.

## What runs

`queries.json` is shared by both clients: 18 SQL checks (literals, arithmetic, strings, aggregates,
joins, windows, a CTE, NULL handling, decimals and dates as strings, arrays, empty results) and
three error checks (`DIVIDE_BY_ZERO`, `UNRESOLVED_COLUMN`, `PARSE_SYNTAX_ERROR`: the error class must
appear in the client's exception). Each client renders a value as `NULL` or its plain string form, so
the expected rows are strings. Four API checks per client are not SQL strings: row count, schema field
names, a config round trip and a collect.

## Delta reads and TPC-DS (pyspark client)

Two more runs use the pinned pyspark client of `bench/` (run with `uv`, no Docker) and write their own summaries:

- `delta` (`delta_read.py`, default in `run.sh`): reads the Spark-written tables of `tests/fixtures/delta-4.4.1`
  three ways (`SELECT` from ``delta.`path` ``, `read.format("delta").load`, `read.load(format=...)`) and compares the
  rows with what Spark recorded in each `EXPECTED.json`; the table Krishiv refuses on purpose must fail with an error
  class. Needs a server built with the `delta` cargo feature (`cargo build -p krishiv-cli --features server,delta`);
  without it every read says `DATA_SOURCE_NOT_FOUND`. Output: `delta-read.json`, `delta-read-summary.md`.
- `tpcds` (`tpcds_connect.py`, `--clients tpcds --tpcds-data DIR`): registers the TPC-DS tables with
  `CREATE TEMPORARY VIEW .. USING parquet` and runs the 99 queries of `bench/queries/tpcds`; the report lists each
  failing query's error class. `--reference sc://..` adds a second server (Spark) on the same files and compares row
  counts. Output: `tpcds.json`, `tpcds-summary.md`. It is not in the weekly workflow (no TPC-DS data on the runner).
  For timings use `krishiv-bench run --engine krishiv-connect --suite tpcds` (`bench/README.md`).

To add a check, add an entry to `queries.json` (`rows`, or `error` for a failing statement); both
clients pick it up. Keep values to integers, strings and booleans (cast anything else to a string in
the query) so the two clients' renderings agree.

CI: `.github/workflows/connect-clients.yml` (manual and weekly) builds the server, runs this script
and writes `summary.md` to the job summary. Tests of the report script: `python3 -m pytest` in this
directory.

Note: `error_divide_by_zero` expects Spark's ANSI behaviour (4.x default). Spark 3.5 returns NULL for `1 / 0`, so the check is for 4.x servers only.
