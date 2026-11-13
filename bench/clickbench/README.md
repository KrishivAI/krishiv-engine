# ClickBench entry for Krishiv

The entry folder for [ClickBench](https://github.com/ClickHouse/ClickBench): `benchmark.sh`, `create.sql`, `queries.sql`
(generated, see below), `template.json` and the small scripts ClickBench's driver calls (`install`, `load`, `query`,
`start`, `stop`, `check`, `data-size`). Krishiv runs in-process through its Python wheel; `hits.parquet` is read in
place, so there is no load step and no server.

```bash
./benchmark.sh                                  # downloads hits.parquet (about 14 GB), installs Krishiv, runs all 43 queries 3 times
HITS_PARQUET=/data/hits.parquet ./benchmark.sh  # reuse a downloaded file
KRISHIV_WHEEL=./krishiv-0.2.0-cp311-abi3-manylinux_2_28_x86_64.whl ./benchmark.sh   # a specific wheel instead of PyPI
```

Output, in ClickBench's format: `Load time: <seconds>`, `Data size: <bytes>`, then 43 lines `[first, second, third],` in
seconds, with `null` for a run that failed. In a ClickBench checkout (`../lib/benchmark-common.sh` present) `benchmark.sh`
hands over to ClickBench's own driver; run on its own it uses the loop in the script.

## How a query is timed
- One process per run (`query` runs `query.py`): the Python process starts, registers the `hits` view from `create.sql`
  (the Parquet file is read in place), then times planning, execution and the transfer of the **whole result** into an Arrow
  table. The result is not discarded early. Nothing is cached between runs except the operating system's page cache.
- The first of the three runs of each query is cold: `benchmark.sh` runs `sync` and drops the page cache before it. This
  needs `sudo` without a password; when it is not available the script says so and the result must be tagged `no-cold`.
- Krishiv's result cache is off by default (`krishiv.resultCache.enabled=false`) and this entry does not enable it
  (a query over a temporary view is never stored in it anyway, and every run is a new process). No tuning, indexes,
  projections or materialised views; settings are the defaults, plus `spark.sql.parquet.binaryAsString=true` (the
  file's strings are unannotated BINARY, and Spark has the same setting). The Parquet footer cache lives inside the
  process, so it does not carry over between runs.

## Licence of the queries
ClickBench's `queries.sql` is CC BY-NC-SA 4.0 upstream text, and this repository (Apache-2.0) does not carry or
redistribute it. `make-queries.py` downloads `duckdb-parquet/queries.sql` from a pinned upstream commit
(`0e10b8434eef`), checks its SHA-256, applies the adaptations below and prints the 43 queries; `install` runs it when
`queries.sql` is missing, and `.gitignore` keeps the generated file out of the repository. To submit this entry to the
ClickBench repository, run `./make-queries.py > queries.sql` and add that file in the submission.

## Adaptations
The queries are ClickBench's DuckDB-Parquet text, one per line (q01 to q43), changed only as follows. All changes work on
the raw columns of `hits.parquet` (`EventDate` is a day number, `EventTime` is epoch seconds), so none of them
changes which rows a query selects.

| # | Upstream text | Krishiv (Spark SQL) | Why |
|---|---|---|---|
| 1 | `STRLEN(x)` | `length(x)` | Spark SQL has no `STRLEN`. |
| 2 | `toDateTime(EventTime)` | `EventTime` | The upstream DuckDB macro is not defined here; the column is already epoch seconds. |
| 3 | `EventDate >= '2013-07-01'` (and `<=`) | `EventDate >= 15887` (the day number) | `EventDate` is a day count in the file, not a date. |
| 4 | `extract(minute FROM EventTime)` | `CAST(FLOOR(EventTime % 3600 / 60) AS BIGINT)` | `EventTime` is an integer, not a timestamp. Same minute value. |
| 5 | `DATE_TRUNC('minute', EventTime)` | `EventTime - EventTime % 60` | Same reason: the grouping key is the minute's epoch second, not a timestamp. The rows and their order are the same; the key column prints as an integer. |
| 6 | q29: `REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1')` | `REGEXP_REPLACE(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '$1')` | Spark string literals unescape backslashes, and its replacement string names group 1 as `$1`. |

Adaptations 3 to 5 follow from reading the raw Parquet columns instead of a view that converts them; the views upstream
entries use do the same conversion in `create.sql`. They are the same adaptations Krishiv's own benchmark harness
applies (`bench/src/krishiv_bench/clickbench_queries.py`, which also checks results against DuckDB's).

## Status
The folder has not been run against the full dataset. It is the entry for P3-T091; the first full run, its
`results/<date>/<machine>.json` and any submission to ClickBench come from that task. The wheel version defaults to the
0.2.0 release (`KRISHIV_VERSION`).

Checked so far (2026-11-13): `benchmark.sh` end to end, with a wheel built from `main` (`KRISHIV_WHEEL`), on a 5,000-row
synthetic `hits.parquet` and four of the adapted queries (count, filter, the q29 regular expression, the minute
grouping): the output has the `Load time`, `Data size` and `[a, b, c],` lines and the regular expression returns the
expected domains. Not checked: the other 39 queries, the real file, the cold-cache path (it needs `sudo`), and
ClickBench's own driver (`../lib/benchmark-common.sh`).
