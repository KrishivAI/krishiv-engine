# bench/ — Krishiv benchmark harness (P0-T17/18/19)

Python (uv) harness that runs query suites against engine adapters and records JSON + a Markdown comparison.
Datasets and results live **outside the repo** in `$KRISHIV_DATA` (default `~/krishiv-data`):

```
$KRISHIV_DATA/tpch/sf1/*.parquet   tpcds/sf1/   clickbench/   nexmark/   zipf/
$KRISHIV_DATA/manifests/*.sha256   (sha256sum -c compatible, relative paths)
$KRISHIV_DATA/results/*.json       report_<suite>_sf<N>.md
```

## Quick start
```bash
cd bench && uv sync
export KRISHIV_DATA=$HOME/krishiv-data
datasets/gen.sh tpch --sf 1                    # tpchgen-cli (cargo install tpchgen-cli) -> Parquet + manifest
docker compose -f baselines/docker-compose.yml up -d     # Spark 4.2 Connect + MinIO (project krishiv-bench)
uv run krishiv-bench run --engine duckdb --suite tpch --sf 1
uv run krishiv-bench run --engine spark-connect --suite tpch --sf 1
uv run krishiv-bench report --suite tpch --sf 1          # -> $KRISHIV_DATA/results/report_tpch_sf1.md
uv run pytest                                            # harness + Zipf generator tests
docker compose -f baselines/docker-compose.yml down      # stop (volumes kept; add -v to drop)
```
Useful flags: `--iterations 5 --warmup 1 --queries q01,q05 --duckdb-threads N --spark-remote sc://host:port`.

## Baselines (`baselines/docker-compose.yml`, project `krishiv-bench`)
| Service | Image | Host port | Notes |
|---|---|---|---|
| spark | `apache/spark:4.2.0` | 15002 (Connect), 14040 (UI) | `local[*]`, driver 8g, container cap 12g, `$KRISHIV_DATA` mounted **read-only** at `/data` |
| minio | `minio/minio` | 19100 (S3), 19101 (console) | own instance, creds `krishiv` / `krishiv-bench-secret`; unrelated to `bench-minio` |
| kafka | `apache/kafka` (KRaft, 1 node) | 19092 | profile `kafka`: `docker compose ... --profile kafka up -d` |

Ports are env-overridable (`SPARK_CONNECT_PORT`, `MINIO_S3_PORT`, `KAFKA_PORT`, ...). Spark 4.2.0 is the newest 4.2.x
published on Docker Hub (verified 2026-10-16 via `docker manifest inspect`); the client is pinned to
`pyspark[connect]==4.2.0` to match. Smoke test: `SparkSession.builder.remote("sc://127.0.0.1:15002").getOrCreate().sql("SELECT 1").collect()`.

## Adapters (`src/krishiv_bench/adapters/`)
Interface (`base.Engine`): `setup(dataset)` registers Parquet tables as views (no pre-loading, for parity), `run(query)`
returns all rows, `teardown()`; optional `begin_measure()/end_measure()` bracket each timed run for peak RSS / CPU / spill.
- `duckdb` (in-process): RSS = harness process RSS sampled @5 ms (includes ~100 MB Python baseline); spill = null.
- `spark-connect`: server-side JVM peak RSS via `docker exec` (`VmHWM`, reset before each run with `clear_refs`) and CPU-s from
  `/proc/<pid>/stat`; spill = null (Spark spill metrics are not collected yet). Host data path is rewritten to `/data`.
- `krishiv`: **stub** raising `NotImplementedError` (implemented in P2; the docstring lists the planned shape).
- Adding Sail/Polars/Flink: subclass `Engine`, register in `adapters/__init__.py`.
Per-engine SQL dialect overrides: `queries/<suite>/<engine>/qNN.sql` (none needed so far).

## Result JSON (`$KRISHIV_DATA/results/<ts>_<engine>_<suite>_sf<N>.json`)
`machine` (CPU model, cores, RAM, OS, governor), `git` (krishiv-engine commit/branch/dirty), `engine` (name, version, config),
`dataset` (name, SF, path, tables), and per query: `iterations`, `times_s`, `p50_s`, `p99_s`, `min_s`, `max_s`, `mean_s`,
`throughput_qps`, `rows_returned`, `expected_rows`, `rows_match_expected`, `result_checksum`, `peak_rss_bytes`,
`cpu_seconds_median`, `spill_bytes` (null when not measurable), or `error`. Percentiles are nearest-rank, so with the default
5 iterations p99 equals max; use `--iterations 100` for a meaningful p99. Wall time is client-observed (includes result
transfer; results here are tiny). One warm-up run is discarded. Absolute numbers on this shared laptop are for relative A/B only.

## Correctness
Each query's row count is checked against the official SF1 answer sets (`queries/tpch/expected_rows_sf1.json`) and, across
engines, the report compares row counts and a checksum of the sorted normalised rows (<= 10 000 rows). Normalisation
(`normalize.py`): Decimal/float -> 5 significant digits (DECIMAL vs DOUBLE and summation-order differences vanish), ints exact,
dates ISO, strings right-trimmed, rows sorted (tie order under LIMIT is not guaranteed; a mismatch there is flagged, then judged by hand).

## Datasets (`datasets/gen.sh` -> `krishiv-datasets`)
- **TPC-H** SF1/10/100 via `tpchgen-cli` (deterministic: regeneration reproduces identical sha256). Only SF1 generated so far.
- **TPC-DS** SF via DuckDB `tpcds` extension (`dsdgen`) -> Parquet (SF1 took ~1 min). Harness query files for all 99 are vendored,
  but only TPC-H has been run end-to-end / verified; no TPC-DS expected-row files yet.
- **ClickBench**: `gen.sh clickbench --yes` downloads `hits.parquet` (~14 GB) or `--partitioned` (100 files). Not downloaded.
- **Nexmark**: simplified Python generator (person/auction/bid, default 2/6/92% mix, hot-key skew) producing batch Parquet. E05/E08 may need
  the reference generator or a Rust port (planned in P2); this one is for smoke/relative use.
- **Zipf**: `gen.sh zipf --rows R --keys K --alpha A` (alpha 0.8–1.5 supported; any alpha > 0); bounded Zipf by inverse CDF, key 0 hottest; tested in `tests/test_zipf.py`.

## Query provenance / license
TPC-H and TPC-DS query texts in `queries/` were exported (by `queries/export_queries.py`) from DuckDB 1.5.x's `tpch`/`tpcds`
extensions (`tpch_queries()`, `tpcds_queries()`), which embed the standard qgen query set with default substitution
parameters, and expected row counts from DuckDB's `tpch_answers()`. DuckDB is MIT-licensed; the benchmark specifications and
query templates are © Transaction Processing Performance Council. Results from this harness are *TPC-derived* and not
audited TPC results; they must not be published as official TPC-H/TPC-DS numbers.
