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
datasets/gen.sh tpch-dbgen --sf 1              # official-answer reference data
uv run krishiv-bench run --engine duckdb --suite tpch --sf 1 --dataset tpch-dbgen
uv run krishiv-bench run --engine spark-connect --suite tpch --sf 1 --dataset tpch-dbgen
uv run krishiv-bench report --suite tpch --sf 1 --dataset tpch-dbgen   # -> $KRISHIV_DATA/results/report_tpch-dbgen_sf1.md
uv run pytest                                            # harness + Zipf generator tests
docker compose -f baselines/docker-compose.yml stop      # stop (containers and volumes kept; `down` is denied in agent settings)
```
Useful flags: `--iterations 5 --warmup 2 --queries q01,q05 --timeout 600 --duckdb-threads N --duckdb-memory-limit 12GB --spark-remote sc://host:port`.
For publishable numbers run from a clean committed tree (the result JSON records commit + `dirty`).

## Baselines (`baselines/docker-compose.yml`, project `krishiv-bench`)
| Service | Image | Host port | Notes |
|---|---|---|---|
| spark | `apache/spark:4.2.0` | 15002 (Connect), 14040 (UI) | `local[*]`, driver 8g, container cap 12g, `$KRISHIV_DATA` mounted **read-only** at `/data` |
| minio | `minio/minio` (digest-pinned, RELEASE.2025-09-07) | 19100 (S3), 19101 (console) | own instance, creds `krishiv` / `krishiv-bench-secret`; unrelated to `bench-minio` |
| kafka | `apache/kafka:4.0.0` (KRaft, 1 node) | 19092 | profile `kafka`: `docker compose ... --profile kafka up -d` |

Ports are env-overridable (`SPARK_CONNECT_PORT`, `MINIO_S3_PORT`, `KAFKA_PORT`, ...). Spark 4.2.0 is the newest 4.2.x
published on Docker Hub (verified 2026-10-16 via `docker manifest inspect`); the client is pinned to
`pyspark[connect]==4.2.0` to match. Smoke test: `SparkSession.builder.remote("sc://127.0.0.1:15002").getOrCreate().sql("SELECT 1").collect()`.

## Adapters (`src/krishiv_bench/adapters/`)
Interface (`base.Engine`): `setup(dataset)` registers Parquet tables as views (no pre-loading, for parity), `run(query)`
returns all rows, `teardown()`; optional `begin_measure()/end_measure()` bracket each timed run for peak RSS / CPU / spill.
- `duckdb` (in-process, `memory_limit` 12GB to match Spark's container cap): peak RSS = `/proc/self` VmHWM reset with `clear_refs` before each run
  (whole harness process; its pre-run RSS is recorded as `baseline_rss_bytes`); spill = null.
- `spark-connect`: same VmHWM/`clear_refs` method on the server JVM (`pgrep -x java` + `docker exec`, failures raise); CPU-s from `/proc/<pid>/stat`;
  spill = null (Spark spill is available via the REST API :14040 but not collected yet). Host data path is rewritten to `/data`.
  Spark runs with Spark 4 defaults (ANSI mode on) and `spark.sql.shuffle.partitions=24`. The JVM retains its heap, so Spark peaks are cumulative/order-dependent.
  Probes run outside the timed region (a few ms; no sampler thread).
- `krishiv`: drives `krishiv-bench-runner` (`dist`-profile build of `krishiv-session` like the shipped artifacts and DuckDB's optimized build, bench-only crate, not the public CLI) over a stdin/stdout
  line protocol until the Python facade can execute queries. Tables are Parquet views registered at `setup()`; `timed_run` reports
  runner-side seconds, `run` dumps the result to Parquet (outside the timed region) and returns the rows for correctness checks.
  Build (always `dist` at the shipped x86-64-v3 level; the runner refuses to start on a CPU without it):
  `RUSTFLAGS='-C target-cpu=x86-64-v3' cargo build --profile dist -p krishiv-bench-runner` (on arm64: `-C target-cpu=neoverse-n1`). The result's engine `config` records `build_profile` and `target_cpu` (from `krishiv-bench-runner --build-info`). Override the path with `$KRISHIV_BENCH_RUNNER`.
- `polars` (lazy `scan_parquet` + `SQLContext`), `datafusion` (datafusion-python, fair spill pool), `spark-local` (Spark 4.2 `local[N]`, AQE on,
  `shuffle.partitions` = cores) and `sail` (pysail Spark Connect server on loopback, pinned pyspark client): used by `reproduce`, hosted by
  `worker.py` (one process session per engine). Versions are pinned in `pyproject.toml`/`uv.lock`; Java 21 must be on `PATH` for Spark.
- `clickhouse` (opt-in: `--engines duckdb,clickhouse,...`; a reference engine, not a compatibility target): ClickHouse 26.9 embedded through the pinned
  `chdb==4.4.0` wheel (same query engine and Parquet reader as `clickhouse-local`), one process session per engine like the others. Tables are views over
  `file('<parquet>', Parquet)`; `max_threads` = threads, `max_memory_usage` = the memory cap, GROUP BY / ORDER BY spill to
  `$KRISHIV_DATA/tmp/clickhouse-<pid>` past half of it, `join_use_nulls=1` (SQL outer-join semantics). The result records `chdb <wheel> (ClickHouse <engine version>)`.
  Install: nothing beyond `uv sync` (the wheel is hash-locked in `uv.lock`). `chdb` was chosen over a downloaded `clickhouse-local` binary because it is installed
  and version-pinned like every other engine here, so there is no separate download, checksum or PATH step; the engine is the same ClickHouse release. TPC-DS is not run for ClickHouse (`load_queries` stops with the reason).
- Adding Flink etc.: subclass `Engine`, register in `adapters/__init__.py`.
Per-engine SQL dialect overrides: `queries/<suite>/<engine>/qNN.sql` (TPC-H ClickHouse only; ClickBench is fetched and adapted in code).

## Krishiv vs DuckDB, interleaved ABBA (P2-T142)
```bash
RUSTFLAGS='-C target-cpu=x86-64-v3' cargo build --profile dist -p krishiv-bench-runner
KRISHIV_BENCH_RUNNER=${CARGO_TARGET_DIR:-../target}/dist/krishiv-bench-runner \
  uv run krishiv-bench abba --sf 10 --dataset tpch-duckdb --reps 3 --threads 8 --memory-limit 12g
  uv run krishiv-bench abba --suite clickbench --threads 12   # 43 queries over $KRISHIV_DATA/clickbench/hits.parquet
  uv run krishiv-bench abba --suite tpcds --sf 1 --threads 12 # 99 queries; unsupported ones are reported as failed
```
Per query: both engines run it once (untimed, warms the page cache) and Krishiv's rows must equal DuckDB's (type-aware
`compare.rows_match`); a wrong result, an error or a timeout marks the query `failed` and it is never timed. Then `--reps` blocks of
Krishiv, DuckDB, DuckDB, Krishiv give `2 * reps` samples per engine. Same Parquet files, same thread count and memory limit
(DuckDB `threads`/`memory_limit`, Krishiv executor threads/`krishiv.memory.limit`); the OS page cache is never dropped.
Reported: median per engine, ratio of medians with a bootstrap 95% CI, the geomean of the ratios (bootstrap CI) and the worst query.
`/proc/loadavg` is recorded around every query; a run whose maximum exceeds 4 is marked indicative. The run holds
`$KRISHIV_DATA/.bench.lock` (`flock`). Output: `$KRISHIV_DATA/results/<ts>_abba_tpch_sf<N>.json` plus the Markdown table on stdout.

## Build vs build, interleaved ABBA (`ab`)
Compares two Krishiv builds (for example `main` against a PR branch), or two configs of one build, with the `abba` protocol.
`--runner PATH` (on `run`, `abba`, `reproduce`) picks the runner binary and wins over `$KRISHIV_BENCH_RUNNER`.
```bash
# build both runners at the shipped x86-64-v3 level (each in its own target dir), then compare; ratio = B / A, below 1 means B is faster
git worktree add ../main-build origin/main && git worktree add ../pr-build my-branch
(cd ../main-build && RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$HOME/target/ab-main cargo build --profile dist -p krishiv-bench-runner)
(cd ../pr-build   && RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=$HOME/target/ab-pr cargo build --profile dist -p krishiv-bench-runner)
uv run krishiv-bench ab --suite tpch --sf 10 --dataset tpch-duckdb --reps 3 --queries q01,q03,q09 \
  --a-runner $HOME/target/ab-main/dist/krishiv-bench-runner --a-sha "$(git -C ../main-build rev-parse HEAD)" \
  --b-runner $HOME/target/ab-pr/dist/krishiv-bench-runner   --b-sha "$(git -C ../pr-build rev-parse HEAD)"
# one binary, a setting on versus off (repeat --a-conf / --b-conf for several settings)
uv run krishiv-bench ab --suite tpch --sf 10 --a-runner $R --b-runner $R --b-conf some.feature.enabled=true
```
Before the first sample the run waits (up to 15 min) for the 1-minute load average to fall below 2. Both runners stay alive for the run and each keeps its own memory limit (`--memory-limit`, default 12g), so budget for two. Per query
both sides run once and their rows must match (type-aware, as `abba`); a mismatch or error fails the query and it is never timed. Then
`--reps` blocks of A B B A. The table and `$KRISHIV_DATA/results/<ts>_ab_<suite>_sf<N>.json` hold the per-query medians, B/A ratio with a
bootstrap 95% CI, the spread of each side ((max - min) / median), the geomean and the load. The JSON records per side the runner path,
`--conf` list, `--a-sha`/`--b-sha` and the runner's `--build-info` (stored as `null` for runners that predate it).

## Groupby and join suites (`h2o-groupby`, `h2o-join`)
The 10 groupby and 5 join questions of the open-source `db-benchmark` project (MPL-2.0, <https://github.com/duckdblabs/db-benchmark>),
written here as portable SQL in `queries/h2o-groupby/` and `queries/h2o-join/` (our own text; only the question definitions and the
checksum idea come from upstream). The data generator is a Python reimplementation of the upstream R `groupby-datagen.R` /
`join-datagen.R` distributions (schemas, key cardinalities, 90 % common keys for the joins, value-level NAs, sorted variant); it is
deterministic (fixed seed 108, per-chunk RNG streams) but not byte-identical to the R output.
```bash
# 0.5 GB = 1e7 rows, 5 GB = 1e8, 50 GB = 1e9; --na 5 and --sorted give the NA and sorted variants; --k sets the groupby K
uv run krishiv-datasets h2o-groupby --size 5gb          # -> $KRISHIV_DATA/h2o/v1/G1_1e8_1e2_0_0/x.parquet
uv run krishiv-datasets h2o-join --size 5gb             # -> $KRISHIV_DATA/h2o/v1/J1_1e8_NA_0_0/{x,small,medium,big}.parquet
uv run krishiv-datasets h2o-verify "$KRISHIV_DATA/h2o/v1/G1_1e8_1e2_0_0"   # recompute the checksums against MANIFEST.json
uv run krishiv-bench run --engine duckdb --suite h2o-groupby --size 5gb       # also: krishiv, polars, datafusion
uv run krishiv-bench reproduce --suite h2o-join --size 5gb --methods ours --engines krishiv,duckdb,polars
uv run krishiv-bench ab --suite h2o-groupby --size 0.5gb --a-runner $A --b-runner $B
uv run krishiv-bench report --suite h2o-groupby --size 5gb
```
* Datasets are immutable: `--h2o-version` names the output directory and an existing dataset is never overwritten. `MANIFEST.json` records the
  parameters, library versions, a per-column content digest (independent of Parquet encoding), the Parquet file hashes and one dataset digest, which
  result files copy. A different NumPy version can change the stream, so the digest is the identity to compare. Join generation keeps the key
  arrays in memory (about 3 GB at 1e8 rows, about 35 GB at 1e9); groupby is chunked.
* Answer checksums first. Each question has a checksum query (the answer's row count and the sums of its checked columns, the benchmark's `chk`).
  Every engine runs it once before any timing and the row must equal DuckDB's (sums within the usual 1e-6 relative tolerance); a mismatch marks the
  question failed, it is never timed, and `run` exits non-zero. Timed runs materialise the answer inside the engine and count rows, so large
  answers are never converted to Python objects.
* `run` defaults to 3 timed + 1 warm-up runs; the result JSON (`<ts>_<engine>_<dataset>_sf<GB>.json`) holds the checksums, reference checksums,
  timings and the dataset digest. `reproduce` and `ab` use the same checksum comparison before their interleaved timing (`abba` supports the TPC suites
  and ClickBench only).

## Reference engines, two methods (`reproduce`, P2-T156)
```bash
uv run krishiv-bench reproduce --suite tpch --sf 10 --reps 3        # tpch | tpcds | clickbench; --engines, --methods ours,theirs
```
Every engine runs in its own worker process (12 threads, 32 GiB where the engine has a knob; recorded per engine). Correctness first:
each engine's rows are compared with DuckDB's (and Krishiv's); a mismatch, error or timeout is recorded for that query and never timed.
* `ours`: `--reps` blocks, each engine in list order then reversed (`2 * reps` samples), medians, bootstrap CIs, geomean of ratios to DuckDB.
* `theirs`: a fresh process per (engine, query), `--runs` hot runs, best of them, per-run `--their-timeout`. Defaults per source: TPC-H/TPC-DS
  5 runs / 60 s (Polars benchmark), ClickBench 3 runs / 300 s (3 runs per the ClickBench rules; the timeout is ours). The page cache is never dropped.
Output: `$KRISHIV_DATA/results/<ts>_reproduce_<suite>_sf<N>.json` (versions, thread/memory pins, CPU model, load average, per-query verdicts).

`reproduce` records `peak_rss_bytes` per engine and query (JSON, summary `max_peak_rss_bytes`, and a "peak memory (MiB)" table):
the highest RSS of the engine's whole worker process tree (runner child, JVM and helpers) during its timed runs, sampled
every 20 ms and bounded below by the per-process VmHWM (`adapters/proc_mem.py`, `TreeProc`). `ours`: max over the
interleaved samples (a JVM keeps its heap, so later queries inherit it); `theirs`: max over the runs in the fresh process.
`--profile` adds `profile` to Krishiv's entry (the runner's `PROFILE` command): wall time of parse+analyse and
plan+execute plus the executor's per-operator rows and compute time (`DataFrame::collect_with_metrics`; seconds are summed over drivers).
ClickBench reads `$KRISHIV_DATA/clickbench` (`hits.parquet` or a `hits/` directory of parts).

## Result JSON (`$KRISHIV_DATA/results/<ts>_<engine>_<suite>_sf<N>.json`)
`machine` (CPU model, cores, RAM, OS, governor), `git` (krishiv-engine commit/branch/dirty), `engine` (name, version, config (Krishiv: threads, memory limit, `build_profile`, `target_cpu`), build, and `allocator` when the engine reports it, e.g. `mimalloc`; the CLI prints it as `allocator=<name>` in `krishiv --version`, the wheel via `krishiv.build_info()`),
`dataset` (name, SF, path, tables, generator version, manifest sha256), and per query: `iterations`, `times_s`, `p50_s`, `p99_s`, `min_s`, `max_s`, `mean_s`,
`throughput_qps`, `rows_returned`, `has_order_by`, `answer_check`, `rows_stored`, `peak_rss_bytes`, `baseline_rss_bytes`,
`cpu_seconds_median`, `spill_bytes` (null when not measurable), or `error`. Percentiles are nearest-rank, so with the default
5 iterations p99 equals max; use `--iterations 100` for a meaningful p99. Wall time is client-observed (includes result
transfer; results here are tiny). Two warm-up runs per query are discarded (`--warmup`); a per-query `--timeout` cancels hung queries (recorded as error). Absolute numbers on this shared laptop are for relative A/B only.

## Correctness
Two independent checks, both type-aware (`compare.py`, unit-tested):
1. **Official answers**: at run time each engine's rows are validated against DuckDB's bundled official TPC-H answer sets (`tpch_answers()`, SF-specific).
2. **Cross-engine**: the report compares the full stored result rows (`<run>.rows.json.gz` next to the JSON; up to 1M rows/query, anything not stored is listed as skipped).
Rules: ints/strings/dates exact (strings right-trimmed); DECIMAL vs DECIMAL at the smaller scale (so Spark's `AVG(DECIMAL)` -> scale 6 matches an
exact value, while equal-scale sums must match to the cent); DECIMAL vs double within half a unit of the decimal scale; double vs double rel 1e-6.
Row order is verified when the query has a top-level `ORDER BY` (ordered comparison; a pure order difference is reported as such), else rows compare as multisets.
The report never prints "all agree" if any check was skipped: skipped checks are listed explicitly.

## Datasets (`datasets/gen.sh` -> `krishiv-datasets`)
- **TPC-H** SF1/10/100 via `tpchgen-cli` (deterministic: regeneration reproduces identical sha256). Only SF1 generated so far.
- **TPC-H dbgen** (`gen.sh tpch-dbgen --sf 1`, dir `tpch-dbgen/sf1`): DuckDB's reference dbgen. tpchgen-cli's random *text* columns differ from dbgen, so
  q02/q10/q13/q15/q20 do not reproduce the official answers on `tpch` data (aggregates over numeric columns do). Official-answer validation therefore runs only on
  `tpch-dbgen` (`--dataset tpch-dbgen`); runs on `tpch` list the answer check as not performed (cross-engine comparison still runs).
- **TPC-DS** SF via DuckDB `tpcds` extension (`dsdgen`) -> Parquet (SF1 took ~1 min). Harness query files for all 99 are vendored,
  but only TPC-H has been run end-to-end / verified; no TPC-DS expected-row files yet.
- **ClickBench**: `gen.sh clickbench --yes` downloads `hits.parquet` (~14 GB) or `--partitioned` (100 files). Not downloaded.
- **Nexmark**: simplified Python generator (person/auction/bid, default 2/6/92% mix, hot-key skew) producing batch Parquet. E05/E08 may need
  the reference generator or a Rust port (planned in P2); this one is for smoke/relative use.
- **Zipf**: `gen.sh zipf --rows R --keys K --alpha A` (alpha 0.8–1.5 supported; any alpha > 0); bounded Zipf by inverse CDF, key 0 hottest; tested in `tests/test_zipf.py`.

## Query provenance / license
* **TPC-H / TPC-DS**: the query texts in `queries/` were not generated by us with `qgen` / `dsqgen`. They were exported
  (by `queries/export_queries.py`) from DuckDB's `tpch` / `tpcds` extensions (`tpch_queries()`, `tpcds_queries()`), which embed the
  standard query set with the default substitution parameters; expected row counts come from DuckDB's `tpch_answers()`. DuckDB is MIT-licensed;
  the benchmark specifications and query templates are (c) Transaction Processing Performance Council, whose end-user licence permits their use for
  benchmarking. Results from this harness are *TPC-derived* and not audited TPC results; they must not be published as official TPC-H/TPC-DS numbers.
  The harness pins `duckdb==2.0.0` (`pyproject.toml`). TPC-H `dbgen` text columns differ between DuckDB versions, and `tpch_answers()`
  matches the `dbgen` of the same version, so the `tpch-dbgen` reference dataset is generated by the pinned version (SF1 was regenerated with
  DuckDB 2.0.0 `dbgen`, all 22 answer checks pass). `tpch-dbgen/sf10` is still 1.5.6-generated and must be regenerated with
  `datasets/gen.sh tpch-dbgen --sf 10 --force` before its answer checks count (without `--force` a non-empty directory is skipped).
* **ClickBench** (ClickHouse/ClickBench, licensed **CC BY-NC-SA 4.0**): the query texts are **fetched, not redistributed**. They are not in this repository.
  On first use the harness downloads two files from the pinned upstream commit `0e10b8434eef001b43156b860a4e2e3665a35d51`
  (`duckdb-parquet/queries.sql` for every engine but ClickHouse, `clickhouse/queries.sql` for ClickHouse), checks their SHA-256 (pinned in
  `clickbench_queries.py`) and caches them under `$KRISHIV_DATA/clickbench/queries/<commit>/`, as the dataset is downloaded to `$KRISHIV_DATA/clickbench`.
  A download whose hash differs from the pin is rejected and never cached. The harness adapts the text at run time (`clickbench_queries.adapt`), without storing
  a modified copy: `STRLEN(x)` becomes `length(x)`; date-string bounds on `EventDate` become day numbers and `extract(minute FROM EventTime)` /
  `DATE_TRUNC('minute', EventTime)` become integer arithmetic (the raw Parquet columns are a day count and epoch seconds, read without a view); and
  q11, q18, q19, q22, q23, q25, q31-q33, q39-q41 get tie-break sort keys (`TIE_BREAK`) so `LIMIT` / `OFFSET` results are deterministic across engines.

### ClickHouse queries and dialect differences
* **TPC-H**: `queries/tpch/clickhouse/qNN.sql` are ClickHouse's own TPC-H queries, `tests/benchmarks/tpc-h/queries/query_NN.sql` in
  [ClickHouse/ClickHouse](https://github.com/ClickHouse/ClickHouse) at commit `a79b7c063486` (Apache-2.0). Each file's header names its source and any
  adaptation: Q6 uses the alternative formulation from the upstream README (ClickHouse issue 70136), Q15's `CREATE VIEW` / `DROP VIEW` pair is a `WITH` clause (one
  statement per query); the other 20 are verbatim (Q11 is the SF1 file; its `HAVING` fraction is the same constant the shared query uses).
* **ClickBench**: ClickHouse runs the upstream `clickhouse/queries.sql`, fetched and adapted as described above (not stored here).
* **Verdicts**: every ClickHouse result is compared with DuckDB's by the type-aware comparison. A difference that is a verified property of ClickHouse's
  dialect or types is recorded as `mismatch-dialect` with the reason (`dialect.py`; the query is not timed, like any non-`ok` verdict), but only when the
  observed rows differ in the documented way (a per-query check against the DuckDB rows); any other difference, on a listed query too, stays `mismatch`.
  Currently: TPC-H q08 (DECIMAL division truncated to 4 places; DuckDB returns a double), ClickBench q28 and q29 (`length()` counts bytes,
  DuckDB counts characters; the byte mean must be larger for at least one key, and at most 4 times the character mean). ClickBench q04 (`avg` of an
  int64 column) stays a plain `mismatch` until the overflow can be checked against an independent expected value. Checked on TPC-H SF1 (all 22: 21 `ok`, q08 `mismatch-dialect`) and a 2M-row slice of
  `hits.parquet` (40 of 43 `ok`, q28/q29 `mismatch-dialect`, q04 `mismatch`); the full 100M-row file needs the benchmark server.
