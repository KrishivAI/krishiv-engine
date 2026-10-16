---
name: krishiv-benchmark
description: Use when running, adding or comparing Krishiv benchmarks — criterion micro-benchmarks, TPC-H/TPC-DS/ClickBench/Nexmark suites against Krishiv or baselines (DuckDB, Spark 4.2, Sail, Polars, Flink), regression checks, or recording benchmark history. Delegate measurement to the perf-analyst agent.
---

# Benchmarks (ADR 0032, 0029)

Read by path: `../krishiv-internal/adr/0032-experiment-driven-decisions.md`, `0029-code-quality-and-testing.md`;
`../krishiv-internal/spec/10-performance-risks.md`; `budgets.json`; `.github/workflows/nightly-bench.yml`.

## What exists (update this list as P0 lands)
- **P0-T17 (done):** `bench/baselines/docker-compose.yml` (project `krishiv-bench`) — Spark 4.2.0 Connect (host :15002),
  MinIO (:19100), Kafka (profile `kafka`, :19092). Never reuse or modify the unrelated `bench-minio` container.
- **P0-T18 (done):** `bench/datasets/gen.sh` / `uv run krishiv-datasets` — TPC-H (tpchgen-cli) and TPC-DS (duckdb) SF1/10/100 → Parquet,
  ClickBench download (14 GB, `--yes`), simplified Nexmark, Zipf α 0.8–1.5. Data under `$KRISHIV_DATA` (default `~/krishiv-data`),
  sha256 manifests in `$KRISHIV_DATA/manifests/`. Only TPC-H/TPC-DS SF1 generated so far.
- **P0-T19 (done):** `bench/` uv project: `uv run krishiv-bench run --engine {duckdb,spark-connect,krishiv} --suite tpch --sf 1`
  then `uv run krishiv-bench report`; JSON in `$KRISHIV_DATA/results/`. The `krishiv` adapter is a stub until P2 (Sail/Polars adapters
  not written yet). See `bench/README.md` for metrics, normalisation and caveats (p99 == max below 100 iterations; spill = null).
If a needed piece is missing (Sail/Polars/Flink adapter, Krishiv adapter), say so; extend `bench/` rather than building an ad-hoc harness.

## Checklist — a measurement
1. Define the question and the comparison (baseline commit/engine vs candidate) and, for experiments, confirm the
   kill criteria are already written (`krishiv-experiment` skill).
2. Record machine (CPU model, cores, RAM, disk, OS/kernel, governor), commit SHA, toolchain, dataset + SF + checksum,
   engine configs. Laptop = relative A/B only; absolute numbers only from the dedicated machine.
3. Build release (`cargo build --release`, or the profile the harness specifies); warm caches; discard warm-up runs.
4. ≥ 5 repetitions, interleave A and B; collect p50 and p99 latency, throughput, CPU time, peak RSS, spill bytes,
   object-store requests where relevant.
5. Compare: Δ% per query and geomean; report noise (stddev / min–max). Regression gate: > 5% slower fails.
6. Store raw JSON with the harness output; history summaries go to `../krishiv-internal/benchmarks/`; update the
   Metrics section of `STATUS.md` with the latest headline number.

## Adding a benchmark
Criterion bench in `crates/<crate>/benches/` (no `divan`); suite queries under `bench/` with expected row counts so
a wrong answer cannot look fast. Correctness first: verify results against DuckDB/Spark before timing.

Targets for context (proposed, ADR 0032): single-node TPC-H/DS within 1.5x of DuckDB and faster than Sail;
short-query planning < 5 ms.
