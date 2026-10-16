---
name: krishiv-benchmark
description: Use when running, adding or comparing Krishiv benchmarks — criterion micro-benchmarks, TPC-H/TPC-DS/ClickBench/Nexmark suites against Krishiv or baselines (DuckDB, Spark 4.2, Sail, Polars, Flink), regression checks, or recording benchmark history. Delegate measurement to the perf-analyst agent.
---

# Benchmarks (ADR 0032, 0029)

Read by path: `../krishiv-internal/adr/0032-experiment-driven-decisions.md`, `0029-code-quality-and-testing.md`;
`../krishiv-internal/spec/10-performance-risks.md`; `budgets.json`; `.github/workflows/nightly-bench.yml`.

## What exists (update this list as P0 lands)
- **Pending P0-T17:** `bench/baselines/docker-compose.yml` — Spark 4.2 + Spark Connect, MinIO, Kafka.
  Never reuse or modify the unrelated `bench-minio` container.
- **Pending P0-T18:** dataset generators (TPC-H/TPC-DS SF1/10/100 → Parquet, ClickBench, Nexmark, Zipf α 0.8–1.5);
  data lives outside the repo under `$KRISHIV_DATA`, checksummed.
- **Pending P0-T19:** `bench/` runner with engine adapters (DuckDB, Spark Connect, Sail, later Krishiv); JSON +
  Markdown report. Until it lands, only criterion micro-benches (`cargo bench -p <crate>`) are available.
If a needed piece is missing, say which task is pending; do not build an ad-hoc harness.

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
