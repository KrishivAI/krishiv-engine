---
name: krishiv-bench-suite
description: Use when adding a benchmark suite, dataset or engine adapter to the bench/ harness, or a paired A/B comparison of two builds. Covers the dataset generator, queries, adapter, CLI wiring, the interleaved ABBA protocol and the README.
---

# Add a benchmark suite

The harness is `bench/` (uv project, `bench/README.md`). Datasets and results live outside the repo in
`$KRISHIV_DATA`. Model: the db-benchmark suites (`git show 0a792ce0 --stat`) and the `memcap` suite.

## Checklist
1. **Dataset.** A generator in `bench/src/krishiv_bench/datasets.py` (entry points `krishiv-datasets ...`, declared
   in `bench/pyproject.toml`) or its own module (`h2o_data.py`, `memcap_data.py`). Fixed seed, one RNG stream per
   chunk, content digests independent of the Parquet encoder, a versioned output directory that is never
   regenerated in place, and a `$KRISHIV_DATA/manifests/*.sha256` file (`write_manifest`).
2. **Queries.** One `bench/queries/<suite>/qNN.sql` per query (or a Python module of query text, as
   `memcap_queries.py`). Engine-specific text goes in `bench/queries/<suite>/<engine>/qNN.sql`. `load_queries` in
   `bench/src/krishiv_bench/runner.py` finds them; add a branch there for a suite that is not plain files.
   Queries must be your own text if a source's licence forbids copying.
3. **Adapter** (only for a new engine). Subclass `Engine` in `bench/src/krishiv_bench/adapters/base.py`
   (`setup`, `run`, `timed_run`, `teardown`, optional `begin_measure`/`end_measure`), put it in
   `adapters/<name>.py`, and register it in `adapters/__init__.py` (`make_engine` and `ENGINES`). Krishiv itself runs
   through `adapters/krishiv.py` and the `crates/krishiv-bench-runner` binary.
4. **CLI.** Wire the suite in `bench/src/krishiv_bench/cli.py`: the `--suite` choices of `run`, `ab`, `abba`,
   `reproduce`; `ABBA_DATASET` maps a suite to its default dataset directory; `report.py` renders it.
5. **Correctness before timing.** In `abba` and `ab`, both sides' rows must match (`compare.rows_match`)
   before any timed run; a mismatch or error marks the query `failed` and it is never timed.
6. **Protocol.** Comparisons use interleaved blocks A B B A (`abba.py` for Krishiv against the reference engine, `ab.py`
   for two Krishiv builds or two configs). Sequential or single-order runs are not evidence. The run holds
   `$KRISHIV_DATA/.bench.lock`, records `/proc/loadavg`, and reports medians, the ratio with a bootstrap 95% CI and
   the spread of each side.
7. **Tests.** `bench/tests/test_<suite>.py` (generator determinism and digests, query loading, report rows).
   Run `(cd bench && uv run pytest)`.
8. **Docs.** A section in `bench/README.md`: the data layout under `$KRISHIV_DATA`, the commands, what is recorded,
   and the provenance and licence of anything adapted.

## Verify
`(cd bench && uv run pytest)`, then `cd python && uv run ruff check && uv run ruff format --check && uv run ty check`
(the python-quality gate also covers `bench/`). Publishable numbers need a clean committed tree (the result JSON
records the commit and `dirty`).
