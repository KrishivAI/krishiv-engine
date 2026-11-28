# PySpark Connect suite (P3-T030)

Runner: `tests/scripts/pyspark_suite/` (stdlib only). Design and the rationale for the draft exclusion manifest:
krishiv-internal `design/P3-pyspark-suite-runner.md`.

- `manifest.toml` — DRAFT exclusions (reason classes are an enum). The owner freezes it at the first run;
  `UNVERIFIED` entries are undecided candidates.
- `summary/<version>.json` — committed dashboard input (denominator parts, pass rate, pass-list ratchet ids).
- `DASHBOARD.md` — generated: `cd tests/scripts && PYTHONPATH=. python -m pyspark_suite report`.

Current summaries are `source: run` (2026-11-28, pandas<3 in the venvs: pandas 3 breaks `pyspark.pandas`); a new run replaces them via `run --out` and a commit of the summary.
The first CI-sandbox run (pytest `--collect-only`, then the oracle run for `ORACLE_ENV`) is written to the nightly artifact (`run --out`); the owner or lane commits that summary as `summary/<ver>.json`, and nothing replaces the static ones until then.
`run` needs `pytest-timeout` in the version venv and a `krishiv server` command; the nightly `pyspark-suite` job
(`.github/workflows/nightly.yml`) does both, writes to `--out` (never over the committed summary) and then runs
`ratchet`. `run` copies the tag's `sql/tests` and `testing` into the installed `pyspark`, so the wheel stays the
only `pyspark` on the path.

```text
cd tests/scripts
PYTHONPATH=. python -m pyspark_suite static --version 4.3.0 --checkout <sparse checkout of v4.3.0 python/pyspark/sql/tests>
```
