# PySpark Connect suite (P3-T030)

Runner: `tests/scripts/pyspark_suite/` (stdlib only). Design and the rationale for the draft exclusion manifest:
krishiv-internal `design/P3-pyspark-suite-runner.md`.

- `manifest.toml` — DRAFT exclusions (reason classes are an enum). The owner freezes it at the first run;
  `UNVERIFIED` entries are undecided candidates.
- `summary/<version>.json` — committed dashboard input (denominator parts, pass rate, pass-list ratchet ids).
- `DASHBOARD.md` — generated: `cd tests/scripts && PYTHONPATH=. python -m pyspark_suite report`.

Current summaries are `source: static`: an `ast` census of the pinned tag (never imported or executed), 0 passed.
The first CI-sandbox run (pytest `--collect-only`, then the oracle run for `ORACLE_ENV`) replaces them.
The `run` step needs `pytest-timeout` in the version venv and a `krishiv server` command (not wired yet: it
needs the P2b server stub).

```text
cd tests/scripts
PYTHONPATH=. python -m pyspark_suite static --version 4.3.0 --checkout <sparse checkout of v4.3.0 python/pyspark/sql/tests>
```
