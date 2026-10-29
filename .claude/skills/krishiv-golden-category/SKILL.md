---
name: krishiv-golden-category
description: Use when adding a new family of Spark-derived golden data (a new tests/fixtures set with a generator and a test) or changing the golden runner, the exclusion manifest or a ratcheted manifest. Goldens come from a real Spark run, never from memory.
---

# Add a golden category

Golden data states what Spark does. It comes from a real Spark 4.2 run or the vendored Spark tag, never from memory
or by hand (ADR 0008). Three kinds exist; pick the one that fits.

## A. Vendored `SQLQueryTestSuite` files (`tests/spark-golden/<tag>/`)
- Import with `tests/scripts/import_spark_golden.sh <tag>`. Only `v4.2.0` is committed; other tags are gitignored.
- The runner is `crates/krishiv-testkit/src/golden/` (`parse.rs` reads `.sql.out`, `run.rs` has the `Engine` trait and
  the report's three numbers: parse, negative, result). `cargo xtask golden [--engine reject-all|sql-parse]` writes
  `target/golden-report.json` through `crates/krishiv-testkit/src/bin/golden-report.rs`.
- **Exclusions.** A file the engine will not run goes into `tests/spark-golden/exclusions.json` with a reason class
  from the fixed list in `crates/krishiv-testkit/src/golden/manifest.rs` (`ReasonClass`: `jvm_udf`, `python_udf`,
  `jvm_only_api`, `external_system`, `jdk_specific`, `nondeterministic_environment`, `unsupported_by_design`,
  `upstream_defect`). An unknown class, an unknown file or a duplicate is an error. Once `frozen` is true, new
  entries carry `"added_after_freeze": true`.
- There is no pass-rate gate on this runner yet; do not describe one.

## B. Generated fixtures with a consuming test (most new categories)
1. **Generator.** `tests/scripts/gen_<name>_goldens.py`, modelled on `gen_window_goldens.py`: it runs each case on a
   real Spark Connect server (`--remote sc://...`, `pyspark-client==4.2.0`), records names, `simpleString` types,
   nullability and rows (or the error class, SQLSTATE and parameters), is deterministic, and writes no local paths.
   Cases whose result depends on row order get a total `ORDER BY`.
2. **Fixture.** Output to `tests/fixtures/spark-4.2.0/<name>.json` (add `spark-4.0.0`/`4.1.0` where versions differ,
   as `tests/fixtures/spark-*/` does). Record `format_version` and the generator path in the file.
3. **Consumer.** A test that loads the fixture and runs every case through the real path, checking exactly what
   Spark gave. Models: `crates/krishiv-session/tests/window_goldens.rs` (end to end through `Session`),
   `crates/krishiv-testkit/tests/` (kernel and analysis level), `crates/krishiv-connect/tests/goldens.rs` (snapshots).
   Cases the engine cannot do yet are listed in the test, each with a reason, and are asserted to fail in the
   expected way (a `PENDING` list, as in `window_goldens.rs`), so closing one forces the list to shrink.
4. **Fixture tests.** If the generator has pure logic, test it in `tests/scripts/test_*.py` (that
   directory's `pyproject.toml` only configures lint).

## C. Ratcheted manifests
`tests/fixtures/analysis/type-coercion/` (one file per coercion input file) and `registry-matrix/` (one file per
function) list the cases that fail
today, each with a reason; `crates/krishiv-testkit/tests/{type_coercion,registry_matrix}.rs` require the list to
match the actual failures exactly, so a fix shrinks it and a regression fails. The golden baseline is split the same
way: `tests/spark-golden/known-failures-categories/` holds one file per category with the rules that assign it (each
rule has an `order`; rules are tried in ascending order across all files, so leave gaps) and
`tests/spark-golden/known-failures-cases/` one file per golden input file; `known-failures.json` itself holds only the
format version, Spark tag and freeze date.
- Regenerate with `KRISHIV_UPDATE_MANIFEST=1 cargo test -p krishiv-testkit --test <name>` (or `cargo xtask golden
  --engine session --ratchet update` for the golden baseline), then read the diff:
  entries removed are progress, entries added are regressions to explain or fix. Only the files whose content changed
  are rewritten, so two PRs touching different keys never conflict.
- A ratchet list in code (`TPCDS_NOT_PLANNED` in `crates/krishiv-plan-codec/tests/ratchet.rs`) follows the same rule:
  it may only shrink, and the test fails when a listed item starts to pass.

## Verify
`cargo nextest run -p krishiv-testkit --no-tests=pass` and the consuming crate; `(cd bench && uv run --group dev pytest ../tests/scripts/test_*.py)` when the generator has tests; `taplo fmt --check; typos`. Fixture
and golden paths are excluded from markdownlint (`.markdownlint-cli2.yaml`), so write notes in the generator's
docstring.
