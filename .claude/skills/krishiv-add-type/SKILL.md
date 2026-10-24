---
name: krishiv-add-type
description: Use when adding a Spark data type to krishiv-types::DataType (or changing how one maps). Lists every layer a type crosses - Arrow mapping, DDL parser, literals, coercion, kernels, scans, plan codec, Spark Connect, Python - and the checks for each.
---

# Add a data type

`DataType` is `#[non_exhaustive]` and several matches over it end in a wildcard, so the compiler does **not** list
every place to touch. Work through this list and grep for a sibling type (`TimestampNtz` is a good model:
`grep -rln TimestampNtz crates python`). Ground truth for names, coercion and casts is Spark 4.2 at its tag plus a
real run (skill `krishiv-spark-function`, step 2); never memory.

## Checklist
1. **Type.** `crates/krishiv-types/src/data_type.rs`: the variant, `keyword()` (the `simpleString`), `simple_string`.
2. **Arrow mapping, both directions.** `crates/krishiv-types/src/arrow.rs` (`DataType::to_arrow` and `from_arrow`;
   `normalize_ingest_type` if files can carry other Arrow types for it).
3. **Type strings.** `crates/krishiv-types/src/ddl.rs` (`parse_data_type`, `resolve`); the SQL side reuses it
   (`crates/krishiv-sql/src/types.rs`, `parser/types.rs`). Add cases to `crates/krishiv-types/tests/spark_type_strings.rs`;
   the fixture comes from `tests/scripts/gen_type_strings.py`.
4. **Literals.** `crates/krishiv-plan/src/literal.rs` (`Literal` variant), `expr.rs`, `display.rs`, `encode.rs`
   (canonical encoding); SQL lowering in `crates/krishiv-sql/src/lower.rs` and `validate/literal.rs`; evaluation in
   `crates/krishiv-exec/src/planner/eval/mod.rs`; statistics in `crates/krishiv-optimizer/src/stats.rs`
   and `fold.rs`.
5. **Coercion and function signatures.** `crates/krishiv-plan/src/analyzer/coerce.rs` and `exprs.rs`; accepted
   argument types in `crates/krishiv-functions/src/registry/` (`defs.rs`, `number.rs`) and `signature.rs`.
6. **Kernels.** Casts to and from the type in `crates/krishiv-kernels/src/cast/`, per Spark version
   (`crates/krishiv-kernels/src/version.rs`); ANSI on and off.
7. **Scans and writes.** `crates/krishiv-io/src/parquet/` (decode, `write/`), `csv/convert.rs`, `csv/encode.rs`,
   `text/`; pushdown in `crates/krishiv-exec/src/planner/pushdown.rs`.
8. **Plan codec.** `proto/krishiv/plan/v1/plan.proto` `DataType.Kind` (new number, never reused) with a
   `// feature: <id>` comment; the arms in `crates/krishiv-plan-codec/src/encode.rs` and `decode.rs` (the encoder
   has a wildcard arm that returns `CodecError::Unsupported`, so add the arm explicitly); then follow skill
   `krishiv-codec-feature` for the registry line, golden and `plan-protogen` baseline.
9. **Spark Connect.** `crates/krishiv-connect/src/translate/types.rs` and `literal.rs` (both directions, with their
   `tests.rs`).
10. **Python.** `python/krishiv/sql/types.py` (class, `_SIMPLE` table), `_convert.py` if values convert,
    `python/krishiv/_native.pyi` if the native surface changes; tests in `python/tests/`.
11. **Golden harness values.** `crates/krishiv-testkit/src/golden/value.rs` and `cell.rs` so goldens can state
    and compare the type; refresh affected manifests (skill `krishiv-golden-category`).

## Verify
- `cargo nextest run -p krishiv-types -p krishiv-plan -p krishiv-kernels -p krishiv-io -p krishiv-connect -p krishiv-plan-codec -p krishiv-testkit --no-tests=pass`
- `cd python && uv run pytest && uv run ruff check && uv run ty check`
- `cargo xtask layering`; clippy with `-D warnings`.
- Moved counts in `tests/fixtures/analysis/*-manifest.json` are rewritten with `KRISHIV_UPDATE_MANIFEST=1`; read the
  diff, since a manifest entry that disappears is a fix and one that appears is a regression.
