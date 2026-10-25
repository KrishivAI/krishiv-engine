# Architecture

A map of the codebase for contributors and coding agents: how a query moves through the crates, where each
concern lives, and the rules that apply everywhere. It names files, not behaviour; each crate's `CLAUDE.md`
holds that crate's invariants, and the module docs (`//!`) at the top of the named files hold the details.

## The path of a query

1. **Entry.** A SQL string reaches `Session::sql`, or a DataFrame call builds a plan directly.
   - Embedded Rust: `krishiv::Session` (`crates/krishiv/src/lib.rs`) re-exports `krishiv-session`
     (`crates/krishiv-session/src/session.rs`, `dataframe.rs`).
   - Python: the PySpark-shaped package `python/krishiv/sql/` over the PyO3 handles in
     `crates/krishiv-python/src/{session,dataframe,column}.rs`.
   - Spark Connect: protobuf plans are decoded in `crates/krishiv-connect/src/wire.rs` and translated into the
     logical IR by `crates/krishiv-connect/src/translate/`. The protos never reach the analyzer.
   - CLI: `crates/krishiv-cli/src/sql.rs` (`krishiv sql`) drives the embedded session.
2. **Parse** (`krishiv-sql`). `parse_statement` (`crates/krishiv-sql/src/lib.rs`) builds an AST with source
   spans (`ast.rs`, `lexer.rs`, `parser.rs`, `parser/`); `lower` / `parse_and_lower` (`lower.rs`) turn a query
   into the logical IR. `validate/` holds the checks done at parse time.
3. **Logical IR and analysis** (`krishiv-plan`). `Plan`, `PlanNode` and `Expr` are in `plan.rs` and `expr.rs`.
   `DefaultAnalyzer` (`crates/krishiv-plan/src/analyzer/mod.rs`) resolves names (`bind.rs`, `resolve.rs`),
   applies implicit casts (`coerce.rs`), and types function calls from the registry (`builtins.rs`, `catalog.rs`).
4. **Optimize** (`krishiv-optimizer`). `optimize`, `optimize_with` (cardinality estimates) and `optimize_traced`
   (per-rule steps, used by `EXPLAIN VERBOSE`) are in `crates/krishiv-optimizer/src/lib.rs`. Rules run in batches
   to a fixed point; the module doc lists the batches in order. Rules: `fold.rs`, `pushdown.rs`, `prune.rs`,
   `join_order.rs`, `restrict.rs`, `group_limit.rs`, `limit.rs`.
5. **Physical plan** (`krishiv-exec`). `PhysicalPlanner` / `DefaultPlanner`
   (`crates/krishiv-exec/src/planner/`) turn the optimized plan into a `PhysicalPlan`
   (`physical/mod.rs`): a tree of `PhysNode` values with no live handles. `physical::instantiate`
   (`physical/instantiate.rs`) builds the operator graph on the worker that runs it.
6. **Plan codec** (`krishiv-plan-codec`). `encode` / `decode` (`crates/krishiv-plan-codec/src/encode.rs`,
   `decode.rs`) turn a `PhysicalPlan` into the `krishiv.plan.v1` bytes defined by
   `proto/krishiv/plan/v1/plan.proto`, and back. It is the form a plan takes between processes. The in-process
   path in step 5 does not go through it; the round-trip is held by tests (`crates/krishiv-plan-codec/tests/ratchet.rs`).
7. **Execute** (`krishiv-exec`).
   - `pipeline.rs` splits the operator graph into pipelines at breakers (aggregate, sort, join build).
   - `driver.rs` is the async driver of one pipeline: it checks memory aborts, cancellation and spill requests at
     every batch boundary.
   - `executor.rs` (`Executor`, `TokioExecutor`) schedules drivers; `runner.rs` (`QueryRunner`) runs a plan.
   - Operators: `operators/` (scan, filter, project, sort, limit, union, file write), `aggregate/`, `join/`
     (hash, grace, nested loop), `window/`. Expression evaluation: `planner/eval/`.
   - Scans read through `krishiv-io`: `scan.rs` (`ScanStream`), `format.rs` (`FileFormat`), `parquet/`, `csv/`,
     `text/`. Kernels (cast, datetime, hash, string, decimal) are in `krishiv-kernels`.
   - Memory and spill: `krishiv-memory` (`pool.rs`, `reservation.rs`, `spill.rs`, `spillfile/`);
     `krishiv-exec/src/output_spill.rs` holds a breaker's finished output.
8. **Output.** Results are Arrow `RecordBatch` streams (`RecordBatchStream`, `DataFrame::collect` /
   `execute_stream` in `crates/krishiv-session/src/dataframe.rs`). Python reads them through the Arrow PyCapsule
   bridge (`crates/krishiv-python/src/stream.rs`); the CLI prints them (`krishiv-cli/src/sql.rs`).
   `krishiv-flight-sql` and the Connect server side are not implemented yet.

`krishiv-session` ties steps 2 to 8 together for one session: `Session::sql_here` parses and lowers, then analysis,
`physical_traced` (optimize and plan) and `run_on_query` execute.

## Layers

`cargo xtask layering` enforces that a crate depends only on crates in the same or a lower layer
(`xtask/layers.toml`, which also lists packages a crate must never depend on).

| Layer | Crates |
|---|---|
| L0 | `krishiv-common` (errors, config registry, redaction, secrets), `krishiv-alloc` |
| L1 | `krishiv-types` (data types, `Batch`), `krishiv-memory` (pools, spill files), `krishiv-udf-sdk`, `krishiv-connector-sdk` |
| L2 | `krishiv-kernels`, `krishiv-functions` (registry and bindings), `krishiv-io` (formats, scan, runtime filters) |
| L3 | `krishiv-sql`, `krishiv-plan`, `krishiv-optimizer` |
| L4 | `krishiv-exec`, `krishiv-plan-codec`, `krishiv-state`, `krishiv-stream` |
| L5 | `krishiv-session`, `krishiv-shuffle`, `krishiv-scheduler`, `krishiv-connectors-{kafka,iceberg,delta,cdc}` |
| L6 | `krishiv-connect`, `krishiv-flight-sql`, `krishiv-mcp`, `krishiv-server`, `krishiv-python`, `krishiv-ui-assets` |
| L7 | `krishiv` (facade), `krishiv-cli`, `krishiv-bench-runner`; test crates `krishiv-testkit`, `krishiv-sim` |

Several crates are placeholders with a short `lib.rs` and no implementation yet: `krishiv-shuffle`,
`krishiv-scheduler`, `krishiv-state`, `krishiv-stream`, `krishiv-flight-sql`, `krishiv-mcp`, both SDK crates and
the four connector crates.

## Where does X live

| Concern | Location |
|---|---|
| Scalar function signatures and the registry | `crates/krishiv-functions/src/registry/` (`builtins()`, `Registry::lookup`; signatures in `defs.rs`, `math.rs`, ...); arity, kind and availability come from `vendored/spark-functions.tsv` |
| Scalar function execution | `crates/krishiv-functions/src/scalar/` (`bind_scalar`, one module per family) over `krishiv-kernels` |
| Type coercion | `crates/krishiv-plan/src/analyzer/coerce.rs`; cast kernels in `crates/krishiv-kernels/src/cast/` |
| Data types and the Arrow mapping | `crates/krishiv-types/src/data_type.rs`, `arrow.rs`, `ddl.rs` |
| Config keys | `crates/krishiv-common/src/config/registry.rs` (`REGISTRY` and the sorted `INDEX`); lookup and parsing in `config/mod.rs`; per-session values in `crates/krishiv-session/src/conf.rs` |
| Error classes | `crates/krishiv-common/vendored/spark-error-conditions.json` (Spark's, vendored) and `krishiv-error-classes.json` (`KRISHIV_*`), compiled by `crates/krishiv-common/build.rs` into typed constructors |
| Spark-version profiles | `SparkProfile` (3.5, 4.0, 4.1, 4.2) in `crates/krishiv-functions/src/signature.rs`; set with `SessionBuilder::profile`; kernel version in `crates/krishiv-kernels/src/version.rs` |
| Plan codec and its feature flags | `crates/krishiv-plan-codec/src/features.rs` (registry, `feature_of`), `proto/krishiv/plan/v1/plan.proto`, `tools/plan-protogen/`; goldens in `crates/krishiv-plan-codec/tests/golden/` |
| Memory pool, reservations, spill | `crates/krishiv-memory/src/` (`pool.rs`, `reservation.rs`, `spill.rs`, `spillfile/`); the driver side in `crates/krishiv-exec/src/driver.rs` |
| File formats and the scan | `crates/krishiv-io/src/` (`format.rs`, `scan.rs`, `scan_gate.rs`, `parquet/`, `csv/`, `text/`, `split/`); path listing in `crates/krishiv-exec/src/files.rs` |
| Runtime filters | produced and consumed in `crates/krishiv-io/src/runtime_filter/`; planned in `crates/krishiv-exec/src/planner/runtime_filter.rs`; operator in `crates/krishiv-exec/src/operators/runtime_filter.rs` |
| Golden tests | `crates/krishiv-testkit/src/golden/` (runner over `tests/spark-golden/`, exclusions in `tests/spark-golden/exclusions.json`); generated fixtures in `tests/fixtures/` with generators in `tests/scripts/gen_*.py` |
| Ratchets | `crates/krishiv-plan-codec/tests/ratchet.rs` (TPC-H and TPC-DS plans round-trip, with a list of unplanned queries that may only shrink); manifests in `tests/fixtures/analysis/` checked by `crates/krishiv-testkit/tests/{type_coercion,registry_matrix}.rs` |
| Benchmarks | `bench/` (harness `src/krishiv_bench/`, adapters in `adapters/`, queries in `queries/`); Rust runner `crates/krishiv-bench-runner/` |
| EXPLAIN ANALYZE and query profiles | `crates/krishiv-exec/src/profile/`, `crates/krishiv-session/src/profile.rs`, `docs/reference/` |
| The Python facade | `python/krishiv/sql/` (PySpark-shaped API, `types.py`), `python/krishiv/_native.pyi` (stubs), `crates/krishiv-python/` (PyO3 bindings), tests in `python/tests/` |
| Repo tooling | `xtask/` (`cargo xtask --help` lists layering, docs-config, docs-errors, budgets, golden, ...) |

## Rules that apply everywhere

- **Spark-exact semantics.** Observable behaviour matches Spark for the selected `SparkProfile`; ground truth
  is Spark's source at the tag plus a real run, and a deliberate divergence is explicit and opt-in.
- **No `unwrap`, `expect` or `panic` in library code.** Errors are `KrishivError` values carrying a Spark error
  class and SQLSTATE from the generated registry.
- **Cargo features are additive.** Behaviour switches are runtime config (`krishiv.*`,
  `krishiv.experimental.*`), not features (`cargo hack check --each-feature`).
- **Layering.** Lower layers never import higher ones; `krishiv-plan` never depends on protobuf or Connect types.
- **No single-process assumptions.** A physical plan is plain data that can be encoded and instantiated on another
  worker; do not share state between planning and execution through globals or live handles.
- **`unsafe` is confined** to `krishiv-memory`, `krishiv-kernels` and the FFI crates, each block with
  a `// SAFETY:` comment; every other crate has `#![forbid(unsafe_code)]`.
- **Versioned formats.** On-disk and wire formats carry a version; a reader handles the previous one.

## How the pieces are tested

| Kind | Where |
|---|---|
| Unit and module tests | beside the code (`tests.rs`, `tests/` inside `src/`) |
| Crate integration tests | `crates/<crate>/tests/` (for example `crates/krishiv-exec/tests/planner/`, `crates/krishiv-plan-codec/tests/`) |
| Plan and EXPLAIN snapshots | `insta` snapshots next to the tests (`krishiv-sql`, `krishiv-optimizer`, `krishiv-plan`) |
| Spark conformance | `krishiv-testkit`: the golden runner, function matrix, kernel evaluation, hash and type-coercion suites; data under `tests/` |
| Python | `python/tests/` (`cd python && uv run pytest`); lint and types with `ruff` and `ty` |
| Fuzzing | `fuzz/fuzz_targets/`, seed generators in `tests/scripts/` |
| Benchmarks | `bench/` (own `pytest` suite in `bench/tests/`) |

## Files that are generated

Do not edit these by hand; each names its generator.

| File | Generator |
|---|---|
| `crates/krishiv-plan-codec/src/proto/krishiv.plan.v1.pb.rs` | `tools/plan-protogen` (`--write`; CI runs `--check`) |
| `crates/krishiv-connect/src/proto/` | `tools/connect-protogen` |
| `crates/krishiv-common/vendored/spark-error-conditions.json` | `tests/scripts/gen_error_classes.py` |
| `crates/krishiv-functions/vendored/spark-functions.tsv` | `tests/scripts/gen_function_registry.py` |
| `tests/fixtures/spark-*/` and `tests/fixtures/analysis/*-manifest.json` | the `tests/scripts/gen_*.py` scripts and `KRISHIV_UPDATE_MANIFEST=1` |
| `tests/spark-golden/v4.2.0/` | `tests/scripts/import_spark_golden.sh` |
| Config and error-class reference text | `cargo xtask docs-config`, `cargo xtask docs-errors` (print to stdout or a path) |

## Reading order for a change

1. The root `CLAUDE.md` (commands, hard rules), then `crates/<crate>/CLAUDE.md` for each crate you touch.
2. The module doc at the top of the file you change; most files state their contract there.
3. The recipe in `.claude/skills/` if the change has one (see the end of this file).
4. The gates that police the rules: `cargo xtask layering`, `cargo hack check --each-feature --workspace`,
   clippy with `-D warnings`, and the crate's own tests. Run them instead of re-deriving the rules.

## Recipes

Step-by-step checklists for common changes are in `.claude/skills/`: `krishiv-spark-function`,
`krishiv-operator`, `krishiv-add-type`, `krishiv-add-config`, `krishiv-add-error-class`,
`krishiv-codec-feature`, `krishiv-golden-category`, `krishiv-bench-suite`.
