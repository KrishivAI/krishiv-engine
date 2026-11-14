# krishiv-plan

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (logical-plan IR v1: P2-T144; analyzer: P2-T071a).

## Purpose
Krishiv's own logical-plan IR (unresolved and resolved relation/expression nodes, origins, structural hash,
pretty-printer) and the analyzer (`analyzer`: catalog / function-registry / access-policy traits, Bind and
Resolve stages; `BuiltinFunctions` is the production `FunctionRegistry` over `krishiv-functions`, also used by
`analyzer::testing::context`); later physical plans. Front ends (SQL AST, Spark Connect
decoder in `krishiv-connect`, native DataFrame APIs) lower into the IR.

## Key files
Paths are relative to this crate.
- `src/plan.rs` relation nodes (`LogicalPlan`); `src/expr.rs` expression nodes
- `src/origin.rs` origins (excluded from the encoding); `src/literal.rs` literal values; `src/function.rs` function references
- `src/encode.rs` canonical encoding and `StructuralHash`; `src/dag.rs` DAG-aware equality and transforms
- `src/walk.rs` child mapping and tree rewrites; `src/display.rs` pretty-printer (insta snapshots)
- `src/analyzer/mod.rs` analyzer entry; `src/analyzer/bind.rs` Bind stage (async, catalog and access policy)
- `src/analyzer/windows.rs` window binding and extraction into `Window` relations
- `src/analyzer/repartition.rs` `DISTRIBUTE BY` (`PlanNode::Repartition`): expressions checked as Spark checks `RepartitionByExpression`; the node stays resolved with them bound (widened like a filter's when they read a dropped column), and the physical planner evaluates them in a hash exchange (goldens `tests/fixtures/spark-4.2.0/distribute-by-goldens.json`)
- `src/analyzer/char_varchar.rs` `CHAR(n)` / `VARCHAR(n)` (design P2-T001b): `Resolver::declared` maps attributes to the catalog string; read-side padding of `char(n)` scans (pads, never cuts), comparison padding (`ApplyCharTypePadding`, `IN` included), `AnalyzedPlan::output_declared`; the declared type follows bare-column aliases, CTE references, `shifted` instances, `SubqueryAlias` column lists, ROLLUP/CUBE/GROUPING SETS key copies and repeated set-operation columns; column comments (`Resolver::comments`, seeded from `TableMeta::column_comments`, out as `AnalyzedPlan::output_comments`) follow the same sites (`carry_column_meta`), plus a permanent view's internal read cast
- `src/analyzer/prefix.rs` the prefix cache (`CachingAnalyzer`, one slot per session; design P3-T013a): a plan that is the last analysed root plus steps above it starts Resolve from the kept `Rel`; warm == cold exactly (ids included): every root-environment relation of the prefix is seeded by identity key (a step may read below the root, QUALIFY), the prefix must equal the kept input with its origins (spans, view source ids), and any changed table, securable list, context or equal relation outside the prefix is a miss; Bind and authorization run on every call; `CachingAnalyzer::warm_starts` counts hits
- `src/analyzer/lineage.rs` (P3-T063 N1a, design `P3-T063-analyze-api.md` §3.1) column lineage of an analyzed plan: `lineage(&AnalyzedPlan)` gives per output ordinal the `Leaf`s (table, path redacted, unexpanded view) and the strongest `Transform`, plus the statement's `filters` (`FilterKind`) and fixed `LineageWarning`s; pure, exhaustive over `PlanNode` and the expression kinds (`#[deny(clippy::wildcard_enum_match_arm)]`), memoized by (node address, `Shift` offset), a fixed point for `UnionLoop`; `lineage/walk.rs` relations, `lineage/exprs.rs` expressions. The view side table `AnalyzedPlan::views` (`analyzer/views.rs`) is recorded by the session's view inlining and located by `locate_views`; only temporary/`INVOKER` views expand (`via_view`), others are `Leaf::View`. SQL-level tests: `krishiv-sql/tests/lineage.rs`
- `src/analyzer/views.rs` (P3-T063) the view side table: `ViewExpansion` (one entry per view the session inlined, keyed by the `SubqueryAlias`'s alias and origin, `ViewSecurity`), `locate_views` (the resolved node of each entry); `AnalyzedPlan::views`, filled by the session (query and SQL function bodies)
- `src/analyzer/suggest.rs` the edit-distance ranking of `UNRESOLVED_COLUMN` proposals (only classes whose Spark message proposes names use it)
- `src/analyzer/access.rs` `authorize` (the one privilege check: fail closed, filtered readers never change a securable), `table_ident` / `namespace_parts`
- `src/analyzer/table_change.rs` (P3-T020a) `CatalogCapabilities`, `TableChange` / `NamespaceChange` (Spark's `TableChange` factories), and `apply_changes`: the one all-or-none application of a change list to a table's columns, comments, defaults and properties, with the per-format differences in `AlterRules` (`TypeWidening`, `column_ids` for rename and drop, `reshape` for position and nullability); partition columns never change and new columns go before them; `CHAR(n)`/`VARCHAR(n)` lengths are kept; nested paths, constraints and `CLUSTER BY` are `UNSUPPORTED_FEATURE.CATALOG_OPERATION`
- `src/analyzer/catalog_write.rs` (P3-T020a) `CatalogWrite` (namespace and table changes, `alter_table` with the pinned version, `rename_table`), `TableInfo`, `NamespaceInfo`, `CreateMode`; `Catalog::capabilities` / `as_write` default to read-only, so a catalog without a write half has no write method
- `src/analyzer/resolve.rs` Resolve stage (sync); `src/analyzer/builtins.rs` `BuiltinFunctions`; `src/analyzer/coerce.rs` implicit coercion
- `src/tests/inventory.rs` node inventory checked by the tests

## Extension points
- New IR node: add it to `src/plan.rs` or `src/expr.rs`, bump `ENCODING_VERSION` if the layout changes, and update the inventory table, the printers and the mutation sites.
- New analyzer rule: in `src/analyzer/resolve.rs` (or `bind.rs` if it needs metadata), with a golden from the Spark run.

## Invariants
- `Arc` children for relations and expressions; rewrites go through `map_children` / `transform_up` / `transform_down`.
- The canonical encoding (origins excluded, versioned) is the structural equality; each plan node kind has its own tag (`encode::plan_tag`, `plan_tags_are_unique`); `StructuralHash` is its
  XXH3-128 Merkle hash. Bump `ENCODING_VERSION` when the layout changes (`keys_are_stable` pins it).
- Read options are `OptionValue`s (ADR 0038): secrets never print or enter the encoding in plaintext
  (canary tests in `src/tests/secrets.rs`); inline secrets encode as a per-process keyed fingerprint.
- Plans are DAGs: `PartialEq`, `Debug` and `transform_*` stay linear in distinct nodes (`src/dag.rs`;
  30x self-join regression tests in `src/tests/dag.rs`). Never derive them on `LogicalPlan`/`Expr`.
- Every producer checks `MAX_DEPTH` (`check_depth` / `Plan::try_new`); `Drop`, depth checks and expression
  printing never recurse.
- Never depends on `prost`, `protobuf`, `tonic` or `krishiv-connect`, even transitively (`[forbidden]` in
  `xtask/layers.toml`).
- New nodes: in-crate tests match without wildcard arms (`src/tests/`), so add the node to the inventory table,
  the generator, the mutation sites and the printers.
- Pretty-printer output is snapshot-tested with `insta` (`src/tests/snapshots/`).
- Analyzer: Bind (async) loads each table once, pins one version per table and calls `AccessPolicy` per
  securable before Resolve sees metadata; every denial or policy error fails closed. Resolve is sync and reads
  only `AnalysisContext` (its config is `ResolutionConf`); resolved plans are deterministic and a fixed point.
  Memo keys are addresses: keep every keyed `Arc` alive for the memo's lifetime. Walks over expressions use a
  visited set (expressions are DAGs too). Self-joins of a shared subtree get a `Shift` instance, never a copy.
- Windows (`analyzer/windows.rs`): a window's function binds as a window call (not an aggregate of the
  operator); frames are resolved at bind time (names show them); select lists extract windows into
  `PlanNode::Window`s (Spark's `ExtractWindowExpressions`); a `HAVING` over an aggregate with windows filters
  below them. Window goldens: `tests/scripts/gen_window_goldens.py` → `tests/fixtures/spark-4.2.0/window-goldens.json`.
- SQL-bodied functions (`analyzer/sqlfn.rs`) are inlined at analysis: the registry answers `FunctionLookup::Sql`, the
  body resolves in its own scope with a `Frame` binding parameters to the cast arguments, and the resolved plan holds
  no call of the function. Open functions are tracked in `Resolver::fn_path` (recursion is `CYCLIC_FUNCTION_REFERENCE`);
  Bind adds the bodies as extra roots so their tables and functions are bound like the query's.
- Analyzer goldens come from a real Spark run: `tests/scripts/gen_analyzer_goldens.py` →
  `tests/fixtures/spark-4.2.0/analyzer-goldens.json`; the constant-folding and ORDER BY-over-aggregate cases are a
  separate fixture (`--section folding --views`, safe on the shared oracle) `analyzer-folding-goldens.json`.
- Constant arguments (`round`'s scale, `date_part`'s field, `LIMIT`): `analyzer/fold.rs` folds what Spark's
  `foldable` allows through the kernels the executor runs (`fold_ops.rs`, on `krishiv_kernels::Baseline`: one loop per
  kernel instead of nine, which keeps the binary small) and raises what folding raises, at the
  argument. A function is folded only if it is deterministic and in the argument-pure families
  (`krishiv_functions::scalar::is_argument_pure`: string, math, arithmetic; datetime/session/hash read session
  state). `first`/`last` match a boolean `Literal` and do not fold (`CallTypes::is_literal`).
- Over an aggregate, `ORDER BY`/`HAVING` expressions are matched to grouping and output expressions by
  `semantic_key` (Spark's `semanticEquals`: commutative operands sorted, `a > b` as `b < a`); a memo keyed by
  address must never see an expression that is dropped (`group_substitute` keys probes with a fresh memo).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0005, 0006, 0008, 0011, 0036, 0037, 0038

## Tests
`cargo nextest run -p krishiv-plan --no-tests=pass` · `cargo clippy -p krishiv-plan --all-targets -- -D warnings`

## Split modules (T193)
Files split to stay under 2,000 lines; the parent file keeps the types and the entry points.
- `analyzer/ops/` `aggregate.rs`, `distinct.rs`, `join.rs`, `sortlimit.rs`, `qualify.rs` (`impl Resolver` by topic), `lift.rs` (aggregate lifting); `analyzer/subquery/` `scalar.rs`, `domain.rs`, `support.rs`; `analyzer/coerce/tests.rs`; `tests/analyzer/partN.rs`.
