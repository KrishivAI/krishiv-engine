# krishiv-plan

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (logical-plan IR v1: P2-T144; analyzer: P2-T071a).

## Purpose
Krishiv's own logical-plan IR (unresolved and resolved relation/expression nodes, origins, structural hash,
pretty-printer) and the analyzer (`analyzer`: catalog / function-registry / access-policy traits, Bind and
Resolve stages; `BuiltinFunctions` is the production `FunctionRegistry` over `krishiv-functions`, also used by
`analyzer::testing::context`); later physical plans. Front ends (SQL AST, Spark Connect
decoder in `krishiv-connect`, native DataFrame APIs) lower into the IR.

## Invariants
- `Arc` children for relations and expressions; rewrites go through `map_children` / `transform_up` / `transform_down`.
- The canonical encoding (origins excluded, versioned) is the structural equality; `StructuralHash` is its
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
- Analyzer goldens come from a real Spark run: `tests/scripts/gen_analyzer_goldens.py` →
  `tests/fixtures/spark-4.2.0/analyzer-goldens.json`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0005, 0006, 0008, 0011, 0036, 0037, 0038

## Test
`cargo nextest run -p krishiv-plan --no-tests=pass` · `cargo clippy -p krishiv-plan --all-targets -- -D warnings`
