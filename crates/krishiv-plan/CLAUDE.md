# krishiv-plan

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (logical-plan IR v1: P2-T144).

## Purpose
Krishiv's own logical-plan IR (unresolved and resolved relation/expression nodes, origins, structural hash,
pretty-printer); later the analyzer, catalog traits and physical plans. Front ends (SQL AST, Spark Connect
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
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0005, 0006, 0011, 0037, 0038

## Test
`cargo nextest run -p krishiv-plan --no-tests=pass` · `cargo clippy -p krishiv-plan --all-targets -- -D warnings`
