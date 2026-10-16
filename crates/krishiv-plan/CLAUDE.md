# krishiv-plan

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2.

## Purpose
Logical and physical plan models (Spark Connect-shaped unresolved plans, resolved logical, physical).

## Invariants
- `Arc` plan nodes; resolved-plan cache keyed by subtree hash (E10).
- EXPLAIN output snapshot-tested with `insta`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0006, 0011

## Test
`cargo nextest run -p krishiv-plan` · `cargo clippy -p krishiv-plan --all-targets -- -D warnings`
