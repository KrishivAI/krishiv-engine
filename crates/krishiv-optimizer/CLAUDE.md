# krishiv-optimizer

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (batch), P4 (IVM pass).

## Purpose
Query optimizer: rule- and cost-based rewrites, including the IVM rewrite pass.

## Invariants
- Never changes a user's explicit `repartition(n)`.
- Rules are plan-snapshot tested; IVM pass satisfies Σ incremental == query(Σ inputs).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0012, 0017, 0020

## Test
`cargo nextest run -p krishiv-optimizer` · `cargo clippy -p krishiv-optimizer --all-targets -- -D warnings`
