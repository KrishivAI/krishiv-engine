# krishiv-connectors-cdc

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P4.

## Purpose
Change data capture source connectors.

## Invariants
- Behind an additive Cargo feature; ordering and delivery guarantees per ADR 0016.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0015, 0016

## Test
`cargo nextest run -p krishiv-connectors-cdc` · `cargo clippy -p krishiv-connectors-cdc --all-targets -- -D warnings`
