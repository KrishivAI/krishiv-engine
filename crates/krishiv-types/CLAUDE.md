# krishiv-types

**Layer:** L1 (may depend only on crates in L1 or lower; `cargo xtask layering`).  
**Real code arrives:** P2.

## Purpose
Internal type representations: Spark data types mapped onto Arrow, schemas, scalar values, decimal/interval/timestamp rules.

## Invariants
- Type mapping and precision/scale rules follow Spark exactly per supported version.
- 128-bit decimals with Spark precision/scale; binary-safe strings.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0009, 0008

## Test
`cargo nextest run -p krishiv-types` · `cargo clippy -p krishiv-types --all-targets -- -D warnings`
