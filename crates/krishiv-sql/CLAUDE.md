# krishiv-sql

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (after E09).

## Purpose
Spark SQL dialect parser and analyzer (resolution, type coercion).

## Invariants
- Own parser (no sqlparser-rs); parser technology decided by E09.
- Unresolved plans mirror Spark Connect relations/expressions (ADR 0003).
- Error messages/classes match Spark; `insta` snapshots for errors.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0008, 0012

## Test
`cargo nextest run -p krishiv-sql` · `cargo clippy -p krishiv-sql --all-targets -- -D warnings`
