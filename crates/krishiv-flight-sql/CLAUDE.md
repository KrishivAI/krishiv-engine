# krishiv-flight-sql

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P5.

## Purpose
Arrow Flight SQL server (multi-endpoint).

## Invariants
- Versioned public surface (ADR 0026).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0005, 0026

## Test
`cargo nextest run -p krishiv-flight-sql` · `cargo clippy -p krishiv-flight-sql --all-targets -- -D warnings`
