# krishiv-io

**Layer:** L2 (may depend only on crates in L2 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (after E08).

## Purpose
Object store and file format I/O: Parquet, CSV, JSON scans/writes on object_store.

## Invariants
- object_store and parquet versions from the workspace (Arrow 60 family).
- Scan path chosen by E08; commit protocol per ADR 0014.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028

## Test
`cargo nextest run -p krishiv-io --no-tests=pass` · `cargo clippy -p krishiv-io --all-targets -- -D warnings`
