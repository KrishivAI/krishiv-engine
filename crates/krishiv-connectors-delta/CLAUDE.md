# krishiv-connectors-delta

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P3.

## Purpose
Delta Lake connector on `delta_kernel` (no default features) with our own Engine implementation.

## Invariants
- `delta_kernel` with `default-features = false`; never `deltalake`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028

## Test
`cargo nextest run -p krishiv-connectors-delta --no-tests=pass` · `cargo clippy -p krishiv-connectors-delta --all-targets -- -D warnings`
