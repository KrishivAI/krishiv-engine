# krishiv

**Layer:** L7 (may depend only on crates in L7 or lower; `cargo xtask layering`).  
**Real code arrives:** P2.

## Purpose
Facade crate: the public Rust API re-exporting the engine.

## Invariants
- Semver after 1.0; the only Rust surface users should depend on.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0005, 0026

## Test
`cargo nextest run -p krishiv` · `cargo clippy -p krishiv --all-targets -- -D warnings`
