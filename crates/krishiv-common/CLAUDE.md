# krishiv-common

**Layer:** L0 (may depend only on crates in L0 or lower; `cargo xtask layering`).  
**Real code arrives:** P0 (registries exist); grows in every phase.

## Purpose
Shared foundation: Spark error classes + SQLSTATE registry (generated from `error-classes.json` by `build.rs`), typed config registry (`krishiv.*`, `krishiv.experimental.*`), tracing setup.

## Invariants
- No dependencies on other krishiv crates.
- Error classes mirror Spark's `error-conditions.json`; names/SQLSTATEs are API (deprecation cycle before removal).
- Config keys are typed and registered here; `cargo xtask docs-config` / `docs-errors` regenerate docs.
- Experimental behaviour is gated by `krishiv.experimental.*`, never by Cargo features.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0026, 0027, 0029, 0008

## Test
`cargo nextest run -p krishiv-common` · `cargo clippy -p krishiv-common --all-targets -- -D warnings`
