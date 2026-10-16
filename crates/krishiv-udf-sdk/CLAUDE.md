# krishiv-udf-sdk

**Layer:** L1 (may depend only on crates in L1 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (Python UDFs), stabilised in P6.

## Purpose
Stable UDF SDK (Rust/WASM/Python UDF boundary types).

## Invariants
- Small and slowest-changing public surface: semver; Arrow C Data Interface / WASM / PyCapsule at the boundary.
- Depends only on L0–L1 crates.
- `unsafe` allowed; every block has a `// SAFETY:` comment; runs under Miri in CI and nightly ASan/TSan (ADR 0029).
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0022, 0026, 0005

## Test
`cargo nextest run -p krishiv-udf-sdk` · `cargo clippy -p krishiv-udf-sdk --all-targets -- -D warnings`
