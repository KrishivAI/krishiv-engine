# krishiv-connector-sdk

**Layer:** L1 (may depend only on crates in L1 or lower; `cargo xtask layering`).  
**Real code arrives:** P3, stabilised in P6.

## Purpose
Stable connector SDK for third-party sources and sinks.

## Invariants
- Small, semver-stable public surface; Arrow C Data Interface at the boundary.
- Depends only on L0–L1 crates.
- `unsafe` allowed; every block has a `// SAFETY:` comment; runs under Miri in CI and nightly ASan/TSan (ADR 0029).
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0026, 0005

## Test
`cargo nextest run -p krishiv-connector-sdk` · `cargo clippy -p krishiv-connector-sdk --all-targets -- -D warnings`
