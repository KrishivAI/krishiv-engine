# krishiv-connectors-iceberg

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P3.

## Purpose
Iceberg table connector (feature `iceberg`).

## Invariants
- Isolated `iceberg` crate (Arrow 58) crossing to Arrow 60 via the C Data Interface; the only allowed duplicate Arrow (deny.toml exception).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028

## Test
`cargo nextest run -p krishiv-connectors-iceberg` · `cargo clippy -p krishiv-connectors-iceberg --all-targets -- -D warnings`
