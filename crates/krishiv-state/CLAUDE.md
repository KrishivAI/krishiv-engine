# krishiv-state

**Layer:** L4 (may depend only on crates in L4 or lower; `cargo xtask layering`).  
**Real code arrives:** P4 (P6 native default); Opus.

## Purpose
Streaming state store (SlateDB-backed first, then the native `krishiv-state` backend).

## Invariants
- State files carry a format version; N reads N−1 (upgrade tests).
- No RocksDB (ADR 0019); backend chosen by E05.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0019, 0015, 0026

## Test
`cargo nextest run -p krishiv-state` · `cargo clippy -p krishiv-state --all-targets -- -D warnings`
