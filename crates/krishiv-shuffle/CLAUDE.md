# krishiv-shuffle

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P5.

## Purpose
Shuffle service and exchange tiers (local, Celeborn client, native service later).

## Invariants
- Shuffle files carry a format version.
- Observable partitioning uses Murmur3 seed 42 (ADR 0010).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0013, 0020, 0021

## Test
`cargo nextest run -p krishiv-shuffle` · `cargo clippy -p krishiv-shuffle --all-targets -- -D warnings`
