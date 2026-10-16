# krishiv-stream

**Layer:** L4 (may depend only on crates in L4 or lower; `cargo xtask layering`).  
**Real code arrives:** P4.

## Purpose
Streaming runtime: three execution profiles, watermarks, barriers/checkpoints, IVM operators.

## Invariants
- batch == stream invariant holds for every supported query (CI test).
- Delivery guarantees and watermark semantics per ADR 0016; checkpoints/barriers are Opus work.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0015, 0016, 0017, 0018

## Test
`cargo nextest run -p krishiv-stream` · `cargo clippy -p krishiv-stream --all-targets -- -D warnings`
