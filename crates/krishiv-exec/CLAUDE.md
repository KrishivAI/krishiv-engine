# krishiv-exec

**Layer:** L4 (may depend only on crates in L4 or lower; `cargo xtask layering`).  
**Real code arrives:** P2; hash tables/spill: Opus.

## Purpose
Batch execution engine: operators, pipelines, morsel scheduling within a node.

## Invariants
- All memory through krishiv-memory reservations; every spill path tested with tiny-memory tests.
- Cancellation-safe, no blocking in async; metrics feed EXPLAIN ANALYZE.
- New operators follow skill `krishiv-operator`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0006, 0007, 0011, 0013

## Test
`cargo nextest run -p krishiv-exec` · `cargo clippy -p krishiv-exec --all-targets -- -D warnings`
