# krishiv-scheduler

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P5 (E01 informs the local scheduler in P1).

## Purpose
Distributed scheduler: coordinator/worker task scheduling, AQE, fault tolerance.

## Invariants
- Tested with deterministic simulation (madsim/turmoil) for crashes and partitions.
- Hard subsystem: Opus.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0004, 0006, 0013, 0020

## Test
`cargo nextest run -p krishiv-scheduler` · `cargo clippy -p krishiv-scheduler --all-targets -- -D warnings`
