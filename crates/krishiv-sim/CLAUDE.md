# krishiv-sim

**Layer:** L7 test infrastructure (`publish = false`; product crates must not depend on it; `cargo xtask layering`).  
**Status:** P0-T24 seam only (`SimConfig` seed + `Simulation` trait); the first simulated cluster test is P1.

## Purpose
Deterministic simulation testing (DST) harness: reproducible runs from a seed, with fault injection (crash/restart,
partition, clock skew, object-store errors) once a backend is chosen.

## Key files
Paths are relative to this crate.
- `src/lib.rs` `SimConfig` (seed) and the `Simulation` trait
- `README.md` status of the harness

## Extension points
- New scenario: implement `Simulation`. A simulator backend needs an ADR first (none is chosen yet).

## Invariants
- Every run is reproducible from its seed; no wall-clock or ambient randomness.
- Backend choice (`madsim` vs `turmoil`) is open until E01 / P1 and needs an ADR; no simulator dependency yet.
- `#![forbid(unsafe_code)]`.

## ADRs (in `../krishiv-internal/adr/`)
0029

## Tests
`cargo nextest run -p krishiv-sim --no-tests=pass` · `cargo clippy -p krishiv-sim --all-targets -- -D warnings`
