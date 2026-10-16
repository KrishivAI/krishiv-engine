# krishiv-sim

Deterministic simulation testing (DST) harness skeleton (P0-T24, ADR 0029). `publish = false`, layer 7.

**Open question (for E01 / P1):** which simulator backs DST — `madsim` (deterministic async runtime
with simulated time, network and file system; replaces tokio/tonic at compile time via `cfg(madsim)`) or
`turmoil` (network-level simulation on top of the real tokio runtime; simpler, no cfg swap, less control
over scheduling and time). The choice is intentionally not made here and **no simulator dependency is added
yet**. It must be recorded in an ADR before P1 builds the first simulated cluster test (fault injection:
node crash/restart, partition, clock skew, object-store errors).

Until then this crate only fixes the seam: `SimConfig` (a seed, so every run is reproducible) and the
`Simulation` trait that either backend will implement.
