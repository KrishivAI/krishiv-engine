# krishiv-kernels

**Layer:** L2 (may depend only on crates in L2 or lower; `cargo xtask layering`).  
**Real code arrives:** P2; unsafe kernels: Opus.

## Purpose
Vectorised compute kernels (SIMD), Spark-observable Murmur3 (seed 42) and internal fast hashing.

## Invariants
- Spark float ordering (NaN, -0.0) in every compare/sort/hash kernel, nested types included.
- Murmur3_x86_32 seed 42 implemented in-house; fast hash (foldhash/xxh3) only where not observable.
- Generics inside kernels; every specialisation is tested.
- `unsafe` allowed; every block has a `// SAFETY:` comment; runs under Miri in CI and nightly ASan/TSan (ADR 0029).
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0008, 0010, 0011, 0029

## Test
`cargo nextest run -p krishiv-kernels` · `cargo clippy -p krishiv-kernels --all-targets -- -D warnings`
