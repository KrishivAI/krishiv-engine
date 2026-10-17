# krishiv-memory

**Layer:** L1 (may depend only on crates in L1 or lower; `cargo xtask layering`).  
**Real code arrives:** P1 (E-experiments) then P2; hard subsystem: Opus.

## Purpose
Memory management: pools, reservations, the memory arbitrator and spill accounting.

## Invariants
- Every operator allocation is accounted through reservations; pools are empty at query end (tested).
- Arbitration and OOM paths return typed errors or trigger spill; never panic.
- Concurrency primitives tested with loom (`src/loom_tests.rs`; `src/sync.rs` swaps in loom's types).
- Arbitration is cooperative: never call into, lock or mutate another task's operator state; no sync lock across an await.
- `unsafe` allowed; every block has a `// SAFETY:` comment; runs under Miri in CI and nightly ASan/TSan (ADR 0029).
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0007, 0006, 0029

## Test
`cargo nextest run -p krishiv-memory --no-tests=pass` · `cargo clippy -p krishiv-memory --all-targets -- -D warnings` ·
loom models: `RUSTFLAGS="--cfg loom" cargo test -p krishiv-memory --lib --release` (use a separate target dir)
