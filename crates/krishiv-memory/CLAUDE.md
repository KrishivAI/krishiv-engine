# krishiv-memory

**Layer:** L1 (may depend only on crates in L1 or lower; `cargo xtask layering`).  
**Real code arrives:** P1 (E-experiments) then P2; hard subsystem: Opus.

## Purpose
Memory management: pools, reservations, the memory arbitrator and spill accounting, and spill files (`src/spillfile/`: Arrow IPC + LZ4/ZSTD on local disk with a byte cap, ADR 0007).

## Invariants
- Every operator allocation is accounted through reservations; pools are empty at query end (tested).
- Arbitration and OOM paths return typed errors or trigger spill; never panic.
- Concurrency primitives tested with loom (`src/loom_tests.rs`; `src/sync.rs` swaps in loom's types).
- Arbitration is cooperative: never call into, lock or mutate another task's operator state; no sync lock across an await.
- `unsafe` allowed; every block has a `// SAFETY:` comment; runs under Miri in CI and nightly ASan/TSan (ADR 0029).
- Spill files (`spillfile`): a finished `.spill` is complete (temp file + rename; no fsync by design: scratch data, deleted at the
  next start after a crash); every frame is checksummed, so damage is `SpillError::Corrupt`, never wrong data; all names are
  generated, never user-supplied; files go on drop, a query's directory on drop or `SpillScope::cancel`; crashed sessions are
  reclaimed via a held advisory lock (not a pid); write/read scratch is accounted in the caller's `Reservation`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0007, 0006, 0029

## Test
`cargo nextest run -p krishiv-memory --no-tests=pass` · `cargo clippy -p krishiv-memory --all-targets -- -D warnings` ·
loom models: `RUSTFLAGS="--cfg loom" cargo test -p krishiv-memory --lib --release` (use a separate target dir)
