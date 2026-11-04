# krishiv-shuffle

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).
**Real code since:** P2-T176 unit 1 (local shuffle); remote tiers in P5.

## Purpose
Shuffle service and exchange tiers (local now; Celeborn client and a native service later).

## Key files
Paths are relative to this crate.
- `src/frame.rs` exchange frame format v1-unstable (design P2-T176 §2.4): 44-byte header (magic `KXFR`, version,
  kind, compression, exchange id, stage attempt, map, task attempt, sequence, reduce, body length, CRC32C), Arrow IPC
  body. No N/N−1 promise until P5 freezes it (owner, 2026-10-25).
- `src/local.rs` `LocalShuffle`: `krishiv_exec::exchange::Exchange` for one process; frames in memory or one file per
  map output attempt in a spill scope; commit before read, first committed attempt per map partition wins, a dropped
  or losing writer's file is removed; every read checks checksum, version, header identity and the `End` frame.
- `src/tests.rs` format and commit tests.

## Invariants
- Shuffle files carry a format version; damage is `KRISHIV_SPILL_FILE_CORRUPT`, never wrong rows.
- Exchange routing is the plan's routing hash `H_kg` (krishiv-exec `route`); observable `repartition` hashing is
  Murmur3 seed 42 (ADR 0010) when that task exists.
- In memory, committed frames are not charged to a pool (tests and the simulation); files are not counted against
  the spill cap.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0013, 0020, 0021

## Test
`cargo nextest run -p krishiv-shuffle --no-tests=pass` · `cargo clippy -p krishiv-shuffle --all-targets -- -D warnings`
