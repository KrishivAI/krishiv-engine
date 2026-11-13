# krishiv-arrow-bridge

**Layer:** L1 (may depend only on crates in L1 or lower; `cargo xtask layering`).
**Temporary FFI crate** (ADR 0029 update 2026-11-13; `design/P3-lakehouse.md` §5.2, P3-T022a / PR-06).

## Purpose
Moves Arrow values from the Arrow major iceberg-rust is built on (58) to the workspace's (60) over the Arrow C Data
Interface, without copying buffers. Only values iceberg-rust returns as Arrow cross: its Arrow schema conversion
(field metadata such as `PARQUET:field_id`, byte-identical) and the metadata tables (`inspect`). Data batches never
cross; Krishiv reads data files itself.

## Key files
- `src/lib.rs` the public surface: `schema_58_to_60`, `record_batch_58_to_60` (safe), the `arrow58` re-exports.
- `src/unsafe_move.rs` the crate's only `unsafe`: the C Data Interface move (`from_raw`) of schema and array, and the
  `from_ffi` import; compile-time size/alignment checks of both majors' FFI structs.
- `tests/bridge.rs` `ffi_layout` (both majors against the specification's structs) and the `bridge_roundtrip` cases,
  written as a macro over an Arrow major.

## Invariants
- `unsafe` only in `src/unsafe_move.rs`; each block has `// SAFETY:`; runs under Miri in CI (`ci.yml` miri job).
  The `unsafe_sites` budget counts it.
- Public functions are safe and return `KrishivError` (`INTERNAL_ERROR`) on a refused export or import; imported
  batches are fully validated.
- Depends on `krishiv-common` and the two Arrow majors only. The Arrow 58 crates are `deny.toml` `skip` entries
  scoped to this crate's versions.
- Deleted once iceberg-rust is on the workspace's Arrow major (apache/iceberg-rust#3257); the crate-count and
  `unsafe_sites` budgets shrink back. The next Arrow bump (61) changes the crate names in `Cargo.toml` and the
  test macro's arguments, not the cases.

## ADRs (in `../krishiv-internal/adr/`)
0029 (update 2026-11-13), 0014, 0028

## Test
`cargo nextest run -p krishiv-arrow-bridge` · `cargo +nightly-2026-08-12 miri test -p krishiv-arrow-bridge` ·
`cargo clippy -p krishiv-arrow-bridge --all-targets -- -D warnings`
