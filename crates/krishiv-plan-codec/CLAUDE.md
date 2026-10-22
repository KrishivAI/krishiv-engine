# krishiv-plan-codec

**Layer:** L4 (may depend only on crates in L4 or lower; `cargo xtask layering`).

## Purpose
The `krishiv.plan.v1` codec (ADR 0003, P2-T173): `PhysicalPlan` to protobuf bytes and back, losslessly. It is the
enforcement of design P2-T171 invariant I1 (plans are pure data) and the wire format the coordinator ships to workers.
`krishiv-plan` stays free of `prost` (its `[forbidden]` rule); `krishiv-exec` has no protobuf either.

## Invariants
- `proto/krishiv/plan/v1/plan.proto` is the schema. `src/proto/krishiv.plan.v1.pb.rs` is **generated, never edited**:
  `cargo run --locked --manifest-path tools/plan-protogen/Cargo.toml -- --write` (CI runs `--check`; no `protoc`).
- Compatibility is N to N-1 both ways (ADR 0026): add fields with new numbers only, `reserved` the number and name
  of anything removed, never reuse an enum value. A change an older reader would silently misread raises
  `MIN_READER_VERSION` (and `FORMAT_VERSION`) and adds `tests/golden/v<N+1>` (see `tests/golden.rs`); old golden
  directories are never edited.
- Flat tables with child indices, no recursion in the encoder or decoder; the decoder bounds size, counts and depth
  (`Limits`), and every reference must point to an earlier entry.
- A plan carries `secret(scope, key)` references only (ADR 0038): an inline secret or a URL with credentials fails
  to encode, and error messages never name a value.
- A new `PhysNode` variant fails to compile in `encode.rs`, `decode.rs` and `tests/common/inventory.rs`
  (`variant_of`), and the inventory test fails until a case of it exists. Adding a case changes the golden files:
  they only grow for the current version.
- `#![forbid(unsafe_code)]`.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0026, 0038

## Test
`cargo nextest run -p krishiv-plan-codec --no-tests=pass` · tests: `roundtrip` (inventory, secrets, corrupt input),
`golden` (compatibility bytes), `ratchet` (TPC-H 22 and TPC-DS 99 planned plans).
