# krishiv-access-proto

**Layer:** L1 (may depend only on crates in L0–L1; `cargo xtask layering`).

## Purpose
The `krishiv.access.v1` messages (ADR 0049 §3): the request and response of the external access-policy service
that `krishiv.access.provider = external` asks. Messages only (`prost`); the `tonic` client and server stubs are in
`krishiv-server` (`src/access_external/`, feature `access-external`), so `krishiv-plan` and `krishiv-session` stay
transport-free and an adapter can depend on the messages alone.

## Invariants
- `proto/krishiv/access/v1/access.proto` is the schema. `src/access.v1.pb.rs` is **generated, never edited**:
  `cargo run --locked --manifest-path tools/access-protogen/Cargo.toml -- --write` (CI runs `--check`; no `protoc`).
- A public surface (ADR 0026, 0049): fields and enum values are only added, never renumbered, renamed, retyped or
  removed; `*_UNSPECIFIED = 0` reads as a denial; a breaking change is a `v2` package beside `v1`.
- No dependency but `prost`. `#![forbid(unsafe_code)]`.

## ADRs (in `../krishiv-internal/adr/`)
0026, 0036, 0049

## Tests
None of its own: the provider's tests in `krishiv-server` (`tests/access_external.rs`) exercise every message.
