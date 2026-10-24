# krishiv-plan-codec

**Layer:** L4 (may depend only on crates in L4 or lower; `cargo xtask layering`).

## Purpose
The `krishiv.plan.v1` codec (ADR 0003, P2-T173): `PhysicalPlan` to protobuf bytes and back, losslessly. It is the
enforcement of design P2-T171 invariant I1 (plans are pure data) and the wire format the coordinator ships to workers.
`krishiv-plan` stays free of `prost` (its `[forbidden]` rule); `krishiv-exec` has no protobuf either.

## Invariants
- `proto/krishiv/plan/v1/plan.proto` is the schema. `src/proto/krishiv.plan.v1.pb.rs` is **generated, never edited**:
  `cargo run --locked --manifest-path tools/plan-protogen/Cargo.toml -- --write` (CI runs `--check`; no `protoc`).
- Compatibility is by named features (ADR 0040, `src/features.rs`): the header lists the features a plan uses
  (derived from the plan alone), and a reader decodes every plan of its epoch whose features it knows. Add fields
  with new numbers only, `reserved` the number and name of anything removed, never reuse an enum value.
- A construct an older reader would misread or reject unhelpfully needs a feature: one **alphabetically sorted**
  registry line, the arms of `feature_of` (no wildcards, so new variants do not compile until classified), a
  `// feature: <id>` comment on each new proto element (`ignorable` for what an older reader may skip;
  `plan-protogen --check` enforces it against `tools/plan-protogen/baseline.txt`), and `tests/golden/<id>/`.
  New features take legacy level `POST_MIGRATION_READER_VERSION` (6).
- `format_version` is frozen at 5 and `min_reader_version` is kept for readers that predate features; plans
  without an `epoch` are such older plans and decode by their content. `tests/golden/v1`-`v5` are frozen and
  decode-only; `base/` and the feature directories are byte-exact against the encoder forever.
- Function behaviour is not a codec matter: functions travel by name and registry id (registry version).
- Flat tables with child indices, no recursion in the encoder or decoder; the decoder bounds size, counts and depth
  (`Limits`), and every reference must point to an earlier entry.
- A plan carries `secret(scope, key)` references only (ADR 0038): an inline secret or a URL with credentials fails
  to encode, and error messages never name a value.
- A new `PhysNode` variant fails to compile in `encode.rs`, `decode.rs`, `features.rs` and
  `tests/roundtrip.rs` (`variant_of`), and the inventory test fails until a case of it exists; a new case gets its
  golden file with `KRISHIV_BLESS_GOLDEN=1` (then rerun the tests). The decoder maps protobuf enums without
  wildcards (`enum_map!`).
- `#![forbid(unsafe_code)]`.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0026, 0038, 0040

## Test
`cargo nextest run -p krishiv-plan-codec --no-tests=pass` · tests: `roundtrip` (inventory, secrets, corrupt input),
`features` (header, reader checks), `golden` (compatibility bytes, simulated older readers, header honesty),
`ratchet` (TPC-H 22 and TPC-DS 99 planned plans).
