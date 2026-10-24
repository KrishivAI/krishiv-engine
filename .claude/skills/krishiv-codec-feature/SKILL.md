---
name: krishiv-codec-feature
description: Use when a change adds or alters something in the krishiv.plan.v1 plan codec - a new PhysNode, project item, enum value or field that an older reader would misread (ADR 0040). Covers the named feature, its legacy level, the reader refusal, the proto annotation and the goldens.
---

# Add a plan-codec feature (ADR 0040)

Read first: `crates/krishiv-plan-codec/CLAUDE.md` and the module doc of `crates/krishiv-plan-codec/src/features.rs`.
Model: the `window_group_limit` change (`git show 0dac6941 -- crates/krishiv-plan-codec proto`).

## Decide
A construct needs a feature when a reader without it would misread or reject it unhelpfully (a new node kind,
project item, enum value, or a field whose absence changes results). A hint or explain detail an older reader may skip
is *ignorable* and needs no feature, only a `// ignorable` comment on the proto element. There is no third class.
Function behaviour is not a codec matter (functions travel by name and registry id).

Test paths below are under `crates/krishiv-plan-codec/`.

## Checklist
1. **Proto.** `proto/krishiv/plan/v1/plan.proto`: new fields get new numbers; reuse nothing; `reserved` anything
   removed. Put `// feature: <id>` (or `// ignorable`) on every new element.
2. **Generated code.** `cargo run --locked --manifest-path tools/plan-protogen/Cargo.toml -- --write` rewrites
   `crates/krishiv-plan-codec/src/proto/krishiv.plan.v1.pb.rs`; `-- --check` is what CI runs and enforces the
   annotations against `tools/plan-protogen/baseline.txt` (never add a line to the baseline).
3. **Registry line.** `crates/krishiv-plan-codec/src/features.rs`, inside `registry!`, at its alphabetical place:
   `(id, Variant, legacy level, Kind, "meaning")`. The id is `snake_case`, never renamed, never reused. The legacy
   level of a new feature is `POST_MIGRATION_READER_VERSION`. `FeatureKind::Optimization` if the planner has an
   equivalent plan without it (masked while a worker lacks it), `Semantic` if not.
4. **Classify constructs.** Add arms to `feature_of` (no wildcards, so new `PhysNode`, `ProjectItemPlan`,
   `JoinType`, `Partitioning`, `WindowFunction` and `AggregateFunction` variants do not compile until classified),
   and to `constructs_of` if the construct contains others.
5. **Encode and decode.** `crates/krishiv-plan-codec/src/encode.rs` and `decode.rs`. A plan that uses no feature must
   still encode to the same bytes as before.
6. **Test inventory.** Add a case to `crates/krishiv-plan-codec/tests/common/inventory.rs` (and the `variant_of`
   match in `tests/roundtrip.rs` for a new `PhysNode`).
7. **Goldens.** `KRISHIV_BLESS_GOLDEN=1 cargo test -p krishiv-plan-codec --test golden` writes the missing files
   under `crates/krishiv-plan-codec/tests/golden/<id>/`; rerun without it. Existing files never change; a change to
   `base/` or another feature's directory means an undeclared feature or an epoch change. `v1`-`v5` are frozen.
8. **Reader refusal.** `tests/features.rs` holds the checks (`a_reader_without_a_feature_refuses_the_plan_naming_it`):
   a reader built with `ReaderFeatures::only([...])` must fail with `CodecError::UnsupportedFeatures` naming the
   feature. Extend the header and construct-mapping tests for the new id.
9. **Ratchet.** `tests/ratchet.rs` must still round-trip every TPC-H and TPC-DS plan; its `TPCDS_NOT_PLANNED` list may
   only shrink.

## Verify
`cargo nextest run -p krishiv-plan-codec --no-tests=pass`, the `plan-protogen --check` command above,
`cargo xtask layering`, clippy `-D warnings`.
