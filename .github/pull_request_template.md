<!-- Keep PRs under 5,000 counted changed lines (test code at half weight; lock files, snapshots, goldens, fixtures and vendored data excluded). One PR per work unit, one commit per task. -->

## Summary

<!-- What and why, in a few lines. Link the task (P2-Txxx), spec chapter and ADRs. -->

Task / spec / ADRs:

## Tests run

<!-- Paste the commands you ran and their result lines. "CI will tell" is not an answer. -->

```text

```

## Checklist

- [ ] Commits are signed off (`git commit -s`, DCO) and follow Conventional Commits.
- [ ] Source files carry `SPDX-License-Identifier: Apache-2.0`.
- [ ] Tests written first and meaningful; docs updated in the same PR; `CHANGELOG` is generated (do not edit it).
- [ ] Every `unsafe` block has a `// SAFETY:` argument; lint exceptions use `#[expect(..., reason = "...")]`, never `allow`.
- [ ] Errors carry a Spark error class and SQLSTATE from the registry (no ad-hoc strings).
- [ ] Spark-observable behaviour: expectations come from executing Spark in the `krishiv-bench` container, never transcribed.

### Rust API Guidelines (public items)

- [ ] Names follow RFC 430 and the conversion conventions (`as_`, `to_`, `into_`); getters have no `get_` prefix.
- [ ] Public types are `Debug`; errors implement `std::error::Error + Send + Sync + 'static`; `Send`/`Sync` where meaningful.
- [ ] `#[non_exhaustive]` on public enums and structs that may grow; no `pub` fields on evolving types; builders for complex constructors.
- [ ] Every public item is documented; primary entry points have a runnable example.

### New dependency checklist (ADR 0028)

Skip if `Cargo.toml` and `Cargo.lock` are unchanged.

- [ ] Why not own code or an existing dependency?
- [ ] License is on the `deny.toml` allowlist; the crate is maintained.
- [ ] Transitive crates, build time and binary size impact noted (`cargo xtask budgets`).
- [ ] No unsafe/C code or MSRV bump, or the exception is justified.
- [ ] Behind a feature (additive only)? Heavy additions need an ADR.
- [ ] `object_store` stays a single pinned version; the `parquet` crate's `object_store` feature stays off.

### Performance review (ADR 0034)

Mark each item reviewed or not applicable.

- [ ] **Allocations per row**: no allocation, `String`/`Vec` creation or boxing per row in hot loops.
- [ ] **Copies**: no avoidable copies or clones of buffers, batches or Arrow arrays; slices and `Arc` sharing instead.
- [ ] **Dyn dispatch in hot loops**: no `dyn` calls per row; dispatch once per batch or monomorphise.
- [ ] **Locks across `.await`**: no lock guard held across an `.await`.
- [ ] **Blocking in async**: no blocking I/O or CPU-heavy work on async threads (use `spawn_blocking` or the morsel executor).
- [ ] Benchmark or measurement attached when the task has a performance threshold.
