---
name: krishiv-operator
description: Use when adding or changing a physical/logical operator in Krishiv (scan, filter, project, hash/sort-merge join, aggregate, sort, window, exchange, limit, union, streaming/IVM operators). Covers semantics goldens, memory accounting and spill, cancellation, metrics and batch==stream.
---

# Add an operator (ADR 0006, 0007, 0011, 0013, 0029)

Read by path (`../krishiv-internal/`): `adr/0006-execution-model.md`, `0007-memory-management.md`,
`0011-batch-operators-and-algorithms.md`, `0012-optimizer.md` (planning rules), `0013-batch-fault-tolerance-and-exchange.md`
(exchanges), `0015`–`0017` (streaming/IVM operators), `0020-partitioning.md`; `spec/03-batch-execution.md`;
`crates/krishiv-exec/CLAUDE.md`, `crates/krishiv-plan/CLAUDE.md`, `crates/krishiv-memory/CLAUDE.md`.
Hard-subsystem operators (hash tables, spill, exchange, IVM) are **Opus** tasks.

## Checklist
1. **Semantics.** spark-semantics agent writes goldens for the Spark-observable behaviour (null handling in join
   keys, NaN/-0.0 grouping and ordering, output ordering guarantees, `EXPLAIN` shape if exposed) at each supported tag.
2. **Plan side.** Logical node in `krishiv-plan`, rules in `krishiv-optimizer`, `insta` snapshots for plans/EXPLAIN.
3. **Failing tests first:** correctness vs goldens; `proptest` against a naive reference implementation; tiny-memory
   tests that force **every spill path**; cancellation mid-stream; empty input; all-null keys; skewed keys (Zipf).
4. **Implement** in `krishiv-exec` behind the `dyn` operator boundary (generics inside kernels). Rules:
   - Every allocation is accounted through `krishiv-memory` reservations; memory pools are empty at query end (test).
   - Spill is correct and bounded; spill files carry a format version (ADR 0026).
   - No blocking in async; cooperative yielding; cancellation-safe across `.await`.
   - Observable hashing uses Murmur3 seed 42 only where partitioning is user-visible (ADR 0010); otherwise the fast hash.
   - Metrics: rows/batches in-out, time, peak memory, spill bytes (feeds `EXPLAIN ANALYZE`).
5. **Streaming/IVM operators:** batch == stream invariant test; IVM property Σ incremental == query(Σ inputs);
   state through `krishiv-state` traits with versioned format and N−1 restore test.
6. **Verify:** `cargo nextest run -p krishiv-exec --no-tests=pass ...`, clippy `-D warnings`, `cargo xtask layering`; loom for new
   concurrency primitives; criterion bench + perf-analyst comparison if the plan states a threshold.
7. **Review:** reviewer agent with full stage 2 (unsafe/concurrency/memory).

## Never
Hold unaccounted memory, panic on OOM (return a typed error or spill), or change an ADR-fixed algorithm choice
without the experiment + ADR update (`.claude/skills/krishiv-experiment/SKILL.md`, `krishiv-adr`).
