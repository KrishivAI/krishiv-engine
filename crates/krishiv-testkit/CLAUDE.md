# krishiv-testkit

**Layer:** L7 test infrastructure (`publish = false`; product crates must not depend on it; `cargo xtask layering`).  
**Status:** P0 skeleton (P0-T20..T23); SQL-executing suites are `#[ignore]`d until P2 (SQL) and P4 (streaming).

## Purpose
Conformance harness: Spark `hash()`/`xxhash64()` golden fixtures (`tests/fixtures/spark-<ver>/hash.json`),
`SQLQueryTestSuite` golden enumeration over `tests/spark-golden/<tag>/`, the batch == bounded-stream invariant
harness, and a sqllogictest runner over `tests/sqllogictest/`.

## Invariants
- Golden data comes from a real Spark run or the vendored Spark tag, never hand-written (ADR 0008, 0010).
- Ignored tests name the phase that un-ignores them; do not delete them.
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic` outside tests.

## ADRs (in `../krishiv-internal/adr/`)
0008, 0010, 0015, 0017, 0029

## Test
`cargo nextest run -p krishiv-testkit --no-tests=pass` · `cargo clippy -p krishiv-testkit --all-targets -- -D warnings`
