# krishiv-udf-sdk

**Layer:** L1 (may depend only on crates in L1 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (Python UDFs), stabilised in P6.

## Purpose
Stable UDF SDK (Rust/WASM/Python UDF boundary types).

## Layout
- `src/lib.rs` crate docs and re-exports · `src/udf.rs` the scalar trait and boundary types (`Signature`, `ArgField`,
  `ReturnField`, `UdfArg`, `InvokeContext`, `UdfError`, `guard_call`) · `src/aggregate.rs` the aggregate traits (`AggregateUdf`, `GroupsAccumulator`, `StateField`, `AggregateProperties`) · `src/library.rs` `UdfLibrary`, a session's
  registrations.
- `public-api.txt` the reviewed public surface (`cargo +nightly-2026-08-12 public-api -ss -p krishiv-udf-sdk`, checked
  nightly); a change here is a public-surface change (ADR 0026, aggregate UDFs: ADR 0045). User page:
  `docs/reference/rust-udfs.md`.

## Invariants
- Small and slowest-changing public surface: semver; Arrow C Data Interface / WASM / PyCapsule at the boundary.
- Depends only on L0–L1 crates.
- `unsafe` allowed; every block has a `// SAFETY:` comment; runs under Miri in CI and nightly ASan/TSan (ADR 0029).
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0022, 0026, 0005, 0045 (aggregate UDFs)

## Test
`cargo nextest run -p krishiv-udf-sdk --no-tests=pass` · `cargo clippy -p krishiv-udf-sdk --all-targets -- -D warnings`
