# krishiv-python

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P0-T25 skeleton, P2.

## Purpose
Python bindings (PyO3) and the PyCapsule Arrow bridge; Python UDF runtime.

## Invariants
- Own PyCapsule bridge via `arrow::ffi`; no `pyo3-arrow`/`arrow-pyarrow`.
- abi3-py310 + free-threaded wheels.
- `unsafe` allowed; every block has a `// SAFETY:` comment; runs under Miri in CI and nightly ASan/TSan (ADR 0029).
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0005, 0022, 0028

## Test
`cargo nextest run -p krishiv-python` · `cargo clippy -p krishiv-python --all-targets -- -D warnings`
