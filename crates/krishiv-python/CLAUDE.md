# krishiv-python

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P0-T25 skeleton, P2.

## Purpose
Python bindings (PyO3) and the PyCapsule Arrow bridge; Python UDF runtime.

## Invariants
- Own PyCapsule bridge via `arrow::ffi`; no `pyo3-arrow`/`arrow-pyarrow`.
- abi3-py311 + free-threaded wheels.
- `unsafe` allowed; every block has a `// SAFETY:` comment; runs under Miri in CI and nightly ASan/TSan (ADR 0029).
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.
- Arrow PyCapsule bridge (`src/stream.rs`): the C-level code is in `src/stream/ffi.rs`, which must stay free of pyo3 so
  `src/stream/tests.rs` can compile it alone (Miri per PR, ASan nightly). Imports are untrusted: callbacks are checked
  before use, every batch is `validate_full`-ed before it is assembled, arrow-rs panics become errors. Exports never
  unwind into a C callback. A consumer's `requested_schema` capsule is only read, never moved out or released. No
  writable memoryviews over Rust buffers.

## ADRs (in `../krishiv-internal/adr/`)
0005, 0022, 0028

## Test
`krishiv-python` sets `test = false` for its lib (the extension module cannot link a test binary); Python tests are under
`python/` (`maturin develop` then `pytest`). `cargo nextest run -p krishiv-python` runs `src/stream/tests.rs` ·
`cargo +nightly-2026-08-12 miri test -p krishiv-python --test stream_ffi` · `cargo clippy -p krishiv-python --all-targets -- -D warnings`
