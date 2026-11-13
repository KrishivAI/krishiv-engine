# krishiv-connectors-delta

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P3.

## Purpose
Delta Lake connector on `delta_kernel` (no default features) with our own Engine implementation.

## Layout
- `src/error.rs` kernel errors as Krishiv errors (a Krishiv error raised in a handler travels through the kernel and back).
- `src/engine/schema.rs` kernel `StructType`/`DataType`/`Scalar` to Arrow 60 and back (µs timestamps, unshredded Variant, field ids).
- `src/engine/data.rs` `KrishivEngineData`: the kernel's `EngineData`/`GetData` over a `RecordBatch` (port of the kernel's `arrow_data.rs`/`arrow_get_data.rs`).

## Invariants
- `delta_kernel` with `default-features = false`; never `deltalake`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028

## Test
`cargo nextest run -p krishiv-connectors-delta --no-tests=pass` · `cargo clippy -p krishiv-connectors-delta --all-targets -- -D warnings`
