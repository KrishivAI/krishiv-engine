# krishiv-connectors-delta

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).  
**Real code arrives:** P3.

## Purpose
Delta Lake connector on `delta_kernel` (no default features) with our own Engine implementation.

## Key files
- `src/error.rs` kernel errors as Krishiv errors (`from_kernel`; design `P3-T021-delta.md` §3.7)
- `src/engine/data.rs`, `get_data.rs` `EngineData`/`GetData` over Arrow 60 (one `Getter` enum; orphan rule); `schema.rs` kernel types to Arrow, scalars to arrays
- `src/engine/storage.rs` kernel `StorageHandler` over `IoContext`, bound to one store and table root (URLs outside the root refused)

## Invariants
- `delta_kernel` with `default-features = false`; never `deltalake`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028

## Test
`cargo nextest run -p krishiv-connectors-delta --no-tests=pass` · `cargo clippy -p krishiv-connectors-delta --all-targets -- -D warnings`
