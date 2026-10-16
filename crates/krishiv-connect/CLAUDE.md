# krishiv-connect

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P3.

## Purpose
Spark Connect gRPC server (protocols 3.5, 4.0, 4.1, 4.2).

## Invariants
- Protos vendored per tag under `proto/spark-connect/<ver>/`; behaviour gated on the client's declared version.
- Krishiv extensions only in `google.protobuf.Any` slots, package `krishiv.v1`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0005

## Test
`cargo nextest run -p krishiv-connect --no-tests=pass` · `cargo clippy -p krishiv-connect --all-targets -- -D warnings`
