# krishiv

**Layer:** L7 (may depend only on crates in L7 or lower; `cargo xtask layering`).  
**Real code arrives:** P2.

## Purpose
Facade crate: the public Rust API re-exporting the engine.

## Key files
Paths are relative to this crate.
- `src/lib.rs` the re-exports that form the Rust API
- `public-api.txt` the reviewed public surface; a change here is a public-surface change (ADR)
- `Cargo.toml` the `mimalloc` feature (wires `krishiv-alloc` into the CLI)

## Extension points
- New public item: re-export it from `src/lib.rs` and update `public-api.txt` in the same PR.

## Invariants
- Semver after 1.0; the only Rust surface users should depend on.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0005, 0026

## Tests
`cargo nextest run -p krishiv --no-tests=pass` · `cargo clippy -p krishiv --all-targets -- -D warnings`
