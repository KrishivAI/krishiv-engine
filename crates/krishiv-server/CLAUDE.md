# krishiv-server

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Status:** P0-T26 skeleton exists (`router()`, `serve`, `run_blocking`); the real admin API arrives in P3.

## Purpose
Admin and UI API server (REST, OpenAPI), serves the web UI. Today: `/api/v1/version`, JSON 404 for unknown `/api/*` paths, and the embedded SPA (from `krishiv-ui-assets`) with an extension-less-path fallback to `index.html`. Binds loopback by default. Reached via `krishiv server`, which exists only with the `ui` feature of `krishiv`/`krishiv-cli`.

## Key files
Paths are relative to this crate.
- `src/lib.rs` `router()`, `serve` and `run_blocking`
- `tests/serve.rs` server tests

## Extension points
- New route: add it to `router()` in `src/lib.rs` under `/api/v1/` with its OpenAPI entry.

## Invariants
- Admin REST is a versioned surface (OpenAPI via utoipa).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0024, 0004, 0026

## Tests
`cargo nextest run -p krishiv-server --no-tests=pass` · `cargo clippy -p krishiv-server --all-targets -- -D warnings`
