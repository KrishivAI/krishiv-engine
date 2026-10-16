# krishiv-server

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P3.

## Purpose
Admin and UI API server (REST, OpenAPI), serves the web UI.

## Invariants
- Admin REST is a versioned surface (OpenAPI via utoipa).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0024, 0004, 0026

## Test
`cargo nextest run -p krishiv-server` · `cargo clippy -p krishiv-server --all-targets -- -D warnings`
