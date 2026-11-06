# krishiv-server

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Status:** P0-T26 skeleton exists (`router()`, `serve`, `run_blocking`); the real admin API arrives in P3.

## Purpose
Admin and UI API server (REST, OpenAPI), serves the web UI. Today: `/api/v1/version`, JSON 404 for unknown `/api/*` paths, and the embedded SPA (from `krishiv-ui-assets`) with an extension-less-path fallback to `index.html`. Binds loopback by default. Reached via `krishiv server`, which exists only with the `ui` feature of `krishiv`/`krishiv-cli`.

## Key files
Paths are relative to this crate.
- `src/lib.rs` `router()`, `serve` and `run_blocking`
- `src/internal.rs` the UI backend (P2-T133 part a): `router_with_ui(source, token)`, the `UiSource` trait (system-table rows as JSON, cancel), `/api/internal/{health,queries,queries/{id},queries/{id}/cancel,history,compare,resources,events}`; every route needs the token (Bearer, `krishiv_ui_token` cookie or `?token=`), every response of that router carries the CSP / `frame-ancestors 'none'` headers, columns are allowlisted per endpoint, `events` is SSE at one event a second
- `src/metrics.rs` `router_with_metrics` and `render` (feature `metrics-prometheus` only)
- `tests/serve.rs` server tests
- `tests/metrics.rs` `/metrics` exposition tests (feature `metrics-prometheus` only)

## Extension points
- A page's data: add a handler in `src/internal.rs` reading through `UiSource` and an allowlist of columns; a new system table is a `SystemTable` variant the source must serve.
- New route: add it to `router()` in `src/lib.rs` under `/api/v1/` with its OpenAPI entry.

## Invariants
- Admin REST is a versioned surface (OpenAPI via utoipa).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.
- The metrics exporter only reads snapshots (`SnapshotProvider`): it never
  holds the registry, and `/metrics` is registered only with the
  `metrics-prometheus` feature and `krishiv.metrics.enabled` (else 404).

## ADRs (in `../krishiv-internal/adr/`)
0024, 0004, 0026

## Tests
`cargo nextest run -p krishiv-server --no-tests=pass` · `cargo clippy -p krishiv-server --all-targets -- -D warnings`
