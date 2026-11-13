# krishiv-server

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Status:** P0-T26 skeleton exists (`router()`, `serve`, `run_blocking`); the real admin API arrives in P3.

## Purpose
Admin and UI API server (REST, OpenAPI), serves the web UI. Today: `/api/v1/version`, JSON 404 for unknown `/api/*` paths, and the embedded SPA (from `krishiv-ui-assets`) with an extension-less-path fallback to `index.html`. Binds loopback by default. Reached via `krishiv server`, which exists only with the `ui` feature of `krishiv`/`krishiv-cli`.

## Key files
Paths are relative to this crate.
- `src/lib.rs` `router()`, `serve` and `run_blocking`
- `src/guard.rs` (P3-T002) `Guard`, the one check every guarded admin-listener route asks: `GuardRequest` (presented token, malformed flag, peer, `GuardSurface`, `Access`: `Authenticated`, `Metrics` = `Monitor` on `System("metrics")`, `Monitor` on `System("system")`, `Operate` on `System("queries")`), `Refusal` (401 / 503 / 403 / 500 fail closed), `TokenGuard` (the start-up token of a server without authentication), `PeerAddr` (the peer `serve_capped` records as `ConnectInfo`: the address for the throttle and, over TLS, the verified client leaf; `peer_of` / `peer_leaf_of`, passed on as `GuardRequest::with_peer_leaf` to `Credentials::peer_leaf`)
- `src/session_guard.rs` (feature `session`) `SessionGuard`: the server's `ServerAuth`, then the ADR 0046 matrix through `RolePolicy` (no role table of its own); a denial writes one `krishiv::audit` record
- `src/internal.rs` the UI backend (P2-T133 part a): `router_with_ui(source, token)` and `router_with_ui_guard(source, guard)` (GET needs `Access::Monitor`, anything else `Access::Operate`: operator), the `UiSource` trait (system-table rows as JSON, cancel), `/api/internal/{health,queries,queries/{id},queries/{id}/cancel,history,compare,resources,events}`; every route needs the token (`Authorization: Bearer` for any method; the `krishiv_ui_token` cookie or `?token=` only for GET), query text is shown with its literals masked, at most 16 event streams share one snapshot a period, every response of that router carries the CSP / `frame-ancestors 'none'` headers, columns are allowlisted per endpoint, `events` is SSE at one event a second
- `src/admin.rs` `krishiv server`'s admin endpoints (design P3-T001 §3.5): `health_routes` (`/healthz` open, `/readyz` 200/503 from a ready flag, fixed bodies; never guarded), `with_limits` (request timeout 30 s, SSE `/events` exempt; body limit 2 MiB), `serve_capped` (connection cap, graceful shutdown); `tests/admin.rs`
- `src/session_source.rs` `SessionSource` (UI over a runtime) and `EmbeddedUi` (serves it from a host process, token + stop on drop)
- `src/metrics.rs` `router_with_metrics`, `router_with_metrics_auth` (with a token, `/metrics` needs `Authorization: Bearer <token>`, compared in constant time; a refusal is a JSON 401 and takes no snapshot; an empty token refuses everyone; a disabled `/metrics` stays a 404 for every caller) and `render` (feature `metrics-prometheus` only). `metrics_route` is the route alone, `metrics_route_guarded` the same behind a `Guard` (`Access::Metrics`, every role), merged into `krishiv server`'s admin router
- `examples/console_capture.rs` runs TPC-H queries through an embedded runtime and serves the UI over it; the website's console captures (`site/capture-console.sh` in the docs repo) use it
- `src/tls.rs` (feature `tls`, P3-T002 design §3.5) `serve_capped_tls`: the admin listener over rustls (`tokio-rustls`), ALPN `http/1.1`, each handshake on a task of its own within `HANDSHAKE_TIMEOUT` (10 s) holding a connection slot, the verified client leaf on `TlsCappedStream::peer_leaf`; `tests/tls.rs` (features `session` + `metrics-prometheus` + `tls`)
- `tests/serve.rs` server tests
- `tests/auth.rs` (features `session` + `metrics-prometheus`) the admin listener with configured tokens: per-role access, 401 / throttle 503, open probes
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
0024, 0004, 0026, 0046

## Tests
`cargo nextest run -p krishiv-server --no-tests=pass` · `cargo clippy -p krishiv-server --all-targets -- -D warnings`
