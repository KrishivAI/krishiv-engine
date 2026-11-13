# krishiv-connect-server

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).
**Real code since:** P3-T001 (design `../krishiv-internal/design/P3-T001-server.md`, approved 2026-11-12).

## Purpose
The Spark Connect gRPC server: the tonic service, the session manager keyed by (authenticated principal, session
id), the limits applied before admission, and graceful shutdown. Decoding and translation stay in `krishiv-connect`;
the admin HTTP server (`/healthz`, `/readyz`, `/metrics`, UI) is `krishiv-server`; the process around both is
`krishiv server` in `krishiv-cli`.

## Key files
Paths are relative to this crate.
- `src/service/v4_3.pb.rs` **generated** by `tools/connect-protogen` (do not edit): the `SparkConnectService` trait
  (one method per RPC, requests arrive as `krishiv_connect::wire::Decoded<_>`) and the tonic router
- `src/service.rs` `ConnectService`, Krishiv's implementation: `ReleaseSession`; every other RPC is `UNIMPLEMENTED`
  until T011a; a draining server answers `UNAVAILABLE` ("server shutting down")
- `src/codec.rs` `WireCodec`: requests decode only through `wire::decode_request` (bounded size, count, depth, fields)
- `src/status.rs` `status_of`: a `KrishivError` as Spark's gRPC status (`INTERNAL`, `google.rpc.ErrorInfo` with
  `classes`, `sqlState`, `errorClass`, `messageParameters`; Spark 4.2.0 `ErrorUtils.buildStatusFromThrowable`);
  `KRISHIV_SESSION_LIMIT` is `RESOURCE_EXHAUSTED`
- `src/manager.rs` `SessionManager` (Spark's `SparkConnectSessionManager`): get-or-create / lookup / release, tombstones
  (LRU, no expiry), `SESSION_CHANGED` on a changed server-side id, the idle sweep (skips a session running a query),
  `krishiv.connect.maxSessions` / `maxSessionsPerPrincipal`
- `src/server.rs` `ConnectServer::serve`: HTTP/2 limits, connection cap, sweeper task, shutdown order
- `tests/server.rs` end to end over a raw tonic client

## Invariants
- **Sessions are keyed by the authenticated principal**, never by `UserContext.user_id` (ADR 0046 §5, threat model
  T-A8); a session id another principal holds is `INVALID_HANDLE.SESSION_NOT_FOUND`, the same answer as unknown.
  Until T002 the principal is a request extension set by the auth layer, else the local user (loopback only).
- **No lock across an `await`**, and none while a session is built or closed: the manager's mutex guards map
  lookups and inserts only. A racing create of the same key keeps the first and closes the loser.
- **Decode only through `wire`** (`#![deny(clippy::disallowed_methods)]`): the codec is the one decode path.
- Shutdown: draining flag (new RPCs `UNAVAILABLE`, `/readyz` 503), stop accepting, queries get
  `krishiv.server.shutdownTimeout`, then every session is closed (queries cancelled). The engine runtime is dropped
  by the caller after `serve` returns, which removes the spill directory.
- tonic refuses a message over `spark.connect.grpc.maxInboundMessageSize` with `OUT_OF_RANGE` from its length
  prefix (grpc-java answers `RESOURCE_EXHAUSTED`): a recorded difference.
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0026, 0046

## Tests
`cargo nextest run -p krishiv-connect-server` · `cargo clippy -p krishiv-connect-server --all-targets -- -D warnings`
· the generated stub: `cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --check`
