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
- `src/service.rs` `ConnectService`, Krishiv's implementation: `ExecutePlan`, the minimal `ReattachExecute` /
  `ReleaseExecute` (design P3-T011a §3.4: no replay buffer until PR-15a), `ReleaseSession`; the other RPCs are
  `UNIMPLEMENTED` until PR-11b/PR-15; a draining server answers `UNAVAILABLE` ("server shutting down")
- `src/operation.rs` `Operation` (one per `ExecutePlan`: id, plan digest, tags, reattachable, sent response ids,
  cancel) and `Operations`, the per-session map in `SessionEntry` (`krishiv.connect.maxOperationsPerSession`,
  `KRISHIV_OPERATION_LIMIT`); `ExecuteLimits` (`aheadBytes`, plan and batch sizes); tag checks
  (`KRISHIV_INVALID_OPERATION_TAG`)
- `src/execute.rs` `ExecutePlan` for ROOT relations: compressed plans (bounded ZSTD), translation, the encoder task
  and its byte-bounded channel (`krishiv.connect.execute.aheadBytes`, memory consumer `connect_responses`), the
  response order (schema, batches, metrics, `ResultComplete` when reattachable); commands are 56K00 until T012;
  `src/execute/arrow.rs` Spark's Arrow shapes, batch limits, IPC streams, codecs and chunks;
  `src/execute/schema.rs` the `schema` response (`DataTypeProtoConverter.toConnectProtoType`)
- `src/codec.rs` `WireCodec`: requests decode only through `wire::decode_request` (bounded size, count, depth, fields)
- `src/status.rs` `status_of`: a `KrishivError` as Spark's gRPC status (`INTERNAL`, `google.rpc.ErrorInfo` with
  `classes`, `sqlState`, `errorClass`, `messageParameters`; `ErrorUtils.buildStatusFromThrowable`, v4.3.0);
  `exception_classes` picks the Java class list PySpark's `convert_exception` reads (Spark's own for the Connect
  classes, else the Python facade's class/SQLSTATE rules); `KRISHIV_SESSION_LIMIT` and `KRISHIV_OPERATION_LIMIT`
  are `RESOURCE_EXHAUSTED`
- `src/manager.rs` `SessionManager` (Spark's `SparkConnectSessionManager`): get-or-create / lookup / release, tombstones
  (LRU, no expiry), `SESSION_CHANGED` on a changed server-side id, the idle sweep (skips a session running a query or
  holding a live operation; `busy()` is what shutdown waits for), `krishiv.connect.maxSessions` /
  `maxSessionsPerPrincipal`
- `src/server.rs` `ConnectServer::serve`: HTTP/2 limits, connection cap, sweeper task, shutdown order
- `tests/server.rs`, `tests/execute.rs` end to end over a raw tonic client

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
- **Operations live in their session**: an operation id is looked up only in the caller's session (another
  principal's id is `OPERATION_NOT_FOUND`); a released or swept session cancels its operations first.
- **The result is never held in full**: the encoder waits for `aheadBytes` of room; dropping the response stream
  (seen at once, also before the first batch) cancels the query and removes the operation (no tombstone until
  PR-15a).
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0026, 0046

## Tests
`cargo nextest run -p krishiv-connect-server` · `cargo clippy -p krishiv-connect-server --all-targets -- -D warnings`
· the generated stub: `cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --check`
