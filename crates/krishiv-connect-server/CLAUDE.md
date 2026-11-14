# krishiv-connect-server

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).
**Real code since:** P3-T001 (design `../krishiv-internal/design/P3-T001-server.md`, approved 2026-11-12).

## Purpose
The Spark Connect gRPC server: the tonic service, the authentication of every RPC, the session manager keyed by
(authenticated caller, session id), the limits applied before admission, and graceful shutdown. Decoding and translation stay in `krishiv-connect`;
the admin HTTP server (`/healthz`, `/readyz`, `/metrics`, UI) is `krishiv-server`; the process around both is
`krishiv server` in `krishiv-cli`.

## Key files
Paths are relative to this crate.
- `src/service/v4_3.pb.rs` **generated** by `tools/connect-protogen` (do not edit): the `SparkConnectService` trait
  (one method per RPC, requests arrive as `krishiv_connect::wire::Decoded<_>`) and the tonic router
- `src/service.rs` `ConnectService`, Krishiv's implementation: `ExecutePlan`, the minimal `ReattachExecute` /
  `ReleaseExecute` (design P3-T011a §3.4: no replay buffer until PR-15a), `ReleaseSession`, `AnalyzePlan`, `Config`,
  `Interrupt`; the other RPCs are `UNIMPLEMENTED` until PR-15; a draining server answers `UNAVAILABLE` ("server
  shutting down")
- `src/analyze.rs` `AnalyzePlan` (design §3.5): every arm over the translated, analyzed plan (never run);
  `same_semantics`/`semantic_hash` from `krishiv_plan::semantic_plan_hash` (names and subquery aliases erased) of
  the analyzed plan; the cache arms are 56K00 until PR-47; `src/analyze/types.rs` ports of `StructType.treeString`, `toDDL` and the planner's
  `parseDatatypeString`
- `src/config.rs` `Config` (design §3.6) over `RuntimeConf`: the seven operations, `silent`, Spark's per-key
  warnings, redaction for both profiles (Spark's pattern plus `krishiv_common::secret`'s classifier and URL
  credentials), `SettableBy` (`check_settable`; `role_of` is the caller's authenticated role, no role a viewer, the local user of an unauthenticated server the admin)
- `src/interrupt.rs` `Interrupt` (design §3.7): `ALL` and `OPERATION_ID` through the session's operations and
  `Session::cancel` per reported id; `TAG` is `UNSUPPORTED_FEATURE.INTERRUPT_TYPE` until T064
- `src/auth.rs` (P3-T002, design `design/P3-T002-authentication.md` §3.1/§3.6, ADR 0046 §2/§3b/§5) `ConnectAuth`;
  `src/auth/layer.rs` `AuthLayer`, the tower layer in front of the router: every RPC (the unimplemented ones too)
  authenticates from its headers before the body is decoded, through the server's one `Authenticator`
  (`ServerAuth`, shared with the admin listener): the `authorization` header (two entries or a non-`Bearer` scheme
  are malformed) and the peer address; failures `UNAUTHENTICATED` with Spark's messages, throttled or unavailable
  `UNAVAILABLE`, no peer address refused, a caller with no role `PERMISSION_DENIED` (42501, audited) before any
  session; the result rides in the request extensions (`ConnectAuth::caller` fails closed without it); auth off =
  the local user. `role_sessions` is the production session factory (`PathPrefixPolicy(RolePolicy(role))`; each session counts its denials into the server's `DeniedCounts` under `connect`, as does the no-role refusal with `ConnectAuth::with_denials`). `SessionOwner` (method, kind, tenant, name,
  no groups) is the session key. `check_rpc` / `check_command`: the §3b class of every RPC and `Command` arm,
  no wildcard arm (a new proto arm fails to compile); `enforce` asks the session's policy for a
  `Check::Privilege` (`Session::authorize_privilege`, one `krishiv::audit` record per denial); `tests/auth.rs`
- `src/operation.rs` `Operation` (one per `ExecutePlan`: id, plan digest, tags, reattachable, sent response ids,
  one atomic state: running, then finished or cancelled by compare-and-swap, whether its encoder waits for the
  client) and `Operations`, the per-session map in `SessionEntry` (`krishiv.connect.maxOperationsPerSession`,
  `KRISHIV_OPERATION_LIMIT`); `ExecuteLimits` (`aheadBytes`, plan and batch sizes); tag checks
  (`KRISHIV_INVALID_OPERATION_TAG`)
- `src/execute.rs` `ExecutePlan` for ROOT relations: compressed plans (bounded ZSTD), translation, the encoder task
  and its byte-bounded channel (`krishiv.connect.execute.aheadBytes`, memory consumer `connect_responses`), the
  response order (schema, batches, metrics, `ResultComplete` when reattachable); commands are 56K00 until T012;
  `src/execute/arrow.rs` Spark's Arrow shapes, batch limits, IPC streams, codecs and chunks;
  `src/execute/schema.rs` the `schema` response (`DataTypeProtoConverter.toConnectProtoType`)
- `src/codec.rs` `WireCodec`: requests decode only through `wire::decode_request` (bounded size, count, depth, fields)
- `src/status.rs` `status_of`: a `KrishivError` as Spark's gRPC status (`INTERNAL`, `google.rpc.ErrorInfo` with
  `classes`, `sqlState`, `errorClass`, `messageParameters`; `ErrorUtils.buildStatusFromThrowable`, v4.3.0;
  `KRISHIV_INSUFFICIENT_PERMISSIONS` is `PERMISSION_DENIED`, ADR 0046 §8);
  `exception_classes` picks the Java class list PySpark's `convert_exception` reads (Spark's own for the Connect
  classes, else the Python facade's class/SQLSTATE rules); `KRISHIV_SESSION_LIMIT` and `KRISHIV_OPERATION_LIMIT`
  are `RESOURCE_EXHAUSTED`
- `src/manager.rs` `SessionManager` (Spark's `SparkConnectSessionManager`; its factory gets the `Authenticated`
  caller; a new session records transport `connect`, the client's `user_id` (unverified, ADR 0046 §5) and its
  server-side id for `system.sessions`): get-or-create / lookup / release, tombstones
  (LRU, no expiry), `SESSION_CHANGED` on a changed server-side id, the idle sweep (skips a session running a query or
  holding a live operation; `busy()` is what shutdown waits for), `krishiv.connect.maxSessions` /
  `maxSessionsPerPrincipal`
- `src/server.rs` `ConnectServer::serve`: HTTP/2 limits, connection cap, sweeper task, shutdown order; `with_tls` (P3-T002 §3.5): the TLS handshake (ALPN `h2`) on the connection's own task within `handshakeTimeout`, `PeerInfo` (address and verified client leaf) as the connection's `ConnectInfo`, which `AuthLayer` passes to the authenticator; `tests/tls.rs`
- `tests/server.rs`, `tests/execute.rs`, `tests/rpcs.rs` (AnalyzePlan, Config, Interrupt), `tests/auth.rs`, `tests/tls.rs`
  end to end over a raw tonic client

## Invariants
- **Sessions are keyed by the authenticated caller's `SessionOwner`**, never by `UserContext.user_id` (ADR 0046
  §5, threat model T-A8); a session id another owner holds is `INVALID_HANDLE.SESSION_NOT_FOUND`, the same answer
  as unknown. Groups are not in the key; the method is (a token and an OIDC subject of one name are two owners).
- **A session's policy is its caller's role at creation, for life**: a caller whose role differs never gets that
  session; it is closed without a tombstone and the lookup goes on as for a free id (a create builds a new one,
  a client that saw the old id gets `INVALID_HANDLE.SESSION_CHANGED`, an RPC that creates nothing
  `SESSION_NOT_FOUND`). `ReleaseSession` ends it whatever the role.
- **Every RPC authenticates before its body is decoded** (`AuthLayer`), once per RPC (tokens expire, connections
  outlive them); a class `check_rpc` gives that needs more than the session lookup or the analyzer is refused
  until its RPC applies it (fail closed).
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
