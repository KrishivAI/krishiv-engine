# krishiv-connect-server

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).
**Real code since:** P3-T001 (design `../krishiv-internal/design/P3-T001-server.md`, approved 2026-11-12).

## Purpose
The Spark Connect gRPC server: the tonic service, the session manager keyed by (authenticated principal, session
id), the limits applied before admission, and graceful shutdown. Decoding and translation stay in `krishiv-connect`;
the admin HTTP server (`/healthz`, `/readyz`, `/metrics`, UI) is `krishiv-server`; the process around both is
`krishiv server` in `krishiv-cli`, which runs Krishiv single-node (deployment modes: embedded, single-node,
distributed; clients connect over Spark Connect).

## Key files
Paths are relative to this crate.
- `src/service/v4_3.pb.rs` **generated** by `tools/connect-protogen` (do not edit): the `SparkConnectService` trait
  (one method per RPC, requests arrive as `krishiv_connect::wire::Decoded<_>`) and the tonic router
- `src/service.rs` `ConnectService`, Krishiv's implementation: `ExecutePlan`, `ReattachExecute` / `ReleaseExecute`
  (design P3-T011a §3.8: resume after `last_response_id`, `release_until` trims, `release_all` closes; the server's
  `ResponseBudget`), `ReleaseSession`, `AnalyzePlan`, `Config`,
  `Interrupt` (each looks the session up and `observe`s the request: `client_type`, user and the Connect transport go to `system.sessions`; `ExecutePlan` also passes the request's tags (`QueryOptions::tags`: they reach `system.queries`, resource-group `clientTags` selectors and cancel by tag) and its `traceparent`/`baggage` metadata (`QueryOptions::inbound_trace`, per request, baggage included via the statement's `Session::baggage`: concurrent requests of one session keep their own trace, a malformed header means no trace and never fails the request, P3-T068; the T068 acceptance, a client span and the engine span sharing one trace id in an OTLP collector, is not tested yet)); the other RPCs are `UNIMPLEMENTED` until PR-15; a draining server answers `UNAVAILABLE` ("server
  shutting down")
- `src/analyze.rs` `AnalyzePlan` (design §3.5): every arm over the translated, analyzed plan (never run);
  `same_semantics`/`semantic_hash` from `krishiv_plan::semantic_plan_hash` (names and subquery aliases erased) of
  the analyzed plan; the cache arms are 56K00 until PR-47; `src/analyze/types.rs` ports of `StructType.treeString`, `toDDL` and the planner's
  `parseDatatypeString`
- `src/config.rs` `Config` (design §3.6) over `RuntimeConf`: the seven operations, `silent`, Spark's per-key
  warnings, redaction for both profiles (Spark's pattern plus `krishiv_common::secret`'s classifier and URL
  credentials), `SettableBy` (`check_settable`; `role_of` makes every principal admin until T002)
- `src/interrupt.rs` `Interrupt` (design §3.7): `ALL` and `OPERATION_ID` through the session's operations and
  `Session::cancel` per reported id; an interrupted operation stays registered (Spark keeps the `ExecuteHolder`), so a
  reattach gets `OPERATION_CANCELED`; `TAG` is `UNSUPPORTED_FEATURE.INTERRUPT_TYPE` until T064
- `src/operation.rs` `Operation` (one per `ExecutePlan`: id, plan digest, tags, reattachable, its `Replay`, one
  atomic state: running, then finished or cancelled by compare-and-swap; `close` = Spark's `ExecuteHolder.close`)
  and `Operations`, the per-session map in `SessionEntry` (`krishiv.connect.maxOperationsPerSession`,
  `KRISHIV_OPERATION_LIMIT`; the same plan under a live id is `Admitted::Existing`); `ExecuteLimits` (`aheadBytes`,
  plan and batch sizes, retry buffer, sender caps, detach timeout, maintenance interval, `maxResponseBytes`); tag
  checks (`KRISHIV_INVALID_OPERATION_TAG`)
- `src/operation/replay.rs` `Replay` (Spark's `ExecuteResponseObserver`): responses numbered from 1, the retry-buffer
  eviction, `release_until`, every response id kept (index lookups), attach generations, the attached sender and its
  deadline (`expire_sender`), detach time; a response holds its ahead permit until consumed and its pool reservation
  until it leaves the buffer; the operation's admission (its share of `maxResponseBytes`) is held until `close`;
  permits and reservations are dropped outside its mutex
- `src/sender.rs` `attach` (Spark's `ExecuteGrpcResponseSender`): one response stream from an index; a newer sender
  or `close` ends it with `INVALID_CURSOR.DISCONNECTED`; reattachable streams end OK without `ResultComplete` at
  `senderMaxStreamSize` / `senderMaxStreamDuration`; dropping it detaches (a non-reattachable operation is closed);
  maintenance detaches a sender past its deadline even if hyper never polls its stream again (a stalled client)
- `src/execute.rs` `ExecutePlan` for ROOT relations: compressed plans (bounded ZSTD), translation, the encoder task
  admitted once against the server's `ResponseBudget` (`krishiv.connect.maxResponseBytes`: its worst case,
  `aheadBytes` + retry buffer, reserved before the first response, held until close), then pushing into the
  operation's `Replay` under the ahead budget (`krishiv.connect.execute.aheadBytes`) and memory consumer
  `connect_responses` (the Arrow payload is `Bytes`, shared with the copy sent), the
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
  (LRU, no expiry), abandoned-operation tombstones and `abandon_detached` (Spark's execution maintenance), `SESSION_CHANGED` on a changed server-side id, the idle sweep (skips a session running a query or
  holding a live operation; `busy()` is what shutdown waits for), `krishiv.connect.maxSessions` /
  `maxSessionsPerPrincipal`
- `src/server.rs` `ConnectServer::serve`: HTTP/2 limits, connection cap, sweeper and abandon tasks, shutdown order
- `tests/server.rs`, `tests/execute.rs`, `tests/rpcs.rs` (AnalyzePlan, Config, Interrupt), `tests/reattach.rs`
  (replay, reattach, caps, detach, races) end to end over a raw tonic client

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
- **The result is never held in full**: the encoder waits for `aheadBytes` of unread responses, attached or not. Only a
  new operation waits on the server's `maxResponseBytes` (admission); a running one never does. `serve` refuses a cap
  below one operation's worst case (`ExecuteLimits::validate`). Dropping a non-reattachable operation's stream (seen at once) cancels its
  query and removes it; a reattachable one detaches and stays until `release_all`, its session's release, or
  `detachedTimeout` (then tombstoned: `OPERATION_ABANDONED`). Live operations, detached or finished but unreleased,
  keep a session from the idle sweep; shutdown waits only for operations still running or read by an attached
  client (`Operation::is_active`).
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0026, 0046

## Tests
`cargo nextest run -p krishiv-connect-server` · `cargo clippy -p krishiv-connect-server --all-targets -- -D warnings`
· the generated stub: `cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --check`
