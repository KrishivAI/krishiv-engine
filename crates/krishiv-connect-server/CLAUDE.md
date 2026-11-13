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
- `src/service.rs` `ConnectService`, Krishiv's implementation: `ExecutePlan`, `ReattachExecute` / `ReleaseExecute`
  (design P3-T011a §3.8: resume after `last_response_id`, `release_until` trims, `release_all` closes; the server's
  `ResponseBudget`), `ReleaseSession`, `AnalyzePlan`, `Config`,
  `Interrupt`, `AddArtifacts`, `ArtifactStatus`, `FetchErrorDetails`, `CloneSession`, `GetStatus` (all twelve RPCs);
  `ExecutePlan` (until its encoder starts), `AnalyzePlan` and `Config` hold the session's statement gate shared; a
  draining server answers `UNAVAILABLE` ("server shutting down"); an error after the
  session is known goes through `entry.errors().status` (it gets an `errorId`), one before it through `status_of`
- `src/artifacts.rs` `AddArtifacts` / `ArtifactStatus` (design §3.9): the chunk protocol and CRC-32 (`crc32`), names
  from `krishiv_connect::artifact` checked before any file, the privilege gate `config::check_add_artifact`, the
  per-session quota (`krishiv.connect.artifacts.maxBytes`, `maxArtifactBytes`, `ARTIFACT_OVERHEAD` per name segment)
  and the server-wide one (`maxTotalBytes`, checked `>= maxArtifactBytes` at startup)
  counted before each write, `Transfer` (runs on the blocking pool; dropping it removes what it staged and returns its
  quota), `SessionArtifacts` (per session: directory, `cache/<hash>` blobs in the pool as `connect_artifacts`,
  `cached(hash)` for T012; a blob's pool reservation covers its whole capacity; the installed files are indexed, and
  `snapshot` / `copy_from` copy them to a clone, counted against its quota and `maxTotalBytes` first), `ArtifactStore` (the server's root,
  created on first use, removed by the awaited `shutdown` that `serve` calls after closing the sessions); `src/artifacts/dir.rs` the
  private root and session directories, `O_EXCL | O_NOFOLLOW` staging, `linkat` install (never replaces), every name
  resolved per segment from a directory descriptor without following links; removal on the blocking pool
- `src/errors.rs` `SessionErrors` (design §3.10): `errorId` on every status sent on a session unless
  `spark.sql.connect.enrichError.enabled=false`, the per-session cache (20, 60 s after last access, given out once),
  `details` (cause chain ≤ 5, each error in its client-safe form, query contexts, no stack trace)
- `src/analyze.rs` `AnalyzePlan` (design §3.5): every arm over the translated, analyzed plan (never run);
  `same_semantics`/`semantic_hash` from `krishiv_plan::semantic_plan_hash` (names and subquery aliases erased) of
  the analyzed plan; the cache arms are 56K00 until PR-47; `src/analyze/types.rs` ports of `StructType.treeString`, `toDDL` and the planner's
  `parseDatatypeString`
- `src/config.rs` `Config` (design §3.6) over `RuntimeConf`: the seven operations, `silent`, Spark's per-key
  warnings, redaction for both profiles (Spark's pattern plus `krishiv_common::secret`'s classifier and URL
  credentials), `SettableBy` (`check_settable`; `role_of` makes every principal admin until T002)
- `src/interrupt.rs` `Interrupt` (design §3.7): `ALL` and `OPERATION_ID` through the session's operations and
  `Session::cancel`/`interrupt_all`; an interrupted operation stays registered (Spark keeps the `ExecuteHolder`), so a
  reattach gets `OPERATION_CANCELED`; `TAG` is `UNSUPPORTED_FEATURE.INTERRUPT_TYPE` until T064
- `src/operation.rs` `Operation` (one per `ExecutePlan`: id, plan digest, tags, reattachable, its `Replay`, one
  atomic state: running, then finished, failed or cancelled by compare-and-swap, plus `ended` once its encoder
  stopped; `state()` for GetStatus: cancelled and not ended is `Terminating`; `close` = Spark's `ExecuteHolder.close`)
  and `Operations`, the per-session map in `SessionEntry` (an operation that leaves it is remembered with its final
  state: LRU, `ENDED_OPERATIONS_KEPT` 1,000, `ENDED_OPERATION_WINDOW` 30 min after the last lookup; recorded under the
  live lock, so a lookup always finds it in one of the two) (`krishiv.connect.maxOperationsPerSession`,
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
  classes, else the Python facade's class/SQLSTATE rules); `KRISHIV_SESSION_LIMIT`, `KRISHIV_OPERATION_LIMIT`, `KRISHIV_ARTIFACT_LIMIT`
  are `RESOURCE_EXHAUSTED`
- `src/manager.rs` `SessionManager` (Spark's `SparkConnectSessionManager`): get-or-create / lookup / release, tombstones
  (LRU, no expiry), abandoned-operation tombstones and `abandon_detached` (Spark's execution maintenance), `SESSION_CHANGED` on a changed server-side id, the idle sweep (skips a session running a query or
  holding a live operation; `busy()` is what shutdown waits for), `krishiv.connect.maxSessions` /
  `maxSessionsPerPrincipal`
- `src/manager/clone.rs` `SessionManager::clone_session` (design §3.11): target checks (`INVALID_CLONE_SESSION_REQUEST.*`,
  42K04, sent as `SparkSQLException`; another principal's id `SESSION_NOT_FOUND`; the session caps), the source's
  gate held exclusively around `Session::clone_session` and the artifact snapshot, the artifact copy on the blocking
  pool without the gate, the clone published only when complete (`Unpublished` closes it otherwise)
- `src/get_status.rs` `GetStatus` (design §3.12): absent → none, empty → live ∪ ended, ids deduplicated in order,
  unknown → `UNKNOWN`; extensions ignored
- `src/server.rs` `ConnectServer::serve`: HTTP/2 limits, connection cap, sweeper and abandon tasks, shutdown order
- `tests/server.rs`, `tests/execute.rs`, `tests/rpcs.rs` (AnalyzePlan, Config, Interrupt), `tests/reattach.rs`
  (replay, reattach, caps, detach, races), `tests/artifacts.rs` (AddArtifacts, ArtifactStatus, FetchErrorDetails,
  quotas, links, client drops, T-C8), `tests/clone_status.rs` (CloneSession, GetStatus, the clone vs concurrent
  statements race) end to end over a raw tonic client

## Invariants
- **Sessions are keyed by the authenticated principal**, never by `UserContext.user_id` (ADR 0046 §5, threat model
  T-A8); a session id another principal holds is `INVALID_HANDLE.SESSION_NOT_FOUND`, the same answer as unknown.
  Until T002 the principal is a request extension set by the auth layer, else the local user (loopback only).
- **No lock across an `await`**, and none while a session is built or closed: the manager's mutex guards map
  lookups and inserts only. A racing create of the same key keeps the first and closes the loser. The one exception
  is each session's statement gate (`SessionEntry::shared`, a tokio `RwLock`): an RPC holds only its own session's
  gate and takes no other lock while waiting for it; `CloneSession` holds the source's exclusively only around the
  synchronous copy (never across the artifact copy or a stream), so it cannot deadlock and never waits on a client.
  The gate is fair (write-preferring): a clone queued behind a slow `ExecutePlan` setup (`df.schema()` listing many
  files) holds back every later `Config`, `AnalyzePlan` and `ExecutePlan` of that session until that setup ends:
  head-of-line blocking, not a deadlock. Commands (T012) that change session state must run under the shared gate,
  and their side effects must not run in the encoder (after the gate is dropped).
- **Cached artifact blobs copied to a clone are reserved twice in the pool** while both sessions hold them (the
  `Bytes` are shared): deliberate, conservative accounting (review L4); a tight pool can refuse such a clone.
- **GetStatus and the operation states follow design §3.12, not Spark's handler** (unverified, not in the vendored
  sources): a setup failure is `FAILED`, a setup whose RPC was dropped is `CANCELLED`, an interrupt whose encoder still
  runs is `TERMINATING`.
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
  keep a session busy.
- **Artifacts never leave their session's directory**: names are validated before any file exists, every path is
  resolved segment by segment from a directory descriptor with `O_NOFOLLOW`, installs use `linkat` (no replace), and
  every byte is counted against the quota before it is written. A released, swept or shut-down session removes its
  directory; the root goes with the service.
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0026, 0046

## Tests
`cargo nextest run -p krishiv-connect-server` · `cargo clippy -p krishiv-connect-server --all-targets -- -D warnings`
· the generated stub: `cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --check`
