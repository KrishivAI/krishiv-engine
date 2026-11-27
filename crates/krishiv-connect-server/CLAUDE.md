# krishiv-connect-server

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).
**Real code since:** P3-T001 (design `../krishiv-internal/design/P3-T001-server.md`, approved 2026-11-12).

## Purpose
The Spark Connect gRPC server: the tonic service, the authentication of every RPC, the session manager keyed by
(authenticated caller, session id), the limits applied before admission, and graceful shutdown. Decoding and translation stay in `krishiv-connect`;
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
  `Interrupt`, `AddArtifacts`, `ArtifactStatus`, `FetchErrorDetails`, `CloneSession`, `GetStatus` (all twelve RPCs;
  each looks the session up and `observe`s the request: `client_type`, user and the Connect transport go to `system.sessions`; `ExecutePlan` also passes the request's tags (`QueryOptions::tags`: they reach `system.queries`, resource-group `clientTags` selectors and cancel by tag) and its `traceparent`/`baggage` metadata (`QueryOptions::inbound_trace`, per request, baggage included via the statement's `Session::baggage`: concurrent requests of one session keep their own trace, a malformed header means no trace and never fails the request, P3-T068; the T068 acceptance, a client span and the engine span sharing one trace id in an OTLP collector, is not tested yet));
  `AddArtifacts` applies its class itself (`caller_checked`, then `auth::enforce` on the session before any artifact is staged);
  `ExecutePlan` (until its encoder starts), `AnalyzePlan` and `Config` hold the session's statement gate shared; a
  draining server refuses new work with `KRISHIV_SERVER_DRAINING` (held work finishes until the drain deadline); an error after the
  session is known goes through `entry.errors().status` (it gets an `errorId`), one before it through `status_of`
- `src/artifacts.rs` `AddArtifacts` / `ArtifactStatus` (design §3.9): the chunk protocol and CRC-32 (`crc32`), names
  from `krishiv_connect::artifact` checked before any file (who may add any artifact, `cache/` included, is the RPC's
  class: client code, `Usage` on `System("udf")`), the
  per-session quota (`krishiv.connect.artifacts.maxBytes`, `maxArtifactBytes`, `ARTIFACT_OVERHEAD` per name segment)
  and the server-wide one (`maxTotalBytes`, checked `>= maxArtifactBytes` at startup)
  counted before each write, `Transfer` (runs on the blocking pool; dropping it removes what it staged and returns its
  quota), `SessionArtifacts` (per session: directory, `cache/<hash>` blobs in the pool as `connect_artifacts`,
  `cached(hash)` for T012; a blob's pool reservation covers its whole capacity; the installed files are indexed, and
  `snapshot` / `copy_from` copy them to a clone, counted against its quota and `maxTotalBytes` first; `python_artifacts`: the installed `pyfiles/` and `archives/` files as absolute paths (each archive with its `#fragment`, `PythonArchive`), which `SessionPython::sync_artifacts` hands to the session's UDF runtime, the whole set when it changed, after every transfer (a failed one too: it may have installed some) and after a clone's copy, never after the session's release; `tests/artifacts.rs` `python_artifacts_reach_the_sessions_udf_runtime_and_its_clones`, `a_transfer_that_fails_still_hands_over_what_it_installed`), `ArtifactStore` (the server's root,
  created on first use, removed by the awaited `shutdown` that `serve` calls after closing the sessions); `src/artifacts/dir.rs` the
  private root and session directories, `O_EXCL | O_NOFOLLOW` staging, `linkat` install (never replaces), every name
  resolved per segment from a directory descriptor without following links; removal on the blocking pool
- `src/errors.rs` `SessionErrors` (design §3.10): `errorId` on every status sent on a session unless
  `spark.sql.connect.enrichError.enabled=false`, the per-session cache (20, 60 s after last access, given out once),
  `details` (cause chain ≤ 5, each error in its client-safe form, query contexts, no stack trace)
- `src/analyze.rs` `AnalyzePlan` (design §3.5): every arm over the translated, analyzed plan (never run);
  `same_semantics`/`semantic_hash` from `krishiv_plan::semantic_plan_hash` (names and subquery aliases erased) of
  the analyzed plan; `persist`, `unpersist` and `get_storage_level` are the frame's own (`src/cache.rs` converts the protocol's `StorageLevel` both ways); access (ADR 0046 §3b): `schema`, `tree_string`, `is_local`, `is_streaming`, `semantic_hash`, `same_semantics` analyze for metadata (`Browse` on catalog tables and views; reads by path keep `Select`, inference reads file contents), `input_files` keeps `Select`; `explain` is `DataFrame::explain_for_metadata` (P3 PR-14d: the ordinary text when the `Select` check passes, else the plan with statistics withheld; `cost` keeps `Select`; `tests/analyze_access.rs`); `src/analyze/types.rs` ports of `StructType.treeString`, `toDDL` and the planner's
  `parseDatatypeString`
- `src/config.rs` `Config` (design §3.6) over `RuntimeConf`: the seven operations, `silent` (a failed pair's warning
  shows a redacted value as `Some(*********(redacted))`, `failed_to_set`; Spark echoes it), Spark's per-key
  warnings, redaction for both profiles (Spark's pattern plus `krishiv_common::secret`'s classifier and URL
  credentials), who may `set`/`unset` a key (a `Startup` key is `CANNOT_MODIFY_STATIC_CONFIG` for everyone, `check_static`; any other key asks the session's policy as SQL `SET` does, `Session::authorize_set_key`: admin keys and server-scoped store keys need `Manage`, refusals audited and counted; no listener-local role check, ADR 0046)
- `src/interrupt.rs` `Interrupt` (design §3.7): `ALL` and `OPERATION_ID` through the session's operations and
  `Session::cancel` per reported id; an interrupted operation stays registered (Spark keeps the `ExecuteHolder`), so a
  reattach gets `OPERATION_CANCELED`; `TAG` is `UNSUPPORTED_FEATURE.INTERRUPT_TYPE` until T064
- `src/auth.rs` (P3-T002, design `design/P3-T002-authentication.md` §3.1/§3.6, ADR 0046 §2/§3b/§5) `ConnectAuth`;
  `src/auth/layer.rs` `AuthLayer`, the tower layer in front of the router: every RPC (the unimplemented ones too)
  authenticates from its headers before the body is decoded, through the server's one `Authenticator`
  (`ServerAuth`, shared with the admin listener): the `authorization` header (two entries or a non-`Bearer` scheme
  are malformed; on `Surface::FlightSql` only, `Basic` is decoded by `ConnectAuth::authenticate` in `src/auth.rs`:
  the password as the bearer, the user name bound to the principal, P3-T101 note §3.1) and the peer address; failures `UNAUTHENTICATED` with Spark's messages, throttled or unavailable
  `UNAVAILABLE`, no peer address refused, a caller with no role `PERMISSION_DENIED` (42501, audited) before any
  session; the result rides in the request extensions (`ConnectAuth::caller` fails closed without it); auth off =
  the local user. The layer takes the request's `Surface` from its path (`surface_of_path`: `/arrow.flight.protocol.FlightService/` is
  `FlightSql`, anything else `Connect`; P3-T101 §2 H2) and passes it to `ConnectAuth::authenticate` (credentials and failure counter,
  no-role record and denial, act-as records) and into the extensions (`request_surface`, read by `caller`'s refusal log);
  `tests/flight_surface.rs` checks each site per surface. `role_sessions` is the production session factory (`PathPrefixPolicy(RolePolicy(role))`; each session counts its denials into the server's `DeniedCounts` under its transport's surface (`connect`, `flight-sql`, or `ui` for the UI editor's sessions, which `krishiv-cli` builds with `Transport::Ui`; they counted under `connect` before P3-T101), as does the no-role refusal with `ConnectAuth::with_denials`). `SessionOwner` (method, kind, tenant, name,
  `delegated_by`; no groups, except an impersonated principal's, which pin its act-as identity) is the session key.
  `src/auth/act_as.rs` (P3-T061 I2/I3, note amendment A12) trusted impersonation, per RPC in the layer: only
  `x-krishiv-act-as-user` (+ `-groups`, `-tenant`, URL-encoded) or `x-krishiv-act-as-token` asks, never
  `UserContext.user_id`; decided by `ServerAuth::impersonate` / `impersonate_signed` from the connection's own peer
  (`ConnectAuth::with_impersonation`; without it, or with authentication off, act-as is refused); unreadable metadata,
  or plain metadata that disagrees with the assertion beside it, is refused through `refuse_impersonation`; every
  refusal `KRISHIV_AUTH_IMPERSONATION_DENIED`, `PERMISSION_DENIED`; `tests/impersonation.rs` (its `signed` module, feature `oidc`: assertions against a loopback JWK set, metadata beside them that disagrees refused). `check_rpc` / `check_command`: the §3b class of every RPC and `Command` arm,
  no wildcard arm (a new proto arm fails to compile); `enforce` asks the session's policy for a
  `Check::Privilege` (`Session::authorize_privilege`, one `krishiv::audit` record per denial); `tests/auth.rs`
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
  response order (schema, batches, metrics, `ResultComplete` when reattachable); `sql_command` (`spark.sql()`) runs as a command (tags and trace carried), `register_function` with a Python UDF runs (no response but the end), `checkpoint_command` (`src/execute/cache_command.rs`: `DataFrame::checkpoint` / `local_checkpoint`, the result held under a UUID in `SessionEntry`'s map and answered as `checkpoint_command_result`) and `remove_cached_remote_relation_command` (drops the id, an unknown one is fine) run; a `cached_remote_relation` is looked up at translation (`Requested::bind`, `CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND` when absent) and spliced in by `Session::from_plan_over` (the new frame owns what the held frames own; a read inside a subquery expression is not replaced; a lazy checkpoint's rows are taken by the first analysis that reads it, `AnalyzePlan` included, a recorded difference from Spark pinned by `tests/cache.rs`); `persist`, `unpersist` and `checkpoint_command` need the operator floor (`Check::SideEffect`: `Operate` on `System("cache")`, `tests/auth.rs`); `write_operation` and `write_operation_v2` (`src/execute/write_command.rs`, PR-24: the session's `write_to_with` / `write_table_with` / `write_to_table_with`, the same calls the Python facade makes; `save()` without a path takes the `path` option, else `_LEGACY_ERROR_TEMP_2050`; the writers' own checks apply; `tests/write.rs`) run; `SessionEntry::close` drops the held frames; other commands are 56K00 until T012;
  `request_plan` (shared with `AnalyzePlan`) translates twice when the plan holds local relations: the first pass records
  them, `Session::local_relation` builds them (a cached one is read from the requesting session's own `cache/<hash>`
  artifact, never from the relation store; `CANNOT_FIND_CACHED_LOCAL_RELATION` without one), the second pass resolves them (`krishiv-connect` CLAUDE.md); `clean_cache` (PySpark's exit cleanup) and a `register_function` that ran end
  through `Work::NoResult` (only `ResultComplete`); a drop guard (`Unsettled`) settles an operation whose encoder
  task unwinds (`INTERNAL_ERROR`, or `OPERATION_CANCELED` after an interrupt), so no stream waits on an orphan;
  `src/execute/hooks.rs` (cfg(test) only) injects an encoder panic or a held preparation per operation id, used by
  `src/execute/fault_tests.rs`;
  `src/execute/catalog_call.rs` (PR-46) the `Catalog` relation: the `CatalogCall` the translator recorded runs on `Session::catalog()` while the plan is prepared (`build_local_relation`), its rows (Spark's `Database`, `Table`, `Column`, `Function`, `CatalogMetadata`, `TablePartition`, a `value` string or boolean, or the empty relation of a call that returns nothing; names and nullability as Spark's encoders) held as a local relation; `cache_table`, `uncache_table` and `clear_cache` need the operator floor (`Check::SideEffect`); `tests/catalog.rs`;
  `src/execute/arrow.rs` Spark's Arrow shapes, batch limits, IPC streams, codecs and chunks;
  `src/execute/schema.rs` the `schema` response (`DataTypeProtoConverter.toConnectProtoType`)
- `src/udf.rs` (P3-T014, PR-26d part 1) Python UDFs: `UdfHost` (the server's one pool, `krishiv_udf_pool::Pool` on
  Linux, set with `SessionManager::with_udf_host`; `krishiv server` builds it from its config) and `SessionPython`, each
  session's UDFs: its runtime (`SessionUdfs`, owner = principal and server-side id) is installed on the session only at
  the first passed `Usage` check on `System("udf")` (`register_function`'s command class; a plan with an inline UDF
  checked in `execute::execution_plan` after translation, before anything is registered), released with the session;
  `InlineUdfs` is the translator's host (inline UDFs as the session's encoded unregistered-UDF names, one version per
  distinct definition by digest; `AnalyzePlan` types them, registers nothing, needs no privilege; the UDFs of
  `mapInPandas`/`applyInPandas`/cogroup relations (PR-26d part 2, `relation_python_udf`) take a version the same way
  under the same check and run on the session's workers (the pool's `prepare_relation`); `tests/udf.rs`
  `a_relation_udf_is_client_code_and_is_typed_without_a_privilege`, `admin_runs_map_in_pandas_on_the_pool`,
  `admin_runs_map_in_arrow`, `admin_runs_each_grouped_map_kind_per_group`, `admin_runs_each_cogrouped_map_kind`); `copy_for` gives a
  `CloneSession` clone a runtime of its own (`manager/clone.rs`); `SessionUdfRuntime::set_artifacts` takes the session's Python artifacts (default: ignored; the pool's `SessionUdfs::set_artifacts` arrives with PR-26c-3); `tests/udf.rs` (real workers with
  `KRISHIV_TEST_UDF_PYTHON`)
- `src/codec.rs` `WireCodec`: requests decode only through `wire::decode_request` (bounded size, count, depth, fields)
- `src/status.rs` `status_of`: a `KrishivError` as Spark's gRPC status (`INTERNAL`, `google.rpc.ErrorInfo` with
  `classes`, `sqlState`, `errorClass`, `messageParameters`; `ErrorUtils.buildStatusFromThrowable`, v4.3.0;
  `KRISHIV_INSUFFICIENT_PERMISSIONS` is `PERMISSION_DENIED`, ADR 0046 §8; `PYTHON_EXCEPTION` carries Spark's `PythonException` classes, `KRISHIV_PYTHON_WORKER_EXITED` `SparkException`);
  `exception_classes` picks the Java class list PySpark's `convert_exception` reads (Spark's own for the Connect
  classes, else the Python facade's class/SQLSTATE rules); `KRISHIV_SESSION_LIMIT`, `KRISHIV_OPERATION_LIMIT`, `KRISHIV_ARTIFACT_LIMIT`
  are `RESOURCE_EXHAUSTED`
- `src/manager.rs` `SessionManager` (Spark's `SparkConnectSessionManager`; its factory gets the `Authenticated`
  caller and the `Transport`; a new session records transport `connect` (`get_or_create_with` / `get_with` take the
  transport / surface for Flight SQL, P3-T101 §2 H3: `system.sessions`, the factory's denial surface, the foreign-id
  record), `is_live` / `release_owned` (H5: owner-keyed, no access stamp, no foreign-id record, abandoned tombstones only
  for reattachable operations), `SessionOwner::cap_key` public (H6), the client's `user_id` (unverified, ADR 0046 §5) and its
  server-side id for `system.sessions`): get-or-create / lookup / release, tombstones
  (LRU, no expiry), abandoned-operation tombstones and `abandon_detached` (Spark's execution maintenance),
  `SESSION_CHANGED` on a changed server-side id, a `krishiv::audit` record (not counted) for a
  session or operation id another owner holds (ADR 0046 §3b; `audit_foreign_operation`, `tests/foreign_handles_audit.rs`), the idle sweep (skips a session running a query or
  holding a live operation; `busy()` is what shutdown waits for), `krishiv.connect.maxSessions` /
  `maxSessionsPerPrincipal`
- `src/manager/clone.rs` `SessionManager::clone_session` (design §3.11): target checks (`INVALID_CLONE_SESSION_REQUEST.*`,
  42K04, sent as `SparkSQLException`; another principal's id `SESSION_NOT_FOUND`; the session caps), the source's
  gate held exclusively around `Session::clone_session` and the artifact snapshot, the artifact copy on the blocking
  pool without the gate, the clone published only when complete (`Unpublished` closes it otherwise)
- `src/get_status.rs` `GetStatus` (design §3.12): absent → none, empty → live ∪ ended, ids deduplicated in order,
  unknown → `UNKNOWN`; extensions ignored
- `src/server.rs` `ConnectServer::serve`: HTTP/2 limits, connection cap, sweeper and abandon tasks, shutdown order; `with_routes` (P3-T101 §2 H1) adds further tonic services (Flight SQL, composed by `krishiv-cli`; Connect never depends on Flight) under the same builder limits, TLS, `AuthLayer` and shutdown; `with_tls` (P3-T002 §3.5): the TLS handshake (ALPN `h2`) on the connection's own task within `handshakeTimeout`, `PeerInfo` (address and verified client leaf) as the connection's `ConnectInfo`, which `AuthLayer` passes to the authenticator; `tests/tls.rs`
- `src/local.rs` `LocalServer` (P3-T016, `krishiv.start_local()`; no failure throttle: every local client is peer 127.0.0.1, so wrong tokens from another process would lock the owner out): a Connect server on `127.0.0.1:0` and `[::1]` (same port; skipped without IPv6: `localhost` may mean `::1`, which no other process may hold) on a thread of its own (current-thread runtime), authenticated by a token made for each start (`spark.connect.authenticate.token`, 64 hex digits, SHA-256 digest compared in constant time by `ServerAuth`), sessions from the base session (`SessionManager::from_base`, no path confinement: the embedded API's rights); `connection_string()` is `sc://localhost:<port>/;token=<token>` (`localhost`, not the IP: pyspark-client sends a token over a plain channel only to that host name); `stop()`/`Drop` drain, join the thread and drop the engine runtime (its compute/I/O pool threads leave just after, `shutdown_background`); `tests/local.rs`
- `tests/server.rs`, `tests/execute.rs`, `tests/rpcs.rs` (AnalyzePlan, Config, Interrupt), `tests/analyze_access.rs` (AnalyzePlan under `RolePolicy` viewer/operator), `tests/reattach.rs`
  (replay, reattach, caps, detach, races), `tests/auth.rs`, `tests/tls.rs`, `tests/artifacts.rs` (AddArtifacts,
  ArtifactStatus, FetchErrorDetails, quotas, links, client drops, T-C8), `tests/clone_status.rs` (CloneSession,
  GetStatus, the clone vs concurrent statements race) end to end over a raw tonic client; `src/execute/fault_tests.rs`
  the same with the test-only fault points (encoder panic, cancel or release during plan preparation)

## Invariants
- **Sessions are keyed by the authenticated caller's `SessionOwner`**, never by `UserContext.user_id` (ADR 0046
  §5, threat model T-A8); a session id another owner holds is `INVALID_HANDLE.SESSION_NOT_FOUND`, the same answer
  as unknown. Groups are not in the key; the method is (a token and an OIDC subject of one name are two owners).
  An impersonated principal's key also holds its delegator and its groups (P3-T061 A4, rule 5): alice through a
  front end, alice herself and alice through another front end are three owners, and a changed act-as identity
  finds no session.
- **A session's policy is its caller's role at creation, for life**: a caller whose role differs never gets that
  session; it is closed without a tombstone and the lookup goes on as for a free id (a create builds a new one,
  a client that saw the old id gets `INVALID_HANDLE.SESSION_CHANGED`, an RPC that creates nothing
  `SESSION_NOT_FOUND`). `ReleaseSession` ends it whatever the role.
- **Every RPC authenticates before its body is decoded** (`AuthLayer`), once per RPC (tokens expire, connections
  outlive them); a class `check_rpc` gives that needs more than the session lookup or the analyzer is refused
  until its RPC applies it (fail closed).
- **No lock across an `await`**, and none while a session is built or closed: the manager's mutex guards map
  lookups and inserts only. Lock order: the manager's, then a session's `Operations` (never the reverse). A racing create of the same key keeps the first and closes the loser. The one exception
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
  runs is `TERMINATING`. An operation's final state is decided (interrupted) before it is recorded; one known window
  remains: a release that records `SUCCEEDED` between a failed last response and `fail_finished` (the record says
  `SUCCEEDED`, the removed operation `FAILED`; needs a pool refusal on the last response).
- **Decode only through `wire`** (`#![deny(clippy::disallowed_methods)]`): the codec is the one decode path.
- Shutdown: draining flag (new work `KRISHIV_SERVER_DRAINING`, `/readyz` 503), stop accepting, queries get
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
- **Artifacts never leave their session's directory**: names are validated before any file exists, every path is
  resolved segment by segment from a directory descriptor with `O_NOFOLLOW`, installs use `linkat` (no replace), and
  every byte is counted against the quota before it is written. A released, swept or shut-down session removes its
  directory; the root goes with the service.
- **Client code runs only after its `Usage` check on `System("udf")`** (ADR 0046 §3b, N1): a session that never
  passed one has no Python UDF runtime, so no plan of it starts a worker.
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0026, 0046

## Tests
`cargo nextest run -p krishiv-connect-server` · `cargo clippy -p krishiv-connect-server --all-targets -- -D warnings` · feature `oidc` (signed act-as assertions): `--features oidc`
· the generated stub: `cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --check`

## Access providers (ADR 0049)
`auth.rs` `provider_sessions(base, config, denials, Option<ExtraProvider>)`: `role_sessions` with another provider, each caller's policy `PathPrefixPolicy(IntersectionPolicy[role matrix, extra])`; `ExtraProvider::roles_decide_data` false wraps the role matrix in `SystemOnlyPolicy`. `krishiv-cli` passes the `external` provider here.

## Persisted Python UDFs (P3-T072c)
`udf.rs` `FunctionStore` (list/put/owner/remove, `StoreChange`, keyed by `SessionPython::store_key`: `t:<tenant>`, else `p:<principal>` on an authenticated server, else the empty key of the auth-off local principal; blocking, run on `spawn_blocking`), `SessionPython::{with_store, load_pending, load_persisted, persist, skip_load}`; `function_store.rs` (feature `function-catalog`) `SqliteFunctionStore` over `krishiv-catalogs` `python_functions` (a payload that fails its stored SHA-256 digest is skipped). The `krishiv.v1.PersistPythonFunction` command extension (class `client_code()`, as `register_function`) stores the session's registered UDF or, with `drop`, removes the stored one; replacing or dropping a function another principal stored needs `Manage` on `System("cluster")` (audited refusal; the store checks the owner inside the write transaction: `put`/`remove` take `override_owner` and answer `StoreChange::OwnedByOther`, then `persist` checks `Manage` and retries with the override). A session loads its functions once, at its first `ExecutePlan` or `AnalyzePlan` (`execute.rs` `load_persisted_udfs`; every concurrent first request waits for the one load, `SessionPython::loading`), only if the principal passes the `Usage` on `System("udf")` check (one check per session, so about one audit record for a principal without it: concurrent first requests may each check; a catalog read failure is retried at the next request); a name the session registered itself is kept; a row this build cannot convert is skipped. Tests: `udf/tests.rs` (fake store: tenants, owner, untenanted keys, concurrent load), `tests/udf.rs` `a_persisted_udf_survives_a_restart_and_is_typed_by_the_first_analysis` (`--features function-catalog`, Connect service, real catalog file, fake host; another principal of the same file does not see it; a tenant needs act-as, covered only by the unit tests). SQL (P3-T072d): `CREATE [OR REPLACE] FUNCTION [IF NOT EXISTS] f AS 'python' USING PYTHON 'registered'` (`Statement::CreatePersistedFunction`; a Krishiv extension, stored under the registered name) and `DROP FUNCTION [IF EXISTS] f` reach this catalog through `krishiv_session::PersistedFunctions`, which `SqlPersist` implements over `SessionPython` (installed on each session by `new_entry`; same `Usage` on `System("udf")` check and owner rule as the command; a name that is no stored function is `ROUTINE_NOT_FOUND`; test `sql_creates_and_drops_a_persisted_udf`). Not done: a test with real workers; the capability `persistpythonfunction` is advertised whether or not a store is configured (a use without one is refused).
