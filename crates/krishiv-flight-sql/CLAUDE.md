# krishiv-flight-sql

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).
**Real code since:** P3-T101 (design `../krishiv-internal/design/P3-T101-flight-sql.md`, v5, design-3; IP PR-65).

## Purpose
Arrow Flight SQL, single endpoint (P3): a second gRPC service on the Spark Connect listener, on Connect's session
manager, auth layer, operations and drain flag (it depends on `krishiv-connect-server`; Connect never depends on it).
Multi-endpoint tickets and stored results are P5 (P5-T030, P5-T047).

## Key files
Paths are relative to this crate.
- `src/cookies.rs` `session_cookie` / `set_session_cookie` (design §3.2): the `arrow_flight_session_id` cookie over
  every `cookie` header; `Absent` (cookie-less client), `Present(id)` for the manager, `Malformed` (gets a new session
  and cookie, as a refused id); ids of 1-64 letters, digits, `-` only
- `src/encode.rs` `encode_results` (design §4.2): the `DoGet` pull stream, arrow-flight `FlightDataEncoder` over the
  query's batches; each batch reserved as consumer `flight_sql_responses` until the encoder asks for the next (so
  only after tonic took its last piece); cut by rows past 4 MiB, each row measured by what it covers (`row_sizes`:
  offsets, list/map children, struct fields, dictionaries as their hydrated values, as the encoder sends them), a row over it alone `KRISHIV_FLIGHT_SQL_ROW_TOO_LARGE`; pieces of 2 MiB; errors end it with `flight_status`
- `src/handles.rs` `Handles<K, P>` (design §4.1): 128-bit random `HandleId`s bound to an owner and a session, no rows held;
  another owner's id answers as unknown; `HandleLimits` (TTL from the end of the last operation, never while
  `begin`..`end` holds one live; count per session and per owner, bytes per owner:
  at a cap expired handles go, then the scope's oldest idle statement handles (prepared never; `reserve` keeps a
  slot in flight until `end`, `set_payload` after the run); refused with
  `KRISHIV_FLIGHT_SQL_HANDLE_LIMIT` only when every handle of the scope is fetching); `get` (no operation), `begin`/`end` (a fetch),
  `remove_session`, `sweep(now, alive)` (`alive` must not stamp session access); `insert`/`reserve` return
  `Inserted { id, evicted }` (the sessions of the handles removed to make room, v4 review m4)
- `src/sessions.rs` `Sessions<O>` (design §3.2, v4 review m1/m2): Flight's session table, minted id (`mint_session_id`,
  a v4 UUID) -> owner, `Provisional{changed}`/`Cookie`, calls in flight, end of the last call. One mutex; it decides
  releases and returns `Release`s, the caller runs `release_owned` and drops the handles after the lock; liveness
  (`is_live`) is asked outside it (`snapshot` then `sweep(now, gone)`). `insert` (after the manager created it),
  `enter` (promotes when via the cookie), `promote`, `mark_changed`, `exit(Exit::CookielessFetch)` (rule 1),
  `release_if_handleless` (rule 3), `close`, `sweep` (rule 2: unchanged provisional past `provisionalTtl`); a rule met
  with calls in flight waits for the last `exit`. Every row op checks the caller's owner; `holds` closures read the
  handle map (lock order: sessions, then handles)
- `src/metrics.rs` `Requests` (design §7): `krishiv_flight_sql_requests_total{command, outcome}`, fixed `Command` and
  `Outcome` enums equal to the registry's label values (test); lock-free cells, `counts()` for the server's scrape
- `src/server.rs` `FlightSqlServer` (design §2): built from the Connect listener's `SessionManager`, `ConnectAuth`,
  drain flag and `ResponseBudget`; `FlightSqlConfig::from_config` (the `krishiv.flightSql.*` keys); `service()` is the
  tonic service for `ConnectServer::with_routes`; `spawn_sweeper` (abort-on-drop `SweepGuard`), `sweep_at`,
  `close_all` (after `serve` returns); a handle's `Payload` (full owner, SQL, pinned clock, `Kind`: pending, query
  schema, or a command's stored rows held as `flight_sql_handles` past 64 KiB)
- `src/call.rs` the session of one call (design §3.2): the cookie's, else created once (provisional without a cookie,
  a cookie session for a refused one, "session reset" at info); `CallSession` is the drop guard that ends the call in
  the table and releases a session the call created unless its answer went out (`kept`): an error or a dropped
  future, whatever the row's state (review M4); `traced` gives the queries a request starts its `traceparent` / `baggage` headers (P3-T068: per request, as Connect's `ExecutePlan`; applied where a query or update runs: `DoGet` of a statement, a command run by `GetFlightInfo`, `run_update`; `tests/service.rs` `a_requests_traceparent_names_the_trace_of_its_query_and_update`)
- `src/statement.rs` `GetFlightInfo(CommandStatementQuery)` (slot, `Operation`, classify with `sql_query_with`, a
  command runs once and is stored), `DoGet(TicketStatementQuery)` (re-analysis with the pinned clock,
  `SCHEMA_CHANGED`, the stored rows of a command), `described` (the announced schemas carry `ARROW:FLIGHT:SQL:PRECISION`/`SCALE`
  on decimal columns, note §17 A3; the batches none), `CancelFlightInfo` (only the operations live on the handle, `Payload::live`); marks are drop guards released in reverse
  (operation, fetch, response budget, call; §16 M3)
- `src/update.rs` `DoPut(CommandStatementUpdate)` (design §4.3, §16 M2): classified with `sql_query_with` (a query
  is `KRISHIV_FLIGHT_SQL_QUERY_AS_UPDATE`, nothing runs), run once under an `Operation`; `record_count` from a
  `num_affected_rows` column, else 0 (note §17 A2: Arrow's JDBC driver reads -1 as a result set and runs it again); a cookie-less update that marks nothing changed runs in a transient session
  (no `set-cookie`, released at the end of the call)
- `src/prepared.rs` prepared statements (design §4.4): `CreatePreparedStatement` classifies with `sql_query_with`
  (a parameterized statement with each `?` bound to `NULL`; nothing runs), stores a `Kind::Prepared` handle (never
  evicted) and marks the session changed; `dataset_schema` of every query, `parameter_schema` one nullable field per
  marker typed from its context (note §17 A1: `krishiv_session::parameter_types::marker_types` over the `NULL`-bound
  analysis, `Session::insert_parameter_types` for an `INSERT ... VALUES` cell, else a string); `DoPut(CommandPreparedStatementQuery)` binds one row (`Session::bind_parameters`, literals as
  `EXECUTE IMMEDIATE ... USING`), `GetFlightInfo` of it runs as §4.1 in the handle's session,
  `DoPut(CommandPreparedStatementUpdate)` runs once per row (all bound first; counts summed, a run that reports
  none counts 0; over `maxBatchRows` `KRISHIV_FLIGHT_SQL_INVALID_PARAMETERS`, 07001); `ClosePreparedStatement` of a gone or
  foreign handle is OK; every call but create is handle-carrying (`handle_call`)
- `src/metadata.rs` catalog metadata (design §5): `GetCatalogs`, `GetDbSchemas`, `GetTables`, `GetTableTypes` and the
  key commands; `GetFlightInfo` answers the fixed schema and a ticket that is the command (no session), `DoGet` runs it
  in the call's session (cookie-less: transient, no `set-cookie`) under `metadataTimeout`
  (`KRISHIV_FLIGHT_SQL_METADATA_TIMEOUT`, `DEADLINE_EXCEEDED`); reads `Session::catalog()` (`list_embedded_databases`,
  `list_tables`: the `SHOW` checks); schemas and tables of `spark_catalog` only (declared catalogs list none yet),
  temporary views once with the empty catalog (`GetDbSchemas` lists `("", "global_temp")`, and `("", "")` while local
  temp views exist); a namespace a pattern matched whose `SHOW TABLES IN` is refused is absent (an exact name is
  refused); Flight SQL patterns (`%`, `_`, `\`) ignoring case; `include_schema` reads each schema through
  `schema_for_metadata` (`Browse`, as `DESCRIBE`), over `maxSchemaTables` is `KRISHIV_FLIGHT_SQL_SCHEMA_TABLES_LIMIT`
  (54000), a table whose schema fails is skipped and logged; a refused `global_temp` hides the temporary views and
  their schemas unless named exactly (`global\_temp`); the statement-gate wait is inside the timeout; key
  commands empty with FlightSql.proto's schemas after a `tableExists` check (a catalog alone means its current
  database)
- `src/gate.rs` `FlightGate` (design §16 m5): the tonic `FlightService` in front of the arrow-rs dispatch; every call
  is a `FlightRpc` (method + command type URL or action type) and `check_flight` (exhaustive, no wildcard) answers what
  the server does not serve `UNIMPLEMENTED` (an undecodable command `INVALID_ARGUMENT`) before any body decoding, auth or
  session work, counted `unsupported`; `DoPut` and the prepared-statement actions are dispatched here (first message peeked; the create's answer carries the cookie), not by arrow-rs; the session-option actions too; `ListActions` is served (`served_actions`: `CancelFlightInfo`, `CancelQuery`, `CreatePreparedStatement`, `ClosePreparedStatement`, `SetSessionOptions`, `GetSessionOptions`, `CloseSession`, the actions it lets through; counted `list_actions`, no session)
- `src/service.rs` the arrow-rs `FlightSqlService` impl: caller first, and new work (handshake, statements, updates, prepared-statement create/execute/bind, `GetSqlInfo`, `GetXdbcTypeInfo` and the eight listings (info and `DoGet`), `SetSessionOptions`/`GetSessionOptions`; `admit_new`) refused with `KRISHIV_SERVER_DRAINING` (`FAILED_PRECONDITION`) while the drain flag is set (fetches of held statement tickets, cancels, prepared close and `CloseSession` still served, T066 §5.1); Handshake echoes the bearer
  (a `Basic` login's password as `Bearer`, note §3.1; `Basic` is decoded by Connect's `AuthLayer` on this surface only);
  `GetSqlInfo` without a session; `CancelFlightInfo` and the deprecated `CancelQuery` (a malformed ticket is
  `INVALID_HANDLE.FORMAT`)
- `tests/service.rs` end to end over a Connect listener with token auth (round trip, cookie and cookie-less sessions,
  commands once, re-fetch clock, owner binding, cancel, limits, auth)
- `src/session_options.rs` Flight's session options (design §3.2; Flight.proto messages declared here with prost,
  arrow-flight 60 has none): `SetSessionOptions` (`catalog`/`schema` set the current catalog/database, other keys
  `check_static` -> `authorize_set_key` -> `conf.set`, valueless = unset; marks the session changed, sets the cookie,
  under the statement gate), `GetSessionOptions` (`conf().entries()` minus `is_redacted`, plus `catalog`/`schema`;
  cookie-less: transient session), `CloseSession` (close then release; always `CLOSED`)
- `src/xdbc_info.rs` `xdbc_type_info_data()` (design §5): the `GetXdbcTypeInfo` table, built once in `Inner`, one row
  per Spark type a column can have (Spark SQL names; all 13 interval field ranges, each with its subcode; complex,
  variant, geo and `VOID` types have no XDBC code and are not listed); served without a session, counted `xdbc_type_info`
- `src/sql_info.rs` `sql_info_data(&ServerInfo)` (design §5): the `GetSqlInfo` table (arrow-flight `SqlInfoDataBuilder`):
  name `Krishiv`, not read-only, SQL yes, Substrait no, transactions none (ADBC reads it at connect), cancel yes,
  backtick quotes, case-insensitive identifiers; versions, keywords and the four function lists come from the caller
- `src/status.rs` `flight_status` / `code_of` (design §6): a `KrishivError` as the Flight SQL gRPC status, from its
  client-safe form (`KrishivError::for_client`); Spark's message text capped at 2 KiB (`... [truncated; error id <id>]`), the full error logged under the id, trailing metadata `x-krishiv-error-class`,
  `x-krishiv-sqlstate`, `x-krishiv-error-id`. Connect's `status_of` is not used here.

## Invariants
- Versioned public surface (ADR 0026, ADR 0005).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0005, 0026, 0046

## Test
`cargo nextest run -p krishiv-flight-sql --no-tests=pass` · `cargo clippy -p krishiv-flight-sql --all-targets -- -D warnings`
