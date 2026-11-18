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
  only after tonic took its last piece); cut by rows past 4 MiB, pieces of 2 MiB; errors end it with `flight_status`
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
- `src/sql_info.rs` `sql_info_data(&ServerInfo)` (design §5): the `GetSqlInfo` table (arrow-flight `SqlInfoDataBuilder`):
  name `Krishiv`, not read-only, SQL yes, Substrait no, transactions none (ADBC reads it at connect), cancel yes,
  backtick quotes, case-insensitive identifiers; versions, keywords and the four function lists come from the caller
- `src/status.rs` `flight_status` / `code_of` (design §6): a `KrishivError` as the Flight SQL gRPC status, from its
  client-safe form (`KrishivError::for_client`); Spark's message text, trailing metadata `x-krishiv-error-class`,
  `x-krishiv-sqlstate`, `x-krishiv-error-id`. Connect's `status_of` is not used here.

## Invariants
- Versioned public surface (ADR 0026, ADR 0005).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0005, 0026, 0046

## Test
`cargo nextest run -p krishiv-flight-sql --no-tests=pass` · `cargo clippy -p krishiv-flight-sql --all-targets -- -D warnings`
