# krishiv-cli

**Layer:** L7 (may depend only on crates in L7 or lower; `cargo xtask layering`).  
`krishiv sql` (P2-T092): `-e`/`-f`/stdin, `-o table|csv|json`, `--conf`, `--table name=path`, `--timing`; exit 0 ok, 1 query/I/O error, 2 usage. The REPL on a terminal is a `reedline` editor (`src/editor.rs`: Tab menu from `complete.rs`, history, multi-line statements); piped stdin uses the plain loop (P2-T132).

## Purpose
The `krishiv` binary: sql, server, coordinator, worker, mcp, ui, doctor, migrate subcommands.

## Key files
Paths are relative to this crate.
- `src/main.rs` subcommand dispatch and the allocator/memory hook set-up
- `src/sql.rs` the `krishiv sql` subcommand: input (`-e`, `-f`, stdin), output (`-o`), `--conf`, `--table`, `--timing`, `--agent`, `--max-rows`, `--max-seconds`
- `src/server.rs` `krishiv server` (feature `server`; `ui` implies it; design P3-T001): host checks before binding (`check_listener`: non-loopback needs `krishiv.auth.*`, bearer methods there need TLS, client certificates need TLS; root refused without `--allow-root`, core dumps off unless `krishiv.server.allowCoreDumps`), one tokio runtime for the Spark Connect listener (`krishiv-connect-server`, `--connect-port`, default 15002) and the admin listener (`--port` 8080: `/healthz`, `/readyz`, `/metrics` and the UI with `ui`; with `krishiv.auth.*` tokens configured they are guarded by `SessionGuard` over one `ServerAuth`, else by a random token printed once; `krishiv.auth.*` and `krishiv.server.tls.*` settings never reach the engine configuration, nor do catalog declarations (`krishiv.catalog.<name>.*`, `spark.sql.catalog.*`: handed to the runtime as a server-mode `CatalogSetup` with the factories compiled in, `iceberg-rest` with feature `catalog-iceberg-rest`, and checked before either listener binds, a refusal exits 1); with `krishiv.server.tls.certFile`/`keyFile` both listeners speak TLS through one `ServerTls`, reloaded on SIGHUP (handler installed before the addresses print), and the printed addresses are `sc://...;use_ssl=true` / `https://`). With authentication on, the same `ServerAuth` authenticates every Connect RPC (`ConnectAuth`) and each caller's session is `Session::new_session_as(principal, PathPrefixPolicy(RolePolicy(role)))` (`krishiv_connect_server::role_sessions`; client code admin-only until T014b, ADR 0046 N1; the path confinement is checked at startup; a caller with no role is refused before any session), SIGTERM/SIGINT start `Drain { grace = krishiv.server.shutdownTimeout, exit_when_drained }` through `Drainer` (shared with the admin API's `Drain`/`ResumeServing`): `/readyz` 503, new work refused with `KRISHIV_SERVER_DRAINING`, held work finishes until the deadline and is then cancelled (`KRISHIV_QUERY_KILLED`), the listeners stop on DRAINED and the process exits 0 after the engine runtime is dropped; a second signal exits 130; `tests/server.rs`. With feature `flight-sql` (implies `server`, P3-T101) and `krishiv.flightSql.enabled` (default on) the same listener also serves Arrow Flight SQL: a `krishiv_flight_sql::FlightSqlServer` over the Connect manager, a `ConnectAuth` of the same `ServerAuth`, the drain flag and the response budget, added with `with_routes`; its sweep task runs on the Connect maintenance interval and `close_all` runs after `serve`; `tests/flight_sql.rs`
  The console's Resources page lists the server's Python UDF workers: `run_listeners` hands `admin_router` a provider over `Pool::monitor()` (`udf_workers_of`; Linux only, where the pool exists), which `SessionSource::with_udf_workers` serves at `/api/internal/udf-workers`; the monitor does not keep the pool, so a stopping server's workers stop as before (`tests/server.rs` `the_console_lists_the_servers_udf_worker_pool`)
- `src/ui.rs` (feature `ui`, P3-T050) `krishiv ui --profiles <dir>`: the web UI over a history store (`krishiv.history.path`) with no engine; loopback, start-up token printed once. `krishiv server` serves the UI for loopback names and, on a loopback bind, its bound address (a non-loopback bind with `krishiv.ui.allowedHosts` set answers only to those names: add the bound IP to use it); `krishiv.ui.allowedHosts` (and `KRISHIV_UI_ALLOWED_HOSTS`) name the hosts the UI answers to (a non-loopback bind checks only when names are set); the UI session cookie is `Secure` over TLS; the editor's sessions come from `role_sessions`
- `src/check_compat/` (P3-T029) `krishiv check-compat <paths>`: `mod.rs` args, file walk, exit codes (0 none, 1 findings, 2 read errors), JSON/text report; `scan.rs` SQL / Python / notebook-cell scanners; `alternatives.json` workarounds for not-yet functions (names only supported functions; a unit test checks), `functions.json` the function matrix (generated from the SQL reference pages by `tests/scripts/gen_compat_matrix.py`); corpus `tests/check_compat_corpus/` (`handwritten/` with a hand-checked `expected.json`, `supported_only/` must give no findings, `generated/` from `tests/scripts/gen_check_compat_corpus.py`); JSON report is `{schema_version, files_scanned, findings, errors}`
- `src/repl.rs` the interactive loop (no `-e`/`-f` on a terminal): `.timer`, `.format`, `.complete`, `.history`, history persistence
- `src/complete.rs` completion sources (keywords, function registry, session catalog, config registry)
- `src/highlight.rs` ANSI highlighting of statements
- `src/history.rs` `~/.krishiv/history` (10,000 entries; sensitive `SET` lines not persisted)
- `src/format.rs` one renderer per output mode (agent output lives in `krishiv-session::agent`)
- `tests/sql.rs` end-to-end tests of `krishiv sql`; `tests/repl.rs` modes, agent detection, caps; `tests/version.rs` version output
- `Cargo.toml` features (`mimalloc` links `krishiv-alloc` and `krishiv-memory`)

## Extension points
- New subcommand: a variant and dispatch in `src/main.rs`, the logic in its own module like `src/sql.rs`, tests in `tests/`.
- New output format for `krishiv sql`: extend the `-o` handling in `src/sql.rs`.

## Invariants
- One executable with subcommands (ADR 0026).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0004, 0026

## Tests
`cargo nextest run -p krishiv-cli --no-tests=pass` · `cargo clippy -p krishiv-cli --all-targets -- -D warnings`
