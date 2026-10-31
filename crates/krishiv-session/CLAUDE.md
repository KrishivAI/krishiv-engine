# krishiv-session

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).
**Real code since:** P2-T090 (session, configuration, catalog, DataFrame actions).

## Purpose
The session every API surface shares: configuration, the session catalog (temporary tables and views),
and the query pipeline (parse → analyze → optimize → physical plan → run). The Rust facade (`krishiv`, L7)
re-exports it; the Python facade and the Connect and Flight SQL servers (L6) wrap the same type.

## Key files
Paths are relative to this crate.
- `src/lib.rs` public entry and re-exports; `public-api.txt` the reviewed public surface
- `src/session.rs` `Session` and the query pipeline entry
- `src/conf.rs` configuration snapshot taken per query
- `src/catalog.rs` session catalog (temporary tables and views)
- `src/view_schema.rs` schema binding modes of permanent views (`WITH SCHEMA BINDING|COMPENSATION|TYPE EVOLUTION|EVOLUTION`, Spark's `ViewSchemaMode`): the read projection per mode, the checks against the analyzed view query, the evolved schema stored after analysis; `tests/views_schema_binding.rs`
- `src/dataframe.rs` DataFrame transformations (IR only) and async actions; `src/column.rs` and `src/functions.rs` user API
- `src/access.rs` `StatementInfo` / `Session::statement_info` (class + read securables with columns), write/DDL checks (`authorize_statement`), the `krishiv::audit` denial record, `PathPrefixPolicy`
- `src/explain.rs` EXPLAIN and `DataFrame::estimate` (ADR 0041: estimated table-scan bytes and rows from the optimizer's statistics, `None` when unknown; the CLI's agent cost gate uses it); `src/profile.rs` profile for EXPLAIN ANALYZE
- `src/agent.rs` the `krishiv.agent-result` v1 formatter (compact tables, errors, estimates) plus the shared Spark-style value renderer
- `tests/session.rs`, `tests/explain_analyze.rs`, `tests/spill.rs`, `tests/tpch_q1.rs` and `tests/window_goldens.rs` (window functions against Spark goldens) end-to-end tests

## Extension points
- New DataFrame action: an async method in `src/dataframe.rs` running on the session executor; transformations only build IR.
- New config key: add it to the typed registry in `krishiv-common` (see that crate's map).

## Invariants
- One pipeline: no API surface plans or runs queries any other way.
- Each query reads a snapshot of the configuration; a concurrent `conf().set` never changes a running query.
- Transformations only build IR; actions are `async` and run on the session's executor, so they work from
  any runtime. Dropping an action's future or result stream cancels its query.
- SQL-bodied temporary functions live in a per-session map (`sql_functions.rs`) replaced on every change (never edited); a
  query takes the map when it starts and reads one snapshot, and every change bumps the registry version.
- The IR is reachable only behind the `unstable-ir` feature (not covered by semver).
- A read of a permanent view analyzes the view's query once on its own (`Session::check_stored_views`) before the whole query, so its binding mode can check types and names (`INCOMPATIBLE_VIEW_SCHEMA_CHANGE`, `CANNOT_UP_CAST_DATATYPE`); `TYPE EVOLUTION`/`EVOLUTION` views store their new schema only after the whole query analyzed, with a compare-and-swap (a concurrent change wins).
- Access control: one principal and one `AccessPolicy` per session, for life. Reads are checked in analysis; what a statement writes, creates or drops (`krishiv_sql::statement_targets`) is checked in `sql` before it is dispatched, and `df.write` paths before `DirectoryWrite::begin`, all through `krishiv_plan::analyzer::authorize` (fail closed, 42501, no reason, no existence oracle). Permanent views are inlined before binding, so `authorize_reads` asks the policy about every catalog-store lookup in plan order before any read (name as written and folded, then the view securable); a failure inside a stored view's text is raised only after that. Every denial logs one `krishiv::audit` record. Deny-all and inverse matrices: `tests/access_policy.rs` (merge-blocking).
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class
  and SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0005, 0026, 0027, 0037, 0038

## Tests
`cargo nextest run -p krishiv-session --all-features` · `cargo clippy -p krishiv-session --all-targets --all-features -- -D warnings`
