# krishiv-session

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).
**Real code since:** P2-T090 (session, configuration, catalog, DataFrame actions).

## Purpose
The session every API surface shares: configuration, the session catalog (temporary tables and views),
and the query pipeline (parse → analyze → optimize → physical plan → run). The Rust facade (`krishiv`, L7)
re-exports it; the Python facade and the Connect and Flight SQL servers (L6) wrap the same type.

## Invariants
- One pipeline: no API surface plans or runs queries any other way.
- Each query reads a snapshot of the configuration; a concurrent `conf().set` never changes a running query.
- Transformations only build IR; actions are `async` and run on the session's executor, so they work from
  any runtime. Dropping an action's future or result stream cancels its query.
- The IR is reachable only behind the `unstable-ir` feature (not covered by semver).
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class
  and SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0005, 0026, 0027, 0037, 0038

## Test
`cargo nextest run -p krishiv-session --all-features` · `cargo clippy -p krishiv-session --all-targets --all-features -- -D warnings`
