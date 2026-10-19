# krishiv-cli

**Layer:** L7 (may depend only on crates in L7 or lower; `cargo xtask layering`).  
`krishiv sql` (P2-T092): `-e`/`-f`/stdin, `-o table|csv|json`, `--conf`, `--table name=path`, `--timing`; exit 0 ok, 1 query/I/O error, 2 usage. REPL polish is P2-T132.

## Purpose
The `krishiv` binary: sql, server, coordinator, worker, mcp, ui, doctor, migrate subcommands.

## Invariants
- One executable with subcommands (ADR 0026).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0004, 0026

## Test
`cargo nextest run -p krishiv-cli --no-tests=pass` · `cargo clippy -p krishiv-cli --all-targets -- -D warnings`
