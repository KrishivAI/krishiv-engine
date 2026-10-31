# krishiv-cli

**Layer:** L7 (may depend only on crates in L7 or lower; `cargo xtask layering`).  
`krishiv sql` (P2-T092): `-e`/`-f`/stdin, `-o table|csv|json`, `--conf`, `--table name=path`, `--timing`; exit 0 ok, 1 query/I/O error, 2 usage. REPL polish is P2-T132.

## Purpose
The `krishiv` binary: sql, server, coordinator, worker, mcp, ui, doctor, migrate subcommands.

## Key files
Paths are relative to this crate.
- `src/main.rs` subcommand dispatch and the allocator/memory hook set-up
- `src/sql.rs` the `krishiv sql` subcommand: input (`-e`, `-f`, stdin), output (`-o`), `--conf`, `--table`, `--timing`, `--agent`, `--max-rows`, `--max-seconds`
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
