# krishiv-mcp

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (stdio), P5 (HTTP).

## Purpose
MCP adapter (stdio read-only first, HTTP + auth later).

## Invariants
- Read-only by default; no write tools without auth (ADR 0025).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0025, 0005

## Test
`cargo nextest run -p krishiv-mcp --no-tests=pass` · `cargo clippy -p krishiv-mcp --all-targets -- -D warnings`
