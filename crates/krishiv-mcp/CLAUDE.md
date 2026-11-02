# krishiv-mcp

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P2 (stdio), P5 (HTTP).

## Purpose
MCP adapter (stdio read-only first, HTTP + auth later).

## Key files
Paths are relative to this crate.
- `src/server.rs` the rmcp server pinned to the 2026-07-28 revision (`KrishivMcpServer::with_tools(McpConfig)` adds the query tools)
- `src/tools.rs` the seven read-only tools, caps (rows, bytes, seconds), the scan-estimate gate (`allow_heavy`), one running query and a queue of 4; every call builds a fresh session over `--data`
- `src/gate.rs` the read gate: one statement of class `Read` (`KRISHIV_MCP_STATEMENT_NOT_ALLOWED`), `SetupLock` (tables are registered, then only `Select` is allowed through `ReadOnlyPolicy`), root check
- `tests/tools.rs` results, caps, the negative suite, statelessness; `tests/stdio_conformance.rs` protocol handshake
- Not done (design P2-T095): `explain_analyze`, `get_query_profile`, `query_history`, `count_total`, `krishiv.mcp.serve()` in Python, OTel audit records; metadata tools answer in the agent text, not `structuredContent`.

## Invariants
- Read-only by default; no write tools without auth (ADR 0025).
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0025, 0005

## Test
`cargo nextest run -p krishiv-mcp --no-tests=pass` · `cargo clippy -p krishiv-mcp --all-targets -- -D warnings`
