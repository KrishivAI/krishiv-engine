# CLAUDE.md — krishiv-engine

**Read first:** `../krishiv-internal/CLAUDE.md`, then `../krishiv-internal/STATUS.md`, the active plan in `../krishiv-internal/plans/`, and the newest note in `../krishiv-internal/handoffs/`. Start sessions here with `--add-dir ../krishiv-internal`.

This file is a placeholder; P0-T16 expands it with commands and per-crate notes.

## Hard rules (summary)
- **No DataFusion, RocksDB or OpenSSL** (ADR 0002, 0019, 0028). Keep Arrow on the workspace-pinned version.
- **Layering L0–L7** (ADR 0026): a crate depends only on lower layers, enforced by `cargo xtask layering`.
  - L0 common
  - L1 types, memory
  - L2 kernels, functions, io
  - L3 sql, plan, optimizer
  - L4 exec, state, stream
  - L5 shuffle, scheduler, connectors
  - L6 connect, flight-sql, mcp, server, python, ui-assets
  - L7 krishiv (facade), krishiv-cli
  - SDKs: udf-sdk, connector-sdk
- **Cargo features are additive only.** Behaviour flags are runtime config (`krishiv.*`, `krishiv.experimental.*`) (ADR 0027).
- **`unsafe` only** in krishiv-memory, krishiv-kernels and FFI crates, each block with a `// SAFETY:` comment. No `unwrap`/`expect`/`panic` in library code. Errors carry a Spark error class and SQLSTATE (ADR 0029).
- **Spark-observable behaviour must match Spark exactly** (ADR 0008, 0010). Any divergence is explicit and opt-in.
- **Process:** one task = one worktree, one branch, one PR, with TDD and two-stage review (ADR 0031).
- **Hard stops (ask the user):**
  - publishing to crates.io or PyPI
  - pushing or merging to `main` after bootstrap
  - deleting data or containers
  - touching other projects on this machine
- **Commit identity:** the user's only (`gopal tathe`, noreply). End commit messages with the attribution trailer given in the session.
