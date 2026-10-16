# Krishiv

**Krishiv** is a Spark-compatible batch & streaming engine written in Rust.
- Runs embedded, single-node or distributed.
- APIs: Python (PySpark-compatible), Rust, SQL, Spark Connect, Flight SQL and MCP.
- Incremental materialized views and queryable streaming state.

> Private repository. Status: **P0 foundation in progress** on branch `p0/foundation` (PR #1); no query engine yet.
> What exists: a 30-crate layered workspace skeleton (layering, dependency, size and lint gates in CI), the
> Spark error-class and config registries (`krishiv-common`), a testkit with Spark golden/hash fixtures and a
> batch==stream harness, a simulation seam, a benchmark harness with DuckDB and Spark baselines, a Python package
> and Svelte UI skeleton, and the Claude Code setup under `.claude/`. Nothing is published anywhere.
> Design, decisions, plans and status live in the private repo `KrishivAI/krishiv-internal` (cloned alongside as `../krishiv-internal`).

License: to be decided (ADR 0033, proposed Apache-2.0).
