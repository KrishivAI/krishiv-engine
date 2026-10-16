# CLAUDE.md — krishiv-engine

Krishiv: a Spark-compatible batch + streaming engine in Rust (no DataFusion, no JVM). This repo holds code, tests,
benchmarks and the Claude Code setup. **Decisions, plans and status live in `../krishiv-internal`** (ADRs are there,
not here). Start sessions here with `claude --add-dir ../krishiv-internal`.

## Read first (every session — skill `krishiv-handoff`)
1. This file, then `crates/<crate>/CLAUDE.md` for each crate you touch.
2. `../krishiv-internal/CLAUDE.md` once, then `../krishiv-internal/STATUS.md` → the active plan it links
   (`plans/P<n>-*.md`) → newest `handoffs/*.md`. The SessionStart hook prints STATUS.md and the latest handoff head.
3. Only the ADRs your task lists: `../krishiv-internal/adr/NNNN-*.md`. Spec chapters in `../krishiv-internal/spec/`
   only when an ADR or the task points there. Read by path; never paste docs into prompts.
4. `git status`, `git log --oneline -10`, `git worktree list`, `gh pr list` before changing anything.

From a worktree, the docs repo is `$(git rev-parse --path-format=absolute --git-common-dir)/../../krishiv-internal`
(or set `KRISHIV_INTERNAL`).

## Commands
```
cargo build --workspace                          # toolchain pinned: rust-toolchain.toml (1.99.1)
cargo nextest run --workspace                    # tests (CI: --all-features --locked)
cargo test --workspace --doc                     # doctests
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo deny check                                 # licenses, bans, advisories, duplicate versions
cargo machete                                    # unused deps
cargo hack check --each-feature --workspace      # features stay additive
cargo xtask layering                             # L0–L7 layer rule (xtask/layers.toml)
cargo xtask docs-config | docs-errors            # regenerate config / error-class docs
cargo xtask budgets [--build-time]               # crate-count / build-time / size budgets (budgets.json)
```
Pre-commit hook (fmt, clippy, deny; skips clippy/deny for docs-only commits or with `SKIP_CLIPPY=1`). Git config is
not versioned, so enable it once per fresh clone (repo-local, never `--global`): `git config core.hooksPath .githooks`.

`.claude/settings.json` permission rules use mid-pattern `*` globs (e.g. `Bash(git push * --force*)`): a **recent
Claude Code is required** (older versions only honour a trailing `:*`, which would leave those denies inert).
After upgrading Claude Code or editing the rules, verify with a fresh session. The denies are a backstop; the hard
stops below apply regardless.

## Layout
- `crates/` — the workspace (layers below). `xtask/` — repo tooling. `.github/workflows/` — ci, nightly-bench, compat.
- `deny.toml`, `budgets.json`, `renovate.json` — dependency and size gates.
- Arriving in P0: `bench/` (baselines P0-T17, datasets P0-T18, harness P0-T19), `tests/spark-golden/` (P0-T20/T22),
  `python/` (P0-T25), `ui/` (P0-T26), `sim/` (P0-T24).
- `.claude/agents/` (implementer, spark-semantics, reviewer, perf-analyst), `.claude/skills/krishiv-*`,
  `.claude/settings.json` + `.claude/hooks/` (SessionStart/Stop), `.githooks/pre-commit`.

## Layering (ADR 0026; enforced by `cargo xtask layering`)
A crate depends only on crates in the same or a lower layer.
- L0 common
- L1 types, memory; SDKs udf-sdk, connector-sdk (L1: depend on L0–L1 only)
- L2 kernels, functions, io
- L3 sql, plan, optimizer
- L4 exec, state, stream
- L5 shuffle, scheduler, connectors-{kafka,iceberg,delta,cdc}
- L6 connect, flight-sql, mcp, server, python, ui-assets
- L7 krishiv (facade), krishiv-cli (binary `krishiv`)

New crate → add it to `xtask/layers.toml` and root `Cargo.toml`, and write its `CLAUDE.md`.

## Hard rules
- **No DataFusion, RocksDB or OpenSSL** (ADR 0002, 0019, 0028); full ban list in `deny.toml`. Arrow sub-crates at the
  workspace-pinned major; all deps in `[workspace.dependencies]`, `default-features = false` (skill `krishiv-dependency`).
- **Spark-observable behaviour matches Spark exactly**, keyed by Spark version 3.5/4.0/4.1/4.2 (ADR 0008, 0010).
  Ground truth comes from apache/spark source at tags + a real Spark run (spark-semantics agent), never memory.
  Divergence is explicit and opt-in per expression; no global allowIncompatible switch. Observable hashing is
  Murmur3 seed 42.
- **Cargo features are additive only**; behaviour flags are runtime config `krishiv.*` / `krishiv.experimental.*`
  (ADR 0027).
- **`unsafe` only** in krishiv-memory, krishiv-kernels and FFI crates (krishiv-python, SDK bridges), each block with
  `// SAFETY:`; everything else `#![forbid(unsafe_code)]` (ADR 0029).
- No `unwrap`/`expect`/`panic` in library code; `thiserror` errors with a Spark error class + SQLSTATE from
  `crates/krishiv-common/error-classes.json`; no `anyhow` in libraries.
- On-disk formats carry a version; N reads N−1 (ADR 0026). Public surfaces change only via ADR.
- Identifiers are `krishiv` / `krishiv-*` (ADR 0030). Decisions change only through an ADR (skill `krishiv-adr`).
- Never commit secrets. Commit identity: the user's only (`gopal tathe`, noreply); end commit messages with the
  attribution trailer given in the session.

## Hard stops — always ask the user first
- Publishing to crates.io, PyPI, npm or any registry (skill `krishiv-release` is dry-run only).
- Pushing or merging to `main`; force-pushing anything.
- Deleting data, volumes or containers the session did not create (never touch `bench-minio`).
- Touching other projects on this machine, or global `~/.claude` settings.
- Starting a new phase without the user's gate approval.

## Process (ADR 0031, `../krishiv-internal/plans/README.md`)
- One task = one git worktree (`.worktrees/<task>`) = one branch (`p<n>/<task>-<slug>`) = one small PR.
  Parallelism only along crate boundaries; never two agents in one crate; at most 2–3 implementers at once.
- Per task: spark-semantics goldens first (if Spark-observable) → implementer with TDD → reviewer (stage 1
  spec/ADR compliance, stage 2 quality/unsafe/concurrency) → fix → merge via PR (user merges `main`).
- Skills: `krishiv-handoff`, `krishiv-spark-function`, `krishiv-operator`, `krishiv-adr`, `krishiv-dependency`,
  `krishiv-benchmark`, `krishiv-experiment`, `krishiv-release`; plus the superpowers workflow skills.
- Session end: tick the plan, update STATUS.md, push branch / draft PR, write a handoff. The Stop hook reminds you.

## Token efficiency
- **Briefs and reports are files.** Dispatch with a brief file path; subagents write full reports to a file and
  reply in < 15 lines (status, commits, verification, concerns, path). Never paste ADRs/spec into prompts — cite paths.
- **Model per role** (ADR 0031): Opus for planning, review, ADRs and hard subsystems (scheduler, arbitrator, hash
  tables, unsafe kernels, IVM, krishiv-state, checkpoints, exactly-once, distributed failure) or any task with open
  design questions; Sonnet for specified work ("make these tests pass"); Haiku for mechanical chores.
- **Batch same-shape work** (e.g. a family of functions) into one brief and one agent, not many tiny dispatches.
- **Review scaled to risk**: light for docs/mechanical/golden-backed bulk work, deep for unsafe, concurrency,
  formats and public API.
- **CI gates are the police**: fmt, clippy, deny, machete, hack, layering, budgets, tests. Run them instead of
  re-reading code to check rules; don't hand-verify what a gate already enforces.
- Read only what the task needs: the task entry, its ADRs, the crate CLAUDE.md files, then the code.
