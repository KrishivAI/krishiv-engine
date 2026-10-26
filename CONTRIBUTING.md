# Contributing to Krishiv

Thank you for your interest. Krishiv is a Rust-native, Spark-compatible batch, streaming and incremental
materialized-view engine. It is **pre-alpha** and under active construction, so expect churn. Please read the
[Code of Conduct](CODE_OF_CONDUCT.md) first; report security issues as described in [SECURITY.md](SECURITY.md).

## Quick start (target: green `cargo nextest run` in under 15 minutes)

```text
git clone https://github.com/KrishivAI/krishiv-engine && cd krishiv-engine
cargo xtask doctor           # checks toolchain, uv/Python, Docker, disk; names a fix for anything missing
cargo xtask setup --install  # installs git hooks, then runs the user-level fixes (rustup, cargo install, uv)
cargo nextest run --workspace --locked --no-tests=pass
```

`cargo xtask setup` without `--install` only prints the fixes. xtask never runs `sudo`; commands that need it
(for example `sysctl kernel.perf_event_paranoid=1` for benchmark counters) are printed for you to run. Running
several worktrees: give each its own `CARGO_TARGET_DIR` on a large disk (doctor reminds you).
A [dev container](.devcontainer/devcontainer.json) provides the same setup for VS Code and Codespaces.
Measured on a 12-core Linux workstation (2026-10-17, tools already installed), a fresh clone with a cold target
directory reached a green `cargo nextest run` in 23 s; compiling the cargo tools from source with `setup --install`
adds a few minutes.

The baseline stack (Spark Connect 4.2 + MinIO, optionally Kafka) is `cargo xtask stack up|down|status [--kafka]`.
It addresses only the `krishiv-bench` compose project and `down` keeps its volumes.

## Build and test

The toolchain is pinned in `rust-toolchain.toml`.

```text
cargo build --workspace
cargo nextest run --workspace --locked --no-tests=pass
cargo test --workspace --doc --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
taplo fmt --check; typos                                                # TOML format and spelling
```

Markdown is checked with markdownlint-cli2 (config in `.markdownlint-cli2.yaml`). `CLAUDE.md` lists the full set of
checks CI runs, including `cargo deny`, `cargo xtask layering` and the size budgets.

## License policy

Rust crates may use only the licenses in `deny.toml` (`[licenses] allow`, enforced by `cargo deny check`). The
dependency-review check (`security.yml`) applies the same list plus `CC0-1.0`, which it needs for Python dependencies:
numpy's compound license (BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0) is permissive, but CC0-1.0 carries no
patent grant, so it is kept out of the Rust allow list to keep such code out of the engine binary. A policy test in
`xtask/tests/policy.rs` fails if the two lists differ in anything but `CC0-1.0`. GitHub Actions used only in CI
(for example the LGPL-3.0 `Swatinem/rust-cache`) are excepted by name in `security.yml`: they run on the build runner and
are never distributed or linked, and the same test requires each such exception to be a real workflow dependency.

## Coverage

Pull requests must cover at least 90% of the executable lines they add (`diff-coverage` is a required
check); performance regressions fail the required CodSpeed check. Locally:

```text
cargo xtask coverage --all-features --html   # cargo-llvm-cov line+branch (branch needs `rustup toolchain install nightly -c llvm-tools-preview`)
cargo xtask coverage-summary                 # per-crate line/branch %
cargo xtask diff-coverage --base origin/main --fail-under 90 # the merge gate, on your branch
```

Pull requests get a diff-coverage report in the job summary; the nightly job publishes the full HTML and JSON report
as an artifact. Both upload to [Codecov](https://about.codecov.io/), which is informational only and never blocks a merge.

## Fuzzing and mutation testing

Nightly CI fuzzes every target of `fuzz/` for an hour and the kernels differential fuzzer for ten minutes; a weekly
run gives each 4.5 hours (`fuzz.yml`, `fuzz-weekly.yml`). The corpus lives only in the CI cache, never in the
repository. A failing run replays its inputs through `tests/scripts/fuzz_triage.py`, which shrinks each reproducer
into `fuzz/regressions/<target>/` and uploads it as the `fuzz-regression-<target>` artifact. To close a finding:

```text
gh run download <run> -n fuzz-regression-<target> -D /tmp/fz    # or reproduce locally with cargo fuzz
mkdir -p fuzz/regressions
cp -r /tmp/fz/regressions/<target> fuzz/regressions/            # commit these with the fix
python3 tests/scripts/fuzz_triage.py replay <target>            # every run replays this directory first
```

`cargo-mutants` runs nightly (`mutants.yml`, one seventh of the mutants per night, the full set weekly) on the kernels, the hash functions, the SQL parser and the spill-file
format. The score is `caught / (caught + missed + timeout)` per area, reported in the run summary and the
`mutation-score` artifact; the floors are in `tests/mutation-floors.json`. To try it on one file:
`cargo mutants --package krishiv-kernels --file crates/krishiv-kernels/src/hash/murmur3.rs --test-tool nextest`.

## Precheck before you push

`cargo xtask precheck` runs what the required PR checks run, scoped to what your branch changed against the base, and
prints one table of pass, fail or skipped (with the reason) per check. It exits non-zero on any failure, so a PR
should pass CI on its first run.

```text
cargo xtask precheck [--base origin/main] [--fast] [--full] [--allow-stale] [--coverage-dependents] [--memory-cap 12G]
```

The default is the full gate. `--fast` is the quick loop (no coverage, budgets or Python wheel tests).

- `base-fresh`: fails when your branch lacks the newest commits of the base. CI tests the merge of the PR with
  `main`, so a stale branch can pass locally and fail there (registry coverage, `ty`, Python tests). Merge the base
  and re-run; `--allow-stale` skips the check.
- Rust checks run on the touched crates (tests also on their dependents); a change to `Cargo.toml`, `Cargo.lock`, the
  toolchain or lint config runs the whole workspace.
- `diff-coverage` runs `cargo llvm-cov nextest` on the touched crates only, in a dedicated target directory
  (`$CARGO_TARGET_DIR-precheck-cov`, or `target/precheck-cov`) that precheck deletes afterwards, then applies CI's
  90% gate to the changed lines. Tests in dependent crates also cover lines in CI: if only those cover your lines, add
  `--coverage-dependents` (slower), or better, a test in the crate itself.
- `policy` runs everything CI's policy job runs: SPDX, source guard, registry coverage, large files, DCO, the
  AI-attribution ban (your commits; the PR title and body are checked in CI) and the PR size cap, plus `workflow-lint`.
- Python: `python-quality` (ruff check, ruff format, ty) and `python-tests` (bench) run when their directories change;
  `python-wheel-tests` builds the extension with `maturin develop` into a throwaway venv and runs `python/tests` when
  `python/` changed or `krishiv-python` depends on a touched crate (needs `uv` and network).
- `codeql-proxy` is a cheap stand-in for the one CodeQL query that has failed PRs (`rust/cleartext-logging`: a
  variable named `secret`, `password`, ... in a log, print, panic or assert message, also across lines and in untracked files). **CodeQL itself stays CI-only.**
- `--memory-cap SIZE` puts the heavy steps (clippy, doc, tests, coverage, budgets, wheel tests) under a `systemd-run`
  memory cap. Putting the whole command under `systemd-run --user --scope -p MemoryMax=...` works too.
- Budgets run only when a manifest changed, or with `--full`.
- A tool that is not installed (`cargo-nextest`, `cargo-llvm-cov`, `zizmor`, `typos`, `taplo`, `uv`, ...) is reported as
  `not installed`, never skipped silently. Full logs are in `target/precheck/<check>.log`.
- Checks CI runs elsewhere (`hack`, `msrv`, the wheel matrix, `wheel-sizes`, `dependency-review`, CodeQL, CodSpeed,
  OSV) are not run locally.

The mapping from each required CI check to a precheck step is in `xtask/src/precheck.rs`; a policy test in
`xtask/tests/precheck.rs` fails when a required check has no mapping. When branch protection gains a required check,
add it to `REQUIRED_CI_CHECKS` and map it.

## Profiling a query

`cargo xtask profile` builds the release CLI with line tables, records the query profile as JSON, and samples
a second run for a flamegraph:

```text
cargo xtask profile <QUERY | -f FILE> [--table NAME=PATH]... [--conf KEY=VALUE]... [--tool auto|perf|samply] [--out DIR] [--no-flamegraph]
```

`<QUERY>` is SQL text, or a file with `-f`. `--table` registers a Parquet file or directory as a temp view
(repeatable); `--conf` sets a session entry (repeatable). Pass a single statement: the CLI prints one JSON
line per statement and the summary reads one document. The out directory defaults to
`target/xtask-profile/<UTC milliseconds>` and is never overwritten; pass `--out DIR` to choose it.

The sampler is chosen by `--tool` (`auto` by default): `perf` runs
`perf record -F 999 -g --call-graph dwarf` and pipes `perf script` through `inferno-collapse-perf` and
`inferno-flamegraph`; `samply` runs `samply record --save-only`. `auto` prefers `perf` when `perf` and both
inferno binaries are on `PATH`, and otherwise uses `samply` when it is installed (when neither combination
is available the command reports the profile JSON as done and then fails with the install hints). Install
what is missing with the
distribution's `linux-tools` package (for `perf`), `cargo install inferno` (for the two inferno binaries) or
`cargo install samply`. When no sampler is installed the command reports the profile JSON as done and then
fails with the install hints.

Output files are `<out>/profile.json` (the query profile, one JSON line, as printed by
`krishiv sql --explain-analyze --json`) and, with `samply`, `<out>/profile.json.gz` in samply's own format
alongside it; with `perf`, `<out>/perf.data` with `<out>/flamegraph.svg`. `--no-flamegraph` skips the
sampling run. The sampling run re-runs the same `krishiv sql --explain-analyze --json` command, so the
flamegraph profiles the instrumented query (profiling overhead included), not a plain run; that is
intended, keeping one command line for both artifacts.
The command ends with a summary: the output paths, the query wall time and the three slowest operators by
compute time.

The profile JSON (`format` `krishiv.query-profile`, version 1) holds the query wall time, per-pipeline driver
counts and the per-operator counters (rows in and out, compute time, blocked time, spills, peak memory);
the schema is `docs/reference/query-profile-v1.schema.json`. The SVG shows where the sampled run spent time;
wide boxes are the functions to look at first.

`perf record` needs the kernel to allow unprivileged profiling. When it refuses, the command prints the
`perf_event_paranoid` note: a privileged user lowers the setting (for example
`sysctl kernel.perf_event_paranoid=1`). xtask never changes the setting and never runs `sudo`.

## Python quality

All Python (`python/`, `bench/`, `tests/scripts/`) is checked with the Astral toolchain: ruff (lint and format) and
ty (types). Versions are pinned in each `uv.lock`; configuration lives in each `pyproject.toml`. The required CI job
`python-quality` runs, per directory:

```text
uv run --frozen ruff check
uv run --frozen ruff format --check
uv run --frozen ty check
```

Any `ignore` or `noqa` needs a reason comment. Python line coverage (`pytest --cov=krishiv`, Codecov flag `python`)
is reported from the `wheels` job; a nightly job also reports the Rust code reached through the Python API (flag
`rust-from-python`). Both are informational.

## Python versions

Krishiv supports CPython 3.11 and newer (`requires-python = ">=3.11"`). The default wheel is a single abi3 wheel
(`abi3-py311`) that runs on every CPython from 3.11 on; the free-threaded build (`cp314t`) cannot use abi3 and is
built separately. When a new CPython is released it is added to the CI matrix; support for the oldest one is dropped
no earlier than its end of life. See [python/README.md](python/README.md).

CI runs only on GitHub-hosted runners; self-hosted runners are not used (`cargo xtask workflow-lint` enforces this).

## Developer Certificate of Origin

Contributions are accepted under the Apache-2.0 license and the Developer Certificate of Origin (DCO). Every
commit must carry a sign-off:

```text
git commit -s
```

CI fails on commits without a `Signed-off-by` line (`cargo xtask dco`).

## Pull requests

- Keep a PR under 5,000 counted changed lines. Test code counts at half weight; lock files,
  snapshots, goldens, test fixtures and vendored data are excluded. Group related work into one PR with one commit per task. A maintainer can apply the
  `size-approved` label to allow a larger one.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`feat:`, `fix:`, `chore:`).
- Fill in the [pull request template](.github/pull_request_template.md), including the tests you ran.
- Source files start with an `SPDX-License-Identifier: Apache-2.0` header (`cargo xtask spdx`).
- Lint exceptions use `#[expect(lint, reason = "...")]`, never a bare `#[allow]`.
- Every `unsafe` block needs a `// SAFETY:` comment. Write tests first; update docs in the same PR.
- Distributed execution readiness: every execution change must also work when the plan runs on several
  worker processes. Physical plans stay serializable data (no service handles, locks, atomics or closures in
  plan nodes; resources are resolved when a worker instantiates the plan); operator state is partition-local
  and combined by explicit merges, never shared in-process state; partitioning uses a named, stable hash,
  never a per-process seed; a task's output does not depend on its attempt number. Reviewers ask: does this
  assume one process?

## Design records

Design documents (ADRs and the specification) are maintained in a private repository for now and are referenced by
number, for example "ADR 0034". If you need context for a referenced record, ask in the issue or PR and a
maintainer will summarize it.

## Spark compatibility wording

Say "Spark-compatible" to describe behavior. Apache Spark is a trademark of the Apache Software Foundation (ASF);
Krishiv is an independent project and is not affiliated with or endorsed by the ASF. Spark-observable
expectations must come from executing Spark, never from transcription.
