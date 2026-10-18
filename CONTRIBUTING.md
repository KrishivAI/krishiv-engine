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
cp -r /tmp/fz/regressions/<target> fuzz/regressions/            # commit these with the fix
python3 tests/scripts/fuzz_triage.py replay <target>            # every run replays this directory first
```

`cargo-mutants` runs nightly (`mutants.yml`, one seventh of the mutants per night, the full set weekly) on the kernels, the hash functions, the SQL parser and the spill-file
format. The score is `caught / (caught + missed + timeout)` per area, reported in the run summary and the
`mutation-score` artifact; the floors are in `tests/mutation-floors.json`. To try it on one file:
`cargo mutants --package krishiv-kernels --file crates/krishiv-kernels/src/hash/murmur3.rs --test-tool nextest`.

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

## Design records

Design documents (ADRs and the specification) are maintained in a private repository for now and are referenced by
number, for example "ADR 0034". If you need context for a referenced record, ask in the issue or PR and a
maintainer will summarize it.

## Spark compatibility wording

Say "Spark-compatible" to describe behavior. Apache Spark is a trademark of the Apache Software Foundation (ASF);
Krishiv is an independent project and is not affiliated with or endorsed by the ASF. Spark-observable
expectations must come from executing Spark, never from transcription.
