# Governance

This document describes how Krishiv is run: who decides what, and how a change gets in. It applies to the
`krishiv-engine` repository. The project's code of conduct is in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and
security reports follow [SECURITY.md](SECURITY.md).

## Roles

- **Maintainers** merge changes, cut releases and decide on design questions for the project. The current maintainer
  is listed in [`.github/CODEOWNERS`](.github/CODEOWNERS). The project currently has one maintainer and no selection
  process yet.
- **Contributors** are anyone who opens an issue, a pull request, a review or a discussion. No membership or
  approval is needed to contribute; see [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

## Decisions

- **Small changes** (bug fixes, documentation, tests, internal refactors that keep behaviour) are decided in the pull
  request: a maintainer reviews, asks for changes if needed, and merges once the checks pass and review threads are
  resolved.
- **Spark-observable behaviour** (SQL and DataFrame semantics, error classes) must match Apache Spark for the Spark
  versions Krishiv targets, or state the divergence explicitly and make it opt-in. Public Rust and Python APIs,
  on-disk formats and `krishiv.*` configuration keys follow [the stability policy](docs/concepts/stability.md). The
  reference for Spark behaviour is the Apache Spark source at the supported tags, checked against a real Spark run,
  not memory.
- **Architectural decisions** are recorded as numbered design records (ADRs). Those records are kept in a private
  repository for now, and are referenced by number in code and issues. A maintainer summarises a record on request.
  An accepted record changes only through a new record that supersedes it.
- **Disagreements** are settled first in the issue or pull request. If they remain open, a maintainer decides and
  gives the reasons in writing. Maintainers aim to explain a decision rather than only state it.

## Releases

Krishiv follows semantic versioning. While the version is `0.x`, a minor release may contain breaking changes; the
changelog says so, and deprecations last at least one minor release. Release notes are generated from the changelog
fragments that each pull request adds (`changelog.d/`).

## Changing this document

Changes to governance go through a pull request like any other change and need approval from a maintainer.
