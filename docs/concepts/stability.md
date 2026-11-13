---
description: Krishiv follows semantic versioning; what may break in a 0.x release and what never does.
---

<!-- SPDX-License-Identifier: Apache-2.0 -->
# API stability

Krishiv follows [semantic versioning](https://semver.org). While the major version is 0, a **minor** release
(0.x.0) may break the public API; a patch release (0.x.y) never does. From 1.0 only a major release breaks.

## What is public

- The Rust crates `krishiv`, `krishiv-session` and `krishiv-udf-sdk`. Their API is snapshotted in
  `crates/<crate>/public-api.txt`, and the nightly `public-api` job fails when the code and the snapshot differ,
  so every API change is a visible diff in its PR.
- The Python package `krishiv`. Its names and signatures are snapshotted in `python/public-api.txt`, and the
  `python-quality` job fails when the source and the snapshot differ (`python tests/scripts/python_public_api.py --write` updates it).
- SQL and DataFrame behaviour that Spark defines: it matches Spark for the supported versions (see
  [Spark compatibility](compatibility.md) and [known differences](known-differences.md)). A fix that moves
  Krishiv closer to Spark is not a breaking change.
- Configuration keys `krishiv.*`, and the on-disk and wire formats, which carry a version: release N reads N-1.

## What is not

- Anything under `krishiv.experimental.*`, and Cargo features named `unstable-*` (for example `unstable-ir`).
- Other workspace crates, which are implementation detail even when they are published.
- Items documented as internal or hidden from the docs.

## Breaking changes and deprecation

- Every breaking change is listed under a "Breaking" heading in the changelog for its release.
- A deprecated item stays for at least one minor release, with `#[deprecated]` (Rust) or a `DeprecationWarning`
  (Python) naming the replacement. Security fixes and Spark-exactness fixes may shorten this; the changelog says so.
- A config key is renamed by keeping the old name as an alias for one minor.

## Checks

- `public-api` (nightly): the snapshot diff above.
- `semver-checks` (nightly): `cargo-semver-checks` against the last released version. It is **non-blocking**
  until the 0.2 release exists to compare against, and **blocking** after that.
