# Releasing

**Publishing is a hard stop (ADR 0031).** No agent, script or CI job may
publish anything: no `cargo publish` (other than `--dry-run`), `maturin publish`,
`uv publish`, `twine upload`, `pnpm publish`, no `release-plz release` (not even
`--dry-run`, without the user's explicit approval), no tags or GitHub releases.
Only the maintainer (gopal) decides to publish, explicitly, and only after the
steps below.

## What happens on merge or on a tag: nothing

- No workflow publishes on push, merge, tag or schedule. The only workflow that
  can upload a package is `publish-testpypi.yml`, and it runs on
  `workflow_dispatch` alone (see "TestPyPI" below).
- Every crate has `publish = false`, so even a hand-typed `cargo publish`
  refuses.
- `release-plz.toml` disables release, publish, git release and git tags for
  the whole workspace, and no workflow runs release-plz.
- The unit test `release_publishing_is_manual_and_gated` in
  `xtask/tests/policy.rs` fails CI if any of that changes: a publish trigger
  other than `workflow_dispatch`, a publish step outside the TestPyPI
  workflow, a missing `environment:` gate, a switch turned on in
  `release-plz.toml`, or a crate without `publish = false`.

Enabling a release is therefore always a reviewed change that edits those
guards, made by the maintainer.

## Current state: `0.2.0`

- The version is `[workspace.package] version = "0.2.0"` in `Cargo.toml`,
  inherited by every crate. Python gets the same version from
  maturin.
- `CHANGELOG.md` holds the notes. Later entries come from
  `release-plz update` (Conventional Commits; edits a local checkout only) and
  are edited by hand.
- The first release has not been published anywhere.

## Dry runs (safe, run in CI)

- **`cargo publish --dry-run` for `krishiv`**: the `publish-dry-run` job of
  `ci.yml` runs `.github/scripts/publish-dry-run.sh`. Because the repository
  forbids publishing, the script works on a scratch copy of the tracked files
  in which the `krishiv` dependency closure has `publish = false` dropped and
  versions added to its path dependencies (the same edit that enabling a real
  release needs), then packages every crate, builds each from its packaged
  files and stops before any upload. The repository is not modified.
  `--package-only` runs `cargo package` instead, for machines whose policy
  refuses the word `publish`.
- **Wheels**: `wheels.yml` builds and tests every wheel in a clean venv.
- **Quickstart**: the `wheels` job of `ci.yml` runs
  `.github/scripts/quickstart-test.sh`, which executes every block of the
  README's Quickstart from a clean venv against the freshly built wheel. A
  change to the README example that does not run fails CI.

## TestPyPI (manual, owner's go required)

`publish-testpypi.yml` builds all wheels (via `wheels.yml`) and uploads them to
TestPyPI. It is gated twice:

1. It runs only when dispatched by hand, on `main` or a `v*` tag.
2. Its upload job targets the `testpypi` GitHub environment. The maintainer
   creates that environment with **required reviewers** (Settings, Environments),
   restricts its deployment branches/tags to `main` and `v*`, and registers
   the workflow as a **trusted publisher** on test.pypi.org (project `krishiv`
   and, for the older-CPU wheel, `krishiv-compat`; workflow
   `publish-testpypi.yml`, environment `testpypi`). Until the environment
   exists with reviewers, nobody can approve the run, and no token exists in
   the repository: upload uses OIDC trusted publishing.

To publish after the owner's go: confirm `CHANGELOG.md` and the version, then
Actions, "Publish to TestPyPI", "Run workflow" on `main` (or the tag), and
approve the `testpypi` deployment. Verify the install from TestPyPI in a clean
venv: `pip install --index-url https://test.pypi.org/simple/ --extra-index-url
https://pypi.org/simple/ "krishiv[pyarrow]==0.2.0"`.

## Release artifacts (manual, publishes nothing)

`release.yml` (dispatch on `main` or a `v*` tag) builds the wheels, the Linux and macOS arm64 CLI binaries,
`SHA256SUMS`, a CycloneDX SBOM and build-provenance attestations from one commit, and keeps them as the
`release-bundle` workflow artifact. It has no publish step, tag or GitHub Release; the gated publish jobs are a
later change by the maintainer.

## Release channels for 0.2.0 (owner decision 2026-11-07)

0.2.0 goes to three channels. Docker and Helm are **not** part of it; they move to P3 with the server and Spark
Connect work. Every channel stays a hard stop: tag and publish only on the owner's explicit go.

| Channel | What ships | Dry run today |
|---|---|---|
| PyPI `krishiv` | abi3 (cp311+) and free-threaded cp314t wheels: manylinux x86_64/aarch64, musllinux, macOS arm64 | `wheels.yml` builds and tests every wheel; `publish-testpypi.yml` (manual) uploads them to TestPyPI |
| crates.io `krishiv` | the facade plus the internal crates at the same version (ADR 0026) | `publish-dry-run` job of `ci.yml` (`.github/scripts/publish-dry-run.sh`) |
| GitHub Release `v0.2.0` | changelog, quickstart, `krishiv` CLI binaries for Linux x86_64, Linux aarch64 and macOS arm64, SBOM, signatures | `binaries.yml` builds and smoke-tests the three binaries (and the `krishiv-compat` x86-64 baseline); the SBOM, the signatures and the release upload are **not wired yet** |

Not wired yet (each is a reviewed change by the maintainer, because it edits the guards in
`release_publishing_is_manual_and_gated`):

- dropping `publish = false` and adding versions to the path dependencies of the crates that ship (what
  `publish-dry-run.sh` simulates), and the PyPI and crates.io credentials (a PyPI trusted publisher over OIDC and a
  scoped crates.io token) behind `environment:` gates with required reviewers;
- a gated release workflow (`workflow_dispatch` only) that downloads the `binaries.yml` artifacts, writes checksums
  and an SBOM, signs them and drafts the GitHub Release. Its dry-run mode must stop before any upload or tag.

Owner actions before the gate are listed in `ops/needs-owner.md` ("Release 0.2.0 channels and pre-publish setup"):
crates.io keywords and description (drop `datafusion`), the PyPI classifier, the credentials, and the go/no-go.

## Release integrity

- **Provenance:** every file in the `release-bundle` artifact (wheels, binaries, SBOMs) has a GitHub build-provenance
  attestation, and `release.yml` verifies each one with `gh attestation verify` before it keeps the bundle. Anyone can
  check a downloaded file: `gh attestation verify <file> --repo KrishivAI/krishiv-engine`.
- **SBOMs:** one CycloneDX file per shipped package (facade, CLI, Python extension) sits in the bundle; attach them to
  the GitHub Release with the binaries and `SHA256SUMS`.
- **PyPI:** trusted publishing over OIDC, no token (`publish-pypi.yml`, `publish-testpypi.yml`).
- **crates.io:** still a scoped token in the `crates-io` environment. crates.io supports trusted publishing
  (`rust-lang/crates-io-auth-action`); moving to it means registering the workflow on each crate and editing the
  publish guard, so it is a reviewed change by the maintainer after 0.2.0, not part of the dry run.
- **Signed tags:** the owner creates the release tag signed, `git tag -s v0.2.0 -m "krishiv 0.2.0"`, and checks it with
  `git verify-tag v0.2.0` before pushing it. Agents never tag.

## The 0.2.0 release, step by step (owner only)

Each step is a hard stop for agents; the maintainer runs it and approves its environment.

1. **Merge** the release PR on a green main, after the P2b gate is approved. Then tag `v0.2.0` on that commit and push the tag.
2. **TestPyPI:** in Actions, run "Publish to TestPyPI" on the tag. Approve the `testpypi` environment, then install from TestPyPI in a clean virtualenv: `pip install -i https://test.pypi.org/simple/ krishiv==0.2.0` (and `krishiv-compat`), and run the quickstart.
3. **PyPI:** run "Publish to PyPI" (`publish-pypi.yml`) on the tag and approve the `pypi` environment. Check <https://pypi.org/p/krishiv> and `pip install krishiv==0.2.0`.
4. **crates.io:** run "Publish to crates.io" (`publish-crates.yml`) on the tag and approve the `crates-io` environment. It publishes the 20 crates of `.github/scripts/publish-crates.txt` in that order. A crate already at 0.2.0 is skipped, so a failed run can be dispatched again. Check `cargo add krishiv@0.2.0` in a scratch project.
5. **Yank the old 0.1 releases** once 0.2.0 is verified: PyPI `krishiv` 0.1.0; crates.io `krishiv` 0.1.0-nightly.202607030148 and `krishiv-common` 0.1.0 (`cargo yank --version <v> <crate>`). See below.
6. **GitHub Release** for `v0.2.0`, with the CHANGELOG 0.2.0 section as its notes.

## Yanking a bad release (owner only)

Every step is a hard stop for agents; the maintainer runs them. Yanking hides a version from new resolves and
deletes nothing, so exact pins keep working.

1. **PyPI**: project `krishiv` (and `krishiv-compat`), Manage, Releases, the version, Options, Yank, with a reason.
   There is no token in the repository, so the web UI is the way.
2. **crates.io**: `cargo yank --version 0.2.0 <crate>` for every published crate (the facade and the internal
   crates at that version, ADR 0026). `cargo yank --undo` reverses it.
3. **GitHub Release**: mark `v0.2.0` as a pre-release and say in its notes that it is yanked. Keep the tag; never
   delete or move a published tag.
4. **Record it**: retitle the CHANGELOG entry `## 0.2.0 (yanked)` with the reason and the fixed version.
5. **Fix forward**: land the fix on `main`, bump the patch version and release `0.2.1` through the same gates.
   A yanked version number is never reused.

## Release train

Owner decision 2026-11-07 (default adopted): a minor release every **6 weeks**.

- **Cut:** one week before the release date, branch `release-X.Y` from `main`. That starts the one-week
  **RC freeze**: only fixes for regressions, doc corrections and version bumps go onto the branch (cherry-picked
  from `main`, or fixed there and merged back). `main` stays open for the next minor.
- **RC:** tag `vX.Y.0-rc.N` from the branch; it publishes nowhere until the owner says go (see above).
- **Release:** on the date, tag `vX.Y.0` from `release-X.Y`. Patch releases (`X.Y.Z`) are cut from the same branch.
  Support covers the latest minor only while the major is 0.
- **Dates:** every date is 6 weeks (42 days) after the previous minor's release date. 0.3 is planned as
  *0.2 release date + 42 days*, its branch cut 7 days before that; the 0.2 date is set by the owner's publish go, so
  record the 0.3 dates in `STATUS.md` the day 0.2 ships. A slipped release does not move the next one: the train
  keeps its dates and the next minor takes what is ready.
- Breaking changes (docs/concepts/stability.md) go into a minor only, with the changelog "Breaking" heading.

## Before any release

- `ui/scripts/sync-assets.sh` refreshes the committed UI assets
  (`crates/krishiv-ui-assets/assets/`).
- The exit-gate checks of the phase plan are green and the maintainer has
  approved the gate review.
- The README says "Spark-compatible" only as a goal and API shape, never as a
  certification, and carries the trademark notice.
