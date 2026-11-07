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

## Before any release

- `ui/scripts/sync-assets.sh` refreshes the committed UI assets
  (`crates/krishiv-ui-assets/assets/`).
- The exit-gate checks of the phase plan are green and the maintainer has
  approved the gate review.
- The README says "Spark-compatible" only as a goal and API shape, never as a
  certification, and carries the trademark notice.
