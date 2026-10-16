# Releasing

**Publishing is a hard stop (ADR 0031).** No agent, script or CI job may
publish anything: no `cargo publish`, `maturin publish`, `uv publish`,
`pnpm publish`, no `release-plz release` without `--dry-run`, no tags or
GitHub releases. Only the maintainer (gopal) decides to publish, explicitly.

Current state (P0-T27, dry run only):

- Every crate has `publish = false`.
- `release-plz.toml` disables release, publish, git release, git tags and
  changelog edits for the whole workspace.
- Allowed check: `release-plz release --dry-run`. Today it reports
  "no public packages found", which is the intended safe result.

Enabling real releases requires a deliberate, reviewed change by the user to
the `publish` keys in each `Cargo.toml` and to `release-plz.toml`, plus the
licence decision in ADR 0033 and registry credentials, none of which exist in
this repository.

Prebuilt UI assets (`crates/krishiv-ui-assets/assets/`) are committed; refresh
them with `ui/scripts/sync-assets.sh` before a release. Python wheels are built
per `python/README.md` and are never uploaded from this repository's tooling.
