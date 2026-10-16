---
name: krishiv-release
description: Use for anything release-related in Krishiv — versioning, changelog, release-plz dry runs, wheel builds, SBOM, signing setup. DRY RUN ONLY; publishing to crates.io, PyPI, npm or a registry is a hard stop that requires the user.
---

# Release (dry run only) — ADR 0026, 0028, 0031

> **HARD STOP.** Never run `cargo publish`, `cargo release`, `uv publish`, `maturin publish/upload`, `twine upload`,
> `npm/pnpm publish`, `docker push`, `gh release create`, `release-plz release`, or push a
> release tag. `.claude/settings.json` denies
> several of these; that is a backstop, not permission for the rest. If a release is wanted, prepare everything,
> then stop and ask the user to run the publish step themselves.

Read by path: `../krishiv-internal/adr/0026-crate-layering-public-surfaces-formats.md` (surfaces, semver promises,
binary/wheel matrix), `0028-dependency-policy-and-arrow-skew.md` (release tooling, SBOM, signing),
`0033-license-apache2-dco.md` (license status — must be Accepted before any public release).
Tooling is **release-plz** (config: `release-plz.toml`, everything disabled; all crates `publish = false`; see `RELEASING.md`). Cargo-dist is not used yet (revisit in P6).

## Checklist — release candidate (dry run)
1. On a release branch (`release/vX.Y.Z`), never `main` directly; CI green on the commit.
2. Versions: workspace version bumped once; internal crates share the facade version ("no stability guarantees"
   in their docs); public crates pass `cargo semver-checks`.
3. Changelog from merged PRs; compatibility matrix (Spark Connect 3.5/4.0/4.1/4.2 pass rates) and benchmark
   headline numbers linked.
4. Gates: `cargo deny check`, `cargo xtask budgets`, `cargo msrv verify`, `cargo hack check --each-feature --workspace`,
   full test suite, upgrade tests (N reads N−1 formats).
5. Local-only dry runs: `cargo package --list -p <crate>`, `release-plz update` (local
   changes only), `maturin build --release` (wheels: abi3-py310 + cp313t/cp314t). `cargo publish --dry-run` is
   denied in settings because it shares the `cargo publish` prefix; use `cargo package` instead. `release-plz release --dry-run` is also denied by settings (`release-plz release:*`): run it only with the user's explicit approval, never on your own.
6. SBOM (`cargo cyclonedx`) and signing (sigstore) configured, not executed against public infra.
7. Write the release notes and a checklist of the publish commands into the PR for the user. **Stop.**
