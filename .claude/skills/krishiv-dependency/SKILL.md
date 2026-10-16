---
name: krishiv-dependency
description: Use before adding, upgrading or removing any Rust (or Python/JS) dependency in krishiv-engine, including Renovate PRs and Arrow major bumps. Applies the ADR 0028 checklist, cargo-deny, machete and the budgets gate.
---

# Add or change a dependency (ADR 0028, 0026, 0029)

Read by path: `../krishiv-internal/adr/0028-dependency-policy-and-arrow-skew.md` (versions, bans, checklist),
`0026-crate-layering-public-surfaces-formats.md`; `deny.toml`; `budgets.json`; root `Cargo.toml` `[workspace.dependencies]`.

## Banned (deny.toml enforces): `datafusion*`, `deltalake`, `iceberg-datafusion`, `rocksdb`, `openssl` (use rustls),
`anyhow` in libraries, `async-std`, `actix`, `figment`, `chrono-tz`, `pyo3-arrow`, `arrow-pyarrow`, `divan`.

## Checklist (copy into the PR description, answered)
1. **Why not own code or an existing dependency?** ADR 0028 prefers writing small things ourselves (Murmur3, sketches,
   Java regex translation, datetime patterns, PyCapsule bridge, config registry).
2. **License** in the allowlist (Apache-2.0, MIT, BSD-2/3, ISC, Zlib, Unicode-3.0, MPL-2.0 file-level); **maintenance**
   (recent release, responsive issues, bus factor).
3. **Cost:** transitive crate count (`cargo tree -p <crate> -e normal | wc -l` before/after), clean build time,
   binary/wheel size — compare with `cargo xtask budgets` (fails on > 5% jump without justification).
4. **Risk:** `unsafe`/C/C++ build, MSRV (stable − 2), proc-macro weight, version skew with Arrow 60 / tokio / prost / pyo3.
5. **Placement:** declared once in `[workspace.dependencies]` with `default-features = false`; crates use
   `<dep>.workspace = true`; heavy or C deps behind an additive Cargo feature; depend on Arrow sub-crates, not `arrow`.
   Only the crate that needs it, respecting layers.
6. **Heavy addition → ADR first** (`.claude/skills/krishiv-adr/SKILL.md`); stop and ask if unsure.

## Verify
```
cargo update -p <crate>            # never a blanket `cargo update` in a feature PR
cargo deny check                   # licenses, bans, advisories, multiple-versions (arrow/tokio/prost/pyo3)
cargo machete                      # no unused deps
cargo hack check --each-feature --workspace   # features stay additive
cargo xtask budgets                # crate count / build time / size budgets
cargo xtask layering
cargo clippy --workspace --all-targets -- -D warnings && cargo nextest run --workspace
```
`Cargo.lock` is committed. Duplicate Arrow is allowed only for the Iceberg isolation exception written in `deny.toml`.

## Upgrades
Renovate groups Arrow family, tonic+prost, OpenTelemetry weekly. Adopt each new **Arrow major within 2 weeks**.
Bumps are mechanical (Haiku/Sonnet) but still get review; never mix a bump with feature work.
