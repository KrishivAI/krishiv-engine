# krishiv-ui-assets

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P0-T26 placeholder, P2+.

## Purpose
Embedded prebuilt web UI assets (`ui/` dist via rust-embed).

## Key files
Paths are relative to this crate.
- `src/lib.rs` the embedded asset set (rust-embed)
- `assets/index.html` built entry page; `assets/assets/` built JS and CSS

## How the UI is built (ADR 0024 update 2026-11-06)
`ui/` is a pnpm workspace (shared-UI note step 1): `ui/packages/ui` is `@krishiv/ui` (tokens, components, data helpers, Vitest
tests), `ui/apps/console` the engine console that builds into `ui/dist`. Svelte 5 + Tailwind CSS v4 + shadcn-svelte components (on Bits UI),
copied into `ui/packages/ui/src/components/` and restyled once to the brand. `ui/packages/ui/src/tokens/index.css` maps Tailwind's theme and
shadcn's CSS variables to the brand `--k-*` tokens (`tokens/tokens.css`, dark and light); Red Hat Display/Text/Mono are self-hosted
(`fonts-LICENSE.txt` ships in `ui/apps/console/public`). Rebuild with `ui/scripts/sync-assets.sh`.
`ui/tests/check_pages.py` is the snapshot check (10 pages x 360/390/820/1440 px x dark/light, run against `ui/tests/fixture_server.py` under the
real CSP: no sideways scroll, no console error, no request outside the origin, fonts loaded, touch targets, the mobile menu), then `page_checks`: token out of the address and storage, hostile text rendered as text (XSS), the editor's cancel within 2 s, `***` on the environment page, and axe-core (`ui/node_modules`) with no serious violation on queries, catalog, SQL and environment.

## Extension points
- UI changes are made in `ui/` (Svelte source); rebuild and replace the built files here. Do not hand-edit them.

## Invariants
- Contains only built assets; UI source lives in `ui/`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0024

## Tests
`cargo nextest run -p krishiv-ui-assets --no-tests=pass` · `cargo clippy -p krishiv-ui-assets --all-targets -- -D warnings`
