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
Svelte 5 + Tailwind CSS v4 + shadcn-svelte components (on Bits UI), copied into `ui/src/lib/components/ui/` and restyled once to the
brand. `ui/src/app.css` maps Tailwind's theme and shadcn's CSS variables to the brand `--k-*` tokens (`ui/src/brand/tokens.css`, dark and
light); Red Hat Display/Text/Mono are self-hosted (`fonts-LICENSE.txt` ships beside them). Rebuild with `ui/scripts/sync-assets.sh`.
`ui/tests/check_pages.py` is the snapshot check (5 pages x 390/820/1440 px x dark/light, run against `ui/tests/fixture_server.py` under the
real CSP: no sideways scroll, no console error, no request outside the origin, fonts loaded, touch targets, the mobile menu).

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
