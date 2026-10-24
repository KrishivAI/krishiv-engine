# krishiv-ui-assets

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P0-T26 placeholder, P2+.

## Purpose
Embedded prebuilt web UI assets (`ui/` dist via rust-embed).

## Key files
Paths are relative to this crate.
- `src/lib.rs` the embedded asset set (rust-embed)
- `assets/index.html` built entry page; `assets/assets/` built JS and CSS

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
