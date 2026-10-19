# krishiv-bench-runner

**Layer:** L7 (may depend only on crates in L7 or lower; `cargo xtask layering`).

## Purpose
Bench-only helper (P2-T142): a line-protocol process around `krishiv-session` that the Python bench harness
(`bench/`) drives while the Python facade cannot execute queries. Not part of the public CLI; `publish = false`.

## Invariants
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic`.
- The protocol is documented in `src/lib.rs`; the Python adapter (`bench/src/krishiv_bench/adapters/krishiv.py`) is its only client.

## Test
`cargo nextest run -p krishiv-bench-runner --no-tests=pass`
