# krishiv-bench-runner

**Layer:** L7 (may depend only on crates in L7 or lower; `cargo xtask layering`).

## Purpose
Bench-only helper (P2-T142): a line-protocol process around `krishiv-session` that the Python bench harness
(`bench/`) drives while the Python facade cannot execute queries. Not part of the public CLI; `publish = false`.

## Key files
Paths are relative to this crate.
- `src/lib.rs` the line protocol (documented there) and the commands around `krishiv-session`
- `src/main.rs` binary entry; installs the mimalloc release hook from `krishiv-alloc`
- `build.rs` build-info values checked by `tests/build_info.rs`
- `tests/protocol.rs` protocol tests; `tests/memory_bound.rs` memory-bound tests; `tests/spill_equivalence.rs` spill tests

## Extension points
- New protocol command: add it to the protocol in `src/lib.rs` and cover it in `tests/protocol.rs`.

## Invariants
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic`.
- The protocol is documented in `src/lib.rs`; the Python adapter (`bench/src/krishiv_bench/adapters/krishiv.py`) is its only client.

## Tests
`cargo nextest run -p krishiv-bench-runner --no-tests=pass`
