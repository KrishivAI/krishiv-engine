# krishiv-alloc

**Layer:** L0 (may depend only on crates in L0 or lower; `cargo xtask layering`). Linked only by binaries
(`krishiv-cli` under the `mimalloc` feature, `krishiv-bench-runner`); libraries never depend on it.

## Purpose
mimalloc policy for the Krishiv binaries: under a configured memory limit it sets allocator options before the first
arena is reserved, and it provides the pressure release hook (`mi_collect`) for `krishiv-memory`.

## Key files
Paths are relative to this crate.
- `src/lib.rs` the allocator policy (about 460 lines, with its tests):
- `LIMITED_OPTIONS` the three `MIMALLOC_*` options (purge delay, THP off, minimal purge size) and their mimalloc option numbers
- `limit_in_args` detects `--memory-limit` or `krishiv.memory.limit=` in program arguments
- `LIMIT_VAR` the `KRISHIV_MEMORY_LIMIT` environment variable
- `init` (glibc only, inside the `.init_array` constructor) applies `LIMITED_OPTIONS` when a limit is known at process start
- `release` the pressure release hook (`mi_collect(true)`)
- `install` links the constructor and returns `release` for `krishiv_memory::set_release_hook`
- `src/heap.rs` (feature `heap-stats`) `Counting<A>`, a global-allocator wrapper that counts live and peak heap bytes for memory measurements; `tests/heap_stats.rs` covers it
- `Cargo.toml` depends on `libmimalloc-sys` (`extended`) only

## Extension points
- A new allocator option: add a row to `LIMITED_OPTIONS` with its mimalloc `mi_option_e` number and pin the number in a test.
- A new way to signal a limit at process start: extend `limit_in_args` (or `LIMIT_VAR`) and its test.
- A host that embeds the engine owns its allocator; it calls `krishiv_memory::set_release_hook` itself rather than using this crate.

## Tests
`systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- cargo nextest run -p krishiv-alloc --no-tests=pass` ·
`cargo clippy -p krishiv-alloc --all-targets -- -D warnings`. Unit tests in `src/lib.rs` (`mod tests`) pin the option
numbers to the linked mimalloc and check the argument detection and the null-`argv` case.

## Invariants
- The options must be in force before mimalloc reserves its first arena, so they are applied by the constructor, not from `main`.
- They apply only when the limit is known at process start and the user has not set the `MIMALLOC_*` variable; with no limit nothing changes.
- A limit from a configuration file or a runtime `SET` is enforced by the memory pools and uses `release`, without the options.
- Its `unsafe` blocks (the constructor, the mimalloc calls) each carry a `// SAFETY:` comment.
