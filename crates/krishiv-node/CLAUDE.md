# krishiv-node

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).
**Real code since:** P3, npm package `krishiv` 0.3.1 (design `design/node-engine-napi.md`, layout `design/js-package-layout.md`).

## Purpose
The embedded engine of the npm package `krishiv`: a Node-API addon (napi-rs 3) that `js/src/embedded.ts` wraps as the
package's embedded backend. Built by `napi build` from `js/` into `krishiv.<platform>.node`, shipped in the
per-platform packages `@krishiv/engine-<platform>`. `publish = false`: never on crates.io. An FFI crate under
ADR 0029: the `#[napi]` macros generate the `unsafe` glue; hand-written `unsafe` needs a `// SAFETY:` comment and
a reason here.

## Key files
Paths are relative to this crate.
- `src/lib.rs` module wiring, the global allocator (feature `mimalloc`), `version()`
- `src/session.rs` `NativeSession` (`open(conf)`, `execute(sql)`, `setConf`, `close`): one `krishiv_session::Session`
  with its own runtime, source `krishiv-node`; configuration applied in sorted key order
- `src/query.rs` `NativeQuery` (`next()`, `cancel()`): the first `next()` analyzes and starts the statement and
  returns the IPC schema message, then one chunk per batch, then the end-of-stream marker, then `null`; `Cancel`
  races every step against a cancel (dropping the step's future or the stream cancels the engine query); a failure
  or cancel ends the query
- `src/ipc.rs` `Encoder`: one Arrow IPC stream handed out a message at a time (`StreamWriter` over a drained `Vec`)
- `src/error.rs` engine errors as a JS `Error` whose message is JSON `{errorClass, sqlState, messageParameters, message}`
- `src/tests.rs` the Rust side without Node (stream, errors, cancel, close)

## Invariants
- Nothing runs on the JavaScript thread but the hand-over: `next()` is an `async fn` (napi `async` feature, its tokio
  runtime); the engine work runs on the session's own executor.
- No result is held whole: one batch per `next()`.
- A `.node` is never unloaded; `close()` frees the session.
- Features mirror the Python wheel's defaults (`mimalloc`, `cloud`, `delta`); on musl mimalloc uses
  `local_dynamic_tls` (the addon is `dlopen`ed).

## Tests
`cargo nextest run -p krishiv-node --lib` · `cargo clippy -p krishiv-node --all-targets -- -D warnings`;
the JavaScript side (vitest, Node 22/24, Bun and Deno smoke) lives in `js/test/`.
