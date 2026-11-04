# krishiv-store

**Layer:** L1 (may depend only on crates in L1 or lower; `cargo xtask layering`). Depends on `krishiv-common` and
`object_store` only (ADR 0044). **Real code arrives:** P3, units S1-S4 of `design/P3-shared-storage-layer.md`.

## Purpose
The object-store layer the engine, the platform and the Iceberg bridge share: URL parsing, store building from `fs.*`
settings and per-read options, typed secrets, expiring per-location credentials with refresh, and the retry and pacing
policies. A published crate (manifest `publish = false` until the owner's first release): its public API is for outside
users.

## Key files
Paths are relative to this crate.
- `src/lib.rs` crate docs (skeleton until S1 moves `krishiv-io/src/io/stores.rs` here)

## Invariants
- No engine-internal type in a public signature: no `IoScheduler`, `MemoryPool`, plan, session or Arrow type. Settings
  come through `StoreSettings`, not `krishiv_common::config::Config` directly.
- Public API stability: after the first publish, changing or removing a public item needs an ADR (ADR 0044, 0026).
- Secrets (ADR 0038): sensitive values are `SecretString` from the moment they are read and exposed only at the
  object_store builder or credential call; `Debug`, errors and store ids never carry a secret or a key id.
- Store ids are unique per process (the `IoScheduler` is process-wide): scoped ids carry a random nonce.
- Cargo features are additive: `cloud` (object_store `aws`, `azure`, `gcp`), later `remote-signing`.
- `#![forbid(unsafe_code)]`. No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0026, 0027, 0028, 0038, 0044

## Tests
`cargo nextest run -p krishiv-store --no-tests=pass` · `cargo clippy -p krishiv-store --all-targets -- -D warnings`
