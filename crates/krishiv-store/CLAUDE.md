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
- `src/lib.rs` crate docs and re-exports; `src/stores.rs` URL parsing (`ObjectScheme`, `ParsedUrl`, `parse_object_url`), the `fs.*` key constants, `StoreRegistry` (store building and caching per `StoreId`, per-read option overrides, `open`/`open_with_options` → `OpenedStore`) and the scheme builders (S3, GCS, Azure, https, `hf://` behind `cloud`)
- `src/settings.rs` `StoreSettings` (raw `fs.*` text by key, exact then ASCII case-insensitive; implemented for the engine's `Config`, `BTreeMap<String, String>`, `HashMap<String, String>` and `Arc<T>`); `src/id.rs` `StoreId` (redacting `Display`); `src/policy.rs` `StorePolicy` (pacing budgets per scheme); `src/error.rs` the `Result` alias and the two error helpers
- `tests/settings.rs` (S2): every key resolves the same from a `Config` and from maps, case-insensitively

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
