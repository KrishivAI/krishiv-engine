# krishiv-jose

**Layer:** L1 (may depend only on crates in L0–L1; `cargo xtask layering`). It depends on no workspace crate.

## Purpose
A standalone token verifier, usable without the engine's query crates (ADR 0050):
JWS compact tokens, JWK sets and OpenID discovery documents parsed strictly, the claim rules, and the signature
check. Taken unchanged from `krishiv-session`'s `auth/jose.rs` and `oidc.rs` `verify_signature`. The key-set cache,
its fetch, `OidcConfig` and every mapping to `KrishivError` stay in `krishiv-session` (`src/auth/`).

## Key files
- `src/lib.rs` the parsers: `Alg`, `Curve`, `JoseError` (its own small error, no `KrishivError`), `Object` (a JSON
  object with distinct members), `parse_jws` / `Jws` / `Header`, `looks_like_jws`, `check_claims` / `ClaimRules` /
  `Identity` / `Actor`, `ExchangeClaims`, `parse_jwks` / `Jwk` / `KeyMaterial`, `discovered_jwks_uri`,
  `is_format_char`, the size limits; `tests.rs`
- `src/verify.rs` (feature `verify`) `verify_signature` on aws-lc-rs: RSA PKCS#1 v1.5 2048-8192 bits, ECDSA
  P-256/P-384 in the fixed `r || s` form; the key must fit the algorithm (`Jwk::fits`); `verify_tests.rs`
- `src/der_signature.rs` `ecdsa_der_to_fixed`: a DER ECDSA signature (a key service's) to the fixed form, strict
  DER only; `der_signature_tests.rs` (RFC 6979 A.2.5 vector, aws-lc-rs round trips, malformed DER refusals)

## Invariants
- Parsing uses only `std`, `serde`, `serde_json`: no clock, network or cryptography; `aws-lc-rs` only behind
  `verify` (`krishiv-session`'s `oidc` turns it on).
- No `none`, no HMAC: such tokens are not parsed. Keys never come from the token header.
- A public surface (ADR 0026, 0050): changing or removing a public item needs an ADR; additions are allowed.
  Enums and structs that may grow are `#[non_exhaustive]`; `ClaimRules` is built by callers, so a new rule is a
  new field and an ADR.
- Never put token content in an error (`JoseError` carries fixed reasons only).
- `#![forbid(unsafe_code)]`; no `unwrap`/`expect`/`panic` outside tests.
- Fuzzed by `fuzz/fuzz_targets/jwt_parse.rs`.
- `public-api.txt` is the snapshot of the public surface with every feature (nightly `public-api` job):
  regenerate with `cargo +nightly-2026-08-12 public-api -ss -p krishiv-jose --all-features`.

## ADRs (in `../krishiv-internal/adr/`)
0026, 0027, 0028, 0029, 0046, 0050

## Tests
`cargo nextest run -p krishiv-jose --all-features` · `cargo clippy -p krishiv-jose --all-targets --all-features -- -D warnings`
