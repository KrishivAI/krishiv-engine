# krishiv-jose

A standalone token verifier, usable without the engine's query crates (ADR 0050): strict JWS compact token and JWK
set parsing, OpenID discovery, claim rules, and (feature `verify`) RSA PKCS#1 v1.5 and ECDSA P-256/P-384 signature
checks on aws-lc-rs. `none` and HMAC are never parsed. `ecdsa_der_to_fixed` turns a DER ECDSA signature into the
fixed `r || s` form JOSE uses.

Licensed under Apache-2.0.
