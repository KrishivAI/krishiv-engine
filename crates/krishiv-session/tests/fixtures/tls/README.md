# TLS test fixtures (P3-T002 PR-14c)

Test-only certificates and keys, made with the `openssl` CLI (not a dependency) on 2026-11-12;
never use them anywhere else. EC P-256, valid until 2126 except `expired.pem`.

- `ca.pem`: the test CA (`CN=krishiv test CA`), signs everything below. Its key is not kept.
- `server.pem`/`.key`/`.der`, `server2.*`: `CN=localhost` / `CN=localhost-2`, SAN `DNS:localhost`, `IP:127.0.0.1`, serverAuth.
- `client.*`: `O=krishiv, CN=etl-job`, SAN `URI:spiffe://krishiv.test/etl`, `DNS:etl.krishiv.test`, clientAuth.
- `client2.*`: two CNs (`a`, `b`), two URIs and two DNS names: refused by every `principalFrom` rule.
- `nosan.*`: `CN=reporting`, no subjectAltName.
- `expired.*`: SAN URI `spiffe://krishiv.test/old`, valid 2020-01-01 to 2020-01-02.

Key files are mode 0600 in the tests (git does not keep modes; the tests copy them).
