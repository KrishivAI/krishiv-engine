---
description: "The three deployment modes: embedded, single-node and distributed."
---

# Deployment

| Mode | Status |
|---|---|
| Embedded: `pip install krishiv`, the engine runs inside your Python process | Runs today |
| Single-node: one `krishiv server` process that clients reach over Spark Connect | In development |
| Distributed: a scheduler and workers | In development |

## Embedded

Nothing to deploy: import the package. Supported CPUs are x86-64-v3 (Intel Haswell or AMD Zen and newer) and Armv8.2
(AWS Graviton2 or newer, Apple silicon). On an older x86-64 CPU install `krishiv-compat` instead (the same package,
imports as `krishiv`; install one of the two, not both). An unsupported CPU is detected at import time and reported by
name instead of crashing.

Memory use is bounded by `krishiv.*` settings (see [configuration](../reference/config/index.md)); operators spill to disk when the
limit is reached.

Nothing has been published to PyPI or crates.io yet. Until the first release, build the wheel from the repository
(`maturin build` in `python/`).

## Single-node

One process, started with `krishiv server` (a build with the `server` feature), runs the engine and serves Spark Connect; clients connect to it with a
Spark Connect client such as PySpark. It listens on `127.0.0.1` port `15002` by default (`krishiv.connect.port`, see
[configuration](../reference/config/index.md)). Where it differs from Spark see
[known differences](../concepts/known-differences.md#krishiv-server-spark-connect).

### Authentication, roles and TLS

Without authentication settings the server binds loopback only, and the admin port (`/metrics`, the web UI) prints a
random token at start. To serve another address, configure at least one method with `--conf` (these settings are
read by the server itself and never become session configuration):

- **Tokens:** `krishiv.auth.tokens.<name>.secret` (an `env:` or `file:` reference; a file readable by group or others
  is refused), `.principal`, `.role` (`viewer`, `operator` or `admin`). Spark's `spark.connect.authenticate.token` is
  one admin token.
- **OIDC:** `krishiv.auth.oidc.issuer`, `audience` and the role lists `krishiv.auth.oidc.roles.*`.
- **Client certificates:** `krishiv.auth.clientCertificates` (`optional` or `required`), `krishiv.auth.clientCa`,
  `krishiv.auth.mtls.principalFrom`, and the role lists `krishiv.auth.roles.*`.

A viewer sees metadata and its own queries and sessions, an operator also reads data and acts on others' running
work, an admin may also write, create, drop and change the cluster. A caller with no role is refused.

TLS: `krishiv.server.tls.certFile` and `keyFile` (the key file must not be readable by group or others) turn on TLS for
both ports; clients then connect with `sc://host:15002/;use_ssl=true` and `https://`. The server refuses to start
when the settings do not fit: a non-loopback address without an authentication method, a token or OIDC on a
non-loopback address without TLS, or client certificates without TLS. On Unix, `SIGHUP` reloads the certificate and
key; new connections get the new ones, open connections keep theirs, and a file that fails to load leaves the old
pair serving.

Every refused request is counted on `/metrics`: `krishiv_auth_failed_total` (unauthenticated or throttled, by port and
method) and `krishiv_auth_denied_total` (permission denied, by port and privilege). The `krishiv::auth` and
`krishiv::audit` log records describe each refusal, but logs are rate limited (`KRISHIV_LOG_RATE`) and best effort:
they are not an audit trail. Use the counters to see how many refusals happened.

## Distributed

A scheduler and workers run one query across machines. It is in development; the mode is not available yet.
