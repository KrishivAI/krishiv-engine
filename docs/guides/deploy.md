---
description: "The three deployment modes: embedded, single-node and distributed."
---

# Deployment

| Mode | Status in 0.2 |
|---|---|
| Embedded: `pip install krishiv`, the engine runs inside your Python process | Runs today |
| Single-node: one `krishiv server` process that clients reach over Spark Connect | In development |
| Distributed: a scheduler and workers | In development |

## Embedded

Nothing to deploy: import the package. Krishiv 0.2 is on PyPI (`pip install krishiv`). Supported CPUs are
x86-64-v3 (Intel Haswell or AMD Zen and newer) and Armv8.2 (AWS Graviton2 or newer, Apple silicon). On an older x86-64
CPU install `krishiv-compat` instead (the same package, imports as `krishiv`; install one of the two, not both). An
unsupported CPU is detected at import time and reported by name instead of crashing.

Memory use is bounded by `krishiv.*` settings (see [configuration](../reference/config/index.md)); operators spill to
disk when the limit is reached.

### The memory limit in a container

`krishiv.memory.limit` is the limit of the process's memory pool. Left unset, it is 80% of the effective memory limit:
the tightest cgroup limit that applies to the process (a Docker `--memory`, a Kubernetes memory limit, a systemd
`MemoryMax`), else the machine's physical memory, or 2 GiB where neither can be read; when both a cgroup limit and the
physical memory are known, the smaller of the two. A 4 GiB container on a 62 GB host therefore gets a pool of about
3.2 GiB, not 50 GB. Set `krishiv.memory.limit` (a startup setting) to use another value.

`SET krishiv.memory.limit` answers `<undefined>` while the key is unset, as `SET` does for any unset Spark key; it does
not print the derived value. `EXPLAIN ANALYZE` prints the limit the query ran under (for example `limit 3.2 GiB`).

## Single-node

The 0.2.0 command line has no `server` subcommand: it lists only `sql` and `debug-bundle`. In builds after 0.2.0,
`krishiv server` (a binary built with the `server` feature) starts one process that serves Spark Connect, by default on
`127.0.0.1` port `15002` (`--connect-port`, `krishiv.connect.port`), and the admin API on port `8080` (`--port`).
Settings are given as `--conf KEY=VALUE`. See [configuration](../reference/config/index.md) and
[known differences](../concepts/known-differences.md#krishiv-server-spark-connect).

```bash
krishiv server --port 8080 --connect-port 15002
curl -s http://127.0.0.1:8080/healthz     # 200
curl -s http://127.0.0.1:8080/readyz      # 200
```

`/metrics` answers `401` without the admin token. For local use on a loopback address,
`--conf krishiv.server.metrics.public=true` lets it answer without one (aggregate counters only), and
`krishiv.metrics.enabled=true` folds each query's metrics into it. The web console needs a binary built with the `ui`
feature and `krishiv.ui.enabled` (default `true`); a `server`-only build has no console page.

### Authentication, roles and TLS

These settings belong to the single-node server. The Spark Connect listener and its settings are not part of 0.2.

Without authentication settings the server binds loopback only, and the admin port (`/metrics`, the web UI) prints a
random token at start. To serve another address, configure at least one method with `--conf` (these settings are
read by the server itself and never become session configuration):

- **Tokens:** `krishiv.auth.tokens.<name>.secret` (an `env:` or `file:` reference; a file readable by group or others
  is refused), `.principal`, `.role` (`viewer`, `operator` or `admin`). Spark's `spark.connect.authenticate.token` is
  one admin token.
- **OIDC:** `krishiv.auth.oidc.issuer`, `audience` and the role lists `krishiv.auth.oidc.roles.*`. No released build
  enables the `oidc` feature, so a build refuses these settings at start-up until it does.
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
