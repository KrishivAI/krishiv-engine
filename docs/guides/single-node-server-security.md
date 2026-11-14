---
description: "Securing a single-node Krishiv server: admin token roles, what the admin endpoints allow, path and object-store confinement, and credential rules."
---

# Securing a single-node server

This page covers a single-node `krishiv server`: one process that serves Spark Connect clients, the metrics
endpoint and the web console. It describes what the server checks today.

## Admin endpoints and token roles

The server's admin listener serves `/metrics`, the web console and its API. Those routes check a bearer token
against static tokens you configure:

```text
krishiv.auth.tokens.<name>.secret     the token value
krishiv.auth.tokens.<name>.principal  the user name the token acts as
krishiv.auth.tokens.<name>.role       viewer, operator or admin
```

Each role may do this:

| Route | viewer | operator | admin |
|---|---|---|---|
| `/metrics` (Prometheus scrape) | yes | yes | yes |
| Console pages that show queries | no | yes | yes |
| Cancel a query from the console | no | yes | yes |

Some routes stay open without a token: `/healthz` and `/readyz`.

Failures are reported as follows:

- A missing or wrong token gets `401` with a `WWW-Authenticate: Bearer` header. The token is never echoed back.
- A token that is allowed to reach the route but lacks the role gets `403`. The server records one audit entry with
  the principal, role, route and the action.
- After repeated failures, the client is throttled and gets `503`. The threshold is `krishiv.auth.failureThrottle.maxFailures`.

If you configure no tokens, the server prints a random start-up token, as before, and accepts only that token.

## Path confinement

By default, a server lets clients read and write any path the server process can reach. Once you configure
authentication, you can limit that to the places you name.

| Setting | What it allows |
|---|---|
| `krishiv.io.local.allowedRoots` | Local paths under the listed absolute directories, for every role, admin included. |
| `krishiv.io.objectStore.allowedPrefixes` | Object-store locations under the listed prefixes, for example `s3://bucket/prefix` or `abfss://container@account.dfs.core.windows.net/prefix`. |

Rules that apply:

- Anything outside these lists is refused for every role. Inside them, the token's role decides.
- Object-store prefixes match whole path segments. `s3://` and `s3a://` are the same store. An Azure prefix must name
  its account host, so a path that takes its account from the client's settings never matches.
- Glob characters, `%`, `?`, `#`, embedded credentials, and `.` or `..` segments are refused when the server starts.
- With `allowedRoots` set, a symbolic link that points outside every listed root is refused.
- `http://` and `https://` locations are refused until you list them in `allowedPrefixes`. Listing one is the way
  to enable it.

These settings are read at start-up.

## Credentials and store options

In server mode, the server's own ambient cloud credentials (for example an instance role) reach only the prefixes
in `allowedPrefixes`. Listing a prefix is the only opt-in.

Store options set by a client are split into two groups:

- **Inline credentials** (access keys, secrets, session tokens, SAS tokens, the Hugging Face token) may be
  passed by any client that can read the data.
- **Everything else** (endpoints, regions, accounts, proxies, key files) needs the admin role on the cluster. This
  check runs on reads, writes, `CREATE TABLE ... OPTIONS`, `SET` and `spark.conf.set`, before the value is used or stored.

S3 regions are validated, so a region cannot point the client at another host.

Embedded use, where the Python or Rust code runs in your own process, is not affected by these rules.

## What is not in this release

- **TLS and mutual TLS on the listeners.** The settings and the checks for them exist in the code, but the server
  build in this release does not enable them. Run the server behind a TLS-terminating proxy, or on a loopback
  address, until they ship.
- **OpenID Connect (OIDC) sign-in.** Not in the server build in this release. Use static tokens.
- **Connect-side tokens.** The Spark Connect listener does not yet check the admin tokens described above. Keep it
  on a loopback address or behind your own access controls.

Check the release notes for when each of these arrives.
