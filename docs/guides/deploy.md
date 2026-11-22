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
`krishiv.metrics.enabled=true` folds each query's metrics into it. The [web console](#the-web-console) is served on
the admin port.

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

## The web console

One console, the same pages, runs in three places. It comes with builds after 0.2.0: the released `krishiv` binary
and the Python wheel both carry it (a build from source needs the `ui` feature, and `server` for `krishiv server`).

| | In your process | On a server | Saved profiles |
|---|---|---|---|
| Start it with | `krishiv.ui()` (Python), `krishiv::ui(&session)` (Rust) | `krishiv server` | `krishiv ui --profiles <dir>` |
| Shows | the one session that started it | every session of every client | the queries a history folder holds |
| Listens on | `127.0.0.1`, a free port | the admin port (`--port`, default `8080`) | `127.0.0.1`, a free port (`--port`) |
| Who may open it | whoever has the URL with its token | a token, by role | whoever has the URL with its token |
| Pages | all but Sessions, Workload and Advisor | all; Environment for an administrator only | History and a query's page |

**In your process.** `krishiv.ui()` starts the console of the active session, prints its URL and returns a handle
(`url`, `port`, `token`, `stop()`); `krishiv::ui(&session)` returns the same handle in Rust and prints nothing, so
show `console.url()` yourself. The token in the URL is generated at each start: treat the URL as a password for the
session. The SQL page runs statements in your session, so it sees your temporary views; you can cancel your own
queries; Environment lists the session's settings and Resources the memory and caches of the process. The console
stops when the session stops.

```python
import krishiv

spark = krishiv.session()
console = krishiv.ui()   # prints: Krishiv console: http://127.0.0.1:<port>/?token=<token>
spark.stop()             # the console stops with the session
```

**On a server.** `krishiv server` serves the console at the admin port's address unless `krishiv.ui.enabled` is
`false`. Without authentication settings it prints a URL with a start-up token; with them, a token's role decides
what its holder sees (the table in [single-node server security](single-node-server-security.md)): an operator
sees and cancels every query and runs SQL as the signed-in user in a session of its own, and only an administrator
opens Environment, where secrets show as `***`. The History page filters by state, principal, client (any part of
the name a client reports, such as `python`), tag (`team` or `team=risk`) and session; the Queries page has one
filter box over the running queries (principal, client, tag, text or id). A client chooses its operation ids, so
two sessions can run queries under one id: a query's link and its **Cancel** carry the row's session
(`/queries/<id>?session=<session id>`), and Cancel ends that session's query only.

The Sessions page lists the sessions of every client: who opened each, with which client and over which protocol,
its current catalog and namespace, when it was last active and how many queries it runs and has started (the
count opens History on that session). Sessions that ended (closed, expired, killed) are listed on request.
**End session** asks once, then ends the session: its running queries fail with `KRISHIV_QUERY_KILLED` and the
client's next call is refused with `INVALID_HANDLE.SESSION_CLOSED`, so the client must open a new session. Ending
a session needs the operator role, like a cancel, and each writes one `krishiv::audit` record. The session the
console's own editor runs in is listed and cannot be ended from the page. The Workload page lists
the resource groups of `krishiv.workload.resourceGroups.file`, each under its parent, with what runs and waits
against its limits, its memory in use and its scan quota; a server started without that file admits every query
as it arrives, and the page says so.

The Advisor page lists the partition advice of `system.partition_advice`: for each table the query history
covers, the column its finished queries filter on by equality that would let them skip the most data, with how
many of the table's queries filter on it, how many partitions it would make and the share of bytes it is expected
to save; other candidate columns follow in the same row. `EXPLAIN` prints the same advice for the tables a query
reads. The advice comes from the stored history, so a server needs `krishiv.history.path` to have any, and the
page reads it when opened and on **Read the history again**, not on a timer. Nothing is applied: to follow a
suggestion, write the table again partitioned by the column.

A server's history is kept in memory unless `krishiv.history.path` names a folder. With it, each finished query's
profile is stored there, and a restarted server lists the newest 1,000 stored queries again. A query from before
the restart has no tags and no client, so the tag and client filters leave it out.

```bash
krishiv server --port 8080 --conf krishiv.history.path=/var/lib/krishiv/history
```

**Saved profiles.** `krishiv ui --profiles /var/lib/krishiv/history` opens the same folder with no engine running:
the History page and each query's page, nothing live and nothing to cancel. Use it to read the profiles of a
server that is stopped, or a folder copied from another machine.

Not in the console yet: Resources does not list UDF worker pools.

## Distributed

A scheduler and workers run one query across machines. It is in development; the mode is not available yet.
