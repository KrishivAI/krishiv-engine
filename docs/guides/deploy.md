---
description: The three deployment modes: embedded, single-node and distributed.
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

One process, started with `krishiv server`, runs the engine and serves Spark Connect; clients connect to it with a
Spark Connect client such as PySpark. It listens on `127.0.0.1` port `15002` by default (`krishiv.connect.port`, see
[configuration](../reference/config/index.md)). A non-loopback address needs authentication, and the server refuses to
start on one until it is configured. Where it differs from Spark see
[known differences](../concepts/known-differences.md#krishiv-server-spark-connect).

## Distributed

A scheduler and workers run one query across machines. It is in development; the mode is not available yet.
