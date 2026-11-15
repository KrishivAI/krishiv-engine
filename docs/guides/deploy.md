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

Nothing to deploy: import the package. Krishiv 0.2 is on PyPI (`pip install "krishiv[pyarrow]"`). Supported CPUs are
x86-64-v3 (Intel Haswell or AMD Zen and newer) and Armv8.2 (AWS Graviton2 or newer, Apple silicon). On an older x86-64
CPU install `krishiv-compat` instead (the same package, imports as `krishiv`; install one of the two, not both). An
unsupported CPU is detected at import time and reported by name instead of crashing.

Memory use is bounded by `krishiv.*` settings (see [configuration](../reference/config/index.md)); operators spill to
disk when the limit is reached.

## Single-node

Not part of 0.2. This mode is in development. When it ships, one process started with `krishiv server` will run the
engine and serve Spark Connect, and PySpark will connect to it on `127.0.0.1` port `15002` by default
(`krishiv.connect.port`, see [configuration](../reference/config/index.md)). Until authentication is available, keep
it on loopback and reach it through a tunnel or proxy. The behaviour it will have is described in
[known differences](../concepts/known-differences.md#krishiv-server-spark-connect).

## Distributed

A scheduler and workers run one query across machines. It is in development; the mode is not available yet.
