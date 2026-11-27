---
description: "Krishiv is a Rust-native, Spark-compatible engine for batch SQL and DataFrames, in process and with no JVM."
---

# Krishiv documentation

Krishiv is a Rust-native, Apache Spark-compatible compute engine for batch SQL and DataFrames. It runs in your
process, on one vectorized, Arrow-native engine, with no JVM. Keep the PySpark and Spark SQL you already write.

> **Krishiv 0.3 adds a server.** In-process batch SQL and DataFrames from Python and Rust, plus `krishiv server`: Spark
> Connect, Arrow Flight SQL and a web console from one process. Streaming, incremental views and distributed execution
> are not in this release. The APIs are not stable yet.

## Install

Krishiv is on PyPI: `pip install krishiv`. The [Quickstart](get-started/quickstart.md) has the steps.
You can also build the wheel from the repository (`maturin build` in `python/`).

## Where to start

- **New here:** [Get started](get-started/index.md) installs Krishiv and runs a first query.
- **Coming from Spark:** [Guides](guides/index.md), beginning with
  [Migrate a PySpark job](guides/pyspark-migration.md).
- **Looking something up:** [Reference](reference/index.md): the [SQL language](reference/sql/index.md) and
  [functions](reference/sql/functions/index.md), the [Python package](reference/python-package.md) and
  [API](reference/python/index.md), every [setting](reference/config/index.md), metrics and query profiles.
- **Deciding what to rely on:** [Concepts](concepts/index.md): what is [supported](concepts/compatibility.md) against
  Spark and the [known differences](concepts/known-differences.md).
- **What changed:** [Releases](releases/index.md).
