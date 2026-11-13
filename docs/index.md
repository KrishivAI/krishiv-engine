---
description: Krishiv is a Rust-native, Spark-compatible engine for batch SQL, DataFrames and streaming, with no JVM.
---

# Krishiv documentation

Krishiv is a Rust-native, Apache Spark-compatible compute engine: batch SQL, DataFrames, streaming and incrementally
maintained views on one vectorized, Arrow-native engine, with no JVM. Keep the PySpark and Spark SQL you already
write.

> **Alpha (`0.2.0-alpha`).** The embedded core runs: in-process batch SQL and DataFrames from Python. There is no
> stable API yet. Streaming, incremental views, Spark Connect and distributed execution are being built.

## Install

Nothing is on PyPI yet, so build the wheel from the repository (`maturin build` in `python/`) and install it. Once
released, `pip install "krishiv[pyarrow]"` is all it takes; the [Quickstart](get-started/quickstart.md) has the steps.

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
