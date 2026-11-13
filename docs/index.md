# Krishiv documentation

Krishiv is a Rust-native, Apache Spark-compatible compute engine: batch SQL, DataFrames, streaming and incrementally
maintained views on one vectorized, Arrow-native engine, with no JVM. Keep the PySpark and Spark SQL you already
write.

> **Alpha (`0.2.0-alpha`).** The embedded core runs: in-process batch SQL and DataFrames from Python. There is no
> stable API yet. Streaming, incremental views, Spark Connect and distributed execution are being built.

## Where to go

The pages follow four kinds of need.

- **[Get started](get-started/quickstart.md)** teach by doing: [install and run your first query](get-started/quickstart.md).
- **[Guides](guides/deploy.md)** solve one task: [migrate a PySpark job](guides/pyspark-migration.md), [deploy](guides/deploy.md) and [find a stalled task with tokio-console](guides/tokio-console.md).
- **[Reference](reference/index.md)** lists facts: the [SQL language](reference/sql/index.md) and
  [function reference](reference/sql/functions/index.md), the [Python package](reference/python-package.md) and
  [API](reference/python/index.md), every [setting](reference/config/index.md), metrics and query profiles.
- **[Concepts](concepts/compatibility.md)** gives background: what is supported against Spark
  ([compatibility](concepts/compatibility.md)) and the [known differences](concepts/known-differences.md).
