# Krishiv documentation

Krishiv is a Rust-native, Apache Spark-compatible compute engine: batch SQL, DataFrames, streaming and incrementally
maintained views on one vectorized, Arrow-native engine, with no JVM. Keep the PySpark and Spark SQL you already
write.

> **Alpha (`0.2.0-alpha`).** The embedded core runs: in-process batch SQL and DataFrames from Python. There is no
> stable API yet. Streaming, incremental views, Spark Connect and distributed execution are being built.

## Where to go

The pages follow four kinds of need.

- **[Tutorials](tutorials/quickstart.md)** teach by doing: [install and run your first query](tutorials/quickstart.md).
- **[How-to guides](how-to/deploy.md)** solve one task: [deploy](how-to/deploy.md) and [find a stalled task with tokio-console](how-to/tokio-console.md).
- **[Reference](reference/index.md)** lists facts: the [SQL language](reference/sql-language.md) and
  [function reference](reference/sql/index.md), the [Python package](reference/python-package.md) and
  [API](reference/python/index.md), every [setting](reference/configuration.md), metrics and query profiles.
- **[Explanation](explanation/compatibility.md)** gives background: what is supported against Spark
  ([compatibility](explanation/compatibility.md)) and the [known differences](explanation/known-differences.md).
