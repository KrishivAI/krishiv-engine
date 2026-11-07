# Krishiv documentation

Krishiv is a Rust-native, Apache Spark-compatible compute engine: batch SQL, DataFrames, streaming and incrementally
maintained views on one vectorized, Arrow-native engine, with no JVM. Keep the PySpark and Spark SQL you already
write.

> **Alpha (`0.2.0-alpha`).** The embedded core runs: in-process batch SQL and DataFrames from Python. There is no
> stable API yet. Streaming, incremental views, Spark Connect and distributed execution are being built.

## Where to go

- [Quickstart](quickstart.md): install and run your first query.
- [SQL](sql.md): the SQL dialect and the [function reference](reference/sql/index.md).
- [Python](python.md): the `krishiv` package and the [API reference](reference/python/index.md).
- [Configuration](configuration.md): every `krishiv.*` and Spark setting.
- [Deployment](deployment.md): embedded today; what comes next.
- [Compatibility](compatibility.md): what is supported against Spark, and the [known differences](known-differences.md).
- [Reference](reference/index.md): SQL functions, Python API, Rust API, metrics and query profiles.
