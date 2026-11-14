# Krishiv

**Spark alternative in Rust: PySpark and Spark SQL without the JVM.**

Krishiv runs Spark SQL and a PySpark-style DataFrame API inside your Python process. No cluster, no JVM, no
server to start. Execution is vectorized on Apache Arrow, and results come back as Arrow tables you can hand to
pandas, Polars or DuckDB without copying.

## Install

```bash
pip install krishiv
```

Needs Python 3.11 or newer. `pyarrow` is installed with it. For `toPandas()` add the extra:
`pip install "krishiv[pandas]"`.

## 30-second example

```python
import krishiv
from krishiv.sql import functions as F

spark = krishiv.session()  # a PySpark-style SparkSession, in process

spark.sql("SELECT 1 AS answer").show()

df = spark.sql("SELECT * FROM VALUES ('east', 10.0), ('west', 20.0), ('east', 5.5) AS t(region, amount)")
df.groupBy("region").agg(F.sum("amount").alias("revenue")).orderBy("region").show()

table = df.toArrow()  # a pyarrow.Table; also pa.table(df), polars.DataFrame(df), duckdb.from_arrow(df)
```

## What works in 0.2

- Batch Spark SQL and DataFrames, in process, over Arrow. `krishiv.sql` mirrors `pyspark.sql`
  (import paths and signatures); errors in `krishiv.errors` carry Spark's error classes.
- Reading and writing Parquet files; `EXPLAIN` and `EXPLAIN ANALYZE` with per-operator rows, time and memory.
- Spark's function library, with Spark's results (hashing, casts, null handling and error behaviour follow Spark).
- `krishiv.testing`: `assertDataFrameEqual` and a pytest fixture, to run your PySpark tests without a JVM.

Streaming, incremental views, Spark Connect and distributed execution are still being built. This is an alpha:
there is no stable API yet.

## Speed

Benchmark results against Spark 4.2.0 on TPC-H-derived queries (not official TPC results) are published with their
method, queries and raw numbers at <https://krishiv.ai/benchmarks>.

## Supported CPUs

The default wheels need an x86-64-v3 CPU (Intel Haswell, 2013, or AMD Zen and newer) or, on arm64, Armv8.2 (AWS
Graviton2, Apple silicon). On an older x86-64 CPU install `krishiv-compat` instead: the same package and the
same `import krishiv`, built for the plain x86-64 baseline (Linux only). Install one of the two, not both.

## Links

- Website: <https://krishiv.ai>
- Documentation: <https://krishiv.ai/docs>
- Benchmarks: <https://krishiv.ai/benchmarks>
- Source and issues: <https://github.com/KrishivAI/krishiv-engine>
- Releases and changelog: <https://github.com/KrishivAI/krishiv-engine/releases>

Krishiv is licensed under Apache-2.0. Apache Spark is a trademark of the Apache Software Foundation; Krishiv is
not affiliated with it.
