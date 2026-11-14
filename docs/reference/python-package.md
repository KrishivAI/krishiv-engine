---
description: "The krishiv Python package, which mirrors pyspark.sql and needs CPython 3.11 or newer."
---

# Python

The `krishiv` package runs in your process and mirrors `pyspark.sql`: the same import paths, signatures and error
classes (`krishiv.errors`). Needs CPython 3.11 or newer.

```python
import krishiv
from krishiv.sql import functions as F

spark = krishiv.session()  # SparkSession.builder.getOrCreate()
df = spark.sql("SELECT 1 AS a").withColumn("b", F.col("a") + 1)
df.show()
table = df.toArrow()  # a pyarrow.Table
```

- Actions (`collect`, `show`, `toArrow`, `toPandas`, `count`) run the query in-process with the GIL released.
- A DataFrame implements the Arrow PyCapsule interface, so `pa.table(df)`, `polars.DataFrame(df)` and
  `duckdb.from_arrow(df)` read the result without copying.
- pyarrow is a required dependency (since 0.2.1; `krishiv[pyarrow]` still works as an alias); `toPandas` also needs pandas (`krishiv[pandas]`).
- `df.profile()` (or `krishiv.ui(df)`) returns a viewer of the query's profile; in a notebook it renders itself.
- Test your PySpark code without a JVM: `krishiv.testing` has `assertDataFrameEqual`, `assertSchemaEqual` and a
  pytest fixture (`pytest_plugins = ["krishiv.testing.pytest_plugin"]`, then use the `krishiv_session` fixture).

The full [Python API reference](python/index.md) is generated from the package's docstrings.
