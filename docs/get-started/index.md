---
description: "Install Krishiv, run your first SQL and DataFrame query, and decide where to go next."
---

# Get started

This section is for someone who has not run Krishiv yet. The steps below install the Python package, start a session,
run a SQL query and a DataFrame query, and read the result back as Arrow data. They take a few minutes.

<!-- The code blocks equal the Quickstart section of the repository README (a test checks it); CI runs the README's. -->

Install the package with pyarrow for the Arrow results:

```bash
pip install "krishiv[pyarrow]"
```

Krishiv needs Python 3.11 or newer (the floor of the Spark 4.3 Python client); the examples run on 3.11 to 3.14. The `pyarrow` extra is needed for `collect()` and `toArrow()`, which return Arrow data.

SQL and DataFrames:

```python
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import krishiv
from krishiv.sql import functions as F

spark = krishiv.session()  # a PySpark-style SparkSession, in process

# SQL.
spark.sql("SELECT 1 AS answer").show()

# DataFrames over a Parquet file; the same file is also a temp view for SQL.
path = str(Path(tempfile.mkdtemp()) / "orders.parquet")
pq.write_table(
    pa.table(
        {
            "region": ["east", "west", "east", "north", "west"],
            "amount": [10.0, 20.0, 5.5, 7.25, 30.0],
        }
    ),
    path,
)
orders = spark.read.parquet(path)
orders.createOrReplaceTempView("orders")

by_region = (
    orders.filter(F.col("amount") > 6)
    .groupBy("region")
    .agg(F.sum("amount").alias("revenue"))
    .orderBy("region")
)
by_region.show()
print(by_region.toArrow())  # a pyarrow.Table; also pa.table(by_region), polars.DataFrame(by_region)

same = spark.sql("SELECT region, sum(amount) AS revenue FROM orders WHERE amount > 6 GROUP BY region ORDER BY region")
assert same.collect() == by_region.collect()
```

`EXPLAIN` and `EXPLAIN ANALYZE` (per-operator rows, time, peak memory and spill bytes):

```python
by_region.explain()  # the physical plan
by_region.explain(analyze=True)  # runs the query and prints each operator's metrics

plan = spark.sql("EXPLAIN ANALYZE SELECT region, sum(amount) FROM orders GROUP BY region").collect()[0][0]
assert plan.startswith("== Physical Plan (analyzed) ==")
```

## Next

- Moving an existing Spark job: [migrate a PySpark job](../guides/pyspark-migration.md), and scan your code first with
  [check-compat](../guides/check-compat.md).
- Trying it on sample data: the [notebooks and datasets](../guides/notebooks-and-datasets.md).
- Running it somewhere other than your Python process: [deployment](../guides/deploy.md).
