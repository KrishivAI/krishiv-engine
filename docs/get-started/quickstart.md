---
description: "Install the Python package and run your first SQL and DataFrame query."
---

# Quickstart

<!-- The code blocks equal the Quickstart section of the repository README (a test checks it); CI runs the README's. -->

Krishiv 0.2 is on PyPI (you can also build the wheel from the repository with `maturin build`):

```bash
pip install krishiv
```

The command-line tool (the `krishiv` binary) installs from source with Rust 1.97 or newer: `cargo install krishiv-cli --features server,ui` (without the features the binary has no `krishiv server`). Prebuilt `krishiv` CLI binaries are not available yet.

Krishiv needs Python 3.11 or newer (the floor of the Spark 4.3 Python client); the examples run on 3.11 to 3.14. `pyarrow` is installed with it (a required dependency since 0.2.1); `collect()` and `toArrow()` return Arrow data.

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

## From Rust

`cargo add krishiv futures` (Rust 1.97 or newer). Results are Arrow record batches; actions are `async`, so any
runtime, or none with `futures::executor::block_on`, can await them.

```rust
use std::sync::Arc;
use krishiv::arrow::array::{Int32Array, RecordBatch};
use krishiv::arrow::datatypes::{DataType, Field, Schema};
use krishiv::functions::col;
use krishiv::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let session = Session::new()?;
let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![3, 1, 2]))])?;
session.register_batches("t", schema, vec![batch])?;

let df = session.table("t").order_by([col("a").desc()]).limit(2);
let rows: Vec<RecordBatch> = futures::executor::block_on(df.collect())?;
Ok(())
}
```

To print a `TIMESTAMP` column with `arrow-cast`'s `array_value_to_string`, enable the `chrono-tz` feature of
`arrow-array` in your own `Cargo.toml` (`cargo add arrow-array --features chrono-tz`, at the version `krishiv` uses):
results carry the zone `UTC`, and Arrow cannot format a named zone without it.

## Next

- [Notebooks and sample data](../guides/notebooks-and-datasets.md): three notebooks over generated order data.
- [Migrate a PySpark job](../guides/pyspark-migration.md): what differs from Spark and the workarounds.
