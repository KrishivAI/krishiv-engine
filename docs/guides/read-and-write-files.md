---
description: "Read a CSV or Parquet file, transform it, and write the result as Parquet or CSV, with the commit directory explained."
---

# Read and write files

This guide reads a file of your own, transforms it and writes the result. It uses the Python package, and the
examples run on Krishiv 0.2.0 from PyPI (`pip install "krishiv[pyarrow]"`) with local files.

## Read a CSV file

Give the reader the path and the two options most files need: `header` for a first row of names, and `inferSchema`
to let Krishiv choose column types from the data.

```python
import krishiv

spark = krishiv.session()
sales = spark.read.csv("sales.csv", header=True, inferSchema=True)
sales.printSchema()
print(sales.count())
```

Without `inferSchema`, every column is read as a string. The full set of CSV options is in the
[Python API reference](../reference/python/krishiv.sql.readwriter.md).

## Transform the data

The DataFrame calls are the same as in PySpark:

```python
from krishiv.sql import functions as F

revenue = (
    sales.filter(F.col("amount") > 100)
    .groupBy("region")
    .agg(F.sum("amount").alias("revenue"))
    .orderBy("region")
)
revenue.show()
```

## Write the result

Choose a mode and a format, then give the output directory. `overwrite` replaces an existing directory; without it,
a write to an existing directory fails with `PATH_ALREADY_EXISTS`.

```python
revenue.write.mode("overwrite").parquet("revenue_parquet")

revenue.write.mode("overwrite").option("header", True).csv("revenue_csv")
```

Read the Parquet output back with `spark.read.parquet("revenue_parquet")`.

### The `_krishiv_log` directory

A write creates a `_krishiv_log/` directory next to the data files. It is Krishiv's commit log: it records each write,
and Krishiv readers of the directory see only the files of committed writes.

If you need a directory that other tools read as a plain Spark output, set the commit protocol to `NONE` before the
write. The output then has `_SUCCESS` and the data files, and no `_krishiv_log/`:

```python
spark.conf.set("krishiv.sql.files.commitProtocol", "NONE")
revenue.write.mode("overwrite").parquet("revenue_plain")
```

The setting applies to the session. A directory that already has a `_krishiv_log/` keeps using it.

## Limits in 0.2.0

- Local files work. The PyPI package of 0.2.0 does not include the object-store readers, so `s3://`, `gs://`, `az://`
  and `https://` paths fail there (see the 0.2.0 release notes).
- This guide covers the options it shows. For the rest, see the [Python API reference](../reference/python/krishiv.sql.readwriter.md).
