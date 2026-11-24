---
description: "Read and write your own formats from Python with the Python Data Source API: register a DataSource, read it with spark.read.format, and what the batch API does not cover yet."
---

# Python data sources

The Python Data Source API lets you read from, and write to, a system Krishiv has no connector for, in plain Python.
It has the shape of `pyspark.sql.datasource` in Spark 4.3. In 0.3 the batch API is available; streaming sources and
sinks are not.

## Write a source

Subclass `DataSource` and give it a schema and a reader. A reader splits the work into partitions and reads each one:

```python
from krishiv.sql import SparkSession
from krishiv.sql.datasource import DataSource, DataSourceReader, InputPartition


class Fixed(DataSource):
    """Three partitions of two rows each: (partition, i)."""

    @classmethod
    def name(cls):
        return "fixed"

    def schema(self):
        return "p INT, i INT"

    def reader(self, schema):
        class R(DataSourceReader):
            def partitions(self):
                return [InputPartition(k) for k in range(3)]

            def read(self, partition):
                for i in range(2):
                    yield (partition.value, i)

        return R()
```

## Register and read it

```python
spark = SparkSession.builder.create()
spark.dataSource.register(Fixed)

df = spark.read.format("fixed").load()
df.filter("i = 1").count()          # 3
df.createOrReplaceTempView("v")
spark.sql("SELECT sum(i) FROM v").collect()[0][0]   # 3
```

The source's name is case-insensitive (`format("FIXED")` works). Without a `name()` method the class name is the name.
Options set with `.option(k, v)` and the path given to `load` reach `self.options`, a dictionary whose keys are
case-insensitive (`options["path"]` and `options["Path"]` are the same key). A schema given with `.schema(...)` wins
over the source's.

`read` may yield tuples, `Row` objects, lists or `pyarrow.RecordBatch` objects. Tuples, `Row` objects and lists are
converted to the declared types. A `RecordBatch` is not cast: its Arrow schema must match the declared one exactly, or the
read fails with `DATA_SOURCE_RETURN_SCHEMA_MISMATCH`.

## Write

Give the source a `writer(schema, overwrite)` method that returns a `DataSourceWriter` (or `DataSourceArrowWriter`):

```python
from krishiv.sql.datasource import DataSourceWriter, WriterCommitMessage


class Sink(DataSource):
    def schema(self):
        return "a INT"

    def writer(self, schema, overwrite):
        class W(DataSourceWriter):
            def write(self, iterator):
                rows = [tuple(r) for r in iterator]
                return WriterCommitMessage()

            def commit(self, messages):
                pass

            def abort(self, messages):
                pass

        return W()


spark.dataSource.register(Sink)
df.write.format("sink").mode("append").save()
```

The write runs as one task over every row of the DataFrame and then calls `commit`; if the task fails it calls `abort`
instead, with `None` for the failed task's message. `mode("overwrite")` passes `overwrite=True` to `writer`.

## Limits

- Batch only. `streamReader` and `streamWriter` raise `NOT_IMPLEMENTED`; the streaming classes arrive after 0.3.
- Partitions run one after the other in the calling process, not in parallel.
- The rows of a read are held in memory while the scan runs. The cap is a fixed 1 GiB; a read that holds
  more fails.
- Filter pushdown is off, as in Spark 4.3 (`spark.sql.python.filterPushdown.enabled`). A reader that overrides
  `pushFilters` fails with `DATA_SOURCE_PUSHDOWN_DISABLED`; with the key on, `pushFilters` is still not called and
  Krishiv applies every filter itself.
- One `DataSource` object is created per `load` and reused by every action on that DataFrame. Spark pickles it once
  `load` has its schema (after `schema()` when the schema is inferred; with `.schema(...)` neither Spark nor Krishiv
  calls `schema()`, so state set there does not exist) and unpickles that copy for every planning: state set in
  `__init__` (or in `schema()` when it ran) reaches the reader, and so does state the reader gets while it is planned
  (its constructor's arguments, for example), but state set during one planning or action does not reach the next.
  Do not rely on state carried from one action to the next; Krishiv will behave the same once readers run on workers.

The full API is in the [Python reference](../reference/python/krishiv.sql.datasource.md).
