---
description: "Write a Python table function (UDTF), register it, and call it from SQL with FROM, TABLE arguments and LATERAL."
---

# Python table functions (UDTFs)

A Python table function returns rows, zero or more for each call, where a scalar UDF returns one value. You write it as
in PySpark (`@udtf`, `@udtf(useArrow=True)` or `@arrow_udtf`), register it with `spark.udtf.register`, and call it from
SQL. A Krishiv server runs it in a worker of the Python UDF pool, in the [sandbox](python-udf-sandbox.md). Its answers
match PySpark 4.2.0's on the cases in `python/tests/pyspark_udf/udtf/goldens.json`, apart from the
[known differences](../concepts/known-differences.md#python-table-functions-udtfs).

## Write and register one

```python
from pyspark.sql.functions import udtf

@udtf(returnType="word: string, length: int")
class SplitWords:
    def eval(self, text: str):
        for word in (text or "").split():
            yield word, len(word)

spark.udtf.register("split_words", SplitWords)
```

`eval` runs for each input row and yields tuples that match `returnType`. A `terminate` method, when the class has one,
runs once at the end of each partition of the input and can yield more rows.

The three forms differ in how rows move:

| Form | Eval type | Rows reach `eval` as |
|---|---|---|
| `@udtf` | 300 | Python values, one call per row |
| `@udtf(useArrow=True)` | 301 | Python values; the rows travel as Arrow batches |
| `@arrow_udtf` | 302 | a `pyarrow.RecordBatch` or table per call |

## Call it from SQL

In the `FROM` clause, with constant arguments:

```sql
SELECT * FROM split_words('a quick test');
```

With a row of a table at a time, using `LATERAL`:

```sql
SELECT d.id, w.word, w.length
FROM docs AS d, LATERAL split_words(d.text) AS w;
```

Each result row leads with the columns of the input row it came from. An inner `JOIN LATERAL ... ON condition` works too.
`LEFT JOIN LATERAL` without a condition returns the rows of the inner join; with `ON` it is refused
with `KRISHIV_UNSUPPORTED_FEATURE` for now, and `USING` and `NATURAL` are refused earlier, by the SQL validator. An `@arrow_udtf` cannot be the target of a `LATERAL` call.

## Hand a whole table to the function

A `TABLE(...)` argument passes a relation, so `eval` sees each of its rows. Say how the rows are split and ordered:

```sql
SELECT * FROM summarize(TABLE(events) PARTITION BY user_id ORDER BY ts);
SELECT * FROM summarize(TABLE(events) WITH SINGLE PARTITION);
```

Each partition gets a new instance of the class and its own `terminate` call, so `terminate` is where per-group totals
are emitted. A function takes one `TABLE` argument. A call that passes two, or that passes a table to a function that
does not take one, fails at analysis.

## Decide the schema when the query is planned: `analyze`

A function whose output columns depend on its arguments declares a static `analyze` method instead of a fixed
`returnType`. It runs at analysis time with the constant arguments and returns an `AnalyzeResult`:

```python
from pyspark.sql.functions import udtf
from pyspark.sql.udtf import AnalyzeArgument, AnalyzeResult
from pyspark.sql.types import StructType, StringType

@udtf
class Echo:
    @staticmethod
    def analyze(arg: AnalyzeArgument) -> AnalyzeResult:
        return AnalyzeResult(StructType().add(arg.value, StringType()))

    def eval(self, arg):
        yield (str(arg),)
```

`analyze` can also ask for a partitioning (`withSinglePartition`, `partitionBy`, `orderBy`) and for a `select` of the
input columns. Two limits to know: a constant that the analyzer does not fold, such as `array(1, 2)`, reaches `analyze`
as no constant (`value` is `None`), and the JSON response of `analyze`, which holds the pickled result as hex (about twice its size), may not exceed 64 MiB.
A pickle above roughly 32 MiB fails with `TABLE_VALUED_FUNCTION_FAILED_TO_ANALYZE_IN_PYTHON`.

## Where it runs

- **In a server.** Register through a Spark Connect client (`pyspark-client`, or [`krishiv.start_local()`](embedded-connect.md)).
  The function runs in the pool, so the server needs Python workers and the sandbox the
  [UDF sandbox guide](python-udf-sandbox.md) describes. [Per-tenant limits](python-udf-tenants.md) apply.
- **Result types.** In the `@udtf` forms a row value is converted to the declared column type as Spark's JVM converts it:
  an integer in a string column becomes its text, and a value that cannot be converted gives NULL. An `@arrow_udtf` result
  is cast to its declared schema when the cast is safe (`int32` for an `int64` column works); an incompatible or unsafe
  conversion fails with `PYTHON_EXCEPTION`.
- **Re-registering** a name replaces the function and runs a fresh analysis for the next query. A built-in table function
  such as `range` or `explode` keeps its name; a UDTF of the same name does not shadow it.

## Related

- [Python UDFs in the server: the sandbox](python-udf-sandbox.md)
- [Profile a Python UDF](python-udf-profiler.md)
- [Migrate a PySpark job](pyspark-migration.md)
