# SQL

Krishiv runs Spark SQL. Semantics, results and error classes follow Apache Spark; the differences that remain are
listed in [known differences](../explanation/known-differences.md) and in the [compatibility matrix](../explanation/compatibility.md).

Run SQL from Python with `spark.sql("...")`, or mix it with DataFrames: register a DataFrame with
`createOrReplaceTempView` and query it by name (see the [quickstart](../tutorials/quickstart.md)).

- Every built-in function and operator has a page in the [SQL function reference](sql/index.md): Spark's
  description and examples, and whether Krishiv implements it.
- `EXPLAIN` prints the physical plan; `EXPLAIN ANALYZE` runs the query and reports each operator's rows, time, peak
  memory and spill bytes (see [EXPLAIN ANALYZE](explain-analyze.md) and the
  [query profile](query-profile-v1.schema.json)).
- Behaviour that depends on the Spark release (3.5, 4.0, 4.1, 4.2) is keyed by the session's Spark version.
- Settings such as ANSI mode are in the [configuration reference](configuration.md).
