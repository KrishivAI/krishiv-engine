# SQL

Krishiv runs Spark SQL. Semantics, results and error classes follow Apache Spark; the differences that remain are
listed in [known differences](../../concepts/known-differences.md) and in the [compatibility matrix](../../concepts/compatibility.md).

Run SQL from Python with `spark.sql("...")`, or mix it with DataFrames: register a DataFrame with
`createOrReplaceTempView` and query it by name (see the [quickstart](../../get-started/quickstart.md)).

- Every built-in function and operator has a page in the [SQL function reference](functions/index.md): Spark's
  description and examples, and whether Krishiv implements it.
- `EXPLAIN` prints the physical plan; `EXPLAIN ANALYZE` runs the query and reports each operator's rows, time, peak
  memory and spill bytes (see [EXPLAIN ANALYZE](../explain-analyze.md) and the
  [query profile](../query-profile-v1.schema.json)).
- `EXPLAIN FORMAT JSON <statement>` returns one row with one column, `plan STRING`, holding the statement's analysis
  as a [`krishiv.analyze.v1`](../analyze/krishiv.analyze.v1.schema.json) document: its class, the securables it
  needs, its output schema and column lineage. It never runs the statement: `EXPLAIN FORMAT JSON INSERT ...` writes
  nothing, `EXPLAIN FORMAT JSON DROP TABLE t` drops nothing and lists the drop target. (The one exception, once
  Python UDTFs are supported: a UDTF's static `analyze` method, which Spark also runs while planning.) When the
  statement's reads are known, they are checked as a run checks them, so a denied read fails with the same error;
  write and drop targets are listed, not checked. For statements whose reads are not fully known yet (`UPDATE` and
  `DELETE` predicates, a multi-table `INSERT` with a shared `FROM`, `SET VARIABLE`) the document says
  `"partial": true` and lists only what is certain. The outer statement is itself a read.
  `FORMAT JSON` comes right after `EXPLAIN`; other formats are a syntax error, and `EXPLAIN FORMAT JSON PHYSICAL` is
  reserved (`KRISHIV_UNSUPPORTED_FEATURE`). It is a Krishiv extension (see
  [known differences](../../concepts/known-differences.md)).
- Behaviour that depends on the Spark release (3.5, 4.0, 4.1, 4.2) is keyed by the session's Spark version.
- Settings such as ANSI mode are in the [configuration reference](../config/index.md).
