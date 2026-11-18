# Adapted PySpark 4.2.0 UDF tests (design P2-T059 section 6.1 and 6.2)

- `test_udf.py`, `test_pandas_udf_scalar.py`, `test_arrow_python_udf.py`: apache/spark tag `v4.2.0`,
  `python/pyspark/sql/tests/` (Apache-2.0, see the repository `NOTICE`), with the imports pointed at `krishiv`
  and `_support.py`. `adapt.py` regenerates them from a checkout of the tag (`ADAPTATIONS` lists every edit of a
  test body: the scalar-iterator variants are dropped from the loops that cover both kinds, `Row` imports, and
  nested-column `createDataFrame` goes through a pyarrow table).
- `manifest.toml`: every test method, `included` (must pass) or `excluded` with a category and a reason.
  `conftest.py` skips the excluded ones; `test_manifest.py` checks the manifest is complete.
- `coercion/` and `test_coercion_goldens.py`: Spark's four coercion golden tables and their harnesses, compared
  cell for cell; `KNOWN_GAPS` lists the differing cells with a reason.
- `window_iter/` and `test_window_iter_goldens.py`: Krishiv-authored cases for window aggregates (eval types
  203, 253: seven window specs, bounded and unbounded frames, NULLs, empty frames) and scalar iterators (204, 251:
  re-batched output and the worker's row checks), built against `pyspark` or `krishiv` by `window_iter/cases.py`.
  `window_iter/goldens.json` holds PySpark 4.2.0's answers from a real local run (`window_iter/gen_goldens.py`,
  its docstring has the command); `KNOWN_GAPS` lists the cases the embedded facade does not match yet (all of
  them until PR-48); `crates/krishiv-udf-pool/tests/window_iter_goldens.rs` replays the same goldens on the worker pool.
- Run: `cd python && uv run --group dev --extra pandas pytest tests/pyspark_udf`.

## Goldens owed

Spark-observable Python UDF behaviour the pool runs with no PySpark 4.2.0 golden yet (the spark-semantics flow
makes them; each lands with or before the change that makes it reachable from a query):

- Python UDTFs (`crates/krishiv-udf-pool/tests/udtf.rs`, `python/tests/test_udf_udtf.py` check the port only),
  due with the planner half of P3-T014 PR-26c-3 part B: for `@udtf` (300), `@udtf(useArrow=True)` (301) and
  `@arrow_udtf` (302), the rows of `eval` and `terminate`; a `TABLE` argument with `PARTITION BY` and `ORDER BY`
  (a `terminate` and a fresh instance per partition) and `WITH SINGLE PARTITION`; each `UDTF_*` error of
  `read_udtf`; `analyze` (the result schema, `partitionBy`, `orderBy`, `select`, a constant argument of each type
  as `analyze` sees it, the `withSinglePartition` refusal without a table argument, and a failing `analyze`'s
  `TABLE_VALUED_FUNCTION_FAILED_TO_ANALYZE_IN_PYTHON` message).
- A `@udtf` (300) returning values of another type than the declared one, nested rows and `None` rows: Spark's
  JVM converts them (`EvaluatePython.makeFromJava`), Krishiv converts them to Arrow as for 301
  (`docs/concepts/known-differences.md`, "Python table functions"); match the golden or record the difference.
- Grouped-aggregate Python UDFs (202, 252) in SQL `GROUP BY`
  (`crates/krishiv-udf-pool/tests/window_iter_goldens.rs` `grouped_aggregate_udfs_in_sql_group_by` checks the
  results and the classes from the Spark source): the same queries in a PySpark 4.2.0 run, including the
  `pythonudaf` name in `INVALID_SQL_SYNTAX.FUNCTION_WITH_UNSUPPORTED_SYNTAX`.
