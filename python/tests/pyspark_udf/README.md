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
  its docstring has the command); `KNOWN_GAPS` lists the cases Krishiv does not match yet.
- Run: `cd python && uv run --group dev --extra pandas pytest tests/pyspark_udf`.
