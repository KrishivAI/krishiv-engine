# Python UDTF goldens (PySpark 4.2.0)

`goldens.json` is the answer of a real local PySpark 4.2.0 run of `cases.py` (354 cases; `gen_goldens.py` has the
command). Spark 4.2.0 on JDK 21.0.13, `local[1]`, pandas 3.0.6, pyarrow 26.0.0, CPython 3.12. Every case is a known gap
in `test_udtf_goldens.py` ("PR-26c-4: UDTF planning"). Cases per area: rows 42, table 54, analyze 90, mistyped 84,
errors 83, definition 1 (kinds 300: 124, 301: 112, 302: 112, plus 5 kind-less definition errors).

Facts the run showed (all are in the goldens; Krishiv must match them or record the difference):

- A bare `@udtf` is Arrow-optimized in 4.2.0: `spark.sql.execution.pythonUDTF.arrow.enabled` defaults to `true`
  (`defaults` in the golden file; `pyspark/sql/udtf.py` `_create_py_udtf`). Kind 300 is `@udtf(useArrow=False)` here.
- 300 converts a mistyped eval result in the JVM and mostly yields NULL, not an error: a string, a float, an int for a
  double or a bool, a Decimal for an int, or a 64-bit value for an int column all become NULL (the overflow `2**40` for
  an int column becomes 0). Only `None` for a non-nullable column (`UDTF_EXEC_ERROR`) and a short tuple for a struct
  (`STRUCT_ARRAY_LENGTH_MISMATCH`) raise. 301 converts through Arrow and raises `UDTF_ARROW_DATA_CONVERSION_ERROR`
  for most of these, but accepts a float, an int for a double and a Decimal for an int. 302 raises the pyarrow error
  (`PYTHON_EXCEPTION`, `pyarrow.lib.ArrowInvalid: ...`) or casts safely (a numeric string for an int).
- 300 accepts a dict, list or `Row` as an output row; 301 refuses dict and list (`UDTF_ARROW_DATA_CONVERSION_ERROR`) and
  accepts a bare scalar that 300 refuses (`UDTF_INVALID_OUTPUT_ROW_TYPE`); a `None` row is one all-NULL row.
- `@arrow_udtf` cannot have `analyze` (`INVALID_ARROW_UDTF_WITH_ANALYZE`, raised at definition) and cannot be a `LATERAL`
  target (`LATERAL_JOIN_WITH_ARROW_UDTF_UNSUPPORTED`, 0A000); 300 and 301 can.
- A `TABLE` argument over an empty relation (and a lateral over an empty input) with a `terminate` throws a JVM
  `NullPointerException` in 4.2.0 (`rows/*/empty_table_input`, `empty_lateral_input`; a Spark bug: Krishiv will not
  reproduce it, a decision for the planner PR); `PARTITION BY` over an empty relation returns no rows.
- `analyze` sees constants as Python values (`datetime.date`, `Decimal`, `bytes`, `datetime.timedelta`, `Row` for a
  struct, `None` for NULL with data type `void` for an untyped NULL); `is_constant` is false for a column reference.
  A `dataclasses.replace` rebuilds the `AnalyzeResult` for `__init__`, so a subclass carries extra data only as
  dataclass fields.
- `ORDER BY` without `PARTITION BY` is a parse error; a multi-expression `ORDER BY` needs parentheses
  (`_LEGACY_ERROR_TEMP_0064`).

Error records hold the exception class, condition, SQLSTATE, the first message line and the worker's `[CLASS] message`
or traceback line. Py4J errors (a JVM exception with no condition) record the JVM exception line with stage, task and
host normalised. Hand-written expectations: none.
