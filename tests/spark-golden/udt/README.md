# UDT rendering oracle (P3-T079)

`render_oracle.py` run with `spark-submit` in `apache/spark:4.2.0` (no numpy in the image, so the UDT is hand-written
over `array<double>`). Results, 2026-11-19:

- Python `simpleString`: `udt`. JVM `simpleString`/`typeName`: the class's simple name without `Type`/`UDT`,
  lower case (`examplepoint`); `printSchema` and `DESCRIBE` show it; `catalogString`: `array<double>`; `sql`:
  `ARRAY<DOUBLE>`.
- JSON: `{"type":"udt","class":<jvm class>,"pyClass":<python class>,"sqlType":{...}}`.
- A Parquet write and re-read keeps the UDT.

Tested in `crates/krishiv-types/src/tests.rs` `a_udt_renders_as_spark_does`.

## Python-only UDTs and the JSON key sets (2026-11-19, pyspark 4.2.0)

`probe_python_only.py` (`uv run --no-project --python 3.12 --with pyspark==4.2.0 python probe_python_only.python_only.py`), output
in `spark-4.2.0-probe.txt`. A UDT with no `scalaUDT` is `PythonUserDefinedType`: `typeName`/`simpleString`
`pythonuserdefined`, JSON `{type, pyClass, serializedClass, sqlType}`. `DataType.fromJson` takes exactly
`{type, class, pyClass (null allowed), sqlType}` or `{type, pyClass, serializedClass, sqlType}`; Krishiv writes those
and reads `"pyClass":null`. Not probed: the 3.5/4.0/4.1/4.3 profiles (the UDT JSON is unchanged in Spark's source
since 2.x as far as the 4.2.0 run shows; a 4.3.0 run is a follow-up for the profile pass).
