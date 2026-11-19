# UDT rendering oracle (P3-T079)

`render_oracle.py` run with `spark-submit` in `apache/spark:4.2.0` (no numpy in the image, so the UDT is hand-written
over `array<double>`). Results, 2026-11-19:

- Python `simpleString`: `udt`. JVM `simpleString`/`typeName`: the class's simple name without `Type`/`UDT`,
  lower case (`examplepoint`); `printSchema` and `DESCRIBE` show it; `catalogString`: `array<double>`; `sql`:
  `ARRAY<DOUBLE>`.
- JSON: `{"type":"udt","class":<jvm class>,"pyClass":<python class>,"sqlType":{...}}`.
- A Parquet write and re-read keeps the UDT.

Tested in `crates/krishiv-types/src/tests.rs` `a_udt_renders_as_spark_does`.
