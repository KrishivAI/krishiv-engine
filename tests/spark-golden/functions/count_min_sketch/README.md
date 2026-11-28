# `count_min_sketch` (Spark's `CountMinSketchAgg`)

`fixture.json` is what Spark 4.3.0 answers for the statements of `gen.py`: serialized sketches byte for byte
(`hex`, or `sha2` for the large ones), output column names, and error classes with SQLSTATE. Spark 4.2.0 gave the
same answers for every statement.

Not in the fixture, observed on 4.3.0: a positive `eps` that is NaN or infinite sizes the rows at 0 counters and
fails at the first non-NULL value with a division by zero; an `eps` near `1e-10` fails with an
`OutOfMemoryError`. A collated string value is `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`.

Regenerate: `python gen.py fixture.json` (PySpark 4.3.0, JDK 21). Test:
`crates/krishiv-session/tests/aggregate_count_min_sketch.rs`.

## Spark sources (tags `v4.2.0` and `v4.3.0`; the aggregate's line numbers are the same in both)

- `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/CountMinSketchAgg.scala`:
  `CountMinSketchAgg` (line 43), `checkInputDataTypes` (71: the argument checks and their error classes),
  `createAggregationBuffer` (134), `update` (138), `merge` (152), `eval`/`serialize` (157-159), `deserialize` (163).
- `common/sketch/src/main/java/org/apache/spark/util/sketch/CountMinSketchImpl.java` (v4.3.0): `addLong` (169),
  `hash` (201), `mergeInPlace` (264), `writeTo` (299: the serialized format).
