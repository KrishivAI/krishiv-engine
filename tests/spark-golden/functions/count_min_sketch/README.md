# `count_min_sketch` (Spark's `CountMinSketchAgg`)

`fixture.json` is what Spark 4.3.0 answers for the statements of `gen.py`: serialized sketches byte for byte
(`hex`, or `sha2` for the large ones), output column names, and error classes with SQLSTATE. Spark 4.2.0 gave the
same answers for every statement.

Not in the fixture, observed on 4.3.0: a positive `eps` that is NaN or infinite sizes the rows at 0 counters and
fails at the first non-NULL value with a division by zero; an `eps` near `1e-10` fails with an
`OutOfMemoryError`. A collated string value is `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`.

Regenerate: `python gen.py fixture.json` (PySpark 4.3.0, JDK 21). Test:
`crates/krishiv-session/tests/aggregate_count_min_sketch.rs`.
