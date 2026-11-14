`spark.read.json` reads nested JSON: objects become structs, arrays become arrays, and a DDL schema may also declare
maps, with Spark's partial results for values that do not fit. Inference now orders fields by name, reads a top-level
array of objects as one row per object, and adds `_corrupt_record` for a malformed line, as Spark 4.2 does.
Scalar values follow Spark's JSON token rules: a boolean column reads only `true`/`false`, an integer column only an
integral number, and a string column holds a nested value's text exactly as written.
