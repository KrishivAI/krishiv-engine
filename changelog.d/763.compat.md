`spark.read.json` reads nested JSON: objects become structs, arrays become arrays, and a DDL schema may also declare
maps (duplicate keys kept), with Spark's partial results for values that do not fit. Inference now orders fields by
name, reads a top-level array of objects as one row per object, adds `_corrupt_record` for a malformed line, and
refuses repeated field names with `COLUMN_ALREADY_EXISTS`, as Spark 4.2 does. Scalar values follow Spark's JSON token
rules: a boolean column reads only `true`/`false`, an integer column only an integral number, a string column holds
a nested value's text exactly as written, and `allowNonNumericNumbers` is honoured. Differences from Spark: a type
nested more than 32 levels deep is refused (`FAILED_TO_PARSE_TOO_COMPLEX`), and a record nested past 1,000 levels is
wholly malformed. Writing nested JSON is not supported yet.
