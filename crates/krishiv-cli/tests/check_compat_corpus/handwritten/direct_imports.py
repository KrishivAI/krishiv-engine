from pyspark.sql.functions import (
    col,
    count_min_sketch,
    expr,
    lower as lc,
    row_number,
    nope_fn,
)
from pyspark.sql.functions import (
    histogram_numeric,
    trim,
)

df = spark.table("t").select(col("a"), lc("b"), trim("c"))
df = df.select(count_min_sketch("a", 0.1, 0.1, 1), histogram_numeric("a", 3))
df = df.select(expr("cast(a as int)"), nope_fn("a"), row_number())
result = df.collect()

# ruff: noqa
