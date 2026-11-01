# SPDX-License-Identifier: Apache-2.0
"""Writes crates/krishiv-types/vendored/collations.tsv: `SELECT * FROM collations()` on Spark 4.2.0.

Run: uvx --from pyspark==4.2.0 python -I tests/scripts/gen_collations.py > crates/krishiv-types/vendored/collations.tsv
"""

import sys

from pyspark.sql import SparkSession

s = SparkSession.builder.master("local[1]").getOrCreate()
rows = s.sql("select * from collations()").collect()
print(len(rows), file=sys.stderr)
print(s.sql("select * from collations()").columns, file=sys.stderr)
for r in rows:
    print("\t".join("" if v is None else str(v) for v in r))
