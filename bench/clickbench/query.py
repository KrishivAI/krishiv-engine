# SPDX-License-Identifier: Apache-2.0
"""Runs one ClickBench query (SQL on stdin) on Krishiv in-process; the elapsed seconds are the last stderr line.

The timed span is planning plus execution plus the transfer of the whole result into an Arrow table; the result is
not discarded early. The ``hits`` view is registered before the clock starts. One process per run, so nothing is
cached between runs except the operating system's page cache.
"""

import os
import sys
import time

import krishiv  # ty: ignore[unresolved-import]  # installed in the entry's own venv, not the bench environment

query = sys.stdin.read().strip()
if not query:
    sys.exit("no query on stdin")

here = os.path.dirname(os.path.abspath(__file__))
spark = krishiv.session()
spark.sql("SET spark.sql.parquet.binaryAsString=true")
with open(os.path.join(here, "create.sql")) as f:
    ddl = "\n".join(ln for ln in f.read().splitlines() if not ln.lstrip().startswith("--"))
spark.sql(ddl.replace("`hits.parquet`", "`" + os.path.join(here, "hits.parquet") + "`").strip().rstrip(";"))

start = time.perf_counter()
table = spark.sql(query.rstrip(";")).toArrow()
elapsed = time.perf_counter() - start

print(table.num_rows)
print(f"{elapsed:.4f}", file=sys.stderr)
