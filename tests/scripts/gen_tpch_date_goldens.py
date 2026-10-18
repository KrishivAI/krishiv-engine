#!/usr/bin/env python3
"""Generate Spark expectations for the TPC-H date predicates (P2-T025a).

Runs a REAL Spark (never from memory) in a throw-away container. For each TPC-H query that filters on dates
(Q1, Q3-Q7, Q10, Q12, Q14, Q15, Q20; the substitution parameters of the specification's validation run) it
evaluates every comparison of the WHERE clause against probe dates just below, on and just above each bound,
the bound expressions themselves (`DATE '1998-12-01' - INTERVAL '90' DAY`, `... + INTERVAL '3' MONTH`) and
`extract(year FROM probe)` (Q7-Q9 group on it). A conjunction is recorded as its comparisons. Each case is a
single-expression SELECT with Spark's typed result (`boolean`, `date` as epoch days, `int`). Usage (from repo
root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_tpch_date_goldens.py \
        > tests/fixtures/spark-4.2.0/tpch-date-predicates.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import datetime
import json
import sys

from pyspark.sql import SparkSession

# (query, lower-bound comparisons, bound expression strings). `{d}` is the probe date.
QUERIES = [
    ("Q1", [("<=", "DATE'1998-12-01' - INTERVAL '90' DAY")]),
    ("Q3", [("<", "DATE'1995-03-15'"), (">", "DATE'1995-03-15'")]),
    ("Q4", [(">=", "DATE'1993-07-01'"), ("<", "DATE'1993-07-01' + INTERVAL '3' MONTH")]),
    ("Q5", [(">=", "DATE'1994-01-01'"), ("<", "DATE'1994-01-01' + INTERVAL '1' YEAR")]),
    ("Q6", [(">=", "DATE'1994-01-01'"), ("<", "DATE'1994-01-01' + INTERVAL '1' YEAR")]),
    ("Q7", [(">=", "DATE'1995-01-01'"), ("<=", "DATE'1996-12-31'")]),
    ("Q10", [(">=", "DATE'1993-10-01'"), ("<", "DATE'1993-10-01' + INTERVAL '3' MONTH")]),
    ("Q12", [(">=", "DATE'1994-01-01'"), ("<", "DATE'1994-01-01' + INTERVAL '1' YEAR")]),
    ("Q14", [(">=", "DATE'1995-09-01'"), ("<", "DATE'1995-09-01' + INTERVAL '1' MONTH")]),
    ("Q15", [(">=", "DATE'1996-01-01'"), ("<", "DATE'1996-01-01' + INTERVAL '3' MONTH")]),
    ("Q20", [(">=", "DATE'1994-01-01'"), ("<", "DATE'1994-01-01' + INTERVAL '1' YEAR")]),
]
# Q12 also compares the three date columns of a line item with each other.
PAIRS = [("1994-02-01", "1994-02-02"), ("1994-02-02", "1994-02-01"), ("1994-03-03", "1994-03-03")]


def probes(bound_sql, spark):
    bound = spark.sql(f"SELECT {bound_sql}").collect()[0][0]
    out = {bound + datetime.timedelta(days=k) for k in (-31, -1, 0, 1, 30)}
    out |= {datetime.date(1992, 1, 1), datetime.date(1998, 12, 31)}
    return sorted(out)


def main() -> None:
    spark = SparkSession.builder.appName("tpch-date-goldens").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    cases = []

    def record(query, sql, ty, wrap=None):
        row = spark.sql(f"SELECT {wrap % sql if wrap else sql}").collect()[0][0]
        cases.append({"query": query, "sql": f"SELECT {sql}", "type": ty, "value": row})

    for query, comparisons in QUERIES:
        for op, bound in comparisons:
            record(query, bound, "date", "unix_date(%s)")
            for d in probes(bound, spark):
                record(query, f"DATE'{d.isoformat()}' {op} {bound}", "boolean")
    for d in (
        datetime.date(1992, 1, 2),
        datetime.date(1995, 12, 31),
        datetime.date(1996, 1, 1),
        datetime.date(1998, 12, 1),
    ):
        record("Q7", f"extract(year FROM DATE'{d.isoformat()}')", "int")
    for a, b in PAIRS:
        record("Q12", f"DATE'{a}' < DATE'{b}'", "boolean")
    json.dump({"spark_version": spark.version, "cases": cases}, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
