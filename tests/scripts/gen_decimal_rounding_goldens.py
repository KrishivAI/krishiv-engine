#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate decimal AVG / division rounding goldens (queue item "Decimal goldens that catch truncation").

Sail returned wrong rows by truncating instead of rounding HALF_UP at the result scale (TPC-H q01 `avg(l_discount)`
0.050006 vs Spark 0.050007; TPC-DS q83 16.6666 vs 16.666667). Every expectation here comes from a REAL Spark 4.2
run over Spark Connect, never from memory. The script defines small temporary views (values cast from text so the
engine under test builds the same columns), runs each case through `spark.sql` and records output names,
`simpleString` types, nullability and rows (as text, sorted), or the error class, SQLSTATE and parameters.
Cases cover: AVG whose exact value needs rounding at the result scale (exact .5 ties and just-below / just-above
ties), negatives (HALF_UP rounds away from zero), GROUP BY, large precision (the 38-digit cap and
`spark.sql.decimalOperations.allowPrecisionLoss` off), division in projections and over aggregates (`sum/count`,
`sum/sum`), and the TPC-H q01 / TPC-DS q83 shapes. Row order never matters: every row carries a key.
Output is deterministic and contains no local paths. Only Spark 4.2.0 is recorded; fixtures for
3.5, 4.0 and 4.1 need their own oracle runs (ADR 0010).

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \\
      python tests/scripts/gen_decimal_rounding_goldens.py --remote sc://127.0.0.1:15012 \\
      --out tests/fixtures/spark-4.2.0/decimal-rounding-goldens.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_decimal_rounding_goldens.py"
LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")
M38 = "9" * 38

rng = random.Random(20261104)


def dec(unscaled: int, scale: int) -> str:
    sign = "-" if unscaled < 0 else ""
    digits = str(abs(unscaled)).rjust(scale + 1, "0")
    return f"{sign}{digits[:-scale]}.{digits[-scale:]}" if scale else f"{sign}{digits}"


# TPC-H lineitem-like: discounts in 0.00..0.10 (decimal(15,2)), many rows, three groups; averages land on the
# 7th decimal digit boundary (q01 shape: avg(decimal(15,2)) is decimal(19,6)).
LINEITEM = [
    (
        str(i % 3),
        dec(rng.randrange(0, 11), 2),
        dec(rng.randrange(90000, 10500000), 2),
        dec(rng.randrange(1, 51) * 100, 2),
    )
    for i in range(200)
]
# Exact ties at the result scale: avg of decimal(5,0) over 32 rows is x/32 (5 decimal digits of 1/32 = .03125,
# result scale 4 so .03125 is a tie); avg(decimal(10,2)) of {0.01,0.00,...} gives ties at scale 6.
TIES = [("a", dec(1, 0))] + [("a", dec(0, 0))] * 31
TIES += [("b", dec(-1, 0))] + [("b", dec(0, 0))] * 31
TIES += [("c", dec(1, 0)), ("c", dec(2, 0)), ("c", dec(2, 0))]  # 1.6667
TIES += [("d", dec(-1, 0)), ("d", dec(-2, 0)), ("d", dec(-2, 0))]  # -1.6667
TIES += [("e", dec(5, 0)), ("e", dec(5, 0)), ("e", dec(6, 0))]  # 5.3333
TIES += [("f", dec(1, 0)), ("f", dec(1, 0)), ("f", dec(1, 0)), ("f", dec(2, 0))]  # 1.25 exactly

# decimal(10,2) values: group e is inexact (5/3 = 1.666667); groups g and h are exact ties at the result scale 6
# (0.01 / 32 = 0.0003125 -> 0.000313 and its negative -> -0.000313); the rest divide exactly.
HALF = (
    [
        ("a", "0.01"),
        ("a", "0.00"),
        ("b", "-0.01"),
        ("b", "0.00"),
        ("c", "0.03"),
        ("c", "0.00"),
        ("c", "0.00"),
        ("d", "0.02"),
        ("d", "0.00"),
        ("d", "0.00"),
        ("d", "0.00"),
        ("e", "0.01"),
        ("e", "0.02"),
        ("e", "0.02"),
    ]
    + [("g", "0.01")]
    + [("g", "0.00")] * 31
    + [("h", "-0.01")]
    + [("h", "0.00")] * 31
)

BIG = [
    (str(i % 2), str(rng.randrange(-(10**19), 10**19)) + "." + str(rng.randrange(10**9)).rjust(9, "0"))
    for i in range(42)
]
HUGE = [("0", M38), ("0", M38), ("0", "1"), ("1", "-" + M38), ("1", "-" + M38), ("1", "-1"), ("1", "7")]
WIDE = [(str(i % 2), dec(rng.randrange(-(10**27), 10**27), 10)) for i in range(25)]  # decimal(38,10)

# Division operands: (id, a, b), a decimal(12,2), b decimal(9,3); many do not divide exactly.
DIVIDE: list[tuple[str | None, ...]] = []
for i in range(60):
    a = dec(rng.randrange(-(10**9), 10**9), 2)
    b = dec(rng.choice([1, -1]) * rng.randrange(1, 10**6), 3)
    DIVIDE.append((str(i), a, b))
DIVIDE += [
    ("100", "1.00", "6.000"),
    ("101", "-1.00", "6.000"),
    ("102", "2.00", "3.000"),
    ("103", "-2.00", "3.000"),
    ("104", "1.00", "8.000"),
    ("105", "-1.00", "8.000"),
    ("106", "0.01", "3.000"),
    ("107", "5.00", "9.000"),
    ("108", "100.00", "6.000"),
    ("109", "0.00", "7.000"),
    ("110", "1.00", "0.000"),
    ("111", None, "3.000"),
    ("112", "0.01", "16.384"),
    ("113", "-0.01", "16.384"),  # 0.0006103515625: an exact tie at scale 12
]
# TPC-DS q83 shape: three return quantities per item; value = qty / (a + b + c) / 3 * 100.
Q83 = [("a", 7, 5, 4), ("b", 1, 1, 1), ("c", 2, 5, 11), ("d", 3, 3, 3), ("e", 10, 1, 3), ("f", 13, 2, 7)]

TABLES: dict[str, tuple[str, Sequence[Sequence[str | None]]]] = {
    "li": ("k INT, disc DECIMAL(15,2), price DECIMAL(15,2), qty DECIMAL(15,2)", LINEITEM),
    "ties": ("k STRING, x DECIMAL(5,0)", TIES),
    "half": ("k STRING, x DECIMAL(10,2)", HALF),
    "big": ("k INT, x DECIMAL(30,9)", BIG),
    "huge": ("k INT, x DECIMAL(38,0)", HUGE),
    "wide": ("k INT, x DECIMAL(38,10)", WIDE),
    "dv": ("id INT, a DECIMAL(12,2), b DECIMAL(9,3)", DIVIDE),
    "q83": (
        "k STRING, sr DECIMAL(7,0), cr DECIMAL(7,0), wr DECIMAL(7,0)",
        [(k, str(a), str(b), str(c)) for k, a, b, c in Q83],
    ),
}

NO_LOSS = {"spark.sql.decimalOperations.allowPrecisionLoss": "false"}
NO_ANSI = {"spark.sql.ansi.enabled": "false"}
CASES: list[tuple[str, str, dict[str, str]]] = [
    (
        "q01_avg_disc",
        "SELECT k, avg(disc) AS a, avg(price) AS p, avg(qty) AS q, sum(disc) AS s, count(*) AS c FROM li GROUP BY k",
        {},
    ),
    ("q01_avg_disc_global", "SELECT avg(disc), avg(price), avg(qty) FROM li", {}),
    ("q01_avg_disc_no_ansi", "SELECT k, avg(disc) AS a FROM li GROUP BY k", NO_ANSI),
    ("q01_sum_over_count", "SELECT k, sum(disc) / count(*) AS d FROM li GROUP BY k", {}),
    ("q01_sum_over_sum", "SELECT k, sum(disc) / sum(qty) AS d FROM li GROUP BY k", {}),
    (
        "q01_sum_disc_price",
        "SELECT k, sum(price * (1 - disc)) AS a, avg(price * (1 - disc)) AS b FROM li GROUP BY k",
        {},
    ),
    ("q01_avg_no_loss", "SELECT k, avg(disc) AS a FROM li GROUP BY k", NO_LOSS),
    ("ties_avg_group", "SELECT k, avg(x) AS a FROM ties GROUP BY k", {}),
    ("ties_avg_global", "SELECT avg(x) FROM ties", {}),
    ("ties_avg_negated", "SELECT k, avg(-x) AS a FROM ties GROUP BY k", {}),
    ("half_avg_group", "SELECT k, avg(x) AS a FROM half GROUP BY k", {}),
    ("half_avg_negated", "SELECT k, avg(-x) AS a FROM half GROUP BY k", {}),
    ("half_avg_distinct", "SELECT k, avg(DISTINCT x) AS a FROM half GROUP BY k", {}),
    ("half_avg_window", "SELECT k, x, avg(x) OVER (PARTITION BY k) AS a FROM half", {}),
    ("half_avg_filter", "SELECT k, avg(x) FILTER (WHERE x <> 0) AS a FROM half GROUP BY k", {}),
    ("big_avg_group", "SELECT k, avg(x) AS a FROM big GROUP BY k", {}),
    ("big_avg_no_loss", "SELECT k, avg(x) AS a FROM big GROUP BY k", NO_LOSS),
    ("huge_avg_group", "SELECT k, avg(x) AS a FROM huge GROUP BY k", {}),
    ("huge_avg_no_ansi", "SELECT k, avg(x) AS a FROM huge GROUP BY k", NO_ANSI),
    ("wide_avg_group", "SELECT k, avg(x) AS a FROM wide GROUP BY k", {}),
    ("wide_avg_no_loss", "SELECT k, avg(x) AS a FROM wide GROUP BY k", NO_LOSS),
    ("divide_rows", "SELECT id, a / b AS q FROM dv WHERE b <> 0", {}),
    ("divide_rows_no_loss", "SELECT id, a / b AS q FROM dv WHERE b <> 0", NO_LOSS),
    ("divide_rows_no_ansi", "SELECT id, a / b AS q FROM dv WHERE b <> 0", NO_ANSI),
    ("divide_swapped", "SELECT id, b / a AS q FROM dv WHERE a <> 0", {}),
    ("divide_literal_int", "SELECT id, a / 3 AS q, b / 7 AS r FROM dv WHERE b <> 0", {}),
    ("divide_literal_decimal", "SELECT id, a / 3.0 AS q, 1.00 / b AS r FROM dv WHERE b <> 0", {}),
    ("divide_self", "SELECT id, a / a AS q FROM dv WHERE a <> 0", {}),
    ("divide_wide", "SELECT k, x / 3 AS q, x / 7.0 AS r FROM wide", {}),
    ("divide_huge", "SELECT k, x / 3 AS q, x / CAST(7 AS DECIMAL(38,0)) AS r FROM huge", {}),
    ("divide_huge_no_loss", "SELECT k, x / 3 AS q FROM huge", NO_LOSS),
    ("divide_then_avg", "SELECT avg(a / b) FROM dv WHERE b <> 0", {}),
    ("divide_then_sum", "SELECT sum(a / b) FROM dv WHERE b <> 0", {}),
    ("divide_by_zero", "SELECT id, a / b AS q FROM dv WHERE id = '110'", {}),
    ("q83_value", "SELECT k, sr / (sr + cr + wr) / 3 * 100 AS v FROM q83", {}),
    ("q83_value_cast", "SELECT k, CAST(sr AS DECIMAL(17,2)) / (sr + cr + wr) / 3 * 100 AS v FROM q83", {}),
    ("q83_avg_of_three", "SELECT k, (sr + cr + wr) / 3.0 AS v FROM q83", {}),
    ("q83_avg_dev", "SELECT k, sr / ((sr + cr + wr) / 3.0) * 100 AS v FROM q83", {}),
    ("q83_group_avg", "SELECT avg(sr), avg(sr + cr + wr), sum(sr) / sum(cr) FROM q83", {}),
]


def view_sql(name: str, columns: str, rows: Sequence[Sequence[str | None]]) -> str:
    cols = [c.strip().split(" ", 1) for c in columns.split(", ")]

    def lit(text: str | None, ty: str) -> str:
        if text is None:
            return f"CAST(NULL AS {ty})"
        quoted = text.replace("'", "''")
        return f"CAST('{quoted}' AS {ty})"

    values = ", ".join("(" + ", ".join(lit(v, ty) for v, (_, ty) in zip(r, cols, strict=True)) + ")" for r in rows)
    names = ", ".join(n for n, _ in cols)
    return f"CREATE OR REPLACE TEMPORARY VIEW {name} AS SELECT * FROM VALUES {values} AS {name}({names})"


def text_of(value, type_name: str) -> str | None:
    """A result value as the fixture records it: decimals in plain notation, integers in decimal."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise SystemExit(f"unexpected {type(value).__name__} value for {type_name}")


def record_error(e) -> dict:
    cond = getattr(e, "getCondition", None) or getattr(e, "getErrorClass", None)
    params = e.getMessageParameters() if hasattr(e, "getMessageParameters") else {}
    return {
        "error_class": cond() if cond else "UNCLASSIFIED",
        "sql_state": e.getSqlState() if hasattr(e, "getSqlState") else None,
        "message_parameters": dict(sorted((params or {}).items())),
    }


def main() -> None:
    from pyspark.errors import PySparkException
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://127.0.0.1:15012")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/decimal-rounding-goldens.json")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    if spark.version != "4.2.0":
        raise SystemExit(f"expected Spark 4.2.0, found {spark.version}")
    tables = {}
    for name, (columns, rows) in TABLES.items():
        spark.sql(view_sql(name, columns, rows))
        schema = spark.table(name).schema
        tables[name] = {
            "columns": [
                {"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable} for f in schema.fields
            ],
            "rows": [list(r) for r in rows],
        }

    cases = []
    for case_id, sql, conf in CASES:
        for k, v in conf.items():
            spark.conf.set(k, v)
        try:
            df = spark.sql(sql)
            fields = df.schema.fields
            rows = df.collect()
            types = [f.dataType.simpleString() for f in fields]
            out = [[text_of(v, ty) for v, ty in zip(r, types, strict=True)] for r in rows]
            result = {
                "ok": True,
                "output": [
                    {"name": f.name, "type": ty, "nullable": f.nullable} for f, ty in zip(fields, types, strict=True)
                ],
                "rows": sorted(out, key=lambda r: json.dumps(r)),
            }
        except PySparkException as e:
            result = {"ok": False, **record_error(e)}
        finally:
            for k in conf:
                spark.conf.unset(k)
        cases.append({"id": case_id, "sql": sql, "conf": conf, **result})

    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": spark.version,
        "generator": GENERATOR,
        "notes": "Output columns and rows (as text, sorted) or error (condition, SQLSTATE, message parameters) "
        "of each case over the temporary views `tables` (rows as the text each value is cast from).",
        "tables": tables,
        "cases": cases,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {args.out}")


if __name__ == "__main__":
    main()
