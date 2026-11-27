#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the `show_string` fixture (P3-T012): what Spark 4.3.0's `Dataset.showString` answers.

Every expectation comes from a REAL Spark 4.3.0 run (Spark Connect), never from memory. A case is a query
(`sql`), `num_rows`, `truncate` and `vertical`; the answer is the text of the ShowString relation that
`DataFrame._show_string` sends. Queries use only literals and `range`, so the shared oracle takes no tables.

    uv run --no-project --python 3.12 --with 'pyspark-client==4.3.0' --with pandas --with pyarrow \\
      python tests/scripts/gen_show_string_goldens.py --remote sc://127.0.0.1:15077 \\
      --out tests/fixtures/spark-4.3.0/show-string.json
"""

from __future__ import annotations

import argparse
import json

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_show_string_goldens.py"

LONG = "abcdefghijklmnopqrstuvwxyz0123456789"
# (name, sql, num_rows, truncate, vertical)
CASES = [
    ("basic", "SELECT id, id * 2 AS twice FROM range(3)", 20, 20, False),
    ("empty_rows", "SELECT id FROM range(0)", 20, 20, False),
    ("one_row_footer", "SELECT id FROM range(3)", 1, 20, False),
    ("two_rows_footer", "SELECT id FROM range(5)", 2, 20, False),
    ("exactly_num_rows", "SELECT id FROM range(2)", 2, 20, False),
    ("num_rows_zero", "SELECT id FROM range(2)", 0, 20, False),
    ("num_rows_negative", "SELECT id FROM range(2)", -3, 20, False),
    ("nulls", "SELECT CAST(NULL AS STRING) AS s, CAST(NULL AS INT) AS i, 1 AS one", 20, 20, False),
    ("binary", "SELECT X'0102' AS b, X'' AS empty, X'ff' AS ff", 20, 20, False),
    ("truncate_default", f"SELECT '{LONG}' AS s", 20, 20, False),
    ("truncate_5", f"SELECT '{LONG}' AS s", 20, 5, False),
    ("truncate_3", f"SELECT '{LONG}' AS s", 20, 3, False),
    ("truncate_2", f"SELECT '{LONG}' AS s", 20, 2, False),
    ("truncate_1", f"SELECT '{LONG}' AS s", 20, 1, False),
    ("truncate_0_left_aligned", f"SELECT '{LONG}' AS s, 1 AS i, 'x' AS c", 20, 0, False),
    ("truncate_negative", f"SELECT '{LONG}' AS s, 1 AS i", 20, -1, False),
    ("header_truncated", f"SELECT 1 AS `{LONG}`", 20, 10, False),
    ("min_width_three", "SELECT 1 AS a, 'bb' AS b", 20, 20, False),
    ("escapes", "SELECT 'a\\nb\\tc\\\\d' AS s, 'q\"r' AS t", 20, 20, False),
    ("wide_chars", "SELECT '中文' AS s, 'ab' AS t, 'é' AS e", 20, 20, False),
    ("wide_chars_truncated", "SELECT '中文中文中文' AS s", 20, 4, False),
    (
        "types",
        "SELECT true AS b, 1Y AS t, 2S AS s, 3 AS i, 4L AS l, 1.5F AS f, 2.5D AS d, "
        "CAST(1.10 AS DECIMAL(5,2)) AS dec, DATE'2024-02-29' AS dt, "
        "TIMESTAMP_NTZ'2024-02-29 01:02:03.5' AS ntz",
        20,
        20,
        False,
    ),
    (
        "special_floats",
        "SELECT CAST('NaN' AS DOUBLE) AS n, CAST('Infinity' AS DOUBLE) AS p, "
        "CAST('-Infinity' AS DOUBLE) AS m, -0.0D AS z, 1e10D AS big, 1e-5D AS small",
        20,
        20,
        False,
    ),
    ("array", "SELECT array(1, 2, 3) AS a, array() AS e, array('x', 'y') AS s", 20, 20, False),
    ("array_with_null", "SELECT array(1, CAST(NULL AS INT), 3) AS a", 20, 20, False),
    ("map", "SELECT map('k', 1, 'j', 2) AS m", 20, 20, False),
    ("map_with_null", "SELECT map('k', CAST(NULL AS INT)) AS m", 20, 20, False),
    ("struct", "SELECT named_struct('a', 1, 'b', 'x') AS s", 20, 20, False),
    ("struct_with_null", "SELECT named_struct('a', CAST(NULL AS INT), 'b', 'x') AS s", 20, 20, False),
    ("nested", "SELECT array(named_struct('a', array(1, 2))) AS n", 20, 20, False),
    ("array_of_binary", "SELECT array(X'0102') AS a", 20, 20, False),
    ("vertical", "SELECT id, 'v' || id AS name FROM range(2)", 20, 20, True),
    ("vertical_empty", "SELECT id FROM range(0)", 20, 20, True),
    ("vertical_footer", "SELECT id, 'v' AS n FROM range(5)", 2, 20, True),
    ("vertical_truncate_0", f"SELECT '{LONG}' AS s, 1 AS i", 20, 0, True),
    ("vertical_nulls", "SELECT CAST(NULL AS STRING) AS s, 1 AS i", 20, 20, True),
    ("many_columns", "SELECT 1 AS a, 2 AS b, 3 AS c, 4 AS d, 5 AS e, 6 AS f", 20, 20, False),
    ("no_columns", "SELECT * FROM (SELECT 1 AS a) WHERE false", 20, 20, False),
]


def main() -> None:
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    spark.conf.set("spark.sql.ansi.enabled", "true")
    cases = []
    for name, sql, num_rows, truncate, vertical in CASES:
        case = {"name": name, "sql": sql, "num_rows": num_rows, "truncate": truncate, "vertical": vertical}
        try:
            case["expected"] = spark.sql(sql)._show_string(num_rows, truncate, vertical)
        except Exception as e:  # recorded, so a refusal is also pinned
            case["error"] = type(e).__name__ + ": " + str(e).splitlines()[0][:200]
        cases.append(case)
    out = {"format_version": FORMAT_VERSION, "generator": GENERATOR, "spark": spark.version, "cases": cases}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(cases)} cases, Spark {spark.version}")


if __name__ == "__main__":
    main()
