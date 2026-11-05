#!/usr/bin/env python3
"""Generate the Spark 4.3 correctness-check fixture (queue item "Four correctness checks").

Every expectation comes from a REAL Spark 4.3.0 run (Spark Connect), never from memory. Cases:
  nd_*      (1) non-deterministic predicates (`rand`) are evaluated after the scan, never pushed into it.
  gen_*     (2) with a generator in the SELECT list, HAVING is applied before window functions.
  opts_*    (4) two references to the same relation with different `WITH (...)` options keep their own.
Check (3), a failing partition predicate, needs a partitioned table and is not generated here.

Data is read from the Spark image's read-only example files (`{kv}` is kv1.txt, `{people}` is
people.csv); copies live beside the fixture. Temporary views only; nothing is written. Each row is
recorded as the text of its values, rows sorted. Output is deterministic and has no local paths.

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.3.0' \\
      python tests/scripts/gen_spark43_checks_goldens.py --remote sc://127.0.0.1:15013 \\
      --out tests/fixtures/spark-4.3.0/spark43-checks.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_spark43_checks_goldens.py"
LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")
RES = "/opt/spark/examples/src/main/resources"
KV = f"csv.`{RES}/kv1.txt` WITH ('delimiter'='\\u0001')"
KV_SQL = "csv.`{kv}` WITH ('delimiter'='\\u0001')"
PEOPLE = "csv.`{people}` WITH ('delimiter'=';','header'='{h}')"

VIEW_D = "create temporary view d as select * from values (1,10),(2,20),(3,30) as t(k,v)"

CASES: list[tuple[str, list[str], str]] = [
    # (1) non-deterministic predicates stay after the scan.
    ("nd_alone", [], f"select count(*) from {KV_SQL} where rand(7) < 0.1"),
    ("nd_after_det", [], f"select count(*) from {KV_SQL} where cast(_c0 as int) % 2 = 0 and rand(7) < 0.3"),
    ("nd_before_det", [], f"select count(*) from {KV_SQL} where rand(7) < 0.3 and cast(_c0 as int) % 2 = 0"),
    ("nd_subquery", [], f"select sum(cast(_c0 as int)) from (select _c0 from {KV_SQL} where rand(11) < 0.2)"),
    ("nd_two_seeds", [], f"select count(*) from {KV_SQL} where rand(1) < 0.5 and rand(2) < 0.5"),
    ("nd_in_select", [], f"select count(*), sum(cast(_c0 as int)) from (select _c0, rand(5) r from {KV_SQL}) where r < 0.25"),
    # (2) HAVING before window functions beside a generator.
    ("gen_window_sum", [VIEW_D], "select k, explode(array(1,2)) x, sum(v) over () s from d group by k, v having k > 1"),
    ("gen_window_count", [VIEW_D], "select k, explode(array(1,2)) x, count(*) over () c from d group by k having k > 1"),
    ("nogen_window_sum", [VIEW_D], "select k, sum(v) over (order by k) s from d group by k, v having k > 1 order by k"),
    # (4) each reference keeps its own options.
    ("opts_self_join_header_first", [], f"select a.name, b._c0 from {PEOPLE.format(h='true', people='{people}')} a join {PEOPLE.format(h='false', people='{people}')} b on true order by 1, 2"),
    ("opts_self_join_header_second", [], f"select a._c0, b.name from {PEOPLE.format(h='false', people='{people}')} a join {PEOPLE.format(h='true', people='{people}')} b on true order by 1, 2"),
    ("opts_union", [], f"select name from {PEOPLE.format(h='true', people='{people}')} union all select _c0 from {PEOPLE.format(h='false', people='{people}')} order by 1"),
]


def run(remote: str) -> list[dict]:
    from pyspark.sql import SparkSession

    out = []
    for cid, setup, sql in CASES:
        # A fresh session per case; temporary views only.
        spark = SparkSession.builder.remote(remote).create()
        try:
            spark.sql("SET spark.sql.ansi.enabled=true").collect()
            for stmt in setup:
                spark.sql(stmt).collect()
            real = sql.replace("{kv}", f"{RES}/kv1.txt").replace("{people}", f"{RES}/people.csv")
            case = {"id": cid, "setup": setup, "sql": sql}
            try:
                rows = [[None if v is None else str(v) for v in r] for r in spark.sql(real).collect()]
                case["rows"] = rows if cid.endswith(("_first", "_second", "union")) or "order by" in sql else sorted(rows)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                m = re.search(r"\[([A-Z_.]+)\]", msg)
                case["error_class"] = m.group(1) if m else "UNKNOWN"
            out.append(case)
        finally:
            spark.stop()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pyspark.sql import SparkSession

    s = SparkSession.builder.remote(a.remote).create()
    version = s.version
    s.stop()
    doc = {"format_version": FORMAT_VERSION, "generator": GENERATOR, "spark_version": version, "cases": run(a.remote)}
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    assert not LOCAL_PATH.search(text), "local path in fixture"
    Path(a.out).write_text(text)
    print("wrote", a.out, len(doc["cases"]), "cases, Spark", version)


if __name__ == "__main__":
    main()
