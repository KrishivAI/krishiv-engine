#!/usr/bin/env python3
"""Generate the Spark 4.3 correctness-check fixture (queue item "Four correctness checks").

Every expectation comes from a REAL Spark 4.3.0 run (Spark Connect), never from memory. Cases:
  nd_*      (1) non-deterministic predicates (`rand`) are evaluated after the scan, never pushed into it.
  gen_*     (2) with a generator in the SELECT list, HAVING is applied before window functions.
  opts_*    (4) two references to the same relation with different `WITH (...)` options keep their own.
  part_*    (3) a partition predicate whose evaluation fails raises the error; it never keeps the
            partition or lets a DELETE remove it. These cases build a small partitioned parquet table
            under `{tmp}` (a scratch directory of the Spark server, never a shared one).

Data is read from the Spark image's read-only example files (`{kv}` is kv1.txt, `{people}` is
people.csv); copies live beside the fixture. Other cases use temporary views only. Each row is
recorded as the text of its values, rows sorted. Output is deterministic and has no local paths.

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.3.0' \\
      python tests/scripts/gen_spark43_checks_goldens.py --remote sc://127.0.0.1:15013 \\
      --out tests/fixtures/spark-4.3.0/spark43-checks.json
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_spark43_checks_goldens.py"
LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")
RES = "/opt/spark/examples/src/main/resources"
KV = f"csv.`{RES}/kv1.txt` WITH ('delimiter'='\\u0001')"
KV_SQL = "csv.`{kv}` WITH ('delimiter'='\\u0001')"
PEOPLE = "csv.`{people}` WITH ('delimiter'=';','header'='{h}')"

P_HDR = PEOPLE.format(h="true", people="{people}")
P_RAW = PEOPLE.format(h="false", people="{people}")
VIEW_D = "create temporary view d as select * from values (1,10),(2,20),(3,30) as t(k,v)"
# (3): `dep` holds a non-numeric partition value, `pn` only numeric ones.
PT = (
    "create table pt using parquet partitioned by (dep) location '{tmp}/pt' as "
    "select * from values (1,'hr'),(2,'eng') as v(id,dep)"
)
PN = (
    "create table pn using parquet partitioned by (dep) location '{tmp}/pn' as "
    "select * from values (1,'1'),(2,'2') as v(id,dep)"
)

CASES: list[tuple[str, list[str], str]] = [
    # (1) non-deterministic predicates stay after the scan.
    ("nd_alone", [], f"select count(*) from {KV_SQL} where rand(7) < 0.1"),
    ("nd_after_det", [], f"select count(*) from {KV_SQL} where cast(_c0 as int) % 2 = 0 and rand(7) < 0.3"),
    ("nd_before_det", [], f"select count(*) from {KV_SQL} where rand(7) < 0.3 and cast(_c0 as int) % 2 = 0"),
    ("nd_subquery", [], f"select sum(cast(_c0 as int)) from (select _c0 from {KV_SQL} where rand(11) < 0.2)"),
    ("nd_two_seeds", [], f"select count(*) from {KV_SQL} where rand(1) < 0.5 and rand(2) < 0.5"),
    (
        "nd_in_select",
        [],
        f"select count(*), sum(cast(_c0 as int)) from (select _c0, rand(5) r from {KV_SQL}) where r < 0.25",
    ),
    # (2) HAVING before window functions beside a generator.
    ("gen_window_sum", [VIEW_D], "select k, explode(array(1,2)) x, sum(v) over () s from d group by k, v having k > 1"),
    (
        "gen_window_count",
        [VIEW_D],
        "select k, explode(array(1,2)) x, count(*) over () c from d group by k having k > 1",
    ),
    (
        "pipe_where_after_gen_window",
        [VIEW_D],
        "(select k, explode(array(1,2)) x, sum(v) over () s from d group by k, v) |> where k > 1",
    ),
    (
        "pipe_where_on_generated",
        [VIEW_D],
        "(select k, explode(array(1,2)) x, sum(v) over () s from d group by k, v) |> where x > 1",
    ),
    ("nogen_window_sum", [VIEW_D], "select k, sum(v) over (order by k) s from d group by k, v having k > 1 order by k"),
    # (3) a failing partition predicate raises; it does not keep or remove the partition.
    ("part_cast_fails", [PT], "select id from pt where cast(dep as int) = 1"),
    ("part_cast_fails_with_data_filter", [PT], "select id from pt where id > 0 and cast(dep as int) = 1"),
    ("part_try_cast_tolerates", [PT], "select id from pt where try_cast(dep as int) = 1"),
    ("part_cast_ok_numeric_values", [PN], "select id, dep from pn where cast(dep as int) = 1"),
    ("part_delete_cast_fails", [PT], "delete from pt where cast(dep as int) = 1"),
    # (4) each reference keeps its own options.
    (
        "opts_self_join_header_first",
        [],
        f"select a.name, b._c0 from {P_HDR} a join {P_RAW} b on true order by 1, 2",
    ),
    (
        "opts_self_join_header_second",
        [],
        f"select a._c0, b.name from {P_RAW} a join {P_HDR} b on true order by 1, 2",
    ),
    (
        "opts_union",
        [],
        f"select name from {P_HDR} union all select _c0 from {P_RAW} order by 1",
    ),
]


def sub(sql: str, tmp: str) -> str:
    return sql.replace("{kv}", f"{RES}/kv1.txt").replace("{people}", f"{RES}/people.csv").replace("{tmp}", tmp)


def run(remote: str) -> list[dict]:
    from pyspark.sql import SparkSession

    out = []
    for cid, setup, sql in CASES:
        # A fresh session per case; temporary views only.
        spark = SparkSession.builder.remote(remote).create()
        tmp = f"/tmp/p3-check3-{uuid.uuid4().hex[:12]}"
        try:
            spark.sql("SET spark.sql.ansi.enabled=true").collect()
            for t in ("pt", "pn"):  # tables live in the server's shared catalog: start from none
                spark.sql(f"drop table if exists {t}").collect()
            for stmt in (sub(x, tmp) for x in setup):
                spark.sql(stmt).collect()
            real = sub(sql, tmp)
            case = {"id": cid, "setup": setup, "sql": sql}
            try:
                rows = [[None if v is None else str(v) for v in r] for r in spark.sql(real).collect()]
                case["rows"] = (
                    rows if cid.endswith(("_first", "_second", "union")) or "order by" in sql else sorted(rows)
                )
            except Exception as e:
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
