#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record Spark 4.2.0 SQL scripting behaviour for the Krishiv-authored goldens (P2-T196 unit 0).

Reads `*.sql` files of cases and runs every case on a REAL Spark in a fresh database holding an empty scratch
table `s(i INT)`. A file is a list of cases:

    -- CASE <name>            starts a case
    -- next                   separates the statements of a case (each runs through `spark.sql`, in order)
    -- KRISHIV-EXPECT: <...>  documented divergence: Krishiv's expected outcome, recorded beside Spark's
    -- <anything else>        comment, kept in the input only

Per statement the output records the schema and rows, or the error (class, SQLSTATE, parameters with Spark's
random scope ids replaced by `<scope>`; a non-Spark Java failure is recorded by its Java class). Session
configuration is reset (`RESET`) after every case. The phase of a structural error is read from the case's own
INSERT: the case runs `BEGIN INSERT INTO s ...; <offending construct> END`, then `SELECT count(*) FROM s`.

Usage (from the repo root; `out` must be writable by the container user):

    mkdir -p /tmp/out && chmod 777 /tmp/out
    docker run --rm -v "$PWD/tests:/work:ro" -v /tmp/out:/out apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' --conf spark.sql.shuffle.partitions=2 \
        /work/scripts/gen_scripting_semantics.py /work/spark-golden/v4.2.0/scripting/semantics /out
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pyspark.sql import SparkSession

UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


# The scratch database of the case being run: only its own name is normalised, never a user identifier
# that happens to look like one (a column `c1`).
SCRATCH_DB = ""


def norm(v):
    text = UUID.sub("<scope>", str(v))
    return text.replace(f"`{SCRATCH_DB}`", "`<db>`") if SCRATCH_DB else text


def jsonable(v):
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return repr(v)
    return norm(v)


def error_of(e):
    cond = getattr(e, "getCondition", lambda: None)()
    if cond is not None:
        params = getattr(e, "getMessageParameters", lambda: None)() or {}
        return {
            "class": cond,
            "sqlstate": e.getSqlState(),
            "params": {k: norm(v) for k, v in sorted(params.items())},
        }
    jex = getattr(e, "java_exception", None)
    if jex is not None:
        try:
            params = dict(jex.getMessageParameters())
            return {
                "class": jex.getCondition(),
                "sqlstate": jex.getSqlState(),
                "params": {k: norm(params[k]) for k in sorted(params)},
                "exception": jex.getClass().getSimpleName(),
            }
        except Exception:
            return {"java": jex.getClass().getName(), "message": norm(jex.getMessage())[:200]}
    return {"python": type(e).__name__, "message": norm(e).split("\n")[0][:200]}


@dataclass
class Case:
    name: str
    statements: list[str] = field(default_factory=list)
    expect: list[str] = field(default_factory=list)


def parse(path: Path) -> list[Case]:
    cases: list[Case] = []
    cur: Case | None = None
    stmt: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"-- CASE (\S+)", line)
        if m:
            if cur:
                cur.statements.append("\n".join(stmt).strip())
            cur = Case(m.group(1))
            stmt = []
            cases.append(cur)
        elif cur is None:
            continue
        elif line.strip() == "-- next":
            cur.statements.append("\n".join(stmt).strip())
            stmt = []
        elif line.startswith("-- KRISHIV-EXPECT:"):
            cur.expect.append(line.split(":", 1)[1].strip())
        elif line.startswith("--"):
            continue
        else:
            stmt.append(line)
    if cur:
        cur.statements.append("\n".join(stmt).strip())
    for c in cases:
        c.statements = [s for s in c.statements if s]
    return cases


def run_case(spark, n, case):
    global SCRATCH_DB
    db = f"c{n}"
    SCRATCH_DB = db
    spark.sql(f"CREATE DATABASE {db}")
    spark.sql(f"USE {db}")
    spark.sql("CREATE TABLE s (i INT)")
    out = []
    for q in case.statements:
        rec = {"query": q}
        try:
            df = spark.sql(q)
            rec["schema"] = norm(df.schema.simpleString())
            rec["rows"] = [[jsonable(c) for c in row] for row in df.collect()]
        except Exception as e:
            rec["error"] = error_of(e)
        out.append(rec)
    spark.sql("RESET")
    res = {"name": case.name, "statements": out}
    if case.expect:
        res["divergence"] = {"krishiv_expected": case.expect, "spark_actual_not_expected": out[-2:-1] or out}
    return res


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    spark = SparkSession.builder.master("local[2]").appName("t196-semantics").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    n = 0
    for f in sorted(src.glob("*.sql")):
        cases = []
        for case in parse(f):
            n += 1
            cases.append(run_case(spark, n, case))
        doc = {"spark": spark.version, "file": f.name, "cases": cases}
        (dst / (f.stem + ".json")).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{f.name}: {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
