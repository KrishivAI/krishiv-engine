#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record Spark 4.2.0 handler classes and table checksums for the SQL scripting goldens (P2-T196 unit 0).

Runs the 95 scripts of `tests/spark-golden/v4.2.0/inputs/scripting/randomly_generated_scripts.sql` on a REAL
Spark, each on fresh copies of the six tables (a new database per run):

  (a) as is: the result rows (the script's last result statement), the error if the script failed, and per table
      `count(*)`, `sum(hash(*))` ("hash") and the same sum over the non-TIMESTAMP columns ("stable_hash"; the
      scripts insert CURRENT_TIMESTAMP, so only `stable_hash` is comparable between runs);
  (b) with the script's `DECLARE EXIT HANDLER ...;` line removed: the error class Spark raises (the condition
      that would have fired the handler), its SQLSTATE, or "none" if the script then runs to the end.

Scripts that call RAND/RANDOM/UUID, or whose results differ between two identical runs, carry
`deterministic: false`; their rows and checksums are not comparable. Usage (from the repo root):

    mkdir -p /tmp/out && chmod 777 /tmp/out
    docker run --rm -v "$PWD/tests:/work:ro" -v /tmp/out:/out apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' --conf spark.sql.shuffle.partitions=2 \
        /work/scripts/gen_scripting_record.py /out/handler-record.json
    cp /tmp/out/handler-record.json tests/spark-golden/v4.2.0/scripting/handler-record.json

The JSON goes to the file named by the argument (the image logs to stdout, so stdout cannot carry it).
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession

SRC = "/work/spark-golden/v4.2.0/inputs/scripting/randomly_generated_scripts.sql"
HANDLER = re.compile(r"^[ \t]*DECLARE\s+EXIT\s+HANDLER\b.*;[ \t]*$", re.MULTILINE)
NONDET = re.compile(r"\b(RAND|RANDOM|UUID|SHUFFLE)\s*\(", re.IGNORECASE)


def load():
    text = Path(SRC).read_text(encoding="utf-8")
    setup, _, rest = text.partition("---- Scripts")
    body = "\n".join(ln for ln in setup.splitlines() if not ln.lstrip().startswith("--"))
    stmts = [s.strip() for s in re.split(r";[ \t]*\n", body + "\n") if s.strip()]
    tables = re.findall(r"^CREATE TABLE (\w+)", setup, re.MULTILINE)
    scripts = re.findall(r"--QUERY-DELIMITER-START\n(.*?)\n--QUERY-DELIMITER-END", rest, re.DOTALL)
    return stmts, tables, scripts


def jsonable(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return repr(v)
    return str(v)


def error_of(e):
    scope = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f-]{23}")
    cond = getattr(e, "getCondition", lambda: None)()
    if cond is not None:
        params = getattr(e, "getMessageParameters", lambda: None)() or {}
        sqlstate = e.getSqlState()
    else:
        jex: Any = getattr(e, "java_exception", None)
        try:
            cond, sqlstate, params = jex.getCondition(), jex.getSqlState(), dict(jex.getMessageParameters())
        except Exception:
            return {"class": type(e).__name__, "sqlstate": None, "message": str(e).split("\n")[0][:300]}
    return {
        "class": cond,
        "sqlstate": sqlstate,
        "params": {k: scope.sub("<scope>", str(v)) for k, v in sorted(params.items())},
    }


def stable(rec):
    """A run's record without the timestamp-dependent `hash` sums."""
    return {
        k: ({t: {c: v for c, v in d.items() if c != "hash"} for t, d in rec[k].items()} if k == "tables" else rec[k])
        for k in rec
    }


def checksums(spark, db, tables):
    out = {}
    for t in tables:
        cols = [f for f in spark.table(f"{db}.{t}").schema.fields if f.dataType.simpleString() != "timestamp"]
        names = ", ".join(f"`{f.name}`" for f in cols)
        r = spark.sql(f"SELECT count(*), sum(hash(*)), (SELECT sum(hash({names})) FROM {db}.{t}) FROM {db}.{t}").first()
        out[t] = {"count": r[0], "hash": r[1], "stable_hash": r[2]}
    return out


class Runner:
    def __init__(self, spark, setup, tables):
        self.spark, self.tables, self.n = spark, tables, 0
        spark.sql("CREATE DATABASE tmpl")
        spark.sql("USE tmpl")
        for s in setup:
            spark.sql(s)

    def fresh(self):
        self.n += 1
        db = f"r{self.n}"
        self.spark.sql(f"CREATE DATABASE {db}")
        for t in self.tables:
            self.spark.sql(f"CREATE TABLE {db}.{t} AS SELECT * FROM tmpl.{t}")
        self.spark.sql(f"USE {db}")
        return db

    def run(self, script):
        db = self.fresh()
        rec = {}
        try:
            df = self.spark.sql(script)
            rec["schema"] = df.schema.simpleString()
            rec["rows"] = [[jsonable(c) for c in row] for row in df.collect()]
        except Exception as e:
            rec["error"] = error_of(e)
        # The scratch database name differs per run: leave it out of what is compared and recorded.
        rec = json.loads(json.dumps(rec).replace(f"`{db}`", "`<db>`").replace(f"{db}.", "<db>."))
        rec["tables"] = checksums(self.spark, db, self.tables)
        return rec


def main():
    setup, tables, scripts = load()
    assert len(scripts) == 95, len(scripts)
    spark = SparkSession.builder.master("local[2]").appName("t196-record").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    runner = Runner(spark, setup, tables)
    out = []
    for i, script in enumerate(scripts, 1):
        assert len(HANDLER.findall(script)) == 1, i
        first = runner.run(script)
        again = runner.run(script)
        det = not NONDET.search(script) and stable(first) == stable(again)
        bare = runner.run(HANDLER.sub("", script))
        fired = bare.get("error", "none")
        out.append(
            {
                "script": i,
                "deterministic": det,
                "as_is": first,
                "fired": fired,
                "without_handler_tables": bare["tables"],
            }
        )
        print(f"script {i}: fired={fired if fired == 'none' else fired['class']} det={det}", file=sys.stderr)
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump({"spark": spark.version, "tables": tables, "scripts": out}, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
