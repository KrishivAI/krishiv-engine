#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record what Spark 4.2.0 does when a Connect Interrupt hits a SQL script (P2b row 15, R3).

Needs a Spark Connect server and the `pyspark-client==4.2.0` package on the host:

    docker run -d --name r3-probe-srv -p 15099:15002 apache/spark:4.2.0 /opt/spark/bin/spark-submit \
        --class org.apache.spark.sql.connect.service.SparkConnectServer --master 'local[2]' \
        /opt/spark/jars/spark-connect_2.13-4.2.0.jar
    python3 gen_scripting_interrupt.py sc://localhost:15099 > interrupt.out

Per case: the script runs on a thread through `spark.sql(...).collect()`; after a delay the main thread sends
interruptAll / interruptTag / interruptOperation; the outcome (error class, rows, side table) is printed.
A side table `r3_log(tag STRING)` records which leaves ran (INSERTs), so "a later leaf still runs" is observable.
"""

import sys
import threading
import time
import uuid

from pyspark.sql import SparkSession

URL = sys.argv[1] if len(sys.argv) > 1 else "sc://localhost:15099"
LOOP = "WHILE true DO SET i = i + 1; END WHILE;"
LONG = "SELECT count(*) FROM range(0, 100000000000) a JOIN range(0, 3) b ON a.id % 3 = b.id"
HANDLER = "DECLARE EXIT HANDLER FOR SQLEXCEPTION BEGIN INSERT INTO r3_log VALUES ('handler'); END;"


def script(body, handler):
    h = HANDLER if handler else ""
    return f"BEGIN DECLARE i INT DEFAULT 0; {h} {body} INSERT INTO r3_log VALUES ('after'); END"


def run(spark, name, sql, how, delay=4.0):
    spark.sql("DROP TABLE IF EXISTS r3_log")
    spark.sql("CREATE TABLE r3_log(tag STRING) USING parquet")
    tag = f"t{uuid.uuid4().hex[:8]}"
    out = {}

    def target():
        t0 = time.time()
        spark.addTag(tag)  # tags are thread-local in the client
        try:
            df = spark.sql(sql)
            out["rows"] = [tuple(r) for r in df.collect()]
        except BaseException as e:  # noqa: BLE001
            out["err"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            out["cls"] = getattr(e, "getCondition", lambda: None)() or getattr(e, "_errorClass", None)
        out["secs"] = round(time.time() - t0, 1)

    th = threading.Thread(target=target)
    th.start()
    time.sleep(delay)
    if how == "all":
        ids = spark.interruptAll()
    elif how == "tag":
        ids = spark.interruptTag(tag)
    else:
        raise ValueError(how)
    th.join(30)
    done = not th.is_alive()
    log = [r[0] for r in spark.sql("SELECT tag FROM r3_log").collect()] if done else "?"
    print(f"CASE {name} interrupt={how} returned_ids={len(ids)} finished={done}")
    print(f"  outcome={out}")
    print(f"  r3_log={log}")


def main():
    spark = SparkSession.builder.remote(URL).getOrCreate()
    spark.sql("SET spark.sql.ansi.enabled=true")
    for body_name, body in (("loop", LOOP), ("select", LONG + ";")):
        for handler in (False, True):
            for how in ("all", "tag"):
                run(spark, f"{body_name}-{'handler' if handler else 'nohandler'}", script(body, handler), how)
    print("SESSION still usable:", spark.sql("SELECT 1").collect())


main()
