#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, SIM115, B023
"""Follow-up Dataset.observe goldens (P3-T084, follow-up 2026-11-14). Real Spark 4.2.0 run, never transcribed.

Sections (each case records a plan description, params, and what Spark did):
  limit_counts  : observed row counts when a LIMIT is above / below / beside the observed node, over 1, 4, 16 partitions.
  ansi_sum      : ANSI on/off behaviour of sum/avg/try_sum/decimal overflow inside an observe metric.
  empty_shapes  : proven-empty shapes above/below the node: is the CollectMetrics pruned, what `get` returns.
  streaming     : observe on a streaming Dataset (Observation object and string name).
  value_types   : struct/map/array/interval/binary/decimal/date/timestamp metric values (classic python values;
                  Connect python values + the raw ObservedMetrics literal protos).

Each limit/ansi case is run REPS times with a fresh Observation and records the observed values of every run
plus `stable` (all runs equal). Plans are recorded as the ordered operator names of the executed plan.

Usage (JDK 21, pyspark==4.2.0), from the repo root:
    python -I tests/scripts/gen_observe_goldens_followup.py classic /tmp/f-classic.json
    python -I tests/scripts/gen_observe_goldens_followup.py connect /tmp/f-connect.json
    python -I tests/scripts/gen_observe_goldens_followup.py merge /tmp/f-classic.json /tmp/f-connect.json \\
        tests/fixtures/spark-4.2.0/observe-goldens-followup.json
"""

import contextlib
import json
import math
import re
import sys
import threading
import time

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_observe_goldens_followup.py"
MODE = sys.argv[1]
if MODE == "merge":
    c, n, out = (json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4])
    doc = {
        "format_version": FORMAT_VERSION,
        "generator": GENERATOR,
        "spark_version": c["spark_version"],
        "classic": c["cases"],
        "connect": n["cases"],
    }
    with open(out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    sys.exit(0)
OUT = sys.argv[2]
REPS = 3

from pyspark.sql import Observation, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

b = SparkSession.builder.config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2")
spark = b.remote("local[2]").getOrCreate() if MODE == "connect" else b.master("local[2]").getOrCreate()
CONNECT = MODE == "connect"
print("VERSION", spark.version, file=sys.stderr)

RAW = []  # raw ObservedMetrics protos captured in Connect mode
if CONNECT:
    from google.protobuf import json_format
    from pyspark.sql.connect.client.core import SparkConnectClient

    _orig = SparkConnectClient._verify_response_integrity

    def _cap(self, response):
        for om in getattr(response, "observed_metrics", None) or []:
            RAW.append(json_format.MessageToDict(om, preserving_proto_field_name=True))
        return _orig(self, response)

    SparkConnectClient._verify_response_integrity = _cap


def norm(v):
    from decimal import Decimal

    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if hasattr(v, "asDict") and isinstance(v, tuple):  # Row (a tuple subclass)
        return {"py_type": "Row", "fields": {k: norm(x) for k, x in v.asDict(recursive=False).items()}}
    if isinstance(v, (bytes, bytearray)):
        return {"py_type": type(v).__name__, "hex": bytes(v).hex()}
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {"py_type": "dict", "items": [[norm(k), norm(x)] for k, x in v.items()]}
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, Decimal):
        return {"py_type": "Decimal", "repr": str(v)}
    return {"py_type": type(v).__name__, "repr": str(v)}


def err(e):
    d = {"python_class": type(e).__name__}
    with contextlib.suppress(Exception):
        d["errorClass"] = e.getErrorClass()
    with contextlib.suppress(Exception):
        d["sqlState"] = e.getSqlState()
    with contextlib.suppress(Exception):
        d["messageParameters"] = {k: str(v) for k, v in (e.getMessageParameters() or {}).items()}
    m = str(e)
    m = re.sub(r"\n+== (DataFrame|SQL).*", "", m, flags=re.S)
    d["message"] = m.split("\n\n")[0].split("\n\tat ")[0][:500]
    return d


def with_timeout(f, secs=15):
    box = {}

    def run():
        try:
            box["v"] = f()
        except BaseException as e:
            box["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(secs)
    if t.is_alive():
        return ("blocked", None)
    return ("error", box["e"]) if "e" in box else ("ok", box["v"])


def getres(o, secs=10):
    st, v = with_timeout(lambda: o.get, secs)
    if st == "ok":
        return {
            "get": "ok",
            "metrics": {k: norm(x) for k, x in v.items()},
            "types": {k: type(x).__name__ for k, x in v.items()},
        }
    if st == "blocked":
        return {"get": "blocked"}
    return {"get": "error", **err(v)}


def plan_nodes(d):
    """Operator names (executed plan, after execution) in tree order; classic only."""
    if CONNECT:
        return None
    s = d._jdf.queryExecution().executedPlan().toString()
    out = []
    for line in s.splitlines():
        m = re.match(r"^[\s:+\-|*()0-9]*([A-Z][A-Za-z]+)", line)
        if m and m.group(1) not in ("Output", "Arguments"):
            out.append(m.group(1))
    return out


CASES = {}
DONE = {"classic": None, "connect": None}


def reg(section, name, desc, params, result):
    CASES.setdefault(section, {})[name] = {"plan": desc, "params": params, **result}
    print("done", section, name, file=sys.stderr)


def safe(section, name, desc, params, f):
    try:
        r = f()
    except BaseException as e:
        r = {"outcome": "error", **err(e)}
    reg(section, name, desc, params, r)


spark.range(0).count()

# ---------------------------------------------------------------- 1. limit counts


def rng(p, n=1000, name="v"):
    return spark.range(0, n, 1, p).toDF(name)


def small(p):
    return spark.range(0, 20, 1, p).toDF("v")


def obs_run(build, action, reps=REPS, plan_of=lambda d: d):
    """build(o1, o2) -> (df_to_act_on, [observations]); action(df) -> value. Fresh observations each rep."""
    runs = []
    plan = None
    for _ in range(reps):
        os_ = [Observation(), Observation()]
        df = build(*os_)
        try:
            res = norm(action(df))
        except BaseException as e:
            res = {"error": err(e)}
        runs.append({"action": res, "observed": [getres(o, 6) for o in os_ if _registered(o)]})
        if plan is None:
            plan = plan_nodes(plan_of(df)) if not CONNECT else None
    metr = [json.dumps([r["observed"]], sort_keys=True) for r in runs]
    return {"runs": runs, "stable": len(set(metr)) == 1, "plan_nodes": plan}


def _registered(o):
    return (o._jo is not None) if not CONNECT else (o._name is not None)


def _count_plan_df(df):
    return df.groupBy().count()


def action_count(d):
    return d.count()


def action_collect_len(d):
    return len(d.collect())


# (name, description, builder, actions{name: fn}) -- builder takes (p, n, o1, o2)
def mk_cases():
    N = [5, 100]
    cases = []

    def add(name, desc, build):
        cases.append((name, desc, build))

    add(
        "observe_limit",
        "range(0,1000,1,P).observe(n=count(1)).limit(n)",
        lambda p, n, o1, o2: rng(p).observe(o1, F.count(F.lit(1)).alias("n")).limit(n),
    )
    add(
        "observe_filter_limit",
        "range(0,1000,1,P).observe(n=count(1)).filter(v%2=0).limit(n)",
        lambda p, n, o1, o2: rng(p).observe(o1, F.count(F.lit(1)).alias("n")).where("v % 2 = 0").limit(n),
    )
    add(
        "observe_selective_filter_limit",
        "range(0,1000,1,P).observe(n).filter(v%100=0).limit(n): selective filter, ~10 matches",
        lambda p, n, o1, o2: rng(p).observe(o1, F.count(F.lit(1)).alias("n")).where("v % 100 = 0").limit(n),
    )
    add(
        "limit_then_observe",
        "range(0,1000,1,P).limit(n).observe(n=count(1))  (limit below the node)",
        lambda p, n, o1, o2: rng(p).limit(n).observe(o1, F.count(F.lit(1)).alias("n")),
    )
    add(
        "observe_sort_limit",
        "range(0,1000,1,P).observe(n).orderBy(v desc).limit(n)  (TakeOrderedAndProject above the node)",
        lambda p, n, o1, o2: rng(p).observe(o1, F.count(F.lit(1)).alias("n")).orderBy(F.col("v").desc()).limit(n),
    )
    add(
        "sort_observe_limit",
        "range(0,1000,1,P).orderBy(v desc).observe(n).limit(n)  (sort below the node)",
        lambda p, n, o1, o2: rng(p).orderBy(F.col("v").desc()).observe(o1, F.count(F.lit(1)).alias("n")).limit(n),
    )
    add(
        "observe_offset_limit",
        "range(0,1000,1,P).observe(n).offset(3).limit(n)",
        lambda p, n, o1, o2: rng(p).observe(o1, F.count(F.lit(1)).alias("n")).offset(3).limit(n),
    )
    add(
        "join_observe_left_limit",
        "observe(range(0,1000,1,P)) JOIN range(0,1000,1,P) ON v (broadcast build side = the other side), limit(n) above",
        lambda p, n, o1, o2: rng(p).observe(o1, F.count(F.lit(1)).alias("n")).join(rng(p), "v").limit(n),
    )
    add(
        "join_observe_right_limit",
        "range(0,1000,1,P) JOIN observe(range(0,1000,1,P)) ON v (observed side is the broadcast/build side), limit(n) above",
        lambda p, n, o1, o2: rng(p).join(rng(p).observe(o1, F.count(F.lit(1)).alias("n")), "v").limit(n),
    )
    add(
        "join_observe_left_shuffle_limit",
        "as join_observe_left_limit but autoBroadcastJoinThreshold=-1 and preferSortMergeJoin (shuffled join), limit(n) above",
        lambda p, n, o1, o2: rng(p).observe(o1, F.count(F.lit(1)).alias("n")).join(rng(p).hint("merge"), "v").limit(n),
    )
    add(
        "union_of_limit_branches",
        "observe(A:range P).limit(n) UNION ALL observe(B:range P).limit(n), each branch observed below its own limit",
        lambda p, n, o1, o2: (
            rng(p)
            .observe(o1, F.count(F.lit(1)).alias("n"))
            .limit(n)
            .union(rng(p).observe(o2, F.count(F.lit(1)).alias("n")).limit(n))
        ),
    )
    add(
        "union_of_limit_branches_then_observe",
        "(range P.limit(n) UNION ALL range P.limit(n)).observe(n)  (observe above the union of limits)",
        lambda p, n, o1, o2: rng(p).limit(n).union(rng(p).limit(n)).observe(o1, F.count(F.lit(1)).alias("n")),
    )
    add(
        "observe_union_limit",
        "(observe(A).union(range P)).limit(n)  (limit above a union whose first branch is observed)",
        lambda p, n, o1, o2: rng(p).observe(o1, F.count(F.lit(1)).alias("n")).union(rng(p)).limit(n),
    )
    return N, cases


N_LIST, CASE_LIST = mk_cases()
ACTIONS = {"count": action_count, "collect": action_collect_len}

if MODE == "classic" or CONNECT:
    for name, desc, build in CASE_LIST:
        for p in (1, 4, 16):
            for n in N_LIST:
                for an, af in ACTIONS.items():
                    key = f"{name}/P{p}/n{n}/{an}"

                    def f(build=build, p=p, n=n, af=af, an=an):
                        if name == "join_observe_left_shuffle_limit":
                            spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
                            spark.conf.set("spark.sql.join.preferSortMergeJoin", "true")
                        try:
                            r = obs_run(
                                lambda o1, o2: build(p, n, o1, o2),
                                af,
                                plan_of=(lambda d: d.groupBy().count()) if an == "count" else (lambda d: d),
                            )
                        finally:
                            spark.conf.unset("spark.sql.autoBroadcastJoinThreshold")
                            spark.conf.unset("spark.sql.join.preferSortMergeJoin")
                        return r

                    safe("limit_counts", key, desc, {"partitions": p, "limit": n, "action": an}, f)

# ---------------------------------------------------------------- 2. ANSI sum overflow
MAXL = 9223372036854775807


def ansi_case(name, desc, params, df_fn, exprs_fn, ansi, reps=5, build_ansi=None, action="count"):
    def f():
        prev = spark.conf.get("spark.sql.ansi.enabled")
        runs = []
        try:
            for _ in range(reps):
                spark.conf.set("spark.sql.ansi.enabled", build_ansi if build_ansi is not None else ansi)
                exprs = exprs_fn()  # expression objects are built under build_ansi (or ansi)
                o = Observation()
                d = df_fn().observe(o, *exprs)
                spark.conf.set("spark.sql.ansi.enabled", ansi)  # run under `ansi`
                try:
                    r = norm(d.count() if action == "count" else len(d.collect()))
                except BaseException as e:
                    r = {"error": err(e)}
                runs.append({"action": r, **getres(o, 6)})
        finally:
            spark.conf.set("spark.sql.ansi.enabled", prev)
        distinct = sorted({json.dumps(x, sort_keys=True) for x in runs})
        return {"runs": runs, "distinct_runs": len(distinct), "stable": len(distinct) == 1}

    safe("ansi_sum", name, desc, params, f)


def rbig(p):
    # two rows: MAX-1 and MAX; sum overflows. p=1: both in one task; p=2: one per task.
    return lambda: spark.range(0, 2, 1, p).select((F.col("id") + (MAXL - 1)).alias("v"))


def rint(p):
    return lambda: spark.range(0, 3, 1, p).select(F.lit(2147483647).cast("int").alias("v"))


def rdec(p):
    return lambda: spark.range(0, 2, 1, p).select(
        F.expr("cast(99999999999999999999999999999999999999 as decimal(38,0))").alias("v")
    )


for ansi in ("true", "false"):
    for p in (1, 2):
        tag = f"ansi_{ansi}/P{p}"
        ansi_case(
            f"sum_bigint_overflow/{tag}",
            f"observe(sum(v)) over bigint rows MAX-1 and MAX, {p} partition(s), count() action",
            {"ansi": ansi, "partitions": p},
            rbig(p),
            lambda: [F.sum("v").alias("s")],
            ansi,
        )
        ansi_case(
            f"avg_bigint/{tag}",
            f"observe(avg(v)) over bigint rows MAX-1 and MAX, {p} partition(s)",
            {"ansi": ansi, "partitions": p},
            rbig(p),
            lambda: [F.avg("v").alias("a")],
            ansi,
        )
        ansi_case(
            f"sum_int_no_overflow/{tag}",
            f"observe(sum(v)) over int rows 3 x 2147483647 (result bigint), {p} partition(s)",
            {"ansi": ansi, "partitions": p},
            rint(p),
            lambda: [F.sum("v").alias("s"), F.avg("v").alias("a")],
            ansi,
        )
        ansi_case(
            f"sum_decimal_overflow/{tag}",
            f"observe(sum(v)) over 2 x decimal(38,0) 10^38-1, {p} partition(s)",
            {"ansi": ansi, "partitions": p},
            rdec(p),
            lambda: [F.sum("v").alias("s")],
            ansi,
        )
        ansi_case(
            f"avg_decimal_overflow/{tag}",
            f"observe(avg(v)) over 2 x decimal(38,0) 10^38-1, {p} partition(s)",
            {"ansi": ansi, "partitions": p},
            rdec(p),
            lambda: [F.avg("v").alias("a")],
            ansi,
        )
        ansi_case(
            f"try_sum_bigint_overflow/{tag}",
            f"observe(try_sum(v)) over bigint MAX-1, MAX, {p} partition(s)",
            {"ansi": ansi, "partitions": p},
            rbig(p),
            lambda: [F.try_sum("v").alias("s")],
            ansi,
        )
        ansi_case(
            f"add_in_metric_overflow/{tag}",
            f"observe(max(v)+1 ) over bigint MAX, {p} partition(s): non-aggregate Add overflows",
            {"ansi": ansi, "partitions": p},
            rbig(p),
            lambda: [(F.max("v") + 2).alias("p")],
            ansi,
        )

# EvalMode is baked into the aggregate expression when the Column is built, not read at action time.
ansi_case(
    "evalmode_built_off_run_on/P1",
    "Column built with ansi=false, action runs with ansi=true",
    {"build_ansi": "false", "run_ansi": "true", "partitions": 1},
    rbig(1),
    lambda: [F.sum("v").alias("s")],
    "true",
    build_ansi="false",
)
ansi_case(
    "evalmode_built_on_run_off/P1",
    "Column built with ansi=true, action runs with ansi=false",
    {"build_ansi": "true", "run_ansi": "false", "partitions": 1},
    rbig(1),
    lambda: [F.sum("v").alias("s")],
    "false",
    build_ansi="true",
)
ansi_case(
    "evalmode_sql_string_built_off_run_on/P1",
    "F.expr('sum(v)') built with ansi=false, action runs with ansi=true",
    {"build_ansi": "false", "run_ansi": "true", "partitions": 1},
    rbig(1),
    lambda: [F.expr("sum(v)").alias("s")],
    "true",
    build_ansi="false",
)
# Same overflow but the action itself also runs the aggregate: shows the action does not hide the metric failure
ansi_case(
    "sum_bigint_overflow_collect/ansi_true/P2",
    "as sum_bigint_overflow P2 but collect() action",
    {"ansi": "true", "partitions": 2, "action": "collect"},
    rbig(2),
    lambda: [F.sum("v").alias("s")],
    "true",
    action="collect",
)

# ---------------------------------------------------------------- 3. proven-empty shapes


def empty_case(name, desc, build, action=lambda d: d.count(), classic_plan=True):
    def f():
        o = Observation()
        d = build(o)
        try:
            res = norm(action(d))
        except BaseException as e:
            res = {"error": err(e)}
        out = {"action": res, **getres(o, 8)}
        if not CONNECT:
            qe = d._jdf.queryExecution()
            out["optimized_has_CollectMetrics"] = "CollectMetrics" in qe.optimizedPlan().toString()
            out["executed_has_CollectMetrics"] = "CollectMetrics" in qe.executedPlan().toString()
            out["optimized_root"] = qe.optimizedPlan().toString().split("\n")[0][:120]
            with contextlib.suppress(Exception):
                out["jvm_get_row"] = str(o._jo.getRow()).replace("\n", " ")[:120]
        else:
            out["connect_flows"] = (
                [[n, norm_flow(m)] for n, m in d.executionInfo.flows] if getattr(d, "executionInfo", None) else None
            )
        return out

    safe("empty_shapes", name, desc, {}, f)


def norm_flow(m):
    return {k: norm(v) for k, v in m.items()}


def base3(p=2):
    return spark.range(0, 6, 1, p).toDF("v")


def OBS(o, df=None):
    return (df if df is not None else base3()).observe(o, F.count(F.lit(1)).alias("n"))


empty_case("where_lit_false_above", "observe(range).where(lit(False))", lambda o: OBS(o).where(F.lit(False)))
empty_case("where_false_sql_above", "observe(range).where('false')", lambda o: OBS(o).where("false"))
empty_case(
    "where_lit_null_above",
    "observe(range).where(lit(None).cast(boolean))",
    lambda o: OBS(o).where(F.lit(None).cast("boolean")),
)
empty_case("where_1_eq_0_above", "observe(range).where('1 = 0')", lambda o: OBS(o).where("1 = 0"))
empty_case(
    "where_contradiction_above",
    "observe(range).where(v > 5 AND v < 0)  (not constant-foldable: data-dependent)",
    lambda o: OBS(o).where("v > 5 and v < 0"),
)
empty_case(
    "where_v_null_and_notnull_above",
    "observe(range).where(v IS NULL AND v IS NOT NULL)",
    lambda o: OBS(o).where("v is null and v is not null"),
)
empty_case(
    "where_lit_false_below",
    "range.where(lit(False)).observe  (empty below the node)",
    lambda o: OBS(o, base3().where(F.lit(False))),
)
empty_case("limit0_above", "observe(range).limit(0)", lambda o: OBS(o).limit(0))
empty_case("limit0_below", "range.limit(0).observe(n)", lambda o: OBS(o, base3().limit(0)))
empty_case("filter_false_then_limit", "observe(range).where(false).limit(5)", lambda o: OBS(o).where("false").limit(5))
empty_case(
    "two_step_filter_false", "observe(range).where(v>1).where(false)", lambda o: OBS(o).where("v > 1").where("false")
)
empty_case(
    "project_where_false_above",
    "observe(range).select(v+1).where(false)",
    lambda o: OBS(o).select((F.col("v") + 1).alias("w")).where("false"),
)
empty_case(
    "sort_where_false_above", "observe(range).orderBy(v).where(false)", lambda o: OBS(o).orderBy("v").where("false")
)
empty_case(
    "groupby_where_false_above",
    "observe(range).where(false).groupBy(v).count()",
    lambda o: OBS(o).where("false").groupBy("v").count(),
)
empty_case(
    "global_agg_where_false_above",
    "observe(range).where(false).agg(count(1))  (global aggregate keeps one row)",
    lambda o: OBS(o).where("false").agg(F.count(F.lit(1)).alias("c")),
    action=lambda d: d.collect(),
)
empty_case(
    "inner_join_empty_local",
    "observe(range) JOIN (empty LocalRelation) ON v (inner join with empty side)",
    lambda o: OBS(o).join(spark.range(0, 3).where("false").toDF("v"), "v"),
)
empty_case(
    "inner_join_filtered_empty",
    "observe(range) JOIN range.where(false)",
    lambda o: OBS(o).join(base3().where("false"), "v"),
)
empty_case(
    "left_join_empty_right",
    "observe(range) LEFT JOIN range.where(false)  (left side must survive)",
    lambda o: OBS(o).join(base3().where("false"), "v", "left"),
)
empty_case(
    "right_join_empty_left_observed",
    "range.where(false) RIGHT JOIN observe(range)",
    lambda o: base3().where("false").join(OBS(o), "v", "right"),
)
empty_case(
    "left_semi_empty_right",
    "observe(range) LEFT SEMI JOIN range.where(false)",
    lambda o: OBS(o).join(base3().where("false"), "v", "left_semi"),
)
empty_case(
    "left_anti_empty_right",
    "observe(range) LEFT ANTI JOIN range.where(false)  (anti with empty right = left)",
    lambda o: OBS(o).join(base3().where("false"), "v", "left_anti"),
)
empty_case(
    "union_with_empty", "observe(range) UNION ALL range.where(false)", lambda o: OBS(o).union(base3().where("false"))
)
empty_case(
    "union_empty_first", "range.where(false) UNION ALL observe(range)", lambda o: base3().where("false").union(OBS(o))
)
empty_case(
    "union_both_empty_above",
    "(observe(range).where(false)) UNION ALL range.where(false)",
    lambda o: OBS(o).where("false").union(base3().where("false")),
)
empty_case(
    "intersect_empty", "observe(range) INTERSECT range.where(false)", lambda o: OBS(o).intersect(base3().where("false"))
)
empty_case(
    "except_empty_right", "observe(range) EXCEPT range.where(false)", lambda o: OBS(o).exceptAll(base3().where("false"))
)
empty_case(
    "sample_fraction0", "observe(range).sample(0.0)  (not proven empty at plan time)", lambda o: OBS(o).sample(0.0, 1)
)
empty_case(
    "collect_empty_pruned",
    "observe(range).where(false) with collect()",
    lambda o: OBS(o).where("false"),
    action=lambda d: d.collect(),
)
empty_case("cache_then_where_false", "observe(range).cache().where(false)", lambda o: OBS(o).cache().where("false"))


# first action pruned, second action not: classic first-wins vs connect last-wins interplay
def _two():
    o = Observation()
    d = OBS(o)
    r1 = d.where("false").count()
    g1 = getres(o, 6)
    r2 = d.count()
    g2 = getres(o, 6)
    return {"first_action_pruned": r1, "after_first": g1, "second_action": r2, "after_second": g2}


safe(
    "empty_shapes",
    "first_action_pruned_then_normal",
    "d=observe(range); d.where(false).count() then d.count() same Observation",
    {},
    _two,
)

# ---------------------------------------------------------------- 4. streaming


def stream_df():
    return spark.readStream.format("rate").option("rowsPerSecond", 5).load()


safe(
    "streaming",
    "observation_object",
    "rate stream .observe(Observation(), count)",
    {"api": "observe(Observation, ...)"},
    lambda: (
        {"outcome": "ok_unexpected"}
        if stream_df().observe(Observation("m"), F.count(F.lit(1)).alias("c")) is not None
        else {}
    ),
)


def _stream_action():
    d = stream_df().observe(Observation("m2"), F.count(F.lit(1)).alias("c"))
    q = d.writeStream.format("memory").queryName("obs_stream_q").start()
    q.stop()
    return {"outcome": "ok_unexpected"}


safe(
    "streaming",
    "observation_object_then_start",
    "rate stream .observe(Observation) then writeStream.start",
    {"api": "observe(Observation, ...)"},
    _stream_action,
)


def _stream_string():
    d = stream_df().observe("sm", F.count(F.lit(1)).alias("c"), F.max("value").alias("mx"))
    q = d.writeStream.format("memory").queryName("obs_stream_s").trigger(processingTime="1 second").start()
    time.sleep(6)
    prog = [p for p in q.recentProgress if p.get("observedMetrics")]
    q.stop()
    return {
        "outcome": "ok",
        "has_observedMetrics_in_progress": bool(prog),
        "observedMetrics_keys": sorted(prog[-1]["observedMetrics"].keys()) if prog else None,
        "metric_names": sorted(prog[-1]["observedMetrics"]["sm"].asDict().keys())
        if prog and "sm" in prog[-1]["observedMetrics"]
        else None,
        "note": "per-microbatch values are scheduling-dependent and not recorded",
    }


safe(
    "streaming",
    "string_name_observe_runs",
    "rate stream .observe('sm', count, max) writeStream memory 6s; metrics in StreamingQueryProgress.observedMetrics",
    {"api": "observe(str, ...)"},
    _stream_string,
)

# ---------------------------------------------------------------- 5. value types


def types_case(name, desc, expr_sql):
    def f():
        o = Observation()
        d = spark.sql(
            "select * from values (1,'a',cast('a' as binary),1.5),(2,'b',X'FF00FE',null) as t(i,k,bin,x)"
        ).observe(o, *[F.expr(s).alias(a) for a, s in expr_sql])
        res = {"action": norm(d.count()), **getres(o, 8)}
        if CONNECT:
            res["connect_raw_observed_metrics"] = RAW[-1:] if RAW else None
        return res

    safe("value_types", name, desc, {"exprs": dict(expr_sql)}, f)


T = [
    (
        "struct",
        "struct(max(i) AS mx, min(i) AS mn, collect_list(k) AS l)",
        [("v", "struct(max(i) as mx, min(i) as mn, collect_list(k) as l)")],
    ),
    (
        "struct_with_null_field",
        "struct(max(x), cast(null as string))",
        [("v", "struct(max(x) as mx, cast(null as string) as ns)")],
    ),
    (
        "struct_nested",
        "struct(struct(max(i)) AS inner, array(min(i)) AS arr)",
        [("v", "named_struct('inner', named_struct('a', max(i)), 'arr', array(min(i)))")],
    ),
    ("map_string_int", "map('a', max(i), 'b', min(i))", [("v", "map('a', max(i), 'b', min(i))")]),
    (
        "map_with_null_value",
        "map('a', max(x), 'b', cast(null as double))",
        [("v", "map('a', cast(null as double), 'b', max(x))")],
    ),
    ("map_int_key", "map(1, max(k))", [("v", "map(1, max(k))")]),
    (
        "map_agg",
        "map_from_arrays(collect_list(k), collect_list(i))",
        [("v", "map_from_arrays(collect_list(k), collect_list(i))")],
    ),
    (
        "map_empty",
        "map_from_arrays(collect_list(k), collect_list(i)) over zero rows",
        [("v", "map_from_arrays(collect_list(k), collect_list(i))")],
    ),
    ("array_int", "collect_list(i)", [("v", "collect_list(i)")]),
    (
        "array_with_null",
        "collect_list(x) (nulls skipped) and array(max(x), null)",
        [("a", "collect_list(x)"), ("b", "array(max(x), cast(null as double))")],
    ),
    ("array_of_struct", "collect_list(struct(i, k))", [("v", "collect_list(struct(i, k))")]),
    (
        "array_nan",
        "array(cast('NaN' as double), cast(-0.0 as double), max(x))",
        [("v", "array(cast('NaN' as double), cast('-Infinity' as double), max(x))")],
    ),
    ("array_empty", "collect_list(i) over zero rows", [("v", "collect_list(i)")]),
    ("binary_max", "max(bin)  (non-UTF-8 bytes)", [("v", "max(bin)")]),
    ("binary_first_lit", "first(X'FF00FE')", [("v", "first(X'FF00FE')")]),
    ("binary_collect", "collect_list(bin)", [("v", "collect_list(bin)")]),
    (
        "interval_daytime",
        "max(make_dt_interval(0, 0, 0, i))  (INTERVAL DAY TO SECOND)",
        [("v", "max(make_dt_interval(i, 0, 0, 0))")],
    ),
    ("interval_daytime_sum", "sum(make_dt_interval(i,1,2,3.5))", [("v", "sum(make_dt_interval(i, 1, 2, 3.5))")]),
    (
        "interval_yearmonth",
        "max(make_ym_interval(0, i))  (INTERVAL YEAR TO MONTH)",
        [("v", "max(make_ym_interval(0, i))")],
    ),
    ("interval_yearmonth_sum", "sum(make_ym_interval(1, i))", [("v", "sum(make_ym_interval(1, i))")]),
    (
        "interval_calendar",
        "first(make_interval(0, 0, 0, i, 1, 2, 3.5))  (CalendarIntervalType)",
        [("v", "first(make_interval(0, 0, 0, i, 1, 2, 3.5))")],
    ),
    (
        "interval_literal_dt",
        "first(INTERVAL '1 02:03:04.5' DAY TO SECOND)",
        [("v", "first(INTERVAL '1 02:03:04.5' DAY TO SECOND)")],
    ),
    ("decimal", "sum(cast(i as decimal(10,2)))", [("v", "sum(cast(i as decimal(10,2)))")]),
    ("date", "max(date'2024-02-29' + i)", [("v", "max(date'2024-02-29' + i)")]),
    (
        "timestamp",
        "max(timestamp'2024-03-10 01:59:59Z' + make_dt_interval(0,0,0,i))",
        [("v", "max(timestamp'2024-03-10 01:59:59Z' + make_dt_interval(0,0,0,i))")],
    ),
    (
        "timestamp_ntz",
        "max(timestamp_ntz'2024-03-10 01:59:59.123456' )",
        [("v", "max(timestamp_ntz'2024-03-10 01:59:59.123456')")],
    ),
    (
        "long_max",
        "max(cast(9223372036854775807 as bigint)) and min(-9223372036854775808)",
        [("a", "max(cast(9223372036854775807 as bigint))"), ("b", "min(cast(-9223372036854775808 as bigint))")],
    ),
    (
        "boolean_null_string",
        "bool_and(i > 0), max(cast(null as string)), first(k)",
        [("a", "bool_and(i > 0)"), ("b", "max(cast(null as string))"), ("c", "first(k)")],
    ),
    (
        "float_special",
        "max(cast('NaN' as float)), min(cast('-Infinity' as double))",
        [("a", "max(cast('NaN' as float))"), ("b", "min(cast('-Infinity' as double))")],
    ),
    ("unicode", "max('日本語 é 😀')", [("v", "max('日本語 é 😀')")]),
]
for nm, desc, ex in T:
    if nm in ("map_empty", "array_empty"):

        def f(ex=ex):
            o = Observation()
            d = spark.sql("select 1 as i, 'a' as k where 1=0").observe(o, *[F.expr(s).alias(a) for a, s in ex])
            res = {"action": norm(d.count()), **getres(o, 8)}
            if CONNECT:
                res["connect_raw_observed_metrics"] = RAW[-1:] if RAW else None
            return res

        safe("value_types", nm, desc + " (zero input rows reach the node)", {"exprs": dict(ex)}, f)
    else:
        types_case(nm, desc, ex)

# ---------------------------------------------------------------- output
with open(OUT, "w") as fh:
    json.dump({"spark_version": spark.version, "cases": CASES}, fh, indent=1, sort_keys=True, default=str)
    fh.write("\n")
print("WROTE", OUT, file=sys.stderr)
