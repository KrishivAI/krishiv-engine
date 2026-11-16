#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the Dataset.observe / Observation golden fixture (P3-T084).

Every expectation comes from a REAL Spark 4.2.0 run, never from docs or memory. Each case builds a tiny
DataFrame, attaches an `Observation`, runs an action and records what `Observation.get` returned (values,
Python value types), or the error class / SQLSTATE / message parameters Spark raised. Two client modes are
recorded because they behave differently (see research/t084-observe-semantics.md in krishiv-internal):

  classic  : PySpark over the JVM (local[2]); Observation is the JVM `org.apache.spark.sql.Observation`.
  connect  : PySpark Spark Connect client against a local Connect server (`remote("local[2]")`).

`get` is called through a 15 s watchdog: `{"get": "blocked"}` means it did not return (it blocks until an
action finishes). Values: ints as ints, doubles as Python `repr` text (`nan`, `inf`), decimals as
`{"py_type": "Decimal", "repr": ...}`. Plan trees and random seeds are stripped from messages; the output
contains no local paths. Cases listed in NONDETERMINISTIC record only the shape, not values.

Usage (from repo root; needs JDK 21 and `pyspark==4.2.0`):

    ops/scripts/spark-local.sh tests/scripts/gen_observe_goldens.py classic /tmp/observe-classic.json
    uv run --no-project --python 3.12 --with 'pyspark[connect]==4.2.0' --with pandas --with pyarrow \\
      python -I tests/scripts/gen_observe_goldens.py connect /tmp/observe-connect.json
    python3 tests/scripts/merge_observe_goldens.py /tmp/observe-classic.json /tmp/observe-connect.json \\
      tests/fixtures/spark-4.2.0/observe-goldens.json
"""

import contextlib
import json
import math
import re
import sys
import threading
from pathlib import Path

from pyspark.sql import Observation, SparkSession, Window
from pyspark.sql import functions as F

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_observe_goldens.py"
NONDETERMINISTIC = {"ok_nondeterministic_inside_agg", "failed_query", "failed_query_then_reuse_df_succeeds"}
MODE = sys.argv[1]  # classic | connect
OUT = sys.argv[2]
b = SparkSession.builder.config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2")
spark = b.remote("local[2]").getOrCreate() if MODE == "connect" else b.master("local[2]").getOrCreate()
print("VERSION", spark.version, file=sys.stderr)


def norm(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    if v is None:
        return None
    if isinstance(v, (int, bool, str)):
        return v
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
    d["message"] = m.split("\n\n")[0][:600]
    return d


def with_timeout(f, secs=20):
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
    if "e" in box:
        return ("error", box["e"])
    return ("ok", box["v"])


def getres(o, secs=15):
    st, v = with_timeout(lambda: o.get, secs)
    if st == "ok":
        return {"get": "ok", "metrics": norm(v), "types": {k: type(x).__name__ for k, x in v.items()}}
    if st == "blocked":
        return {"get": "blocked"}
    return {"get": "error", **err(v)}


def rows(df):
    return [[norm(x) for x in r] for r in df.collect()]


spark.range(0).count()
R = {}


def case(name):
    def deco(f):
        try:
            R[name] = f()
        except BaseException as e:
            R[name] = {"outcome": "error", **err(e)}
        print("done", name, file=sys.stderr)

    return deco


def base():
    return spark.createDataFrame([("a", 1, 1.5), ("b", 2, None), ("c", None, 3.0)], "k string, v int, x double")


def empty():
    return base().where("1=0")


def run(df_fn, exprs_fn, action=lambda d: d.count(), name=None):
    o = Observation(name) if name else Observation()
    df = df_fn().observe(o, *exprs_fn())
    res = {"action_result": norm(action(df))}
    res.update(getres(o))
    res["obs_name_given"] = bool(name)
    return res


@case("basic_alias")
def _():
    return run(
        base, lambda: [F.count(F.lit(1)).alias("rows"), F.sum("v").alias("sv"), F.max("k").alias("mk")], name="m"
    )


@case("no_alias_names")
def _():
    return run(base, lambda: [F.count(F.lit(1)), F.max("v"), F.sum(F.col("v") + 1), F.lit(42)], name="m")


@case("literal_only")
def _():
    return run(base, lambda: [F.lit(42).alias("c"), F.lit("s").alias("s")], name="m")


@case("composite_expr")
def _():
    return run(base, lambda: [(F.sum("v") + F.avg("x") - F.lit(1)).alias("e"), F.count("*").alias("n")], name="m")


@case("empty_input")
def _():
    return run(
        empty,
        lambda: [
            F.count(F.lit(1)).alias("c"),
            F.sum("v").alias("s"),
            F.max("k").alias("mx"),
            F.avg("x").alias("a"),
            F.collect_list("k").alias("l"),
        ],
        name="m",
    )


@case("empty_input_no_rows_collected")
def _():
    return run(empty, lambda: [F.count(F.lit(1)).alias("c")], action=lambda d: d.collect(), name="m")


@case("types")
def _():
    return run(
        lambda: spark.sql("select cast(1.10 as decimal(5,2)) d, cast('NaN' as double) n, cast(-0.0 as double) z, 1 i"),
        lambda: [
            F.sum("d").alias("sd"),
            F.max("n").alias("mn"),
            F.min("z").alias("mz"),
            F.sum("i").alias("si"),
            F.avg("i").alias("ai"),
            F.collect_list("i").alias("l"),
            F.struct(F.max("i"), F.min("i")).alias("st"),
        ],
        name="m",
    )


@case("nulls_all")
def _():
    return run(
        lambda: spark.sql("select cast(null as int) v"),
        lambda: [F.sum("v").alias("s"), F.count("v").alias("c"), F.max("v").alias("m")],
        name="m",
    )


@case("anonymous_observation")
def _():
    o = Observation()
    df = base().observe(o, F.count(F.lit(1)).alias("n"))
    df.count()
    return {"name_is_str": isinstance(o._name, str), "name_len": len(o._name) if o._name else None, **getres(o)}


# errors
def err_case(exprs_fn, df_fn=base):
    def f():
        o = Observation("m")
        df = df_fn().observe(o, *exprs_fn())
        df.count()  # analysis error surfaces at action (lazily)
        return {"outcome": "ok_unexpected", **getres(o)}

    return f


for nm, fn in {
    "err_attribute": lambda: [F.col("v")],
    "err_attribute_alias_mixed": lambda: [F.count(F.lit(1)).alias("c"), F.col("v").alias("raw")],
    "err_nondeterministic_rand": lambda: [F.rand().alias("r")],
    "ok_nondeterministic_inside_agg": lambda: [F.sum(F.rand()).alias("r")],
    "err_nondeterministic_plus_agg": lambda: [(F.count(F.lit(1)) + F.rand()).alias("r")],
    "err_nested_aggregate": lambda: [F.sum(F.max("v")).alias("n")],
    "err_distinct": lambda: [F.countDistinct("v").alias("d")],
    "err_distinct_sum": lambda: [F.sum_distinct(F.col("v")).alias("d")],
    "err_filter": lambda: [F.expr("count(v) filter (where v > 1)").alias("f")],
    "err_window": lambda: [F.sum("v").over(Window.orderBy("k")).alias("w")],
    "err_generator": lambda: [F.explode(F.array(F.lit(1), F.lit(2))).alias("g")],
    "err_unresolved_column": lambda: [F.sum("nope").alias("s")],
    "err_scalar_subquery": lambda: [F.expr("(select max(id) from range(3))").alias("s")],
    "ok_aggregate_over_subquery_free_udf_like": lambda: [F.sum(F.length("k")).alias("s")],
    "err_unresolved_star": lambda: [F.expr("count(*)").alias("c"), F.col("*")],
    "ok_approx_count_distinct": lambda: [F.approx_count_distinct("v").alias("a")],
    "err_first_ignorenulls_ok": lambda: [F.first("v", True).alias("f")],
    "ok_count_if": lambda: [F.expr("count_if(v>0)").alias("c")],
}.items():
    case(nm)(err_case(fn))


@case("err_empty_name_string_api")
def _():
    df = base().observe("", F.count(F.lit(1)).alias("n"))
    df.count()
    return {"outcome": "ok_unexpected"}


@case("err_empty_observation_name")
def _():
    Observation("")
    return {"outcome": "ok_unexpected"}


@case("err_observation_name_type")
def _():
    Observation(5)
    return {"outcome": "ok_unexpected"}


@case("err_no_exprs")
def _():
    Observation("m")
    base().observe(Observation("m"))
    return {"outcome": "ok_unexpected"}


@case("err_no_exprs_string")
def _():
    base().observe("m")
    return {"outcome": "ok_unexpected"}


@case("err_get_before_observe")
def _():
    o = Observation("m")
    return getres(o, 3)


@case("err_reuse")
def _():
    o = Observation("m")
    base().observe(o, F.count(F.lit(1)).alias("n"))
    base().observe(o, F.count(F.lit(1)).alias("n"))
    return {"outcome": "ok_unexpected"}


@case("get_blocks_without_action")
def _():
    o = Observation("m")
    base().observe(o, F.count(F.lit(1)).alias("n"))
    return getres(o, 4)


@case("get_after_transformation_only_no_action_then_action")
def _():
    o = Observation("m")
    df = base().observe(o, F.count(F.lit(1)).alias("n"))
    r1 = getres(o, 3)
    df.count()
    r2 = getres(o, 10)
    return {"before_action": r1, "after_action": r2}


@case("multiple_actions_first_wins")
def _():
    o = Observation("m")
    df = base().observe(o, F.count(F.lit(1)).alias("n"), F.sum("v").alias("s"))
    c1 = df.count()
    r1 = getres(o)
    c2 = df.where("v is not null").count()
    c3 = df.collect()
    r2 = getres(o)
    return {"counts": [c1, c2, len(c3)], "after_first": r1, "after_more": r2}


@case("count_action_prunes_columns")
def _():
    # df.count() prunes columns; metric on v still computed? (CollectMetrics keeps referenced cols)
    return run(base, lambda: [F.sum("v").alias("s")], action=lambda d: d.count(), name="m")


@case("filter_after_observe")
def _():
    return run(base, lambda: [F.count(F.lit(1)).alias("n")], action=lambda d: d.where("v = 1").count(), name="m")


@case("filter_before_observe")
def _():
    o = Observation("m")
    df = base().where("v = 1").observe(o, F.count(F.lit(1)).alias("n"))
    df.collect()
    return getres(o)


@case("limit_after_observe")
def _():
    return run(base, lambda: [F.count(F.lit(1)).alias("n")], action=lambda d: d.limit(1).collect(), name="m")


@case("take_after_observe")
def _():
    return run(base, lambda: [F.count(F.lit(1)).alias("n")], action=lambda d: d.take(1), name="m")


@case("limit_zero")
def _():
    return run(base, lambda: [F.count(F.lit(1)).alias("n")], action=lambda d: d.limit(0).collect(), name="m")


@case("limit_big_range")
def _():
    return run(
        lambda: spark.range(0, 100000, 1, 4).toDF("v"),
        lambda: [F.count(F.lit(1)).alias("n"), F.sum("v").alias("s")],
        action=lambda d: d.limit(3).collect(),
        name="m",
    )


@case("exact_under_parallelism")
def _():
    return run(
        lambda: spark.range(0, 100000, 1, 8).toDF("v"),
        lambda: [F.count(F.lit(1)).alias("n"), F.sum("v").alias("s"), F.max("v").alias("mx")],
        name="m",
    )


@case("cache")
def _():
    o = Observation("m")
    df = base().observe(o, F.count(F.lit(1)).alias("n")).cache()
    a = df.count()
    r1 = getres(o)
    b2 = df.count()
    r2 = getres(o)
    return {"counts": [a, b2], "after_first": r1, "after_second": r2}


@case("cache_then_new_observe_on_cached")
def _():
    df = base().cache()
    df.count()
    o = Observation("m")
    d2 = df.observe(o, F.count(F.lit(1)).alias("n"))
    d2.collect()
    return getres(o)


@case("union_two_branches_same_name_diff_df")
def _():
    a = base().observe("dup", F.count(F.lit(1)).alias("n"))
    b2 = base().observe("dup", F.count(F.lit(1)).alias("n"))
    a.union(b2).count()
    return {"outcome": "ok_unexpected"}


@case("union_two_branches_distinct_names")
def _():
    oa, ob = Observation("a"), Observation("b")
    a = base().observe(oa, F.count(F.lit(1)).alias("n"))
    b2 = base().where("v=1").observe(ob, F.count(F.lit(1)).alias("n"))
    c = a.union(b2).count()
    return {"count": c, "a": getres(oa), "b": getres(ob)}


@case("self_join_same_df")
def _():
    o = Observation("m")
    d = base().observe(o, F.count(F.lit(1)).alias("n"))
    c = d.alias("l").join(d.alias("r"), F.col("l.k") == F.col("r.k")).count()
    return {"count": c, **getres(o)}


@case("self_union_same_df")
def _():
    o = Observation("m")
    d = base().observe(o, F.count(F.lit(1)).alias("n"))
    c = d.union(d).count()
    return {"count": c, **getres(o)}


@case("join_two_observed")
def _():
    oa, ob = Observation("a"), Observation("b")
    a = base().observe(oa, F.count(F.lit(1)).alias("n"))
    b2 = base().observe(ob, F.sum("v").alias("s"))
    c = a.join(b2, "k").count()
    return {"count": c, "a": getres(oa), "b": getres(ob)}


@case("same_name_string_same_df_in_both_branches")
def _():
    d = base().observe("dup", F.count(F.lit(1)).alias("n"))
    return {"count": d.union(d).count()}


@case("nested_observe_different_names")
def _():
    oa, ob = Observation("a"), Observation("b")
    d = base().observe(oa, F.count(F.lit(1)).alias("n")).where("v=1").observe(ob, F.count(F.lit(1)).alias("m"))
    c = d.count()
    return {"count": c, "a": getres(oa), "b": getres(ob)}


@case("nested_observe_same_name_diff_df")
def _():
    d = base().observe("dup", F.count(F.lit(1)).alias("n")).where("v=1").observe("dup", F.count(F.lit(1)).alias("n"))
    return {"count": d.count()}


@case("observed_metric_column_name_collision")
def _():
    return run(base, lambda: [F.count(F.lit(1)).alias("n"), F.sum("v").alias("n")], name="m")


@case("failed_query")
def _():
    spark.conf.set("spark.sql.ansi.enabled", "true")
    try:
        o = Observation("m")
        d = spark.range(5).toDF("v").observe(o, F.count(F.lit(1)).alias("n")).selectExpr("1/(v-2) as r")
        try:
            d.collect()
            a = "ok_unexpected"
        except Exception as e:
            a = err(e)
        return {"action": a, **getres(o, 15)}
    finally:
        spark.conf.set("spark.sql.ansi.enabled", "true")


@case("failed_query_then_reuse_df_succeeds")
def _():
    o = Observation("m")
    base0 = spark.range(5).toDF("v").observe(o, F.count(F.lit(1)).alias("n"))
    with contextlib.suppress(Exception):
        base0.selectExpr("1/(v-2) as r").collect()
    r1 = getres(o, 8)
    base0.count()
    return {"after_fail": r1, "after_success": getres(o, 8)}


@case("observation_on_action_collect_schema_only")
def _():
    o = Observation("m")
    d = base().observe(o, F.count(F.lit(1)).alias("n"))
    s = d.schema.simpleString()
    c = d.columns
    return {"schema": s, "columns": c, "get_after_schema": getres(o, 3)}


@case("explain_does_not_trigger")
def _():
    o = Observation("m")
    d = base().observe(o, F.count(F.lit(1)).alias("n"))
    d.explain()
    return getres(o, 3)


@case("write_action")
def _():
    import tempfile

    o = Observation("m")
    d = base().observe(o, F.count(F.lit(1)).alias("n"), F.sum("v").alias("s"))
    p = tempfile.mkdtemp() + "/o"
    d.write.parquet(p)
    return getres(o)


@case("string_observe_returns_df_only")
def _():
    d = base().observe("nm", F.count(F.lit(1)).alias("n"))
    return {"count": d.count(), "type": type(d).__name__}


@case("observation_in_groupby_downstream")
def _():
    o = Observation("m")
    d = base().observe(o, F.count(F.lit(1)).alias("n")).groupBy("k").count()
    return {"rows": sorted(rows(d)), **getres(o)}


@case("observation_through_sort_limit_topn")
def _():
    return run(
        lambda: spark.range(0, 1000, 1, 4).toDF("v"),
        lambda: [F.count(F.lit(1)).alias("n")],
        action=lambda d: d.orderBy("v").limit(5).collect(),
        name="m",
    )


@case("show_action_limit20")
def _():
    return run(
        lambda: spark.range(0, 1000, 1, 4).toDF("v"),
        lambda: [F.count(F.lit(1)).alias("n")],
        action=lambda d: d.show(3) or 0,
        name="m",
    )


@case("python_attrs")
def _():
    o = Observation("m")
    return {
        "has_get_async": hasattr(o, "get_async"),
        "has_future": hasattr(o, "future"),
        "has_name": hasattr(o, "name"),
        "public": [a for a in dir(o) if not a.startswith("__")],
    }


@case("aggregate_with_alias_inside_nested_alias")
def _():
    return run(base, lambda: [F.sum("v").alias("a").alias("b")], name="m")


@case("ansi_off_overflow_sum")
def _():
    spark.conf.set("spark.sql.ansi.enabled", "false")
    try:
        return run(
            lambda: spark.sql("select cast(9223372036854775807 as bigint) v union all select 1"),
            lambda: [F.sum("v").alias("s"), (F.max("v") + 1).alias("p1")],
            name="m",
        )
    finally:
        spark.conf.set("spark.sql.ansi.enabled", "true")


@case("ansi_on_overflow_sum")
def _():
    spark.conf.set("spark.sql.ansi.enabled", "true")
    return run(
        lambda: spark.sql("select cast(9223372036854775807 as bigint) v union all select 1"),
        lambda: [F.sum("v").alias("s")],
        name="m",
    )


@case("ansi_on_cast_error_inside_metric")
def _():
    return run(base, lambda: [F.sum(F.col("k").cast("int")).alias("s")], name="m")


@case("conditional_metric_no_raise_unselected")
def _():
    return run(
        lambda: spark.sql("select * from values (0),(2) as t(v)"),
        lambda: [F.sum(F.when(F.col("v") != 0, F.lit(10) / F.col("v"))).alias("s")],
        name="m",
    )


# ---- second-action probes (first-wins in classic, last-wins in Connect) ----
@case("first_then_second_action")
def _():
    o = Observation("m")
    df = base().observe(o, F.count(F.lit(1)).alias("n"))
    df.limit(1).collect()
    first = dict(o.get)
    df.count()
    second = dict(o.get)
    return {"after_limit1": first, "after_count": second}


if MODE == "connect":

    @case("flows_string_observe")
    def _():
        d = base().observe("nm", F.count(F.lit(1)).alias("n"), F.sum("v").alias("s"), F.lit(7))
        d.toPandas()
        return d.executionInfo.flows

    @case("flows_two_nested")
    def _():
        d = base().observe("a", F.count(F.lit(1)).alias("n")).observe("b", F.sum("v").alias("s"))
        d.toPandas()
        return d.executionInfo.flows

    @case("flows_no_alias")
    def _():
        d = base().observe("x", F.count(F.lit(1)), F.max("v"))
        d.toPandas()
        return d.executionInfo.flows

    @case("flows_observe_pruned_by_empty_relation")
    def _():
        d = base().observe("e", F.count(F.lit(1)).alias("n")).where("1=0")
        d.toPandas()
        return d.executionInfo.flows


def _scrub(m):
    m = re.sub(r"rand\(-?\d+\)", "rand(<seed>)", m)
    m = re.sub(r"#\d+L?", "#<id>", m)
    return re.sub(r"\], \d+", "], <dataframe-id>", m)


def _clean_msg(m):
    return _scrub(m.split("\n==")[0].split(";\n")[0].strip())


def clean(name, v):
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k == "message" and isinstance(x, str):
                out[k] = _clean_msg(
                    x if "AssertionError" not in x else "java.lang.AssertionError in PythonSQLUtils.toPyRow"
                )
            elif k == "messageParameters" and isinstance(x, dict):
                out[k] = {a: _scrub(b) for a, b in x.items() if a != "plan"}
            elif name == "first_then_second_action" and k == "after_limit1" and MODE == "connect":
                out[k] = "<unstable>"  # Connect: 1 in four isolated sessions, 3 once inside the full suite
            elif k in ("metrics", "types") and name in NONDETERMINISTIC:
                out[k] = "<nondeterministic>"
            else:
                out[k] = clean(name, x)
        return out
    return v


R = {k: clean(k, v) for k, v in R.items()}
Path(OUT).write_text(
    json.dumps(
        {
            "format_version": FORMAT_VERSION,
            "generator": GENERATOR,
            "mode": MODE,
            "spark_version": spark.version,
            "cases": R,
        },
        indent=1,
        sort_keys=True,
    )
)
spark.stop()
