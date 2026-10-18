#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Writes the Spark Connect plans of crates/krishiv-connect/tests/fixtures/pyspark-4.2/.

Each case is a DataFrame built with the real PySpark 4.2 Connect client; its
plan is what the client would put in `ExecutePlanRequest.plan`. No server is
involved: building a plan only needs the client library, so the plans have
the exact shape a stock PySpark sends (plan ids, null orderings, literal
types, canonical operator names).

Run it from the repository root inside the pinned Spark image, with the
client's Python dependencies mounted (the image has none of grpc, pandas,
pyarrow). The dependencies are installed outside the image so that the shared
container is untouched:

    uv pip install --python-version 3.10 --python-platform x86_64-manylinux_2_28 \
        --target "$DEPS" grpcio grpcio-status googleapis-common-protos pandas pyarrow numpy protobuf
    docker run --rm -v "$PWD:/repo" -v "$DEPS:/deps:ro" \
        -e PYTHONPATH=/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.9-src.zip:/deps \
        apache/spark:4.2.0 python3 /repo/tests/scripts/gen_connect_goldens.py \
        /repo/crates/krishiv-connect/tests/fixtures/pyspark-4.2

The canary strings in the secret cases are test data: the tests assert that
they never reach the IR, its printers or an error message.
"""

import datetime
import decimal
import faulthandler
import json
import os
import sys

from google.protobuf.json_format import MessageToDict
from pyspark.sql import types as T
from pyspark.sql.connect import functions as F
from pyspark.sql.connect import plan as P
from pyspark.sql.connect import proto as pb2
from pyspark.sql.connect.column import Column
from pyspark.sql.connect.expressions import LiteralExpression
from pyspark.sql.connect.session import SparkSession

CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


def df_of(spark, name="t"):
    return spark.read.format("parquet").load(f"/data/{name}.parquet")


# --- reads ---------------------------------------------------------------


@case
def read_parquet(s):
    return df_of(s)


@case
def read_parquet_paths(s):
    return s.read.parquet("/data/a.parquet", "/data/b.parquet")


@case
def read_parquet_options(s):
    return s.read.option("mergeSchema", "true").option("pathGlobFilter", "*.parquet").parquet("/data/p")


@case
def read_load_default_format(s):
    return s.read.load("/data/default")


@case
def read_secret_options(s):
    return (
        s.read.format("parquet")
        .option("fs.s3a.access.key", "CANARY-ACCESS-KEY")
        .option("fs.s3a.secret.key", "CANARY-SECRET-KEY")
        .option("mergeSchema", "true")
        .load("s3a://bucket/events")
    )


@case
def read_signed_url_path(s):
    return s.read.parquet("https://host.example/data.parquet?X-Amz-Signature=CANARY-SIGNATURE&X-Amz-Expires=60")


@case
def read_schema_string(s):
    return s.read.schema("a INT, b STRING").parquet("/data/p")


@case
def read_table(s):
    return s.read.table("db.events")


@case
def read_table_quoted(s):
    return s.table("`my db`.`ev.ents`")


@case
def read_table_options(s):
    return s.read.option("versionAsOf", "3").table("events")


# --- relations -----------------------------------------------------------


@case
def sql(s):
    return _from_plan(s, P.SQL("SELECT a, count(*) FROM t GROUP BY a"))


@case
def sql_named_args(s):
    return _from_plan(
        s, P.SQL("SELECT * FROM t WHERE a > :lo AND b = :name", named_args={"lo": F.lit(5), "name": F.lit("x")})
    )


@case
def sql_positional_args(s):
    return _from_plan(s, P.SQL("SELECT * FROM t WHERE a > ? AND b = ?", args=[F.lit(5), F.lit("x")]))


def _from_plan(s, plan):
    from pyspark.sql.connect.dataframe import DataFrame

    return DataFrame(plan, s)


@case
def project(s):
    df = df_of(s)
    return df.select("a", df.b, (F.col("c") + 1).alias("d"))


@case
def project_star(s):
    return df_of(s).select("*")


@case
def project_qualified_star(s):
    return df_of(s).alias("x").select("x.*")


@case
def project_dataframe_star(s):
    df = df_of(s)
    return df.select(df["*"])


@case
def project_dotted_names(s):
    return df_of(s).select("a.b.c", "`we.ird`.x", "`a``b`")


@case
def filter_comparison(s):
    return df_of(s).filter(F.col("a") > 1)


@case
def filter_logic(s):
    a, b = F.col("a"), F.col("b")
    return df_of(s).where(((a >= 1) & (b <= 2)) | ~(a == b) | (a != 3) | a.eqNullSafe(b) | (a < 4))


@case
def filter_null_tests(s):
    a = F.col("a")
    return df_of(s).filter(a.isNull() | a.isNotNull())


@case
def filter_string_predicates(s):
    c = F.col("c")
    return df_of(s).filter(c.like("a%") | c.rlike("^a") | c.ilike("A%") | c.isin("x", "y"))


@case
def filter_expression_string(s):
    return df_of(s).filter("a > 1 AND b < 2")


@case
def select_expr(s):
    return df_of(s).selectExpr("a + 1 AS x")


@case
def arithmetic(s):
    a, b = F.col("a"), F.col("b")
    return df_of(s).select(
        (a + b).alias("add"),
        (a - b).alias("sub"),
        (a * b).alias("mul"),
        (a / b).alias("div"),
        (a % b).alias("rem"),
        (-a).alias("neg"),
    )


@case
def conditional(s):
    a = F.col("a")
    return df_of(s).select(
        F.when(a > 1, "x").when(a > 0, "y").otherwise("z").alias("w"),
        F.concat(F.col("c"), F.lit("-"), F.col("d")).alias("cc"),
    )


@case
def between(s):
    return df_of(s).filter(F.col("a").between(1, 5))


@case
def function_calls(s):
    return df_of(s).select(F.upper("c"), F.abs("a"), F.coalesce("a", "b"), F.substring("c", 1, 2))


@case
def aggregate_group_by(s):
    return df_of(s).groupBy("a", "b").agg(F.sum("c"), F.count("*"), F.avg("d").alias("m"))


@case
def aggregate_global(s):
    return df_of(s).agg(F.max("a"), F.countDistinct("b"))


@case
def aggregate_count(s):
    return df_of(s).groupBy("a").count()


@case
def aggregate_rollup(s):
    return df_of(s).rollup("a", "b").agg(F.sum("c"))


@case
def aggregate_cube(s):
    return df_of(s).cube("a", "b").agg(F.sum("c"))


@case
def aggregate_pivot(s):
    return df_of(s).groupBy("a").pivot("b", [1, 2]).agg(F.sum("c"))


def _two(s):
    return df_of(s, "l"), df_of(s, "r")


def _join_how(how):
    def build(s):
        left, right = _two(s)
        return left.join(right, left["id"] == right["id"], how)

    build.__name__ = f"join_{how}"
    return build


for _how in ["inner", "left", "right", "full", "left_semi", "left_anti", "cross"]:
    case(_join_how(_how))


@case
def join_using(s):
    left, right = _two(s)
    return left.join(right, "id")


@case
def join_using_columns_left(s):
    left, right = _two(s)
    return left.join(right, ["id", "k"], "left")


@case
def join_no_condition(s):
    left, right = _two(s)
    return left.crossJoin(right)


@case
def join_aliased(s):
    left, right = _two(s)
    a, b = left.alias("a"), right.alias("b")
    return a.join(b, F.col("a.id") == F.col("b.id"))


@case
def join_self(s):
    df = df_of(s)
    return df.join(df.alias("o"), df["id"] == F.col("o.id"))


@case
def sort_default(s):
    return df_of(s).orderBy("a", "b")


@case
def sort_directions(s):
    a, b = F.col("a"), F.col("b")
    return df_of(s).orderBy(a.desc(), b.asc_nulls_last(), a.desc_nulls_first(), b.asc_nulls_first())


@case
def sort_within_partitions(s):
    return df_of(s).sortWithinPartitions("a")


@case
def limit(s):
    return df_of(s).limit(10)


@case
def with_column(s):
    return df_of(s).withColumn("x", F.col("a") + 1)


@case
def with_columns(s):
    return df_of(s).withColumns({"x": F.col("a") + 1, "y": F.lit("k")})


@case
def with_column_renamed(s):
    return df_of(s).withColumnRenamed("a", "z")


@case
def alias(s):
    return df_of(s).alias("t1")


@case
def to_df(s):
    return df_of(s).toDF("x", "y")


@case
def distinct(s):
    return df_of(s).distinct()


@case
def union(s):
    left, right = _two(s)
    return left.union(right)


@case
def show_string(s):
    from pyspark.sql.connect.dataframe import DataFrame

    df = df_of(s)
    return DataFrame(P.ShowString(df._plan, 20, 20, False), s)


@case
def chain(s):
    df = df_of(s)
    return (
        df.filter(F.col("a") > 1)
        .withColumn("x", F.col("b") * 2)
        .groupBy("x")
        .agg(F.sum("a").alias("s"))
        .orderBy(F.col("s").desc())
        .limit(5)
    )


@case
def column_by_dataframe(s):
    left, right = _two(s)
    j = left.join(right, left["id"] == right["id"])
    return j.select(left["a"], right["b"])


@case
def result_to_arrow(s):
    # `toArrow` and `toPandas` send the DataFrame's own plan; the Arrow batch
    # format is a request option, not part of the plan.
    return df_of(s).filter(F.col("a") > 0)


# --- expressions: literals and casts -----------------------------------------


def _lit(value, data_type=None):
    return Column(
        LiteralExpression(value, data_type) if data_type is not None else LiteralExpression._from_value(value)
    )


@case
def literals_scalar(s):
    return df_of(s).select(
        _lit(True),
        _lit(1),
        _lit(1 << 40),
        _lit(1.5),
        _lit("text"),
        _lit(b"bin"),
        _lit(decimal.Decimal("12.345")),
        _lit(datetime.date(2020, 2, 29)),
        _lit(datetime.datetime(2020, 2, 29, 1, 2, 3, 456789, tzinfo=datetime.timezone.utc)),  # noqa: UP017 (Python 3.10 image)
        _lit(datetime.datetime(2020, 2, 29, 1, 2, 3, 456789)),
    )


@case
def literals_sized(s):
    return df_of(s).select(
        _lit(7, T.ByteType()),
        _lit(300, T.ShortType()),
        _lit(70000, T.IntegerType()),
        _lit(1.25, T.FloatType()),
        _lit(2.5, T.DoubleType()),
        _lit(decimal.Decimal("1.50"), T.DecimalType(10, 4)),
    )


@case
def literals_null(s):
    return df_of(s).select(
        _lit(None, T.IntegerType()),
        _lit(None, T.StringType()),
        _lit(None, T.NullType()),
        _lit(None, T.DecimalType(12, 3)),
    )


@case
def literals_special_floats(s):
    return df_of(s).select(_lit(float("nan")), _lit(float("inf")), _lit(-0.0))


@case
def literals_interval(s):
    return df_of(s).select(_lit(datetime.timedelta(days=1, seconds=2, microseconds=3)))


@case
def literals_unicode_string(s):
    return df_of(s).select(_lit("h\u00e9llo \u4e16\u754c \U0001f600"), _lit(""))


@case
def literals_array(s):
    return df_of(s).select(F.array(F.lit(1), F.lit(2)), _lit([1, 2, 3]))


@case
def cast_type(s):
    return df_of(s).select(
        F.col("a").cast("int"),
        F.col("a").cast(T.LongType()),
        F.col("a").cast(T.DecimalType(10, 2)),
        F.col("a").cast("timestamp_ntz"),
        F.col("a").cast("decimal(5)"),
        F.col("a").cast(T.StringType()),
    )


@case
def cast_try(s):
    return df_of(s).select(F.col("a").try_cast("int"))


@case
def cast_unsupported_type(s):
    return df_of(s).select(F.col("a").cast("array<int>"))


@case
def alias_expression(s):
    return df_of(s).select(F.col("a").alias("x"), (F.col("b") + 1).alias("y"))


@case
def alias_multi(s):
    return df_of(s).select(F.posexplode("arr").alias("pos", "v"))


@case
def user_defined_function_call(s):
    return df_of(s).select(F.call_function("db.f", F.col("a")))


# ---------------------------------------------------------------------------


def main(out_dir):
    faulthandler.dump_traceback_later(240, exit=True)
    os.makedirs(out_dir, exist_ok=True)
    spark = SparkSession.builder.remote("sc://localhost:1").create()
    for name, build in sorted(CASES.items()):
        print(name, flush=True)
        # Number the plans of every case from 0, as a fresh session does, so
        # that adding a case does not renumber the others.
        P.LogicalPlan._nextPlanId = 0
        # `to_proto` would first ask the server for its plan-compression
        # setting; the relation itself needs no server.
        plan = pb2.Plan()
        plan.root.CopyFrom(build(spark)._plan.plan(spark.client))
        text = json.dumps(MessageToDict(plan, preserving_proto_field_name=False), indent=2, sort_keys=True)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            f.write(text + "\n")
    print(f"wrote {len(CASES)} plans to {out_dir}")
    sys.stdout.flush()
    # Building a plan opens no connection, but the client's exit hook tries to
    # reach the (absent) server and retries for minutes.
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
