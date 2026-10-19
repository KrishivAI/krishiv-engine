# SPDX-License-Identifier: Apache-2.0
"""The PySpark-shaped facade (``krishiv.sql``): plan building and analysis (no execution)."""

from __future__ import annotations

from typing import Any

import pytest

import krishiv
from krishiv import errors
from krishiv.sql import DataFrame, Row, SparkSession
from krishiv.sql import functions as F
from krishiv.sql import types as T

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        pa.table({"id": pa.array([1, 2, 3], pa.int64()), "name": ["a", None, "c"], "v": [1.5, 2.5, None]})
    )


def test_session_is_shared_and_stoppable() -> None:
    first = krishiv.session()
    assert krishiv.session() is first
    assert SparkSession.builder.config("spark.sql.ansi.enabled", False).getOrCreate() is first
    assert first.conf.get("spark.sql.ansi.enabled") == "false"
    first.stop()
    assert krishiv.session() is not first
    krishiv.session().stop()


def test_builder_options(spark: SparkSession) -> None:
    assert SparkSession.builder is not SparkSession.builder
    configured = SparkSession.builder.appName("x").config(map={"spark.sql.ansi.enabled": "false"}).create()
    assert configured.conf.get("spark.sql.ansi.enabled") == "false"
    assert configured.version == "4.2.0"
    assert spark.conf.get("no.such.key", None) is None
    assert spark.conf.get("no.such.key", 5) == "5"
    with pytest.raises(errors.PySparkValueError):
        spark.conf.get("no.such.key")
    spark.conf.set("spark.sql.ansi.enabled", True)
    assert spark.conf.get("spark.sql.ansi.enabled") == "true"
    spark.conf.unset("spark.sql.ansi.enabled")
    with pytest.raises(errors.PySparkTypeError):
        spark.conf.set("k", 1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkNotImplementedError):
        SparkSession.builder.remote("sc://x")


def test_schema_and_columns_by_analysis(spark: SparkSession) -> None:
    df = spark.sql("SELECT 1 AS a, 'x' AS b, CAST(1 AS DECIMAL(10,2)) AS c, DATE'2020-01-01' AS d")
    assert df.columns == ["a", "b", "c", "d"]
    assert df.dtypes == [("a", "int"), ("b", "string"), ("c", "decimal(10,2)"), ("d", "date")]
    assert df.schema == T.StructType(
        [
            T.StructField("a", T.IntegerType(), False),
            T.StructField("b", T.StringType(), False),
            T.StructField("c", T.DecimalType(10, 2), True),
            T.StructField("d", T.DateType(), False),
        ]
    )
    assert df.schema.simpleString() == "struct<a:int,b:string,c:decimal(10,2),d:date>"
    assert df.schema["b"].dataType == T.StringType()
    assert df.schema[0].name == "a"
    assert df.schema.fieldNames() == ["a", "b", "c", "d"]
    assert len(df.schema) == 4
    assert [f.name for f in df.schema] == df.columns


def test_print_schema(df: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    df.printSchema()
    assert capsys.readouterr().out.splitlines() == [
        "root",
        " |-- id: bigint (nullable = true)",
        " |-- name: string (nullable = true)",
        " |-- v: double (nullable = true)",
    ]


def test_column_access(df: DataFrame) -> None:
    assert df.select("id", df["name"], df.v).columns == ["id", "name", "v"]
    assert df.select(["id", "v"]).columns == ["id", "v"]
    assert df[["id", "v"]].columns == ["id", "v"]
    assert df[df.id > 1].columns == df.columns
    assert df[1].alias("n").__class__.__name__ == "Column"
    assert df.select("*").columns == df.columns
    with pytest.raises(AttributeError):
        _ = df.nope
    with pytest.raises(errors.PySparkTypeError):
        df[1.5]
    with pytest.raises(errors.PySparkTypeError):
        df.select(1)  # ty: ignore[invalid-argument-type]


def test_with_column_filter_alias(df: DataFrame) -> None:
    out = (
        df.withColumn("w", F.abs(F.col("id") - 5) * 2 + 1)
        .withColumn("s", F.upper("name"))
        .filter((F.col("w") > 1) & ~F.col("v").isNull() | F.col("id").isin(1, 2))
        .where(F.col("id").between(1, 3) & F.col("name").like("a%") & F.col("id").eqNullSafe(1))
        .alias("t")
    )
    assert out.columns == ["id", "name", "v", "w", "s"]
    assert out.schema["s"].dataType == T.StringType()
    assert (
        df.select((F.col("id") / 2).alias("half"), (-F.col("id")) % 2, 2 * F.col("id"), 1 - F.col("id")).columns[0]
        == "half"
    )
    assert df.select(F.col("id").cast("string").alias("s"), F.col("id").astype(T.DoubleType()).alias("d")).dtypes == [
        ("s", "string"),
        ("d", "double"),
    ]
    assert df.select(F.col("id").isin([1, 2]), (F.col("id") != 1), (F.col("id") <= 1), (F.col("id") >= 1)).columns


def test_withcolumn_argument_errors(df: DataFrame) -> None:
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        df.withColumn("c", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.filter(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkValueError, match="CANNOT_CONVERT_COLUMN_INTO_BOOL"):
        bool(F.col("a"))
    with pytest.raises(errors.PySparkTypeError):
        iter(F.col("a"))
    with pytest.raises(errors.PySparkTypeError):
        df.limit("3")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkValueError):
        F.col("a").alias("a", "b")
    with pytest.raises(errors.PySparkValueError):
        F.col("a").alias("a", metadata={})
    with pytest.raises(errors.PySparkTypeError):
        F.col("a").alias(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        F.col("a").like(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        F.col("a").cast(1)  # ty: ignore[invalid-argument-type]


def test_analysis_errors_surface_at_first_action(df: DataFrame) -> None:
    bad = df.select("missing")  # building does not analyze
    with pytest.raises(errors.AnalysisException) as e:
        _ = bad.columns
    assert e.value.getCondition() == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    assert e.value.getSqlState() == "42703"
    assert isinstance(e.value, errors.PySparkException)
    with pytest.raises(errors.ParseException, match="UNSUPPORTED_DATATYPE"):
        _ = df.withColumn("x", F.col("id").cast("not_a_type")).columns


def test_group_by_and_agg(df: DataFrame) -> None:
    out = df.groupBy("name").agg(F.sum("v").alias("s"), F.avg(F.col("v")), F.max("id"), F.min("id"), F.count("*"))
    assert out.columns[:2] == ["name", "s"]
    assert df.groupBy().count().columns == ["count"]
    assert df.agg(F.count_distinct("id", "name").alias("d"), F.countDistinct("id")).columns == [
        "d",
        "count(DISTINCT id)",
    ]
    assert df.agg(F.mean("v"), F.count("id")).columns[0].startswith("avg")
    assert df.groupby(["id"]).agg(F.sum(F.coalesce("v", F.lit(0.0))).alias("c")).columns == ["id", "c"]
    with pytest.raises(errors.PySparkNotImplementedError):
        df.groupBy("id").agg({"v": "sum"})  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.groupBy("id").agg("v")  # ty: ignore[invalid-argument-type]


def test_joins(df: DataFrame) -> None:
    other = df.select("id", F.col("v").alias("v2"))
    assert df.join(other, "id").columns == ["id", "name", "v", "v2"]
    assert df.join(other, ["id"], "left").columns == ["id", "name", "v", "v2"]
    left, right = df.alias("l"), other.alias("r")
    assert left.join(right, F.col("l.id") == F.col("r.id")).columns == ["id", "name", "v", "id", "v2"]
    with pytest.raises(errors.AnalysisException, match="AMBIGUOUS_COLUMN_REFERENCE"):
        _ = df.join(other, df.id == other.id).columns
    both = [F.col("l.id") == F.col("r.id"), F.col("l.v") == F.col("r.v2")]
    assert left.join(right, both, "left_outer").columns == ["id", "name", "v", "id", "v2"]
    assert left.join(right, F.col("l.id") == F.col("r.id"), "left_semi").columns == df.columns
    assert left.join(right, F.col("l.id") == F.col("r.id"), "anti").columns == df.columns
    for how in ("right", "outer", "full_outer", "rightouter", "fullouter"):
        assert df.join(other, "id", how).columns
    for how in ("leftsemi", "leftanti"):
        assert left.join(right, F.col("l.id") == F.col("r.id"), how).columns
    assert df.crossJoin(other).columns == ["id", "name", "v", "id", "v2"]
    assert df.join(other).columns == df.crossJoin(other).columns
    assert df.alias("l").join(df.alias("r"), F.col("l.id") == F.col("r.id")).columns == df.columns * 2
    with pytest.raises(errors.PySparkValueError, match="VALUE_NOT_ALLOWED"):
        df.join(other, "id", "sideways")
    with pytest.raises(errors.PySparkValueError):
        df.join(other, how="left")
    with pytest.raises(errors.PySparkTypeError):
        df.join(other, ["id", df.id])  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.join(1)  # ty: ignore[invalid-argument-type]


def test_order_by_and_limit(df: DataFrame) -> None:
    out = df.orderBy("id", F.col("v").desc_nulls_first()).limit(2)
    assert out.columns == df.columns
    assert df.sort(
        F.col("id").asc_nulls_last(), F.col("v").desc(), F.col("name").asc_nulls_first(), F.col("v").desc_nulls_last()
    ).columns
    assert df.orderBy(["id", "v"], ascending=[True, False]).columns == df.columns
    assert df.orderBy("id", ascending=False).columns == df.columns
    assert df.orderBy(F.desc("id"), F.asc("v")).columns == df.columns
    with pytest.raises(errors.PySparkValueError):
        df.orderBy("id", "v", ascending=[True])


def test_functions_and_literals(df: DataFrame) -> None:
    out = df.select(
        F.lit(1),
        F.lit(2**40),
        F.lit(1.5),
        F.lit("s"),
        F.lit(True),
        F.lit(F.col("id")),
        F.sqrt("v"),
        F.floor("v"),
        F.ceil("v"),
        F.lower("name"),
        F.length("name"),
        F.trim("name"),
        F.ltrim("name"),
        F.rtrim("name"),
        F.concat("name", F.lit("!")),
        F.greatest("id", "v"),
        F.least("id", "v"),
        F.call_function("upper", F.col("name")),
        F.column("id"),
    )
    assert [t for _, t in out.dtypes[:5]] == ["int", "bigint", "double", "string", "boolean"]
    with pytest.raises(errors.PySparkTypeError):
        F.lit([1])
    with pytest.raises(errors.PySparkNotImplementedError):
        F.lit(None)
    with pytest.raises(errors.PySparkValueError):
        F.lit(2**70)
    with pytest.raises(errors.PySparkNotImplementedError):
        F.expr("a + 1")


def test_string_expressions_are_not_implemented_yet(df: DataFrame) -> None:
    with pytest.raises(errors.PySparkNotImplementedError, match="selectExpr"):
        df.selectExpr("id + 1")
    with pytest.raises(errors.PySparkNotImplementedError):
        df.where("id > 1")
    with pytest.raises(errors.PySparkNotImplementedError):
        df.filter("id > 1")


def test_session_tables_and_reader(spark: SparkSession, df: DataFrame, tmp_path: Any) -> None:
    with pytest.raises(errors.AnalysisException) as e:
        _ = spark.table("missing").columns
    assert e.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"
    with pytest.raises(errors.ParseException):
        spark.sql("SELECT 1 +")
    with pytest.raises(errors.PySparkTypeError):
        spark.sql(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkNotImplementedError):
        spark.sql("SELECT :a", args={"a": 1})
    with pytest.raises(errors.PySparkNotImplementedError):
        spark.createDataFrame([(1, "a")])
    with pytest.raises(errors.PySparkNotImplementedError):
        spark.createDataFrame(pa.table({"a": [1]}), schema="a int")
    with pytest.raises(errors.PySparkTypeError):
        spark.read.load()
    reader = spark.read.format("parquet").option("mergeSchema", True).options(x="y")
    assert reader.load(str(tmp_path / "none.parquet")).__class__ is DataFrame
    assert spark.read.csv(str(tmp_path / "none.csv"), header=True).__class__ is DataFrame
    assert spark.read.json([str(tmp_path / "none.json")]).__class__ is DataFrame
    assert spark.read.parquet(str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")).__class__ is DataFrame
    assert df.sparkSession is not None


def test_create_dataframe_from_other_producers(spark: SparkSession) -> None:
    polars = pytest.importorskip("polars")
    duckdb = pytest.importorskip("duckdb")
    assert spark.createDataFrame(polars.DataFrame({"a": [1, 2]})).columns == ["a"]
    assert spark.createDataFrame(duckdb.sql("SELECT 1 AS x, 'q' AS y")).dtypes == [("x", "int"), ("y", "string")]
    batch = pa.record_batch({"b": [1.0]})
    assert spark.createDataFrame(batch).columns == ["b"]


def test_row() -> None:
    r = Row(a=1, b="x")
    assert (r.a, r["b"], r[0], r.asDict()) == (1, "x", 1, {"a": 1, "b": "x"})
    assert repr(r) == "Row(a=1, b='x')"
    assert r == (1, "x")
    positional = Row(1, 2)
    assert positional[1] == 2
    assert repr(positional) == "<Row(1, 2)>"
    with pytest.raises(errors.PySparkValueError):
        Row(1, a=2)
    with pytest.raises(errors.PySparkValueError):
        positional.asDict()
    with pytest.raises(KeyError):
        r["zz"]
    with pytest.raises(errors.PySparkAttributeError):
        _ = r.zz
    with pytest.raises(AttributeError):
        _ = r.__nope__


def test_types() -> None:
    assert T.IntegerType() == T.IntegerType()
    assert T.IntegerType() != T.LongType()
    assert hash(T.DecimalType(5, 2)) == hash(T.DecimalType(5, 2))
    assert repr(T.StringType()) == "StringType()"
    assert repr(T.DecimalType(5, 2)) == "DecimalType(5,2)"
    assert T.IntegerType.typeName() == "integer"
    assert T.TimestampNTZType.typeName() == "timestamp_ntz"
    assert T.StructType().simpleString() == "struct<>"
    field = T.StructField("a", T.LongType())
    assert repr(field) == "StructField('a', LongType(), True)"
    assert field.simpleString() == "a:bigint"
    assert hash(T.StructType([field])) == hash(T.StructType([field]))
    with pytest.raises(KeyError):
        T.StructType([field])["b"]
    with pytest.raises(errors.PySparkTypeError):
        T.StructField("a", "int")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkValueError):
        T._from_simple_string("array<int>")
    for t in (T.NullType, T.BooleanType, T.ByteType, T.ShortType, T.FloatType, T.BinaryType, T.TimestampType):
        assert T._from_simple_string(t().simpleString()) == t()
