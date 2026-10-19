# SPDX-License-Identifier: Apache-2.0
"""DataFrame actions (P2-T091b part 3): ``toArrow``, ``collect``, ``show``, ``toPandas``, ``count``,
``__arrow_c_stream__`` (pyarrow, Polars, DuckDB), temp views, file reads and ``lit(None)``."""

from __future__ import annotations

import gc
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, Row, SparkSession
from krishiv.sql import functions as F
from krishiv.sql._convert import cell, rows_of, show_string

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        pa.table({"id": pa.array([1, 2, 3], pa.int64()), "name": ["a", None, "c"], "v": [1.5, 2.5, None]})
    )


def shown(df: DataFrame, capsys: pytest.CaptureFixture[str], *args: Any, **kwargs: Any) -> str:
    df.show(*args, **kwargs)
    return capsys.readouterr().out


def test_smoke_select_one_to_arrow(spark: SparkSession) -> None:
    """P2-T103's acceptance: ``sql("select 1").toArrow()``."""
    table = spark.sql("select 1").toArrow()
    assert table.to_pydict() == {"1": [1]}


def test_to_arrow_and_collect(df: DataFrame) -> None:
    table = df.orderBy("id").toArrow()
    assert table.to_pydict() == {"id": [1, 2, 3], "name": ["a", None, "c"], "v": [1.5, 2.5, None]}
    rows = df.orderBy("id").collect()
    assert rows[0] == Row(id=1, name="a", v=1.5)
    assert rows[1].asDict() == {"id": 2, "name": None, "v": 2.5}
    assert rows[2].v is None
    assert df.take(2) == df.limit(2).collect()
    assert df.head() is not None
    assert len(df.head(2)) == 2  # ty: ignore[invalid-argument-type]
    assert df.first() is not None
    assert df.filter(F.col("id") > 99).head() is None
    assert df.filter(F.col("id") > 99).collect() == []


def test_empty_result_keeps_its_schema(df: DataFrame) -> None:
    empty = df.filter(F.col("id") > 99).toArrow()
    assert empty.num_rows == 0
    assert empty.schema.names == ["id", "name", "v"]


def test_rows_have_spark_python_types() -> None:
    """The engine has no nested types yet; the conversion is tested on pyarrow tables."""
    table = pa.table(
        {
            "d": pa.array([date(2020, 1, 2)]),
            "dec": pa.array([Decimal("1.50")], pa.decimal128(5, 2)),
            "s": pa.array([{"x": 1, "y": "z"}]),
            "m": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int64())),
            "l": pa.array([[1, 2]]),
        }
    )
    [row] = rows_of(table)
    assert row.d == date(2020, 1, 2)
    assert row.dec == Decimal("1.50")
    assert row.s == Row(x=1, y="z")
    assert row.m == {"k": 1}
    assert row.l == [1, 2]
    assert Row(a=1, b=2) == (1, 2)


def test_collect_with_duplicate_column_names(spark: SparkSession) -> None:
    [row] = spark.sql("select 1 as a, 2 as a").collect()
    assert tuple(row) == (1, 2)


def test_count(df: DataFrame, spark: SparkSession) -> None:
    assert df.count() == 3
    assert df.filter(F.col("id") > 1).count() == 2
    assert spark.sql("select 1 where false").count() == 0


def test_actions_rerun_the_query(spark: SparkSession) -> None:
    df = spark.sql("select 1 as a")
    assert df.collect() == df.collect()
    assert df.toArrow().equals(df.toArrow())
    stream = df.__arrow_c_stream__()
    assert stream is not df.__arrow_c_stream__()


def test_errors_surface_as_krishiv_exceptions(spark: SparkSession) -> None:
    df = spark.sql("select 1 / 0 as x")
    with pytest.raises(errors.PySparkException):
        df.collect()
    with pytest.raises(errors.PySparkException):
        df.toArrow()
    with pytest.raises(errors.PySparkException):
        df.count()


def test_lit_none(spark: SparkSession) -> None:
    df = spark.sql("select 1 as a").withColumn("n", F.lit(None))
    assert df.dtypes == [("a", "int"), ("n", "void")]
    assert df.collect() == [Row(a=1, n=None)]
    assert spark.sql("select 1 as a").filter(F.col("a").isNotNull() & F.lit(None).isNull()).count() == 1


def test_lit_rejects_other_types() -> None:
    with pytest.raises(errors.PySparkTypeError, match="None, bool, int, float or str"):
        F.lit(b"x")


def test_to_pandas_needs_pandas(df: DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match="toPandas needs pandas"):
        df.toPandas()


def test_to_pandas(df: DataFrame) -> None:
    pd = pytest.importorskip("pandas")
    frame = df.orderBy("id").toPandas()
    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1, 2, 3]


def test_arrow_actions_need_pyarrow(df: DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    for action in (df.toArrow, df.collect, df.toPandas, df.show):
        with pytest.raises(ImportError, match=r"krishiv\[pyarrow\]"):
            action()
    assert df.count() == 3  # needs no Arrow library
    assert df.columns == ["id", "name", "v"]


# -- show: formatted as Spark 4.2.0 does (checked against the Spark oracle) -------------------------------------


def test_show(df: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert shown(df.orderBy("id"), capsys) == (
        "+---+----+----+\n"
        "| id|name|   v|\n"
        "+---+----+----+\n"
        "|  1|   a| 1.5|\n"
        "|  2|NULL| 2.5|\n"
        "|  3|   c|NULL|\n"
        "+---+----+----+\n"
    )


def test_show_limits_and_footer(df: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    out = shown(df.orderBy("id").select("id"), capsys, 2)
    assert out == "+---+\n| id|\n+---+\n|  1|\n|  2|\n+---+\nonly showing top 2 rows\n"
    assert shown(df.orderBy("id").select("id"), capsys, 1).endswith("only showing top 1 row\n")
    assert shown(df.select("id"), capsys, 0) == "+---+\n| id|\n+---+\n+---+\nonly showing top 0 rows\n"
    assert shown(df.filter(F.col("id") > 9).select("id"), capsys) == "+---+\n| id|\n+---+\n+---+\n"


def test_show_truncation_and_wide_characters(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    q = spark.sql("select 'abcdefghijklmnopqrstuvwxyz' as long, '日本語' as cjk")
    assert shown(q, capsys) == (
        "+--------------------+------+\n"
        "|                long|   cjk|\n"
        "+--------------------+------+\n"
        "|abcdefghijklmnopq...|日本語|\n"
        "+--------------------+------+\n"
    )
    assert shown(q, capsys, truncate=False) == (
        "+--------------------------+------+\n"
        "|long                      |cjk   |\n"
        "+--------------------------+------+\n"
        "|abcdefghijklmnopqrstuvwxyz|日本語|\n"
        "+--------------------------+------+\n"
    )
    assert shown(q, capsys, truncate=5).splitlines()[3] == "|ab...|日本語|"


def test_show_vertical(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    q = spark.sql("select 1 as a, 'x' as b, 1.5 as c, cast(null as int) as d")
    assert shown(q, capsys, vertical=True) == ("-RECORD 0---\n a   | 1    \n b   | x    \n c   | 1.5  \n d   | NULL \n")
    assert shown(q.filter(F.col("a") > 9), capsys, vertical=True) == "(0 rows)\n"
    two = spark.sql("select 1 as a union all select 2").orderBy("a")
    assert shown(two, capsys, 1, vertical=True) == "-RECORD 0--\n a   | 1   \nonly showing top 1 row\n"


def test_show_rejects_bad_arguments(df: DataFrame) -> None:
    with pytest.raises(errors.PySparkTypeError):
        df.show("3")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.show(truncate="x")  # ty: ignore[invalid-argument-type]


def test_cell_formats_like_spark() -> None:
    f64, f32 = pa.float64(), pa.float32()
    assert [cell(v, f64) for v in (1e10, 1e-5, 100.0, 1234567.0, 12345678.0, 0.001, 0.00099, -2.5)] == [
        "1.0E10",
        "1.0E-5",
        "100.0",
        "1234567.0",
        "1.2345678E7",
        "0.001",
        "9.9E-4",
        "-2.5",
    ]
    assert [cell(v, f64) for v in (float("nan"), float("inf"), float("-inf"), 0.0, -0.0)] == [
        "NaN",
        "Infinity",
        "-Infinity",
        "0.0",
        "-0.0",
    ]
    assert [cell(v, f32) for v in (0.10000000149011612, 100.0, 1e10)] == ["0.1", "100.0", "1.0E10"]
    assert cell(Decimal("5.000"), pa.decimal128(10, 3)) == "5.000"
    assert cell(b"\x01\xff", pa.binary()) == "[01 FF]"
    assert cell(True, pa.bool_()) == "true"
    assert cell(None, pa.int32()) == "NULL"
    from datetime import datetime

    assert cell(datetime(2020, 1, 2, 3, 4, 5, 100000), pa.timestamp("us")) == "2020-01-02 03:04:05.1"
    assert cell(datetime(2020, 1, 2, 3, 4, 5), pa.timestamp("us")) == "2020-01-02 03:04:05"
    assert cell(date(2020, 1, 2), pa.date32()) == "2020-01-02"
    nested = pa.struct([("x", pa.int64()), ("y", pa.string())])
    assert cell({"x": 1, "y": "z"}, nested) == "{1, z}"
    assert cell([("k", 1)], pa.map_(pa.string(), pa.int64())) == "{k -> 1}"
    assert cell([1, None], pa.list_(pa.int64())) == "[1, NULL]"


def test_show_string_without_columns() -> None:
    assert show_string([], [], 20, True, False) == "++\n||\n++\n++\n"


# -- the Arrow C stream -------------------------------------------------------------------------------------


def ranges(table: Any) -> list[tuple[int, int]]:
    """The address ranges of the buffers of every chunk of every column of ``table``."""
    return [
        (b.address, b.address + b.size)
        for col in table.columns
        for ch in col.chunks
        for b in ch.buffers()
        if b is not None
    ]


def inside(buffers: list[tuple[int, int]], source: list[tuple[int, int]]) -> bool:
    return all(any(lo <= a and e <= hi for lo, hi in source) for a, e in buffers)


@pytest.fixture
def big() -> Any:
    n = 100_000
    return pa.table(
        {
            "id": pa.array(range(n), pa.int64()),
            "x": pa.array([0.5] * n, pa.float64()),
            "name": pa.array([f"name-{i % 1000}" for i in range(n)]),
        }
    )


def test_pyarrow_reads_operator_output_without_copying(spark: SparkSession, big: Any) -> None:
    source = ranges(big)
    out = pa.table(spark.createDataFrame(big).select("id", "x"))
    assert out.num_rows == big.num_rows
    assert out.column("id").to_pylist()[-1] == big.num_rows - 1
    # Every output buffer is a slice of a source buffer: the operators passed the memory through.
    assert inside(ranges(out), source)
    assert out.column("id").num_chunks > 1


def test_variable_width_data_is_shared_with_the_source(spark: SparkSession, big: Any) -> None:
    out = pa.table(spark.createDataFrame(big).select("name"))
    data_buffers = [
        (b.address, b.address + b.size)
        for ch in out.column("name").chunks
        for b in ch.buffers()[2:]  # views of a string_view array: the variadic data buffers
        if b is not None
    ]
    assert data_buffers
    assert inside(data_buffers, ranges(big.select(["name"])))
    assert out.column("name").to_pylist()[5] == "name-5"


def test_polars_reads_operator_output_without_copying(spark: SparkSession, big: Any) -> None:
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame(spark.createDataFrame(big).select("id", "x"))
    assert frame.height == big.num_rows
    assert frame["id"].sum() == sum(range(big.num_rows))
    assert inside(ranges(frame.to_arrow()), ranges(big))


def test_duckdb_reads_operator_output(spark: SparkSession, big: Any) -> None:
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    relation = con.from_arrow(spark.createDataFrame(big).select("id", "name"))
    assert relation.aggregate("sum(id), count(distinct name)").fetchone() == (sum(range(big.num_rows)), 1000)


def test_dataframe_is_a_stream_for_duckdb_twice(df: DataFrame) -> None:
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    for _ in range(2):  # each scan executes the query again
        assert con.from_arrow(df).aggregate("sum(id)").fetchone() == (6,)


def test_requested_schema_and_capsule(df: DataFrame) -> None:
    own = df.toArrow().schema
    wanted = pa.schema([(name, own.field(i).type) for i, name in enumerate("abc")])
    capsule = df.__arrow_c_stream__(wanted.__arrow_c_schema__())
    got = pa.RecordBatchReader._import_from_c_capsule(capsule).read_all()
    assert got.schema.names == ["a", "b", "c"]
    with pytest.raises(ValueError, match="field"):
        df.__arrow_c_stream__(pa.schema([("a", pa.int64())]).__arrow_c_schema__())


def test_releasing_the_stream_cancels_without_blocking(spark: SparkSession, big: Any) -> None:
    df = spark.createDataFrame(big)
    reader = pa.RecordBatchReader.from_stream(df)
    first = reader.read_next_batch()
    assert first.num_rows > 0
    del reader  # releases the stream mid-query; must neither block nor raise
    gc.collect()
    capsule = df.__arrow_c_stream__()
    del capsule  # never consumed: the capsule destructor releases the stream
    gc.collect()
    assert df.count() == big.num_rows  # the session still works


def test_a_stream_released_unread_leaves_no_state(spark: SparkSession, big: Any) -> None:
    df = spark.createDataFrame(big).select("id")
    for _ in range(20):
        df.__arrow_c_stream__()
    assert df.count() == big.num_rows


# -- temp views and file reads ------------------------------------------------------------------------------


@pytest.fixture
def parquet_file(tmp_path: Path) -> str:
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"k": [1, 2, 3, 4], "g": ["a", "b", "a", "b"]}), path)
    return str(path)


def test_read_parquet_and_actions(spark: SparkSession, parquet_file: str) -> None:
    df = spark.read.parquet(parquet_file)
    assert df.columns == ["k", "g"]
    assert df.count() == 4
    assert df.orderBy("k").collect()[0] == Row(k=1, g="a")
    assert df.groupBy("g").agg(F.count("*").alias("n")).orderBy("g").collect() == [Row(g="a", n=2), Row(g="b", n=2)]


def test_create_or_replace_temp_view(spark: SparkSession, parquet_file: str, tmp_path: Path) -> None:
    spark.read.parquet(parquet_file).createOrReplaceTempView("t")
    assert spark.sql("select sum(k) as s from t").collect() == [Row(s=10)]
    other = tmp_path / "other.parquet"
    pq.write_table(pa.table({"k": [7, 8]}), other)
    spark.read.parquet(str(other)).createOrReplaceTempView("t")  # replaces
    assert spark.sql("select count(*) as n from t").collect() == [Row(n=2)]
    with pytest.raises(errors.PySparkTypeError):
        spark.read.parquet(parquet_file).createOrReplaceTempView(1)  # ty: ignore[invalid-argument-type]


def test_temp_view_of_a_computed_dataframe_is_not_supported_yet(df: DataFrame) -> None:
    with pytest.raises(errors.UnsupportedOperationException, match="temporary views other than reads by path"):
        df.createOrReplaceTempView("v")


def test_string_expressions_are_listed_as_not_implemented(df: DataFrame) -> None:
    for call in (lambda: df.selectExpr("id + 1"), lambda: df.where("id > 1"), lambda: F.expr("id + 1")):
        with pytest.raises(errors.PySparkNotImplementedError):
            call()


def test_release_does_not_wait_for_the_query(spark: SparkSession, big: Any) -> None:
    """Releasing a stream of a 10^10-row cross join returns at once (cancellation does not join the engine), and
    the session keeps working. That the release drops the engine stream, with no poll afterwards, is checked
    deterministically in ``stream_ffi`` (``releasing_the_exported_query_stream_drops_the_engine_stream``); the
    executor's cancel-on-drop in ``krishiv-exec``'s ``executor`` tests."""
    import time

    left = spark.createDataFrame(big.select(["id"]))
    right = spark.createDataFrame(big.select(["x"]))
    reader = pa.RecordBatchReader.from_stream(left.crossJoin(right))
    reader.read_next_batch()
    started = time.perf_counter()
    del reader
    gc.collect()
    assert time.perf_counter() - started < 5
    assert spark.sql("select 1 as a").collect() == [Row(a=1)]
