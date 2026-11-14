# SPDX-License-Identifier: Apache-2.0
"""The PySpark names the 0.2.0 acceptance sweep found missing (PY-010, 017, 018, 026, 027, 028, 033, 035,
037, 040). Expected values come from a real PySpark 4.2.0 run (``ops/scripts/spark-local.sh``)."""

from __future__ import annotations

import pytest

import krishiv
from krishiv import errors
from krishiv.sql import Row, SparkSession
from krishiv.sql import functions as F
from krishiv.sql import types as T
from krishiv.sql.catalog import CatalogMetadata, Database, Table

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def df(spark: SparkSession):
    return spark.createDataFrame([(1, "a", 1.5), (2, "b", 2.5)], "a long, b string, c double")


# Row (PY-010, PY-028) -----------------------------------------------------------------------------


def test_row_as_dict_recursive() -> None:
    r = Row(a=Row(b=[Row(c=1)]), d={"k": Row(e=1)})
    assert r.asDict(True) == {"a": {"b": [{"c": 1}]}, "d": {"k": {"e": 1}}}
    assert isinstance(r.asDict()["a"], Row)
    assert r.asDict(recursive=False) == {"a": r.a, "d": r.d}


def test_row_without_fields_cannot_be_a_dict() -> None:
    with pytest.raises(errors.PySparkTypeError, match="CANNOT_CONVERT_TYPE"):
        Row(1, 2).asDict()


def test_row_class_is_callable() -> None:
    r = Row("x", "y")
    assert repr(r) == "<Row('x', 'y')>"
    assert r(1, 2) == Row(x=1, y=2)
    assert repr(r(1, 2)) == "Row(x=1, y=2)"
    assert r(1) == Row(x=1)
    with pytest.raises(errors.PySparkValueError, match=r"TOO_MANY_VALUES\] Expected 2 values for `fields`, got 3"):
        r(1, 2, 3)


def test_collect_keeps_duplicate_field_names(spark: SparkSession, df) -> None:
    """PY-037: ``struct("a", "a")`` and ``arrays_zip`` of a column with itself."""
    rows = df.select(F.struct("a", "a")).sort("a").collect()
    assert rows[0][0] == Row(a=1, a=1) if False else tuple(rows[0][0]) == (1, 1)
    assert rows[0][0].__fields__ == ["a", "a"]
    nested = spark.sql("select array(named_struct('n', 1, 'n', 2)) as xs, map(1, named_struct('n', 3, 'n', 4)) as m")
    [row] = nested.collect()
    assert [tuple(x) for x in row.xs] == [(1, 2)]
    assert [tuple(v) for v in row.m.values()] == [(3, 4)]
    assert spark.sql("select cast(null as struct<n: int, n: int>) as s").collect() == [Row(s=None)]


# printSchema (PY-040) -----------------------------------------------------------------------------


def test_print_schema_names_types_as_spark(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    spark.sql(
        "select cast(1 as int) i, cast(1 as smallint) s, cast(1 as tinyint) t, cast(1 as bigint) l, "
        "cast(1.5 as float) fl, cast(1.5 as double) d, cast(1 as decimal(5,2)) dc, true bo, 'x' st, "
        "array(1, 2) ar, map('k', 1) m, named_struct('a', 1, 'b', 'x') nested, current_date() dt, "
        "current_timestamp() ts, X'00' bin"
    ).printSchema()
    assert capsys.readouterr().out == (
        "root\n"
        " |-- i: integer (nullable = false)\n"
        " |-- s: short (nullable = false)\n"
        " |-- t: byte (nullable = false)\n"
        " |-- l: long (nullable = false)\n"
        " |-- fl: float (nullable = false)\n"
        " |-- d: double (nullable = false)\n"
        " |-- dc: decimal(5,2) (nullable = false)\n"
        " |-- bo: boolean (nullable = false)\n"
        " |-- st: string (nullable = false)\n"
        " |-- ar: array (nullable = false)\n"
        " |    |-- element: integer (containsNull = false)\n"
        " |-- m: map (nullable = false)\n"
        " |    |-- key: string\n"
        " |    |-- value: integer (valueContainsNull = false)\n"
        " |-- nested: struct (nullable = false)\n"
        " |    |-- a: integer (nullable = false)\n"
        " |    |-- b: string (nullable = false)\n"
        " |-- dt: date (nullable = false)\n"
        " |-- ts: timestamp (nullable = false)\n"
        " |-- bin: binary (nullable = false)\n"
        "\n"
    )


def test_print_schema_level(df, capsys: pytest.CaptureFixture[str]) -> None:
    nested = df.select(F.struct("a", "b").alias("s"))
    nested.printSchema(1)
    assert capsys.readouterr().out == "root\n |-- s: struct (nullable = false)\n\n"
    assert nested.schema.treeString(1) == "root\n |-- s: struct (nullable = false)\n"


# DataFrame methods (PY-017) -----------------------------------------------------------------------


def test_is_empty_and_streaming(df) -> None:
    assert df.isEmpty() is False
    assert df.limit(0).isEmpty() is True
    assert df.isStreaming is False


def test_rollup_cube_grouping_sets(df) -> None:
    rolled = df.rollup("b").count().sort("b").collect()
    assert rolled == [Row(b=None, count=2), Row(b="a", count=1), Row(b="b", count=1)]
    cubed = df.cube("a", "b").count().collect()
    assert len(cubed) == 1 + 2 + 2 + 2
    assert Row(a=None, b=None, count=2) in cubed
    sets = df.groupingSets([["a"], []], "a").count().sort("a").collect()
    assert sets == [Row(a=None, count=2), Row(a=1, count=1), Row(a=2, count=1)]
    summed = df.rollup("a", "b").agg(F.sum("c").alias("s")).collect()
    assert Row(a=None, b=None, s=4.0) in summed
    assert Row(a=1, b=None, s=1.5) in summed


def test_with_columns_replaces_in_place_and_adds(df) -> None:
    out = df.withColumns({"c": F.col("a") + 10, "x": F.lit(1), "y": "a"})
    assert out.columns == ["a", "b", "c", "x", "y"]
    assert out.sort("a").collect()[0] == Row(a=1, b="a", c=11, x=1, y=1)
    assert df.withColumns({"A": F.lit(5)}).columns == ["A", "b", "c"]
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        df.withColumns({"x": 1})  # ty: ignore[invalid-argument-type]


def test_col_regex(df) -> None:
    assert df.select(df.colRegex("`^[ab]$`")).columns == ["a", "b"]
    assert df.select(df.colRegex("`^A$`")).columns == ["a"]
    assert df.select(df.colRegex("`x.*`")).columns == []
    assert df.select("c", df.colRegex("`a`")).columns == ["c", "a"]


def test_sort_within_partitions_and_repartition_by_range(df) -> None:
    assert [r.a for r in df.sortWithinPartitions("a", ascending=False).collect()] == [2, 1]
    assert df.repartitionByRange(2, "a").count() == 2
    assert df.repartitionByRange("a").count() == 2
    with pytest.raises(errors.PySparkValueError, match="CANNOT_BE_EMPTY"):
        df.repartitionByRange(2)


def test_create_temp_view(spark: SparkSession, df) -> None:
    df.createTempView("v1")
    assert spark.sql("select count(*) from v1").collect()[0][0] == 2
    with pytest.raises(errors.AnalysisException, match="TEMP_TABLE_OR_VIEW_ALREADY_EXISTS"):
        df.createTempView("v1")
    df.createOrReplaceTempView("v1")


def test_global_temp_views_are_not_implemented(df) -> None:
    with pytest.raises(errors.PySparkNotImplementedError, match="NOT_IMPLEMENTED"):
        df.createGlobalTempView("g")
    with pytest.raises(errors.PySparkNotImplementedError, match="NOT_IMPLEMENTED"):
        df.createOrReplaceGlobalTempView("g")


def test_to_schema(spark: SparkSession, df) -> None:
    out = df.to(T.StructType([T.StructField("B", T.StringType()), T.StructField("a", T.StringType())]))
    assert out.columns == ["B", "a"]
    assert out.sort("a").collect()[0] == Row(B="a", a="1")
    # PySpark 4.2: a field the DataFrame lacks reads NULL.
    assert df.to(T.StructType([T.StructField("zz", T.IntegerType())])).sort("zz").collect() == [
        Row(zz=None),
        Row(zz=None),
    ]
    with pytest.raises(errors.PySparkTypeError):
        df.to("a int")  # ty: ignore[invalid-argument-type]


def test_foreach_and_foreach_partition(df) -> None:
    seen: list[int] = []
    df.sort("a").foreach(lambda row: seen.append(row.a))
    assert seen == [1, 2]
    parts: list[list[int]] = []
    df.sort("a").foreachPartition(lambda rows: parts.append([r.a for r in rows]))
    assert parts == [[1, 2]]


def test_to_json_omits_null_fields(spark: SparkSession) -> None:
    out = spark.sql("select 1 as a, cast(null as int) as g, 'x' as s").toJSON()
    assert [r.value for r in out.collect()] == ['{"a":1,"s":"x"}']


def test_same_semantics(df) -> None:
    assert df.select("a").sameSemantics(df.select("a"))
    assert not df.select("a").sameSemantics(df.select("b"))
    assert df.select("a").semanticHash() == df.select("a").semanticHash()
    assert -(2**31) <= df.semanticHash() < 2**31


@pytest.mark.parametrize(
    "call",
    [
        lambda df: df.randomSplit([0.5, 0.5], seed=1),
        lambda df: df.inputFiles(),
        lambda df: df.writeStream,
        lambda df: df.withWatermark("a", "1 minute"),
        lambda df: df.dropDuplicatesWithinWatermark(),
    ],
)
def test_names_the_engine_cannot_run_are_not_implemented(df, call) -> None:
    with pytest.raises(errors.PySparkNotImplementedError, match="NOT_IMPLEMENTED"):
        call(df)


# mapInPandas, applyInPandas, cogroup ---------------------------------------------------------------


pd = pytest.importorskip("pandas")


def test_map_in_pandas(df) -> None:
    def double(batches):
        for pdf in batches:
            pdf["a"] = pdf["a"] * 2
            yield pdf[["a", "b"]]

    out = df.mapInPandas(double, "a long, b string")
    assert out.sort("a").collect() == [Row(a=2, b="a"), Row(a=4, b="b")]


def test_map_in_arrow(df) -> None:
    def keep_big(batches):
        for b in batches:
            yield pa.RecordBatch.from_pandas(b.to_pandas().query("a > 1")[["a"]])

    assert df.mapInArrow(keep_big, "a long").collect() == [Row(a=2)]


def test_apply_in_pandas_with_and_without_key(spark: SparkSession) -> None:
    data = spark.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)], "id long, v double")

    def demean(pdf):
        return pdf.assign(v=pdf.v - pdf.v.mean())

    out = data.groupBy("id").applyInPandas(demean, "id long, v double").sort("id", "v").collect()
    assert out == [
        Row(id=1, v=-0.5),
        Row(id=1, v=0.5),
        Row(id=2, v=-3.0),
        Row(id=2, v=-1.0),
        Row(id=2, v=4.0),
    ]

    def total(key, pdf):
        return pd.DataFrame([{"id": key[0], "total": float(pdf.v.sum())}])

    out = data.groupBy("id").applyInPandas(total, "id long, total double").sort("id").collect()
    assert out == [Row(id=1, total=3.0), Row(id=2, total=18.0)]


def test_apply_in_arrow(spark: SparkSession) -> None:
    data = spark.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0)], "id long, v double")

    def count(key, table):
        return pa.table({"id": [key[0]], "n": [table.num_rows]})

    out = data.groupBy("id").applyInArrow(count, "id long, n long").sort("id").collect()
    assert out == [Row(id=1, n=2), Row(id=2, n=1)]


def test_cogroup_apply_in_pandas(spark: SparkSession) -> None:
    left = spark.createDataFrame([(1, 10), (2, 20)], "id long, x long")
    right = spark.createDataFrame([(1, 100), (1, 200), (3, 300)], "id long, y long")

    def merge(key, lpdf, rpdf):
        return pd.DataFrame([{"id": key[0], "nx": len(lpdf), "ny": len(rpdf)}])

    out = left.groupBy("id").cogroup(right.groupBy("id")).applyInPandas(merge, "id long, nx long, ny long")
    assert out.sort("id").collect() == [Row(id=1, nx=1, ny=2), Row(id=2, nx=1, ny=0), Row(id=3, nx=0, ny=1)]


def test_grouped_map_over_an_expression_key_is_not_implemented(df) -> None:
    with pytest.raises(errors.PySparkNotImplementedError, match="NOT_IMPLEMENTED"):
        df.groupBy(F.col("a") + 1).applyInPandas(lambda p: p, "a long")
    with pytest.raises(errors.PySparkNotImplementedError, match="NOT_IMPLEMENTED"):
        df.groupBy("a").apply(None)


# reader and writer (PY-018, PY-033) ----------------------------------------------------------------


def test_reader_table_and_text(spark: SparkSession, df, tmp_path) -> None:
    df.createOrReplaceTempView("t")
    assert spark.read.table("t").count() == 2
    path = tmp_path / "x.txt"
    path.write_text("one\ntwo\n")
    assert [r.value for r in spark.read.text(str(path)).sort("value").collect()] == ["one", "two"]
    assert spark.read.text(str(path), wholetext=True).collect() == [Row(value="one\ntwo\n")]


def test_orc_and_unsupported_readers_and_writers(spark: SparkSession, df, tmp_path) -> None:
    with pytest.raises(errors.PySparkException, match="orc"):
        df.write.orc(str(tmp_path / "o"))
    with pytest.raises(errors.PySparkException, match="orc"):
        spark.read.orc(str(tmp_path / "o")).collect()
    with pytest.raises(errors.UnsupportedOperationException, match="KRISHIV_UNSUPPORTED_FEATURE"):
        df.write.xml(str(tmp_path / "x"))
    with pytest.raises(errors.UnsupportedOperationException, match="KRISHIV_UNSUPPORTED_FEATURE"):
        df.write.clusterBy("a").parquet(str(tmp_path / "p"))
    with pytest.raises(errors.PySparkTypeError):
        df.write.clusterBy(1)  # ty: ignore[invalid-argument-type]


# session (PY-035) ------------------------------------------------------------------------------------


def test_active_session() -> None:
    SparkSession.builder.getOrCreate().stop()
    assert SparkSession.getActiveSession() is None
    with pytest.raises(errors.PySparkRuntimeError, match="NO_ACTIVE_OR_DEFAULT_SESSION"):
        SparkSession.active()
    s = SparkSession.builder.getOrCreate()
    try:
        assert SparkSession.getActiveSession() is s
        assert SparkSession.active() is s
        assert s.getActiveSession() is s
    finally:
        s.stop()


@pytest.mark.parametrize(
    "name",
    ["sparkContext", "readStream", "streams", "tvf", "udtf", "dataSource", "profile"],
)
def test_session_properties_the_engine_lacks_are_not_implemented(spark: SparkSession, name: str) -> None:
    with pytest.raises(errors.PySparkNotImplementedError, match="NOT_IMPLEMENTED"):
        getattr(spark, name)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.addArtifacts("x"),
        lambda s: s.interruptAll(),
        lambda s: s.interruptTag("t"),
        lambda s: s.interruptOperation("o"),
        lambda s: s.addTag("t"),
        lambda s: s.getTags(),
    ],
)
def test_session_calls_the_engine_lacks_are_not_implemented(spark: SparkSession, call) -> None:
    with pytest.raises(errors.PySparkNotImplementedError, match="NOT_IMPLEMENTED"):
        call(spark)


# Catalog (PY-026) --------------------------------------------------------------------------------------


def test_catalog_returns_pyspark_lists(spark: SparkSession, df) -> None:
    cat = spark.catalog
    df.createOrReplaceTempView("v1")
    assert cat.currentCatalog() == "spark_catalog"
    assert cat.listCatalogs() == [CatalogMetadata("spark_catalog", None)]
    assert cat.listCatalogs("nope*") == []
    dbs = cat.listDatabases()
    assert [d.name for d in dbs] == ["default"]
    assert isinstance(dbs[0], Database)
    assert cat.getDatabase("default").name == "default"
    assert cat.databaseExists("default") and not cat.databaseExists("nope")
    cat.setCurrentDatabase("default")
    with pytest.raises(errors.AnalysisException, match="SCHEMA_NOT_FOUND"):
        cat.setCurrentDatabase("nope")
    tables = cat.listTables()
    assert tables == [Table("v1", None, [], None, "TEMPORARY", True)]
    assert cat.listTables("default", "v*") == tables
    assert cat.listTables(pattern="x*") == []
    assert cat.getTable("v1") == tables[0]
    with pytest.raises(errors.AnalysisException, match="TABLE_OR_VIEW_NOT_FOUND"):
        cat.getTable("nope")
    cols = cat.listColumns("v1")
    assert [(c.name, c.dataType, c.nullable) for c in cols] == [("a", "bigint", True), ("b", "string", True), ("c", "double", True)]
    assert cols[0].isPartition is False


def test_catalog_functions(spark: SparkSession) -> None:
    cat = spark.catalog
    names = [f.name for f in cat.listFunctions()]
    assert "abs" in names
    assert [f.name for f in cat.listFunctions(pattern="ab*")][:1] == ["abs"]
    assert cat.functionExists("abs") is True
    assert cat.functionExists("no_such_fn") is False
    assert cat.getFunction("abs").name == "abs"
    with pytest.raises(errors.AnalysisException):
        cat.getFunction("no_such_fn")


def test_runtime_config_is_modifiable_and_get_all(spark: SparkSession) -> None:
    assert spark.conf.isModifiable("spark.sql.shuffle.partitions") is True
    assert spark.conf.isModifiable("spark.app.name") is False
    spark.conf.set("spark.sql.shuffle.partitions", "7")
    assert spark.conf.getAll["spark.sql.shuffle.partitions"] == "7"


# build_info (PY-027) -----------------------------------------------------------------------------------


def test_build_info_names_the_commit() -> None:
    info = krishiv.build_info()
    assert {"version", "allocator", "git_sha"} <= set(info)
    assert isinstance(info["git_sha"], str) and info["git_sha"]
