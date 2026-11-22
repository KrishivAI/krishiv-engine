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
    assert repr(r(1)) == "Row(x=1)"
    assert r(1).asDict() == {"x": 1}
    assert r() == Row()
    with pytest.raises(errors.PySparkValueError, match=r"TOO_MANY_VALUES\] Expected 2 values for `fields`, got 3"):
        r(1, 2, 3)


def test_collect_keeps_duplicate_field_names(spark: SparkSession, df) -> None:
    """PY-037: ``struct("a", "a")`` and ``arrays_zip`` of a column with itself."""
    rows = df.select(F.struct("a", "a")).sort("a").collect()
    assert tuple(rows[0][0]) == (1, 1)
    assert rows[0][0].__fields__ == ["a", "a"]
    nested = spark.sql("select array(named_struct('n', 1, 'n', 2)) as xs, map(1, named_struct('n', 3, 'n', 4)) as m")
    [row] = nested.collect()
    assert [tuple(x) for x in row.xs] == [(1, 2)]
    assert [tuple(v) for v in row.m.values()] == [(3, 4)]
    assert spark.sql("select cast(null as struct<n: int, n: int>) as s").collect() == [Row(s=None)]


# printSchema (PY-040) -----------------------------------------------------------------------------


def test_print_schema_names_types_as_spark(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    ddl = (
        "i int, s short, t byte, l long, fl float, d double, dc decimal(5,2), bo boolean, st string, "
        "dt date, ts timestamp, bin binary"
    )
    spark.createDataFrame([], ddl).printSchema()
    assert capsys.readouterr().out == (
        "root\n"
        + "".join(
            f" |-- {n}: {t} (nullable = true)\n"
            for n, t in [
                ("i", "integer"),
                ("s", "short"),
                ("t", "byte"),
                ("l", "long"),
                ("fl", "float"),
                ("d", "double"),
                ("dc", "decimal(5,2)"),
                ("bo", "boolean"),
                ("st", "string"),
                ("dt", "date"),
                ("ts", "timestamp"),
                ("bin", "binary"),
            ]
        )
        + "\n"
    )


def test_tree_string_of_nested_types() -> None:
    schema = T.DataType.fromDDL("ar array<int>, m map<string,int>, nested struct<a:int,b:string>")
    assert isinstance(schema, T.StructType)
    assert schema.treeString() == (
        "root\n"
        " |-- ar: array (nullable = true)\n"
        " |    |-- element: integer (containsNull = true)\n"
        " |-- m: map (nullable = true)\n"
        " |    |-- key: string\n"
        " |    |-- value: integer (valueContainsNull = true)\n"
        " |-- nested: struct (nullable = true)\n"
        " |    |-- a: integer (nullable = true)\n"
        " |    |-- b: string (nullable = true)\n"
    )


def test_print_schema_level_zero_or_negative_is_unlimited(df) -> None:
    nested = df.select(F.struct("a", "b").alias("s"))
    full = nested.schema.treeString()
    assert nested.schema.treeString(0) == full
    assert nested.schema.treeString(-1) == full
    assert "|    |-- a:" in full


def test_foreach_streams_rows(df) -> None:
    seen: list[int] = []
    df.sort("a").foreach(lambda r: seen.append(r.a))
    assert seen == [1, 2]


def test_col_regex_of_a_plain_name_is_that_column(df) -> None:
    assert df.select(df.colRegex("a")).columns == ["a"]
    assert df.select(df.colRegex("t.`^[ab]$`")).columns == ["a", "b"]


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
        df.withColumns({"x": 1})


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


def _struct(*fields: T.StructField) -> T.StructType:
    return T.StructType(list(fields))


def test_to_reorders_casts_and_fills(spark: SparkSession, df) -> None:
    out = df.to(_struct(T.StructField("B", T.StringType()), T.StructField("a", T.StringType())))
    assert out.columns == ["B", "a"]
    assert out.sort("a").collect()[0] == Row(B="a", a="1")
    # PySpark 4.2: a nullable field the DataFrame lacks reads NULL, extra columns are dropped.
    assert df.to(_struct(T.StructField("zz", T.IntegerType()))).sort("zz").collect() == [Row(zz=None), Row(zz=None)]
    with pytest.raises(errors.PySparkTypeError):
        df.to("a int")


def test_to_missing_non_nullable_field_is_unresolved(df) -> None:
    with pytest.raises(errors.AnalysisException, match=r"UNRESOLVED_COLUMN.WITH_SUGGESTION.*`zz`.*\[`a`, `b`, `c`\]"):
        df.to(_struct(T.StructField("zz", T.IntegerType(), False)))


def test_to_nullable_into_non_nullable_is_refused(df) -> None:
    with pytest.raises(errors.AnalysisException, match=r"NULLABLE_COLUMN_OR_FIELD.*`a`"):
        df.to(_struct(T.StructField("a", T.LongType(), False)))


def test_to_string_to_int_is_refused_at_analysis(df) -> None:
    with pytest.raises(errors.AnalysisException, match=r'INVALID_COLUMN_OR_FIELD_DATA_TYPE.*"STRING".*"INT"'):
        df.to(_struct(T.StructField("b", T.IntegerType())))


def test_to_reorders_nested_structs_by_name(spark: SparkSession) -> None:
    nested = spark.sql("select named_struct('p', 1, 'q', 's') s")
    out = nested.to(
        _struct(T.StructField("s", _struct(T.StructField("q", T.StringType()), T.StructField("p", T.LongType()))))
    )
    assert out.collect() == [Row(s=Row(q="s", p=1))]
    with pytest.raises(errors.AnalysisException, match=r"INVALID_COLUMN_OR_FIELD_DATA_TYPE.*`s`.`q`"):
        nested.to(_struct(T.StructField("s", _struct(T.StructField("q", T.IntegerType())))))
    with pytest.raises(errors.AnalysisException, match=r"UNRESOLVED_FIELD.WITH_SUGGESTION.*`z`.*`s`"):
        nested.to(_struct(T.StructField("s", _struct(T.StructField("z", T.IntegerType(), False)))))
    null_struct = spark.sql("select cast(null as struct<p:int>) s")
    assert null_struct.to(_struct(T.StructField("s", _struct(T.StructField("p", T.IntegerType()))))).collect() == [
        Row(s=None)
    ]


def test_to_matches_the_spark_4_2_0_cast_matrix(spark: SparkSession) -> None:
    """256 (from, to) pairs of atomic types, each run on PySpark 4.2.0 (``fixtures/to_schema_matrix_*.json``)."""
    import json
    from pathlib import Path

    matrix = json.loads((Path(__file__).parent / "fixtures" / "to_schema_matrix_spark_4_2_0.json").read_text())
    wrong = []
    for pair, allowed in matrix.items():
        have, want = pair.split("->")
        frame = spark.sql(f"select cast(null as {have}) x")
        try:
            _ = frame.to(_struct(T.StructField("x", T.DataType.fromDDL(want)))).schema
            got = True
        except errors.AnalysisException:
            got = False
        if got != allowed:
            wrong.append((pair, allowed, got))
    assert wrong == []


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


def test_to_json_of_string_columns_from_data(spark: SparkSession) -> None:
    """``struct()`` over a string column of a table typed Utf8 inside a Utf8View struct (engine bug, fixed)."""
    data = spark.createDataFrame([(1, "x", 2.0), (2, None, 3.0)], "a int, b string, c double")
    assert [r.value for r in data.sort("a").toJSON().collect()] == ['{"a":1,"b":"x","c":2.0}', '{"a":2,"c":3.0}']
    assert data.select(F.struct("a", "b")).sort("a").collect()[0][0] == Row(a=1, b="x")


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


# mapInPandas, applyInPandas, cogroup: need the Python UDF worker pool (planned) ----------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda df: df.mapInPandas(lambda it: it, "a long"),
        lambda df: df.mapInArrow(lambda it: it, "a long"),
        lambda df: df.groupBy("a").applyInPandas(lambda p: p, "a long"),
        lambda df: df.groupBy("a").applyInArrow(lambda t: t, "a long"),
        lambda df: df.groupBy("a").cogroup(df.groupBy("a")),
        lambda df: df.groupBy("a").apply(None),
    ],
)
def test_map_functions_are_refused_until_the_udf_worker_pool_runs_them(df, call) -> None:
    with pytest.raises(errors.PySparkNotImplementedError, match=r"NOT_IMPLEMENTED.*worker pool"):
        call(df)


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
        df.write.clusterBy(1)


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
    ["readStream", "streams", "tvf", "udtf", "dataSource", "profile"],
)
def test_session_properties_the_engine_lacks_are_not_implemented(spark: SparkSession, name: str) -> None:
    with pytest.raises(errors.PySparkNotImplementedError, match="NOT_IMPLEMENTED"):
        getattr(spark, name)
    # ... and an AttributeError, so probing code still works.
    assert hasattr(spark, name) is False
    assert getattr(spark, name, "fallback") == "fallback"


def test_spark_context_is_a_jvm_attribute_error_as_in_connect(spark: SparkSession) -> None:
    with pytest.raises(errors.PySparkAttributeError, match="JVM_ATTRIBUTE_NOT_SUPPORTED"):
        _ = spark.sparkContext
    assert hasattr(spark, "sparkContext") is False


def test_write_stream_is_an_attribute_error_too(df) -> None:
    assert hasattr(df, "writeStream") is False


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
    assert cat.databaseExists("default")
    assert not cat.databaseExists("nope")
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
    assert [(c.name, c.dataType, c.nullable) for c in cols] == [
        ("a", "bigint", True),
        ("b", "string", True),
        ("c", "double", True),
    ]
    assert cols[0].isPartition is False


def test_catalog_function_exists_is_the_sql_function_namespace(spark: SparkSession) -> None:
    cat = spark.catalog
    for dataframe_only in ("col", "lit", "udf", "column"):
        assert cat.functionExists(dataframe_only) is False
        assert dataframe_only not in [f.name for f in cat.listFunctions()]
    assert cat.functionExists("ABS") is True
    assert cat.functionExists("abs", "default") is True
    for bad in ("a b", "abs(1) + abs", "x;y"):
        with pytest.raises(errors.ParseException, match="PARSE_SYNTAX_ERROR"):
            cat.functionExists(bad)


def test_catalog_name_patterns_are_spark_regexes(spark: SparkSession, df) -> None:
    df.createOrReplaceTempView("t_one")
    df.createOrReplaceTempView("t_two")
    names = lambda pattern: sorted(t.name for t in spark.catalog.listTables(pattern=pattern))  # noqa: E731
    assert names("t_*") == ["t_one", "t_two"]
    assert names("T_ONE|nope") == ["t_one"]
    assert names("t_[a-z]ne") == ["t_one"]
    assert names("t.one") == ["t_one"]


def test_catalog_get_table_error_has_parameters_and_hides_internal_tables(spark: SparkSession, df) -> None:
    with pytest.raises(errors.AnalysisException) as info:
        spark.catalog.getTable("nope")
    assert info.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"
    assert info.value.getMessageParameters() == {"relationName": "`nope`"}
    [hidden] = [n for n in spark._native.temp_table_names() if n.startswith("__krishiv_arrow_")][:1]
    with pytest.raises(errors.AnalysisException, match="TABLE_OR_VIEW_NOT_FOUND"):
        spark.catalog.getTable(hidden)


def test_catalog_create_table_does_not_replace(spark: SparkSession, df, tmp_path) -> None:
    path = str(tmp_path / "p")
    df.write.parquet(path)
    assert spark.catalog.createTable("created", path, "parquet").count() == 2
    with pytest.raises(errors.AnalysisException, match="TEMP_TABLE_OR_VIEW_ALREADY_EXISTS"):
        spark.catalog.createTable("created", path, "parquet")


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
    assert spark.conf.isModifiable("krishiv.memory.limit") is False
    assert spark.conf.isModifiable("no.such.key") is False
    spark.conf.set("spark.sql.session.timeZone", "Europe/Paris")
    # The engine names the key `krishiv.sql.session.timeZone`.
    assert "Europe/Paris" in spark.conf.getAll.values()


# build_info (PY-027) -----------------------------------------------------------------------------------


def test_build_info_names_the_commit() -> None:
    info = krishiv.build_info()
    assert {"version", "allocator", "git_sha"} <= set(info)
    assert isinstance(info["git_sha"], str)
    assert info["git_sha"]
