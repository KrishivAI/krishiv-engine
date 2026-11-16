# SPDX-License-Identifier: Apache-2.0
"""``spark.catalog`` returns lists of the records ``pyspark.sql.catalog`` defines, not DataFrames (PY-026)."""

from __future__ import annotations

import pytest

from krishiv.errors import AnalysisException
from krishiv.sql import SparkSession
from krishiv.sql.catalog import CatalogMetadata, Column, Database, Function, Table

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def test_list_tables_is_a_list_of_table_records(spark: SparkSession) -> None:
    spark.sql("CREATE TEMPORARY VIEW v AS SELECT 1 AS a, 'x' AS b")
    tables = spark.catalog.listTables()
    assert isinstance(tables, list)
    [v] = [t for t in tables if t.name == "v"]
    assert isinstance(v, Table)
    assert v == Table(name="v", catalog=None, namespace=[], description=None, tableType="TEMPORARY", isTemporary=True)
    assert v._fields == ("name", "catalog", "namespace", "description", "tableType", "isTemporary")


def test_create_dataframe_tables_stay_hidden(spark: SparkSession) -> None:
    spark.createDataFrame([(1,)], ["a"])
    assert spark.catalog.listTables() == []


def test_get_table_and_exists(spark: SparkSession) -> None:
    spark.sql("CREATE TEMPORARY VIEW v AS SELECT 1 AS a")
    assert spark.catalog.getTable("v").name == "v"
    assert spark.catalog.tableExists("v")
    assert spark.catalog.tableExists("V")
    assert not spark.catalog.tableExists("nope")
    with pytest.raises(AnalysisException) as e:
        spark.catalog.getTable("nope")
    assert "TABLE_OR_VIEW_NOT_FOUND" in str(e.value)


def test_databases_and_current(spark: SparkSession) -> None:
    dbs = spark.catalog.listDatabases()
    assert [d.name for d in dbs] == ["default"]
    assert isinstance(dbs[0], Database)
    assert dbs[0]._fields == ("name", "catalog", "description", "locationUri")
    assert spark.catalog.currentDatabase() == "default"
    assert spark.catalog.databaseExists("default")
    assert not spark.catalog.databaseExists("nope")
    assert spark.catalog.getDatabase("default").catalog == "spark_catalog"
    assert [d.name for d in spark.catalog.listDatabases("def*")] == ["default"]
    assert spark.catalog.listDatabases("zzz*") == []


def test_catalogs(spark: SparkSession) -> None:
    assert spark.catalog.currentCatalog() == "spark_catalog"
    assert CatalogMetadata(name="spark_catalog", description=None) in spark.catalog.listCatalogs()
    spark.catalog.setCurrentCatalog("SPARK_CATALOG")
    with pytest.raises(AnalysisException) as e:
        spark.catalog.setCurrentCatalog("no_such_catalog")
    assert "CATALOG_NOT_FOUND" in str(e.value)
    assert spark.catalog.currentCatalog() == "spark_catalog"


def test_list_columns(spark: SparkSession) -> None:
    spark.sql("CREATE TEMPORARY VIEW v AS SELECT 1 AS a, 'x' AS b")
    cols = spark.catalog.listColumns("v")
    assert [c.name for c in cols] == ["a", "b"]
    assert all(isinstance(c, Column) for c in cols)
    assert cols[0].dataType == "int"
    assert cols[0]._fields == ("name", "description", "dataType", "nullable", "isPartition", "isBucket", "isCluster")


def test_functions(spark: SparkSession) -> None:
    assert spark.catalog.functionExists("abs")
    assert not spark.catalog.functionExists("no_such_function")
    f = spark.catalog.getFunction("abs")
    assert isinstance(f, Function)
    assert f.name == "abs"
    assert f.isTemporary
    assert any(x.name == "abs" for x in spark.catalog.listFunctions())
    assert [x.name for x in spark.catalog.listFunctions(pattern="abs")] == ["abs"]
    with pytest.raises(AnalysisException):
        spark.catalog.getFunction("no_such_function")


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("setCurrentCatalog", (1,)),
        ("listCatalogs", (1,)),
        ("setCurrentDatabase", (1,)),
        ("listDatabases", (1,)),
        ("getDatabase", (1,)),
        ("databaseExists", (1,)),
        ("listTables", (1,)),
        ("listTables", (None, 1)),
        ("getTable", (1,)),
        ("tableExists", (1,)),
        ("tableExists", ("t", 1)),
        ("listColumns", (1,)),
        ("listColumns", ("t", 1)),
        ("listFunctions", (1,)),
        ("listFunctions", (None, 1)),
        ("getFunction", (1,)),
        ("functionExists", (1,)),
        ("functionExists", ("f", 1)),
    ],
)
def test_argument_types_are_checked(spark: SparkSession, method: str, args: tuple[object, ...]) -> None:
    with pytest.raises(TypeError):
        getattr(spark.catalog, method)(*args)


def test_function_and_database_arguments(spark: SparkSession) -> None:
    assert spark.catalog.listFunctions("default", "abs")[0].name == "abs"
    assert isinstance(spark.catalog.functionExists("abs", "default"), bool)  # built-ins live in no database
    assert not spark.catalog.functionExists("no_such_function")
    for call in (lambda: spark.catalog.listFunctions("nope"), lambda: spark.catalog.getDatabase("nope")):
        with pytest.raises(AnalysisException) as err:
            call()
        assert err.value.getCondition() == "SCHEMA_NOT_FOUND"
    assert spark.catalog.databaseExists("default")
    assert not spark.catalog.databaseExists("nope")


def test_dataframe_tables_are_not_found_by_name(spark: SparkSession) -> None:
    before = set(spark._native.temp_table_names())
    df = spark.createDataFrame(pa.table({"a": [1]}))  # the table lives as long as the frame
    [hidden] = set(spark._native.temp_table_names()) - before
    assert not spark.catalog.tableExists(hidden)
    assert df.count() == 1
    for call in (spark.catalog.getTable, spark.catalog.listColumns):
        with pytest.raises(AnalysisException) as err:
            call(hidden)
        assert err.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"


def test_a_permanent_table_and_the_database_paths(spark: SparkSession) -> None:
    spark.sql("CREATE DATABASE d1")
    spark.sql("CREATE TABLE d1.t (a INT, b STRING) USING parquet")
    [t] = spark.catalog.listTables("d1")
    assert (t.name, t.catalog, t.namespace, t.tableType, t.isTemporary) == (
        "t",
        "spark_catalog",
        ["d1"],
        "MANAGED",
        False,
    )
    assert t.database == "d1"
    assert spark.catalog.tableExists("t", "d1")
    assert [c.name for c in spark.catalog.listColumns("t", "d1")] == ["a", "b"]
    spark.catalog.setCurrentDatabase("d1")
    assert spark.catalog.currentDatabase() == "d1"
    assert spark.catalog.tableExists("t")
    with pytest.raises(AnalysisException) as err:
        spark.catalog.setCurrentDatabase("nope")
    assert err.value.getCondition() == "SCHEMA_NOT_FOUND"
    with pytest.raises(AnalysisException) as err:
        spark.catalog.listTables("nope")
    assert err.value.getCondition() == "SCHEMA_NOT_FOUND"


def test_list_catalogs_pattern(spark: SparkSession) -> None:
    assert [c.name for c in spark.catalog.listCatalogs("spark*")] == ["spark_catalog"]
    assert spark.catalog.listCatalogs("other") == []
