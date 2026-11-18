# SPDX-License-Identifier: Apache-2.0
"""Names of Spark that Krishiv lacks give the engine's ``KRISHIV_UNSUPPORTED_FEATURE`` (PY-045,
PY-046, PY-050, PY-051), never a bare ``AttributeError`` or ``ModuleNotFoundError``."""

from __future__ import annotations

import importlib
import warnings

import pytest

import krishiv
from krishiv.errors import UnsupportedOperationException


@pytest.fixture(scope="module")
def spark() -> krishiv.SparkSession:
    return krishiv.SparkSession.builder.create()


def _refused(excinfo: pytest.ExceptionInfo[UnsupportedOperationException], needle: str) -> None:
    error = excinfo.value
    assert error.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert error.getSqlState() == "0A000"
    assert needle in str(error)


def test_spark_context_is_refused_with_the_dataframe_way_named(spark: krishiv.SparkSession) -> None:
    with pytest.raises(UnsupportedOperationException) as info:
        _ = spark.sparkContext
    _refused(info, "use the DataFrame API")


def test_dataframe_rdd_is_refused_with_the_dataframe_way_named(spark: krishiv.SparkSession) -> None:
    with pytest.raises(UnsupportedOperationException) as info:
        _ = spark.range(3).rdd
    _refused(info, "use the DataFrame API")


def test_spark_conf_is_refused_with_the_session_way_named() -> None:
    with pytest.raises(UnsupportedOperationException) as info:
        krishiv.SparkConf()
    _refused(info, "SparkSession.builder.config")


@pytest.mark.parametrize(("module", "needle"), [("krishiv.pandas", "toPandas"), ("krishiv.ml", "MLlib")])
def test_importing_an_absent_module_names_the_feature(module: str, needle: str) -> None:
    with pytest.raises(UnsupportedOperationException) as info:
        importlib.import_module(module)
    _refused(info, needle)


def test_hive_and_jdbc_names_are_refused(spark: krishiv.SparkSession) -> None:
    with pytest.raises(UnsupportedOperationException) as info:
        krishiv.SparkSession.builder.enableHiveSupport()
    _refused(info, "Hive")
    df = spark.range(2)
    with pytest.raises(UnsupportedOperationException) as info:
        df.write.jdbc("jdbc:postgresql://localhost/db", "t")
    _refused(info, "JDBC")
    with pytest.raises(UnsupportedOperationException) as info:
        spark.read.jdbc("jdbc:postgresql://localhost/db", "t")
    _refused(info, "JDBC")


@pytest.mark.parametrize("how", ["reader", "writer"])
def test_format_jdbc_is_refused_where_it_is_named(spark: krishiv.SparkSession, how: str) -> None:
    # Not later as `NOT_EXPECTED_TYPE Argument path should be str`.
    target = spark.read if how == "reader" else spark.range(1).write
    with pytest.raises(UnsupportedOperationException) as info:
        target.format("JDBC")
    _refused(info, "jdbc")


def test_cluster_settings_warn_and_still_start() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        builder = (
            krishiv.SparkSession.builder.master("yarn")
            .config("spark.executor.instances", "4")
            .config("spark.submit.deployMode", "cluster")
        )
        session = builder.create()
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(m.startswith("spark.master=yarn has no effect") for m in messages)
    assert any(m.startswith("spark.executor.instances=4 has no effect") for m in messages)
    assert any(m.startswith("spark.submit.deployMode=cluster has no effect") for m in messages)
    assert session.range(2).count() == 2


def test_local_settings_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        krishiv.SparkSession.builder.master("local[*]").config("spark.sql.shuffle.partitions", "4").create()


def test_pandas_utils_are_importable_as_in_spark() -> None:
    from krishiv.errors import PySparkImportError
    from krishiv.sql.pandas.utils import (
        require_minimum_pandas_version,
        require_minimum_pyarrow_version,
    )

    # pandas and pyarrow are optional: either they satisfy the minimums or the check says which
    # package is missing, as Spark's does (`PySparkImportError`, an `ImportError`).
    for check, package in ((require_minimum_pandas_version, "Pandas"), (require_minimum_pyarrow_version, "PyArrow")):
        failures: list[PySparkImportError] = []
        try:
            check()
        except PySparkImportError as error:
            failures.append(error)
        assert [(e.getCondition(), package in str(e)) for e in failures] in ([], [("PACKAGE_NOT_INSTALLED", True)])
