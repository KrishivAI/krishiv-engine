# SPDX-License-Identifier: Apache-2.0
"""The slice of ``pyspark.testing`` / ``pyspark.util`` that the adapted PySpark 4.2.0 UDF tests use.

``ReusedSQLTestCase`` follows ``pyspark/testing/sqlutils.py`` (a class-wide session, ``sql_conf`` and the
``table``/``temp_view`` helpers), ``check_error`` follows ``pyspark/testing/utils.py`` (``PySparkErrorTestUtils``).
Both are Apache-2.0 code from apache/spark v4.2.0 (see NOTICE), reduced to what a Krishiv session offers.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import tempfile
import unittest
from contextlib import contextmanager
from typing import Any

from krishiv import _native
from krishiv.errors import (  # noqa: F401  (re-exported names the adapted tests import)
    AnalysisException,
    PySparkException,
    PySparkNotImplementedError,
    PySparkTypeError,
)
from krishiv.sql import Row, SparkSession
from krishiv.sql import session as _session
from krishiv.sql.udf import PythonEvalType  # noqa: F401
from krishiv.testing import assertDataFrameEqual  # noqa: F401

#: Spark's ``PythonException`` is a captured JVM exception; Krishiv raises a ``PySparkException`` with the
#: ``PYTHON_EXCEPTION`` condition. The alias keeps ``assertRaises(PythonException)`` as strict as the class
#: hierarchy allows.
PythonException = PySparkException


def have_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


have_pandas = have_package("pandas")
have_pyarrow = have_package("pyarrow")
have_numpy = have_package("numpy")
numpy_requirement_message = "" if have_numpy else "No module named 'numpy'"
pandas_requirement_message = "" if have_pandas else "No module named 'pandas'"
pyarrow_requirement_message = "" if have_pyarrow else "No module named 'pyarrow'"
test_compiled = True
test_not_compiled_message = ""


def is_remote_only() -> bool:
    return False


def timeout(seconds: int) -> Any:
    """``pyspark.testing.utils.timeout``: an alarm raising ``TimeoutError``."""

    def decorator(func: Any) -> Any:
        def handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)

        return wrapper

    return decorator


class _Unavailable:
    """Stands in for a Spark name Krishiv lacks; tests using it are excluded in the manifest."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise unittest.SkipTest("not available in Krishiv")


class ExamplePoint(_Unavailable):
    pass


class ExamplePointUDT(_Unavailable):
    pass


class PySparkLogger(_Unavailable):
    pass


class TaskContext(_Unavailable):
    pass


class SparkContext(_Unavailable):
    pass


class SQLContext(_Unavailable):
    pass


class _SparkContextStub:
    """``cls.sc``: Krishiv has no SparkContext; only the worker ``environment`` map exists (it has no effect: UDFs
    run in this process, which already has the environment)."""

    def __init__(self) -> None:
        self.environment: dict[str, str] = {}


class SQLTestUtils:
    """``pyspark/testing/sqlutils.py`` ``SQLTestUtils`` over a Krishiv session."""

    spark: SparkSession

    @contextmanager
    def sql_conf(self, pairs: dict[str, Any]) -> Any:
        keys = list(pairs.keys())
        old_values = [self.spark.conf.get(key, None) for key in keys]
        for key, new_value in pairs.items():
            self.spark.conf.set(key, new_value)
        try:
            yield
        finally:
            for key, old_value in zip(keys, old_values, strict=True):
                if old_value is None:
                    self.spark.conf.unset(key)
                else:
                    self.spark.conf.set(key, old_value)

    @contextmanager
    def table(self, *tables: str) -> Any:
        try:
            yield
        finally:
            for t in tables:
                self.spark.sql(f"DROP TABLE IF EXISTS {t}")

    @contextmanager
    def temp_view(self, *views: str) -> Any:
        try:
            yield
        finally:
            for v in views:
                self.spark.catalog.dropTempView(v)

    @contextmanager
    def temp_func(self, *functions: str) -> Any:
        try:
            yield
        finally:
            for f in functions:
                self.spark.sql(f"DROP TEMPORARY FUNCTION IF EXISTS {f}")

    @contextmanager
    def temp_env(self, pairs: dict[str, str | None]) -> Any:
        old_values = {key: os.environ.get(key) for key in pairs}
        for key, new_value in pairs.items():
            if new_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = new_value
        try:
            yield
        finally:
            for key, old_value in old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value


class QuietTest:
    """``pyspark.testing.utils.QuietTest`` silences the JVM's log4j; there is none here."""

    def __init__(self, sc: Any) -> None:
        pass

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


class ReusedSQLTestCase(unittest.TestCase, SQLTestUtils):
    """A class-wide session, as Spark's: 4 UDF pool threads (the pool is process-wide)."""

    _previous_active: Any = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        config = [("krishiv.python.udf.threads", "4")]
        cls.spark = SparkSession(_native.Session(config), config)
        cls._previous_active, _session._active = _session._active, cls.spark
        cls.sc = _SparkContextStub()
        cls.tempdir = tempfile.NamedTemporaryFile(delete=False)
        os.unlink(cls.tempdir.name)
        cls.testData = [Row(key=i, value=str(i)) for i in range(100)]
        cls.df = cls.spark.createDataFrame(cls.testData)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            shutil.rmtree(cls.tempdir.name, ignore_errors=True)
        finally:
            _session._active = cls._previous_active
            super().tearDownClass()

    def quiet(self) -> QuietTest:
        return QuietTest(self.sc)

    def create_data_frame(self, data: Any, schema: Any) -> Any:
        """``spark.createDataFrame(data, schema)``; for the nested types it cannot build from rows, through a
        pyarrow table of the same schema (the test data stays Spark's)."""
        from krishiv.errors import UnsupportedOperationException

        try:
            return self.spark.createDataFrame(data, schema)
        except UnsupportedOperationException:
            import pyarrow as pa

            from krishiv._udf.pandas_types import to_arrow_type

            columns = {
                f.name: pa.array(
                    [_nested(f.dataType, row[i]) for row in data], type=to_arrow_type(f.dataType, timezone="UTC")
                )
                for i, f in enumerate(schema.fields)
            }
            return self.spark.createDataFrame(pa.table(columns))

    def check_error(
        self,
        exception: PySparkException,
        errorClass: str,
        messageParameters: dict[str, str] | None = None,
        query_context_type: Any = None,
        fragment: str | None = None,
        matchPVals: bool = False,
    ) -> None:
        """``PySparkErrorTestUtils.check_error`` (condition and parameters; the query context is not compared)."""
        self.assertIsInstance(exception, PySparkException)
        self.assertEqual(errorClass, exception.getCondition())
        actual = exception.getMessageParameters()
        if matchPVals:
            self.assertEqual(len(messageParameters or {}), len(actual or {}))
            for key, value in (messageParameters or {}).items():
                self.assertIn(key, actual or {})
                self.assertRegex((actual or {})[key], value)
        else:
            self.assertEqual(messageParameters, actual)


def _nested(data_type: Any, value: Any) -> Any:
    """A Python value as pyarrow takes it for the Spark type (rows as dicts, maps as pair lists)."""
    from krishiv.sql.types import ArrayType, MapType, StructType

    if value is None:
        return None
    if isinstance(data_type, StructType):
        names = getattr(value, "_fields", None)
        return {
            f.name: _nested(f.dataType, value[f.name] if names else value[i]) for i, f in enumerate(data_type.fields)
        }
    if isinstance(data_type, ArrayType):
        return [_nested(data_type.elementType, v) for v in value]
    if isinstance(data_type, MapType):
        return [(_nested(data_type.keyType, k), _nested(data_type.valueType, v)) for k, v in value.items()]
    return value
