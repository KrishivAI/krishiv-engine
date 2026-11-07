# SPDX-License-Identifier: Apache-2.0
"""The Python UDF facades (design P2-T059 unit 2c): ``udf``, ``pandas_udf``, ``arrow_udf``,
``UserDefinedFunction`` and ``spark.udf.register``, over the embedded runtime.

Spark facts are cited from the PySpark 4.2.0 sdist (``pyspark/...:line``). The pandas cases need the
``pandas`` extra (``uv run --group dev --extra pandas pytest``) and skip without it; without pandas, Spark's
``udf`` falls back to ``SQL_BATCHED_UDF`` with a ``RuntimeWarning`` (``sql/udf.py:114-124``), which Krishiv runs
Arrow-batched (the documented divergence).
"""

from __future__ import annotations

import datetime
import importlib.util
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from krishiv.errors import PySparkException, PySparkTypeError
from krishiv.sql import SparkSession
from krishiv.sql import functions as F
from krishiv.sql.types import (
    ArrayType,
    DateType,
    DecimalType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from krishiv.sql.udf import UserDefinedFunction

HAVE_PANDAS = importlib.util.find_spec("pandas") is not None
if HAVE_PANDAS:
    # At module level: the UDFs' string annotations (`from __future__ import annotations`) resolve in the
    # module's globals (`typing.get_type_hints`, `sql/pandas/functions.py:934-939`).
    import pandas as pd
needs_pandas = pytest.mark.skipif(not HAVE_PANDAS, reason="the pandas extra is not installed")

pytestmark = [
    # Without pandas Spark's `udf` warns that Arrow cannot be used (`sql/udf.py:114-124`), and Krishiv that it
    # runs the pickled kind Arrow-batched; both are covered by their own tests below.
    pytest.mark.filterwarnings("ignore:Arrow optimization failed:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:Krishiv runs Python UDFs on the Arrow path only:UserWarning"),
]


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    # The UDF pool is process-wide and sized by the session making the first call: 4 threads, as in
    # `test_python_udf_runtime.py`, whose `thread_safe` control needs more than one. The session is the
    # active one while the module runs, as `udf` reads the active session's conf (`sql/udf.py:101-110`).
    from krishiv import _native
    from krishiv.sql import session as _session

    config = [("krishiv.python.udf.threads", "4")]
    created = SparkSession(_native.Session(config), config)
    previous, _session._active = _session._active, created
    try:
        yield created
    finally:
        _session._active = previous


def _values(rows: list[Any]) -> list[Any]:
    return [r[0] for r in rows]


# --- the facade forms and their column names ---


def test_the_decorator_forms_and_their_column_names(spark: SparkSession) -> None:
    # `sql/functions/builtin.py:30489-30507`: `slen(name)` (aliased there), `to_upper(name)`, `add_one(age)`.
    slen = F.udf(lambda s: len(s), IntegerType())

    @F.udf
    def to_upper(s: Any) -> Any:
        if s is not None:
            return s.upper()

    @F.udf(returnType=IntegerType())
    def add_one(x: Any) -> Any:
        if x is not None:
            return x + 1

    @F.udf("int")
    def times_two(x: Any) -> Any:
        return None if x is None else x * 2

    df = spark.sql("SELECT 1 AS id, 'John Doe' AS name, 21 AS age")
    out = df.select(slen("name").alias("slen(name)"), to_upper("name"), add_one("age"), times_two(df.age))
    assert out.columns == ["slen(name)", "to_upper(name)", "add_one(age)", "times_two(age)"]
    assert [tuple(r) for r in out.collect()] == [(8, "JOHN DOE", 22, 42)]


def test_a_lambda_is_named_lambda(spark: SparkSession) -> None:
    # The name is `func.__name__` (`sql/udf.py:204-206`), so `<lambda>` for a lambda.
    f = F.udf(lambda x: x + 1, LongType())
    assert f.__name__ == "<lambda>"
    df = spark.range(3).select(f("id"))
    assert df.columns == ["<lambda>(id)"]
    assert sorted(_values(df.collect())) == [1, 2, 3]


def test_return_type_defaults_to_string_and_takes_a_ddl_string_or_a_data_type(spark: SparkSession) -> None:
    # `udf`'s `returnType` defaults to `StringType()` (`sql/functions/builtin.py:30453-30458`); a DDL
    # string is parsed when `returnType` is first read (`sql/udf.py:391-402`).
    default = F.udf(lambda x: x)
    assert default.returnType == StringType()
    as_string = F.udf(lambda x: x, "array<int>")
    assert as_string.returnType == ArrayType(IntegerType())
    as_type = F.udf(lambda x: x, DoubleType())
    assert as_type.returnType == DoubleType()
    # A default-string UDF over a number returns its text (`LocalDataToArrowConversion`'s string
    # coercion, `sql/conversion.py:744-760`).
    assert sorted(_values(spark.range(2).select(default("id")).collect())) == ["0", "1"]


def test_udf_attributes_are_sparks(spark: SparkSession) -> None:
    # `_wrapped` (`sql/udf.py:515-548`) carries func, returnType, evalType, deterministic,
    # asNondeterministic and _unwrapped.
    def plus(x: Any) -> Any:
        return x + 1

    f = F.udf(plus, "bigint")
    assert f.func is plus
    assert f.deterministic is True
    assert f.evalType == (101 if HAVE_PANDAS else 100)
    assert isinstance(f._unwrapped, UserDefinedFunction)
    explicit = F.udf(plus, "bigint", useArrow=True)
    assert explicit.evalType == (101 if HAVE_PANDAS else 100)


def test_use_arrow_false_keeps_sparks_eval_type_and_warns_of_the_divergence(spark: SparkSession) -> None:
    # `useArrow=False` is `SQL_BATCHED_UDF` (`sql/udf.py:137-143`); Krishiv runs it Arrow-batched.
    with pytest.warns(UserWarning, match="Arrow path only"):
        f = F.udf(lambda x: x * 3, "bigint", useArrow=False)
    assert f.evalType == 100
    assert sorted(_values(spark.range(3).select(f("id")).collect())) == [0, 3, 6]


def test_arrow_disabled_by_conf_gives_the_batched_kind(spark: SparkSession) -> None:
    # `spark.sql.execution.pythonUDF.arrow.enabled=false` with `useArrow=None` (`sql/udf.py:101-110`).
    spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", "false")
    try:
        with pytest.warns(UserWarning, match="Arrow path only"):
            f = F.udf(lambda x: -x, "bigint")
        assert f.evalType == 100
        assert sorted(_values(spark.range(2).select(f("id")).collect())) == [-1, 0]
    finally:
        spark.conf.unset("spark.sql.execution.pythonUDF.arrow.enabled")


def test_as_nondeterministic_clears_deterministic(spark: SparkSession) -> None:
    # `asNondeterministic` (`sql/udf.py:550-560`, exposed by `_wrapped`, `:544-546`).
    f = F.udf(lambda x: x, "bigint")
    g = f.asNondeterministic()
    assert g.deterministic is False
    assert f.deterministic is True
    assert sorted(_values(spark.range(2).select(g("id")).collect())) == [0, 1]


def test_keyword_arguments_are_not_supported_yet(spark: SparkSession) -> None:
    f = F.udf(lambda a: a, "bigint")
    with pytest.raises(PySparkException, match="NOT_IMPLEMENTED"):
        f(a="id")


# --- spark.udf.register ---


def test_register_a_function_then_sql_and_the_dataframe(spark: SparkSession) -> None:
    # `sql/udf.py:621-631`: string by default; `stringLengthString(test)` and `stringLengthString(text)`.
    strlen = spark.udf.register("stringLengthString", lambda x: len(x))
    df = spark.sql("SELECT stringLengthString('test')")
    assert df.columns == ["stringLengthString(test)"]
    assert _values(df.collect()) == ["4"]
    df = spark.sql("SELECT 'foo' AS text").select(strlen("text"))
    assert df.columns == ["stringLengthString(text)"]
    assert _values(df.collect()) == ["3"]
    spark.udf.register("stringLengthInt", lambda x: len(x), IntegerType())
    assert _values(spark.sql("SELECT stringLengthInt('test')").collect()) == [4]


def test_register_a_udf_keeps_its_type_and_determinism(spark: SparkSession) -> None:
    # `sql/udf.py:644-657`.
    slen = F.udf(lambda s: len(s), IntegerType())
    registered = spark.udf.register("slen", slen)
    assert _values(spark.sql("SELECT slen('test')").collect()) == [4]
    assert registered.returnType == IntegerType()
    nondet = spark.udf.register("nondet_id", F.udf(lambda x: x, "bigint").asNondeterministic())
    assert nondet.deterministic is False
    assert sorted(_values(spark.sql("SELECT nondet_id(id) FROM range(2)").collect())) == [0, 1]


def test_register_a_udf_with_a_return_type_is_refused(spark: SparkSession) -> None:
    # `CANNOT_SPECIFY_RETURN_TYPE_FOR_UDF` (`sql/udf.py:681-686`).
    with pytest.raises(PySparkTypeError) as caught:
        spark.udf.register("x", F.udf(lambda s: s), "int")
    assert caught.value.getCondition() == "CANNOT_SPECIFY_RETURN_TYPE_FOR_UDF"


def test_an_unregistered_udf_is_not_callable_from_sql(spark: SparkSession) -> None:
    # Only `spark.udf.register` makes a name SQL-callable (`sql/udf.py:726`).
    def not_registered_plus(x: Any) -> Any:
        return x + 1

    f = F.udf(not_registered_plus, "bigint")
    assert _values(spark.range(1).select(f("id")).collect()) == [1]
    for query in ("SELECT not_registered_plus(id) FROM range(1)", "SELECT `<lambda>`(id) FROM range(1)"):
        with pytest.raises(PySparkException) as caught:
            spark.sql(query).collect()
        assert caught.value.getCondition() == "UNRESOLVED_ROUTINE", query


def test_a_udf_runs_in_another_session(spark: SparkSession) -> None:
    f = F.udf(lambda x: x * 10, "bigint")
    other = spark.newSession()
    assert sorted(_values(other.range(2).select(f("id")).collect())) == [0, 10]
    assert sorted(_values(spark.range(2).select(f("id")).collect())) == [0, 10]


# --- each eval type over each type, nulls in and out ---

_TYPES_QUERY = """
SELECT * FROM (
  SELECT 1 AS i, 'x' AS s, 1.5D AS d, CAST(1.25 AS DECIMAL(10,2)) AS dec, DATE'2024-01-02' AS dt,
         TIMESTAMP'2024-01-02 03:04:05' AS ts, array(1, 2) AS arr, named_struct('a', 1, 'b', 'y') AS st, 0 AS k
  UNION ALL
  SELECT NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1
) ORDER BY k
"""

_COLUMNS = [
    ("i", IntegerType()),
    ("s", StringType()),
    ("d", DoubleType()),
    ("dec", DecimalType(10, 2)),
    ("dt", DateType()),
    ("ts", TimestampType()),
    ("arr", ArrayType(IntegerType())),
    ("st", StructType([StructField("a", IntegerType()), StructField("b", StringType())])),
]

_EXPECTED = [
    1,
    "x",
    1.5,
    Decimal("1.25"),
    datetime.date(2024, 1, 2),
    datetime.datetime(2024, 1, 2, 3, 4, 5),
    [1, 2],
    (1, "y"),
]


def _check_identity(spark: SparkSession, make: Any, null_struct: Any = None) -> None:
    df = spark.sql(_TYPES_QUERY)
    for (name, data_type), expected in zip(_COLUMNS, _EXPECTED, strict=True):
        f = make(data_type)
        rows = df.select(f(name)).collect()
        got = _values(rows)
        first = tuple(got[0]) if isinstance(got[0], tuple) else got[0]
        assert first == expected, (name, got)
        if name == "st" and null_struct is not None:
            assert tuple(got[1]) == null_struct, (name, got)
        else:
            assert got[1] is None, (name, got)


def test_arrow_batched_round_trips_every_type_and_null(spark: SparkSession) -> None:
    _check_identity(spark, lambda t: F.udf(lambda v: v, t))


def test_arrow_udf_round_trips_every_type_and_null(spark: SparkSession) -> None:
    _check_identity(spark, lambda t: F.arrow_udf(lambda a: a, t))


@needs_pandas
def test_pandas_udf_round_trips_every_type_and_null(spark: SparkSession) -> None:
    # A struct reaches a pandas UDF as a DataFrame (`worker.py:2467-2475`, `df_for_struct`), where a NULL
    # struct is a row of NULL fields: it comes back as a struct of NULLs, not a NULL struct.
    _check_identity(spark, lambda t: F.pandas_udf(lambda s: s, t), null_struct=(None, None))


def test_arrow_batched_converts_results_to_the_declared_type(spark: SparkSession) -> None:
    struct = F.udf(lambda i: {"a": i, "b": str(i)}, "struct<a:bigint,b:string>")
    rows = spark.range(2).select(struct("id")).collect()
    assert sorted(tuple(r[0]) for r in rows) == [(0, "0"), (1, "1")]
    as_dict = F.udf(lambda i: {str(i): i}, "map<string,bigint>")
    assert sorted(_values(spark.range(2).select(as_dict("id")).collect()), key=str) == [{"0": 0}, {"1": 1}]


def test_a_zero_argument_udf_gives_one_value_per_row(spark: SparkSession) -> None:
    seven = F.udf(lambda: 7, "int")
    assert _values(spark.range(3).select(seven()).collect()) == [7, 7, 7]


# --- the arrow and pandas kinds ---


def test_arrow_udf_works_on_whole_arrays(spark: SparkSession) -> None:
    @F.arrow_udf("bigint")
    def plus_one(a: pa.Array) -> pa.Array:
        return pc.call_function("add", [a, 1])

    assert plus_one.evalType == 250
    df = spark.range(3).select(plus_one("id"))
    assert df.columns == ["plus_one(id)"]
    assert sorted(_values(df.collect())) == [1, 2, 3]


def test_arrow_udf_gets_sparks_string_type_not_a_view(spark: SparkSession) -> None:
    seen: list[Any] = []

    def kind(a: pa.Array) -> pa.Array:
        seen.append(a.type)
        return pc.call_function("utf8_upper", [a])

    f = F.arrow_udf(kind, "string")
    assert _values(spark.sql("SELECT 'ab' AS s").select(f("s")).collect()) == ["AB"]
    assert seen == [pa.string()]


def test_arrow_udf_returning_no_pyarrow_array_is_refused(spark: SparkSession) -> None:
    # Krishiv imports pyarrow's own arrays only (design "Trusted producer").
    f = F.arrow_udf(lambda a: a.to_pylist(), "bigint")
    with pytest.raises(PySparkException, match="UDF_RETURN_TYPE") as caught:
        spark.range(3).select(f("id")).collect()
    assert caught.value.getCondition() == "PYTHON_EXCEPTION"


def test_a_udf_inferred_from_arrow_type_hints_is_an_arrow_udf(spark: SparkSession) -> None:
    # `udf` with `pa.Array -> pa.Array` hints is `ArrowUDFType.SCALAR` (`sql/functions/builtin.py:30573-30580`,
    # `sql/pandas/typehints.py:380-403`). Inferring tries pandas first, so it needs pandas.
    @F.udf(returnType=IntegerType())
    def pa_calc(a: pa.Array, b: pa.Array) -> pa.Array:
        return pc.call_function("add", [a, pc.call_function("multiply", [b, 10])])

    if HAVE_PANDAS:
        assert pa_calc.evalType == F.ArrowUDFType.SCALAR
    df = spark.sql("SELECT 1 AS a, 2 AS b").select(pa_calc("a", "b"))
    assert _values(df.collect()) == [21]


def test_pandas_udf_needs_a_return_type() -> None:
    # `CANNOT_BE_NONE` (`sql/pandas/functions.py:817-821`), checked after pandas is found.
    if not HAVE_PANDAS:
        with pytest.raises(PySparkException) as caught:
            F.pandas_udf(lambda s: s, "int")
        assert caught.value.getCondition() == "PACKAGE_NOT_INSTALLED"
        return
    with pytest.raises(PySparkTypeError) as caught:
        F.pandas_udf(lambda s: s)
    assert caught.value.getCondition() == "CANNOT_BE_NONE"


@needs_pandas
def test_pandas_udf_over_a_series(spark: SparkSession) -> None:
    @F.pandas_udf("bigint")
    def add_one(s: pd.Series) -> pd.Series:
        assert isinstance(s, pd.Series)
        return s + 1

    assert add_one.evalType == F.PandasUDFType.SCALAR
    spark.udf.register("add_one_pd", add_one)
    # `sql/udf.py:661-667`: `add_one(id)` = 1, 2, 3.
    rows = spark.sql("SELECT add_one_pd(id) FROM range(3)").collect()
    assert sorted(_values(rows)) == [1, 2, 3]
    df = spark.range(3).select(add_one("id"))
    assert df.columns == ["add_one(id)"]
    assert sorted(_values(df.collect())) == [1, 2, 3]


@needs_pandas
def test_pandas_udf_of_the_wrong_length_fails(spark: SparkSession) -> None:
    # `SCHEMA_MISMATCH_FOR_PANDAS_UDF` (`worker.py:394-403`).
    f = F.pandas_udf(lambda s: s.iloc[:1], "bigint")
    with pytest.raises(PySparkException, match="SCHEMA_MISMATCH_FOR_PANDAS_UDF"):
        spark.range(3).select(f("id")).collect()


def test_an_arrow_udf_of_the_wrong_length_fails(spark: SparkSession) -> None:
    f = F.arrow_udf(lambda a: pa.array([1]), "bigint")
    with pytest.raises(PySparkException, match="SCHEMA_MISMATCH_FOR_PANDAS_UDF"):
        spark.range(3).select(f("id")).collect()


# --- stream export ---


def test_a_stream_read_with_the_gil_held_finishes() -> None:
    # Design §2.5 "Stream export": the C stream's get_next, called from a thread holding the GIL (a ctypes
    # PYFUNCTYPE call keeps it), must not block the UDF pool. Run in a child process: without the detach the
    # holder never lets go of the GIL, so nothing in this process could time it out.
    code = textwrap.dedent(
        """
        import ctypes
        import pyarrow.compute as pc
        from krishiv.sql import SparkSession, functions as F

        class ArrowArray(ctypes.Structure):
            pass

        ArrowArray._fields_ = [
            ("length", ctypes.c_int64), ("null_count", ctypes.c_int64), ("offset", ctypes.c_int64),
            ("n_buffers", ctypes.c_int64), ("n_children", ctypes.c_int64), ("buffers", ctypes.c_void_p),
            ("children", ctypes.c_void_p), ("dictionary", ctypes.c_void_p),
            ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowArray))), ("private_data", ctypes.c_void_p),
        ]

        class Stream(ctypes.Structure):
            pass

        Stream._fields_ = [
            ("get_schema", ctypes.c_void_p),
            ("get_next", ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.POINTER(Stream), ctypes.POINTER(ArrowArray))),
            ("get_last_error", ctypes.c_void_p),
            ("release", ctypes.PYFUNCTYPE(None, ctypes.POINTER(Stream))),
            ("private_data", ctypes.c_void_p),
        ]

        spark = SparkSession.builder.getOrCreate()
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "50")
        plus = F.arrow_udf(lambda a: pc.call_function("add", [a, 1]), "bigint")
        capsule = spark.range(0, 1000, 1, 4).select(plus("id")).__arrow_c_stream__()
        get = ctypes.pythonapi.PyCapsule_GetPointer
        get.restype = ctypes.c_void_p
        get.argtypes = [ctypes.py_object, ctypes.c_char_p]
        stream = ctypes.cast(get(capsule, b"arrow_array_stream"), ctypes.POINTER(Stream))
        rows = 0
        while True:
            out = ArrowArray()
            assert stream.contents.get_next(stream, ctypes.byref(out)) == 0
            if not out.release:
                break
            rows += out.length
            out.release(ctypes.byref(out))
        assert rows == 1000, rows
        print("ok")
        """
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "ok"
