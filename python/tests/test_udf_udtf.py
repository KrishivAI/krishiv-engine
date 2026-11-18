# SPDX-License-Identifier: Apache-2.0
"""Python table functions (``krishiv._udf.udtf``: ``@udtf`` 300, ``@udtf(useArrow=True)`` 301, ``@arrow_udtf``
302, and ``analyze``; P3-T014 PR-26c-3 part B), run in this process over a scripted frame source.

The server runs the same code in a pool worker (``krishiv-udf-pool``'s ``tests/udtf.rs`` drives real workers with
``pyspark-client``); that interpreter is not measured, so this file runs each path here: ``eval`` per row or
batch, ``terminate`` at each ``PARTITION BY`` change and at the end, ``cleanup``, ``SkipRestOfInputTableException``,
``analyze``'s result passed to ``__init__``, and the ``UDTF_*`` errors of pyspark 4.2.0's ``read_udtf``.
``analyze`` needs pyspark's ``AnalyzeResult`` classes, which the facade does not have yet: stand-ins with the same
fields take their place.
"""

from __future__ import annotations

import dataclasses
import pickle
import types
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

pa = pytest.importorskip("pyarrow")

import krishiv.errors as kerrors  # noqa: E402
from krishiv._udf import _host, udtf  # noqa: E402
from krishiv._udf.udtf import SQL_ARROW_TABLE_UDF, SQL_ARROW_UDTF, SQL_TABLE_UDF, UdtfCall  # noqa: E402
from krishiv.sql.types import DataType, IntegerType, LongType, StructField, StructType  # noqa: E402

AB = StructType([StructField("a", LongType()), StructField("b", LongType())])


class Source:
    """A scripted frame source: the frames in order."""

    def __init__(self, frames: list[tuple[Any, ...]]) -> None:
        self.frames = list(frames)

    def next(self) -> tuple[Any, ...]:
        return self.frames.pop(0)


@pytest.fixture(autouse=True)
def conf(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    values = {"spark.sql.session.timeZone": "UTC"}
    monkeypatch.setattr(_host, "runner_conf", lambda: values)
    return values


def call(
    n: int = 1,
    return_type: StructType = AB,
    named: dict[int, str] | None = None,
    partition: list[int] | None = None,
    tables: list[int] | None = None,
    analyzed: Any = None,
) -> UdtfCall:
    names = named or {}
    return UdtfCall.of(
        {
            "name": "my_udtf",
            "return_type": return_type.json(),
            "args": [[i, names.get(i)] for i in range(n)],
            "partition_child_indexes": partition or [],
            "table_arg_offsets": tables or [],
            "analyze_result": None if analyzed is None else pickle.dumps(analyzed).hex(),
        }
    )


def x_frames() -> list[tuple[Any, ...]]:
    return [
        ("batch", 0, pa.RecordBatch.from_pydict({"_0": [1, 2]})),
        ("batch", 0, pa.RecordBatch.from_pydict({"_0": [3]})),
        ("end",),
    ]


def run(handler: Any, eval_type: int, c: UdtfCall, frames: list[tuple[Any, ...]]) -> list[list[Any]]:
    runner = udtf.make_udtf_runner(handler, eval_type, c)
    out: list[list[Any]] = []
    for batch in runner(Source(frames), 0, 0):
        out.extend(list(r.values()) for r in batch.to_pylist())
    return out


class Counting:
    cleaned: ClassVar[list[int]] = []

    def __init__(self) -> None:
        self.n = 0

    def eval(self, x: int) -> Iterator[tuple[int, int]]:
        self.n += 1
        yield x, x * 10

    def terminate(self) -> Iterator[tuple[int, int]]:
        yield -1, self.n

    def cleanup(self) -> None:
        Counting.cleaned.append(self.n)


@pytest.mark.parametrize("eval_type", [SQL_TABLE_UDF, SQL_ARROW_TABLE_UDF])
def test_eval_runs_per_row_then_terminate_and_cleanup(eval_type: int) -> None:
    Counting.cleaned = []
    assert run(Counting, eval_type, call(), x_frames()) == [[1, 10], [2, 20], [3, 30], [-1, 3]]
    assert Counting.cleaned == [3]


def test_an_arrow_udtf_gets_each_batch_and_its_tables_are_coerced() -> None:
    class Batches:
        def eval(self, x: Any) -> Iterator[Any]:
            # int32 results are cast to the declared bigint.
            yield pa.table({"a": x.cast(pa.int32()), "b": x})

        def terminate(self) -> Iterator[Any]:
            yield pa.record_batch({"a": [-1], "b": [-1]})

    assert run(Batches, SQL_ARROW_UDTF, call(), x_frames()) == [[1, 1], [2, 2], [3, 3], [-1, -1]]

    class Nothing:
        def eval(self, x: Any) -> None:
            return None

    # An empty result is one empty batch per eval.
    assert run(Nothing, SQL_ARROW_UDTF, call(), x_frames()) == []


def test_named_arguments_and_a_zero_argument_call() -> None:
    class Named:
        def eval(self, a: int, *, b: int) -> Iterator[tuple[int, int]]:
            yield a, b

    frames = [("batch", 0, pa.RecordBatch.from_pydict({"_0": [1], "_1": [2]})), ("end",)]
    assert run(Named, SQL_TABLE_UDF, call(2, named={1: "b"}), frames) == [[1, 2]]

    class OnlyNamed:
        def eval(self, b: int) -> Iterator[tuple[int, int]]:
            yield b, b

    frames = [("batch", 0, pa.RecordBatch.from_pydict({"_0": [4]})), ("end",)]
    assert run(OnlyNamed, SQL_TABLE_UDF, call(1, named={0: "b"}), frames) == [[4, 4]]

    class NoArgs:
        def eval(self) -> Iterator[tuple[int, int]]:
            yield 7, 7

    one_row = [("batch", 0, pa.RecordBatch.from_pydict({"_0": [0]}).select([])), ("end",)]
    assert run(NoArgs, SQL_TABLE_UDF, call(0), one_row) == [[7, 7]]


def test_a_none_row_and_a_single_column_value() -> None:
    one = StructType([StructField("a", LongType())])

    class Rows:
        def eval(self, x: int) -> Iterator[Any]:
            yield None

    assert run(Rows, SQL_TABLE_UDF, call(), x_frames()) == [[None, None]] * 3

    class Bare:
        def eval(self, x: int) -> Iterator[Any]:
            yield x

    assert run(Bare, SQL_ARROW_TABLE_UDF, call(return_type=one), x_frames()) == [[1], [2], [3]]


def table_frames() -> list[tuple[Any, ...]]:
    def rows(v: list[int], k: list[int]) -> Any:
        struct = pa.StructArray.from_arrays([pa.array(v), pa.array(k)], names=["v", "partition_by_0"])
        return pa.RecordBatch.from_arrays([struct], names=["_0"])

    return [("batch", 0, rows([1, 2, 3], [7, 7, 8])), ("batch", 0, rows([4, 5], [8, 9])), ("end",)]


def test_partition_by_terminates_each_partition_and_hides_its_values() -> None:
    class Sum:
        def __init__(self) -> None:
            self.s = 0

        def eval(self, row: Any) -> Iterator[Any]:
            assert list(row.asDict()) == ["v"]
            self.s += row["v"]
            return iter(())

        def terminate(self) -> Iterator[tuple[int, int]]:
            yield self.s, 0

    c = call(partition=[1], tables=[0])
    assert run(Sum, SQL_TABLE_UDF, c, table_frames()) == [[3, 0], [7, 0], [5, 0]]

    class ArrowSum:
        def __init__(self) -> None:
            self.s = 0

        def eval(self, batch: Any) -> Iterator[Any]:
            assert batch.schema.names == ["v"]
            self.s += sum(batch.column(0).to_pylist())
            return iter(())

        def terminate(self) -> Iterator[Any]:
            yield pa.table({"a": [self.s], "b": [0]})

    assert run(ArrowSum, SQL_ARROW_UDTF, c, table_frames()) == [[3, 0], [7, 0], [5, 0]]


def test_partition_columns_only_leave_an_empty_batch() -> None:
    class Count:
        def __init__(self) -> None:
            self.n = 0

        def eval(self, batch: Any) -> Iterator[Any]:
            assert batch.num_columns == 0
            self.n += batch.num_rows
            return iter(())

        def terminate(self) -> Iterator[Any]:
            yield pa.table({"a": [self.n], "b": [0]})

    struct = pa.StructArray.from_arrays([pa.array([1, 1, 2])], names=["partition_by_0"])
    frames = [("batch", 0, pa.RecordBatch.from_arrays([struct], names=["_0"])), ("end",)]
    assert run(Count, SQL_ARROW_UDTF, call(partition=[0], tables=[0]), frames) == [[2, 0], [1, 0]]


def test_skip_rest_of_input_table_skips_to_the_partition_end(monkeypatch: pytest.MonkeyPatch) -> None:
    skip = udtf._skip_rest_exception()

    class FirstOnly:
        def eval(self, row: Any) -> Iterator[tuple[int, int]]:
            yield row["v"], 0
            raise skip

    c = call(partition=[1], tables=[0])
    assert run(FirstOnly, SQL_TABLE_UDF, c, table_frames()) == [[1, 0], [3, 0], [5, 0]]

    class ArrowFirst:
        def eval(self, batch: Any) -> Iterator[Any]:
            yield pa.table({"a": batch.column(0).slice(0, 1), "b": [0]})
            raise skip

    assert run(ArrowFirst, SQL_ARROW_UDTF, c, table_frames()) == [[1, 0], [3, 0], [5, 0]]

    class Stop:
        def eval(self, x: int) -> Iterator[tuple[int, int]]:
            raise skip

        def terminate(self) -> Iterator[tuple[int, int]]:
            yield 0, 0

    assert run(Stop, SQL_TABLE_UDF, call(), x_frames()) == [[0, 0]]

    class ArrowStop:
        def eval(self, x: Any) -> Iterator[Any]:
            raise skip

        def terminate(self) -> Iterator[Any]:
            yield pa.table({"a": [0], "b": [0]})

    assert run(ArrowStop, SQL_ARROW_UDTF, call(), x_frames()) == [[0, 0]]


@dataclasses.dataclass(frozen=True)
class Result:
    """A picklable stand-in for an ``AnalyzeResult``."""

    n: int


def test_the_analyze_result_reaches_init() -> None:
    class WithAnalyze:
        def __init__(self, analyze_result: Result) -> None:
            self.n = analyze_result.n

        def eval(self, x: int) -> Iterator[tuple[int, int]]:
            yield x, self.n

    assert run(WithAnalyze, SQL_TABLE_UDF, call(analyzed=Result(5)), x_frames()) == [[1, 5], [2, 5], [3, 5]]

    class NoArgInit:
        def eval(self, x: int) -> Iterator[tuple[int, int]]:
            yield x, 0

    assert run(NoArgInit, SQL_TABLE_UDF, call(analyzed=Result(5)), x_frames())[0] == [1, 0]


def error(handler: Any, eval_type: int, c: UdtfCall | None = None, frames: Any = None) -> Exception:
    with pytest.raises(Exception) as err:  # noqa: PT011  # each caller checks the condition
        run(handler, eval_type, c or call(), frames or x_frames())
    return err.value


def condition(e: Exception) -> str | None:
    get = getattr(e, "getCondition", None)
    return get() if get else None


def test_read_udtf_s_errors() -> None:
    class Raises:
        def eval(self, x: int) -> Any:
            raise ValueError("boom")

    for eval_type in (SQL_TABLE_UDF, SQL_ARROW_TABLE_UDF, SQL_ARROW_UDTF):
        e = error(Raises, eval_type)
        assert condition(e) == "UDTF_EXEC_ERROR"
        assert e.getMessageParameters() == {"method_name": "eval", "error": "boom"}  # ty: ignore[unresolved-attribute]

    class NotIterable:
        def eval(self, x: int) -> Any:
            return 1

    for eval_type in (SQL_TABLE_UDF, SQL_ARROW_UDTF):
        assert condition(error(NotIterable, eval_type)) == "UDTF_RETURN_NOT_ITERABLE"

    class Short:
        def eval(self, x: int) -> Iterator[Any]:
            yield (x,)

    assert condition(error(Short, SQL_TABLE_UDF)) == "UDTF_RETURN_SCHEMA_MISMATCH"
    assert condition(error(Short, SQL_ARROW_TABLE_UDF)) == "UDTF_RETURN_SCHEMA_MISMATCH"

    class Scalar:
        def eval(self, x: int) -> Iterator[Any]:
            yield 1

    assert condition(error(Scalar, SQL_TABLE_UDF)) == "UDTF_INVALID_OUTPUT_ROW_TYPE"
    assert condition(error(Scalar, SQL_ARROW_UDTF)) == "UDTF_ARROW_TYPE_CONVERSION_ERROR"

    class Udt:
        def eval(self, x: int) -> Iterator[Any]:
            class Point:
                __UDT__ = object()

            yield Point()

    assert condition(error(Udt, SQL_TABLE_UDF)) == "UDTF_INVALID_OUTPUT_ROW_TYPE"

    class Strings:
        def eval(self, x: int) -> Iterator[Any]:
            yield "a", "b"

    assert condition(error(Strings, SQL_ARROW_TABLE_UDF)) == "UDTF_ARROW_DATA_CONVERSION_ERROR"

    class Wide:
        def eval(self, x: Any) -> Iterator[Any]:
            yield pa.table({"a": [1], "b": [2], "c": [3]})

    assert condition(error(Wide, SQL_ARROW_UDTF)) == "UDTF_RETURN_SCHEMA_MISMATCH"

    class TwoArgs:
        def eval(self, x: int, y: int) -> Iterator[Any]:
            yield x, y

    assert condition(error(TwoArgs, SQL_TABLE_UDF)) == "UDTF_EVAL_METHOD_ARGUMENTS_DO_NOT_MATCH_SIGNATURE"

    class InitArg:
        def __init__(self, a: Any) -> None:
            pass

        def eval(self, x: int) -> Iterator[Any]:
            yield x, x

    assert condition(error(InitArg, SQL_TABLE_UDF)) == "UDTF_CONSTRUCTOR_INVALID_NO_ANALYZE_METHOD"

    class InitArgs:
        def __init__(self, a: Any, b: Any) -> None:
            pass

        def eval(self, x: int) -> Iterator[Any]:
            yield x, x

    analyzed = call(analyzed={"r": 1})
    assert condition(error(InitArgs, SQL_TABLE_UDF, analyzed)) == "UDTF_CONSTRUCTOR_INVALID_IMPLEMENTS_ANALYZE_METHOD"

    class BadInit:
        def __init__(self) -> None:
            raise RuntimeError("no")

        def eval(self, x: int) -> Iterator[Any]:
            yield x, x

    e = error(BadInit, SQL_TABLE_UDF)
    assert condition(e) == "UDTF_EXEC_ERROR"
    assert e.getMessageParameters()["method_name"] == "__init__"  # ty: ignore[unresolved-attribute]

    class NoEval:
        pass

    assert "has not implemented the 'eval' method" in str(error(NoEval, SQL_TABLE_UDF))
    assert "Invalid UDTF handler type" in str(error(NoEval(), SQL_TABLE_UDF))

    class NotNull:
        def eval(self, x: int) -> Iterator[Any]:
            yield None, x

    strict = StructType([StructField("a", LongType(), False), StructField("b", LongType())])
    e = error(NotNull, SQL_TABLE_UDF, call(return_type=strict))
    assert condition(e) == "UDTF_EXEC_ERROR"
    assert "non-nullable" in e.getMessageParameters()["error"]  # ty: ignore[unresolved-attribute]


def test_a_partition_by_arrow_udtf_needs_a_table_argument() -> None:
    class AnyUdtf:
        def eval(self, x: Any) -> Iterator[Any]:
            return iter(())

    e = error(AnyUdtf, SQL_ARROW_UDTF, call(partition=[0]))
    assert condition(e) == "INVALID_ARROW_UDTF_TABLE_ARGUMENT"


def test_a_return_type_must_be_a_struct_and_legacy_conversion_is_refused(conf: dict[str, str]) -> None:
    with pytest.raises(kerrors.PySparkRuntimeError, match="must be a struct type"):
        udtf.make_udtf_runner(Counting, SQL_TABLE_UDF, UdtfCall.of({"name": "f", "return_type": '"long"'}))
    conf[udtf.LEGACY_PANDAS_CONVERSION] = "true"
    assert condition(error(Counting, SQL_ARROW_TABLE_UDF)) == "NOT_IMPLEMENTED"


def test_the_null_checker_walks_nested_types() -> None:
    from krishiv.sql.types import ArrayType, MapType

    nested = StructType(
        [
            StructField("arr", ArrayType(LongType(), containsNull=False)),
            StructField("map", MapType(LongType(), LongType(), valueContainsNull=False)),
            StructField("loose", MapType(LongType(), LongType())),
            StructField("s", StructType([StructField("f", LongType(), False)])),
        ]
    )
    check = udtf._build_null_checker(nested)
    assert check is not None
    check(([1], {1: 2}, {1: None}, (1,)))
    for bad in (([None], {}, {}, (1,)), ([], {1: None}, {}, (1,)), ([], {}, {None: 1}, (1,)), ([], {}, {}, (None,))):
        with pytest.raises(kerrors.PySparkRuntimeError):
            check(bad)
    assert udtf._build_null_checker(AB) is None


# --- analyze (stand-ins for pyspark's classes) ---
@dataclasses.dataclass(frozen=True)
class AnalyzeArgument:
    dataType: Any
    value: Any
    isTable: bool
    isConstantExpression: bool


@dataclasses.dataclass(frozen=True)
class PartitioningColumn:
    name: str


@dataclasses.dataclass(frozen=True)
class OrderingColumn:
    name: str
    ascending: bool = True
    overrideNullsFirst: bool | None = None


@dataclasses.dataclass(frozen=True)
class SelectedColumn:
    name: str
    alias: str = ""


@dataclasses.dataclass(frozen=True)
class AnalyzeResult:
    schema: Any
    withSinglePartition: bool = False
    partitionBy: Any = ()
    orderBy: Any = ()
    select: Any = ()


@pytest.fixture
def analyze_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        udtf,
        "_analyze_api",
        lambda: (AnalyzeArgument, AnalyzeResult, OrderingColumn, PartitioningColumn, SelectedColumn),
    )
    # The facade's types have no `needConversion` yet; an integer needs none in pyspark either.
    monkeypatch.setattr(DataType, "needConversion", lambda self: False, raising=False)


TABLE = {"type": AB.json(), "is_table": True, "name": None, "column": None}
N = {"type": '"integer"', "column": 0, "is_table": False, "name": "n"}
#: The constants of a call with ``N``: its value, 3, in column 0.
VALUES = pa.record_batch([pa.array([3], pa.int32())], names=["_1"])


class Analyzed:
    @staticmethod
    def analyze(t: AnalyzeArgument, n: AnalyzeArgument) -> AnalyzeResult:
        assert t.isTable
        assert not t.isConstantExpression
        assert t.value is None
        assert n.isConstantExpression
        assert n.value == 3
        assert n.dataType == IntegerType()
        return AnalyzeResult(
            schema=StructType([StructField("c", LongType())]),
            withSinglePartition=False,
            partitionBy=[PartitioningColumn("a")],
            orderBy=[OrderingColumn("b", ascending=False, overrideNullsFirst=True)],
            select=[SelectedColumn("a", "x")],
        )

    def eval(self, row: Any, n: int) -> Iterator[Any]:
        yield (n,)


def test_analyze_gives_spark_s_fields(analyze_api: None) -> None:
    got = udtf.analyze_udtf(Analyzed, "f", [TABLE, N], VALUES)
    assert got["schema"] == StructType([StructField("c", LongType())]).json()
    assert pickle.loads(bytes.fromhex(got["pickled"])).partitionBy == [PartitioningColumn("a")]
    assert got["with_single_partition"] is False
    assert got["partition_by"] == ["a"]
    assert got["order_by"] == [{"name": "b", "ascending": False, "nulls_first": True}]
    assert got["select"] == [{"name": "a", "alias": "x"}]


def analyze_error(handler: Any, args: list[dict[str, Any]]) -> str:
    with pytest.raises(Exception) as err:  # noqa: PT011  # each caller checks the message
        udtf.analyze_udtf(handler, "f", args, VALUES)
    return str(err.value)


def test_analyze_s_refusals(analyze_api: None) -> None:
    def returning(**result: Any) -> Any:
        class F:
            @staticmethod
            def analyze(*args: Any, **kwargs: Any) -> Any:
                if "value" in result:
                    return result["value"]
                return AnalyzeResult(**result)

        return F

    schema = StructType([StructField("c", LongType())])
    assert "expects a result of type" in analyze_error(returning(value=1), [N])
    assert "with a 'schema' field comprising a StructType" in analyze_error(returning(schema=LongType()), [N])
    assert "'withSinglePartition' field set to 'true'" in analyze_error(
        returning(schema=schema, withSinglePartition=True), [N]
    )
    assert "'partitionBy' list set to non-empty" in analyze_error(
        returning(schema=schema, partitionBy=[PartitioningColumn("a")]), [N]
    )
    assert "'partitionBy' field set to a value besides" in analyze_error(
        returning(schema=schema, partitionBy=["a"]), [TABLE]
    )
    assert "'orderBy' field set to a value besides" in analyze_error(returning(schema=schema, orderBy=["a"]), [TABLE])
    assert "'select' field set to a value besides" in analyze_error(returning(schema=schema, select=["a"]), [TABLE])

    class OneArg:
        @staticmethod
        def analyze(a: Any) -> Any:
            return None

    assert "did not match the expected signature of the static 'analyze' method" in analyze_error(OneArg, [N, TABLE])

    class NotStatic:
        def analyze(self) -> Any:
            return None

    assert "has not implemented the 'analyze' static method" in analyze_error(NotStatic, [])
    assert "Invalid UDTF handler type" in analyze_error(NotStatic(), [])


def test_analyze_is_not_run_in_process() -> None:
    with pytest.raises(kerrors.PySparkNotImplementedError):
        udtf.analyze_udtf(Analyzed, "f", [], VALUES)


# --- analyze's constants: Spark's internal value, then `fromInternal` ---
def spark_type(name: str, **fields: Any) -> Any:
    """A stand-in for a pyspark type: what ``_internal_value`` reads (``fields``, ``elementType``, ...)."""
    return types.SimpleNamespace(name=name, **fields)


def internal(array: Any, dt: Any = None) -> Any:
    return udtf._internal_value(array[0], dt or spark_type("atomic"))


def test_constants_take_spark_s_internal_form() -> None:
    import datetime
    import decimal

    assert internal(pa.array([19724], pa.date32())) == 19724
    assert internal(pa.array([1_704_164_645_123_456], pa.timestamp("us", "UTC"))) == 1_704_164_645_123_456
    assert internal(pa.array([1_704_164_645_123_456], pa.timestamp("us"))) == 1_704_164_645_123_456
    assert internal(pa.array([datetime.timedelta(seconds=1)], pa.duration("us"))) == 1_000_000
    assert internal(pa.array([5_000], pa.time64("ns"))) == 5_000
    dec = internal(pa.array([decimal.Decimal("12.50")], pa.decimal128(5, 2)))
    assert str(dec) == "12.50"
    assert isinstance(dec, decimal.Decimal)
    assert internal(pa.array([b"\x00\xff"], pa.binary_view())) == b"\x00\xff"
    assert internal(pa.array(["s"], pa.string_view())) == "s"
    assert internal(pa.array([None], pa.int32())) is None
    date = spark_type("date")
    assert internal(pa.array([[1, None]], pa.list_(pa.date32())), spark_type("array", elementType=date)) == [1, None]
    st = spark_type(
        "struct",
        fields=[spark_type("f", dataType=date), spark_type("f", dataType=spark_type("int"))],
    )
    struct = pa.array([{"x": 0, "y": 7}], pa.struct([("x", pa.date32()), ("y", pa.int32())]))
    assert internal(struct, st) == (0, 7)
    mt = spark_type("map", keyType=spark_type("string"), valueType=date)
    assert internal(pa.array([[("k", 2)]], pa.map_(pa.string(), pa.date32())), mt) == {"k": 2}


class Converting:
    """A stand-in type that needs conversion, as ``DateType`` does."""

    def needConversion(self) -> bool:  # pyspark's name
        return True

    def fromInternal(self, v: int) -> Any:  # pyspark's name
        import datetime

        return datetime.date.fromordinal(v + datetime.date(1970, 1, 1).toordinal())


def test_analyze_gets_a_constant_through_from_internal(analyze_api: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import datetime

    seen: list[Any] = []

    class Dated:
        @staticmethod
        def analyze(d: AnalyzeArgument) -> AnalyzeResult:
            seen.append(d)
            return AnalyzeResult(schema=StructType([StructField("c", LongType())]))

    monkeypatch.setattr(udtf, "_parse_type", lambda text: Converting())
    values = pa.record_batch([pa.array([19724], pa.date32())], names=["_0"])
    udtf.analyze_udtf(Dated, "f", [{"type": '"date"', "column": 0, "is_table": False, "name": None}], values)
    assert seen[0].value == datetime.date(2024, 1, 2)
    assert seen[0].isConstantExpression


@pytest.mark.parametrize(
    ("values", "typ"),
    [
        (["x", None, "yz"], pa.string_view()),
        ([b"x", None, b"yz"], pa.binary_view()),
        (["x", None, "yz"], pa.string()),
        ([1, None, 3], pa.int64()),
    ],
)
def test_a_lateral_call_carries_its_input_row_of_any_type(values: list[Any], typ: Any) -> None:
    # A LATERAL call leads each result row with its input row's carried columns; pyarrow has no ``take`` for the
    # view types the engine sends (opus-1's Connect lateralJoin goldens, rows/udtf30{0,1}/dataframe_lateral).
    raw = pa.record_batch([pa.array([10, 20, 30], pa.int64()), pa.array(values, typ)], names=["x", "g"])
    cursor = udtf._Cursor()
    cursor.ref = (raw, 2)
    out = udtf._carry(pa.record_batch([pa.array([7, 8], pa.int64())], names=["r"]), cursor, 1)
    assert out.schema.names == ["g", "r"]
    assert out.schema.field("g").type == typ
    assert out.column(0).to_pylist() == [values[2], values[2]]
    assert udtf._carry(pa.record_batch([pa.array([], pa.int64())], names=["r"]), cursor, 1).num_rows == 0


def lateral_call(return_type: StructType = AB) -> UdtfCall:
    """One argument (frame column 0); frame column 1 is the input column a LATERAL call carries."""
    return UdtfCall.of(
        {
            "name": "my_udtf",
            "return_type": return_type.json(),
            "args": [[0, None]],
            "passthrough": [1],
        }
    )


def run_lateral(handler: Any, eval_type: int, frames: list[tuple[Any, ...]]) -> list[list[Any]]:
    runner = udtf.make_udtf_runner(handler, eval_type, lateral_call())
    out: list[list[Any]] = []
    for batch in runner(Source(frames), 0, 0):
        out.extend(list(r.values()) for r in batch.to_pylist())
    return out


@pytest.mark.parametrize("eval_type", [SQL_TABLE_UDF, SQL_ARROW_TABLE_UDF])
def test_a_lateral_call_leads_each_row_with_its_input_row_and_terminate_with_the_last(eval_type: int) -> None:
    Counting.cleaned = []
    frames = [
        ("batch", 0, pa.record_batch([pa.array([1, 2]), pa.array(["p", "q"], pa.string_view())], names=["_0", "g"])),
        ("batch", 0, pa.record_batch([pa.array([3]), pa.array(["r"], pa.string_view())], names=["_0", "g"])),
        ("end",),
    ]
    assert run_lateral(Counting, eval_type, frames) == [
        ["p", 1, 10],
        ["q", 2, 20],
        ["r", 3, 30],
        ["r", -1, 3],
    ]


def test_a_lateral_terminate_before_any_row_leads_with_nulls() -> None:
    # An input batch with no rows: `eval` never runs, so `terminate`'s rows lead with NULL input columns.
    frames = [
        ("batch", 0, pa.record_batch([pa.array([], pa.int64()), pa.array([], pa.string())], names=["_0", "g"])),
        ("end",),
    ]
    assert run_lateral(Counting, SQL_TABLE_UDF, frames) == [[None, -1, 0]]


def test_a_lateral_terminate_with_no_input_batch_at_all_is_an_error() -> None:
    with pytest.raises(Exception, match="before any input batch"):
        run_lateral(Counting, SQL_TABLE_UDF, [("end",)])


def test_an_arrow_udtf_is_no_lateral_call() -> None:
    frames = [
        ("batch", 0, pa.record_batch([pa.array([1]), pa.array(["p"])], names=["_0", "g"])),
        ("end",),
    ]
    with pytest.raises(Exception, match="cannot be a LATERAL call"):
        run_lateral(Counting, SQL_ARROW_UDTF, frames)


def test_java_kind_names_what_pyrolite_unpickles() -> None:
    import datetime
    import decimal

    kinds = [
        (True, "Boolean"),
        (1, "Integer"),
        (2**40, "Long"),
        (2**70, "BigInteger"),
        (1.5, "Double"),
        ("s", "String"),
        (b"b", "byte[]"),
        (bytearray(b"b"), "byte[]"),
        (decimal.Decimal("1.5"), "BigDecimal"),
        (datetime.date(2024, 1, 2), "Calendar"),
        ((1, 2), "Object[]"),
        (object, "Object"),
    ]
    for value, kind in kinds:
        assert udtf._java_kind(value) == kind, value
    assert udtf._java_kind([1]) == "List"
    assert udtf._java_kind({1: 2}) == "Map"


def test_java_narrowing_and_text() -> None:
    assert udtf._wrap(2**31, 32) == -(2**31)
    assert udtf._wrap(300, 8) == 44
    assert udtf._wrap(-1, 16) == -1
    assert udtf._java_string(True) == "true"
    assert udtf._java_string(False) == "false"
    assert udtf._java_string([1, None, "a"]) == "[1, null, a]"
    # A dict is a `java.util.HashMap` to the JVM: `toString` iterates it by bucket, not insertion. Expected texts
    # are JDK 21.0.13's `HashMap.toString` of the same puts in the same order.
    maps: list[tuple[dict[Any, Any], str]] = [
        ({"k": None, "j": 1.0}, "{j=1.0, k=null}"),
        ({"zeta": 1, "alpha": 2, "mid": 3, "éß": 4}, "{zeta=1, éß=4, alpha=2, mid=3}"),
        ({100: "a", 3: "b", 17: "c", -5: "d"}, "{17=c, 3=b, 100=a, -5=d}"),
        ({5000000000: "x", 1 << 40: "y"}, "{1099511627776=y, 5000000000=x}"),
        ({True: 1, False: 2}, "{false=2, true=1}"),
        ({2.5: "p", -0.0: "q", float("nan"): "r"}, "{-0.0=q, 2.5=p, NaN=r}"),
        ({None: 1, "a": 2}, "{null=1, a=2}"),
        (
            {f"key{19 - i}": i for i in range(20)},
            "{key1=18, key2=17, key0=19, key5=14, key6=13, key3=16, key4=15, key9=10, key7=12, key8=11, key11=8, "
            "key10=9, key15=4, key14=5, key13=6, key12=7, key19=0, key18=1, key17=2, key16=3}",
        ),
        # A key whose Java hash is identity-based keeps Python's order.
        ({b"b": 1, b"a": 2}, "{b'b'=1, b'a'=2}"),
    ]
    for value, text in maps:
        assert udtf._java_string(value) == text, value
    assert udtf._java_string(7) == "7"
    doubles = [
        (float("nan"), "NaN"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (1.5, "1.5"),
        (100.0, "100.0"),
        (1e7, "1.0E7"),
        (12345678.9, "1.23456789E7"),
        (-0.0001, "-1.0E-4"),
        (1e-10, "1.0E-10"),
    ]
    for value, text in doubles:
        assert udtf._java_double(value) == text, value


def test_from_java_matches_make_from_java() -> None:
    import datetime
    import decimal
    import math

    from krishiv._udf import types as t

    cases: list[tuple[Any, Any, Any]] = [
        (t.BooleanType(), True, True),
        (t.BooleanType(), 1, None),
        (t.ByteType(), 300, 44),
        (t.ShortType(), 2**16 + 5, 5),
        (t.IntegerType(), 2**31, -(2**31)),
        (t.LongType(), 5, 5),
        (t.IntegerType(), "1", None),
        (t.FloatType(), 1.5, 1.5),
        (t.FloatType(), 1, None),
        (t.DoubleType(), 2.5, 2.5),
        (t.DoubleType(), 2, None),
        (t.DecimalType(5, 2), decimal.Decimal("1.005"), decimal.Decimal("1.01")),
        (t.DecimalType(5, 2), 1.5, None),
        (t.DateType(), 19000, 19000),
        (t.DateType(), datetime.date(2024, 1, 2), None),
        (t.TimestampType(), 2**40, 2**40),
        (t.TimestampNTZType(), "x", None),
        (t.StringType(), 1.0, "1.0"),
        (t.StringType(), [True], "[true]"),
        (t.BinaryType(), "é", "é".encode()),
        (t.BinaryType(), b"ab", b"ab"),
        (t.BinaryType(), 1, None),
        (t.ArrayType(t.IntegerType()), [1, "x"], [1, None]),
        (t.ArrayType(t.IntegerType()), (2,), [2]),
        (t.ArrayType(t.IntegerType()), 3, None),
        (t.MapType(t.StringType(), t.LongType()), {"a": 1}, [("a", 1)]),
        (t.MapType(t.StringType(), t.LongType()), [1], None),
        (t.StructType([t.StructField("x", t.IntegerType())]), (1,), {"x": 1}),
        (t.StructType([t.StructField("x", t.IntegerType())]), [1], None),
        (t.NullType(), 1, None),
        (t.IntegerType(), None, None),
    ]
    for dt, value, expected in cases:
        assert udtf._from_java(dt, value) == expected, (dt, value)
    huge = udtf._from_java(t.FloatType(), 1e39)
    assert math.isinf(huge)
    assert huge > 0


def test_from_java_raises_the_jvm_classes() -> None:
    import decimal

    from krishiv._udf import types as t

    with pytest.raises(udtf.SparkJvmError) as e:
        udtf._from_java(t.DecimalType(3, 1), decimal.Decimal("12345.6"))
    assert e.value.condition == "NUMERIC_VALUE_OUT_OF_RANGE.WITHOUT_SUGGESTION"
    assert e.value.parameters["precision"] == "3"
    assert "NUMERIC_VALUE_OUT_OF_RANGE" in str(e.value)
    two = t.StructType([t.StructField("x", t.IntegerType()), t.StructField("y", t.IntegerType())])
    with pytest.raises(udtf.SparkJvmError) as e:
        udtf._from_java(two, (1,))
    assert e.value.condition == "STRUCT_ARRAY_LENGTH_MISMATCH"
    assert e.value.parameters == {"expected": "2", "actual": "1"}
