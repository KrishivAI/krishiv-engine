# SPDX-License-Identifier: Apache-2.0
"""The Python table-function (UDTF) cases, built against ``pyspark`` or ``krishiv``.

One module builds every case for both engines: ``gen_goldens.py`` runs it on PySpark 4.2.0 (local, a real JVM) and
writes ``goldens.json``; ``test_udtf_goldens.py`` runs it on a Krishiv session and compares. The cases cover the Python
eval types 300 (``@udtf``, rows through the JVM's ``EvaluatePython``), 301 (``@udtf(useArrow=True)``) and 302
(``@arrow_udtf``), with SQL (``spark.udtf.register``, ``TABLE(...)``, ``LATERAL``) and DataFrame calls:

- ``rows``: the rows of ``eval`` and ``terminate``, several ``eval`` calls and one instance per partition, NULL
  inputs, empty input;
- ``table``: a ``TABLE`` argument with ``PARTITION BY``, ``ORDER BY`` and ``WITH SINGLE PARTITION`` (a fresh
  instance and a ``terminate`` per partition);
- ``analyze``: the static method (result schema, ``withSinglePartition``, ``partitionBy``, ``orderBy``, ``select``,
  the constant arguments of each type as ``analyze`` sees them, ``analyze`` with ``@arrow_udtf``) and its failures;
- ``mistyped``: ``eval`` results of another type than the declared one, for 300, 301 and 302;
- ``errors``: the ``UDTF_*`` classes of ``read_udtf`` that a user can reach.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from collections.abc import Callable
from typing import Any

import pyarrow as pa

#: The table input: (id, g, v), inserted out of order so ``ORDER BY`` shows; ``v`` has a NULL; ``g`` has three groups.
TABLE_ROWS: list[tuple[Any, ...]] = [(2, "a", 20), (1, "a", 10), (4, "b", None), (3, "b", 30), (5, "c", 50)]
TABLE_SCHEMA = "id int, g string, v int"

#: Session conf for every case.
CONF = {
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.shuffle.partitions": "3",
}

KINDS = ("300", "301", "302")


class Api:
    """``pyspark`` or ``krishiv`` under one set of names (both packages mirror ``pyspark.sql``)."""

    AnalyzeArgument: Any
    AnalyzeResult: Any
    PartitioningColumn: Any
    OrderingColumn: Any
    SelectedColumn: Any
    SkipRestOfInputTableException: Any

    def __init__(self, root: str) -> None:
        self.F = importlib.import_module(f"{root}.sql.functions")
        self.T = importlib.import_module(f"{root}.sql.types")
        try:
            u = importlib.import_module(f"{root}.sql.udtf")
        except ImportError:
            u = None
        for name in (
            "AnalyzeArgument",
            "AnalyzeResult",
            "PartitioningColumn",
            "OrderingColumn",
            "SelectedColumn",
            "SkipRestOfInputTableException",
        ):
            setattr(self, name, getattr(u, name, None))

    def udtf(self, kind: str, return_type: Any) -> Callable[[type], Any]:
        """The decorator of one kind: 300 ``@udtf``, 301 ``@udtf(useArrow=True)``, 302 ``@arrow_udtf``."""
        if kind == "300":
            # ``useArrow=False`` is explicit: PySpark 4.2.0 defaults a bare ``@udtf`` to Arrow-optimized
            # (``spark.sql.execution.pythonUDTF.arrow.enabled`` is true), which is 301.
            return self.F.udtf(returnType=return_type, useArrow=False)  # type: ignore[no-any-return]
        if kind == "301":
            return self.F.udtf(returnType=return_type, useArrow=True)  # type: ignore[no-any-return]
        return self.F.arrow_udtf(returnType=return_type)  # type: ignore[no-any-return]


def prepare(spark: Any) -> None:
    """The views of every SQL case: ``t`` (the table input) and ``e`` (the same schema, no rows)."""
    df = spark.createDataFrame(TABLE_ROWS, TABLE_SCHEMA)
    df.createOrReplaceTempView("t")
    df.where("id < 0").createOrReplaceTempView("e")


def _ordered(df: Any) -> Any:
    """``df`` ordered by all its columns, so that the rows compare; a map column cannot be ordered by."""
    if "map<" in df.schema.simpleString():
        return df
    return df.orderBy(*df.columns)


def _query(api: Api, spark: Any, udtf: Any, sql: str, name: str = "f") -> Any:
    prepare(spark)
    spark.udtf.register(name, udtf)
    return _ordered(spark.sql(sql))


def _dataframe(spark: Any, build: Callable[[Any], Any]) -> Any:
    prepare(spark)
    return _ordered(build(spark.table("t")))


# ---------------------------------------------------------------------------------------------------------------
# The handler classes. Each ``make_*`` returns the decorated class of one kind; the row kinds (300, 301) share a
# body, the Arrow kind (302) has its own (``eval`` gets arrays or a record batch and yields tables).
# ---------------------------------------------------------------------------------------------------------------
def _tbl(**cols: tuple[pa.DataType, list[Any]]) -> pa.Table:
    return pa.table({k: pa.array(v, type=t) for k, (t, v) in cols.items()})


def make_repeat(api: Api, kind: str) -> Any:
    """``eval(a, n)`` yields ``(a, i)`` for ``i < n``."""
    if kind == "302":

        @api.udtf(kind, "a int, i int")
        class RepeatArrow:
            def eval(self, a: pa.Array, n: pa.Array) -> Any:
                rows = [(x, i) for x, k in zip(a.to_pylist(), n.to_pylist(), strict=True) for i in range(k)]
                yield _tbl(a=(pa.int32(), [r[0] for r in rows]), i=(pa.int32(), [r[1] for r in rows]))

        return RepeatArrow

    @api.udtf(kind, "a int, i int")
    class Repeat:
        def eval(self, a: int, n: int) -> Any:
            for i in range(n):
                yield (a, i)

    return Repeat


def make_counter(api: Api, kind: str) -> Any:
    """An instance counts its ``eval`` rows; ``terminate`` yields ``(-1, count)``."""
    if kind == "302":

        @api.udtf(kind, "v int, n int")
        class CounterArrow:
            def __init__(self) -> None:
                self.n = 0

            def eval(self, v: pa.Array) -> Any:
                vals = v.to_pylist()
                ns = list(range(self.n + 1, self.n + 1 + len(vals)))
                self.n += len(vals)
                yield _tbl(v=(pa.int32(), vals), n=(pa.int32(), ns))

            def terminate(self) -> Any:
                yield _tbl(v=(pa.int32(), [-1]), n=(pa.int32(), [self.n]))

        return CounterArrow

    @api.udtf(kind, "v int, n int")
    class Counter:
        def __init__(self) -> None:
            self.n = 0

        def eval(self, v: int | None) -> Any:
            self.n += 1
            yield (v, self.n)

        def terminate(self) -> Any:
            yield (-1, self.n)

    return Counter


def make_nullcheck(api: Api, kind: str) -> Any:
    """Yields whether the argument is NULL."""
    if kind == "302":

        @api.udtf(kind, "is_null boolean")
        class NullCheckArrow:
            def eval(self, a: pa.Array) -> Any:
                yield _tbl(is_null=(pa.bool_(), [x is None for x in a.to_pylist()]))

        return NullCheckArrow

    @api.udtf(kind, "is_null boolean")
    class NullCheck:
        def eval(self, a: Any) -> Any:
            yield (a is None,)

    return NullCheck


def make_empty(api: Api, kind: str) -> Any:
    """``eval`` yields nothing; ``terminate`` yields one row, so that a call of it shows."""
    if kind == "302":

        @api.udtf(kind, "r int")
        class EmptyArrow:
            def eval(self, *args: Any) -> Any:
                return
                yield

            def terminate(self) -> Any:
                yield _tbl(r=(pa.int32(), [-1]))

        return EmptyArrow

    @api.udtf(kind, "r int")
    class Empty:
        def eval(self, *args: Any) -> Any:
            return
            yield

        def terminate(self) -> Any:
            yield (-1,)

    return Empty


def make_table_rows(api: Api, kind: str) -> Any:
    """A ``TABLE(t)`` argument: ``eval`` yields ``(g, id, n)`` with the instance's running count and ``terminate``
    yields ``(last g, -1, count)``; the ``seen`` column is the ids as the instance saw them, in order."""
    rt = "g string, id int, n int, seen string"
    if kind == "302":

        @api.udtf(kind, rt)
        class TableArrow:
            def __init__(self) -> None:
                self.ids: list[int] = []
                self.g: str | None = None

            def eval(self, batch: pa.RecordBatch) -> Any:
                gs, ids = batch.column("g").to_pylist(), batch.column("id").to_pylist()
                ns, seen = [], []
                for g, i in zip(gs, ids, strict=True):
                    self.g = g
                    self.ids.append(i)
                    ns.append(len(self.ids))
                    seen.append(",".join(map(str, self.ids)))
                yield _tbl(g=(pa.string(), gs), id=(pa.int32(), ids), n=(pa.int32(), ns), seen=(pa.string(), seen))

            def terminate(self) -> Any:
                seen = ",".join(map(str, self.ids))
                yield _tbl(
                    g=(pa.string(), [self.g]),
                    id=(pa.int32(), [-1]),
                    n=(pa.int32(), [len(self.ids)]),
                    seen=(pa.string(), [seen]),
                )

        return TableArrow

    @api.udtf(kind, rt)
    class TableRows:
        def __init__(self) -> None:
            self.ids: list[int] = []
            self.g: str | None = None

        def eval(self, row: Any) -> Any:
            self.g = row["g"]
            self.ids.append(row["id"])
            yield (row["g"], row["id"], len(self.ids), ",".join(map(str, self.ids)))

        def terminate(self) -> Any:
            yield (self.g, -1, len(self.ids), ",".join(map(str, self.ids)))

    return TableRows


def make_skip_rest(api: Api, kind: str) -> Any:
    """``SkipRestOfInputTableException`` after the first row of a partition: ``terminate`` still runs."""
    skip = api.SkipRestOfInputTableException
    rt = "g string, id int"
    if kind == "302":

        @api.udtf(kind, rt)
        class SkipArrow:
            def eval(self, batch: pa.RecordBatch) -> Any:
                yield _tbl(
                    g=(pa.string(), batch.column("g").to_pylist()[:1]),
                    id=(pa.int32(), batch.column("id").to_pylist()[:1]),
                )
                raise skip()

            def terminate(self) -> Any:
                yield _tbl(g=(pa.string(), ["end"]), id=(pa.int32(), [-1]))

        return SkipArrow

    @api.udtf(kind, rt)
    class Skip:
        def eval(self, row: Any) -> Any:
            yield (row["g"], row["id"])
            raise skip()

        def terminate(self) -> Any:
            yield ("end", -1)

    return Skip


# ---------------------------------------------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------------------------------------------
def make_describe(api: Api, kind: str) -> Any:
    """``analyze`` records what it was given for each argument; ``eval`` yields it (the analyze result is passed to
    ``__init__``)."""
    AnalyzeResult = api.AnalyzeResult
    T = api.T
    schema = (
        T.StructType()
        .add("position", T.IntegerType())
        .add("data_type", T.StringType())
        .add("value", T.StringType())
        .add("is_constant", T.BooleanType())
        .add("is_table", T.BooleanType())
    )

    def describe(args: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        return [
            (i, a.dataType.simpleString(), repr(a.value), a.isConstantExpression, a.isTable) for i, a in enumerate(args)
        ]

    # Spark rebuilds the result for ``__init__`` with ``dataclasses.replace``, which keeps declared fields only: a
    # subclass carries its extra data as a dataclass field.
    @dataclasses.dataclass
    class ResultWithInfo(AnalyzeResult):  # type: ignore[valid-type,misc]
        info: list[Any] = dataclasses.field(default_factory=list)

    if kind == "302":

        @api.udtf(kind, None)
        class DescribeArrow:
            def __init__(self, result: Any = None) -> None:
                self.info = getattr(result, "info", [])

            @staticmethod
            def analyze(*args: Any) -> Any:
                return ResultWithInfo(schema=schema, info=describe(args))  # ty: ignore[unknown-argument]

            def eval(self, *args: Any) -> Any:
                yield pa.table(
                    {
                        "position": pa.array([r[0] for r in self.info], pa.int32()),
                        "data_type": pa.array([r[1] for r in self.info], pa.string()),
                        "value": pa.array([r[2] for r in self.info], pa.string()),
                        "is_constant": pa.array([r[3] for r in self.info], pa.bool_()),
                        "is_table": pa.array([r[4] for r in self.info], pa.bool_()),
                    }
                )

        return DescribeArrow

    @api.udtf(kind, None)
    class Describe:
        def __init__(self, result: Any = None) -> None:
            self.info = getattr(result, "info", [])

        @staticmethod
        def analyze(*args: Any) -> Any:
            return ResultWithInfo(schema=schema, info=describe(args))  # ty: ignore[unknown-argument]

        def eval(self, *args: Any) -> Any:
            yield from self.info

    return Describe


def make_analyze_schema(api: Api, kind: str) -> Any:
    """The result schema comes from the constant argument: the column is named by it."""
    AnalyzeResult = api.AnalyzeResult
    T = api.T
    if kind == "302":

        @api.udtf(kind, None)
        class NamedArrow:
            @staticmethod
            def analyze(name: Any) -> Any:
                return AnalyzeResult(schema=T.StructType().add(name.value, T.IntegerType()))

            def eval(self, name: pa.Array) -> Any:
                yield pa.table({"x": pa.array([1], pa.int32())})

        return NamedArrow

    @api.udtf(kind, None)
    class Named:
        @staticmethod
        def analyze(name: Any) -> Any:
            return AnalyzeResult(schema=T.StructType().add(name.value, T.IntegerType()))

        def eval(self, name: str) -> Any:
            yield (1,)

    return Named


def make_analyzed_table(api: Api, kind: str, **options: Any) -> Any:
    """A ``TABLE`` argument UDTF whose ``analyze`` asks for ``options`` (``withSinglePartition``, ``partitionBy``,
    ``orderBy``, ``select`` are lists of the names, built with the SDK classes)."""
    AnalyzeResult = api.AnalyzeResult
    PartitioningColumn, OrderingColumn, SelectedColumn = (
        api.PartitioningColumn,
        api.OrderingColumn,
        api.SelectedColumn,
    )
    T = api.T
    single = options.get("withSinglePartition", False)
    part = [PartitioningColumn(n) for n in options.get("partitionBy", [])]
    order = [OrderingColumn(n, asc) for n, asc in options.get("orderBy", [])]
    select = [SelectedColumn(n, a) for n, a in options.get("select", [])]
    schema = T.StructType().add("g", T.StringType()).add("id", T.IntegerType()).add("n", T.IntegerType())

    def analyze_fn(*args: Any) -> Any:
        return AnalyzeResult(schema=schema, withSinglePartition=single, partitionBy=part, orderBy=order, select=select)

    if select:
        # With ``select``, the instance gets exactly the selected columns, by alias or name.
        schema = T.StructType().add("a", T.StringType()).add("b", T.StringType()).add("n", T.IntegerType())

    if kind == "302":

        @api.udtf(kind, None)
        class AnalyzedArrow:
            analyze = staticmethod(analyze_fn)

            def __init__(self) -> None:
                self.n = 0

            def eval(self, batch: pa.RecordBatch) -> Any:
                cols = [c.to_pylist() for c in batch.columns]
                names = batch.schema.names
                if select:
                    n = list(range(self.n + 1, self.n + 1 + batch.num_rows))
                    self.n += batch.num_rows
                    yield _tbl(
                        a=(pa.string(), [str(x) for x in cols[0]]),
                        b=(pa.string(), [",".join(names)] * batch.num_rows),
                        n=(pa.int32(), n),
                    )
                    return
                ids = batch.column("id").to_pylist()
                n = list(range(self.n + 1, self.n + 1 + len(ids)))
                self.n += len(ids)
                yield _tbl(g=(pa.string(), batch.column("g").to_pylist()), id=(pa.int32(), ids), n=(pa.int32(), n))

        return AnalyzedArrow

    @api.udtf(kind, None)
    class Analyzed:
        analyze = staticmethod(analyze_fn)

        def __init__(self) -> None:
            self.n = 0

        def eval(self, row: Any) -> Any:
            self.n += 1
            if select:
                yield (str(row[0]), ",".join(row.__fields__), self.n)
            else:
                yield (row["g"], row["id"], self.n)

    return Analyzed


def make_analyze_raises(api: Api, kind: str, how: str) -> Any:
    """An ``analyze`` that fails: raises, returns the wrong type, or takes the wrong arguments."""
    AnalyzeResult = api.AnalyzeResult
    T = api.T
    schema = T.StructType().add("x", T.IntegerType())

    def analyze_raises(a: Any) -> Any:
        raise ValueError("analyze refuses this call")

    def analyze_wrong_type(a: Any) -> Any:
        return schema  # a StructType, not an AnalyzeResult

    def analyze_two(a: Any, b: Any) -> Any:
        return AnalyzeResult(schema=schema)

    def analyze_no_schema(a: Any) -> Any:
        return AnalyzeResult(schema=None)

    analyze_fn = {
        "raises": analyze_raises,
        "wrong_type": analyze_wrong_type,
        "two": analyze_two,
        "none": analyze_no_schema,
    }[how]
    if kind == "302":

        @api.udtf(kind, None)
        class FailArrow:
            analyze = staticmethod(analyze_fn)

            def eval(self, a: pa.Array) -> Any:
                yield pa.table({"x": pa.array([1], pa.int32())})

        return FailArrow

    @api.udtf(kind, None)
    class Fail:
        analyze = staticmethod(analyze_fn)

        def eval(self, a: Any) -> Any:
            yield (1,)

    return Fail


# ---------------------------------------------------------------------------------------------------------------
# eval results of another type than the declared one, and the read_udtf errors
# ---------------------------------------------------------------------------------------------------------------
def make_yielding(api: Api, kind: str, return_type: Any, rows: Callable[[], Any], arrow: Callable[[], Any]) -> Any:
    """A UDTF whose ``eval(a)`` yields ``rows()`` (300, 301) or ``arrow()`` (302)."""
    if kind == "302":

        @api.udtf(kind, return_type)
        class YieldingArrow:
            def eval(self, a: pa.Array) -> Any:
                return arrow()

        return YieldingArrow

    @api.udtf(kind, return_type)
    class Yielding:
        def eval(self, a: Any) -> Any:
            return rows()

    return Yielding


def _gen(*values: Any) -> Callable[[], Any]:
    def run() -> Any:
        yield from values

    return run


def _tables(**cols: pa.Array) -> Callable[[], Any]:
    def run() -> Any:
        yield pa.table(cols)

    return run


def _scalar_type(api: Api, name: str) -> Any:
    T = api.T
    return {"int": T.IntegerType(), "string": T.StringType()}[name]


def _mistyped(api: Api, kind: str, case: str) -> Any:
    """``(declared type, python rows, arrow tables)`` of each mistyped-result case."""
    import datetime
    import decimal

    T = api.T
    nn_int = T.StructType([T.StructField("a", T.IntegerType(), False)])
    table: dict[str, tuple[Any, Callable[[], Any], Callable[[], Any]]] = {
        "int_for_string": ("a string", _gen((1,)), _tables(a=pa.array([1], pa.int32()))),
        "string_for_int": ("a int", _gen(("x",)), _tables(a=pa.array(["x"], pa.string()))),
        "numeric_string_for_int": ("a int", _gen(("12",)), _tables(a=pa.array(["12"], pa.string()))),
        "float_for_int": ("a int", _gen((1.5,)), _tables(a=pa.array([1.5], pa.float64()))),
        "whole_float_for_int": ("a int", _gen((2.0,)), _tables(a=pa.array([2.0], pa.float64()))),
        "int_for_double": ("a double", _gen((1,)), _tables(a=pa.array([1], pa.int64()))),
        "bool_for_int": ("a int", _gen((True,)), _tables(a=pa.array([True], pa.bool_()))),
        "int_for_bool": ("a boolean", _gen((1,)), _tables(a=pa.array([1], pa.int32()))),
        "overflow_int": ("a int", _gen((2**40,)), _tables(a=pa.array([2**40], pa.int64()))),
        "long_for_int_small": ("a int", _gen((7,)), _tables(a=pa.array([7], pa.int64()))),
        "none_for_not_null": (nn_int, _gen((None,)), _tables(a=pa.array([None], pa.int32()))),
        "none_nullable": ("a int", _gen((None,)), _tables(a=pa.array([None], pa.int32()))),
        "string_for_date": ("a date", _gen(("2020-01-02",)), _tables(a=pa.array(["2020-01-02"], pa.string()))),
        "date_for_string": (
            "a string",
            _gen((datetime.date(2020, 1, 2),)),
            _tables(a=pa.array([datetime.date(2020, 1, 2)], pa.date32())),
        ),
        "int_for_decimal": ("a decimal(5,2)", _gen((3,)), _tables(a=pa.array([3], pa.int64()))),
        "float_for_decimal": ("a decimal(5,2)", _gen((1.25,)), _tables(a=pa.array([1.25], pa.float64()))),
        "decimal_for_int": (
            "a int",
            _gen((decimal.Decimal("3"),)),
            _tables(a=pa.array([decimal.Decimal("3")], pa.decimal128(5, 0))),
        ),
        "bytes_for_string": ("a string", _gen((b"ab",)), _tables(a=pa.array([b"ab"], pa.binary()))),
        "string_for_binary": ("a binary", _gen(("ab",)), _tables(a=pa.array(["ab"], pa.string()))),
        "list_for_array": ("a array<int>", _gen(([1, 2],)), _tables(a=pa.array([[1, 2]], pa.list_(pa.int32())))),
        "mixed_list_for_array": (
            "a array<int>",
            _gen((["x", 2],)),
            _tables(a=pa.array([["x", "y"]], pa.list_(pa.string()))),
        ),
        "tuple_for_struct": (
            "a struct<x:int,y:string>",
            _gen(((1, "p"),)),
            _tables(a=pa.array([{"x": 1, "y": "p"}], pa.struct([("x", pa.int32()), ("y", pa.string())]))),
        ),
        "dict_for_struct": (
            "a struct<x:int,y:string>",
            _gen(({"x": 1, "y": "p"},)),
            _tables(a=pa.array([{"x": 1, "y": "p"}], pa.struct([("x", pa.int32()), ("y", pa.string())]))),
        ),
        "short_tuple_for_struct": (
            "a struct<x:int,y:string>",
            _gen(((1,),)),
            _tables(a=pa.array([{"x": 1}], pa.struct([("x", pa.int32())]))),
        ),
        "none_for_struct": (
            "a struct<x:int>",
            _gen((None,)),
            _tables(a=pa.array([None], pa.struct([("x", pa.int32())]))),
        ),
        "struct_nested_mistyped": (
            "a struct<x:int>",
            _gen((("s",),)),
            _tables(a=pa.array([{"x": "s"}], pa.struct([("x", pa.string())]))),
        ),
        "dict_for_map": (
            "a map<string,int>",
            _gen(({"k": 1},)),
            _tables(a=pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int32()))),
        ),
        "two_columns_first_wrong": (
            "a int, b string",
            _gen(("x", "y")),
            _tables(a=pa.array(["x"], pa.string()), b=pa.array(["y"], pa.string())),
        ),
    }
    return table[case]


MISTYPED = (
    "int_for_string",
    "string_for_int",
    "numeric_string_for_int",
    "float_for_int",
    "whole_float_for_int",
    "int_for_double",
    "bool_for_int",
    "int_for_bool",
    "overflow_int",
    "long_for_int_small",
    "none_for_not_null",
    "none_nullable",
    "string_for_date",
    "date_for_string",
    "int_for_decimal",
    "float_for_decimal",
    "decimal_for_int",
    "bytes_for_string",
    "string_for_binary",
    "list_for_array",
    "mixed_list_for_array",
    "tuple_for_struct",
    "dict_for_struct",
    "short_tuple_for_struct",
    "none_for_struct",
    "struct_nested_mistyped",
    "dict_for_map",
    "two_columns_first_wrong",
)


def _mistyped_case(api: Api, spark: Any, kind: str, case: str) -> Any:
    rt, rows, arrow = _mistyped(api, kind, case)
    u = make_yielding(api, kind, rt, rows, arrow)
    return _query(api, spark, u, "SELECT * FROM f(1)")


#: Shapes of the eval output that are no row: (name, python rows, arrow result).
_BAD_ROWS = (
    "scalar_int",
    "list_row",
    "string_row",
    "dict_row",
    "row_object",
    "none_row",
    "wide_row",
    "narrow_row",
    "not_iterable",
    "none_result",
    "list_result",
    "iterator_of_rows_then_none",
)


def _bad_rows(api: Api, kind: str, case: str) -> tuple[Callable[[], Any], Callable[[], Any]]:
    wide = _tables(a=pa.array([1], pa.int32()), b=pa.array([2], pa.int32()))
    table = {
        "scalar_int": (_gen(1), lambda: iter([1])),
        "list_row": (_gen([1]), lambda: iter([[1]])),
        "string_row": (_gen("1"), lambda: iter(["1"])),
        "dict_row": (_gen({"a": 1}), lambda: iter([{"a": 1}])),
        "row_object": (lambda: iter([_row(api, 1)]), lambda: iter([_row(api, 1)])),
        "none_row": (_gen(None), lambda: iter([None])),
        "wide_row": (_gen((1, 2)), wide),
        "narrow_row": (_gen(()), lambda: iter([pa.table({})])),
        "not_iterable": (lambda: 5, lambda: 5),
        "none_result": (lambda: None, lambda: None),
        "list_result": (lambda: [(1,), (2,)], lambda: [pa.table({"a": pa.array([1, 2], pa.int32())})]),
        "iterator_of_rows_then_none": (
            _gen((1,), None),
            lambda: iter([pa.record_batch({"a": pa.array([1], pa.int32())}), None]),
        ),
    }
    return table[case]


def _row(api: Api, value: Any) -> Any:
    return importlib.import_module(f"{api.F.__name__.split('.')[0]}.sql").Row(a=value)


def _bad_rows_case(api: Api, spark: Any, kind: str, case: str) -> Any:
    rows, arrow = _bad_rows(api, kind, case)
    u = make_yielding(api, kind, "a int", rows, arrow)
    return _query(api, spark, u, "SELECT * FROM f(1)")


def make_raising(api: Api, kind: str, where: str) -> Any:
    """A UDTF that raises in ``__init__``, ``eval`` or ``terminate``."""
    if kind == "302":

        @api.udtf(kind, "a int")
        class RaisingArrow:
            def __init__(self) -> None:
                if where == "init":
                    raise ValueError("init failed")

            def eval(self, a: pa.Array) -> Any:
                if where == "eval":
                    raise ValueError("eval failed")
                yield _tbl(a=(pa.int32(), [1]))

            def terminate(self) -> Any:
                if where == "terminate":
                    raise ValueError("terminate failed")
                yield _tbl(a=(pa.int32(), [2]))

        return RaisingArrow

    @api.udtf(kind, "a int")
    class Raising:
        def __init__(self) -> None:
            if where == "init":
                raise ValueError("init failed")

        def eval(self, a: int) -> Any:
            if where == "eval":
                raise ValueError("eval failed")
            yield (1,)

        def terminate(self) -> Any:
            if where == "terminate":
                raise ValueError("terminate failed")
            yield (2,)

    return Raising


def make_ctor(api: Api, kind: str, with_analyze: bool) -> Any:
    """A constructor with an argument: refused without ``analyze``, and with an ``analyze`` whose result it does not
    accept (a second positional)."""
    AnalyzeResult = api.AnalyzeResult
    T = api.T
    if with_analyze:

        @api.udtf(kind, None)
        class WithAnalyze:
            @staticmethod
            def analyze(a: Any) -> Any:
                return AnalyzeResult(schema=T.StructType().add("a", T.IntegerType()))

            def __init__(self, x: Any, y: Any) -> None:
                self.x = x

            def eval(self, a: Any) -> Any:
                yield (1,)

        return WithAnalyze

    @api.udtf(kind, "a int")
    class NoAnalyze:
        def __init__(self, x: Any) -> None:
            self.x = x

        def eval(self, a: Any) -> Any:
            yield (1,)

    return NoAnalyze


# ---------------------------------------------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------------------------------------------
def _cases() -> dict[str, Callable[[Api, Any], Any]]:
    cases: dict[str, Callable[[Api, Any], Any]] = {}

    def add(name: str, fn: Callable[[Api, Any], Any]) -> None:
        cases[name] = fn

    for kind in KINDS:
        k = f"udtf{kind}"

        # rows: eval, terminate, several calls per partition, NULLs, empty input
        add(
            f"rows/{k}/eval_sql",
            lambda api, spark, kind=kind: _query(api, spark, make_repeat(api, kind), "SELECT * FROM f(3, 2)"),
        )
        add(
            f"rows/{k}/eval_dataframe",
            lambda api, spark, kind=kind: _dataframe(
                spark, lambda t: make_repeat(api, kind)(api.F.lit(7), api.F.lit(3))
            ),
        )
        add(
            f"rows/{k}/eval_lateral",
            lambda api, spark, kind=kind: _query(
                api,
                spark,
                make_repeat(api, kind),
                "SELECT t.id, f.* FROM t, LATERAL f(t.id, t.id - 3) AS f",
            ),
        )
        add(
            f"rows/{k}/lateral_left_join",
            lambda api, spark, kind=kind: _query(
                api,
                spark,
                make_repeat(api, kind),
                "SELECT t.id, f.* FROM t LEFT JOIN LATERAL f(t.id, t.id - 3) AS f",
            ),
        )
        add(
            f"rows/{k}/dataframe_lateral",
            lambda api, spark, kind=kind: _dataframe(
                spark,
                lambda t: t.lateralJoin(
                    make_repeat(api, kind)(api.F.col("id").outer(), api.F.col("id").outer() - 3).alias("f")
                ),
            ),
        )
        add(
            f"rows/{k}/terminate_constant_args",
            lambda api, spark, kind=kind: _query(api, spark, make_counter(api, kind), "SELECT * FROM f(5)"),
        )
        add(
            f"rows/{k}/terminate_lateral_one_instance",
            lambda api, spark, kind=kind: _query(
                api, spark, make_counter(api, kind), "SELECT t.id, f.* FROM t, LATERAL f(t.v) AS f"
            ),
        )
        add(
            f"rows/{k}/null_constant",
            lambda api, spark, kind=kind: _query(
                api, spark, make_nullcheck(api, kind), "SELECT * FROM f(CAST(NULL AS INT))"
            ),
        )
        add(
            f"rows/{k}/null_lateral",
            lambda api, spark, kind=kind: _query(
                api, spark, make_nullcheck(api, kind), "SELECT t.id, f.* FROM t, LATERAL f(t.v) AS f"
            ),
        )
        add(
            f"rows/{k}/empty_eval_output",
            lambda api, spark, kind=kind: _query(api, spark, make_empty(api, kind), "SELECT * FROM f(1)"),
        )
        add(
            f"rows/{k}/empty_lateral_input",
            lambda api, spark, kind=kind: _query(
                api, spark, make_empty(api, kind), "SELECT e.id, f.* FROM e, LATERAL f(e.v) AS f"
            ),
        )
        add(
            f"rows/{k}/empty_table_input",
            lambda api, spark, kind=kind: _query(api, spark, make_empty(api, kind), "SELECT * FROM f(TABLE(e))"),
        )
        add(
            f"rows/{k}/empty_table_partitioned",
            lambda api, spark, kind=kind: _query(
                api, spark, make_empty(api, kind), "SELECT * FROM f(TABLE(e) PARTITION BY g)"
            ),
        )
        add(
            f"rows/{k}/zero_rows_requested",
            lambda api, spark, kind=kind: _query(api, spark, make_repeat(api, kind), "SELECT * FROM f(3, 0)"),
        )

        # table arguments
        tbl = make_table_rows
        for name, sql in (
            ("plain", "SELECT * FROM f(TABLE(t))"),
            ("partition_by", "SELECT * FROM f(TABLE(t) PARTITION BY g)"),
            ("partition_order", "SELECT * FROM f(TABLE(t) PARTITION BY g ORDER BY id)"),
            ("partition_order_desc", "SELECT * FROM f(TABLE(t) PARTITION BY g ORDER BY id DESC)"),
            ("partition_order_expr", "SELECT * FROM f(TABLE(t) PARTITION BY g ORDER BY coalesce(v, 0) DESC, id)"),
            ("partition_expr", "SELECT * FROM f(TABLE(t) PARTITION BY id % 2 ORDER BY id)"),
            ("partition_two_keys", "SELECT * FROM f(TABLE(t) PARTITION BY (g, v IS NULL) ORDER BY id)"),
            ("single_partition", "SELECT * FROM f(TABLE(t) WITH SINGLE PARTITION)"),
            ("single_partition_order", "SELECT * FROM f(TABLE(t) WITH SINGLE PARTITION ORDER BY id DESC)"),
            ("subquery", "SELECT * FROM f(TABLE(SELECT * FROM t WHERE id > 1) PARTITION BY g ORDER BY id)"),
            ("named_view_select", "SELECT * FROM f(row => TABLE(t) PARTITION BY g ORDER BY id)"),
            ("order_without_partition", "SELECT * FROM f(TABLE(t) ORDER BY id)"),
        ):
            add(
                f"table/{k}/{name}",
                lambda api, spark, kind=kind, sql=sql, tbl=tbl: _query(api, spark, tbl(api, kind), sql),
            )
        add(
            f"table/{k}/dataframe_partition_order",
            lambda api, spark, kind=kind: _dataframe(
                spark,
                lambda t: make_table_rows(api, kind)(t.asTable().partitionBy("g").orderBy("id")),
            ),
        )
        add(
            f"table/{k}/dataframe_single_partition",
            lambda api, spark, kind=kind: _dataframe(
                spark,
                lambda t: make_table_rows(api, kind)(t.asTable().withSinglePartition()),
            ),
        )
        add(
            f"table/{k}/dataframe_plain",
            lambda api, spark, kind=kind: _dataframe(spark, lambda t: make_table_rows(api, kind)(t.asTable())),
        )
        add(
            f"table/{k}/skip_rest_of_input",
            lambda api, spark, kind=kind: _query(
                api, spark, make_skip_rest(api, kind), "SELECT * FROM f(TABLE(t) PARTITION BY g ORDER BY id)"
            ),
        )
        add(
            f"table/{k}/table_and_scalar_mix",
            lambda api, spark, kind=kind: _query(
                api, spark, make_table_rows(api, kind), "SELECT * FROM f(TABLE(t), 1)"
            ),
        )
        add(
            f"table/{k}/null_values_partition_key",
            lambda api, spark, kind=kind: _query(
                api, spark, make_table_rows(api, kind), "SELECT * FROM f(TABLE(t) PARTITION BY v ORDER BY id)"
            ),
        )

        # analyze
        add(
            f"analyze/{k}/schema_from_argument",
            lambda api, spark, kind=kind: _query(
                api, spark, make_analyze_schema(api, kind), "SELECT * FROM f('colname')"
            ),
        )
        for const, expr in (
            ("int", "1"),
            ("expression", "1 + 1"),
            ("string", "'abc'"),
            ("boolean", "true"),
            ("double", "1.5D"),
            ("decimal", "1.50BD"),
            ("decimal_cast", "CAST(1.5 AS DECIMAL(10, 3))"),
            ("date", "DATE '2020-01-02'"),
            ("timestamp", "TIMESTAMP '2020-01-02 03:04:05'"),
            ("timestamp_ntz", "TIMESTAMP_NTZ '2020-01-02 03:04:05'"),
            ("binary", "X'0AFF'"),
            ("null", "NULL"),
            ("null_typed", "CAST(NULL AS DATE)"),
            ("array", "array(1, 2)"),
            ("struct", "named_struct('x', 1, 'y', 'a')"),
            ("map", "map('k', 1)"),
            ("interval", "INTERVAL '1' DAY"),
            ("non_constant_column", "t.id"),
        ):
            if kind != "300" and const not in ("date", "timestamp", "decimal", "binary", "null", "non_constant_column"):
                continue
            if const == "non_constant_column":
                sql = "SELECT t.id, f.* FROM t, LATERAL f(t.id) AS f"
            else:
                sql = f"SELECT * FROM f({expr})"
            add(
                f"analyze/{k}/argument_{const}",
                lambda api, spark, kind=kind, sql=sql: _query(api, spark, make_describe(api, kind), sql),
            )
        add(
            f"analyze/{k}/named_arguments",
            lambda api, spark, kind=kind: _query(
                api, spark, make_describe(api, kind), "SELECT * FROM f(b => 2, a => 'x')"
            ),
        )
        add(
            f"analyze/{k}/table_argument_seen",
            lambda api, spark, kind=kind: _query(api, spark, make_describe(api, kind), "SELECT * FROM f(TABLE(t), 5) "),
        )
        options: dict[str, dict[str, Any]] = {
            "single_partition": {"withSinglePartition": True},
            "partition_by": {"partitionBy": ["g"]},
            "partition_by_order_by": {"partitionBy": ["g"], "orderBy": [("id", False)]},
            "partition_by_expr_order_by_expr": {"partitionBy": ["id % 2"], "orderBy": [("coalesce(v, 0)", True)]},
            "order_by_without_partition_by": {"orderBy": [("id", True)]},
            "select": {"select": [("id", "a"), ("g", "g2")]},
            "select_expression_needs_alias": {"select": [("id + 1", "")]},
            "select_expression": {"select": [("id + 1", "a"), ("upper(g)", "g2")]},
            "select_unknown_column": {"select": [("nope", "")]},
            "partition_by_unknown_column": {"partitionBy": ["nope"]},
            "single_partition_and_partition_by": {"withSinglePartition": True, "partitionBy": ["g"]},
            "partition_by_select_order_by": {
                "partitionBy": ["g"],
                "orderBy": [("id", True)],
                "select": [("id", "a")],
            },
        }
        for opt, kw in options.items():
            add(
                f"analyze/{k}/table_{opt}",
                lambda api, spark, kind=kind, kw=kw: _query(
                    api, spark, make_analyzed_table(api, kind, **kw), "SELECT * FROM f(TABLE(t))"
                ),
            )
        add(
            f"analyze/{k}/partition_by_and_explicit_partition_by",
            lambda api, spark, kind=kind: _query(
                api,
                spark,
                make_analyzed_table(api, kind, partitionBy=["g"]),
                "SELECT * FROM f(TABLE(t) PARTITION BY g)",
            ),
        )
        add(
            f"analyze/{k}/single_partition_and_explicit_single_partition",
            lambda api, spark, kind=kind: _query(
                api,
                spark,
                make_analyzed_table(api, kind, withSinglePartition=True),
                "SELECT * FROM f(TABLE(t) WITH SINGLE PARTITION)",
            ),
        )
        add(
            f"analyze/{k}/single_partition_without_table",
            lambda api, spark, kind=kind: _query(
                api, spark, make_analyzed_table(api, kind, withSinglePartition=True), "SELECT * FROM f(1)"
            ),
        )
        add(
            f"analyze/{k}/partition_by_without_table",
            lambda api, spark, kind=kind: _query(
                api, spark, make_analyzed_table(api, kind, partitionBy=["g"]), "SELECT * FROM f(1)"
            ),
        )
        add(
            f"analyze/{k}/dataframe_partition_by",
            lambda api, spark, kind=kind: _dataframe(
                spark, lambda t: make_analyzed_table(api, kind, partitionBy=["g"], orderBy=[("id", True)])(t.asTable())
            ),
        )
        for how in ("raises", "wrong_type", "two", "none"):
            add(
                f"errors/{k}/analyze_{how}",
                lambda api, spark, kind=kind, how=how: _query(
                    api, spark, make_analyze_raises(api, kind, how), "SELECT * FROM f(1)"
                ),
            )

        # mistyped results
        for case in MISTYPED:
            add(
                f"mistyped/{k}/{case}",
                lambda api, spark, kind=kind, case=case: _mistyped_case(api, spark, kind, case),
            )

        # errors
        for case in _BAD_ROWS:
            add(
                f"errors/{k}/rows_{case}",
                lambda api, spark, kind=kind, case=case: _bad_rows_case(api, spark, kind, case),
            )
        for where in ("init", "eval", "terminate"):
            add(
                f"errors/{k}/raises_in_{where}",
                lambda api, spark, kind=kind, where=where: _query(
                    api, spark, make_raising(api, kind, where), "SELECT * FROM f(1)"
                ),
            )
        add(
            f"errors/{k}/eval_too_few_arguments",
            lambda api, spark, kind=kind: _query(api, spark, make_repeat(api, kind), "SELECT * FROM f(1)"),
        )
        add(
            f"errors/{k}/eval_too_many_arguments",
            lambda api, spark, kind=kind: _query(api, spark, make_repeat(api, kind), "SELECT * FROM f(1, 2, 3)"),
        )
        add(
            f"errors/{k}/eval_wrong_argument_name",
            lambda api, spark, kind=kind: _query(api, spark, make_repeat(api, kind), "SELECT * FROM f(a => 1, z => 2)"),
        )
        add(
            f"errors/{k}/constructor_argument_without_analyze",
            lambda api, spark, kind=kind: _query(api, spark, make_ctor(api, kind, False), "SELECT * FROM f(1)"),
        )
        add(
            f"errors/{k}/constructor_arguments_with_analyze",
            lambda api, spark, kind=kind: _query(api, spark, make_ctor(api, kind, True), "SELECT * FROM f(1)"),
        )
        add(
            f"errors/{k}/table_argument_to_scalar_eval",
            lambda api, spark, kind=kind: _query(api, spark, make_repeat(api, kind), "SELECT * FROM f(TABLE(t), 2)"),
        )
        add(
            f"errors/{k}/scalar_argument_to_table_eval",
            lambda api, spark, kind=kind: _query(api, spark, make_table_rows(api, kind), "SELECT * FROM f(1)"),
        )

    # Definition-time errors (300 and 301 only: ``arrow_udtf`` shares the checks).
    def bad_return_type(api: Api, spark: Any, rt: Any) -> Any:
        @api.F.udtf(returnType=rt)
        class Bad:
            def eval(self, a: Any) -> Any:
                yield (1,)

        return _query(api, spark, Bad, "SELECT * FROM f(1)")

    add("errors/definition/return_type_not_struct", lambda api, spark: bad_return_type(api, spark, "int"))
    add("errors/definition/return_type_array", lambda api, spark: bad_return_type(api, spark, "array<int>"))

    def no_eval(api: Api, spark: Any) -> Any:
        @api.F.udtf(returnType="a int")
        class NoEval:
            def terminate(self) -> Any:
                yield (1,)

        return _query(api, spark, NoEval, "SELECT * FROM f(1)")

    add("errors/definition/no_eval_method", no_eval)

    def default_is_arrow(api: Api, spark: Any) -> Any:
        """A bare ``@udtf`` (no ``useArrow``): the session conf decides, and 4.2.0's default is Arrow (301)."""

        @api.F.udtf(returnType="a string")
        class Bare:
            def eval(self, a: Any) -> Any:
                yield (b"ab",)

        return _query(api, spark, Bare, "SELECT * FROM f(1)")

    add("definition/default_udtf_eval_type", default_is_arrow)

    def no_return_type_no_analyze(api: Api, spark: Any) -> Any:
        @api.F.udtf
        class NoType:
            def eval(self, a: Any) -> Any:
                yield (1,)

        return _query(api, spark, NoType, "SELECT * FROM f(1)")

    add("errors/definition/no_return_type_no_analyze", no_return_type_no_analyze)

    def return_type_and_analyze(api: Api, spark: Any) -> Any:
        AnalyzeResult = api.AnalyzeResult
        T = api.T

        @api.F.udtf(returnType="a int")
        class Both:
            @staticmethod
            def analyze(a: Any) -> Any:
                return AnalyzeResult(schema=T.StructType().add("a", T.IntegerType()))

            def eval(self, a: Any) -> Any:
                yield (1,)

        return _query(api, spark, Both, "SELECT * FROM f(1)")

    add("errors/definition/return_type_and_analyze", return_type_and_analyze)

    return cases


CASES = _cases()


def cell(value: Any) -> Any:
    """A result cell as JSON: NaN and infinities as strings, containers recursively, types JSON lacks (date,
    timestamp, decimal, bytes, Row) as their ``repr``."""
    if isinstance(value, str):
        # PySpark 4.2.0's 300 puts the JVM's ``toString`` of a value it cannot convert into a string column
        # (``[B@7a6ed6b4``, ``java.util.GregorianCalendar[time=?...]``): keep the class, drop the identity.
        value = re.sub(r"\[B@[0-9a-f]+", "[B@HASH", value)
        return re.sub(r"^(java\.util\.GregorianCalendar)\[.*\]$", r"\1[...]", value, flags=re.S)
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return str(value) if value != value or value in (float("inf"), float("-inf")) else value
    if isinstance(value, list | tuple) and type(value).__name__ != "Row":
        return [cell(v) for v in value]
    if isinstance(value, dict):
        return {str(k): cell(v) for k, v in value.items()}
    return repr(value)
