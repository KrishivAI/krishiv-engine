# SPDX-License-Identifier: Apache-2.0
"""The Python Data Source helpers called on the test thread.

The engine calls a source's ``read`` from a native worker thread, which coverage.py does not trace, so these
tests call the planning and conversion helpers directly to keep their lines measured. The behaviour through
``spark.read.format`` is in ``test_python_datasource.py``."""

from __future__ import annotations

import gc
from typing import Any

import pytest

from krishiv.errors import (
    AnalysisException,
    PySparkAssertionError,
    PySparkRuntimeError,
    PySparkTypeError,
)
from krishiv.sql import SparkSession
from krishiv.sql import datasource as ds
from krishiv.sql.datasource import (
    CaseInsensitiveDict,
    DataSource,
    DataSourceReader,
    InputPartition,
)
from krishiv.sql.row import Row
from krishiv.sql.session import _pa_schema
from krishiv.sql.types import StructType

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def _schema(ddl: str) -> StructType:
    return ds._struct_of(ddl)


def _source(reader: DataSourceReader, ddl: str = "a INT, b STRING") -> DataSource:
    class Src(DataSource):
        def schema(self) -> str:
            return ddl

        def reader(self, schema: Any) -> DataSourceReader:
            return reader

    return Src(options={})


def _read(spark: SparkSession, reader: DataSourceReader, ddl: str = "a INT, b STRING") -> list[tuple[Any, ...]]:
    _, read = ds._plan_and_read(spark, _source(reader, ddl), _schema(ddl), "t")
    table = read()
    return [tuple(r.values()) for r in table.to_pylist()]


class Rows(DataSourceReader):
    def __init__(self, *items: Any, partitions: Any = None) -> None:
        self.items = items
        self._partitions = partitions

    def partitions(self) -> Any:
        if self._partitions is None:
            raise NotImplementedError
        return self._partitions

    def read(self, partition: InputPartition) -> Any:
        return iter(self.items)


def test_case_insensitive_dict_methods() -> None:
    d = CaseInsensitiveDict({"A": 1, "b": 2})
    del d["a"]
    assert "A" not in d
    assert d.pop("B") == 2
    assert d.pop("missing", 9) == 9
    assert d.setdefault("K", 3) == 3
    d.update({"X": 1}, Y=2)
    assert sorted(d) == ["k", "x", "y"]
    assert isinstance(d.copy(), CaseInsensitiveDict)


def test_input_partition_repr_and_abstract_defaults(spark: SparkSession) -> None:
    assert "7" in repr(InputPartition(7))

    class Bare(DataSource):
        pass

    bare = Bare(options={})
    assert bare.name() == "Bare"
    for call in (bare.schema, lambda: bare.reader(_schema("a INT")), lambda: bare.writer(_schema("a INT"), False)):
        with pytest.raises(NotImplementedError):
            call()
    for stream in (bare.streamReader, lambda s: bare.streamWriter(s, False)):
        with pytest.raises(NotImplementedError):
            stream(_schema("a INT"))
    assert [p.value for p in DataSourceReader.partitions(Rows())] == [None]
    assert list(DataSourceReader.pushFilters(Rows(), [])) == []


def test_source_and_mismatch_errors() -> None:
    err = ds._source_error("create", "instance", "boom")
    assert isinstance(err, AnalysisException)
    with pytest.raises(AnalysisException):
        ds._guard("create", "instance", lambda: 1 / 0)
    with pytest.raises(PySparkTypeError):
        ds._struct_of(5)  # ty: ignore[invalid-argument-type]
    assert ds._struct_of("INT").fieldNames() == ["value"]


def test_tuples_lists_rows_and_arrow(spark: SparkSession) -> None:
    assert _read(spark, Rows((1, "x"), [2, "y"], Row(b="z", a=3))) == [(1, "x"), (2, "y"), (3, "z")]
    schema = _schema("a INT, b STRING")
    batch = pa.RecordBatch.from_pydict({"a": [4], "b": ["w"]}, schema=_pa_schema(schema, pa))
    assert _read(spark, Rows(batch, batch)) == [(4, "w"), (4, "w")]
    assert _read(spark, Rows()) == []


def test_partitions_planning(spark: SparkSession) -> None:
    reader = Rows((1, "x"), partitions=[InputPartition(0), InputPartition(1)])
    assert _read(spark, reader) == [(1, "x"), (1, "x")]
    assert _read(spark, Rows((1, "x"), partitions=[])) == [(1, "x")]
    with pytest.raises(AnalysisException):
        ds._plan_and_read(spark, _source(Rows(partitions=(InputPartition(0),))), _schema("a INT"), "t")
    with pytest.raises(AnalysisException):
        ds._plan_and_read(spark, _source(Rows(partitions=[1])), _schema("a INT"), "t")


@pytest.mark.parametrize(
    "items",
    [
        ((1,),),  # width
        ((1, "x"), 5),  # not a row
        (Row(a=1, c="x"),),  # named row with other names
    ],
)
def test_bad_rows_are_refused(spark: SparkSession, items: tuple[Any, ...]) -> None:
    with pytest.raises(PySparkRuntimeError):
        _read(spark, Rows(*items))


def test_arrow_batch_mismatches(spark: SparkSession) -> None:
    for batch in (
        pa.RecordBatch.from_pydict({"a": [1]}),
        pa.RecordBatch.from_pydict({"a": [1], "c": ["x"]}),
        pa.RecordBatch.from_pydict({"a": pa.array([1], pa.int64()), "b": ["x"]}),
    ):
        with pytest.raises(PySparkRuntimeError):
            _read(spark, Rows(batch))


def test_reader_shape_errors(spark: SparkSession) -> None:
    schema = _schema("a INT")

    class NotAReader(DataSource):
        def reader(self, schema: Any) -> Any:
            return object()

    with pytest.raises(AnalysisException):
        ds._plan_and_read(spark, NotAReader(options={}), schema, "t")

    class Filtering(Rows):
        def pushFilters(self, filters: Any) -> Any:
            return filters

    with pytest.raises(AnalysisException):
        ds._plan_and_read(spark, _source(Filtering()), schema, "t")

    class ListRead(Rows):
        def read(self, partition: Any) -> Any:
            return [(1,)]

    with pytest.raises(PySparkRuntimeError):
        ds._plan_and_read(spark, _source(ListRead()), schema, "t")[1]()


def test_pushdown_conf_is_read_and_tolerates_a_broken_session() -> None:
    class Broken:
        @property
        def conf(self) -> Any:
            raise RuntimeError("no conf")

    assert ds._pushdown_enabled(Broken()) is False  # ty: ignore[invalid-argument-type]


def test_byte_limit(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ds, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(PySparkRuntimeError):
        _read(spark, Rows((1, "x")))


def test_load_read_callback_and_dropped_session(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    real = spark._native

    class Native:
        def register_python_source(self, name: str, empty: Any, read: Any, plan: Any) -> Any:
            seen["read"] = read
            return real.sql("SELECT 1 AS a")

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

    # main keeps the engine session in `_base_native` (`_native` is a read-only property there).
    monkeypatch.setattr(spark, "_base_native" if hasattr(spark, "_base_native") else "_native", Native())

    class Src(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            return Rows((1,))

    entry = ds._Registered(Src, "src", 1)
    ds.load(spark, entry, {}, None)
    assert seen["read"]().to_pylist() == [{"a": 1}]
    ds.load(spark, entry, {}, "a INT")
    assert ds._registry(spark) is ds._registry(spark)
    assert ds.registered(spark, "SRC") is None

    class Holder:
        _native = Native()

    holder = Holder()
    ds.load(holder, entry, {}, None)  # ty: ignore[invalid-argument-type]
    del holder
    gc.collect()
    with pytest.raises(PySparkRuntimeError):
        seen["read"]()


def test_register_refuses_a_bad_name(spark: SparkSession) -> None:
    class Nameless(DataSource):
        @classmethod
        def name(cls) -> str:
            return ""

    with pytest.raises(PySparkRuntimeError):
        ds.DataSourceRegistration(spark).register(Nameless)


def test_save_refuses_unknown_mode(spark: SparkSession) -> None:
    class Src(DataSource):
        pass

    entry = ds._Registered(Src, "src", 1)
    with pytest.raises(PySparkRuntimeError):
        ds.save(spark, entry, spark.range(1), "bogus", {})
    with pytest.raises(PySparkAssertionError):
        ds._writer_of(type("S", (DataSource,), {"writer": lambda self, s, o: 1})(options={}), _schema("a INT"), False)
    gc.collect()


def test_reader_and_writer_pass_the_path_and_struct_schemas(spark: SparkSession) -> None:
    seen: dict[str, Any] = {}

    class Src(DataSource):
        @classmethod
        def name(cls) -> str:
            return "pathsrc"

        def __init__(self, options: Any) -> None:
            super().__init__(options)
            seen["options"] = dict(options)

        def schema(self) -> StructType:
            return _schema("a INT")

        def reader(self, schema: Any) -> DataSourceReader:
            return Rows((1,))

    assert ds._struct_of(_schema("a INT")).fieldNames() == ["a"]
    spark.dataSource.register(Src)
    spark.read.format("pathsrc").load(["/one"])
    assert seen["options"]["path"] == "/one"
    spark.read.format("pathsrc").load("/two")
    assert seen["options"]["path"] == "/two"
    with pytest.raises(PySparkTypeError):
        spark.read.format("pathsrc").load(["/a", "/b"])

    class Sink(DataSource):
        @classmethod
        def name(cls) -> str:
            return "pathsink"

        def __init__(self, options: Any) -> None:
            super().__init__(options)
            seen["sink"] = dict(options)

        def writer(self, schema: Any, overwrite: bool) -> Any:
            class W(ds.DataSourceWriter):
                def write(self, iterator: Any) -> Any:
                    return ds.WriterCommitMessage()

            return W()

    spark.dataSource.register(Sink)
    spark.range(1).write.format("pathsink").mode("append").save("/out")
    assert seen["sink"]["path"] == "/out"
    # a StructType given to ``schema()`` reaches the source as the user schema
    assert spark.read.format("pathsrc").schema(_schema("a INT")).load("/three").collect() == [Row(a=1)]


def test_push_filters_marks_what_the_reader_took(spark: SparkSession) -> None:
    schema = _schema("a INT, b STRING")
    eq = ds.EqualTo(("a",), 1)
    gt = ds.GreaterThan(("a",), 0)

    class Takes(Rows):
        def pushFilters(self, filters: Any) -> Any:
            return [filters[1]]  # keeps the second, takes the first

    pushed, _ = ds._plan_and_read(spark, _source(Takes()), schema, "t", [eq, gt], True)
    assert pushed == [True, False]

    class TakesAll(Rows):
        def pushFilters(self, filters: Any) -> Any:
            return None

    pushed, _ = ds._plan_and_read(spark, _source(TakesAll()), schema, "t", [eq], True)
    assert pushed == [True]

    class Invents(Rows):
        def pushFilters(self, filters: Any) -> Any:
            return [ds.EqualTo(("a",), 1)]  # equal, but not one that was offered

    with pytest.raises(AnalysisException):
        ds._plan_and_read(spark, _source(Invents()), schema, "t", [eq], True)

    # no filters offered: pushFilters is not called
    pushed, _ = ds._plan_and_read(spark, _source(Invents()), schema, "t", [], True)
    assert pushed == []


def test_filter_of_builds_each_kind() -> None:
    assert ds._filter_of(("EqualTo", ("a",), 1)) == ds.EqualTo(("a",), 1)
    assert ds._filter_of(("IsNull", ("a",), None)) == ds.IsNull(("a",))
    assert ds._filter_of(("IsNotNull", ("a",), None)) == ds.IsNotNull(("a",))
    assert ds._filter_of(("Not", None, ("StringContains", ("b",), "x"))) == ds.Not(ds.StringContains(("b",), "x"))


def test_plan_callback_offers_filters_only_when_pushdown_is_on(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real = spark._native

    class Native:
        def register_python_source(self, name: str, empty: Any, read: Any, plan: Any) -> Any:
            seen["plan"] = plan
            return real.sql("SELECT 1 AS a")

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

    monkeypatch.setattr(spark, "_base_native" if hasattr(spark, "_base_native") else "_native", Native())

    class Taking(Rows):
        def pushFilters(self, filters: Any) -> Any:
            return []

    class Src(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            return Taking((1,))

    ds.load(spark, ds._Registered(Src, "src", 1), {}, None)
    specs = [("EqualTo", ("a",), 1)]
    monkeypatch.setattr(ds, "_pushdown_enabled", lambda _s: False)
    assert seen["plan"](specs)[0] == [False]
    monkeypatch.setattr(ds, "_pushdown_enabled", lambda _s: True)
    assert seen["plan"](specs)[0] == [True]
