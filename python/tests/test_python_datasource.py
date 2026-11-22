# SPDX-License-Identifier: Apache-2.0
"""The Python Data Source API (P3-T040): batch read and write through ``spark.dataSource.register``."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from krishiv.errors import (
    AnalysisException,
    PySparkException,
    PySparkNotImplementedError,
    PySparkRuntimeError,
    PySparkTypeError,
    SparkRuntimeException,
)
from krishiv.sql import SparkSession, datasource
from krishiv.sql.datasource import (
    CaseInsensitiveDict,
    DataSource,
    DataSourceArrowWriter,
    DataSourceReader,
    DataSourceWriter,
    EqualTo,
    GreaterThan,
    InputPartition,
    IsNotNull,
    WriterCommitMessage,
)
from krishiv.sql.row import Row

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


class Fixed(DataSource):
    """Three partitions of two rows each: (partition, i)."""

    @classmethod
    def name(cls) -> str:
        return "fixed"

    def schema(self) -> str:
        return "p INT, i INT"

    def reader(self, schema: Any) -> DataSourceReader:
        class R(DataSourceReader):
            def partitions(self) -> list[InputPartition]:
                return [InputPartition(k) for k in range(3)]

            def read(self, partition: InputPartition) -> Any:
                for i in range(2):
                    yield (partition.value, i)

        return R()


def rows(df: Any) -> list[tuple[Any, ...]]:
    return sorted(tuple(r) for r in df.collect())


def test_read_tuples_over_partitions(spark: SparkSession) -> None:
    spark.dataSource.register(Fixed)
    df = spark.read.format("fixed").load()
    assert rows(df) == [(p, i) for p in range(3) for i in range(2)]
    assert [f.name for f in df.schema.fields] == ["p", "i"]


def test_name_is_case_insensitive_and_sql_works(spark: SparkSession) -> None:
    spark.dataSource.register(Fixed)
    df = spark.read.format("FIXED").load()
    assert df.filter("i = 1").count() == 3
    assert df.groupBy("p").count().count() == 3
    df.createOrReplaceTempView("v")
    assert spark.sql("SELECT sum(i) FROM v").collect()[0][0] == 3


def test_options_are_case_insensitive_and_path_is_an_option(spark: SparkSession) -> None:
    seen: dict[str, Any] = {}

    class Opts(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            seen.update(self.options)
            assert self.options["PATH"] == "/some/where"

            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield (1,)

            return R()

    spark.dataSource.register(Opts)
    spark.read.format("Opts").option("Key", "v").load("/some/where").collect()
    assert isinstance(seen, dict)
    assert seen["key"] == "v"
    assert seen["path"] == "/some/where"


def test_case_insensitive_dict() -> None:
    d = CaseInsensitiveDict({"Path": "x"})
    assert d["path"] == "x"
    assert "PATH" in d
    assert d.get("PaTh") == "x"
    d["A"] = 1
    assert list(d) == ["path", "a"]


def test_default_name_is_the_class_name_and_schema_of_user(spark: SparkSession) -> None:
    class Plain(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield (7,)

            return R()

    spark.dataSource.register(Plain)
    assert rows(spark.read.format("Plain").load()) == [(7,)]
    # a user schema wins over the source's
    out = spark.read.format("plain").schema("a BIGINT").load()
    assert out.schema.fields[0].dataType.simpleString() == "bigint"
    assert rows(out) == [(7,)]


def test_read_outputs_rows_lists_and_arrow_batches(spark: SparkSession) -> None:
    class Mixed(DataSource):
        def schema(self) -> str:
            return "a INT, b STRING"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield (1, "x")
                    yield Row(a=2, b="y")
                    yield [3, "z"]

            return R()

    spark.dataSource.register(Mixed)
    assert rows(spark.read.format("Mixed").load()) == [(1, "x"), (2, "y"), (3, "z")]


def test_named_row_is_placed_by_field_name(spark: SparkSession) -> None:
    class Named(DataSource):
        def schema(self) -> str:
            return "a INT, b STRING"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield Row(b="x", a=1)

            return R()

    spark.dataSource.register(Named)
    assert rows(spark.read.format("Named").load()) == [(1, "x")]


def test_named_row_with_other_names_is_refused(spark: SparkSession) -> None:
    class Wrong(DataSource):
        def schema(self) -> str:
            return "a INT, b STRING"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield Row(c=1, d="x")

            return R()

    spark.dataSource.register(Wrong)
    with pytest.raises(PySparkException) as e:
        spark.read.format("Wrong").load().collect()
    assert e.value.getCondition() == "PYTHON_EXCEPTION"
    assert "DATA_SOURCE_RETURN_SCHEMA_MISMATCH" in str(e.value)


def test_a_tuple_then_an_arrow_batch_is_an_invalid_return_type(spark: SparkSession) -> None:
    class Mixed2(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield (1,)
                    yield pa.RecordBatch.from_pydict({"a": pa.array([2], type=pa.int32())})

            return R()

    spark.dataSource.register(Mixed2)
    with pytest.raises(PySparkException) as e:
        spark.read.format("Mixed2").load().collect()
    assert "DATA_SOURCE_INVALID_RETURN_TYPE" in str(e.value)


def test_arrow_batch_of_another_schema_is_refused(spark: SparkSession) -> None:
    class Strict(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield pa.RecordBatch.from_pydict({"a": pa.array([1, 2], type=pa.int64())})

            return R()

    spark.dataSource.register(Strict)
    with pytest.raises(PySparkException) as e:
        spark.read.format("Strict").load().collect()
    assert e.value.getCondition() == "PYTHON_EXCEPTION"
    assert "DATA_SOURCE_RETURN_SCHEMA_MISMATCH" in str(e.value)


def test_arrow_batch_of_the_declared_schema_is_read(spark: SparkSession) -> None:
    class Exact(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield pa.RecordBatch.from_pydict({"a": pa.array([1, 2], type=pa.int32())})

            return R()

    spark.dataSource.register(Exact)
    assert rows(spark.read.format("Exact").load()) == [(1,), (2,)]


def test_empty_iterator_and_missing_partitions_default(spark: SparkSession) -> None:
    class Empty(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    assert partition.value is None  # the default single partition
                    return iter(())

            return R()

    spark.dataSource.register(Empty)
    assert spark.read.format("Empty").load().count() == 0


def test_partitions_are_planned_for_every_execution(spark: SparkSession) -> None:
    calls = {"n": 0, "partitions": 0, "reads": 0}

    class Growing(DataSource):
        def schema(self) -> str:
            return "k INT"

        def reader(self, schema: Any) -> DataSourceReader:
            calls["n"] += 1
            n = calls["n"]

            class R(DataSourceReader):
                def partitions(self) -> list[InputPartition]:
                    calls["partitions"] += 1
                    return [InputPartition(k) for k in range(n)]

                def read(self, partition: InputPartition) -> Any:
                    calls["reads"] += 1
                    yield (partition.value,)

            return R()

    spark.dataSource.register(Growing)
    df = spark.read.format("Growing").load()
    df.explain()  # planning and EXPLAIN read nothing
    assert (calls["n"], calls["partitions"], calls["reads"]) == (0, 0, 0)
    first = df.count()
    second = df.count()
    assert (first, second) == (1, 2)  # a second action sees a source that grew
    # One reader(), one partitions() and one read() per partition, per action (not one more for planning).
    assert (calls["n"], calls["partitions"], calls["reads"]) == (2, 2, 3)


def test_reregistering_a_name_replaces_the_class(spark: SparkSession) -> None:
    spark.dataSource.register(Fixed)

    class Fixed2(Fixed):
        @classmethod
        def name(cls) -> str:
            return "fixed"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield (9, 9)

            return R()

    spark.dataSource.register(Fixed2)
    assert rows(spark.read.format("fixed").load()) == [(9, 9)]


def test_builtin_format_wins_over_a_python_source_of_the_same_name(spark: SparkSession, tmp_path: Path) -> None:
    class Shadow(Fixed):
        @classmethod
        def name(cls) -> str:
            return "parquet"

    spark.dataSource.register(Shadow)
    target = str(tmp_path / "t")
    spark.createDataFrame([(5, 6)], "a int, b int").write.parquet(target)
    assert rows(spark.read.format("parquet").load(target)) == [(5, 6)]


@pytest.mark.parametrize("builtin", ["xml", "jdbc", "noop", "rate", "org.apache.spark.sql.parquet"])
def test_every_spark_registered_short_name_wins_over_a_python_source(spark: SparkSession, builtin: str) -> None:
    class Shadow(Fixed):
        @classmethod
        def name(cls) -> str:
            return builtin

    spark.dataSource.register(Shadow)
    assert datasource.registered(spark, builtin) is None


def test_sessions_do_not_share_registrations(spark: SparkSession) -> None:
    spark.dataSource.register(Fixed)
    other = SparkSession.builder.create()
    with pytest.raises(PySparkException):
        other.read.format("fixed").load().collect()


def test_register_refuses_a_non_data_source(spark: SparkSession) -> None:
    with pytest.raises(PySparkTypeError):
        spark.dataSource.register(int)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_schema_not_implemented_is_the_python_data_source_error(spark: SparkSession) -> None:
    class NoSchema(DataSource):
        pass

    spark.dataSource.register(NoSchema)
    with pytest.raises(AnalysisException) as e:
        spark.read.format("NoSchema").load()
    assert e.value.getCondition() == "PYTHON_DATA_SOURCE_ERROR"
    params = e.value.getMessageParameters()
    assert params is not None
    assert (params["action"], params["type"]) == ("create", "instance")


def test_read_failure_is_the_python_data_source_error(spark: SparkSession) -> None:
    class Boom(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    raise RuntimeError("disk on fire")

            return R()

    spark.dataSource.register(Boom)
    df = spark.read.format("Boom").load()
    with pytest.raises(PySparkException) as e:
        df.collect()
    assert e.value.getCondition() == "PYTHON_EXCEPTION"  # a failed read task, as Spark raises it
    assert "disk on fire" in str(e.value)


def test_read_returning_a_list_is_invalid(spark: SparkSession) -> None:
    class ListRead(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    return [(1,)]

            return R()

    spark.dataSource.register(ListRead)
    with pytest.raises(Exception) as e:  # noqa: PT011
        spark.read.format("ListRead").load().collect()
    assert "DATA_SOURCE_INVALID_RETURN_TYPE" in str(e.value)


def test_wrong_width_and_bad_row_type(spark: SparkSession) -> None:
    def source(out: Any) -> type[DataSource]:
        class S(DataSource):
            @classmethod
            def name(cls) -> str:
                return "s"

            def schema(self) -> str:
                return "a INT, b INT"

            def reader(self, schema: Any) -> DataSourceReader:
                class R(DataSourceReader):
                    def read(self, partition: InputPartition) -> Any:
                        yield out

                return R()

        return S

    spark.dataSource.register(source((1,)))
    with pytest.raises(PySparkException) as e:
        spark.read.format("s").load().collect()
    assert "DATA_SOURCE_RETURN_SCHEMA_MISMATCH" in str(e.value)
    assert "Expected: 2 columns, Found: 1 columns" in str(e.value)
    spark.dataSource.register(source(5))
    with pytest.raises(PySparkException) as e:
        spark.read.format("s").load().collect()
    assert "DATA_SOURCE_INVALID_RETURN_TYPE" in str(e.value)
    assert "Unsupported return type ('int') from Python data source 's'" in str(e.value)


def test_partitions_must_be_a_list_of_input_partitions(spark: SparkSession) -> None:
    def source(parts: Any) -> type[DataSource]:
        class S(DataSource):
            @classmethod
            def name(cls) -> str:
                return "p"

            def schema(self) -> str:
                return "a INT"

            def reader(self, schema: Any) -> DataSourceReader:
                class R(DataSourceReader):
                    def partitions(self) -> Any:
                        return parts

                    def read(self, partition: InputPartition) -> Any:
                        yield (1,)

                return R()

        return S

    for bad in ((InputPartition(1),), [1]):
        spark.dataSource.register(source(bad))
        with pytest.raises(PySparkException) as e:
            spark.read.format("p").load().collect()
        assert isinstance(e.value, AnalysisException)
        assert e.value.getCondition() == "PYTHON_DATA_SOURCE_ERROR"
        params = e.value.getMessageParameters()
        assert params is not None
        assert (params["action"], params["type"]) == ("generate", "read partitions")
        assert "DATA_SOURCE_TYPE_MISMATCH" in params["msg"]


def test_pushfilters_without_the_conf_is_refused(spark: SparkSession) -> None:
    class Plain(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            class NoPush(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield (1,)

            class Push(NoPush):
                def pushFilters(self, filters: list[Any]) -> Any:
                    return filters

            return Push() if self.options.get("push") == "1" else NoPush()

    spark.dataSource.register(Plain)
    assert spark.read.format("Plain").load().count() == 1  # pushFilters not overridden: reads normally
    with pytest.raises(Exception) as e:  # noqa: PT011
        spark.read.format("Plain").option("push", "1").load().collect()
    assert "DATA_SOURCE_PUSHDOWN_DISABLED" in str(e.value)
    spark.conf.set("spark.sql.python.filterPushdown.enabled", "true")
    assert spark.read.format("Plain").option("push", "1").load().filter("a = 1").count() == 1


class Recording(DataSource):
    """A source over `x`, `y` that records what pushFilters saw and takes the filters named in `take`."""

    seen: ClassVar[list[Any]] = []
    calls: ClassVar[list[str]] = []

    @classmethod
    def name(cls) -> str:
        return "recording"

    def schema(self) -> str:
        return "x INT, y INT"

    def reader(self, schema: Any) -> DataSourceReader:
        options = self.options
        Recording.calls.append("reader")

        class R(DataSourceReader):
            def __init__(self) -> None:
                self.taken: list[Any] = []

            def pushFilters(self, filters: list[Any]) -> Any:
                Recording.calls.append("pushFilters")
                Recording.seen.append(list(filters))
                if options.get("mode") == "boom":
                    raise ValueError("boom")
                if options.get("mode") == "extra":
                    return [*filters, EqualTo(("zzz",), 1)]
                if options.get("mode") == "take":
                    self.taken = [f for f in filters if isinstance(f, EqualTo)]
                    return [f for f in filters if not isinstance(f, EqualTo)]
                return filters

            def read(self, partition: InputPartition) -> Any:
                for x in range(1, 5):
                    if all(not (isinstance(f, EqualTo) and f.attribute == ("x",)) or x == f.value for f in self.taken):
                        yield (x, x * 10)

        return R()


@pytest.fixture
def pushdown(spark: SparkSession) -> SparkSession:
    Recording.seen.clear()
    Recording.calls.clear()
    spark.conf.set("spark.sql.python.filterPushdown.enabled", "true")
    spark.dataSource.register(Recording)
    return spark


def test_pushfilters_receives_the_serializable_conjuncts(pushdown: SparkSession) -> None:
    df = pushdown.read.format("recording").load().filter("x = 2 AND y > 5")
    assert [r.x for r in df.collect()] == [2]
    assert Recording.seen == [[IsNotNull(("x",)), IsNotNull(("y",)), EqualTo(("x",), 2), GreaterThan(("y",), 5)]]
    assert Recording.calls == ["reader", "pushFilters"]


def test_no_filter_never_calls_pushfilters(pushdown: SparkSession) -> None:
    assert pushdown.read.format("recording").load().count() == 4
    assert Recording.calls == ["reader"]


def test_a_taken_filter_is_applied_by_the_source_only(pushdown: SparkSession) -> None:
    df = pushdown.read.format("recording").option("mode", "take").load().filter("x = 3")
    assert [r.x for r in df.collect()] == [3]


def test_unhandled_filters_stay_with_the_engine(pushdown: SparkSession) -> None:
    df = pushdown.read.format("recording").load().filter("x = 3 AND (y = 30 OR y = 99)")
    assert [r.x for r in df.collect()] == [3]


def test_residual_division_by_a_filtered_zero(pushdown: SparkSession) -> None:
    df = pushdown.read.format("recording").load().filter("x > 0 AND 10 / x > 1")
    assert sorted(r.x for r in df.collect()) == [1, 2, 3, 4]
    assert Recording.seen[0] == [IsNotNull(("x",)), GreaterThan(("x",), 0)]


def test_pushfilters_failure_is_the_python_data_source_error(pushdown: SparkSession) -> None:
    df = pushdown.read.format("recording").option("mode", "boom").load().filter("x = 1")
    with pytest.raises(Exception) as e:  # noqa: PT011
        df.collect()
    assert "PYTHON_DATA_SOURCE_ERROR" in str(e.value)
    assert "boom" in str(e.value)


def test_extraneous_filters_are_refused(pushdown: SparkSession) -> None:
    df = pushdown.read.format("recording").option("mode", "extra").load().filter("x = 1")
    with pytest.raises(Exception) as e:  # noqa: PT011
        df.collect()
    assert "DATA_SOURCE_EXTRANEOUS_FILTERS" in str(e.value)


def test_filter_classes_are_dataclasses() -> None:
    assert EqualTo(("a", "b"), 3) == EqualTo(("a", "b"), 3)


def test_stream_reader_is_p4() -> None:
    with pytest.raises(PySparkNotImplementedError):
        Fixed({}).streamReader(None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


# ---- writing ---------------------------------------------------------------------------------------


class Log:
    events: ClassVar[list[tuple[str, Any]]] = []


class Msg(WriterCommitMessage):
    def __init__(self, n: int) -> None:
        self.n = n


class Sink(DataSource):
    @classmethod
    def name(cls) -> str:
        return "sink"

    def schema(self) -> str:
        return "a INT"

    def writer(self, schema: Any, overwrite: bool) -> DataSourceWriter:
        Log.events.append(("writer", overwrite))
        fail = self.options.get("fail") == "1"

        class W(DataSourceWriter):
            def write(self, iterator: Any) -> WriterCommitMessage:
                got = [r.a for r in iterator]
                Log.events.append(("write", sorted(got)))
                if fail:
                    raise RuntimeError("write failed")
                return Msg(len(got))

            def commit(self, messages: list[Any]) -> None:
                Log.events.append(("commit", [m.n for m in messages]))

            def abort(self, messages: list[Any]) -> None:
                Log.events.append(("abort", messages))

        return W()


def test_write_append_and_overwrite(spark: SparkSession) -> None:
    Log.events = []
    spark.dataSource.register(Sink)
    df = spark.range(3).selectExpr("cast(id as int) as a")
    df.write.format("sink").mode("append").save()
    df.write.format("sink").mode("overwrite").save()
    assert Log.events == [
        ("writer", False),
        ("write", [0, 1, 2]),
        ("commit", [3]),
        ("writer", True),
        ("write", [0, 1, 2]),
        ("commit", [3]),
    ]


def test_write_failure_calls_abort_and_raises(spark: SparkSession) -> None:
    Log.events = []
    spark.dataSource.register(Sink)
    df = spark.range(2).selectExpr("cast(id as int) as a")
    with pytest.raises(SparkRuntimeException) as e:
        df.write.format("sink").option("fail", "1").mode("append").save()
    assert e.value.getCondition() == "PYTHON_EXCEPTION"
    assert "write failed" in str(e.value)
    assert ("abort", [None]) in Log.events
    assert not any(k == "commit" for k, _ in Log.events)


@pytest.mark.parametrize("mode", ["error", "errorifexists", "ignore"])
def test_write_modes_error_and_ignore_are_refused(spark: SparkSession, mode: str) -> None:
    spark.dataSource.register(Sink)
    df = spark.range(1).selectExpr("cast(id as int) as a")
    with pytest.raises(AnalysisException) as e:
        df.write.format("sink").mode(mode).save()
    assert e.value.getCondition() == "UNSUPPORTED_DATA_SOURCE_SAVE_MODE"
    params = e.value.getMessageParameters()
    assert params is not None
    assert params["createMode"] == ("Ignore" if mode == "ignore" else "ErrorIfExists")


def test_arrow_writer_gets_record_batches(spark: SparkSession) -> None:
    got: list[Any] = []

    class ArrowSink(DataSource):
        def writer(self, schema: Any, overwrite: bool) -> DataSourceWriter:
            class W(DataSourceArrowWriter):
                def write(self, iterator: Any) -> WriterCommitMessage:
                    got.extend(iterator)
                    return Msg(0)

            return W()

    spark.dataSource.register(ArrowSink)
    spark.range(4).selectExpr("cast(id as int) as a").write.format("ArrowSink").mode("append").save()
    assert sum(b.num_rows for b in got) == 4
    assert all(isinstance(b, pa.RecordBatch) for b in got)


def test_round_trip_through_a_python_source_matches_parquet(spark: SparkSession, tmp_path: Any) -> None:
    """The same rows read through a Python source and as Parquet answer one query alike."""
    path = str(tmp_path / "t.parquet")
    data = [(i, f"s{i % 4}", i * 1.5) for i in range(50)]
    spark.createDataFrame(data, "a INT, b STRING, c DOUBLE").write.parquet(path)

    class Same(DataSource):
        def schema(self) -> str:
            return "a INT, b STRING, c DOUBLE"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def partitions(self) -> list[InputPartition]:
                    return [InputPartition(range(0, 25)), InputPartition(range(25, 50))]

                def read(self, partition: InputPartition) -> Any:
                    for i in partition.value:
                        yield data[i]

            return R()

    spark.dataSource.register(Same)
    query = "SELECT b, count(*) AS n, sum(c) AS s FROM {t} WHERE a > 3 GROUP BY b ORDER BY b"
    spark.read.parquet(path).createOrReplaceTempView("pq")
    spark.read.format("Same").load().createOrReplaceTempView("py")
    assert spark.sql(query.format(t="py")).collect() == spark.sql(query.format(t="pq")).collect()


def test_writer_gets_nested_rows_and_duplicate_columns(spark: SparkSession) -> None:
    seen: list[Any] = []

    class Capture(DataSource):
        def writer(self, schema: Any, overwrite: bool) -> DataSourceWriter:
            class W(DataSourceWriter):
                def write(self, iterator: Any) -> WriterCommitMessage:
                    seen.extend(iterator)
                    return WriterCommitMessage()

            return W()

    spark.dataSource.register(Capture)
    df = spark.sql("select 1 as k, 2 as k, named_struct('x', 3, 'y', 'z') as s")
    df.write.format("Capture").mode("append").save()
    assert len(seen) == 1
    assert tuple(seen[0])[:2] == (1, 2)
    assert isinstance(seen[0][2], Row)
    assert tuple(seen[0][2]) == (3, "z")


def test_engine_error_in_a_write_reaches_the_caller_unchanged(spark: SparkSession) -> None:
    Log.events = []
    spark.dataSource.register(Sink)
    df = spark.sql("select 1 div 0 as a")
    with pytest.raises(PySparkException) as e:
        df.write.format("sink").mode("append").save()
    assert e.value.getCondition() == "DIVIDE_BY_ZERO"
    assert ("abort", [None]) in Log.events


def test_a_source_over_the_byte_limit_fails_before_it_is_read_whole(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from krishiv.sql import datasource

    monkeypatch.setattr(datasource, "MAX_SOURCE_BYTES", 1000)
    pulled: list[int] = []

    class Big(DataSource):
        def schema(self) -> str:
            return "a BIGINT"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    for i in range(1000):
                        pulled.append(i)
                        yield pa.RecordBatch.from_pydict({"a": pa.array([i] * 100, type=pa.int64())})

            return R()

    spark.dataSource.register(Big)
    with pytest.raises(PySparkException) as e:
        spark.read.format("Big").load().collect()
    assert e.value.getCondition() == "KRISHIV_PYTHON_DATA_SOURCE_LIMIT"
    params = e.value.getMessageParameters()
    assert params is not None
    assert params["source"] == "Big"
    assert len(pulled) < 10  # stopped at the first batches over the limit


def test_no_active_session_message_is_sparks() -> None:
    error = PySparkRuntimeError(errorClass="NO_ACTIVE_SESSION", messageParameters={})
    assert error.getCondition() == "NO_ACTIVE_SESSION"
    text = "No active Spark session found. Please create a new Spark session before running the code."
    assert error.getMessage() == f"[NO_ACTIVE_SESSION] {text}"


def test_the_last_spelling_of_a_repeated_option_wins(spark: SparkSession) -> None:
    seen: dict[str, Any] = {}

    class Opts(DataSource):
        def schema(self) -> str:
            return "a INT"

        def reader(self, schema: Any) -> DataSourceReader:
            seen.update(self.options)

            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield (1,)

            return R()

    spark.dataSource.register(Opts)
    spark.read.format("Opts").option("Path", "first").option("path", "second").option("Path", "third").load().collect()
    assert seen["path"] == "third"


def test_a_source_without_columns_keeps_its_rows(spark: SparkSession) -> None:
    class NoColumns(DataSource):
        def schema(self) -> Any:
            from krishiv.sql.types import StructType

            return StructType([])

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def read(self, partition: InputPartition) -> Any:
                    yield ()
                    yield ()

            return R()

    spark.dataSource.register(NoColumns)
    assert spark.read.format("NoColumns").load().count() == 2


def test_a_bucketed_save_through_a_python_sink_is_refused(spark: SparkSession) -> None:
    Log.events = []
    spark.dataSource.register(Sink)
    df = spark.range(3).selectExpr("cast(id as int) as a")
    with pytest.raises(AnalysisException) as bucketed:
        df.write.format("sink").mode("append").bucketBy(2, "a").save()
    assert bucketed.value.getCondition() == "_LEGACY_ERROR_TEMP_1312"
    assert "'save' does not support bucketBy right now." in bucketed.value.getMessage()
    with pytest.raises(AnalysisException) as both:
        df.write.format("sink").mode("append").bucketBy(2, "a").sortBy("a").save()
    assert both.value.getCondition() == "_LEGACY_ERROR_TEMP_1313"
    assert "does not support bucketBy and sortBy right now." in both.value.getMessage()
    with pytest.raises(AnalysisException) as sort_only:
        df.write.format("sink").mode("append").sortBy("a").save()
    assert sort_only.value.getCondition() == "SORT_BY_WITHOUT_BUCKETING"
    assert "sortBy must be used together with bucketBy." in sort_only.value.getMessage()
    # Spark asserts "not bucketed" before anything about partitioning.
    with pytest.raises(AnalysisException) as mixed:
        df.write.format("sink").mode("append").partitionBy("a").bucketBy(2, "a").save()
    assert mixed.value.getCondition() == "_LEGACY_ERROR_TEMP_1312"
    assert Log.events == []


def test_a_partitioned_save_through_a_python_sink_is_refused_before_the_writer_runs(spark: SparkSession) -> None:
    from krishiv.errors import UnsupportedOperationException

    Log.events = []
    spark.dataSource.register(Sink)
    df = spark.range(3).selectExpr("cast(id as int) as a")
    with pytest.raises(UnsupportedOperationException):
        df.write.format("sink").mode("append").partitionBy("a").save()
    with pytest.raises(UnsupportedOperationException):
        df.write.format("sink").mode("append").save(partitionBy="nonexistent")
    assert Log.events == []


QUERIES = Path(__file__).parents[2] / "tests" / "connect" / "python-source-queries.sql"


def _equivalence_rows() -> list[tuple[Any, ...]]:
    return [(i, f"n{i}", i % 4, None if i % 7 == 0 else i * 3, date(2026, 1, 1 + i % 5)) for i in range(40)]


def _canonical(df: Any) -> list[str]:
    return sorted(repr(tuple(r)) for r in df.collect())


def test_python_source_matches_parquet(spark: SparkSession, tmp_path: Path) -> None:
    """The 20 statements of ``tests/connect/python-source-queries.sql`` agree on a Python source and Parquet."""
    pq = pytest.importorskip("pyarrow.parquet")
    data = _equivalence_rows()

    class Equiv(DataSource):
        @classmethod
        def name(cls) -> str:
            return "equiv"

        def schema(self) -> str:
            return "id INT, name STRING, grp INT, v INT, d DATE"

        def reader(self, schema: Any) -> DataSourceReader:
            class R(DataSourceReader):
                def partitions(self) -> list[InputPartition]:
                    return [InputPartition(k) for k in range(4)]

                def read(self, partition: InputPartition) -> Any:
                    yield from data[partition.value :: 4]

            return R()

    spark.dataSource.register(Equiv)
    spark.read.format("equiv").load().createOrReplaceTempView("src")
    table = pa.table(
        {
            "id": pa.array([r[0] for r in data], pa.int32()),
            "name": [r[1] for r in data],
            "grp": pa.array([r[2] for r in data], pa.int32()),
            "v": pa.array([r[3] for r in data], pa.int32()),
            "d": pa.array([r[4] for r in data], pa.date32()),
        }
    )
    pq.write_table(table, tmp_path / "t.parquet")
    spark.read.parquet(str(tmp_path / "t.parquet")).createOrReplaceTempView("pq")
    statements = [line for line in QUERIES.read_text().splitlines() if line.strip() and not line.startswith("--")]
    assert len(statements) == 20
    for statement in statements:
        got = _canonical(spark.sql(statement.format(t="src")))
        want = _canonical(spark.sql(statement.format(t="pq")))
        assert got == want, statement


class Stamps(DataSource):
    """A source over a TIMESTAMP and a TIMESTAMP_NTZ column that records the filter values."""

    seen: ClassVar[list[Any]] = []

    @classmethod
    def name(cls) -> str:
        return "stamps"

    def schema(self) -> str:
        return "t TIMESTAMP, n TIMESTAMP_NTZ"

    def reader(self, schema: Any) -> DataSourceReader:
        class R(DataSourceReader):
            def pushFilters(self, filters: list[Any]) -> Any:
                Stamps.seen.extend(filters)
                return filters

            def read(self, partition: InputPartition) -> Any:
                return iter([])

        return R()


def test_timestamp_values_are_aware_utc_and_ntz_values_are_naive(pushdown: SparkSession) -> None:
    """Spark 4.3.0 (`tests/fixtures/spark-4.3.0/python-datasource-pushdown/timestamp_ltz.json`)."""
    Stamps.seen.clear()
    pushdown.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
    pushdown.dataSource.register(Stamps)
    pushdown.read.format("stamps").load().filter(
        "t = timestamp '2020-01-01 00:00:00' AND n = timestamp_ntz '2020-01-01 00:00:00'"
    ).count()
    values = {f.attribute: f.value for f in Stamps.seen if isinstance(f, EqualTo)}
    assert values[("t",)] == datetime(2020, 1, 1, 8, 0, tzinfo=UTC)
    assert values[("t",)].tzinfo is not None
    assert values[("n",)] == datetime(2020, 1, 1, 0, 0)
    assert values[("n",)].tzinfo is None


_ORACLE = Path(__file__).resolve().parents[2] / "tests/fixtures/spark-4.3.0/python-datasource-pushdown/classic.json"
_FILTER_CASES = json.loads(_ORACLE.read_text())["filter_cases"]


class Probe(DataSource):
    """A source of the schema in option `type` (column `x`) that records what pushFilters saw, takes nothing."""

    seen: ClassVar[list[str]] = []

    @classmethod
    def name(cls) -> str:
        return "probe"

    def schema(self) -> str:
        return f"x {self.options['type']}"

    def reader(self, schema: Any) -> DataSourceReader:
        class R(DataSourceReader):
            def pushFilters(self, filters: list[Any]) -> Any:
                Probe.seen.append(repr(list(filters)))
                return filters

            def read(self, partition: InputPartition) -> Any:
                return iter(())

        return R()


_NO_SCHEMA = ("struct<", "array<", "interval")


@pytest.mark.parametrize("case", _FILTER_CASES, ids=lambda c: c["sql_filter"])
def test_filters_reach_the_source_as_spark_4_3_sends_them(pushdown: SparkSession, case: dict[str, Any]) -> None:
    if "collate" in case["sql_filter"]:
        pytest.xfail("collated comparison is not supported yet")
    if "3 / 2" in case["sql_filter"]:
        pytest.xfail("the optimizer does not fold a literal division, so the value never reaches the source")
    if any(case["sql_type"].startswith(t) or t in case["sql_type"] for t in _NO_SCHEMA):
        pytest.xfail("a Python source with a struct, array or interval column is not readable yet (createDataFrame)")
    Probe.seen.clear()
    pushdown.dataSource.register(Probe)
    df = pushdown.read.format("probe").option("type", case["sql_type"]).load()
    df.filter(case["sql_filter"]).count()
    assert Probe.seen == case["filters_repr"]
