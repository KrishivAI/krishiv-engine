# SPDX-License-Identifier: Apache-2.0
"""The Arrow PyCapsule bridge (``krishiv._native.ArrowStream``, P2-T091b).

Data crosses the bridge without copies: every test that moves data compares
buffer addresses before and after.
"""

from __future__ import annotations

from typing import Any

import pytest
from krishiv._native import ArrowStream

pa = pytest.importorskip("pyarrow")


def table() -> Any:
    return pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "name": pa.array(["a", None, "ccc"], pa.string()),
            "score": pa.array([0.5, 1.5, None], pa.float64()),
        }
    )


def addresses(data: Any) -> list[int]:
    """Buffer addresses of every column of a pyarrow table or batch."""
    columns = data.columns
    chunks = [c for col in columns for c in (col.chunks if hasattr(col, "chunks") else [col])]
    return [b.address for chunk in chunks for b in chunk.buffers() if b is not None]


def test_pyarrow_round_trip_is_zero_copy() -> None:
    source = table()
    got = pa.table(ArrowStream.from_arrow(source))
    assert got.equals(source)
    assert addresses(got) == addresses(source)


def test_record_batch_reader_round_trip() -> None:
    source = table()
    reader = pa.RecordBatchReader.from_batches(source.schema, source.to_batches(max_chunksize=2))
    got = pa.RecordBatchReader.from_stream(ArrowStream.from_arrow(reader)).read_all()
    assert got.equals(source)
    assert got.num_rows == 3


def test_single_batch_via_arrow_c_array() -> None:
    batch = table().to_batches()[0]
    stream = ArrowStream.from_arrow(batch)
    got = pa.table(stream)
    assert got.to_batches()[0].equals(batch)
    assert addresses(got) == addresses(batch)


def test_polars_consumes_zero_copy() -> None:
    pl = pytest.importorskip("polars")
    source = pa.table({"id": pa.array([1, 2, 3], pa.int64()), "x": pa.array([1.0, 2.0, 3.0])})
    frame = pl.DataFrame(ArrowStream.from_arrow(source))
    assert frame.to_dict(as_series=False) == {"id": [1, 2, 3], "x": [1.0, 2.0, 3.0]}
    # Polars keeps primitive buffers as they arrive; its own export shows them.
    back = pa.table(frame)
    assert addresses(back)[-1] == addresses(source)[-1]
    assert addresses(back)[1] == addresses(source)[1]


def test_polars_produces() -> None:
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    got = pa.table(ArrowStream.from_arrow(frame))
    assert got.column("id").to_pylist() == [1, 2, 3]
    assert addresses(got) == addresses(pa.table(frame))


def test_duckdb_consumes_and_produces() -> None:
    duckdb = pytest.importorskip("duckdb")
    source = table()
    con = duckdb.connect()
    relation = con.from_arrow(ArrowStream.from_arrow(source))
    rows = relation.project("id, name").order("id").fetchall()
    assert rows == [(1, "a"), (2, None), (3, "ccc")]
    cursor = con.execute("SELECT 41 + 1 AS answer")
    # `to_arrow_reader` replaced `fetch_record_batch` in DuckDB 1.4; musllinux
    # has wheels only up to 1.3.
    reader = getattr(cursor, "to_arrow_reader", None) or cursor.fetch_record_batch
    result = reader()
    got = pa.table(ArrowStream.from_arrow(result))
    assert got.to_pydict() == {"answer": [42]}


def test_duckdb_reads_buffers_in_place() -> None:
    duckdb = pytest.importorskip("duckdb")
    source = pa.table({"id": pa.array(range(100_000), pa.int64())})
    # DuckDB scans the stream without materialising a copy first; the sum
    # proves it read every row of the original buffers.
    con = duckdb.connect()  # kept alive while the relation is read
    relation = con.from_arrow(ArrowStream.from_arrow(source))
    total = relation.aggregate("sum(id)").fetchone()
    assert total == (sum(range(100_000)),)


def test_stream_is_single_use() -> None:
    stream = ArrowStream.from_arrow(table())
    assert not stream.consumed
    pa.table(stream)
    assert stream.consumed
    with pytest.raises(ValueError, match="already consumed"):
        pa.table(stream)


def test_schema_does_not_consume() -> None:
    stream = ArrowStream.from_arrow(table())
    assert pa.schema(stream) == table().schema
    assert not stream.consumed
    assert repr(stream) == "ArrowStream(columns=[id, name, score])"


def test_requested_schema_with_equal_types_renames() -> None:
    stream = ArrowStream.from_arrow(table())
    wanted = pa.schema([("a", pa.int64()), ("b", pa.string()), ("c", pa.float64())])
    capsule = wanted.__arrow_c_schema__()
    got = pa.RecordBatchReader._import_from_c_capsule(stream.__arrow_c_stream__(capsule)).read_all()
    assert got.schema.names == ["a", "b", "c"]
    # The caller's schema capsule was only read: it is still valid.
    assert pa.Schema._import_from_c_capsule(capsule) == wanted


def test_requested_schema_needing_a_cast_keeps_own_schema() -> None:
    stream = ArrowStream.from_arrow(table())
    wanted = pa.schema([("id", pa.int32()), ("name", pa.string()), ("score", pa.float64())])
    capsule = stream.__arrow_c_stream__(wanted.__arrow_c_schema__())
    got = pa.RecordBatchReader._import_from_c_capsule(capsule).read_all()
    assert got.schema == table().schema


def test_requested_schema_with_other_field_count_raises() -> None:
    stream = ArrowStream.from_arrow(table())
    wanted = pa.schema([("id", pa.int64())])
    with pytest.raises(ValueError, match="1 fields, the data has 3"):
        stream.__arrow_c_stream__(wanted.__arrow_c_schema__())
    assert not stream.consumed


def test_requested_schema_must_be_a_schema_capsule() -> None:
    stream = ArrowStream.from_arrow(table())
    with pytest.raises(ValueError, match="arrow_schema"):
        stream.__arrow_c_stream__(table().__arrow_c_stream__())
    with pytest.raises(TypeError, match="PyCapsule"):
        stream.__arrow_c_stream__("not a capsule")


def test_none_requested_schema() -> None:
    stream = ArrowStream.from_arrow(table())
    got = pa.RecordBatchReader._import_from_c_capsule(stream.__arrow_c_stream__(None)).read_all()
    assert got.equals(table())


def test_non_arrow_object_is_rejected() -> None:
    with pytest.raises(TypeError, match="does not implement the Arrow PyCapsule interface"):
        ArrowStream.from_arrow([1, 2, 3])


class FakeStream:
    def __init__(self, capsule: object) -> None:
        self.capsule = capsule

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        return self.capsule


class FakeArray:
    def __init__(self, result: object) -> None:
        self.result = result

    def __arrow_c_array__(self, requested_schema: object = None) -> object:
        return self.result


def test_wrongly_named_capsules_are_rejected() -> None:
    with pytest.raises(ValueError, match="arrow_array_stream"):
        ArrowStream.from_arrow(FakeStream(table().schema.__arrow_c_schema__()))
    with pytest.raises(TypeError, match="PyCapsule"):
        ArrowStream.from_arrow(FakeStream(42))


def test_malformed_array_protocol_results_are_rejected() -> None:
    batch = table().to_batches()[0]
    schema_capsule, array_capsule = batch.__arrow_c_array__()
    with pytest.raises(TypeError, match="tuple"):
        ArrowStream.from_arrow(FakeArray([schema_capsule, array_capsule]))
    with pytest.raises(TypeError, match="schema capsule, array capsule"):
        ArrowStream.from_arrow(FakeArray((schema_capsule,)))
    with pytest.raises(ValueError, match="arrow_array"):
        ArrowStream.from_arrow(FakeArray((schema_capsule, schema_capsule)))


def test_non_table_array_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[KRISHIV_INVALID_ARROW_DATA\].*struct array"):
        ArrowStream.from_arrow(pa.array([1, 2, 3]))


def test_consumed_producer_stream_is_rejected() -> None:
    capsule = table().__arrow_c_stream__()
    pa.RecordBatchReader._import_from_c_capsule(capsule)
    with pytest.raises(ValueError, match=r"\[KRISHIV_INVALID_ARROW_DATA\].*already consumed"):
        ArrowStream.from_arrow(FakeStream(capsule))


def test_invalid_utf8_from_a_producer_is_rejected() -> None:
    bad = pa.Array.from_buffers(
        pa.string(), 1, [None, pa.py_buffer(b"\x00\x00\x00\x00\x02\x00\x00\x00"), pa.py_buffer(b"\xff\xfe")]
    )
    source = pa.Table.from_arrays([bad], names=["s"])
    stream = ArrowStream.from_arrow(source)
    with pytest.raises(pa.ArrowInvalid, match=r"\[KRISHIV_INVALID_ARROW_DATA\].*Invalid UTF8"):
        pa.table(stream)


def test_producer_error_reaches_the_consumer() -> None:
    def batches() -> Any:
        yield pa.record_batch({"x": [1]})
        raise RuntimeError("boom from the producer")

    reader = pa.RecordBatchReader.from_batches(pa.schema([("x", pa.int64())]), batches())
    stream = ArrowStream.from_arrow(reader)
    with pytest.raises(Exception, match="boom from the producer"):
        pa.table(stream)


def test_one_cached_capsule_is_imported_once_across_threads() -> None:
    # A producer handing out one cached capsule: concurrent imports (truly
    # parallel on the free-threaded build) take the stream exactly once.
    import threading

    producer = FakeStream(table().__arrow_c_stream__())
    results: list[str] = []
    barrier = threading.Barrier(8)

    def take() -> None:
        barrier.wait()
        try:
            stream = ArrowStream.from_arrow(producer)
            results.append("ok" if pa.table(stream).equals(table()) else "wrong data")
        except ValueError as e:
            results.append("consumed" if "already consumed" in str(e) else str(e))

    threads = [threading.Thread(target=take) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["consumed"] * 7 + ["ok"]


def test_one_cached_array_capsule_is_imported_once_across_threads() -> None:
    # As above, for a producer whose __arrow_c_array__ returns one cached pair.
    import threading

    batch = table().to_batches()[0]
    producer = FakeArray(batch.__arrow_c_array__())
    results: list[str] = []
    barrier = threading.Barrier(8)

    def take() -> None:
        barrier.wait()
        try:
            stream = ArrowStream.from_arrow(producer)
            results.append("ok" if pa.table(stream).to_batches()[0].equals(batch) else "wrong data")
        except ValueError as e:
            results.append("consumed" if "already consumed" in str(e) else str(e))

    threads = [threading.Thread(target=take) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["consumed"] * 7 + ["ok"]


def typed_table() -> Any:
    """One column per Arrow layout the bridge checks: every buffer count and
    child rule in the import's header check, plus the Null type (no buffers)."""
    return pa.table(
        {
            "null": pa.nulls(3),
            "bool": pa.array([True, None, False]),
            "f32": pa.array([1.5, None, 2.5], pa.float32()),
            "dec": pa.array([1, None, 3], pa.decimal128(10, 2)),
            "ts": pa.array([1, 2, None], pa.timestamp("us", tz="UTC")),
            "date": pa.array([1, None, 3], pa.date32()),
            "bin": pa.array([b"a", None, b"bc"], pa.binary()),
            "fsb": pa.array([b"ab", None, b"cd"], pa.binary(2)),
            "large": pa.array(["x", None, "yz"], pa.large_string()),
            "view": pa.array(["x", None, "a long string that does not fit inline"], pa.string_view()),
            "list": pa.array([[1, 2], None, []], pa.list_(pa.int64())),
            "large_list": pa.array([[1], None, [2, 3]], pa.large_list(pa.int32())),
            "fixed_list": pa.array([[1, 2], None, [3, 4]], pa.list_(pa.int16(), 2)),
            "struct": pa.array([{"a": 1, "b": "x"}, None, {"a": 3, "b": None}]),
            "map": pa.array([[("k", 1)], None, []], pa.map_(pa.string(), pa.int64())),
            "dict": pa.array(["a", "b", "a"]).dictionary_encode(),
        }
    )


def test_every_layout_round_trips_zero_copy() -> None:
    source = typed_table()
    got = pa.table(ArrowStream.from_arrow(source))
    assert got.equals(source)
    assert addresses(got) == addresses(source)
    batch = source.to_batches()[0]
    got = pa.table(ArrowStream.from_arrow(batch))
    assert got.equals(source)


def test_exported_stream_is_taken_once_across_threads() -> None:
    # Eight threads exporting one stream (parallel on the free-threaded
    # build): exactly one gets the capsule, the others the consumed error.
    import threading

    stream = ArrowStream.from_arrow(table())
    results: list[str] = []
    barrier = threading.Barrier(8)

    def take() -> None:
        barrier.wait()
        try:
            capsule = stream.__arrow_c_stream__()
            got = pa.RecordBatchReader._import_from_c_capsule(capsule).read_all()
            results.append("ok" if got.equals(table()) else "wrong data")
        except ValueError as e:
            results.append("consumed" if "already consumed" in str(e) else str(e))

    threads = [threading.Thread(target=take) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["consumed"] * 7 + ["ok"]
    assert stream.consumed
