# SPDX-License-Identifier: Apache-2.0
"""The embedded Python relation UDF kinds (design P3-T085, PR-48a): `mapInPandas`/`mapInArrow`,
`groupBy().applyInPandas`/`applyInArrow`, `cogroup().applyInPandas`/`applyInArrow` and grouped-aggregate
`pandas_udf`, through the `krishiv.sql` facade on a one-thread pool (the GIL default, which is what makes
pool occupancy matter)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from krishiv import _native
from krishiv.errors import UnsupportedOperationException
from krishiv.sql.session import SparkSession


def _spark(threads: str = "1") -> SparkSession:
    config = [("krishiv.python.udf.threads", threads)]
    return SparkSession(_native.Session(config), config)


def _rows(df) -> list[tuple]:
    return sorted(tuple(r) for r in df.collect())


def test_map_in_pandas_sees_every_partition_as_batches() -> None:
    spark = _spark()
    df = spark.range(10).repartition(3)

    def plus_one(batches):
        for pdf in batches:
            yield pdf.assign(id=pdf["id"] + 1)

    assert _rows(df.mapInPandas(plus_one, "id long")) == [(i + 1,) for i in range(10)]


def test_map_in_arrow_may_change_the_row_count_and_the_columns() -> None:
    spark = _spark()
    df = spark.range(5)

    def doubled(batches):
        for b in batches:
            ids = b.column("id").to_pylist()
            yield pa.RecordBatch.from_pydict({"v": ids + ids, "tag": ["x"] * (2 * len(ids))})

    out = df.mapInArrow(doubled, "v long, tag string")
    assert out.columns == ["v", "tag"]
    assert len(out.collect()) == 10


def test_map_in_pandas_barrier_is_refused() -> None:
    spark = _spark()
    with pytest.raises(UnsupportedOperationException):
        spark.range(1).mapInPandas(lambda it: it, "id long", barrier=True)


def test_a_map_result_missing_a_declared_column_fails_the_query() -> None:
    spark = _spark()
    df = spark.range(3).mapInPandas(lambda it: it, "id long, other long")
    with pytest.raises(Exception, match=r"(?i)column|schema|field|other"):
        df.collect()


def test_a_map_result_with_an_extra_column_is_truncated_as_spark_does() -> None:
    spark = _spark()
    df = spark.range(3).mapInPandas(lambda it: (pdf.assign(extra=1) for pdf in it), "id long")
    assert _rows(df) == [(0,), (1,), (2,)]


def test_grouped_map_in_pandas_with_and_without_key() -> None:
    spark = _spark()
    df = spark.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)], ["id", "v"])

    def demean(pdf):
        return pdf.assign(v=pdf["v"] - pdf["v"].mean())

    out = df.groupBy("id").applyInPandas(demean, "id long, v double")
    assert _rows(out) == [(1, -0.5), (1, 0.5), (2, -3.0), (2, -1.0), (2, 4.0)]

    def with_key(key, pdf):
        import pandas as pd

        return pd.DataFrame({"id": [key[0]], "n": [len(pdf)]})

    assert _rows(df.groupBy("id").applyInPandas(with_key, "id long, n long")) == [(1, 2), (2, 3)]


def test_grouped_map_in_arrow() -> None:
    spark = _spark()
    df = spark.createDataFrame([(1, 1), (1, 2), (2, 3)], ["k", "v"])

    def total(table: pa.Table) -> pa.Table:
        return pa.table({"k": [table["k"][0].as_py()], "s": [sum(table["v"].to_pylist())]})

    assert _rows(df.groupBy("k").applyInArrow(total, "k long, s long")) == [(1, 3), (2, 3)]


def test_grouped_map_without_keys_is_one_group() -> None:
    spark = _spark()
    out = spark.range(6).groupBy().applyInPandas(lambda pdf: pdf.assign(id=pdf["id"].sum()).head(1), "id long")
    assert _rows(out) == [(15,)]


def test_cogroup_apply_in_pandas_has_an_empty_side_for_a_key_on_one_side() -> None:
    spark = _spark()
    left = spark.createDataFrame([(1, 1), (1, 2), (2, 3)], ["id", "v"])
    right = spark.createDataFrame([(1, 10), (3, 30)], ["id", "v"])

    def count(lpdf, rpdf):
        import pandas as pd

        key = lpdf["id"].iloc[0] if len(lpdf) else rpdf["id"].iloc[0]
        return pd.DataFrame({"id": [key], "nl": [len(lpdf)], "nr": [len(rpdf)]})

    out = left.groupBy("id").cogroup(right.groupBy("id")).applyInPandas(count, "id long, nl long, nr long")
    assert _rows(out) == [(1, 2, 1), (2, 1, 0), (3, 0, 1)]


def test_cogroup_apply_in_arrow() -> None:
    spark = _spark()
    left = spark.createDataFrame([(1, 1), (2, 2)], ["id", "v"])
    right = spark.createDataFrame([(1, 5)], ["id", "w"])

    def f(lt: pa.Table, rt: pa.Table) -> pa.Table:
        return pa.table({"nl": [lt.num_rows], "nr": [rt.num_rows]})

    out = left.groupBy("id").cogroup(right.groupBy("id")).applyInArrow(f, "nl long, nr long")
    assert _rows(out) == [(1, 0), (1, 1)]


def test_grouped_aggregate_pandas_udf() -> None:
    from krishiv.sql.functions import pandas_udf

    spark = _spark()
    df = spark.createDataFrame([(1, 1.0), (1, 3.0), (2, 5.0)], ["id", "v"])

    mean_udf = pandas_udf(lambda v: v.mean(), "double", 202)  # SQL_GROUPED_AGG_PANDAS_UDF

    out = df.groupBy("id").agg(mean_udf(df["v"]).alias("m"))
    assert _rows(out) == [(1, 2.0), (2, 5.0)]


def test_eight_partitions_finish_on_a_pool_of_one_thread() -> None:
    spark = _spark()
    df = spark.range(800).repartition(8)
    out = df.mapInPandas(lambda it: (pdf for pdf in it), "id long")
    assert len(out.collect()) == 800


def test_a_failing_function_surfaces_its_python_error() -> None:
    spark = _spark()

    def boom(batches):
        raise ValueError("kaboom")
        yield  # pragma: no cover

    with pytest.raises(Exception, match="kaboom"):
        spark.range(3).mapInPandas(boom, "id long").collect()


def test_a_limit_over_a_map_returns_its_rows_and_frees_the_pool() -> None:
    spark = _spark()

    def many(batches):
        import pandas as pd

        for _ in batches:
            pass
        for _ in range(30):
            yield pd.DataFrame({"id": [1]})

    # The executor drains the task before the query ends (it does not cancel it early), so the task
    # must be finite; what matters is that the pool thread is free again afterwards.
    assert len(spark.range(1).mapInPandas(many, "id long").limit(3).collect()) == 3
    assert len(spark.range(2).mapInPandas(lambda it: it, "id long").collect()) == 2


# --- pool occupancy: many groups on a small pool (review of 8e47688fa, findings 1-2). The process's pool is
# started once, with the first session's size, so each case runs in its own interpreter, under a deadline.

_DEADLINE_S = 90

_PRELUDE = """
import sys
import pyarrow as pa
from krishiv import _native
from krishiv.sql.session import SparkSession
from krishiv.sql.functions import pandas_udf

config = [("krishiv.python.udf.threads", sys.argv[1])]
spark = SparkSession(_native.Session(config), config)
"""


def _run_isolated(threads: int, body: str) -> str:
    import subprocess
    import sys

    try:
        done = subprocess.run(
            [sys.executable, "-c", _PRELUDE + body, str(threads)],
            capture_output=True,
            text=True,
            timeout=_DEADLINE_S,
            check=False,
        )
    except subprocess.TimeoutExpired as e:  # a hang is the failure this guards against
        raise AssertionError(f"no result within {_DEADLINE_S}s on {threads} thread(s)") from e
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("threads", [1, 2])
def test_a_scalar_udf_feeding_a_relation_udf_finishes(threads: int) -> None:
    # A relation task holds its pool thread for its whole life; on a pool of one the scalar UDF job feeding it
    # runs on the spare thread (pool.rs `spawn`). Without the spare this query hangs.
    out = _run_isolated(
        threads,
        """
f = pandas_udf(lambda v: v + 1, "long")
df = spark.range(1000).repartition(3).withColumn("x", f("id"))
mapped = df.mapInPandas(lambda it: it, "id long, x long").collect()
grouped = df.groupBy("id").applyInPandas(lambda p: p, "id long, x long").collect()
print("rows", len(mapped), len(grouped))
""",
    )
    assert out == "rows 1000 1000"


_COUNTING = """
import threading, time
active = peak = 0
lock = threading.Lock()
def counting(v):
    global active, peak
    with lock:
        active += 1
        peak = max(peak, active)
    time.sleep(0.002)
    with lock:
        active -= 1
    return v + 1
f = pandas_udf(counting, "long")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
"""


def test_one_thread_pool_never_overlaps_scalar_udf_calls() -> None:
    # `threads = 1` starts a spare OS thread (pool.rs), but scalar calls still run one at a time, on every interpreter.
    out = _run_isolated(
        1,
        _COUNTING
        + """
rows = spark.range(0, 200000, 1, 8).select(f("id")).collect()
print("peak", peak, len(rows))
""",
    )
    assert out == "peak 1 200000"


def test_one_thread_pool_runs_a_relation_udf_beside_gated_scalar_calls() -> None:
    out = _run_isolated(
        1,
        _COUNTING
        + """
df = spark.range(0, 20000, 1, 8).withColumn("x", f("id"))
rows = df.mapInPandas(lambda it: it, "id long, x long").collect()
print("peak", peak, len(rows))
""",
    )
    assert out == "peak 1 20000"


@pytest.mark.parametrize("threads", [1, 2])
def test_apply_in_pandas_over_a_thousand_groups_finishes(threads: int) -> None:
    out = _run_isolated(
        threads,
        """
df = spark.range(1000).repartition(4)
res = df.groupBy("id").applyInPandas(lambda pdf: pdf, "id long")
print("rows", len(res.collect()))
""",
    )
    assert out == "rows 1000"


@pytest.mark.parametrize("threads", [1, 2])
def test_grouped_aggregate_udf_over_a_thousand_groups_finishes(threads: int) -> None:
    out = _run_isolated(
        threads,
        """
df = spark.range(1000).repartition(4)
s = pandas_udf(lambda v: v.sum(), "long", 202)
res = df.groupBy("id").agg(s(df["id"]).alias("s"))
print("rows", len(res.collect()))
""",
    )
    assert out == "rows 1000"


@pytest.mark.parametrize("threads", [1, 2, 3])
def test_two_grouped_aggregate_udfs_in_one_agg_finish_or_name_the_limit(threads: int) -> None:
    out = _run_isolated(
        threads,
        """
df = spark.range(200).repartition(4)
s = pandas_udf(lambda v: v.sum(), "long", 202)
m = pandas_udf(lambda v: v.max(), "long", 202)
try:
    res = df.groupBy("id").agg(s(df["id"]).alias("s"), m(df["id"]).alias("m"))
    print("rows", len(res.collect()))
except Exception as e:
    msg = str(e)
    print("error", "KRISHIV_PYTHON_WORKER_LIMIT" in msg, msg.count('"krishiv.python.udf.threads" is %d' % THREADS))
""".replace("THREADS", str(threads)),
    )
    # Two lanes need two slots. A pool of one has one slot and a pool of two keeps one thread for scalar UDFs,
    # so both refuse; the message reports the configured setting, not the slot count. A pool of three runs it.
    assert out == ("rows 200" if threads == 3 else "error True 1")


@pytest.mark.parametrize("threads", [1, 2])
def test_cogroup_over_many_keys_finishes(threads: int) -> None:
    out = _run_isolated(
        threads,
        """
left = spark.range(500).repartition(3)
right = spark.range(500).repartition(2)

def f(lt, rt):
    return pa.table({"nl": [lt.num_rows], "nr": [rt.num_rows]})

res = left.groupBy("id").cogroup(right.groupBy("id")).applyInArrow(f, "nl long, nr long")
print("rows", len(res.collect()))
""",
    )
    assert out == "rows 500"


@pytest.mark.parametrize("threads", [1, 2])
def test_a_failing_group_frees_the_pool_for_the_next_query(threads: int) -> None:
    out = _run_isolated(
        threads,
        """
def boom(pdf):
    raise ValueError("kaboom")

df = spark.range(300).repartition(3)
try:
    df.groupBy("id").applyInPandas(boom, "id long").collect()
    print("no error")
except Exception as e:
    assert "kaboom" in str(e)
    ok = spark.range(5).mapInPandas(lambda it: it, "id long").collect()
    print("rows", len(ok))
""",
    )
    assert out == "rows 5"


@pytest.mark.parametrize("threads", [1, 2])
def test_chained_map_in_pandas_nodes_finish(threads: int) -> None:
    out = _run_isolated(
        threads,
        """
df = spark.range(2000).repartition(3)
res = df.mapInPandas(lambda it: it, "id long").mapInPandas(lambda it: it, "id long")
print("rows", len(res.collect()))
""",
    )
    assert out == "rows 2000"


@pytest.mark.parametrize("threads", [1, 2])
def test_apply_in_pandas_then_map_in_pandas_finish(threads: int) -> None:
    out = _run_isolated(
        threads,
        """
df = spark.range(1000).repartition(3)
res = (
    df.groupBy("id").applyInPandas(lambda pdf: pdf, "id long").mapInPandas(lambda it: it, "id long")
)
print("rows", len(res.collect()))
""",
    )
    assert out == "rows 1000"


@pytest.mark.parametrize("threads", [1, 2])
def test_two_python_threads_collecting_on_one_slot_both_finish(threads: int) -> None:
    out = _run_isolated(
        threads,
        """
import threading
df = spark.range(1000).repartition(3)
res = df.groupBy("id").applyInPandas(lambda pdf: pdf, "id long")
counts = []
def run():
    counts.append(len(res.collect()))
ts = [threading.Thread(target=run) for _ in range(3)]
[t.start() for t in ts]
[t.join() for t in ts]
print("rows", sorted(counts))
""",
    )
    assert out == "rows [1000, 1000, 1000]"


def test_cancel_during_a_long_group_frees_the_pool() -> None:
    out = _run_isolated(
        1,
        """
import threading, time
def slow(pdf):
    time.sleep(0.2)
    return pdf

df = spark.range(300).repartition(1)
box = []
started = threading.Event()
def slow_started(pdf):
    started.set()
    return slow(pdf)
res = df.groupBy("id").applyInPandas(slow_started, "id long")
def run():
    try:
        box.append(res.collect())
    except Exception as e:
        box.append(e)
t = threading.Thread(target=run)
t.start()
assert started.wait(60), "the UDF never started"
spark.interruptAll()
t.join(30)
assert not t.is_alive(), "the cancelled query did not return"
assert len(box) == 1 and isinstance(box[0], Exception), f"the query was not cancelled: {box!r}"
assert "OPERATION_CANCELED" in str(box[0]) or "cancel" in str(box[0]).lower(), str(box[0])
ok = spark.range(5).mapInPandas(lambda it: it, "id long").collect()
print("rows", len(ok))
""",
    )
    assert out == "rows 5"


def test_a_zero_column_frame_keeps_its_row_count_into_a_map() -> None:
    spark = _spark()
    df = spark.range(5).select()

    def count_pandas(batches):
        import pandas as pd

        total = sum(len(pdf) for pdf in batches)
        yield pd.DataFrame({"n": [total]})

    def count_arrow(batches):
        total = sum(b.num_rows for b in batches)
        yield pa.RecordBatch.from_pydict({"n": [total]})

    assert _rows(df.mapInPandas(count_pandas, "n long")) == [(5,)]
    assert _rows(df.mapInArrow(count_arrow, "n long")) == [(5,)]


def _count_cogroup(lpdf, rpdf):
    import pandas as pd

    return pd.DataFrame({"nl": [len(lpdf)], "nr": [len(rpdf)]})


def test_a_cogroup_with_an_unparsable_right_table_name_raises_the_right_sides_error() -> None:
    spark = _spark()
    left = spark.createDataFrame([(1, 1)], ["id", "v"])
    right = spark.table("a b")  # a deferred build error, not an unresolved table
    out = left.groupBy("id").cogroup(right.groupBy("id")).applyInPandas(_count_cogroup, "nl long, nr long")
    with pytest.raises(Exception, match="PARSE_SYNTAX_ERROR"):
        out.collect()


def test_a_cogroup_owns_its_right_sides_lazy_checkpoint() -> None:
    spark = _spark()
    left = spark.createDataFrame([(1, 1)], ["id", "v"])
    right = spark.createDataFrame([(1, 2)], ["id", "v"]).localCheckpoint(eager=False)
    out = left.groupBy("id").cogroup(right.groupBy("id")).applyInPandas(_count_cogroup, "nl long, nr long")
    del right
    assert _rows(out) == [(1, 1)]


def test_a_map_after_the_pool_stopped_fails_instead_of_returning_a_short_partition() -> None:
    # A task whose job the stopped pool dropped unrun closes its result channel without `finished`: that is an
    # error, never an empty rest of the partition (InProcessTask::next).
    out = _run_isolated(
        1,
        """
df = spark.range(5)
print("first", len(df.mapInPandas(lambda it: it, "id long").collect()))
from krishiv import _native
_native._udf_pool_shutdown()
try:
    print("rows", len(df.mapInPandas(lambda it: it, "id long").collect()))
except Exception:
    print("error")
""",
    )
    assert out == "error"


_VALIDATE_KEYS = [
    "spark.sql.execution.arrow.pyspark.validateSchema.enabled",
    "krishiv.sql.execution.arrow.pyspark.validateSchema.enabled",
]


def _spark_validating(key: str, value: str = "true") -> SparkSession:
    config = [("krishiv.python.udf.threads", "1"), (key, value)]
    return SparkSession(_native.Session(config), config)


def _misnamed(batches):
    for b in batches:
        yield pa.RecordBatch.from_pydict({"wrong": b.column("id").to_pylist()})


@pytest.mark.parametrize("key", _VALIDATE_KEYS)
def test_map_in_arrow_with_validate_schema_refuses_a_misnamed_result(key: str) -> None:
    df = _spark_validating(key).range(3).mapInArrow(_misnamed, "id long")
    with pytest.raises(Exception, match=r"ARROW_TYPE_MISMATCH") as info:
        df.collect()
    assert "SQL_MAP_ARROW_ITER_UDF" in str(info.value)


def test_map_in_arrow_without_validate_schema_relabels_a_misnamed_result() -> None:
    df = _spark().range(3).mapInArrow(_misnamed, "id long")
    assert _rows(df) == [(0,), (1,), (2,)]
    off = _spark_validating(_VALIDATE_KEYS[0], "false").range(3).mapInArrow(_misnamed, "id long")
    assert _rows(off) == [(0,), (1,), (2,)]


def test_a_windowed_grouped_aggregate_pandas_udf_in_sql_is_refused() -> None:
    from krishiv.sql.functions import pandas_udf

    spark = _spark()
    total = pandas_udf(lambda v: v.sum(), "long", 202)  # SQL_GROUPED_AGG_PANDAS_UDF
    spark.udf.register("total_udf", total)
    spark.range(4).createOrReplaceTempView("t")
    with pytest.raises(Exception, match=r"KRISHIV_UNSUPPORTED_FEATURE|NOT_IMPLEMENTED"):
        spark.sql("SELECT total_udf(id) OVER (PARTITION BY id % 2) FROM t").collect()
