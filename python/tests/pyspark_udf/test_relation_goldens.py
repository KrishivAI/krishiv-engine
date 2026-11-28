# SPDX-License-Identifier: Apache-2.0
"""Python relation UDFs (``mapInPandas`` / ``mapInArrow``, grouped and cogrouped maps, grouped aggregates) run
embedded, against PySpark 4.2.0's answers.

``relation/goldens.json`` comes from a real local PySpark 4.2.0 run of ``relation/cases.py``
(``relation/gen_goldens.py``). Each case is one pytest case run on a Krishiv session: the schema and rows must equal
Spark's, or the error must carry Spark's condition and the Python worker's ``[CLASS] message``. Where Spark's JVM
failed on a result the worker did not check (``UNSUPPORTED_CALL``, or no condition at all), Krishiv must fail too, with
any error. A case that differs is a failure unless ``KNOWN_GAPS`` lists it with a reason; a listed case that now
matches fails too, so the list cannot go stale.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from pyspark_udf._support import have_pandas, have_pyarrow

pytestmark = [
    pytest.mark.skipif(not (have_pandas and have_pyarrow), reason="needs pandas and pyarrow"),
    pytest.mark.filterwarnings("ignore"),
]

GOLDENS = json.loads((Path(__file__).resolve().parent / "relation" / "goldens.json").read_text())

#: Conditions Spark raises from inside the JVM when it reads a result the worker let through.
JVM_FAILURES = {None, "UNSUPPORTED_CALL.WITHOUT_SUGGESTION"}

#: Krishiv's own condition where Spark's JVM fails (the golden keeps Spark's failure as provenance).
KRISHIV_REFUSALS: dict[str, str] = {
    "map/arrow/missing_column": "PYTHON_EXCEPTION",
    "map/arrow/reordered_by_name": "KRISHIV_INVALID_ARROW_DATA",
    "map/arrow/wrong_type": "KRISHIV_INVALID_ARROW_DATA",
    "not_null/cogrouped/pandas": "PYTHON_EXCEPTION",
    "not_null/grouped/arrow": "PYTHON_EXCEPTION",
    "not_null/grouped/pandas": "PYTHON_EXCEPTION",
    "not_null/map/arrow": "PYTHON_EXCEPTION",
    "not_null/map/pandas": "PYTHON_EXCEPTION",
}

#: Known gaps: (case regex, reason). A matching case must still differ from Spark.
KNOWN_GAPS: list[tuple[str, str]] = []


def gap(case: str) -> str | None:
    return next((reason for pattern, reason in KNOWN_GAPS if re.fullmatch(pattern, case)), None)


@pytest.fixture(scope="module")
def spark() -> Any:
    from krishiv import _native
    from krishiv.sql.session import SparkSession
    from pyspark_udf.relation import cases

    config = list(cases.CONF.items())
    return SparkSession(_native.Session(config), config)


def difference(spark: Any, case: str) -> str | None:
    """None when Krishiv answers ``case`` as Spark did, else what differs."""
    from krishiv.errors import PySparkException
    from pyspark_udf.relation import cases

    want = GOLDENS["cases"][case]
    try:
        df = cases.CASES[case](cases.Api("krishiv"), spark)
        got: dict[str, Any] = {
            "schema": df.schema.simpleString(),
            "rows": [[cases.cell(v) for v in row] for row in df.collect()],
        }
    except PySparkException as e:
        if "error" not in want:
            return f"raised {type(e).__name__}: {e}"
        err = want["error"]
        if err.get("condition") in JVM_FAILURES:
            # Spark fails inside the JVM; Krishiv's own refusal is pinned here.
            expected = KRISHIV_REFUSALS[case]
            if e.getCondition() != expected:
                return f"condition {e.getCondition()!r}, Krishiv refuses with {expected!r}: {e}"
            return None
        condition = e.getCondition()
        if condition != err["condition"]:
            return f"condition {condition!r}, Spark {err['condition']!r}: {e}"
        worker = err["worker"]
        if worker:
            line = worker["message"]
            if worker["error_class"]:
                line = f"[{worker['error_class']}] {line}"
            if line not in str(e):
                return f"no {line!r} in: {e}"
        return None
    if "error" in want:
        return f"returned {got}, Spark raised {want['error']}"
    return None if got == want else f"returned {got}, Spark {want}"


@pytest.mark.parametrize("case", sorted(GOLDENS["cases"]))
def test_the_case_matches_spark(spark: Any, case: str) -> None:
    reason = gap(case)
    if reason:
        pytest.skip(f"known gap: {reason}")
    diff = difference(spark, case)
    assert diff is None, diff


@pytest.mark.parametrize("case", sorted(c for c in GOLDENS["cases"] if gap(c)))
def test_every_known_gap_still_differs(spark: Any, case: str) -> None:
    assert difference(spark, case) is not None, f"{case} matches Spark now: drop it from KNOWN_GAPS"


def test_the_goldens_cover_every_case() -> None:
    from pyspark_udf.relation import cases

    assert sorted(GOLDENS["cases"]) == sorted(cases.CASES), "rerun relation/gen_goldens.py"
    assert GOLDENS["spark"] == "4.2.0"


@pytest.mark.parametrize("whole_struct_null", [False, True])
def test_a_nested_not_null_field_is_not_enforced(spark: Any, whole_struct_null: bool) -> None:
    """Spark 4.2.0 (a local run of this shape) returns the rows: only the top-level columns' NOT NULL is checked."""
    import pyarrow as pa

    def f(batches: Any) -> Any:
        for batch in batches:
            if whole_struct_null:
                s = pa.array([None] * batch.num_rows, type=pa.struct([("a", pa.int64())]))
            else:
                s = pa.StructArray.from_arrays([batch.column("k")], names=["a"])
            yield pa.RecordBatch.from_arrays([s], names=["s"])

    df = spark.createDataFrame([(1,), (2,)], "k long").coalesce(1).mapInArrow(f, "s struct<a:long not null>")
    want = [None, None] if whole_struct_null else [{"a": 1}, {"a": 2}]
    assert [r["s"].asDict() if r["s"] is not None else None for r in df.collect()] == want
