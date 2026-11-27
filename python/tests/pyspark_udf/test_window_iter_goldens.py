# SPDX-License-Identifier: Apache-2.0
"""Python window aggregates (eval types 203, 253) and scalar iterators (204, 251) against PySpark 4.2.0's answers.

``window_iter/goldens.json`` comes from a real local PySpark 4.2.0 run of ``window_iter/cases.py``
(``window_iter/gen_goldens.py``). Each case is one pytest case run on a Krishiv session: the schema and rows must
equal Spark's, or the error must carry Spark's condition and the Python worker's ``[CLASS] message``. A case that
differs is a failure unless ``KNOWN_GAPS`` lists it with a reason; a listed case that now matches fails too, so the
list cannot go stale.
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

GOLDENS = json.loads((Path(__file__).resolve().parent / "window_iter" / "goldens.json").read_text())

#: Known gaps: (case regex, reason). A matching case must still differ from Spark.
KNOWN_GAPS: list[tuple[str, str]] = [
    (
        r"window/.*|iter/(arrow|pandas)_(first_only|twice)",
        "the embedded facade refuses eval types 202/203 and 252/253 (NOT_IMPLEMENTED), and its scalar iterator "
        "UDFs (204/251) differ on the first_only/twice cases, until its in-process runtime runs them (PR-48, P3-T085); "
        "the server's worker pool matches every case "
        "(krishiv-udf-pool tests/window_iter_goldens.rs replays these goldens as SQL on a session over the pool)",
    ),
]


def gap(case: str) -> str | None:
    return next((reason for pattern, reason in KNOWN_GAPS if re.fullmatch(pattern, case)), None)


@pytest.fixture(scope="module")
def spark() -> Any:
    from krishiv.sql import SparkSession
    from pyspark_udf.window_iter import cases

    builder = SparkSession.builder
    for k, v in cases.CONF.items():
        builder = builder.config(k, v)
    return builder.getOrCreate()


def difference(spark: Any, case: str) -> str | None:
    """None when Krishiv answers ``case`` as Spark did, else what differs."""
    from krishiv.errors import PySparkException
    from pyspark_udf.window_iter import cases

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
        condition = e.getCondition()
        if condition != err["condition"]:
            return f"condition {condition!r}, Spark {err['condition']!r}: {e}"
        worker = err["worker"]
        if worker and f"[{worker['error_class']}] {worker['message']}" not in str(e):
            return f"no [{worker['error_class']}] {worker['message']!r} in: {e}"
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
    from pyspark_udf.window_iter import cases

    assert sorted(GOLDENS["cases"]) == sorted(cases.CASES), "rerun window_iter/gen_goldens.py"
    assert GOLDENS["spark"] == "4.2.0"
