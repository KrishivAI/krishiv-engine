# SPDX-License-Identifier: Apache-2.0
"""``krishiv.testing`` against real ``pyspark.testing`` 4.2.0, scenario by scenario.

The expected outcomes in ``fixtures/pyspark-testing-4.2.0.json`` were recorded by running PySpark itself
(``gen_pyspark_testing_fixture.py``), never written from memory. Each scenario must pass or fail the same way
here, with the same error class and the same message (colour codes removed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _testing_differential import outcome, run_scenario
from _testing_scenarios import SCENARIOS

from krishiv.sql import Row
from krishiv.sql import types as T
from krishiv.testing.utils import assertDataFrameEqual, assertSchemaEqual

EXPECTED = json.loads((Path(__file__).parent / "fixtures" / "pyspark-testing-4.2.0.json").read_text())

pytest.importorskip("pyarrow")

# Where Krishiv deliberately does not copy PySpark. PySpark flags a float mismatch only when
# ``abs(a - b) > atol + rtol * abs(b)``, which is False whenever the left side is NaN or the difference is
# infinite, so ``NaN`` equals ``1.0`` and ``inf`` equals ``-inf`` there and a wrong value slips through a test.
# Krishiv keeps NaN equal only to NaN and an infinity equal only to itself, and fails these two.
DELIBERATE_DIVERGENCES = {"rows-float-nan-vs-value", "rows-float-inf-vs-neginf"}


def test_the_fixture_was_recorded_from_pyspark_4_2_0() -> None:
    assert EXPECTED["spark_version"] == "4.2.0"
    assert set(EXPECTED["outcomes"]) == {sid for sid, *_ in SCENARIOS}


@pytest.mark.parametrize(("sid", "kind", "args", "kwargs"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_agrees_with_pyspark(sid: str, kind: str, args: tuple, kwargs: dict) -> None:
    got = outcome(lambda: run_scenario(kind, args, kwargs, Row, T, assertDataFrameEqual, assertSchemaEqual))
    expected = EXPECTED["outcomes"][sid]
    if sid in DELIBERATE_DIVERGENCES:
        assert expected["ok"], "PySpark is expected to pass this one; the divergence list is stale"
        assert not got["ok"]
        assert got["error_class"] == "DIFFERENT_ROWS"
        return
    assert got == expected
