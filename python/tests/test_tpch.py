# SPDX-License-Identifier: Apache-2.0
"""The 22 TPC-H queries from Python (P2-T091b): ``spark.read.parquet`` + ``createOrReplaceTempView`` + ``spark.sql``,
results compared with Spark 4.2.0's answers.

Runs over the checked-in SF 0.01 tables; set ``KRISHIV_TPCH_SF1_DIR`` (dbgen's tables as Parquet, one file per table)
to also run SF 1 against ``tests/fixtures/tpch-sf1-spark-answers.json`` of ``krishiv-exec``.
"""

from __future__ import annotations

import json
import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from krishiv.sql import Row, SparkSession

pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "crates/krishiv-exec/tests/fixtures"
QUERIES = ROOT / "bench/queries/tpch"
TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]

pytestmark = pytest.mark.skipif(
    not QUERIES.is_dir(), reason="the repository's TPC-H fixtures are not next to the tests"
)


def render(value: Any) -> str:
    """A value as Spark's answers render it (PySpark's ``repr`` for floats)."""
    if value is None:
        return "NULL"
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def lines(rows: list[Row]) -> list[str]:
    return ["|".join(render(v) for v in row) for row in rows]


def session(data: Path) -> SparkSession:
    spark = SparkSession.builder.create()
    for table in TABLES:
        spark.read.parquet(str(data / f"{table}.parquet")).createOrReplaceTempView(table)
    return spark


def check(data: Path, answers_path: Path, queries: range = range(1, 23)) -> None:
    answers = json.loads(answers_path.read_text())["queries"]
    spark = session(data)
    wrong = []
    for q in queries:
        name = f"q{q:02d}"
        got = lines(spark.sql((QUERIES / f"{name}.sql").read_text()).collect())
        if got != answers[name]["rows"]:
            wrong.append(f"{name}: got {len(got)} rows, want {len(answers[name]['rows'])}; first got {got[:1]}")
    assert not wrong, "\n".join(wrong)


def test_tpch_sf0_01_matches_spark() -> None:
    data = FIXTURES / "tpch-sf0.01-parquet"
    check(data, data / "spark-answers.json")


@pytest.mark.skipif("KRISHIV_TPCH_SF1_DIR" not in os.environ, reason="needs the TPC-H SF1 data set")
def test_tpch_sf1_matches_spark() -> None:
    check(Path(os.environ["KRISHIV_TPCH_SF1_DIR"]), FIXTURES / "tpch-sf1-spark-answers.json")
