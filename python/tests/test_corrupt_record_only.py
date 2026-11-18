# SPDX-License-Identifier: Apache-2.0
"""PY-076: a read of only the JSON/CSV corrupt-record column is refused when it runs; ``count()`` is a count.

Spark 4.2.0 goldens: ``tests/fixtures/spark-4.2.0/csv-json-corrupt-record/goldens.json`` (``json.inferred.*``,
``pq_as_json.*``). ``df.count()`` is ``groupBy().count()``: no column is read, so it never raises.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from krishiv import SparkSession
from krishiv.errors import PySparkException

pytest.importorskip("pyarrow")

GOLDENS = Path(__file__).resolve().parents[2] / "tests/fixtures/spark-4.2.0/csv-json-corrupt-record/goldens.json"
CLASS = "UNSUPPORTED_FEATURE.QUERY_ONLY_CORRUPT_RECORD_COLUMN"


@pytest.fixture(scope="module")
def spark() -> Any:
    return SparkSession.builder.getOrCreate()


def _bad_json(tmp_path: Path) -> str:
    g = json.loads(GOLDENS.read_text())
    path = tmp_path / "bad.json"
    path.write_bytes(bytes.fromhex(g["inputs_hex"]["bad.json"]))
    return str(path)


def test_count_of_an_inferred_json_read_is_a_count(spark: Any, tmp_path: Path) -> None:
    df = spark.read.json(_bad_json(tmp_path))
    assert df.count() == 4  # json.inferred.count
    assert df.select("_corrupt_record").count() == 4  # ColumnPruning leaves nothing to read


def test_reading_the_corrupt_column_alone_raises_when_it_runs(spark: Any, tmp_path: Path) -> None:
    df = spark.read.json(_bad_json(tmp_path)).select("_corrupt_record")
    assert df.columns == ["_corrupt_record"]  # building the frame is fine
    with pytest.raises(PySparkException) as e:
        df.collect()
    assert e.value.getCondition() == CLASS
    df.explain()  # the plan is shown


def test_a_parquet_directory_read_as_json(spark: Any, tmp_path: Path) -> None:
    out = str(tmp_path / "pq")
    spark.range(3).write.parquet(out)
    df = spark.read.json(out)
    assert [f.name for f in df.schema.fields] == ["_corrupt_record"]  # pq_as_json.schema
    assert isinstance(df.count(), int)  # pq_as_json.count: Spark 2; the number of lines differs with the file
    with pytest.raises(PySparkException) as e:
        df.collect()
    assert e.value.getCondition() == CLASS
