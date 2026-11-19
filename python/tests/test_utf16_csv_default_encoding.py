# SPDX-License-Identifier: Apache-2.0
"""SQL-30: a UTF-16 CSV file read with the default (UTF-8) options.

Spark returns the rows and column names with NUL (``\\u0000``) and U+FFFD for the BOM and the bytes that are not
UTF-8, and no error. The ``sql030.*.enc=default.*`` probes of
``tests/fixtures/spark-4.2.0/csv-json-corrupt-record/goldens.json`` (real Spark 4.2.0 runs) give the expected names,
rows and counts. A NUL in a column name cannot cross the Arrow C data interface, so ``collect``/``toArrow`` export
such a column under a placeholder name and rename it back; a NUL in a nested field name, or in a column name read
through the bare ``__arrow_c_stream__``, is refused with ``KRISHIV_UNSUPPORTED_FEATURE``.
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


def _goldens() -> dict[str, Any]:
    return json.loads(GOLDENS.read_text())


def _esc(s: str) -> str:
    """The generator's ``esc``: a backslash doubled, anything outside printable ASCII as ``\\uXXXX``."""
    return "".join("\\\\" if c == "\\" else f"\\u{ord(c):04x}" if ord(c) < 0x20 or ord(c) > 0x7E else c for c in s)


def _row(r: Any) -> str:
    return "[" + ",".join("null" if v is None else _esc(v) for v in r) + "]"


@pytest.fixture(scope="module")
def spark() -> Any:
    return SparkSession.builder.getOrCreate()


@pytest.mark.parametrize("endian", ["le", "be"])
@pytest.mark.parametrize("header", ["true", "false"])
def test_a_utf16_csv_read_as_utf8_matches_spark(spark: Any, tmp_path: Path, endian: str, header: str) -> None:
    g = _goldens()
    path = tmp_path / f"utf16{endian}.csv"
    path.write_bytes(bytes.fromhex(g["inputs_hex"][f"utf16{endian}.csv"]))
    probe = f"sql030.utf16{endian}.header={header}.enc=default"
    df = spark.read.option("header", header).csv(str(path))
    assert ",".join(_esc(n) for n in df.columns) == g["probes"][f"{probe}.columns"]["value"]
    assert df.count() == int(g["probes"][f"{probe}.count"]["value"])
    assert " | ".join(_row(r) for r in df.collect()) == g["probes"][f"{probe}.rows"]["value"]
    table = df.toArrow()
    assert [_esc(n) for n in table.column_names] == [_esc(n) for n in df.columns]


def test_a_nul_column_name_is_refused_through_the_bare_capsule(spark: Any, tmp_path: Path) -> None:
    g = _goldens()
    path = tmp_path / "u.csv"
    path.write_bytes(bytes.fromhex(g["inputs_hex"]["utf16le.csv"]))
    df = spark.read.option("header", "true").csv(str(path))
    pa = pytest.importorskip("pyarrow")
    with pytest.raises(PySparkException) as e:
        pa.table(df)  # a consumer that imports the capsule directly cannot rename the column back
    assert e.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"


def test_a_nul_in_a_nested_field_name_is_a_krishiv_error(spark: Any, tmp_path: Path) -> None:
    path = tmp_path / "n.json"
    path.write_text('{"x": {"k\\u0000": 1}}\n')
    df = spark.read.json(str(path))
    with pytest.raises(PySparkException) as e:
        df.collect()
    assert e.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
