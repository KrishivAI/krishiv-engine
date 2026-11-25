# SPDX-License-Identifier: Apache-2.0
"""The Python package returns the rows of ``tests/fixtures/client-parity/cases.json``, as the npm package's embedded
engine does (``js/test/parity.test.ts``); the canonical form is in that folder's README."""

from __future__ import annotations

import base64
import datetime
import decimal
import json
from pathlib import Path
from typing import Any

import pytest

from krishiv.sql import SparkSession

pa = pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "tests/fixtures/client-parity/cases.json").read_text(encoding="utf-8"))

_INTS = {8: "tinyint", 16: "smallint", 32: "int", 64: "bigint"}


def type_string(t: Any) -> str:
    """The canonical type string of an Arrow type."""
    if pa.types.is_integer(t):
        return _INTS[t.bit_width]
    if pa.types.is_float32(t):
        return "float"
    if pa.types.is_float64(t):
        return "double"
    if pa.types.is_decimal(t):
        return f"decimal({t.precision},{t.scale})"
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t):
        return "string"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_binary_view(t):
        return "binary"
    if pa.types.is_boolean(t):
        return "boolean"
    if pa.types.is_date32(t):
        return "date"
    if pa.types.is_timestamp(t):
        return "timestamp" if t.tz else "timestamp_ntz"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return f"array<{type_string(t.value_type)}>"
    if pa.types.is_struct(t):
        fields = (t.field(i) for i in range(t.num_fields))
        return "struct<" + ",".join(f"{f.name}:{type_string(f.type)}" for f in fields) + ">"
    raise AssertionError(f"no canonical form for {t}")


def canonical(v: Any, t: Any) -> Any:
    """The canonical JSON value of ``v``, a Python value of Arrow type ``t``."""
    if v is None:
        return None
    if pa.types.is_boolean(t) or pa.types.is_floating(t):
        return v
    if pa.types.is_integer(t):
        return str(v)
    if pa.types.is_decimal(t):
        assert isinstance(v, decimal.Decimal)
        return str(v.quantize(decimal.Decimal(1).scaleb(-t.scale)))
    if pa.types.is_timestamp(t):
        assert isinstance(v, datetime.datetime)
        utc = v.astimezone(datetime.UTC) if v.tzinfo else v
        return utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if pa.types.is_date32(t):
        return v.isoformat()
    if isinstance(v, bytes):
        return base64.b16encode(v).decode().lower()
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return [canonical(x, t.value_type) for x in v]
    if pa.types.is_struct(t):
        return {t.field(i).name: canonical(v[t.field(i).name], t.field(i).type) for i in range(t.num_fields)}
    return v


@pytest.mark.parametrize("case", CASES["cases"], ids=[c["name"] for c in CASES["cases"]])
def test_the_python_package_returns_the_parity_rows(case: dict[str, Any]) -> None:
    builder = SparkSession.builder
    for k, v in CASES["conf"].items():
        builder = builder.config(k, v)
    spark = builder.create()
    root = ROOT.as_posix()
    for stmt in case["setup"]:
        spark.sql(stmt.replace("{root}", root)).collect()
    table = spark.sql(case["sql"].replace("{root}", root)).toArrow()
    assert [type_string(f.type) for f in table.schema] == case["types"]
    rows = [{f.name: canonical(r[f.name], f.type) for f in table.schema} for r in table.to_pylist()]
    assert rows == case["expected"]
