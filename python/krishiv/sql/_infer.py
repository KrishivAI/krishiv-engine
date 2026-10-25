# SPDX-License-Identifier: Apache-2.0
"""Schema inference for ``createDataFrame`` over Python rows.

The rules of ``pyspark.sql.types._infer_type`` and ``_merge_type`` (PySpark 4.2): each
value has the Spark type its Python type maps to, a row's types are merged with the
types of the rows before it, and a column that is still ``void`` at the end cannot be
typed. Merging is strict: ``bigint`` with ``double`` is an error, not a widening.

Naive datetimes infer ``timestamp``; with ``spark.sql.timestampType = TIMESTAMP_NTZ``
they infer ``timestamp_ntz`` instead (time-zone-aware datetimes always infer
``timestamp``). Two Python types have no Spark type here yet: ``datetime.timedelta``
and ``datetime.time`` are ``UNSUPPORTED_DATA_TYPE`` until the engine can materialize
their types.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from decimal import Decimal
from functools import reduce
from typing import Any, cast

from krishiv.errors import PySparkTypeError, PySparkValueError
from krishiv.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    LongType,
    MapType,
    NullType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

# Keyed by exact type, as PySpark does: a subclass (a ``bool`` is an ``int`` to Python)
# or a type not listed is not inferred.
_ATOMIC: dict[type, Callable[[], DataType]] = {
    type(None): NullType,
    bool: BooleanType,
    int: LongType,
    float: DoubleType,
    str: StringType,
    bytearray: BinaryType,
    bytes: BinaryType,
    # The precision and scale of one value say nothing about the next row's.
    Decimal: lambda: DecimalType(38, 18),
    datetime.date: DateType,
    datetime.datetime: TimestampType,
}


def infer_type(value: Any, timestamp_ntz: bool = False) -> DataType:
    """The Spark type of one Python value: a naive datetime is ``timestamp_ntz``
    when the session prefers ``TIMESTAMP_NTZ``, else ``timestamp``."""
    if type(value) is datetime.datetime and value.tzinfo is None and timestamp_ntz:
        return TimestampNTZType()
    atomic = _ATOMIC.get(type(value))
    if atomic is not None:
        return atomic()
    if isinstance(value, dict):
        key_type: DataType = NullType()
        value_type: DataType = NullType()
        for k, v in value.items():
            if k is not None:
                key_type = merge_type(key_type, infer_type(k, timestamp_ntz))
            if v is not None:
                value_type = merge_type(value_type, infer_type(v, timestamp_ntz))
        return MapType(key_type, value_type, True)
    if isinstance(value, list):
        element = reduce(merge_type, (infer_type(v, timestamp_ntz) for v in value)) if value else NullType()
        return ArrayType(element, True)
    if isinstance(value, tuple):
        return infer_row(value, None, timestamp_ntz)
    raise PySparkTypeError(errorClass="UNSUPPORTED_DATA_TYPE", messageParameters={"data_type": type(value).__name__})


def infer_row(row: Any, names: list[str] | None, timestamp_ntz: bool = False) -> StructType:
    """The struct of one row's values: named by ``names``, else by the row's own
    field names (a ``Row``, a named tuple), else ``_1``, ``_2``, ..."""
    if names is None:
        names = getattr(row, "__fields__", None) or getattr(row, "_fields", None)
    if names is None:
        names = [f"_{i + 1}" for i in range(len(row))]
    return StructType(
        [StructField(name, infer_type(v, timestamp_ntz), True) for name, v in zip(names, row, strict=True)]
    )


def merge_type(a: DataType, b: DataType) -> DataType:
    """The type of values of type ``a`` and of type ``b``: ``void`` merges with
    anything, equal kinds merge (nested types recursively), anything else is
    ``CANNOT_MERGE_TYPE``."""
    if isinstance(a, NullType):
        return b
    if isinstance(b, NullType):
        return a
    if type(a) is not type(b):
        raise PySparkTypeError(
            errorClass="CANNOT_MERGE_TYPE",
            messageParameters={"data_type1": type(a).__name__, "data_type2": type(b).__name__},
        )
    if isinstance(a, StructType) and isinstance(b, StructType):
        others = {f.name: f.dataType for f in b.fields}
        fields = [StructField(f.name, merge_type(f.dataType, others.get(f.name, NullType()))) for f in a.fields]
        known = {f.name for f in fields}
        fields.extend(StructField(n, t) for n, t in others.items() if n not in known)
        return StructType(fields)
    if isinstance(a, ArrayType) and isinstance(b, ArrayType):
        return ArrayType(merge_type(a.elementType, b.elementType), True)
    if isinstance(a, MapType) and isinstance(b, MapType):
        return MapType(merge_type(a.keyType, b.keyType), merge_type(a.valueType, b.valueType), True)
    return a


def has_nulltype(data_type: DataType) -> bool:
    """Whether ``data_type`` is, or contains, ``void``."""
    if isinstance(data_type, StructType):
        return any(has_nulltype(f.dataType) for f in data_type.fields)
    if isinstance(data_type, ArrayType):
        return has_nulltype(data_type.elementType)
    if isinstance(data_type, MapType):
        return has_nulltype(data_type.keyType) or has_nulltype(data_type.valueType)
    return isinstance(data_type, NullType)


def infer_schema(rows: list[list[Any]], names: list[str], timestamp_ntz: bool = False) -> StructType:
    """The struct that every row of ``rows`` (one value per name) fits.

    Raises ``CANNOT_MERGE_TYPE`` at the first row whose types do not merge with the rows
    before it, and ``CANNOT_DETERMINE_TYPE`` when a column is only ever ``None`` (or only
    holds empty lists and dicts)."""
    if not rows:
        raise PySparkValueError(errorClass="CANNOT_INFER_EMPTY_SCHEMA")
    merged = reduce(merge_type, (infer_row(row, names, timestamp_ntz) for row in rows))
    if has_nulltype(merged):
        raise PySparkValueError(errorClass="CANNOT_DETERMINE_TYPE")
    return cast(StructType, merged)
