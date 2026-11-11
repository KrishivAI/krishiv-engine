# SPDX-License-Identifier: Apache-2.0
"""``pyspark.sql.types`` as the vendored PySpark 4.2.0 conversions import it: ``krishiv.sql.types`` plus
``Row``/``_create_row``, ``VariantVal``, ``typing.cast`` (which ``pyspark.sql.types`` re-exports) and
``_NoValue`` (``pyspark._globals``, read as ``pyspark._NoValue``).

``krishiv.sql.types`` has no ``TimeType``, geometry, geography or user-defined types yet: the classes below
stand in for them so that the vendored code keeps Spark's branches unchanged. ``TimeType`` is made when an
input column is a ``time64`` (an engine ``TIME``); the others are only tested against and never made from
engine data.
"""

from __future__ import annotations

from typing import Any, cast

from krishiv.sql.row import Row, _create_row
from krishiv.sql.types import (
    ArrayType,
    AtomicType,
    BinaryType,
    BooleanType,
    ByteType,
    CalendarIntervalType,
    CharType,
    DataType,
    DateType,
    DayTimeIntervalType,
    DecimalType,
    DoubleType,
    FloatType,
    FractionalType,
    IntegerType,
    IntegralType,
    LongType,
    MapType,
    NullType,
    NumericType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
    VarcharType,
    VariantType,
    YearMonthIntervalType,
)
from krishiv.sql.variant import VariantVal

__all__ = [
    "ArrayType",
    "AtomicType",
    "BinaryType",
    "BooleanType",
    "ByteType",
    "CalendarIntervalType",
    "CharType",
    "DataType",
    "DateType",
    "DayTimeIntervalType",
    "DecimalType",
    "DoubleType",
    "FloatType",
    "FractionalType",
    "Geography",
    "GeographyType",
    "Geometry",
    "GeometryType",
    "IntegerType",
    "IntegralType",
    "LongType",
    "MapType",
    "NullType",
    "NumericType",
    "Row",
    "ShortType",
    "StringType",
    "StructField",
    "StructType",
    "TimeType",
    "TimestampNTZType",
    "TimestampType",
    "UserDefinedType",
    "VarcharType",
    "VariantType",
    "VariantVal",
    "YearMonthIntervalType",
    "_NoValue",
    "_create_row",
    "cast",
]


class _NoValueType:
    """``pyspark._globals._NoValueType``: the one "no value" marker (a zero-column batch's rows)."""

    _instance: _NoValueType | None = None

    def __new__(cls) -> _NoValueType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<no value>"


_NoValue = _NoValueType()


class TimeType(AtomicType):
    """Stands in for ``pyspark.sql.types.TimeType`` (``TIME(6)``): made for a ``time64`` input column."""

    def __init__(self, precision: int = 6) -> None:
        self.precision = precision

    def simpleString(self) -> str:
        return f"time({self.precision})"


class UserDefinedType(DataType):
    """Stands in for ``pyspark.sql.types.UserDefinedType``: Krishiv has no UDTs, so nothing is one."""

    def sqlType(self) -> DataType:
        raise NotImplementedError

    def serialize(self, obj: Any) -> Any:
        raise NotImplementedError

    def deserialize(self, datum: Any) -> Any:
        raise NotImplementedError


class _SpatialType(DataType):
    MIXED_SRID = -1

    def __init__(self, srid: int | str) -> None:
        self.srid = srid

    def toInternal(self, obj: Any) -> Any:
        raise NotImplementedError


class GeometryType(_SpatialType):
    """Stands in for ``pyspark.sql.types.GeometryType``: the Python facade has no spatial values yet."""


class GeographyType(_SpatialType):
    """Stands in for ``pyspark.sql.types.GeographyType``: the Python facade has no spatial values yet."""


class _Spatial:
    @classmethod
    def fromWKB(cls, wkb: bytes, srid: int) -> Any:
        raise NotImplementedError("spatial values in Python UDFs")


class Geometry(_Spatial):
    """Stands in for ``pyspark.sql.types.Geometry``."""


class Geography(_Spatial):
    """Stands in for ``pyspark.sql.types.Geography``."""
