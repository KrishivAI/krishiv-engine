# SPDX-License-Identifier: Apache-2.0
"""Spark data types, as ``pyspark.sql.types`` (the flat types the engine represents)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from krishiv.errors import PySparkValueError, not_expected_type


class DataType:
    """Base of the Spark data types."""

    def simpleString(self) -> str:
        return _SIMPLE[type(self)]

    @classmethod
    def typeName(cls) -> str:
        return cls.__name__[: -len("Type")].lower()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(self.__dict__.items()))))


class NullType(DataType):
    @classmethod
    def typeName(cls) -> str:
        return "void"


class BooleanType(DataType):
    pass


class ByteType(DataType):
    @classmethod
    def typeName(cls) -> str:
        return "byte"


class ShortType(DataType):
    @classmethod
    def typeName(cls) -> str:
        return "short"


class IntegerType(DataType):
    @classmethod
    def typeName(cls) -> str:
        return "integer"


class LongType(DataType):
    @classmethod
    def typeName(cls) -> str:
        return "long"


class FloatType(DataType):
    pass


class DoubleType(DataType):
    pass


class StringType(DataType):
    pass


class BinaryType(DataType):
    pass


class DateType(DataType):
    pass


class TimestampType(DataType):
    pass


class TimestampNTZType(DataType):
    @classmethod
    def typeName(cls) -> str:
        return "timestamp_ntz"


class DecimalType(DataType):
    def __init__(self, precision: int = 10, scale: int = 0) -> None:
        self.precision = precision
        self.scale = scale

    def simpleString(self) -> str:
        return f"decimal({self.precision},{self.scale})"

    def __repr__(self) -> str:
        return f"DecimalType({self.precision},{self.scale})"


_SIMPLE: dict[type[DataType], str] = {
    NullType: "void",
    BooleanType: "boolean",
    ByteType: "tinyint",
    ShortType: "smallint",
    IntegerType: "int",
    LongType: "bigint",
    FloatType: "float",
    DoubleType: "double",
    StringType: "string",
    BinaryType: "binary",
    DateType: "date",
    TimestampType: "timestamp",
    TimestampNTZType: "timestamp_ntz",
}
_BY_SIMPLE: dict[str, type[DataType]] = {v: k for k, v in _SIMPLE.items()}
_DECIMAL = re.compile(r"decimal\((\d+),(\d+)\)")


def _from_simple_string(text: str) -> DataType:
    """The type of a Spark ``simpleString`` (as the engine reports it)."""
    if (cls := _BY_SIMPLE.get(text)) is not None:
        return cls()
    if (m := _DECIMAL.fullmatch(text)) is not None:
        return DecimalType(int(m.group(1)), int(m.group(2)))
    raise PySparkValueError(errorClass="UNSUPPORTED_DATA_TYPE", messageParameters={"data_type": text})


class StructField(DataType):
    def __init__(
        self,
        name: str,
        dataType: DataType,
        nullable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(dataType, DataType):
            raise not_expected_type("dataType", "DataType", dataType)
        self.name = name
        self.dataType = dataType
        self.nullable = nullable
        self.metadata = metadata or {}

    def simpleString(self) -> str:
        return f"{self.name}:{self.dataType.simpleString()}"

    def __repr__(self) -> str:
        return f"StructField('{self.name}', {self.dataType!r}, {self.nullable})"

    def __hash__(self) -> int:
        return hash((self.name, self.dataType, self.nullable))


class StructType(DataType):
    def __init__(self, fields: list[StructField] | None = None) -> None:
        self.fields = list(fields or [])
        self.names = [f.name for f in self.fields]

    @classmethod
    def typeName(cls) -> str:
        return "struct"

    def fieldNames(self) -> list[str]:
        return list(self.names)

    def simpleString(self) -> str:
        return "struct<" + ",".join(f.simpleString() for f in self.fields) + ">"

    def __iter__(self) -> Iterator[StructField]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, key: int | str) -> StructField:
        if isinstance(key, str):
            for f in self.fields:
                if f.name == key:
                    return f
            raise KeyError(f"No StructField named {key}")
        return self.fields[key]

    def __repr__(self) -> str:
        return f"StructType([{', '.join(repr(f) for f in self.fields)}])"

    def __hash__(self) -> int:
        return hash(tuple(self.fields))


def _struct_from_columns(columns: list[tuple[str, str, bool]]) -> StructType:
    """The ``StructType`` of the engine's ``(name, simple type, nullable)`` columns."""
    return StructType([StructField(n, _from_simple_string(t), nullable) for n, t, nullable in columns])


def _simple_string_of(data_type: DataType | str) -> str:
    """The Spark type string of a type given to ``cast``."""
    if isinstance(data_type, DataType):
        return data_type.simpleString()
    if isinstance(data_type, str):
        return data_type
    raise not_expected_type("dataType", "DataType or str", data_type)
