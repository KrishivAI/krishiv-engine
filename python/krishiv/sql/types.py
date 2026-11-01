# SPDX-License-Identifier: Apache-2.0
"""Spark data types, as ``pyspark.sql.types``.

Atomic types map to engine types through the existing conversion (the
engine's DDL parser): ``cast`` and ``createDataFrame`` with one of them
behave exactly as the SQL equivalent. The nested types (``ArrayType``,
``MapType``, ``StructType`` with nested fields) and the ``CharType``,
``VarcharType`` and interval types mirror ``pyspark.sql.types`` exactly
(``simpleString``/``jsonValue``/DDL round trips), but the engine has no
equivalent of them yet: using one where an engine type is needed fails
with the engine's own ``KRISHIV_UNSUPPORTED_FEATURE`` error, as the SQL
equivalent does.
"""

from __future__ import annotations

import json as _json
import re
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar, overload


class DataType:
    """Base of the Spark data types."""

    def simpleString(self) -> str:
        return self.typeName()

    @classmethod
    def typeName(cls) -> str:
        return cls.__name__[: -len("Type")].lower()

    def jsonValue(self) -> str | dict[str, Any]:
        return self.typeName()

    def json(self) -> str:
        """This type as a JSON string, as ``pyspark.sql.types`` writes it."""
        return _json.dumps(self.jsonValue(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def fromDDL(cls, ddl: str) -> DataType:
        """The type of a DDL-formatted string, as ``pyspark.sql.types`` reads it.

        A top-level struct may omit the ``struct<>`` (``"a int, b string"``),
        for compatibility with ``createDataFrame``.
        """
        return _parse_datatype_string(ddl)

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


class AtomicType(DataType):
    """A type that is not null, an array, a map or a struct."""


class NumericType(AtomicType):
    """A numeric type."""


class IntegralType(NumericType):
    """An integral type."""


class FractionalType(NumericType):
    """A fractional type."""


class BooleanType(AtomicType):
    pass


class ByteType(IntegralType):
    @classmethod
    def typeName(cls) -> str:
        return "byte"

    def simpleString(self) -> str:
        return "tinyint"


class ShortType(IntegralType):
    @classmethod
    def typeName(cls) -> str:
        return "short"

    def simpleString(self) -> str:
        return "smallint"


class IntegerType(IntegralType):
    @classmethod
    def typeName(cls) -> str:
        return "integer"

    def simpleString(self) -> str:
        return "int"


class LongType(IntegralType):
    @classmethod
    def typeName(cls) -> str:
        return "long"

    def simpleString(self) -> str:
        return "bigint"


class FloatType(FractionalType):
    pass


class DoubleType(FractionalType):
    pass


class DecimalType(FractionalType):
    def __init__(self, precision: int = 10, scale: int = 0) -> None:
        self.precision = precision
        self.scale = scale

    def simpleString(self) -> str:
        return f"decimal({self.precision},{self.scale})"

    def jsonValue(self) -> str:
        return self.simpleString()

    def __repr__(self) -> str:
        return f"DecimalType({self.precision},{self.scale})"


class StringType(AtomicType):
    """A string; ``collation`` is the name as written (``UTF8_BINARY`` by default).

    As in PySpark 4, a collated string prints ``string collate <name>`` and two
    strings are equal only when the collation names are spelled alike.
    """

    def __init__(self, collation: str = "UTF8_BINARY") -> None:
        self.collation = collation

    def _collated(self) -> bool:
        return self.collation != "UTF8_BINARY"

    def simpleString(self) -> str:
        return f"string collate {self.collation}" if self._collated() else "string"

    def jsonValue(self) -> str:
        return self.simpleString()

    def __repr__(self) -> str:
        return f"StringType({self.collation!r})" if self._collated() else "StringType()"


class CharType(AtomicType):
    """A fixed-length blank-padded string, ``char(n)``."""

    def __init__(self, length: int) -> None:
        self.length = length

    def simpleString(self) -> str:
        return f"char({self.length})"

    def jsonValue(self) -> str:
        return self.simpleString()

    def __repr__(self) -> str:
        return f"CharType({self.length})"


class VarcharType(AtomicType):
    """A variable-length string of at most ``n`` characters, ``varchar(n)``."""

    def __init__(self, length: int) -> None:
        self.length = length

    def simpleString(self) -> str:
        return f"varchar({self.length})"

    def jsonValue(self) -> str:
        return self.simpleString()

    def __repr__(self) -> str:
        return f"VarcharType({self.length})"


class BinaryType(AtomicType):
    pass


class DateType(AtomicType):
    pass


class TimestampType(AtomicType):
    pass


class TimestampNTZType(AtomicType):
    @classmethod
    def typeName(cls) -> str:
        return "timestamp_ntz"


class DayTimeIntervalType(AtomicType):
    """An ANSI day-time interval, ``interval day to second`` by default."""

    DAY = 0
    HOUR = 1
    MINUTE = 2
    SECOND = 3

    _fields: ClassVar[dict[int, str]] = {DAY: "day", HOUR: "hour", MINUTE: "minute", SECOND: "second"}
    _inverted_fields: ClassVar[dict[str, int]] = {v: k for k, v in _fields.items()}

    def __init__(self, startField: int | None = None, endField: int | None = None) -> None:
        if startField is None and endField is None:
            startField, endField = self.DAY, self.SECOND
        elif endField is None:
            endField = startField
        if startField not in self._fields or endField not in self._fields:
            from krishiv.errors import PySparkRuntimeError

            raise PySparkRuntimeError(
                errorClass="INVALID_INTERVAL_CASTING",
                messageParameters={"start_field": str(startField), "end_field": str(endField)},
            )
        self.startField = startField
        self.endField = endField

    def _str_repr(self) -> str:
        start = self._fields[self.startField]
        end = self._fields[self.endField]
        return f"interval {start}" if start == end else f"interval {start} to {end}"

    simpleString = _str_repr
    jsonValue = _str_repr

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.startField}, {self.endField})"


class YearMonthIntervalType(AtomicType):
    """An ANSI year-month interval, ``interval year to month`` by default."""

    YEAR = 0
    MONTH = 1

    _fields: ClassVar[dict[int, str]] = {YEAR: "year", MONTH: "month"}
    _inverted_fields: ClassVar[dict[str, int]] = {v: k for k, v in _fields.items()}

    def __init__(self, startField: int | None = None, endField: int | None = None) -> None:
        if startField is None and endField is None:
            startField, endField = self.YEAR, self.MONTH
        elif endField is None:
            endField = startField
        if startField not in self._fields or endField not in self._fields:
            from krishiv.errors import PySparkRuntimeError

            raise PySparkRuntimeError(
                errorClass="INVALID_INTERVAL_CASTING",
                messageParameters={"start_field": str(startField), "end_field": str(endField)},
            )
        self.startField = startField
        self.endField = endField

    def _str_repr(self) -> str:
        start = self._fields[self.startField]
        end = self._fields[self.endField]
        return f"interval {start}" if start == end else f"interval {start} to {end}"

    simpleString = _str_repr
    jsonValue = _str_repr

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.startField}, {self.endField})"


class CalendarIntervalType(AtomicType):
    """A calendar interval (months, days, microseconds)."""

    @classmethod
    def typeName(cls) -> str:
        return "interval"


class ArrayType(DataType):
    """An array of ``elementType`` values, as ``pyspark.sql.types.ArrayType``."""

    def __init__(self, elementType: DataType, containsNull: bool = True) -> None:
        if not isinstance(elementType, DataType):
            from krishiv.errors import not_expected_type

            raise not_expected_type("elementType", "DataType", elementType)
        self.elementType = elementType
        self.containsNull = containsNull

    def simpleString(self) -> str:
        return f"array<{self.elementType.simpleString()}>"

    def jsonValue(self) -> dict[str, Any]:
        return {
            "type": self.typeName(),
            "elementType": self.elementType.jsonValue(),
            "containsNull": self.containsNull,
        }

    @classmethod
    def fromJson(cls, json: dict[str, Any]) -> ArrayType:
        return cls(_parse_datatype_json_value(json["elementType"]), json["containsNull"])

    def __repr__(self) -> str:
        return f"ArrayType({self.elementType!r}, {self.containsNull!r})"


class MapType(DataType):
    """A map from ``keyType`` to ``valueType``, as ``pyspark.sql.types.MapType``."""

    def __init__(self, keyType: DataType, valueType: DataType, valueContainsNull: bool = True) -> None:
        for name, type_ in (("keyType", keyType), ("valueType", valueType)):
            if not isinstance(type_, DataType):
                from krishiv.errors import not_expected_type

                raise not_expected_type(name, "DataType", type_)
        self.keyType = keyType
        self.valueType = valueType
        self.valueContainsNull = valueContainsNull

    def simpleString(self) -> str:
        return f"map<{self.keyType.simpleString()},{self.valueType.simpleString()}>"

    def jsonValue(self) -> dict[str, Any]:
        return {
            "type": self.typeName(),
            "keyType": self.keyType.jsonValue(),
            "valueType": self.valueType.jsonValue(),
            "valueContainsNull": self.valueContainsNull,
        }

    @classmethod
    def fromJson(cls, json: dict[str, Any]) -> MapType:
        return cls(
            _parse_datatype_json_value(json["keyType"]),
            _parse_datatype_json_value(json["valueType"]),
            json["valueContainsNull"],
        )

    def __repr__(self) -> str:
        return f"MapType({self.keyType!r}, {self.valueType!r}, {self.valueContainsNull!r})"


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
    CalendarIntervalType: "interval",
}
_BY_SIMPLE: dict[str, type[DataType]] = {v: k for k, v in _SIMPLE.items()}
_COLLATED = re.compile(r"string collate (\S+)", re.IGNORECASE)
_DECIMAL = re.compile(r"decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_CHAR = re.compile(r"char\(\s*(\d+)\s*\)", re.IGNORECASE)
_VARCHAR = re.compile(r"varchar\(\s*(\d+)\s*\)", re.IGNORECASE)
_DAYTIME = re.compile(r"interval\s+(day|hour|minute|second)(\s+to\s+(day|hour|minute|second))?", re.IGNORECASE)
_YEARMONTH = re.compile(r"interval\s+(year|month)(\s+to\s+(year|month))?", re.IGNORECASE)


def _from_simple_string(text: str) -> DataType:
    """The type of a Spark ``simpleString`` (as the engine reports it)."""
    if (cls := _BY_SIMPLE.get(text)) is not None:
        return cls()
    if (m := _COLLATED.fullmatch(text)) is not None:
        return StringType(m.group(1))
    if (m := _DECIMAL.fullmatch(text)) is not None:
        return DecimalType(int(m.group(1)), int(m.group(2)))
    if (m := _CHAR.fullmatch(text)) is not None:
        return CharType(int(m.group(1)))
    if (m := _VARCHAR.fullmatch(text)) is not None:
        return VarcharType(int(m.group(1)))
    if (m := _DAYTIME.fullmatch(text)) is not None:
        fields = DayTimeIntervalType._inverted_fields
        first = fields[m.group(1).lower()]
        return (
            DayTimeIntervalType(first) if m.group(3) is None else DayTimeIntervalType(first, fields[m.group(3).lower()])
        )
    if (m := _YEARMONTH.fullmatch(text)) is not None:
        fields = YearMonthIntervalType._inverted_fields
        first = fields[m.group(1).lower()]
        return (
            YearMonthIntervalType(first)
            if m.group(3) is None
            else YearMonthIntervalType(first, fields[m.group(3).lower()])
        )
    from krishiv.errors import PySparkValueError

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
            from krishiv.errors import not_expected_type

            raise not_expected_type("dataType", "DataType", dataType)
        self.name = name
        self.dataType = dataType
        self.nullable = nullable
        self.metadata = metadata or {}

    def simpleString(self) -> str:
        return f"{self.name}:{self.dataType.simpleString()}"

    def jsonValue(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.dataType.jsonValue(),
            "nullable": self.nullable,
            "metadata": self.metadata,
        }

    @classmethod
    def fromJson(cls, json: dict[str, Any]) -> StructField:
        return cls(
            json["name"],
            _parse_datatype_json_value(json["type"]),
            json.get("nullable", True),
            json.get("metadata") or {},
        )

    def typeName(self) -> str:
        from krishiv.errors import PySparkTypeError

        raise PySparkTypeError(errorClass="INVALID_TYPENAME_CALL", messageParameters={})

    def __repr__(self) -> str:
        return f"StructField('{self.name}', {self.dataType!r}, {self.nullable})"

    def __hash__(self) -> int:
        return hash((self.name, self.dataType, self.nullable))


class StructType(DataType):
    def __init__(self, fields: Iterable[StructField] | None = None) -> None:
        self.fields = list(fields or [])
        if not all(isinstance(f, StructField) for f in self.fields):
            raise AssertionError("fields should be a list of StructField")
        self.names = [f.name for f in self.fields]

    @classmethod
    def typeName(cls) -> str:
        return "struct"

    def fieldNames(self) -> list[str]:
        return list(self.names)

    def simpleString(self) -> str:
        return "struct<" + ",".join(f.simpleString() for f in self.fields) + ">"

    def jsonValue(self) -> dict[str, Any]:
        return {"type": self.typeName(), "fields": [f.jsonValue() for f in self.fields]}

    @classmethod
    def fromJson(cls, json: dict[str, Any]) -> StructType:
        return cls([StructField.fromJson(f) for f in json["fields"]])

    def add(
        self,
        field: str | StructField,
        data_type: str | DataType | None = None,
        nullable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> StructType:
        """Adds a field to this struct: either a ``StructField``, or a name
        with a type (a ``DataType`` or, as in ``pyspark.sql.types``, a JSON
        type string such as ``"string"`` or ``"decimal(10,2)"``)."""
        if isinstance(field, StructField):
            self.fields.append(field)
            self.names.append(field.name)
            return self
        if not isinstance(field, str) or data_type is None:
            from krishiv.errors import PySparkValueError

            raise PySparkValueError(
                errorClass="ARGUMENT_REQUIRED",
                messageParameters={"arg_name": "data_type", "condition": "passing name of struct_field to create"},
            )
        if isinstance(data_type, str):
            data_type = _parse_datatype_json_value(data_type)
        if not isinstance(data_type, DataType):
            from krishiv.errors import not_expected_type

            raise not_expected_type("data_type", "DataType or str", data_type)
        self.fields.append(StructField(field, data_type, nullable, metadata))
        self.names.append(field)
        return self

    def __iter__(self) -> Iterator[StructField]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @overload
    def __getitem__(self, key: str | int) -> StructField: ...

    @overload
    def __getitem__(self, key: slice) -> StructType: ...

    def __getitem__(self, key: int | str | slice) -> StructField | StructType:
        from krishiv.errors import PySparkIndexError, PySparkKeyError, PySparkTypeError

        if isinstance(key, str):
            for f in self.fields:
                if f.name == key:
                    return f
            raise PySparkKeyError(f"No StructField named {key}")
        if isinstance(key, int):
            try:
                return self.fields[key]
            except IndexError:
                raise PySparkIndexError(f"StructType index out of range: {key}") from None
        if isinstance(key, slice):
            return StructType(self.fields[key])
        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "key",
                "expected_type": "int, slice or str",
                "arg_type": type(key).__name__,
            },
        )

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
    from krishiv.errors import not_expected_type

    raise not_expected_type("dataType", "DataType or str", data_type)


def _parse_datatype_json_string(json_string: str) -> DataType:
    """The type of a JSON type string, as ``pyspark.sql.types`` reads it."""
    return _parse_datatype_json_value(_json.loads(json_string))


def _parse_datatype_json_value(json_value: dict[str, Any] | str) -> DataType:
    """The type of a JSON type value: an atomic type name (``"integer"``,
    ``"decimal(10,2)"``, ``"char(3)"``, ``"interval day to second"``) or a
    ``{"type": "array"|"map"|"struct", ...}`` document."""
    from krishiv.errors import PySparkValueError

    if not isinstance(json_value, dict):
        # JSON names atomic types by ``typeName`` (``"integer"``, ``"long"``),
        # which differs from ``simpleString`` (``"int"``, ``"bigint"``) for four of them.
        if json_value in ("byte", "short", "integer", "long"):
            return {"byte": ByteType, "short": ShortType, "integer": IntegerType, "long": LongType}[json_value]()
        try:
            return _from_simple_string(json_value)
        except Exception:
            raise PySparkValueError(
                errorClass="CANNOT_PARSE_DATATYPE", messageParameters={"msg": str(json_value)}
            ) from None
    type_name = json_value.get("type")
    if type_name == "array":
        return ArrayType.fromJson(json_value)
    if type_name == "map":
        return MapType.fromJson(json_value)
    if type_name == "struct":
        return StructType.fromJson(json_value)
    raise PySparkValueError(errorClass="UNSUPPORTED_DATA_TYPE", messageParameters={"data_type": str(type_name)})


_DDL_ATOMS = {
    "void": NullType,
    "boolean": BooleanType,
    "tinyint": ByteType,
    "byte": ByteType,
    "smallint": ShortType,
    "short": ShortType,
    "int": IntegerType,
    "integer": IntegerType,
    "bigint": LongType,
    "long": LongType,
    "float": FloatType,
    "real": FloatType,
    "double": DoubleType,
    "string": StringType,
    "binary": BinaryType,
    "date": DateType,
    "timestamp": TimestampType,
    "timestamp_ltz": TimestampType,
    "timestamp_ntz": TimestampNTZType,
    "interval": CalendarIntervalType,
}


def _parse_datatype_string(s: str) -> DataType:
    """The type of a DDL string, as ``pyspark.sql.types`` reads it: either a
    single type (``"array<int>"``) or struct fields (``"a int, b string"``,
    with ``:`` or a space between name and type)."""
    from krishiv.errors import ParseException

    try:
        tokens = _ddl_tokens(s)
    except Exception as e:
        raise ParseException(f"Syntax error at or near '{e}'") from None
    if not tokens:
        raise ParseException("[PARSE_EMPTY_STATEMENT] Empty DDL string")
    parser = _DdlParser(tokens)
    # A field list (``a int, ...``), or a single type. Fields are tried
    # first: a bare type word fails as a field name's type (``interval
    # day ...`` fails as fields and parses as one type).
    try:
        try:
            fields = parser.fields()
            parser.end()
            return StructType(fields)
        except ValueError:
            parser = _DdlParser(tokens)
        data_type = parser.data_type()
        parser.end()
        return data_type
    except Exception as e:
        raise ParseException(f"Syntax error at or near '{e}'") from None


def _ddl_tokens(s: str) -> list[str]:
    tokens: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace() or c == ",":
            if c == ",":
                tokens.append(",")
            i += 1
        elif c in "<>():":
            tokens.append(c)
            i += 1
        elif c == "`":
            j = s.find("`", i + 1)
            if j < 0:
                raise ValueError("unterminated quoted identifier")
            tokens.append("`id:" + s[i + 1 : j].replace("``", "`"))
            i = j + 1
        elif c.isalnum() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            raise ValueError(s[i])
    return tokens


class _DdlParser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("end of input")
        self.pos += 1
        return tok

    def expect(self, tok: str) -> None:
        if self.next() != tok:
            raise ValueError(tok)

    def end(self) -> None:
        if self.peek() is not None:
            raise ValueError(self.peek())

    def field_name(self) -> str:
        tok = self.next()
        if tok.startswith("`id:"):
            return tok[4:]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            return tok
        raise ValueError(tok)

    def fields(self) -> list[StructField]:
        fields: list[StructField] = []
        while True:
            name = self.field_name()
            if self.peek() == ":":
                self.next()
            data_type = self.data_type()
            nullable = True
            if (tok := self.peek()) is not None and tok.upper() in ("NOT", "NULL"):
                self.next()
                if tok.upper() == "NOT":
                    if (tok2 := self.peek()) is None or tok2.upper() != "NULL":
                        raise ValueError(tok2 or "end of input")
                    self.next()
                    nullable = False
            fields.append(StructField(name, data_type, nullable))
            if self.peek() == ",":
                self.next()
                continue
            return fields

    def data_type(self) -> DataType:
        tok = self.next()
        if tok.startswith("`id:"):
            raise ValueError(tok[4:])
        lower = tok.lower()
        if lower in ("array", "map", "struct"):
            return self.nested(lower)
        if lower in ("decimal", "dec", "numeric"):
            return self.decimal()
        if lower in ("char", "character"):
            self.expect("(")
            length = self.number()
            self.expect(")")
            return CharType(length)
        if lower == "varchar":
            self.expect("(")
            length = self.number()
            self.expect(")")
            return VarcharType(length)
        if lower == "interval":
            if self.peek() is None or self.peek() in (",", ">", ")"):
                return CalendarIntervalType()
            return self.interval()
        if lower == "time":
            raise ValueError(tok)
        if lower in _DDL_ATOMS:
            if self.peek() == "(":
                raise ValueError(tok)
            if lower == "string" and (self.peek() or "").lower() == "collate":
                self.next()
                return StringType(self.next())
            return _DDL_ATOMS[lower]()
        raise ValueError(tok)

    def nested(self, kind: str) -> DataType:
        self.expect("<")
        if kind == "array":
            element = self.data_type()
            self.expect(">")
            return ArrayType(element)
        if kind == "map":
            key = self.data_type()
            self.expect(",")
            value = self.data_type()
            self.expect(">")
            return MapType(key, value)
        fields = self.fields()
        self.expect(">")
        return StructType(fields)

    def decimal(self) -> DataType:
        if self.peek() != "(":
            return DecimalType()
        self.next()
        precision = self.number()
        scale = 0
        if self.peek() == ",":
            self.next()
            scale = self.number()
        self.expect(")")
        return DecimalType(precision, scale)

    def interval(self) -> DataType:
        first = self.next().lower()
        following = self.peek()
        if first == "year":
            if following is not None and following.lower() == "to":
                self.next()
                if self.next().lower() != "month":
                    raise ValueError("month")
                return YearMonthIntervalType(0, 1)
            return YearMonthIntervalType(0)
        if first == "month":
            return YearMonthIntervalType(1)
        if first in ("day", "hour", "minute", "second"):
            fields = DayTimeIntervalType._inverted_fields
            if following is not None and following.lower() == "to":
                self.next()
                second = self.next().lower()
                if second not in fields or fields[second] < fields[first]:
                    raise ValueError(second)
                return DayTimeIntervalType(fields[first], fields[second])
            return DayTimeIntervalType(fields[first])
        raise ValueError(first)

    def number(self) -> int:
        tok = self.next()
        if re.fullmatch(r"\d+", tok):
            return int(tok)
        raise ValueError(tok)
