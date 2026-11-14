# SPDX-License-Identifier: Apache-2.0
"""``createDataFrame`` over Python rows: the Arrow type of a Spark type, the check of a value
against a type (``verifySchema``, as ``pyspark.sql.types._make_type_verifier``) and the
conversion of a value to what pyarrow takes, for flat and nested (array, map, struct) types.

A struct value is a ``Row`` or tuple (read by position) or a ``dict`` (read by field name); a map
is a ``dict``; an array a ``list``, ``tuple`` or ``array.array``. A ``string`` column takes any
value (``convert_string``: its ``str``); a ``decimal`` is rounded half up to the scale.
"""

from __future__ import annotations

import array
import datetime
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import Any

from krishiv.errors import (
    ArithmeticException,
    PySparkTypeError,
    PySparkValueError,
    unsupported_feature,
)
from krishiv.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DataType,
    DateType,
    DayTimeIntervalType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    ShortType,
    StringType,
    StructType,
    TimestampNTZType,
    TimestampType,
)

_INT_BOUNDS: dict[Any, tuple[int, int]] = {
    ByteType: (-(2**7), 2**7 - 1),
    ShortType: (-(2**15), 2**15 - 1),
    IntegerType: (-(2**31), 2**31 - 1),
    LongType: (-(2**63), 2**63 - 1),
}


def pa_type(data_type: DataType, pa: Any) -> Any:
    """The Arrow type holding ``data_type`` values when ``createDataFrame`` converts rows."""
    if isinstance(data_type, NullType):
        return pa.null()
    if isinstance(data_type, BooleanType):
        return pa.bool_()
    if isinstance(data_type, ByteType):
        return pa.int8()
    if isinstance(data_type, ShortType):
        return pa.int16()
    if isinstance(data_type, IntegerType):
        return pa.int32()
    if isinstance(data_type, LongType):
        return pa.int64()
    if isinstance(data_type, FloatType):
        return pa.float32()
    if isinstance(data_type, DoubleType):
        return pa.float64()
    if isinstance(data_type, StringType):
        if data_type._collated():
            raise unsupported_feature(f"createDataFrame with {data_type.simpleString()} columns")
        return pa.string()
    if isinstance(data_type, BinaryType):
        return pa.binary()
    if isinstance(data_type, DecimalType):
        return pa.decimal128(data_type.precision, data_type.scale)
    if isinstance(data_type, DateType):
        return pa.date32()
    if isinstance(data_type, TimestampType):
        return pa.timestamp("us", tz="UTC")
    if isinstance(data_type, TimestampNTZType):
        return pa.timestamp("us")
    if isinstance(data_type, DayTimeIntervalType):
        return pa.duration("us")
    if isinstance(data_type, ArrayType):
        return pa.list_(pa.field("element", pa_type(data_type.elementType, pa), nullable=data_type.containsNull))
    if isinstance(data_type, MapType):
        return pa.map_(
            pa.field("key", pa_type(data_type.keyType, pa), nullable=False),
            pa.field("value", pa_type(data_type.valueType, pa), nullable=data_type.valueContainsNull),
        )
    if isinstance(data_type, StructType):
        return pa.struct([pa.field(f.name, pa_type(f.dataType, pa), nullable=f.nullable) for f in data_type.fields])
    raise unsupported_feature(f"createDataFrame with {data_type.simpleString()} columns")


def pa_schema(struct: StructType, pa: Any) -> Any:
    """The Arrow schema of ``struct``."""
    return pa.schema([pa.field(f.name, pa_type(f.dataType, pa), nullable=f.nullable) for f in struct.fields])


def _is_sequence(value: Any) -> bool:
    return isinstance(value, list | tuple | array.array)


def _struct_values(value: Any, struct: StructType) -> list[Any] | None:
    """The values of one struct value in field order, or ``None`` when ``value`` is not one."""
    if isinstance(value, dict):
        return [value.get(f.name) for f in struct.fields]
    if isinstance(value, list | tuple):
        return list(value)
    return None


def unacceptable(name: str | None, data_type: DataType, value: Any) -> PySparkTypeError:
    params = {"data_type": repr(data_type), "obj": repr(value), "obj_type": str(type(value))}
    if name is None:
        return PySparkTypeError(errorClass="FIELD_DATA_TYPE_UNACCEPTABLE", messageParameters=params)
    return PySparkTypeError(
        errorClass="FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME", messageParameters={"field_name": name, **params}
    )


def verify(value: Any, data_type: DataType, nullable: bool = True, name: str | None = None) -> None:
    """Checks ``value`` against ``data_type``, nested values included, and raises as PySpark's
    classic ``createDataFrame`` verification does."""
    if value is None:
        if not nullable:
            if name is None:
                raise PySparkValueError(errorClass="FIELD_NOT_NULLABLE")
            raise PySparkValueError(errorClass="FIELD_NOT_NULLABLE_WITH_NAME", messageParameters={"field_name": name})
        return
    if isinstance(data_type, NullType | StringType):
        return
    if isinstance(data_type, BooleanType):
        ok = isinstance(value, bool)
    elif isinstance(data_type, ByteType | ShortType | IntegerType | LongType):
        ok = isinstance(value, int) and not isinstance(value, bool)
        if ok:
            low, high = _INT_BOUNDS[type(data_type)]
            if not low <= value <= high:
                raise PySparkValueError(
                    errorClass="VALUE_OUT_OF_BOUNDS",
                    messageParameters={
                        "arg_name": "obj",
                        "lower_bound": str(low),
                        "upper_bound": str(high),
                        "actual": str(value),
                    },
                )
    elif isinstance(data_type, FloatType | DoubleType):
        ok = isinstance(value, float)
    elif isinstance(data_type, DecimalType):
        ok = isinstance(value, Decimal)
    elif isinstance(data_type, BinaryType):
        ok = isinstance(value, bytes | bytearray)
    elif isinstance(data_type, DateType):
        ok = isinstance(value, datetime.date)
    elif isinstance(data_type, TimestampType | TimestampNTZType):
        ok = isinstance(value, datetime.datetime)
    elif isinstance(data_type, DayTimeIntervalType):
        ok = isinstance(value, datetime.timedelta)
    elif isinstance(data_type, ArrayType):
        if not _is_sequence(value):
            raise unacceptable(name, data_type, value)
        element_name = f"element in array {name}" if name else "element in array"
        for item in value:
            verify(item, data_type.elementType, data_type.containsNull, element_name)
        return
    elif isinstance(data_type, MapType):
        if not isinstance(value, dict):
            raise unacceptable(name, data_type, value)
        key_name = f"key of map {name}" if name else "key of map"
        value_name = f"value of map {name}" if name else "value of map"
        for k, v in value.items():
            verify(k, data_type.keyType, False, key_name)
            verify(v, data_type.valueType, data_type.valueContainsNull, value_name)
        return
    elif isinstance(data_type, StructType):
        values = _struct_values(value, data_type)
        if values is None:
            raise unacceptable(name, data_type, value)
        if not isinstance(value, dict) and len(values) != len(data_type.fields):
            params = {"object_length": str(len(values)), "field_length": str(len(data_type.fields))}
            if name is None:
                raise PySparkValueError(errorClass="FIELD_STRUCT_LENGTH_MISMATCH", messageParameters=params)
            raise PySparkValueError(
                errorClass="FIELD_STRUCT_LENGTH_MISMATCH_WITH_NAME", messageParameters={"field_name": name, **params}
            )
        for field, item in zip(data_type.fields, values, strict=True):
            verify(
                item, field.dataType, field.nullable, f"field {field.name} in {name}" if name else f"field {field.name}"
            )
        return
    else:
        ok = True  # a type this layer does not model: the Arrow conversion reports it
    if not ok:
        raise unacceptable(name, data_type, value)


_WIDE = Context(prec=200)


def _decimal(value: Decimal, data_type: DecimalType) -> Decimal:
    """``value`` rounded half up to the scale, or ``NUMERIC_VALUE_OUT_OF_RANGE`` when its
    integer digits do not fit the precision."""
    if value.is_nan():
        return value
    step = Decimal(1).scaleb(-data_type.scale)
    rounded = value.quantize(step, rounding=ROUND_HALF_UP, context=_WIDE) if value.is_finite() else value
    if not rounded.is_finite() or len(rounded.as_tuple().digits) > data_type.precision:
        raise ArithmeticException(
            errorClass="NUMERIC_VALUE_OUT_OF_RANGE.WITHOUT_SUGGESTION",
            messageParameters={
                "roundedValue": format(rounded, "f"),
                "originalValue": format(value, "f"),
                "precision": str(data_type.precision),
                "scale": str(data_type.scale),
            },
            sqlState="22003",
        )
    return rounded


def convert(value: Any, data_type: DataType, zone: Any) -> Any:
    """One value as pyarrow takes it for ``pa_type(data_type)``.

    ``zone`` gives the session time zone (called only for a naive ``timestamp_ntz`` of an aware
    datetime). A naive datetime for a ``timestamp`` column is read in the Python process's local
    time zone, as PySpark reads it."""
    if value is None:
        return None
    if isinstance(data_type, ArrayType):
        return [convert(v, data_type.elementType, zone) for v in value]
    if isinstance(data_type, MapType):
        return [(convert(k, data_type.keyType, zone), convert(v, data_type.valueType, zone)) for k, v in value.items()]
    if isinstance(data_type, StructType):
        values = _struct_values(value, data_type)
        if values is None:
            return value
        if not isinstance(value, dict) and len(values) != len(data_type.fields):
            raise PySparkValueError(
                errorClass="STRUCT_ARRAY_LENGTH_MISMATCH",
                messageParameters={"expected": str(len(data_type.fields)), "actual": str(len(values))},
                sqlState="2201E",
            )
        return {f.name: convert(v, f.dataType, zone) for f, v in zip(data_type.fields, values, strict=True)}
    if isinstance(data_type, StringType):
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)
    if isinstance(data_type, DecimalType):
        return _decimal(value, data_type) if isinstance(value, Decimal) else value
    if isinstance(data_type, DateType):
        if isinstance(value, datetime.datetime):
            return value.date()
        return value
    if isinstance(value, datetime.datetime):
        if isinstance(data_type, TimestampType):
            if value.tzinfo is None:
                return value.astimezone(datetime.UTC)
            return value
        if isinstance(data_type, TimestampNTZType):
            return value.astimezone(zone()).replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, array.array):
        return list(value)
    return value


def _arrow_unsupported(arrow_type: Any) -> PySparkTypeError:
    return PySparkTypeError(
        errorClass="UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION", messageParameters={"data_type": str(arrow_type)}
    )


def normalize_arrow_type(t: Any, pa: Any) -> Any:
    """The Arrow type the engine holds for the Arrow type ``t`` of a pyarrow or pandas column:
    large and fixed-size lists, large strings and binaries, dictionaries, 256-bit decimals and
    nanosecond timestamps map to their plain forms; types Spark has no counterpart for
    (unsigned integers, ``date64``, ``float16``, ``time``) are refused as PySpark does."""
    types = pa.types
    if types.is_large_string(t):
        return pa.string()
    if types.is_large_binary(t):
        return pa.binary()
    if types.is_dictionary(t):
        return normalize_arrow_type(t.value_type, pa)
    if types.is_map(t):
        return pa.map_(
            pa.field("key", normalize_arrow_type(t.key_type, pa), nullable=False),
            pa.field("value", normalize_arrow_type(t.item_type, pa), nullable=True),
        )
    if types.is_list(t) or types.is_large_list(t) or types.is_fixed_size_list(t):
        return pa.list_(pa.field("element", normalize_arrow_type(t.value_type, pa), nullable=True))
    if types.is_struct(t):
        return pa.struct([pa.field(f.name, normalize_arrow_type(f.type, pa), nullable=True) for f in t])
    if types.is_decimal(t):
        return pa.decimal128(t.precision, t.scale)
    if types.is_timestamp(t):
        return pa.timestamp("us", tz=t.tz)
    if types.is_duration(t):
        return pa.duration("us")
    if types.is_time(t):
        raise PySparkValueError(errorClass="UNSUPPORTED_TIME_TYPE", sqlState="0A000")
    if types.is_date64(t) or types.is_unsigned_integer(t) or types.is_float16(t):
        raise _arrow_unsupported(t)
    return t


def normalize_arrow_table(table: Any, pa: Any, zone: str | None = None) -> Any:
    """``table`` cast column by column to the types of :func:`normalize_arrow_type` (all nullable).

    A column of naive timestamps is a ``timestamp`` (local time, as Spark reads it) when ``zone``
    names the session time zone, else a ``timestamp_ntz``."""
    import pyarrow.compute as pc

    columns = []
    for column in table.columns:
        if zone is not None and pa.types.is_timestamp(column.type) and column.type.tz is None:
            # A time in a DST gap moves forward and an ambiguous one is the earlier, as java.time reads it.
            column = pc.assume_timezone(  # ty: ignore[unresolved-attribute]
                column, zone, ambiguous="earliest", nonexistent="latest"
            )
        columns.append(column)
    table = pa.table(columns, names=table.column_names)
    schema = pa.schema([pa.field(f.name, normalize_arrow_type(f.type, pa), nullable=True) for f in table.schema])
    return table if table.schema.equals(schema) else table.cast(schema)


def arrow_array(values: list[Any], arrow_type: Any, pa: Any, null_on_error: bool) -> Any:
    """``pa.array(values, type=arrow_type)``; with ``null_on_error`` (``verifySchema=False``) a value
    pyarrow cannot take is NULL, as the JVM conversion of classic PySpark leaves it."""
    try:
        return pa.array(values, type=arrow_type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, OverflowError, ValueError, TypeError):
        if not null_on_error:
            raise
    cells = []
    for v in values:
        try:
            cells.append(pa.array([v], type=arrow_type)[0].as_py() if v is not None else None)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, OverflowError, ValueError, TypeError):
            cells.append(None)
    return pa.array(cells, type=arrow_type)
