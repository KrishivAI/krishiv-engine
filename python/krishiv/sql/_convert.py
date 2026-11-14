# SPDX-License-Identifier: Apache-2.0
"""Arrow results as Python values, and the text of ``DataFrame.show``.

pyarrow is imported lazily: the package works without it, and the actions that need it say so.
"""

from __future__ import annotations

import html
import math
import re
import struct
import unicodedata
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from typing import Any

from krishiv.sql.row import Row, _create_row
from krishiv.sql.types import (
    ArrayType,
    CalendarIntervalType,
    DataType,
    MapType,
    StructType,
    YearMonthIntervalType,
)


def require_pyarrow(action: str) -> Any:
    """The ``pyarrow`` module, or an ``ImportError`` saying what to install for ``action``."""
    try:
        import pyarrow
    except ImportError as e:
        raise ImportError(f"{action} needs pyarrow; install it with `pip install 'krishiv[pyarrow]'`") from e
    return pyarrow


def _leaves(t: DataType) -> Iterator[DataType]:
    """The leaf types of ``t``, depth first."""
    if isinstance(t, ArrayType):
        yield from _leaves(t.elementType)
    elif isinstance(t, MapType):
        yield from _leaves(t.keyType)
        yield from _leaves(t.valueType)
    elif isinstance(t, StructType):
        for f in t.fields:
            yield from _leaves(f.dataType)
    else:
        yield t


def not_collectable(t: DataType) -> DataType | None:
    """The first year-month or calendar interval in ``t``, which PySpark's Connect client won't bring back."""
    return next((x for x in _leaves(t) if isinstance(x, YearMonthIntervalType | CalendarIntervalType)), None)


def not_collectable_arrow(arrow_type: Any) -> str | None:
    """The name of the first year-month or calendar interval in an Arrow type."""
    import pyarrow as pa

    if pa.types.is_struct(arrow_type):
        children = [arrow_type.field(i).type for i in range(arrow_type.num_fields)]
    elif pa.types.is_map(arrow_type):
        children = [arrow_type.key_type, arrow_type.item_type]
    elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        children = [arrow_type.value_type]
    else:
        name = str(arrow_type)
        return name if name in ("month_interval", "month_day_nano_interval") else None
    return next((n for n in map(not_collectable_arrow, children) if n is not None), None)


_HOLDS_INTERVAL = re.compile(r"\binterval\b(?!:)")


def holds_interval(simple_type: str) -> bool:
    """Whether a ``simpleString`` type is or holds an interval of any family (a field may be named ``interval``)."""
    return _HOLDS_INTERVAL.search(simple_type) is not None


def rows_of(table: Any) -> list[Row]:
    """The rows of a pyarrow ``Table`` as ``Row``s, with Spark's Python types (structs as ``Row``, maps as ``dict``)."""
    names = list(table.schema.names)
    columns = [_column_values(table.column(i)) for i in range(table.num_columns)]
    if not names:
        return [Row() for _ in range(table.num_rows)]  # no columns: one empty Row per row, as PySpark
    return [_create_row(names, list(values)) for values in zip(*columns, strict=True)]


def _column_values(column: Any) -> list[Any]:
    if _has_duplicate_struct_fields(column.type):
        # ``to_pylist`` raises on a struct with duplicate field names: walk the arrays by position instead.
        return _by_position(column.combine_chunks() if hasattr(column, "combine_chunks") else column, column.type)
    return [_value(v, column.type) for v in column.to_pylist()]


def _has_duplicate_struct_fields(arrow_type: Any) -> bool:
    import pyarrow as pa

    if pa.types.is_struct(arrow_type):
        names = [arrow_type.field(i).name for i in range(arrow_type.num_fields)]
        return len(set(names)) != len(names) or any(
            _has_duplicate_struct_fields(arrow_type.field(i).type) for i in range(arrow_type.num_fields)
        )
    if pa.types.is_map(arrow_type):
        return _has_duplicate_struct_fields(arrow_type.key_type) or _has_duplicate_struct_fields(arrow_type.item_type)
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        return _has_duplicate_struct_fields(arrow_type.value_type)
    return False


def _by_position(array: Any, arrow_type: Any) -> list[Any]:
    """``array`` as PySpark Python values, struct fields read by position (so duplicate names survive)."""
    import pyarrow as pa

    if not _has_duplicate_struct_fields(arrow_type):
        return [_value(v, arrow_type) for v in array.to_pylist()]
    if pa.types.is_struct(arrow_type):
        fields = [arrow_type.field(i) for i in range(arrow_type.num_fields)]
        children = [_by_position(array.field(i), f.type) for i, f in enumerate(fields)]
        names = [f.name for f in fields]
        valid = array.is_valid().to_pylist()
        return [_create_row(names, [c[j] for c in children]) if ok else None for j, ok in enumerate(valid)]
    if pa.types.is_map(arrow_type):
        out: list[Any] = []
        for scalar in array:
            if not scalar.is_valid:
                out.append(None)
                continue
            entries = scalar.values
            keys = _by_position(entries.field(0), arrow_type.key_type)
            vals = _by_position(entries.field(1), arrow_type.item_type)
            out.append(dict(zip(keys, vals, strict=True)))
        return out
    return [None if not scalar.is_valid else _by_position(scalar.values, arrow_type.value_type) for scalar in array]


def _value(value: Any, arrow_type: Any) -> Any:
    """``value`` (as pyarrow's ``to_pylist`` gives it) as PySpark gives it."""
    if value is None:
        return None
    import pyarrow as pa

    if pa.types.is_timestamp(arrow_type) and arrow_type.tz is not None:
        # An instant: PySpark returns it as a naive datetime in the client's local time zone.
        return value.astimezone().replace(tzinfo=None)
    from krishiv.sql.variant import VariantVal, is_variant_struct

    if is_variant_struct(arrow_type):
        return VariantVal(bytes(value["value"]), bytes(value["metadata"]))
    if pa.types.is_struct(arrow_type):
        fields = [arrow_type.field(i) for i in range(arrow_type.num_fields)]
        return _create_row([f.name for f in fields], [_value(value[f.name], f.type) for f in fields])
    if pa.types.is_map(arrow_type):
        return {_value(k, arrow_type.key_type): _value(v, arrow_type.item_type) for k, v in value}
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        return [_value(v, arrow_type.value_type) for v in value]
    return value


# -- show ---------------------------------------------------------------------------------------


_OFFSET = re.compile(r"(?:UTC|GMT|UT)?([+-])(\d{1,2})(?::?(\d{2}))?(?::?(\d{2}))?")


def session_zone(name: str) -> tzinfo:
    """The time zone named by ``spark.sql.session.timeZone``: a region ID, ``Z``/``UTC``/``GMT``/``UT``, or an
    offset (``+h``, ``+hh:mm``, ``+hhmmss``, optionally prefixed ``UTC``, ``GMT`` or ``UT``), as Java's ``ZoneId``."""
    if name in ("Z", "UTC", "GMT", "UT"):
        return UTC
    match = _OFFSET.fullmatch(name)
    if match:
        sign, hours, minutes, seconds = match.groups()
        offset = timedelta(hours=int(hours), minutes=int(minutes or 0), seconds=int(seconds or 0))
        return timezone(-offset if sign == "-" else offset)
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


def cell(value: Any, arrow_type: Any, zone: tzinfo = UTC) -> str:
    """One value (as pyarrow's ``to_pylist`` gives it, of ``arrow_type``) as ``Dataset.show`` prints it;
    instants (timestamps with a time zone) in ``zone``, the session time zone."""
    if value is None:
        return "NULL"
    import pyarrow as pa

    from krishiv.sql.variant import _json, _Metadata, is_variant_struct

    if is_variant_struct(arrow_type):
        out: list[str] = []
        _json(bytes(value["value"]), _Metadata(bytes(value["metadata"])), 0, zone, out)
        return "".join(out)
    if pa.types.is_struct(arrow_type):
        fields = [arrow_type.field(i) for i in range(arrow_type.num_fields)]
        return "{" + ", ".join(cell(value[f.name], f.type, zone) for f in fields) + "}"
    if pa.types.is_map(arrow_type):
        pairs = (f"{cell(k, arrow_type.key_type, zone)} -> {cell(v, arrow_type.item_type, zone)}" for k, v in value)
        return "{" + ", ".join(pairs) + "}"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        return "[" + ", ".join(cell(v, arrow_type.value_type, zone) for v in value) + "]"
    if pa.types.is_timestamp(arrow_type) and arrow_type.tz is not None:
        value = value.astimezone(zone)
    if pa.types.is_float32(arrow_type):
        return _java_double(value, single=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _java_double(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        text = value.strftime("%Y-%m-%d %H:%M:%S")
        return text + f".{value.microsecond:06d}".rstrip("0") if value.microsecond else text
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes | bytearray):
        return "[" + " ".join(f"{b:02X}" for b in value) + "]"
    if isinstance(value, timedelta):
        return _day_time_cell(value)
    return str(value)


def _day_time_cell(value: timedelta) -> str:
    """A day-time interval as ``Dataset.show`` prints it, in its default fields (Arrow carries no others)."""
    micros = abs(value // timedelta(microseconds=1))
    sign = "-" if value < timedelta(0) else ""
    seconds, micros = divmod(micros, 1_000_000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    fraction = f".{micros:06d}".rstrip("0") if micros else ""
    return f"INTERVAL '{sign}{days} {hours:02d}:{minutes:02d}:{seconds:02d}{fraction}' DAY TO SECOND"


def _shortest_single(value: float) -> str:
    """The shortest decimal text that reads back as the same 32-bit float (Java's ``Float.toString`` digits)."""
    for digits in range(1, 10):
        text = f"{value:.{digits - 1}e}"
        if struct.unpack("f", struct.pack("f", float(text)))[0] == value:
            return text
    return f"{value:.8e}"


def _java_double(value: float, single: bool = False) -> str:
    """``java.lang.Double.toString``: plain for 1e-3 <= |v| < 1e7, else ``d.dddE±n``."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    if 1e-3 <= abs(value) < 1e7:
        plain = format(Decimal(_shortest_single(value) if single else repr(value)), "f")
        return plain if "." in plain else plain + ".0"
    text = _shortest_single(abs(value)) if single else repr(abs(value))
    _, digits, exp = Decimal(text).as_tuple()
    digits_text = "".join(map(str, digits)).rstrip("0") or "0"
    power = len(digits) + int(exp) - 1
    return f"{'-' if value < 0 else ''}{digits_text[0]}.{digits_text[1:] or '0'}E{power}"


def _width(text: str) -> int:
    """Display width: East Asian wide and fullwidth characters take two columns, as in Spark."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int, right_align: bool) -> str:
    fill = " " * (width - _width(text))
    return fill + text if right_align else text + fill


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] if limit < 4 else text[: limit - 3] + "..."


def table_cells(table: Any, zone: tzinfo = UTC) -> list[list[str]]:
    """The cells of a pyarrow ``Table``, as ``show`` prints them, one list per row."""
    columns = [[cell(v, col.type, zone) for v in col.to_pylist()] for col in table.columns]
    if not columns:
        return [[] for _ in range(table.num_rows)]
    return [list(row) for row in zip(*columns, strict=True)]


def show_string(names: list[str], rows: list[list[str]], n: int, truncate: bool | int, vertical: bool) -> str:
    """The text ``DataFrame.show`` prints (``rows`` holds up to ``n + 1`` rows of cell text)."""
    limit = 20 if truncate is True else 0 if truncate is False else int(truncate)
    shown = rows[:n]
    body = [[_truncate(v, limit) for v in row] for row in shown]
    lines: list[str] = []
    if vertical:
        name_width = max([3] + [_width(c) for c in names])
        value_width = max([3] + [_width(v) for row in body for v in row])
        for i, row in enumerate(body):
            lines.append(f"-RECORD {i}".ljust(name_width + value_width + 5, "-"))
            lines.extend(
                f" {_pad(c, name_width, False)} | {_pad(v, value_width, False)} "
                for c, v in zip(names, row, strict=True)
            )
        if not body:
            lines.append("(0 rows)")
    else:
        header = list(names)  # Spark truncates cells, never the header
        widths = [max([3, _width(h)] + [_width(r[i]) for r in body]) for i, h in enumerate(header)]
        rule = "+" + "+".join("-" * w for w in widths) + "+"
        right = limit > 0

        def line(items: list[str]) -> str:
            return "|" + "|".join(_pad(t, w, right) for t, w in zip(items, widths, strict=True)) + "|"

        lines = [rule, line(header), rule, *[line(r) for r in body], rule]
    if len(rows) > n:
        lines.append(f"only showing top {n} row{'s' if n != 1 else ''}")
    return "\n".join(lines) + "\n"


def repr_html_string(columns: list[tuple[str, str]], rows: list[list[str]]) -> str:
    """The HTML table ``DataFrame._repr_html_`` returns: one header row of column names, one of
    their types, then one row per preview row. Everything is HTML-escaped, so values cannot
    break the table (``rows`` holds the display text of each cell, as :func:`table_cells` gives)."""
    head = "".join(f"<th>{html.escape(name)}</th>" for name, _ in columns)
    types = "".join(f"<th>{html.escape(type_)}</th>" for _, type_ in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>" for row in rows)
    return (
        '<table border="1" class="dataframe">'
        f"<thead><tr>{head}</tr><tr>{types}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def _arrow_zone_name(zone: tzinfo) -> str:
    """``zone`` as a time zone name Arrow accepts: a region ID or a ``+hh:mm`` offset."""
    key = getattr(zone, "key", None)
    if isinstance(key, str):
        return key
    offset = zone.utcoffset(None)
    if offset is None or offset == timedelta(0):
        return "UTC"
    minutes = int(offset.total_seconds() // 60)
    sign = "-" if minutes < 0 else "+"
    return f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def session_local(table: Any, zone: tzinfo) -> Any:
    """``table`` with its instants (timestamps with a time zone) as the wall-clock time of ``zone``, without a time
    zone: what Spark's ``toPandas`` returns. Timestamps without a time zone are unchanged."""
    import pyarrow as pa
    import pyarrow.compute as pc

    name = _arrow_zone_name(zone)
    columns = []
    for column in table.columns:
        if pa.types.is_timestamp(column.type) and column.type.tz is not None:
            column = pc.call_function("local_timestamp", [column.cast(pa.timestamp(column.type.unit, name))])
        columns.append(column)
    return pa.table(columns, names=table.schema.names)
