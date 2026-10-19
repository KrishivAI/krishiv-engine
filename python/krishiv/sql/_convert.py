# SPDX-License-Identifier: Apache-2.0
"""Arrow results as Python values, and the text of ``DataFrame.show``.

pyarrow is imported lazily: the package works without it, and the actions that need it say so.
"""

from __future__ import annotations

import math
import re
import struct
import unicodedata
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from typing import Any

from krishiv.sql.row import Row, _create_row


def require_pyarrow(action: str) -> Any:
    """The ``pyarrow`` module, or an ``ImportError`` saying what to install for ``action``."""
    try:
        import pyarrow
    except ImportError as e:
        raise ImportError(f"{action} needs pyarrow; install it with `pip install 'krishiv[pyarrow]'`") from e
    return pyarrow


def rows_of(table: Any) -> list[Row]:
    """The rows of a pyarrow ``Table`` as ``Row``s, with Spark's Python types (structs as ``Row``, maps as ``dict``)."""
    names = list(table.schema.names)
    columns = [_column_values(table.column(i)) for i in range(table.num_columns)]
    if not names:
        return [Row() for _ in range(table.num_rows)]  # no columns: one empty Row per row, as PySpark
    return [_create_row(names, list(values)) for values in zip(*columns, strict=True)]


def _column_values(column: Any) -> list[Any]:
    return [_value(v, column.type) for v in column.to_pylist()]


def _value(value: Any, arrow_type: Any) -> Any:
    """``value`` (as pyarrow's ``to_pylist`` gives it) as PySpark gives it."""
    if value is None:
        return None
    import pyarrow as pa

    if pa.types.is_timestamp(arrow_type) and arrow_type.tz is not None:
        # An instant: PySpark returns it as a naive datetime in the client's local time zone.
        return value.astimezone().replace(tzinfo=None)
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
    return str(value)


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
        header = [_truncate(c, limit) for c in names]
        widths = [max([3, _width(h)] + [_width(r[i]) for r in body]) for i, h in enumerate(header)]
        rule = "+" + "+".join("-" * w for w in widths) + "+"
        right = limit > 0

        def line(items: list[str]) -> str:
            return "|" + "|".join(_pad(t, w, right) for t, w in zip(items, widths, strict=True)) + "|"

        lines = [rule, line(header), rule, *[line(r) for r in body], rule]
    if len(rows) > n:
        lines.append(f"only showing top {n} row{'s' if n != 1 else ''}")
    return "\n".join(lines) + "\n"
