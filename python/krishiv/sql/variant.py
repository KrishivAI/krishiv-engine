# SPDX-License-Identifier: Apache-2.0
"""``VariantVal``: a ``VARIANT`` value as PySpark hands it out (``pyspark.sql.types.VariantVal``).

The value and metadata bytes are the Parquet Variant v1 encoding the engine stores. ``toJson`` prints the value as
Spark's ``Variant.toJson`` does (object keys in stored order, strings escaped as Jackson does, doubles and floats as
Java prints them, decimals plain without trailing zeros, timestamps in the given zone with a ``+HH:MM`` offset, binary
as base64); ``toPython`` gives dicts, lists, ``Decimal``, ``date``, ``datetime`` (UTC for a timestamp, naive for
``timestamp_ntz``), ``bytes`` and ``uuid.UUID``.
"""

from __future__ import annotations

import base64
import struct
import uuid
from datetime import UTC, date, datetime, timedelta, tzinfo
from decimal import Decimal
from typing import Any

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_EPOCH_NTZ = datetime(1970, 1, 1)


def _int(b: bytes, pos: int, size: int, signed: bool = False) -> int:
    return int.from_bytes(b[pos : pos + size], "little", signed=signed)


class _Metadata:
    def __init__(self, metadata: bytes) -> None:
        header = metadata[0]
        if header & 0x0F != 1:
            raise ValueError("MALFORMED_VARIANT: unsupported metadata version")
        self.size = ((header >> 6) & 3) + 1
        self.count = _int(metadata, 1, self.size)
        self.offsets = 1 + self.size
        self.strings = self.offsets + (self.count + 1) * self.size
        self.bytes = metadata

    def key(self, i: int) -> str:
        start = _int(self.bytes, self.offsets + i * self.size, self.size)
        end = _int(self.bytes, self.offsets + (i + 1) * self.size, self.size)
        return self.bytes[self.strings + start : self.strings + end].decode("utf-8")


def _container(value: bytes, pos: int, is_object: bool) -> tuple[int, int, int, int, int]:
    """(count, id size, offset size, first id or offset position, data start) of the object or array at ``pos``."""
    info = value[pos] >> 2
    offset_size = (info & 3) + 1
    if is_object:
        id_size = ((info >> 2) & 3) + 1
        large = (info >> 4) & 1
    else:
        id_size = 0
        large = (info >> 2) & 1
    count_size = 4 if large else 1
    count = _int(value, pos + 1, count_size)
    ids = pos + 1 + count_size
    offsets = ids + count * id_size
    data = offsets + (count + 1) * offset_size
    return count, id_size, offset_size, ids, data


def _fields(value: bytes, meta: _Metadata, pos: int) -> list[tuple[str, int]]:
    count, id_size, offset_size, ids, data = _container(value, pos, True)
    offsets = ids + count * id_size
    return [
        (meta.key(_int(value, ids + i * id_size, id_size)), data + _int(value, offsets + i * offset_size, offset_size))
        for i in range(count)
    ]


def _elements(value: bytes, pos: int) -> list[int]:
    count, _, offset_size, offsets, data = _container(value, pos, False)
    return [data + _int(value, offsets + i * offset_size, offset_size) for i in range(count)]


def _decimal(value: bytes, pos: int, width: int) -> tuple[int, int]:
    return _int(value, pos + 2, width, signed=True), value[pos + 1]


def _plain(unscaled: int, scale: int) -> str:
    while scale > 0 and unscaled % 10 == 0:
        unscaled //= 10
        scale -= 1
    text = str(abs(unscaled))
    if scale:
        text = text.rjust(scale + 1, "0")
        text = text[:-scale] + "." + text[-scale:]
    return ("-" if unscaled < 0 else "") + text


def _quote(text: str) -> str:
    """Jackson's ``writeString``."""
    out = ['"']
    for c in text:
        if c == '"':
            out.append('\\"')
        elif c == "\\":
            out.append("\\\\")
        elif c in "\b\f\n\r\t":
            out.append({"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}[c])
        elif ord(c) < 0x20:
            out.append(f"\\u{ord(c):04X}")
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


def _local(micros: int) -> str:
    when = _EPOCH_NTZ + timedelta(microseconds=micros)
    text = when.strftime("%Y-%m-%d %H:%M:%S")
    return text + f".{when.microsecond:06d}".rstrip("0") if when.microsecond else text


def _json(value: bytes, meta: _Metadata, pos: int, zone: tzinfo, out: list[str]) -> None:
    from krishiv.sql._convert import _java_double

    header = value[pos]
    basic, info = header & 3, header >> 2
    if basic == 1:
        out.append(_quote(value[pos + 1 : pos + 1 + info].decode("utf-8")))
    elif basic == 2:
        out.append("{")
        for i, (key, at) in enumerate(_fields(value, meta, pos)):
            out.append(("," if i else "") + _quote(key) + ":")
            _json(value, meta, at, zone, out)
        out.append("}")
    elif basic == 3:
        out.append("[")
        for i, at in enumerate(_elements(value, pos)):
            if i:
                out.append(",")
            _json(value, meta, at, zone, out)
        out.append("]")
    elif info == 0:
        out.append("null")
    elif info in (1, 2):
        out.append("true" if info == 1 else "false")
    elif info in (3, 4, 5, 6):
        out.append(str(_int(value, pos + 1, 1 << (info - 3), signed=True)))
    elif info in (7, 14):
        width, code = (8, "<d") if info == 7 else (4, "<f")
        d = struct.unpack(code, value[pos + 1 : pos + 1 + width])[0]
        text = _java_double(d, single=info == 14)
        out.append(text if d == d and abs(d) != float("inf") else _quote(text))
    elif info in (8, 9, 10):
        out.append(_plain(*_decimal(value, pos, {8: 4, 9: 8, 10: 16}[info])))
    elif info == 11:
        out.append(_quote((date(1970, 1, 1) + timedelta(days=_int(value, pos + 1, 4, signed=True))).isoformat()))
    elif info == 12:
        when = (_EPOCH + timedelta(microseconds=_int(value, pos + 1, 8, signed=True))).astimezone(zone)
        offset = when.utcoffset() or timedelta(0)
        minutes = int(offset.total_seconds()) // 60
        sign = "-" if minutes < 0 else "+"
        local = _local(int((when.replace(tzinfo=None) - _EPOCH_NTZ) / timedelta(microseconds=1)))
        out.append(_quote(f"{local}{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"))
    elif info == 13:
        out.append(_quote(_local(_int(value, pos + 1, 8, signed=True))))
    elif info == 15:
        n = _int(value, pos + 1, 4)
        out.append(_quote(base64.b64encode(value[pos + 5 : pos + 5 + n]).decode("ascii")))
    elif info == 16:
        n = _int(value, pos + 1, 4)
        out.append(_quote(value[pos + 5 : pos + 5 + n].decode("utf-8")))
    elif info == 20:
        out.append(_quote(str(uuid.UUID(bytes=bytes(value[pos + 1 : pos + 17])))))
    else:
        raise ValueError(f"UNKNOWN_PRIMITIVE_TYPE_IN_VARIANT: {info}")


def _python(value: bytes, meta: _Metadata, pos: int) -> Any:
    header = value[pos]
    basic, info = header & 3, header >> 2
    if basic == 1:
        return value[pos + 1 : pos + 1 + info].decode("utf-8")
    if basic == 2:
        return {key: _python(value, meta, at) for key, at in _fields(value, meta, pos)}
    if basic == 3:
        return [_python(value, meta, at) for at in _elements(value, pos)]
    if info == 0:
        return None
    if info in (1, 2):
        return info == 1
    if info in (3, 4, 5, 6):
        return _int(value, pos + 1, 1 << (info - 3), signed=True)
    if info == 7:
        return struct.unpack("<d", value[pos + 1 : pos + 9])[0]
    if info == 14:
        return struct.unpack("<f", value[pos + 1 : pos + 5])[0]
    if info in (8, 9, 10):
        unscaled, scale = _decimal(value, pos, {8: 4, 9: 8, 10: 16}[info])
        return Decimal(unscaled) * (Decimal(10) ** (-scale))
    if info == 11:
        return date(1970, 1, 1) + timedelta(days=_int(value, pos + 1, 4, signed=True))
    if info == 12:
        return _EPOCH + timedelta(microseconds=_int(value, pos + 1, 8, signed=True))
    if info == 13:
        return _EPOCH_NTZ + timedelta(microseconds=_int(value, pos + 1, 8, signed=True))
    if info in (15, 16):
        n = _int(value, pos + 1, 4)
        raw = bytes(value[pos + 5 : pos + 5 + n])
        return raw if info == 15 else raw.decode("utf-8")
    if info == 20:
        return uuid.UUID(bytes=bytes(value[pos + 1 : pos + 17]))
    raise ValueError(f"UNKNOWN_PRIMITIVE_TYPE_IN_VARIANT: {info}")


class VariantVal:
    """A ``VARIANT`` value: its value and metadata bytes (``pyspark.sql.types.VariantVal``)."""

    def __init__(self, value: bytes, metadata: bytes) -> None:
        self.value = value
        self.metadata = metadata

    def __str__(self) -> str:
        return self.toJson()

    def __repr__(self) -> str:
        return f"VariantVal({self.value!r}, {self.metadata!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VariantVal) and (self.value, self.metadata) == (other.value, other.metadata)

    def __hash__(self) -> int:
        return hash((self.value, self.metadata))

    def toPython(self) -> Any:
        """The value as Python objects."""
        return _python(self.value, _Metadata(self.metadata), 0)

    def toJson(self, zone_id: str = "UTC") -> str:
        """The value as Spark's JSON text, timestamps in ``zone_id``."""
        from krishiv.sql._convert import session_zone

        out: list[str] = []
        _json(self.value, _Metadata(self.metadata), 0, session_zone(zone_id), out)
        return "".join(out)


def is_variant_struct(arrow_type: Any) -> bool:
    """Whether a pyarrow type is the engine's variant carrier (``metadata``, ``value``, marked ``krishiv.variant``)."""
    import pyarrow as pa

    if not pa.types.is_struct(arrow_type) or arrow_type.num_fields != 2:
        return False
    fields = [arrow_type.field(i) for i in range(2)]
    return [f.name for f in fields] == ["metadata", "value"] and all(
        (f.metadata or {}).get(b"krishiv.variant") == f.name.encode() for f in fields
    )
