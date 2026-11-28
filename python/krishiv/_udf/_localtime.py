# SPDX-License-Identifier: Apache-2.0
"""Client-local time conversions that also work on Windows (before 1970).

Lives in ``krishiv._udf`` because the UDF worker package ships without ``krishiv.sql``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def _local_naive(value: datetime) -> datetime:
    """``value`` (an aware datetime) as a naive datetime in the client's local time zone.

    Windows' C runtime cannot convert an instant before 1970 (``astimezone()`` raises ``OSError``). There the
    instant is moved forward by whole 400-year Gregorian cycles (146097 days, which keep the weekday and the
    calendar date) to 1970 or later, the zone's offset is read there, and that offset applies to the original
    instant. Windows applies its current rules to every year, so this is the answer its own conversion would give."""
    try:
        return value.astimezone().replace(tzinfo=None)
    except (OSError, OverflowError):
        instant = value.astimezone(UTC)
        cycles = -(-(1970 - instant.year) // 400)
        # at least one cycle: a failure at 1970-01-01 itself means the local time falls before the epoch
        offset = (instant + timedelta(days=146097 * max(cycles, 1))).astimezone().utcoffset() or timedelta(0)
        try:
            return (instant + offset).replace(tzinfo=None)
        except OverflowError:  # within a day of datetime.min
            return instant.replace(tzinfo=None)


def _naive_to_utc(value: datetime) -> datetime:
    """``value`` as an aware UTC datetime; a naive value is read as client-local time (as ``astimezone`` does).

    Windows' C runtime cannot convert a local time before 1970 (``astimezone`` raises ``OSError``). There the
    zone's offset is read at the same wall time moved forward by whole 400-year cycles, as in ``_local_naive``."""
    try:
        return value.astimezone(UTC)
    except (OSError, OverflowError):
        cycles = max(-(-(1970 - value.year) // 400), 1)  # at least one: 1970-01-01 local can be before the epoch
        try:
            offset = (value + timedelta(days=146097 * cycles)).astimezone().utcoffset() or timedelta(0)
            return (value - offset).replace(tzinfo=UTC)
        except OverflowError:  # within a day of datetime.min
            return value.replace(tzinfo=UTC)
