# SPDX-License-Identifier: Apache-2.0
"""``_local_naive`` converts an instant to the client's naive local time. Where the C runtime cannot convert a
pre-1970 instant (Windows: ``astimezone()`` raises ``OSError``) the instant moves by whole 400-year cycles to
1970 or later, so a DST zone keeps its summer and winter offsets."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from krishiv.sql import _convert

NEW_YORK = ZoneInfo("America/New_York")


def test_an_instant_converts_to_naive_local_time():
    value = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert _convert._local_naive(value) == value.astimezone().replace(tzinfo=None)


class _NoPre1970(datetime):
    """A datetime whose ``astimezone()`` (local zone) fails before 1970 as Windows' C runtime does, and
    otherwise answers in New York's rules."""

    def astimezone(self, tz=None):
        if tz is None:
            if self.year < 1970:
                raise OSError(22, "Invalid argument")
            return datetime.fromtimestamp(self.timestamp(), NEW_YORK)
        return super().astimezone(tz)


@pytest.mark.parametrize(
    ("utc", "local"),
    [
        (datetime(1965, 7, 1, 16, 0), datetime(1965, 7, 1, 12, 0)),  # summer: EDT, -4
        (datetime(1965, 1, 1, 16, 0), datetime(1965, 1, 1, 11, 0)),  # winter: EST, -5
        (datetime(1600, 7, 1, 16, 0), datetime(1600, 7, 1, 12, 0)),
    ],
)
def test_a_pre_1970_instant_keeps_the_zones_dst(monkeypatch, utc, local):
    monkeypatch.setattr(_convert, "datetime", _NoPre1970)
    value = _NoPre1970(*utc.timetuple()[:6], tzinfo=UTC)
    assert _convert._local_naive(value) == local


def test_an_instant_at_datetime_min_does_not_overflow(monkeypatch):
    monkeypatch.setattr(_convert, "datetime", _NoPre1970)
    value = _NoPre1970(1, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert _convert._local_naive(value) == datetime(1, 1, 1)


class _NoPre1970Naive(datetime):
    """As ``_NoPre1970``, and a naive value read as local time (``astimezone(UTC)``) fails before 1970 too."""

    def astimezone(self, tz=None):
        if self.year < 1970 and (tz is None or self.tzinfo is None):
            raise OSError(22, "Invalid argument")
        if tz is None:
            return datetime.fromtimestamp(self.timestamp(), NEW_YORK)
        if self.tzinfo is None:
            return datetime.fromtimestamp(self.replace(tzinfo=NEW_YORK).timestamp(), tz)
        return super().astimezone(tz)


def test_a_naive_time_converts_to_utc_as_astimezone_does():
    value = datetime(2024, 1, 2, 3, 4, 5)
    assert _convert._naive_to_utc(value) == value.astimezone(UTC)


@pytest.mark.parametrize(
    ("local", "utc"),
    [
        (datetime(1970, 1, 1, 0, 0), datetime(1970, 1, 1, 5, 0, tzinfo=UTC)),  # the value that failed on Windows
        (datetime(1965, 7, 1, 12, 0), datetime(1965, 7, 1, 16, 0, tzinfo=UTC)),  # summer: EDT, -4
        (datetime(1965, 1, 1, 11, 0), datetime(1965, 1, 1, 16, 0, tzinfo=UTC)),  # winter: EST, -5
    ],
)
def test_a_pre_1970_naive_time_keeps_the_zones_dst(monkeypatch, local, utc):
    monkeypatch.setattr(_convert, "datetime", _NoPre1970Naive)
    value = _NoPre1970Naive(*local.timetuple()[:6])
    assert _convert._naive_to_utc(value) == utc


def test_a_naive_time_at_datetime_min_does_not_overflow(monkeypatch):
    monkeypatch.setattr(_convert, "datetime", _NoPre1970Naive)
    value = _NoPre1970Naive(1, 1, 1, 0, 0, 0)
    result = _convert._naive_to_utc(value)
    assert result.tzinfo is UTC
    assert result.replace(tzinfo=None) - datetime(1, 1, 1) < timedelta(days=1)
