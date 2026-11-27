# SPDX-License-Identifier: Apache-2.0
"""``_local_naive`` converts an instant to the client's naive local time. Where the C runtime cannot convert a
pre-1970 instant (Windows: ``astimezone()`` raises ``OSError``) the instant moves by whole 400-year cycles to
1970 or later, so a DST zone keeps its summer and winter offsets."""

from datetime import UTC, datetime
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
    assert isinstance(_convert._local_naive(value), datetime)
