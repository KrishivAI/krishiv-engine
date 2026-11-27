# SPDX-License-Identifier: Apache-2.0
"""``_local_naive`` converts an instant to the client's naive local time; where the C runtime cannot convert a
pre-1970 instant (Windows, ``astimezone()`` raises ``OSError``) it falls back to the 1970 offset."""

from datetime import UTC, datetime, timedelta, timezone

from krishiv.sql import _convert


def test_an_instant_converts_to_naive_local_time():
    value = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert _convert._local_naive(value) == value.astimezone().replace(tzinfo=None)


def test_a_failing_astimezone_falls_back_to_the_1970_offset(monkeypatch):
    class Failing(datetime):
        def astimezone(self, tz=None):
            if tz is None:
                raise OSError(22, "Invalid argument")
            return super().astimezone(tz)

    class Local(datetime):
        """Stands for the 1970 reference instant: local offset +02:00."""

        def astimezone(self, tz=None):
            return datetime.fromtimestamp(0, timezone(timedelta(hours=2)))

    value = Failing(1960, 1, 1, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(_convert, "datetime", Local)
    assert _convert._local_naive(value) == datetime(1960, 1, 1, 14, 0, 0)
