"""Shared fixtures."""

from __future__ import annotations

import pytest

from krishiv_bench import ab, abba, reproduce


@pytest.fixture(autouse=True)
def no_load_settle(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs must not wait for this machine's load to decay; the tests of ``wait_idle`` itself keep the real one."""
    if "wait_idle" in request.node.name:
        return
    for mod in (abba, ab, reproduce):
        monkeypatch.setattr(mod, "wait_idle", lambda *a, **k: None)
