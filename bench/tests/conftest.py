"""Shared fixtures."""

from __future__ import annotations

import pytest

from krishiv_bench import ab, abba, bisect, flamediff, reproduce
from krishiv_bench.adapters.krishiv import ANY_BUILD_ENV


@pytest.fixture(autouse=True)
def no_load_settle(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs must not wait for this machine's load to decay; the tests of ``wait_idle`` itself keep the real one."""
    if "wait_idle" in request.node.name:
        return
    for mod in (abba, ab, reproduce, bisect, flamediff):
        monkeypatch.setattr(mod, "wait_idle", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def any_runner_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests do not judge this machine's runner build (the dist runner a test may find is the lane's own); the build
    level check has its own tests (``test_build_level.py``), which clear this."""
    monkeypatch.setenv(ANY_BUILD_ENV, "1")
