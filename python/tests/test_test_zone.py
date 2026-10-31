# SPDX-License-Identifier: Apache-2.0
"""The test setup's zone, whatever `TZ` the shell exports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Runs the setup in a process started in another zone, then prints the zone the
# engine reads and the offset this interpreter's own conversions use.
_PROBE = """
import os, runpy, sys, time
runpy.run_path(sys.argv[1])
print(os.environ["TZ"], time.timezone)
"""


def test_the_setup_moves_the_process_to_one_zone() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, str(Path(__file__).with_name("conftest.py"))],
        env={**os.environ, "TZ": "Asia/Tokyo"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["UTC", "0"]
