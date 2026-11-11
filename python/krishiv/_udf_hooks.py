# SPDX-License-Identifier: Apache-2.0
"""Process hooks of the embedded Python UDF pool (design P2-T059 §2.5, unit 2b).

- At interpreter exit the pool is stopped and joined (at most 5 s, the GIL released) before finalization
  starts, so its threads release their Python thread states; a thread still inside a UDF is left running
  and reported here.
- In a forked child the inherited pool does not exist (its threads stayed in the parent): the native side
  sees the pool was started in another process, leaves it alone and starts the child's own on its first
  UDF call (no fork hook is needed).
"""

from __future__ import annotations

import atexit
import logging

from krishiv import _native

_log = logging.getLogger("krishiv.udf")
_registered = False


def _shutdown() -> None:
    stuck = _native._udf_pool_shutdown()
    if stuck:
        _log.warning("%d Python UDF pool thread(s) did not stop within 5 s and were left running", stuck)


def register() -> None:
    """Registers the hook once per process."""
    global _registered
    if _registered:
        return
    _registered = True
    atexit.register(_shutdown)
