"""Memory sizes on the command line (``--cap 4.5g``, ``--engine-memory 3g``): binary units, validated."""

from __future__ import annotations

import re

_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*(g|gb|gib|m|mb|mib)$", re.IGNORECASE)


def parse_size(text: str) -> int:
    """``4.5g`` -> bytes (GiB), ``1024m`` -> bytes (MiB). Rejects other units, zero and negative values."""
    m = _SIZE.match(text.strip())
    if not m:
        raise ValueError(f"memory size must look like 4.5g, 3g or 1024m (got {text!r})")
    value = float(m.group(1)) * (1 << 30 if m.group(2).lower().startswith("g") else 1 << 20)
    if value < 1 << 20:
        raise ValueError(f"memory size must be at least 1m (got {text!r})")
    return int(value)
