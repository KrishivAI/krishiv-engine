#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge the classic and Connect runs of gen_observe_goldens.py into one fixture (P3-T084).

Usage: python3 tests/scripts/merge_observe_goldens.py CLASSIC.json CONNECT.json OUT.json
"""

import json
import sys
from pathlib import Path


def main() -> None:
    classic = json.loads(Path(sys.argv[1]).read_text())
    connect = json.loads(Path(sys.argv[2]).read_text())
    assert classic["spark_version"] == connect["spark_version"]
    merged = {
        "format_version": classic["format_version"],
        "generator": classic["generator"],
        "spark_version": classic["spark_version"],
        "classic": classic["cases"],
        "connect": connect["cases"],
    }
    Path(sys.argv[3]).write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
