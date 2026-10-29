#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plan the CodSpeed shards.

Reads the named shards from `.github/codspeed-shards.json`, finds every
`[[bench]]` target of the workspace, and prints the job matrix as JSON. A
target in no named shard runs in a final `catch-all` shard, so a new
benchmark is measured without editing the workflow. A target listed in two
shards, or listed but not a bench target, fails the plan: either would be
reported twice or break the build.

A target behind `required-features` is not built by a default workspace
build, so it is neither planned nor reported missing.
"""

import glob
import json
import os
import sys
import tomllib
from collections import Counter


def bench_targets() -> set[str]:
    names: set[str] = set()
    for path in glob.glob("**/Cargo.toml", recursive=True):
        if path.startswith(("target/", "node_modules/")) or "/target/" in path:
            continue
        with open(path, "rb") as f:
            manifest = tomllib.load(f)
        for bench in manifest.get("bench", []):
            if not bench.get("required-features"):
                names.add(bench["name"])
    return names


def plan(
    shards: dict[str, list[str]], existing: set[str]
) -> tuple[list[dict], list[str]]:
    errors = []
    listed = Counter(name for names in shards.values() for name in names)
    repeated = sorted(n for n, c in listed.items() if c > 1)
    unknown = sorted(set(listed) - existing)
    if repeated:
        errors.append(
            f"benchmark targets in more than one shard: {', '.join(repeated)}"
        )
    if unknown:
        errors.append(
            f"benchmark targets in a shard but not a bench target: {', '.join(unknown)}"
        )
    matrix = [
        {"shard": shard, "bench_flags": " ".join(f"--bench {n}" for n in names)}
        for shard, names in shards.items()
    ]
    rest = sorted(existing - set(listed))
    if rest:
        matrix.append(
            {
                "shard": "catch-all",
                "bench_flags": " ".join(f"--bench {n}" for n in rest),
            }
        )
    return matrix, errors


def main() -> int:
    with open(".github/codspeed-shards.json") as f:
        shards = json.load(f)
    matrix, errors = plan(shards, bench_targets())
    for e in errors:
        print(f"::error::{e}")
    if errors:
        return 1
    text = json.dumps({"include": matrix})
    print(text)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"matrix={text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
