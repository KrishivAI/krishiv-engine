#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plan the CodSpeed shards.

Reads the named shards from `.github/codspeed-shards.json`, finds every
bench target of the workspace (`cargo metadata`, so targets Cargo discovers
from `benches/` count too), and prints the job matrix as JSON. A
target in no named shard runs in a final `catch-all` shard, so a new
benchmark is measured without editing the workflow. A target listed in two
shards, or listed but not a bench target, fails the plan: either would be
reported twice or break the build.

A target behind `required-features` is not built by a default workspace
build, so it is neither planned nor reported missing.
"""

import json
import os
import subprocess
import sys
from collections import Counter


def bench_targets() -> set[str]:
    """Bench targets of the workspace, auto-discovered ones included."""
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    names: set[str] = set()
    for package in json.loads(out)["packages"]:
        for target in package["targets"]:
            if "bench" in target["kind"] and not target.get("required-features"):
                names.add(target["name"])
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
    empty = sorted(shard for shard, names in shards.items() if not names)
    if empty:
        # `cargo codspeed build --workspace` without `--bench` runs every
        # default benchmark, so an empty shard would duplicate the others.
        errors.append(f"shards without a benchmark target: {', '.join(empty)}")
    matrix = [
        {"shard": shard, "bench_flags": " ".join(f"--bench {n}" for n in names)}
        for shard, names in shards.items()
        if names
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
