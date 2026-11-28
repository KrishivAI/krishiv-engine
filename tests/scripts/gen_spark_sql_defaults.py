#!/usr/bin/env python3
"""Generate the table of Spark's registered SQL configs and what `spark.conf.get` answers for an unset one
(acceptance SV-3, ADR 0008).

Nothing is transcribed: a one-off throw-away container of the default profile's release (image pinned by
digest below) runs `spark-shell` with `spark_sql_defaults.scala`, which lists every entry `SQLConf` registers
(internal and static ones included) with its default. The rows go to
`crates/krishiv-common/src/config/spark_defaults_table.rs`. `config::spark_default` reads it: a key Krishiv
does not register itself, that the client reads unset, answers Spark's default (`Config.get`, SQL `SET key`,
the embedded `spark.conf.get`) instead of `SQL_CONF_NOT_FOUND`. Krishiv's own entries are asked first.

The same run lists the Spark config entries that are not SQL ones (`spark.executor.memory`, `spark.ui.port`): the
keys of `ConfigEntry.knownConfigs` once a query has run, minus the SQL keys. They go to
`crates/krishiv-common/src/config/spark_core_keys_table.rs`; `config::is_spark_core_key` reads it, and a session
refuses to change them at runtime as Spark does (`CANNOT_MODIFY_CONFIG`).

A default Spark computes from its host (`createWithDefaultFunction`: a path, the JVM's zone) is not a Spark
default a client can expect anywhere: those keys are listed in `HOST_DEFAULTS` and answer no value.

Every supported release is run (images pinned as in `gen_legacy_settings.py`): the default profile's release
(4.3.0) gives the table, and each older one the rows it answers differently, or does not register
(`SPARK_SQL_DEFAULTS_BEFORE`); a session reads the rows of its profile's release.

Usage (from the repo root; needs docker; about five minutes):

    python3 tests/scripts/gen_spark_sql_defaults.py

Output is deterministic.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCALA = Path(__file__).resolve().parent / "spark_sql_defaults.scala"
OUT = ROOT / "crates/krishiv-common/src/config/spark_defaults_table.rs"
CORE_OUT = ROOT / "crates/krishiv-common/src/config/spark_core_keys_table.rs"

# One release per supported profile line, pinned by digest as in gen_legacy_settings.py. The default profile's
# (SparkProfile::default(), 4.3) is the base table; the others are stored as their differences from it.
IMAGES = {
    "3.5.9": "apache/spark:3.5.9@sha256:af02a459c8706e031c835c16f9db3c463816776e2543dd0f828af65606bcf392",
    "4.0.5": "apache/spark:4.0.5@sha256:45eb1b8261a89bc7692be05aa4664676856f1c886cecc7b97b4e10f635b45b3c",
    "4.1.4": "apache/spark:4.1.4@sha256:847f91f8bb2f28d632064dfe632d423020f8fd60b27c7704c0e5bf5b0c6432c7",
    "4.2.1": "apache/spark:4.2.1@sha256:0cb70cf0193ec81c93c6c47078208512bbe659dd9dc89535dd2773826f0ea2bf",
    "4.3.0": "apache/spark:4.3.0@sha256:3dedc5c8665a6f472e2f0a3a828207a79a30785090ef325d3af219cb90a73d63",
}
RELEASE = "4.3.0"

# Defaults that depend on the host the release runs on (the JVM zone, the working directory); found by reading the
# run's `function` rows and its path-valued rows (each one named, never a pattern).
HOST_DEFAULTS: frozenset[str] = frozenset({"spark.sql.session.timeZone", "spark.sql.warehouse.dir"})

Row = tuple[str, str, str, bool]


def read_release(image: str) -> tuple[list[Row], list[str]]:
    """(key, kind, value, static) rows of the SQL entries, and the keys of the other entries. The script goes in
    on stdin (`-I` runs silently on 4.x); a cold container can stall, so it is tried twice."""
    cmd = ["docker", "run", "--rm", "-i", "--entrypoint", "", "-e", "HOME=/tmp", "-e", "TZ=UTC", image]
    cmd += ["/opt/spark/bin/spark-shell", "--master", "local[1]"]
    stderr = ""
    for _ in range(2):
        done = subprocess.run(
            cmd, input=SCALA.read_text(encoding="utf-8"), capture_output=True, text=True, check=False, timeout=900
        )
        stderr = done.stderr
        # the REPL echoes a prompt before a line (`scala> @@END`)
        lines = [ln[ln.index("@@") :] for ln in done.stdout.splitlines() if "@@" in ln]
        printed = {ln[:3] for ln in lines}
        if {"@@E", "@@C"} <= printed and "@@END" in lines:
            rows: list[Row] = []
            core: list[str] = []
            for ln in lines:
                if ln.startswith("@@E\t"):
                    _, key, kind, value, static = ln.split("\t")[:5]
                    rows.append((key, kind, unescape(value), static == "true"))
                elif ln.startswith("@@C\t"):
                    core.append(ln.split("\t")[1])
            return rows, core
    raise SystemExit(f"{image}: no registry printed:\n{stderr[-1500:]}")


def unescape(s: str) -> str:
    """The probe's text back: it writes a backslash, tab, line feed and carriage return as two characters."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"\\": "\\", "t": "\t", "n": "\n", "r": "\r"}[s[i + 1]])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def rust_str(s: str) -> str:
    esc = s.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t").replace("\n", "\\n")
    return '"' + esc.replace("\r", "\\r") + '"'


def entry(key: str, kind: str, value: str, static: bool) -> str:
    """The Rust `(SparkDefault, static)` of a row."""
    if kind == "fallback":
        d = f"SparkDefault::Fallback({rust_str(value)})"
    elif kind == "none" or key in HOST_DEFAULTS:
        d = "SparkDefault::NoValue"
    else:
        d = f"SparkDefault::Value({rust_str(value)})"
    return f"{d}, {'true' if static else 'false'}"


def main() -> None:
    runs = {}
    for release, image in IMAGES.items():
        runs[release] = read_release(image)
        print(f"Spark {release}: {len(runs[release][0])} SQL config entries", file=sys.stderr)
    rows, core = runs[RELEASE]
    for release, (r, _) in runs.items():
        for key, kind, value, _ in r:
            if kind == "function" and key not in HOST_DEFAULTS:
                print(f"{release}: computed default, check it is not host-specific: {key} = {value}", file=sys.stderr)
    base = {k: entry(k, kind, v, st) for k, kind, v, st in rows}
    out = [
        "// SPDX-License-Identifier: Apache-2.0",
        "// @generated by tests/scripts/gen_spark_sql_defaults.py from the Spark releases "
        + ", ".join(IMAGES)
        + ". Do not edit.",
        "",
        f"/// Every SQL config entry Spark {RELEASE} registers, sorted by key: the key, what `spark.conf.get` answers",
        "/// for it unset, and whether it is a static (`StaticSQLConf`) key.",
        "pub(super) const SPARK_SQL_DEFAULTS: &[(&str, SparkDefault, bool)] = &[",
    ]
    out += [f"    ({rust_str(k)}, {base[k]})," for k in sorted(base)]
    out += [
        "];",
        "",
        f"/// The rows an older release answers differently from {RELEASE}, sorted by release line and key: `None` for",
        "/// a key that release does not register.",
        "#[rustfmt::skip]",
        "pub(super) const SPARK_SQL_DEFAULTS_BEFORE: &[OlderRow] = &[",
    ]
    deltas = 0
    for release, (r, _) in runs.items():
        if release == RELEASE:
            continue
        line = ".".join(release.split(".")[:2])
        theirs = {k: entry(k, kind, v, st) for k, kind, v, st in r}
        for k in sorted(set(base) | set(theirs)):
            if k not in theirs:
                out.append(f"    ({rust_str(line)}, {rust_str(k)}, None),")
            elif base.get(k) != theirs[k]:
                out.append(f"    ({rust_str(line)}, {rust_str(k)}, Some(({theirs[k]}))),")
            else:
                continue
            deltas += 1
    out.append("];")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(rows)} entries, {deltas} older-release rows -> {OUT.relative_to(ROOT)}", file=sys.stderr)
    core_out = [
        "// SPDX-License-Identifier: Apache-2.0",
        f"// @generated by tests/scripts/gen_spark_sql_defaults.py from the Spark release {RELEASE}. Do not edit.",
        "",
        f"/// Every Spark {RELEASE} config entry that is not a SQL one, sorted: the keys Spark refuses to change at",
        "/// runtime (`CANNOT_MODIFY_CONFIG`).",
        "pub(super) const SPARK_CORE_KEYS: &[&str] = &[",
    ]
    core_out += [f"    {rust_str(k)}," for k in sorted(set(core))]
    core_out.append("];")
    CORE_OUT.write_text("\n".join(core_out) + "\n", encoding="utf-8")
    print(f"{len(set(core))} core keys -> {CORE_OUT.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
