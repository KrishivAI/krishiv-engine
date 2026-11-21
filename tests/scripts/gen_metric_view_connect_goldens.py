#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Metric-view goldens, Spark Connect form (P3 4.2-MV, G1; ADR 0008, 0003).

Three metric-view errors are not `SparkThrowable` in classic Spark (`MetricViewValidationException` for an
expression longer than 1,024 characters, `java.util.NoSuchElementException` for `measure(<dimension>)`,
`java.lang.IllegalArgumentException` for `SHOW CREATE TABLE <mv> AS SERDE`). This script records what a Spark
Connect client sees for them, plus a few ordinary `SparkThrowable` cases as a control, so the Krishiv Connect
server can reproduce the wire form. Nothing is transcribed: every outcome is produced by a real run.

How a run works (needs docker, uv, network for `pyspark-client` from PyPI; about two minutes per release):

1. The host side starts a throw-away container (`o2-mv-g1-spark421`, `o2-mv-g1-spark430`, `--rm`, image pinned by
   digest) running `start-connect-server.sh` in the foreground (`SPARK_NO_DAEMONIZE=1`), `local[2]`, the in-memory
   catalog, UTC, with the container's Connect port published on 127.0.0.1 only. The image ships the Connect jar in
   `/opt/spark/jars`, so no `--packages` is needed.
2. It runs this file again under `uv run --with pyspark-client==<ver>` (Python 3.12, `-I`) in `--client` mode.
   That is the client the Spark project ships for Connect (no JVM). Each case gets its own Connect session
   (`SparkSession.builder.remote(url).create()`) and a fresh database `mv_<case>`, dropped afterwards.
3. The host side stops the container and writes `tests/fixtures/spark-<ver>/metric-views/connect.json`.

The case steps are reused from `gen_metric_view_goldens.py` (same case ids as `cases.json`). A Connect client has
three observable phases, recorded per step: `sql` (`spark.sql(text)`, which on Connect already sends the query to
the server for analysis), `schema` (`df.schema`, an AnalyzePlan RPC) and `collect` (an ExecutePlan RPC). The first
one that raises is the error `phase`.

An error records what the Python client exposes: the exception class (`python_class`, with its `python_mro`),
`condition` (`getCondition()`), `error_class` (`getErrorClass()`, same value on Spark 4), `sqlstate`,
`parameters`, `message` (`getMessage()` without the appended server stack trace), `jvm_class` (the first line of
the server stack trace the client received), `has_jvm_stacktrace`, `query_context`, `grpc_status_code` and
`str_equals_get_message`. `vs_classic` compares with the classic run in `cases.json` of the same release:
`phase_same`, `condition_same`, `sqlstate_same`, `parameters_same`, `message_same` (exact text), and
`message_same_modulo_expr_ids` (expression ids `#123` and `ExprId(123,uuid)` masked).

    python3 tests/scripts/gen_metric_view_connect_goldens.py [--only 4.2.1,4.3.0]
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SELF = Path(__file__).resolve()
ROOT = SELF.parents[2]
FORMAT_VERSION = 1
STACK_MARK = "\n\nJVM stacktrace:\n"

# version -> (container name, image pinned by digest, fixture dir, host port)
RUNS = {
    "4.2.1": (
        "o2-mv-g1-spark421",
        "apache/spark:4.2.1@sha256:0cb70cf0193ec81c93c6c47078208512bbe659dd9dc89535dd2773826f0ea2bf",
        "spark-4.2.1",
        15421,
    ),
    "4.3.0": (
        "o2-mv-g1-spark430",
        "apache/spark:4.3.0@sha256:3dedc5c8665a6f472e2f0a3a828207a79a30785090ef325d3af219cb90a73d63",
        "spark-4.3.0",
        15430,
    ),
}
SERVER_CONF = {
    "spark.sql.catalogImplementation": "in-memory",
    "spark.sql.warehouse.dir": "/tmp/o2mv-warehouse",
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.shuffle.partitions": "2",
    "spark.ui.enabled": "false",
}
CLIENT_DEPS = ["grpcio", "grpcio-status", "protobuf", "googleapis-common-protos", "pandas<3", "pyarrow", "numpy"]

# The three non-SparkThrowable errors (every case id that produces one in cases.json) and the controls.
FOCUS = [
    "yaml_expr_length_1025",
    "yaml_expr_length_1031_measure",
    "yaml_expr_length_1025_concat",
    "yaml_expr_length_1025_measure",
    "yaml_expr_non_bmp_1026_units",
    "yaml_expr_non_bmp_concat_1026_units",
    "read_measure_of_dimension",
    "ddl_show_create_table",
]
CONTROLS = [
    "ddl_drop_view_on_table",  # AnalysisException, WRONG_COMMAND_FOR_OBJECT_TYPE (42809)
    "read_measure_of_expression",  # ExtendedAnalysisException, message ends with the analyzed plan
    "read_bare_measure_in_aggregate",  # ExtendedAnalysisException, MISSING_AGGREGATION
    "parse_temporary",  # ParseException, PARSE_SYNTAX_ERROR
    "parse_missing_with_metrics",  # SparkThrowable INTERNAL_ERROR (4.2.1) / MISSING_CLAUSES_FOR_OPERATION (4.3.0)
    "read_having_dimension",  # success: schema and rows
]
CASE_IDS = FOCUS + CONTROLS

# ---------------------------------------------------------------------------------------------------------------
# Client (runs under uv with pyspark-client)
# ---------------------------------------------------------------------------------------------------------------


def dump_golden(doc: dict) -> str:
    """The golden as JSON with one case per line, so a review diff shows changed cases, not 20,000 lines."""
    head = {k: v for k, v in doc.items() if k != "cases"}
    lines = [json.dumps(head, ensure_ascii=True)[:-1] + ', "cases": [']
    cases = [json.dumps(c, ensure_ascii=True) for c in doc["cases"]]
    lines += [c + ("," if i + 1 < len(cases) else "") for i, c in enumerate(cases)]
    lines.append("]}")
    return "\n".join(lines) + "\n"


def _error(e: Any, phase: str) -> dict:
    import warnings

    rec: dict = {"phase": phase, "python_class": f"{type(e).__module__}.{type(e).__qualname__}"}
    rec["python_mro"] = [c.__qualname__ for c in type(e).__mro__ if c.__module__.startswith("pyspark")]
    full = e.getMessage() if hasattr(e, "getMessage") else str(e)
    head, sep, stack = full.partition(STACK_MARK)
    rec["message"] = head
    rec["has_jvm_stacktrace"] = bool(sep)
    rec["jvm_class"] = stack.split("\n", 1)[0] if sep else None
    rec["str_equals_get_message"] = str(e) == full
    if hasattr(e, "getCondition"):
        rec["condition"] = e.getCondition()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec["error_class"] = e.getErrorClass()
        rec["sqlstate"] = e.getSqlState()
        params = e.getMessageParameters()
        rec["parameters"] = None if params is None else {k: params[k] for k in sorted(params)}
        ctxs = []
        for c in e.getQueryContext() or []:
            item: dict = {}
            for attr in ("contextType", "objectType", "objectName", "startIndex", "stopIndex", "fragment"):
                try:
                    val = getattr(c, attr)()
                    item[attr] = val if isinstance(val, (int, str)) or val is None else str(val)
                except Exception:
                    pass
            ctxs.append(item)
        rec["query_context"] = ctxs
    code = getattr(e, "_grpc_status_code", None)
    rec["grpc_status_code"] = None if code is None else str(code)
    return rec


def _run_sql(s: Any, text: str) -> dict:
    sys.path.insert(0, str(SELF.parent))
    g = importlib.import_module("gen_metric_view_goldens")

    try:
        df = s.sql(text)
    except Exception as e:
        return {"error": _error(e, "sql")}
    out: dict = {}
    try:
        out["schema"] = g._schema(df.schema)
    except Exception as e:
        return {"error": _error(e, "schema")}
    try:
        out.update(g._collect(df, text))
    except Exception as e:
        out["error"] = _error(e, "collect")
    return out


def client(version: str, port: int) -> None:
    from pyspark.sql import SparkSession

    sys.path.insert(0, str(SELF.parent))
    g = importlib.import_module("gen_metric_view_goldens")

    by_name = {c["name"]: c for c in g.CASES}
    results = []
    admin = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").create()
    server_version = admin.version
    for name in CASE_IDS:
        c = by_name[name]
        s = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").create()
        db = "mv_" + name
        s.sql(f"CREATE DATABASE {db}")
        s.sql(f"USE {db}")
        steps = []
        for step in c["steps"]:
            rec = dict(step)
            if step["kind"] == "sql":
                rec["outcome"] = _run_sql(s, step["sql"])
            elif step["kind"] == "conf":
                s.conf.set(step["key"], step["value"])
                rec["outcome"] = {}
            else:
                raise SystemExit(f"{name}: api steps are not supported in Connect mode")
            steps.append(rec)
        results.append(
            {
                "name": name,
                "group": c["group"],
                "note_ref": c["note_ref"],
                "database": db,
                "role": "control" if name in CONTROLS else "non_sparkthrowable",
                "target": "compatible",
                "steps": steps,
            }
        )
        try:
            s.sql("USE default")
            for (dbname,) in s.sql("SHOW DATABASES").collect():
                if dbname == db or dbname.startswith(db + "_"):
                    s.sql(f"DROP DATABASE {dbname} CASCADE")
        except Exception as e:
            print(f"cleanup {db}: {e}", file=sys.stderr)
        s.stop()
    admin.stop()
    print(
        "@@JSON " + json.dumps({"spark_version": server_version, "client_version": _client_version(), "cases": results})
    )


def _client_version() -> str:
    import pyspark

    return pyspark.__version__


# ---------------------------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------------------------


def _mask(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"ExprId\(\d+,[0-9a-f-]+\)", "ExprId(N,U)", text)
    return re.sub(r"#\d+L?", "#N", text)


def _vs_classic(conn: dict, classic: dict | None) -> dict | None:
    if classic is None:
        return None
    cond = classic.get("condition")
    msg = classic.get("message")
    cm = conn.get("message")
    return {
        "classic_java_class": classic.get("java_class"),
        "classic_phase": classic.get("phase"),
        "phase_same": conn.get("phase") == classic.get("phase"),
        "condition_same": conn.get("condition") == cond,
        "sqlstate_same": conn.get("sqlstate") == classic.get("sqlstate"),
        "parameters_same": conn.get("parameters") == classic.get("parameters"),
        "message_same": cm == msg,
        "message_same_modulo_expr_ids": _mask(cm) == _mask(msg),
        "classic_message": msg,
    }


def _attach_classic(version: str, fixture_dir: str, cases: list) -> None:
    classic = json.loads((ROOT / "tests/fixtures" / fixture_dir / "metric-views/cases.json").read_text("utf-8"))
    by_name = {c["name"]: c for c in classic["cases"]}
    for c in cases:
        ref = by_name[c["name"]]["steps"]
        if len(ref) != len(c["steps"]):
            raise SystemExit(f"{c['name']}: step count differs from cases.json")
        for step, rstep in zip(c["steps"], ref, strict=True):
            if step.get("sql") != rstep.get("sql"):
                raise SystemExit(f"{c['name']}: step text differs from cases.json")
            err = step["outcome"].get("error")
            rerr = rstep["outcome"].get("error")
            step["outcome_kind"] = "error" if err else "success"
            step["classic_outcome_kind"] = "error" if rerr else "success"
            if err:
                err["vs_classic"] = _vs_classic(err, rerr)
            elif not rerr:
                out, rout = step["outcome"], rstep["outcome"]
                step["vs_classic"] = {
                    "schema_same": out.get("schema") == rout.get("schema"),
                    "rows_same": out.get("rows") == rout.get("rows"),
                }
            step["match"] = "class_sqlstate_prefix" if err and not err.get("condition") else "exact"


def _wait_ready(name: str) -> None:
    for _ in range(120):
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True, check=False)
        if "Spark Connect server started" in logs.stdout + logs.stderr:
            return
        time.sleep(1)
    raise SystemExit(f"{name}: the Connect server did not start")


def run_release(version: str) -> dict:
    name, image, _, port = RUNS[version]
    cmd = ["docker", "run", "-d", "--rm", "--name", name, "--entrypoint", "", "-e", "HOME=/tmp"]
    cmd += ["-e", "SPARK_NO_DAEMONIZE=1", "-p", f"127.0.0.1:{port}:15002", image]
    cmd += ["/opt/spark/sbin/start-connect-server.sh", "--master", "local[2]"]
    for k, v in SERVER_CONF.items():
        cmd += ["--conf", f"{k}={v}"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    try:
        _wait_ready(name)
        uv = ["uv", "run", "--no-project", "--python", "3.12", "--with", f"pyspark-client=={version}"]
        for dep in CLIENT_DEPS:
            uv += ["--with", dep]
        uv += ["python", "-I", str(SELF), "--client", version, str(port)]
        done = subprocess.run(uv, capture_output=True, text=True, check=False, timeout=1800)
    finally:
        subprocess.run(["docker", "stop", name], capture_output=True, text=True, check=False)
    for line in done.stdout.splitlines():
        if line.startswith("@@JSON "):
            return json.loads(line[len("@@JSON ") :])
    raise SystemExit(f"Spark {version}: no output (exit {done.returncode}):\n{done.stderr[-3000:]}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--client":
        client(args[1], int(args[2]))
        return
    only = set(args[1].split(",")) if args and args[0] == "--only" else None
    for version, (_, image, fixture_dir, _) in RUNS.items():
        if only and version not in only:
            continue
        data = run_release(version)
        if data["spark_version"] != version:
            raise SystemExit(f"image {image} reports Spark {data['spark_version']}, expected {version}")
        _attach_classic(version, fixture_dir, data["cases"])
        doc = {
            "format_version": FORMAT_VERSION,
            "generator": "tests/scripts/gen_metric_view_connect_goldens.py",
            "spark_version": version,
            "image": image,
            "session": "Spark Connect server (local[2], in-memory catalog), Python client pyspark-client",
            "client_version": data["client_version"],
            "server_conf": SERVER_CONF,
            "cases": data["cases"],
        }
        out = ROOT / "tests/fixtures" / fixture_dir / "metric-views/connect.json"
        out.write_text(dump_golden(doc), encoding="utf-8")
        n = sum(len(c["steps"]) for c in data["cases"])
        print(f"Spark {version}: {len(data['cases'])} cases, {n} steps -> {out.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
