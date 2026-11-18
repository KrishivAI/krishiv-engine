# SPDX-License-Identifier: Apache-2.0
"""The SQL form of every case of ``cases.py``, for the replay on the server's worker pool
(``crates/krishiv-udf-pool/tests/udtf_goldens.rs``).

Each case runs against a recording stand-in for the session, with the UDTFs built by ``pyspark``'s decorators (what
a Spark Connect client sends): the UDTFs it registers (name, eval type, return type, determinism and the class
cloudpickled as Spark Connect's ``command``) and the SQL it runs. A case that builds a DataFrame call instead
(``t.asTable()``, ``udtf(...)`` on columns, ``lateralJoin``) is ``dataframe`` (Spark Connect's relation path, not
replayed here); one whose definition fails in ``pyspark`` is ``client_error`` with the condition (raised by the
client, as Spark Connect raises it). Usage: ``python export_cases.py OUT.json``. The session's
``spark.sql.execution.pythonUDTF.arrow.enabled`` comes from ``KRISHIV_UDTF_ARROW_DEFAULT`` (the engine's answer to
``conf.get``), which decides a bare ``@udtf``'s eval type as ``pyspark.sql.udtf._create_py_udtf`` reads it.

The replay reads the export committed as ``crates/krishiv-udf-pool/tests/goldens/udtf_cases.json.gz``, so CI needs
no full PySpark, only the workers' interpreter. Regenerate it after a change to ``cases.py`` or this file (the
replay compares their SHA-256 with the ones recorded) with a full PySpark 4.2.0 on the workers' CPython minor
version (cloudpickled code objects are specific to it), from the repository root::

    KRISHIV_UDTF_ARROW_DEFAULT=true $PYSPARK_PYTHON python/tests/pyspark_udf/udtf/export_cases.py \
        crates/krishiv-udf-pool/tests/goldens/udtf_cases.json.gz

A ``.gz`` output is the committed form: gzip (no timestamp) of ``{"python", "pyspark", "arrow_default",
"sources", "cases"}``, ``sources`` mapping each of ``cases.py`` and ``export_cases.py`` to its SHA-256.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))


class Captured(Exception):
    """The query a case ran."""

    def __init__(self, sql: str) -> None:
        super().__init__(sql)
        self.sql = sql


class NotSql(Exception):
    """A case that calls a DataFrame method beyond the view set-up."""


class _Frame:
    """The views' DataFrames: only the set-up calls ``prepare`` makes are allowed."""

    def where(self, _cond: str) -> _Frame:
        return self

    def createOrReplaceTempView(self, _name: str) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        raise NotSql(name)


class _Registration:
    def __init__(self) -> None:
        self.udtfs: list[tuple[str, Any]] = []

    def register(self, name: str, f: Any) -> Any:
        self.udtfs.append((name, f))
        return f


class _Conf:
    def get(self, key: str, default: Any = None) -> Any:
        if key == "spark.sql.execution.pythonUDTF.arrow.enabled":
            return os.environ.get("KRISHIV_UDTF_ARROW_DEFAULT", "true")
        return default


class _Session:
    def __init__(self) -> None:
        self.udtf = _Registration()
        self.conf = _Conf()

    def createDataFrame(self, *_a: Any, **_k: Any) -> _Frame:
        return _Frame()

    def table(self, _name: str) -> _Frame:
        return _Frame()

    def sql(self, text: str) -> Any:
        raise Captured(text)


def _return_type(t: Any) -> str | None:
    """The declared return type as ``struct<..>`` text: a DDL string (``a int, b string``) is read without a JVM
    (top-level commas split the fields, the first blank a field's name from its type)."""
    if t is None:
        return None
    if not isinstance(t, str):
        return "json:" + str(t.json())  # Spark's JSON keeps each field's nullability
    fields, depth, start = [], 0, 0
    for i, ch in enumerate(t + ","):
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif ch == "," and depth == 0:
            fields.append(t[start:i].strip())
            start = i + 1
    parts = [f.split(None, 1) for f in fields if f]
    if any(len(p) != 2 or p[0].endswith(">") for p in parts):
        return t  # a type, not a field list (``int``, ``array<int>``)
    return "struct<" + ",".join(f"{n}:{ty.strip()}" for n, ty in parts) + ">"


def main() -> None:
    from pyspark import cloudpickle
    from pyspark.sql import SparkSession

    from pyspark_udf.udtf import cases

    cloudpickle.register_pickle_by_value(cases)
    api = cases.Api("pyspark")
    out: dict[str, Any] = {}
    for name, build in sorted(cases.CASES.items()):
        spark = _Session()
        # ``_create_py_udtf`` reads the arrow setting from the instantiated session.
        SparkSession._instantiatedSession = spark  # type: ignore[assignment]
        try:
            build(api, spark)
            record: dict[str, Any] = {"kind": "dataframe"}
        except Captured as c:
            record = {"kind": "sql", "sql": c.sql}
        except NotSql:
            record = {"kind": "dataframe"}
        except Exception as e:
            get = getattr(e, "getCondition", None)
            record = {"kind": "client_error", "condition": get() if get else None, "message": str(e)}
        finally:
            SparkSession._instantiatedSession = None
        if record["kind"] == "sql":
            record["udtfs"] = [
                {
                    "name": n,
                    "eval_type": f.evalType,
                    "return_type": _return_type(f._returnType),
                    "deterministic": bool(f.deterministic),
                    "command": cloudpickle.dumps(f.func).hex(),
                }
                for n, f in spark.udtf.udtfs
            ]
        out[name] = record
    target = Path(sys.argv[1])
    if target.suffix != ".gz":
        target.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        return
    import pyspark

    committed = {
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
        "pyspark": pyspark.__version__,
        "arrow_default": os.environ.get("KRISHIV_UDTF_ARROW_DEFAULT", "true"),
        "sources": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ("cases.py", "export_cases.py")},
        "cases": out,
    }
    text = json.dumps(committed, sort_keys=True, separators=(",", ":")) + "\n"
    target.write_bytes(gzip.compress(text.encode(), compresslevel=9, mtime=0))


if __name__ == "__main__":
    main()
