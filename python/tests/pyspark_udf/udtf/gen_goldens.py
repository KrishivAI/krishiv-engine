# SPDX-License-Identifier: Apache-2.0
"""Run every case of ``cases.py`` on PySpark 4.2.0 and write ``goldens.json``.

A real local Spark (JVM from the host, ``local[1]``), so the answers and the error classes are Spark's own::

    cd python/tests/pyspark_udf/udtf
    env -u SPARK_HOME JAVA_HOME=<a JDK 17 or 21> uv run --no-project --python 3.12 --with 'pyspark==4.2.0' \\
        --with 'pandas==3.0.6' --with 'pyarrow==26.0.0' python gen_goldens.py

A case records its schema and rows, or its error: the exception's class, condition, SQLSTATE and message, and the
Python worker's error class and message (the last ``[CLASS] message`` line of the worker traceback that Spark's
``PythonException`` carries).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases  # noqa: E402

#: The worker's error line inside a PythonException message: ``pyspark...Error: [CLASS] message``.
_WORKER_ERROR = re.compile(r"^(?:[\w.]+\.)?(\w+): \[([A-Z][A-Z0-9_.]*)\] (.*)$")


def worker_error(message: str) -> dict[str, str] | None:
    found = None
    for line in message.splitlines():
        m = _WORKER_ERROR.match(line.strip())
        if m:
            found = {"python_exception": m.group(1), "error_class": m.group(2), "message": m.group(3)}
    return found


def traceback_error(message: str) -> str | None:
    """The exception line of the Python traceback in a ``PythonException`` message (``ValueError: eval failed``)."""
    lines = message.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last):"):
            for after in lines[i + 1 :]:
                if after and not after.startswith((" ", "Traceback")):
                    return after
    return None


def error_record(e: Exception) -> dict[str, Any]:
    """The exception class, the condition and SQLSTATE (None for a plain Python error), the first line of the
    message (the JVM stack and the worker traceback follow it) and the worker's ``[CLASS] message``."""
    text = str(e)
    get_condition = getattr(e, "getCondition", None) or getattr(e, "getErrorClass", None)
    get_sqlstate = getattr(e, "getSqlState", None)
    condition = get_condition() if get_condition else None
    sqlstate = get_sqlstate() if get_sqlstate else None
    first = next((line for line in text.split("\n") if line.strip()), "")
    java = getattr(e, "java_exception", None)
    if java is not None:
        # A Py4J error names the JVM object (``o123``): record the JVM exception's own line instead.
        m = re.search(r"(java\.[\w.]+(?:Exception|Error): [^\n]*)", str(java))
        first = m.group(1) if m else str(java).split("\n")[0]
        # Stage, task and host differ per run.
        first = re.sub(r"stage \d+\.\d+", "stage N.0", first)
        first = re.sub(r"\(TID \d+\) \([\d.]+ executor driver\)", "(TID N) (HOST executor driver)", first)
        first = re.sub(r"task \d+\.\d+", "task N.0", first)
    return {
        "exception": type(e).__name__,
        "condition": condition,
        "sqlstate": sqlstate,
        "message": first,
        "worker": worker_error(text),
        "python_error": traceback_error(text),
    }


def run_case(api: cases.Api, spark: Any, build: Any) -> dict[str, Any]:
    try:
        df = build(api, spark)
        schema = df.schema.simpleString()
        rows = [[cases.cell(v) for v in row] for row in df.collect()]
    except Exception as e:
        return {"error": error_record(e)}
    return {"schema": schema, "rows": rows}


def main() -> None:
    import pandas
    import pyarrow
    import pyspark
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false")
    for k, v in cases.CONF.items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    api = cases.Api("pyspark")
    # The handler classes call helpers of ``cases``, which the Python workers cannot import: ship it by value.
    from pyspark import cloudpickle

    cloudpickle.register_pickle_by_value(cases)
    arrow_default = spark.conf.get("spark.sql.execution.pythonUDTF.arrow.enabled")
    out = {
        "generator": "python/tests/pyspark_udf/udtf/gen_goldens.py",
        "spark": spark.version,
        "pyspark": pyspark.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "conf": cases.CONF,
        "defaults": {"spark.sql.execution.pythonUDTF.arrow.enabled": arrow_default},
        "cases": {name: run_case(api, spark, build) for name, build in sorted(cases.CASES.items())},
    }
    spark.stop()
    (HERE / "goldens.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(out['cases'])} cases")


if __name__ == "__main__":
    main()
