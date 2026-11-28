# SPDX-License-Identifier: Apache-2.0
"""Run every case of ``cases.py`` on PySpark 4.2.0 and write ``goldens.json``.

A real local Spark (JVM from the host, ``local[1]``), so the answers and the error classes are Spark's own::

    cd python/tests/pyspark_udf/relation
    env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' --with 'pandas==3.0.6' \\
        --with 'pyarrow==26.0.0' python gen_goldens.py

A case records its schema and rows, or its error: the JVM exception's class and condition, and the Python
worker's error (the last ``Error: message`` line of the worker traceback that Spark's ``PythonException``
carries, with its ``[CLASS]`` when it has one).
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
_WORKER_ERROR = re.compile(r"^(?:[\w.]+\.)?(\w*(?:Error|Exception|Invalid)): (?:\[([A-Z][A-Z0-9_.]*)\] )?(.*)$")


def worker_error(message: str) -> dict[str, str | None] | None:
    found = None
    for line in message.splitlines():
        m = _WORKER_ERROR.match(line.strip())
        if m and not m.group(1).endswith("PythonException"):
            found = {"python_exception": m.group(1), "error_class": m.group(2), "message": m.group(3)}
    return found


def run_case(api: cases.Api, spark: Any, build: Any) -> dict[str, Any]:
    from pyspark.errors import PySparkException

    try:
        df = build(api, spark)
        schema = df.schema.simpleString()
        rows = [[cases.cell(v) for v in row] for row in df.collect()]
    except PySparkException as e:
        return {
            "error": {
                "exception": type(e).__name__,
                "condition": e.getCondition(),
                "worker": worker_error(str(e)),
            }
        }
    except Exception as e:  # a JVM failure with no Spark condition: kept so a run shows every such case at once
        causes = [line for line in str(e).splitlines() if line.startswith("Caused by: ")]
        return {"error": {"exception": type(e).__name__, "jvm": causes[-1] if causes else str(e).splitlines()[0]}}
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
    spark.sparkContext.setLogLevel("OFF")
    api = cases.Api("pyspark")
    out = {
        "generator": "python/tests/pyspark_udf/relation/gen_goldens.py",
        "spark": spark.version,
        "pyspark": pyspark.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "conf": cases.CONF,
        "cases": {name: run_case(api, spark, build) for name, build in sorted(cases.CASES.items())},
    }
    spark.stop()
    (HERE / "goldens.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(out['cases'])} cases")


if __name__ == "__main__":
    main()
