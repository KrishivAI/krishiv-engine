# SPDX-License-Identifier: Apache-2.0
"""Record real ``pyspark.testing`` (4.2.0) outcomes for ``_testing_scenarios.SCENARIOS``.

    SPARK_HOME=<stub dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \\
        --with 'pandas<3' --with pyarrow python python/tests/gen_pyspark_testing_fixture.py \\
        python/tests/fixtures/pyspark-testing-4.2.0.json

No JVM or Spark session is needed: the assertions run on lists of ``Row`` and on ``StructType``. Outcomes are
the pass/fail, the error class and the message with colour codes removed.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main(out: str) -> None:
    from _testing_differential import outcome, run_scenario
    from _testing_scenarios import SCENARIOS

    pyspark = importlib.import_module("pyspark")
    sql = importlib.import_module("pyspark.sql")
    testing = importlib.import_module("pyspark.testing")
    types = importlib.import_module("pyspark.sql.types")
    if pyspark.__version__ != "4.2.0":
        raise SystemExit(f"expected pyspark 4.2.0, found {pyspark.__version__}")

    def one(kind: str, args: tuple, kwargs: dict) -> dict:
        return outcome(
            lambda: run_scenario(
                kind, args, kwargs, sql.Row, types, testing.assertDataFrameEqual, testing.assertSchemaEqual
            )
        )

    result = {
        "spark_version": pyspark.__version__,
        "outcomes": {sid: one(kind, args, kwargs) for sid, kind, args, kwargs in SCENARIOS},
    }
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print("wrote", out, len(result["outcomes"]), "scenarios")


if __name__ == "__main__":
    main(sys.argv[1])
