# SPDX-License-Identifier: Apache-2.0
"""Regenerates the adapted PySpark 4.2.0 test files from a checkout of the ``v4.2.0`` tag.

Usage: ``python adapt.py <spark-checkout>``. The rewrite is mechanical (imports and the ``__main__`` block); the
test bodies stay Spark's. Hand edits to an adapted file are listed in its header and must be re-applied after a
regeneration (``git diff`` shows them).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FILES = {
    "test_udf.py": "sql/tests/test_udf.py",
    "test_pandas_udf_scalar.py": "sql/tests/pandas/test_pandas_udf_scalar.py",
    "test_arrow_python_udf.py": "sql/tests/arrow/test_arrow_python_udf.py",
    # The coercion golden harnesses (sources, not collected) with their CSVs, and their mixin.
    "coercion/src_python_udf_input_type.py": "sql/tests/coercion/test_python_udf_input_type.py",
    "coercion/src_python_udf_return_type.py": "sql/tests/coercion/test_python_udf_return_type.py",
    "coercion/src_pandas_udf_input_type.py": "sql/tests/coercion/test_pandas_udf_input_type.py",
    "coercion/src_pandas_udf_return_type.py": "sql/tests/coercion/test_pandas_udf_return_type.py",
    "goldenutils.py": "testing/goldenutils.py",
}

HEADER = """# SPDX-License-Identifier: Apache-2.0
#
# Adapted from apache/spark tag v4.2.0, python/pyspark/{source}
# (Licensed to the Apache Software Foundation under the Apache License, Version 2.0; see the NOTICE file at the
# repository root). The test bodies are Spark's; the imports point at `krishiv` and the test support in
# `_support.py`. Which tests run is decided by `manifest.toml` (see `conftest.py`).
#
"""

# Longest first.
REWRITES = [
    (r"pyspark\.sql\.tests\.test_udf", "pyspark_udf.test_udf"),
    (r"pyspark\.testing\.goldenutils", "pyspark_udf.goldenutils"),
    (r"pyspark\.loose_version", "krishiv._udf.loose_version"),
    (r"pyspark\.testing\.sqlutils", "pyspark_udf._support"),
    (r"pyspark\.testing\.utils", "pyspark_udf._support"),
    (r"pyspark\.testing\.objects", "pyspark_udf._support"),
    (r"pyspark\.testing", "pyspark_udf._support"),
    (r"pyspark\.util", "pyspark_udf._support"),
    (r"pyspark\.logger", "pyspark_udf._support"),
    (r"pyspark\.errors", "pyspark_udf._support"),
    (r"pyspark\.sql", "krishiv.sql"),
    (r"from pyspark import", "from pyspark_udf._support import"),
]


# The iterator kind (`SCALAR_ITER`, eval type 204) is out of scope (ADR 0022 §1): the tests that loop over both
# scalar kinds keep the scalar one.
ADAPTATIONS = [
    ("[PandasUDFType.SCALAR, PandasUDFType.SCALAR_ITER]", "[PandasUDFType.SCALAR]"),
    # Krishiv's Row is not a class factory: the same value, built with the keyword form.
    ('Row("namedtuple")(1)', "Row(namedtuple=1)"),
    # `Row` lives in krishiv.sql, not krishiv.sql.types.
    ("from krishiv.sql.types import (\n    Row,\n", "from krishiv.sql import Row\nfrom krishiv.sql.types import (\n"),
    # Nested columns: `createDataFrame` of rows does not build them yet (see `_support.create_data_frame`).
    ("return self.spark.createDataFrame(args, schema)", "return self.create_data_frame(args, schema)"),
    ("            (pandas_iter_check_data, PandasUDFType.SCALAR_ITER),\n", ""),
    ("            (iter_internal_value, PandasUDFType.SCALAR_ITER),\n", ""),
    ("[(scalar_f, PandasUDFType.SCALAR), (iter_f, PandasUDFType.SCALAR_ITER)]", "[(scalar_f, PandasUDFType.SCALAR)]"),
]


def adapt(text: str, source: str) -> str:
    # Drop the ASF licence header (it moves into ours) and the `__main__` runner.
    text = re.sub(r"\A(#[^\n]*\n)+\n?", "", text)
    text = re.sub(r"\nif __name__ == \"__main__\":.*\Z", "\n", text, flags=re.S)
    for pattern, replacement in REWRITES:
        text = re.sub(pattern, replacement, text)
    for old, new in ADAPTATIONS:
        text = text.replace(old, new)
    return HEADER.format(source=source) + text


def main() -> None:
    root = Path(sys.argv[1]) / "python" / "pyspark"
    out = Path(__file__).parent
    for name, source in FILES.items():
        (out / name).parent.mkdir(exist_ok=True)
        (out / name).write_text(adapt((root / source).read_text(), source))
    for csv in sorted((root / "sql/tests/coercion").glob("golden_*_base.csv")) + sorted(
        (root / "sql/tests/coercion").glob("golden_python_udf_*_with_arrow.csv")
    ):
        (out / "coercion" / csv.name).write_bytes(csv.read_bytes())


if __name__ == "__main__":
    main()
