# SPDX-License-Identifier: Apache-2.0
"""``pyspark.sql.pandas.utils``: the version checks Spark's example scripts import.

The checks are the ones the UDF runtime already makes (Spark 4.2.1's minimums: pandas 2.2.0,
pyarrow 18.0.0, NumPy 1.22); they raise ``PySparkImportError`` when a package is missing or too old.
"""

from krishiv._udf.pandas_utils import (
    require_minimum_numpy_version,
    require_minimum_pandas_version,
    require_minimum_pyarrow_version,
)

__all__ = [
    "require_minimum_numpy_version",
    "require_minimum_pandas_version",
    "require_minimum_pyarrow_version",
]
