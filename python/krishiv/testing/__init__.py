# SPDX-License-Identifier: Apache-2.0
"""Test helpers, as ``pyspark.testing``: ``assertDataFrameEqual`` and ``assertSchemaEqual``.

Behaviour (parameters, error classes, message formats) follows PySpark 4.2's
``pyspark.testing.utils``; the code below is written for this package (it works
on :class:`krishiv.sql.DataFrame`, lists of :class:`krishiv.sql.Row`, and pandas
frames when pandas is installed).
"""

from krishiv.testing.utils import assertDataFrameEqual, assertSchemaEqual

__all__ = ["assertDataFrameEqual", "assertSchemaEqual"]
