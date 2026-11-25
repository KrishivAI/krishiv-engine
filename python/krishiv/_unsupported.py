# SPDX-License-Identifier: Apache-2.0
"""Names of Spark that Krishiv does not have, so that a script meets the engine's
``KRISHIV_UNSUPPORTED_FEATURE`` error (with the DataFrame way named) and not a bare
``AttributeError`` or ``ModuleNotFoundError``."""

from __future__ import annotations

from typing import Any

from krishiv.errors import unsupported_feature


class SparkConf:
    """Not supported: Krishiv has no ``SparkConf``; set options with ``SparkSession.builder.config``
    or ``spark.conf.set``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise unsupported_feature("SparkConf (use SparkSession.builder.config or spark.conf.set)")


class SparkContext:
    """Not supported: Krishiv has no ``SparkContext`` or RDDs; use the DataFrame API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise unsupported_feature("SparkContext (use the DataFrame API and SparkSession)")
